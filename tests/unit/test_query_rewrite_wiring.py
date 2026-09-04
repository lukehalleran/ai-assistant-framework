"""
Regression tests for the 2026-09-04 query-rewrite wiring fixes.

Two independent bugs conspired to make the (config-disabled) rewrite path
fire anyway AND crash every time it did:

  1. DaemonOrchestrator built the ContextPipeline config dict inline in
     __init__ with `self.config.get("REWRITE_TIMEOUT_S", 2.0)`. Nothing ever
     sets a root-level "REWRITE_TIMEOUT_S" key on self.config (the real
     value lives at config.yaml `features.rewrite_timeout_s`, resolved by
     app_config into REWRITE_TIMEOUT_S), so the literal 2.0 fallback ALWAYS
     won — even though config.yaml ships `features.rewrite_timeout_s: 0`
     (app_config.REWRITE_TIMEOUT_S == 0.0), the pipeline always got a
     nonzero timeout and ran the rewrite LLM call.
  2. core.context_pipeline.ContextPipeline._rewrite_query called
     model_manager.generate_once(..., model="gpt-4o-mini", ...) — the real
     ModelManager.generate_once signature takes `model_name`, not `model` —
     so every invocation raised TypeError (caught + logged as "Query
     rewrite failed").

Both fixed 2026-09-04:
  - core/orchestrator.py: the dict-building was factored into
    DaemonOrchestrator._build_context_pipeline_config, whose REWRITE_TIMEOUT_S
    fallback is now the live app_config.REWRITE_TIMEOUT_S value (an explicit
    self.config["REWRITE_TIMEOUT_S"] override still wins).
  - core/context_pipeline.py:_rewrite_query now calls
    generate_once(model_name="gpt-4o-mini", ...).

These tests drive the DEPLOYED functions/classes — no getsource string pins.
"""
import types

import pytest

import config.app_config as app_config
from core.orchestrator import DaemonOrchestrator
from core.context_pipeline import ContextPipeline


# ---------------------------------------------------------------------------
# 1. DaemonOrchestrator._build_context_pipeline_config honors the live
#    app_config default instead of a hardcoded literal.
# ---------------------------------------------------------------------------

class TestBuildContextPipelineConfig:
    def test_default_follows_live_app_config_disabled_value(self, monkeypatch):
        """config.yaml's features.rewrite_timeout_s: 0 (disabled) must reach
        the pipeline config as 0.0 — not the old hardcoded 2.0."""
        monkeypatch.setattr(app_config, "REWRITE_TIMEOUT_S", 0.0)
        fake_self = types.SimpleNamespace(config={})
        cfg = DaemonOrchestrator._build_context_pipeline_config(fake_self)
        assert cfg["REWRITE_TIMEOUT_S"] == 0.0

    def test_default_follows_live_app_config_nonzero_value(self, monkeypatch):
        """If the feature were ever re-enabled with a different timeout, the
        pipeline config must track it (proves this isn't just a special-case
        zero check)."""
        monkeypatch.setattr(app_config, "REWRITE_TIMEOUT_S", 3.5)
        fake_self = types.SimpleNamespace(config={})
        cfg = DaemonOrchestrator._build_context_pipeline_config(fake_self)
        assert cfg["REWRITE_TIMEOUT_S"] == 3.5

    def test_explicit_config_override_still_wins(self, monkeypatch):
        monkeypatch.setattr(app_config, "REWRITE_TIMEOUT_S", 0.0)
        fake_self = types.SimpleNamespace(config={"REWRITE_TIMEOUT_S": 5.0})
        cfg = DaemonOrchestrator._build_context_pipeline_config(fake_self)
        assert cfg["REWRITE_TIMEOUT_S"] == 5.0

    def test_falsy_config_dict_still_uses_live_default(self, monkeypatch):
        """self.config is None/{} in some construction paths — the ternary's
        `if self.config else` branch must not reintroduce the hardcoded 2.0."""
        monkeypatch.setattr(app_config, "REWRITE_TIMEOUT_S", 0.0)
        fake_self = types.SimpleNamespace(config=None)
        cfg = DaemonOrchestrator._build_context_pipeline_config(fake_self)
        assert cfg["REWRITE_TIMEOUT_S"] == 0.0

    def test_other_keys_unaffected(self, monkeypatch):
        """Sanity: the refactor into a helper didn't drop the other keys."""
        monkeypatch.setattr(app_config, "REWRITE_TIMEOUT_S", 0.0)
        fake_self = types.SimpleNamespace(
            config={"features": {"use_stm_pass": False, "enable_query_rewrite": False}},
            stm_min_depth=7,
        )
        cfg = DaemonOrchestrator._build_context_pipeline_config(fake_self)
        assert cfg["USE_STM_PASS"] is False
        assert cfg["enable_query_rewrite"] is False
        assert cfg["STM_MIN_CONVERSATION_DEPTH"] == 7


# ---------------------------------------------------------------------------
# 2 & 3. ContextPipeline._rewrite_query: disabled short-circuit + malformed
#        kwarg fix, driven against a fake model_manager with the REAL
#        generate_once signature (no **kwargs — a stray `model=` raises).
# ---------------------------------------------------------------------------

class _RealSignatureModelManager:
    """Mirrors models.model_manager.ModelManager.generate_once's real
    signature exactly (no **kwargs) so a wrong kwarg name raises TypeError,
    the way the live bug did."""

    def __init__(self, result: str = "rewritten query"):
        self.result = result
        self.calls: list[dict] = []

    async def generate_once(
        self,
        prompt: str,
        model_name: str = None,
        system_prompt: str = "You are a concise and helpful assistant.",
        max_tokens: int = 256,
        temperature: float = None,
        top_p: float = None,
        disable_reasoning: bool = False,
    ) -> str:
        self.calls.append({
            "prompt": prompt,
            "model_name": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        return self.result


class _CallRecordingModelManager:
    """Records whether generate_once was invoked at all (for the
    _rewrite_timeout == 0 short-circuit test)."""

    def __init__(self):
        self.calls: list[dict] = []

    async def generate_once(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return "should never be reached"


@pytest.mark.asyncio
class TestRewriteQueryWiring:
    async def test_disabled_timeout_returns_none_and_never_calls_model(self):
        manager = _CallRecordingModelManager()
        pipeline = ContextPipeline(model_manager=manager, topic_manager=None,
                                    config={"REWRITE_TIMEOUT_S": 0})
        assert pipeline._rewrite_timeout == 0

        result = await pipeline._rewrite_query("What is the deal with async debugging today?")

        assert result is None
        assert manager.calls == []

    async def test_enabled_timeout_calls_generate_once_with_model_name_kwarg(self):
        manager = _RealSignatureModelManager(result="a fully rewritten search query")
        pipeline = ContextPipeline(model_manager=manager, topic_manager=None,
                                    config={"REWRITE_TIMEOUT_S": 1.0})
        assert pipeline._rewrite_timeout == 1.0

        # >=10 words, question-shaped, so should_rewrite's fallback heuristic
        # (no query_analysis passed) fires.
        query = "What is actually going on with the async debugging issue we discussed?"
        result = await pipeline._rewrite_query(query)

        assert result == "a fully rewritten search query"
        assert len(manager.calls) == 1
        assert manager.calls[0]["model_name"] == "gpt-4o-mini"
        # The old bug passed `model=` instead — confirm that kwarg is gone by
        # construction (a TypeError would have propagated out of _rewrite_query
        # and been swallowed as a warning, returning None instead of a result).
