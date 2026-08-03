"""
Forced top_p wiring tests (Kimi K3 endpoint mandate).

The provider serving moonshotai/kimi-k3 rejects any top_p other than 0.95
("Invalid value for 'top_p': 0.9. This endpoint requires top_p=0.95",
observed 2026-07-30). MODEL_CAPABILITIES rows may declare "forced_top_p";
ModelManager.resolve_top_p() must override caller-supplied and default values
on every API path that sends top_p. Per the validation rule, these tests drive
THE deployed generate paths (generate_with_openai / generate_once /
generate_async) with a fake transport that captures the request kwargs —
not a re-derivation of the resolution logic.
"""
from types import SimpleNamespace

import pytest

from models.model_manager import (
    API_MODEL_ALIASES,
    MODEL_CAPABILITIES,
    ModelManager,
    _slug_forced_top_p,
)
import models.model_manager as mm_module


KIMI_SLUG = "moonshotai/kimi-k3"


# ---------------------------------------------------------------------------
# Fake OpenAI transports capturing create() kwargs
# ---------------------------------------------------------------------------

def _fake_response():
    msg = SimpleNamespace(content="ok", tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


class _FakeAsyncClient:
    def __init__(self, captured):
        async def _create(**kwargs):
            captured.append(kwargs)
            if kwargs.get("stream"):
                async def _stream():
                    return
                    yield  # pragma: no cover
                return _stream()
            return _fake_response()

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=_create)
        )


class _FakeSyncClient:
    def __init__(self, captured):
        def _create(**kwargs):
            captured.append(kwargs)
            return _fake_response()

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=_create)
        )


@pytest.fixture
def manager(monkeypatch):
    # Skip the SentenceTransformer load; no network happens at construction.
    monkeypatch.setattr(
        ModelManager, "_get_cached_embedder", staticmethod(lambda: None)
    )
    mm = ModelManager(api_key="test-key")
    mm.captured = []
    mm.client = _FakeSyncClient(mm.captured)
    mm.async_client = _FakeAsyncClient(mm.captured)
    return mm


# ---------------------------------------------------------------------------
# Registry declaration
# ---------------------------------------------------------------------------

def test_kimi_k3_declares_forced_top_p():
    assert MODEL_CAPABILITIES[KIMI_SLUG]["forced_top_p"] == 0.95
    assert _slug_forced_top_p(KIMI_SLUG) == 0.95


def test_non_mandated_models_have_no_forced_top_p():
    for slug in ("anthropic/claude-fable-5", "deepseek/deepseek-v4-pro",
                 "openai/gpt-4o"):
        assert _slug_forced_top_p(slug) is None


def test_forced_top_p_values_are_valid():
    for slug, caps in MODEL_CAPABILITIES.items():
        forced = caps.get("forced_top_p")
        if forced is not None:
            assert 0.0 <= forced <= 1.0, f"{slug} forced_top_p={forced}"


# ---------------------------------------------------------------------------
# resolve_top_p
# ---------------------------------------------------------------------------

def test_resolve_top_p_forces_for_alias_and_slug(manager):
    # Both alias forms and the raw slug resolve to the mandated value,
    # regardless of what the caller asked for.
    for name in ("kimi-3", "kimi-k3", KIMI_SLUG):
        assert manager.resolve_top_p(name, 0.9) == 0.95
        assert manager.resolve_top_p(name, None) == 0.95
        assert manager.resolve_top_p(name, 1.0) == 0.95


def test_resolve_top_p_passthrough_for_unconstrained_model(manager):
    assert manager.resolve_top_p("deepseek-v4", 0.7) == 0.7
    assert manager.resolve_top_p("deepseek-v4", None) == mm_module.DEFAULT_TOP_P


# ---------------------------------------------------------------------------
# Deployed request paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_once_sends_forced_top_p(manager):
    result = await manager.generate_once("hi", model_name="kimi-3", top_p=0.9)
    assert result == "ok"
    assert len(manager.captured) == 1
    assert manager.captured[0]["top_p"] == 0.95
    assert manager.captured[0]["model"] == KIMI_SLUG


@pytest.mark.asyncio
async def test_generate_once_passthrough_for_unconstrained_model(manager):
    await manager.generate_once("hi", model_name="deepseek-v4", top_p=0.7)
    assert manager.captured[0]["top_p"] == 0.7


@pytest.mark.asyncio
async def test_generate_async_streaming_sends_forced_top_p(manager):
    manager.switch_model("kimi-3")
    await manager.generate_async("hi")
    assert len(manager.captured) == 1
    assert manager.captured[0]["stream"] is True
    assert manager.captured[0]["top_p"] == 0.95


@pytest.mark.asyncio
async def test_generate_async_caller_top_p_still_overridden(manager):
    manager.switch_model("kimi-3")
    await manager.generate_async("hi", top_p=0.9)
    assert manager.captured[0]["top_p"] == 0.95


def test_generate_with_openai_sends_forced_top_p(manager):
    result = manager.generate_with_openai("hi", KIMI_SLUG, top_p=0.9)
    assert result == "ok"
    assert manager.captured[0]["top_p"] == 0.95


def test_generate_sync_wrapper_sends_forced_top_p(manager):
    # generate() coerces top_p None -> 1.0 before delegating; the resolver
    # inside generate_with_openai must still win for mandated models.
    manager.generate("hi", model_name="kimi-3")
    assert manager.captured[0]["top_p"] == 0.95


def test_alias_roster_unchanged():
    # The fix keys off these aliases; fail loudly if they're renamed.
    assert API_MODEL_ALIASES["kimi-3"] == KIMI_SLUG
    assert API_MODEL_ALIASES["kimi-k3"] == KIMI_SLUG
