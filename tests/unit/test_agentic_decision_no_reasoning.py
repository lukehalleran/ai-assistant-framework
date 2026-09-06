"""Fix 1.5: the agentic decision phase must actually disable reasoning.

Root cause (verified live): AgenticSearchController._generate_decision_no_reasoning
built its create() kwargs directly and left a comment "Explicitly do NOT add
reasoning params" — but for a reasoning-by-default model (moonshotai/kimi-k3,
the active model) simply omitting the ``reasoning`` key does NOT disable
reasoning; the model still burns its token budget on chain-of-thought before
ever emitting an XML tool marker. Live: three memory_search decision rounds
took 45.6s of decision time for 45ms of tool time. The direct call also
bypassed ``resolve_top_p`` (kimi-k3's endpoint 400s on any top_p other than
0.95) and OpenRouter's ``extra_body={"usage":{"include":True}}`` cache
accounting.

The fix routes the decision call through THE deployed
``model_manager.generate_once(disable_reasoning=True)`` — which already sends
``extra_body["reasoning"] = {"enabled": False}`` for reasoning-capable models
and applies ``resolve_top_p`` on every API path. These tests drive the
deployed ``_generate_decision_no_reasoning`` (and, for the native-tools path,
``_generate_with_tools`` -> ``generate_once_with_tools``) with a
kwargs-capturing fake transport (same pattern as test_forced_top_p.py) —
never a re-derivation of what the request "should" contain.
"""
from types import SimpleNamespace

import pytest

from core.agentic.controller import AgenticSearchController
from models.model_manager import ModelManager

KIMI_SLUG = "moonshotai/kimi-k3"


def _fake_response(content="ok"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


class _FakeAsyncClient:
    def __init__(self, captured):
        async def _create(**kwargs):
            captured.append(kwargs)
            return _fake_response()

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


@pytest.fixture
def manager(monkeypatch):
    # Skip the SentenceTransformer load; no network happens at construction.
    monkeypatch.setattr(ModelManager, "_get_cached_embedder", staticmethod(lambda: None))
    mm = ModelManager(api_key="test-key")
    mm.captured = []
    mm.async_client = _FakeAsyncClient(mm.captured)
    mm.switch_model("kimi-3")
    return mm


@pytest.fixture
def controller(manager):
    from unittest.mock import MagicMock
    return AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())


@pytest.mark.asyncio
async def test_decision_no_reasoning_disables_reasoning_and_forces_top_p(controller, manager):
    result = await controller._generate_decision_no_reasoning(
        prompt="What tool should I use?",
        model_name="kimi-3",
        system_prompt="You are an agentic search assistant.",
    )
    assert result == "ok"
    assert len(manager.captured) == 1
    kwargs = manager.captured[0]
    assert kwargs["extra_body"]["reasoning"] == {"enabled": False}
    assert kwargs["top_p"] == 0.95
    assert kwargs["model"] == KIMI_SLUG


@pytest.mark.asyncio
async def test_decision_no_reasoning_active_model_fallback(controller, manager):
    """model_name=None must fall back to the active model (kimi-3), matching
    the pre-fix contract (`model_name or self.model_manager.active_model_name`)."""
    result = await controller._generate_decision_no_reasoning(
        prompt="hi", model_name=None, system_prompt="sys",
    )
    assert result == "ok"
    assert manager.captured[0]["model"] == KIMI_SLUG
    assert manager.captured[0]["extra_body"]["reasoning"] == {"enabled": False}


@pytest.mark.asyncio
async def test_decision_no_reasoning_swallows_errors(controller, manager, monkeypatch):
    """On failure the method must never raise — the caller in
    _get_model_decision has its own except/timeout handling around whatever
    string comes back. generate_once (unlike the old direct client call)
    classifies the failure into a string rather than propagating the
    exception; either way, no exception escapes this method."""
    async def _boom(**kwargs):
        raise RuntimeError("upstream down")

    manager.async_client.chat.completions.create = _boom
    result = await controller._generate_decision_no_reasoning(
        prompt="hi", model_name="kimi-3", system_prompt="sys",
    )
    assert isinstance(result, str)
    assert "upstream down" in result


@pytest.mark.asyncio
async def test_generate_with_tools_disables_reasoning_for_native_model(controller, manager):
    """The native-tools path (_generate_with_tools -> generate_once_with_tools)
    must also explicitly disable reasoning for a reasoning-by-default model —
    the pre-fix code relied on omitting the key entirely, the same bug."""
    response = await controller._generate_with_tools(
        prompt="What tool should I use?",
        system_prompt="You are an agentic search assistant.",
        model_name="kimi-3",
        tools=[{"type": "function", "function": {"name": "noop", "parameters": {}}}],
    )
    assert len(manager.captured) == 1
    kwargs = manager.captured[0]
    assert kwargs["extra_body"]["reasoning"] == {"enabled": False}
    assert kwargs["model"] == KIMI_SLUG
    assert getattr(response, "content", None) == "ok" or response.content == "ok"
