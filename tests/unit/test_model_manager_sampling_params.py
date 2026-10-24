"""
Per-route sampling-parameter shape: `sampling_params: False` (2026-10-23).

The OpenRouter catalog lists NO temperature/top_p/top_k for
moonshotai/kimi-k3.1-pro, and Daemon sent temperature + top_p on every API
path. A route that refuses sampling parameters answers HTTP 400, so a row may
declare `sampling_params: False` and resolve_sampling() must send NONE of them
on ALL four request builders (generate_with_openai, generate_once,
generate_once_with_tools, generate_async streaming). An absent key behaves
exactly as before. Whether kimi-k3.1-pro really needs the flag is unknown until
the paid probe (scripts/probe_model_request_shapes.py --run); these tests use a
patched row, never the shipped one.

class: BC-82 (request shape declared per route), BC-83 (catalog cannot show it).
Drives THE deployed builders with a fake transport capturing the payload.
"""
from types import SimpleNamespace

import pytest

import models.model_manager as mm_module
from models.model_manager import MODEL_CAPABILITIES, ModelManager, resolve_sampling

KIMI31 = "moonshotai/kimi-k3.1-pro"
KIMI3 = "moonshotai/kimi-k3"
TOOLS = [{"type": "function", "function": {"name": "propose_action", "parameters": {"type": "object"}}}]
SAMPLING_KEYS = ("temperature", "top_p", "top_k")


def _fake_response():
    msg = SimpleNamespace(content="ok", tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


class _Async:
    def __init__(self, captured):
        async def _create(**kwargs):
            captured.append(kwargs)
            if kwargs.get("stream"):
                async def _stream():
                    return
                    yield  # pragma: no cover
                return _stream()
            return _fake_response()

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


class _Sync:
    def __init__(self, captured):
        def _create(**kwargs):
            captured.append(kwargs)
            return _fake_response()

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(ModelManager, "_get_cached_embedder", staticmethod(lambda: None))
    mm = ModelManager(api_key="test-key")
    mm.captured = []
    mm.client = _Sync(mm.captured)
    mm.async_client = _Async(mm.captured)
    return mm


@pytest.fixture
def refusing(monkeypatch):
    """kimi-k3.1-pro's row with the constraint declared (same-instance before/after)."""
    monkeypatch.setitem(MODEL_CAPABILITIES, KIMI31, {**MODEL_CAPABILITIES[KIMI31], "sampling_params": False})


def _no_sampling(payload):
    return [k for k in SAMPLING_KEYS if k in payload]


# --- helper ----------------------------------------------------------------

def test_absent_key_sends_what_was_given():
    assert resolve_sampling(KIMI31, 0.3, 0.9) == {"temperature": 0.3, "top_p": 0.9}
    assert resolve_sampling("openai/gpt-4o", 0.3, None) == {"temperature": 0.3}
    assert resolve_sampling("not/registered", None, None) == {}


def test_false_sends_nothing(refusing):
    assert resolve_sampling(KIMI31, 0.3, 0.9) == {}


def test_false_beats_forced_top_p(monkeypatch):
    monkeypatch.setitem(MODEL_CAPABILITIES, KIMI3,
                        {**MODEL_CAPABILITIES[KIMI3], "sampling_params": False})
    assert resolve_sampling(KIMI3, 0.3, 0.95) == {}


def test_true_or_other_rows_unchanged(monkeypatch):
    monkeypatch.setitem(MODEL_CAPABILITIES, KIMI31,
                        {**MODEL_CAPABILITIES[KIMI31], "sampling_params": True})
    assert resolve_sampling(KIMI31, 0.3, 0.9) == {"temperature": 0.3, "top_p": 0.9}
    assert resolve_sampling(KIMI3, 0.3, 0.95) == {"temperature": 0.3, "top_p": 0.95}


def test_shipped_row_declares_no_unprobed_constraint():
    # The row value comes from the paid probe; until then none is declared.
    assert "sampling_params" not in MODEL_CAPABILITIES[KIMI31]
    assert "forced_top_p" not in MODEL_CAPABILITIES[KIMI31]


# --- every deployed builder -----------------------------------------------

@pytest.mark.asyncio
async def test_generate_once_before_after(manager, monkeypatch):
    await manager.generate_once("hi", model_name="kimi-k3.1-pro", temperature=0.4, top_p=0.8)
    before = manager.captured[-1]
    assert before["temperature"] == 0.4 and before["top_p"] == 0.8
    monkeypatch.setitem(MODEL_CAPABILITIES, KIMI31, {**MODEL_CAPABILITIES[KIMI31], "sampling_params": False})
    await manager.generate_once("hi", model_name="kimi-k3.1-pro", temperature=0.4, top_p=0.8)
    assert _no_sampling(manager.captured[-1]) == []
    assert manager.captured[-1]["model"] == KIMI31


@pytest.mark.asyncio
async def test_generate_once_with_tools_omits(manager, refusing):
    await manager.generate_once_with_tools("save", model_name="kimi-k3.1-pro", tools=TOOLS, temperature=0.3)
    assert _no_sampling(manager.captured[0]) == []
    assert manager.captured[0]["tools"] == TOOLS


@pytest.mark.asyncio
async def test_generate_once_with_tools_unchanged_when_absent(manager):
    await manager.generate_once_with_tools("save", model_name="kimi-k3.1-pro", tools=TOOLS, temperature=0.3)
    assert manager.captured[0]["temperature"] == 0.3
    assert "top_p" not in manager.captured[0]  # this path never sent top_p


@pytest.mark.asyncio
async def test_generate_async_streaming_omits(manager, refusing):
    manager.switch_model("kimi-k3.1-pro")
    await manager.generate_async("hi", temperature=0.5, top_p=0.7)
    sent = manager.captured[0]
    assert sent["stream"] is True and _no_sampling(sent) == []


@pytest.mark.asyncio
async def test_generate_async_streaming_unchanged_when_absent(manager):
    manager.switch_model("kimi-k3.1-pro")
    await manager.generate_async("hi", temperature=0.5, top_p=0.7)
    assert manager.captured[0]["temperature"] == 0.5 and manager.captured[0]["top_p"] == 0.7


def test_generate_with_openai_omits(manager, refusing):
    assert manager.generate_with_openai("hi", KIMI31, temperature=0.5, top_p=0.7) == "ok"
    assert _no_sampling(manager.captured[0]) == []


def test_generate_with_openai_accepts_alias_and_slug(manager, refusing):
    manager.generate_with_openai("hi", KIMI31, temperature=0.5)
    manager.generate_with_openai("hi", "kimi-k3.1-pro", temperature=0.5)
    assert all(_no_sampling(c) == [] for c in manager.captured)


def test_forced_top_p_model_still_forced(manager):
    manager.generate_with_openai("hi", KIMI3, top_p=0.9)
    assert manager.captured[0]["top_p"] == 0.95
    assert mm_module.resolve_sampling(KIMI3, 0.3, 0.95)["top_p"] == 0.95
