"""
Per-call model routing (2026-10-15, class: BC-30).

ResponseGenerator used to call ``model_manager.switch_model(model_name)`` inside
its coroutines, mutating the process-wide active model; any concurrent
coroutine (grounding verifier, personal-claim check, a second tab's turn,
best-of candidates) then ran on whatever model the last switch left. The call
is now routed with ``generate_async(route_model=...)`` and the active model is
never written. Tests drive THE deployed ResponseGenerator + ModelManager with a
fake transport capturing the request ``model``.
"""
import asyncio
import logging
from types import SimpleNamespace

import pytest

from core.response_generator import ResponseGenerator
from models.model_manager import API_MODEL_ALIASES, ModelManager
from utils.time_manager import TimeManager

M1, M2 = "kimi-3", "deepseek-v4"


def _chunk(text):
    delta = SimpleNamespace(content=text, reasoning_content="")
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)], usage=None)


class _Client:
    def __init__(self, captured, gate=None):
        async def _create(**kwargs):
            captured.append(kwargs)
            if kwargs.get("stream"):
                async def _stream():
                    if gate is not None:
                        await gate.wait()
                    yield _chunk("hello there friend")
                return _stream()
            msg = SimpleNamespace(content="full", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


@pytest.fixture
def mm(monkeypatch):
    monkeypatch.setattr(ModelManager, "_get_cached_embedder", staticmethod(lambda: None))
    m = ModelManager(api_key="test-key")
    m.captured = []
    m.async_client = _Client(m.captured)
    m.client = m.async_client
    m.switch_model(M1)
    return m


@pytest.fixture
def rg(mm):
    return ResponseGenerator(model_manager=mm, time_manager=TimeManager())


async def _drain(agen):
    return "".join([c async for c in agen])


@pytest.mark.asyncio
async def test_streaming_routes_per_call_and_leaves_active(rg, mm):
    out = await _drain(rg.generate_streaming_response("hi", M2, system_prompt="sp"))
    assert "hello" in out
    assert mm.captured[0]["model"] == API_MODEL_ALIASES[M2]
    assert mm.get_active_model_name() == M1


@pytest.mark.asyncio
async def test_concurrent_streams_route_to_own_models(rg, mm):
    gate = asyncio.Event()
    mm.async_client = _Client(mm.captured, gate)
    t1 = asyncio.create_task(_drain(rg.generate_streaming_response("a", M1, system_prompt="sp")))
    t2 = asyncio.create_task(_drain(rg.generate_streaming_response("b", M2, system_prompt="sp")))
    await asyncio.sleep(0.05)
    gate.set()
    await asyncio.gather(t1, t2)
    by_prompt = {c["messages"][-1]["content"]: c["model"] for c in mm.captured}
    assert by_prompt["a"] == API_MODEL_ALIASES[M1]
    assert by_prompt["b"] == API_MODEL_ALIASES[M2]
    assert mm.get_active_model_name() == M1


@pytest.mark.asyncio
async def test_unregistered_route_model_warns_and_uses_active(mm, caplog):
    with caplog.at_level(logging.WARNING):
        await mm.generate_async("hi", route_model="no-such-model")
    assert mm.captured[0]["model"] == API_MODEL_ALIASES[M1]
    assert any("not registered" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_legacy_model_name_kwarg_still_ignored(mm):
    await mm.generate_async("hi", model_name=M2)
    assert mm.captured[0]["model"] == API_MODEL_ALIASES[M1]
    assert mm.get_active_model_name() == M1


@pytest.mark.asyncio
async def test_generate_full_passes_model_without_switching(rg, mm, monkeypatch):
    seen = {}

    async def _fake_once(**kwargs):
        seen.update(kwargs)
        return "text"

    monkeypatch.setattr(mm, "generate_once", _fake_once)
    assert await rg.generate_full("p", M2, system_prompt="sp") == "text"
    assert seen["model_name"] == M2
    assert mm.get_active_model_name() == M1
