"""First-token deadline + single retry on streamed answers.

Incident: oct27 wave 2 lane L1, 2026-10-27 (AGT turn 10-24: 27 s for 321 tokens
vs 7 s for the same shape - the provider stalled before the first chunk and the
only guard was the 120 s per-chunk STREAM_TIMEOUT). Class: BC-69 (unbounded
wait), BC-41 (provider stall holding the turn), BC-82 (live config).

Drives THE deployed ``ResponseGenerator.generate_streaming_response`` and
``AgenticSearchController._generate_final_response`` with fake streams: a
stream that stalls before its first chunk is closed and re-opened ONCE; a
second stall fails as today; a stall AFTER a chunk is never retried.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import config.app_config as ac
from core.agentic.controller import AgenticSearchController
from core.agentic.types import AgenticSearchSession
from core.first_token_deadline import FirstTokenTimeout, guard_first_chunk
from core.response_generator import ResponseGenerator
from utils.time_manager import TimeManager

PROMPTS = ["Tell me about the lease.", "   Tell me\n   about the lease.\n"]


async def _stalled():
    await asyncio.sleep(30)
    yield "never"


async def _good(*parts):
    for p in parts:
        yield p


async def _chunk_then_stall():
    yield "partial "
    await asyncio.sleep(30)
    yield "never"


class _Manager:
    """generate_async returns the next scripted stream and counts opens."""

    def __init__(self, *streams):
        self.streams = list(streams)
        self.opens = 0
        self.active_model_name = "kimi-3"

    async def generate_async(self, *a, **kw):
        self.opens += 1
        return self.streams.pop(0)


@pytest.fixture(autouse=True)
def _short_deadline(monkeypatch):
    monkeypatch.setattr(ac, "FIRST_TOKEN_TIMEOUT_S", 0.2)


def _rg(manager):
    return ResponseGenerator(model_manager=manager, time_manager=TimeManager())


async def _collect(agen):
    out = []
    async for c in agen:
        out.append(c)
    return "".join(out)


# ---- helper -------------------------------------------------------------
@pytest.mark.asyncio
async def test_helper_retries_once_then_passes_through():
    reopened = []

    async def reopen():
        reopened.append(1)
        return _good("a", "b")

    it = await guard_first_chunk(_stalled(), reopen, 0.1)
    assert [c async for c in it] == ["a", "b"]
    assert reopened == [1]


@pytest.mark.asyncio
async def test_helper_second_stall_raises_and_disabled_guard_passes_through():
    async def reopen():
        return _stalled()

    with pytest.raises(FirstTokenTimeout):
        await guard_first_chunk(_stalled(), reopen, 0.1)
    it = await guard_first_chunk(_good("x"), reopen, 0.0)   # <=0 disables
    assert [c async for c in it] == ["x"]


# ---- ResponseGenerator ----------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", PROMPTS)
async def test_response_generator_retries_once_on_first_token_stall(prompt):
    mgr = _Manager(_stalled(), _good("Hello ", "there."))
    text = await _collect(_rg(mgr).generate_streaming_response(prompt, "kimi-3"))
    assert "Hello" in text and "there." in text
    assert "Error" not in text
    assert mgr.opens == 2          # base: one open, waits the 120 s STREAM_TIMEOUT


@pytest.mark.asyncio
async def test_response_generator_second_stall_fails_as_before():
    mgr = _Manager(_stalled(), _stalled())
    text = await _collect(_rg(mgr).generate_streaming_response(PROMPTS[0], "kimi-3"))
    assert text.startswith("[Error: Stream timeout")
    assert mgr.opens == 2


@pytest.mark.asyncio
async def test_response_generator_no_retry_after_a_chunk(monkeypatch):
    mgr = _Manager(_chunk_then_stall(), _good("SHOULD NOT APPEAR"))
    rg = _rg(mgr)
    # the per-chunk deadline after the first chunk is the existing STREAM_TIMEOUT;
    # stop the test waiting on it: it must not have re-opened within the window
    task = asyncio.ensure_future(_collect(rg.generate_streaming_response(PROMPTS[0], "kimi-3")))
    await asyncio.sleep(0.8)       # > 3x the first-token deadline
    assert mgr.opens == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ---- agentic synthesis -----------------------------------------------------
def _controller(mgr):
    ctl = AgenticSearchController(model_manager=mgr, web_search_manager=MagicMock())
    ctl._build_final_prompt = MagicMock(return_value="FINAL PROMPT")
    return ctl


async def _drain(ctl):
    return await _collect(ctl._generate_final_response(
        query="q", system_prompt="sys", model_name="kimi-3",
        session=AgenticSearchSession(query="q"),
    ))


class _Delta:
    def __init__(self, content):
        self.content = content
        self.reasoning_content = ""


def _chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=_Delta(text), finish_reason=None)], usage=None)


async def _good_chunks(*parts):
    for p in parts:
        yield _chunk(p)


@pytest.mark.asyncio
async def test_agentic_synthesis_retries_once_on_first_token_stall():
    mgr = _Manager(_stalled(), _good_chunks("The answer ", "is here."))
    text = await _drain(_controller(mgr))
    assert "The answer is here." in text
    assert mgr.opens == 2


@pytest.mark.asyncio
async def test_agentic_synthesis_second_stall_fails_as_before():
    mgr = _Manager(_stalled(), _stalled())
    text = await _drain(_controller(mgr))
    assert text.startswith("I apologize, but I encountered an error")
    assert mgr.opens == 2


@pytest.mark.asyncio
async def test_agentic_synthesis_no_retry_after_a_chunk():
    async def one_then_stall():
        yield _chunk("partial ")
        await asyncio.sleep(30)

    mgr = _Manager(one_then_stall(), _good_chunks("SHOULD NOT APPEAR"))
    ctl = _controller(mgr)
    task = asyncio.ensure_future(_drain(ctl))
    await asyncio.sleep(0.8)
    assert mgr.opens == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
