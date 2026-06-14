"""Tests for reasoning-only response recovery.

Regression coverage for the failure where a reasoning model (e.g. deepseek-v4
via OpenRouter) streams its entire answer into the API reasoning channel and
returns empty visible content. The consumer used to emit only the synthetic
"<thinking>" marker (10 chars), which the GUI then surfaced as the dead-end
"caught by the thinking filter" fallback.

The fix: when a stream is reasoning-only, close the dangling <thinking> marker
and retry once with native reasoning disabled (generate_once(disable_reasoning=True)),
which forces the model to emit the answer as normal content.
"""
import pytest
from unittest.mock import MagicMock

from core.response_generator import ResponseGenerator
from models.model_manager import ModelManager
from utils.time_manager import TimeManager


# ── Fake OpenAI-style streaming chunks ───────────────────────────────────────
class _Delta:
    def __init__(self, content="", reasoning_content=""):
        self.content = content
        self.reasoning_content = reasoning_content


class _Choice:
    def __init__(self, content="", reasoning_content="", finish_reason=None):
        self.delta = _Delta(content, reasoning_content)
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, content="", reasoning_content="", finish_reason=None):
        self.choices = [_Choice(content, reasoning_content, finish_reason)]


def _async_returning(value):
    """Build an async function that returns `value` (for awaited mocks)."""
    async def _fn(*args, **kwargs):
        return value
    return _fn


# ──────────────────────────────── ResponseGenerator ─────────────────────────
@pytest.fixture
def response_generator():
    return ResponseGenerator(model_manager=ModelManager(), time_manager=TimeManager())


@pytest.mark.asyncio
async def test_streaming_reasoning_only_recovers_via_retry(response_generator, monkeypatch):
    """Stream ends with reasoning but no content → recover via no-reasoning retry."""
    async def reasoning_only_stream():
        yield _Chunk(reasoning_content="Let me think about the user's week...")
        yield _Chunk(reasoning_content=" they need the disability center.")
        # stream ends — never any content delta

    captured = {}

    async def fake_generate_once(*args, **kwargs):
        captured["disable_reasoning"] = kwargs.get("disable_reasoning")
        return "Here's a real plan for your week."

    monkeypatch.setattr(response_generator.model_manager, "generate_async",
                        _async_returning(reasoning_only_stream()))
    monkeypatch.setattr(response_generator.model_manager, "generate_once", fake_generate_once)

    out = []
    async for chunk in response_generator.generate_streaming_response("plan my week", "deepseek-v4"):
        out.append(chunk)
    full = "".join(out)

    # Recovery retry was made with reasoning disabled
    assert captured.get("disable_reasoning") is True
    # Recovered answer is present; dead-end error is not
    assert "real plan for your week" in full
    assert "empty response" not in full.lower()
    # The dangling <thinking> was closed so the GUI can't get stuck on it
    assert "</thinking>" in full


@pytest.mark.asyncio
async def test_streaming_reasoning_only_finish_reason_path_recovers(response_generator, monkeypatch):
    """Empty content + finish_reason=stop also triggers recovery, not the error."""
    async def reasoning_then_stop():
        yield _Chunk(reasoning_content="thinking...")
        yield _Chunk(content="", finish_reason="stop")

    monkeypatch.setattr(response_generator.model_manager, "generate_async",
                        _async_returning(reasoning_then_stop()))
    monkeypatch.setattr(response_generator.model_manager, "generate_once",
                        _async_returning("Recovered answer."))

    full = "".join([c async for c in
                    response_generator.generate_streaming_response("q", "deepseek-v4")])

    assert "Recovered answer." in full
    assert "empty response" not in full.lower()


@pytest.mark.asyncio
async def test_streaming_reasoning_only_failed_recovery_shows_error(response_generator, monkeypatch):
    """If the recovery retry also yields nothing, fall back to the error string."""
    async def reasoning_then_stop():
        yield _Chunk(reasoning_content="thinking...")
        yield _Chunk(content="", finish_reason="stop")

    monkeypatch.setattr(response_generator.model_manager, "generate_async",
                        _async_returning(reasoning_then_stop()))
    monkeypatch.setattr(response_generator.model_manager, "generate_once",
                        _async_returning(""))  # recovery produces nothing

    full = "".join([c async for c in
                    response_generator.generate_streaming_response("q", "deepseek-v4")])

    assert "empty response" in full.lower()


@pytest.mark.asyncio
async def test_streaming_normal_content_no_recovery(response_generator, monkeypatch):
    """A normal content stream must not trigger a recovery retry."""
    async def normal_stream():
        yield _Chunk(content="Hello ")
        yield _Chunk(content="world.")

    called = {"once": False}

    async def fake_once(*a, **k):
        called["once"] = True
        return "SHOULD NOT BE USED"

    monkeypatch.setattr(response_generator.model_manager, "generate_async",
                        _async_returning(normal_stream()))
    monkeypatch.setattr(response_generator.model_manager, "generate_once", fake_once)

    full = "".join([c async for c in
                    response_generator.generate_streaming_response("q", "deepseek-v4")])

    assert "Hello" in full and "world" in full
    assert called["once"] is False
    assert "SHOULD NOT BE USED" not in full


@pytest.mark.asyncio
async def test_recover_reasoning_only_helper_yields_text(response_generator, monkeypatch):
    monkeypatch.setattr(response_generator.model_manager, "generate_once",
                        _async_returning("  recovered  "))
    out = [c async for c in response_generator._recover_reasoning_only("p", "sp", 256)]
    assert out == ["recovered"]


@pytest.mark.asyncio
async def test_recover_reasoning_only_helper_swallows_exception(response_generator, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("api down")
    monkeypatch.setattr(response_generator.model_manager, "generate_once", boom)
    out = [c async for c in response_generator._recover_reasoning_only("p", "sp", 256)]
    assert out == []


# ──────────────────────────────── AgenticSearchController ────────────────────
@pytest.fixture
def controller():
    from core.agentic.controller import AgenticSearchController
    manager = MagicMock()
    manager.api_models = {}
    return AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())


@pytest.mark.asyncio
async def test_agentic_final_response_reasoning_only_recovers(controller, monkeypatch):
    """The agentic final-response stream recovers a reasoning-only answer."""
    from core.agentic.types import AgenticSearchSession

    async def reasoning_only_stream():
        yield _Chunk(reasoning_content="The user needs guidance counselor + disability center.")
        # no content ever

    monkeypatch.setattr(controller, "_build_final_prompt", lambda **kw: "FINAL PROMPT")
    controller.model_manager.generate_async = _async_returning(reasoning_only_stream())

    captured = {}

    async def fake_once(*args, **kwargs):
        captured["disable_reasoning"] = kwargs.get("disable_reasoning")
        return "Here's how to reach both offices this week."

    controller.model_manager.generate_once = fake_once

    session = AgenticSearchSession(query="plan my week")
    out = []
    async for chunk in controller._generate_final_response(
        query="plan my week", system_prompt="sys", model_name="deepseek-v4", session=session
    ):
        out.append(chunk)
    full = "".join(out)

    assert captured.get("disable_reasoning") is True
    assert "reach both offices" in full
    assert "</thinking>" in full  # dangling marker closed
    # The raw output is no longer just the 10-char "<thinking>" sentinel
    assert full.replace("<thinking>", "").replace("</thinking>", "").strip()


@pytest.mark.asyncio
async def test_agentic_final_response_normal_content_no_recovery(controller, monkeypatch):
    from core.agentic.types import AgenticSearchSession

    async def normal_stream():
        yield _Chunk(content="Direct answer.")

    monkeypatch.setattr(controller, "_build_final_prompt", lambda **kw: "FINAL PROMPT")
    controller.model_manager.generate_async = _async_returning(normal_stream())

    called = {"once": False}

    async def fake_once(*a, **k):
        called["once"] = True
        return "UNUSED"

    controller.model_manager.generate_once = fake_once

    session = AgenticSearchSession(query="q")
    full = "".join([c async for c in controller._generate_final_response(
        query="q", system_prompt="sys", model_name="deepseek-v4", session=session)])

    assert "Direct answer." in full
    assert called["once"] is False
