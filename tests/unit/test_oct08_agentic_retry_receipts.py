"""BC-72/BC-58: the silent agentic retry inside enhanced mode ran tools but its
receipts never reached the turn telemetry (only the agentic-search closure
attached them). `_silent_agentic_retry(ctx=...)` now attaches them whether the
retry is accepted or rejected, and never attributes a stale session."""
import asyncio
from types import SimpleNamespace

import gui.handlers as handlers

_RECEIPTS = [{"tool": "search_memory", "status": "ok", "reason": ""}]
_ORIGINAL = "alpha beta gamma delta epsilon zeta eta theta"


class _FakeController:
    def __init__(self, text, set_session=True, stale=None):
        self._text = text
        self._set_session = set_session
        self._last_session = stale

    async def run_agentic_search(self, **_kw):
        if self._set_session:
            self._last_session = SimpleNamespace(
                tool_receipts=list(_RECEIPTS), providers_failed={"tavily": "402"})
        yield self._text


def _run(controller, ctx):
    orch = SimpleNamespace(agentic_controller=controller)
    return asyncio.run(handlers._silent_agentic_retry(
        orch, "q", "sys", "m", "ctx", _ORIGINAL, "hint", "TEST", ctx=ctx))


def test_accepted_retry_attaches_receipts():
    ctx = SimpleNamespace(telemetry={})
    clean, _ = _run(_FakeController("entirely different words here now"), ctx)
    assert clean is not None
    assert ctx.telemetry["tool_calls"] == _RECEIPTS
    assert ctx.telemetry["providers_failed"] == {"tavily": "402"}


def test_rejected_retry_still_attaches_receipts():
    ctx = SimpleNamespace(telemetry={})
    clean, _ = _run(_FakeController(_ORIGINAL), ctx)
    assert clean is None
    assert ctx.telemetry["tool_calls"] == _RECEIPTS


def test_stale_session_is_not_attributed():
    stale = SimpleNamespace(tool_receipts=[{"tool": "old"}])
    ctx = SimpleNamespace(telemetry={})
    _run(_FakeController("some other answer text", set_session=False, stale=stale), ctx)
    assert "tool_calls" not in ctx.telemetry


def test_no_ctx_is_unchanged_behaviour():
    clean, _ = _run(_FakeController("entirely different words here now"), None)
    assert clean is not None


def test_helper_never_raises_on_bad_telemetry():
    handlers._attach_agentic_tool_receipts(SimpleNamespace(telemetry=None), SimpleNamespace(tool_receipts=[]))
    handlers._attach_agentic_tool_receipts(SimpleNamespace(), None)
