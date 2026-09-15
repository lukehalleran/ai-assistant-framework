"""A05b-2: buffered delivery in correct mode — no unreviewed draft reaches
the wire (docs/execution/generalization/A05_design.md, "Decision: design A").

Drives the deployed api.chat_service.submit_stream / gui.handlers.handle_submit
for both routes (agentic and enhanced), plus the new
gui.handlers._buffer_grounding_draft wrapper directly for the one shape
neither route can produce itself (both catch every internal exception into a
final "debug" chunk). Fakes only — no LLM, no network, no live store.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import config.app_config as config
import gui.handlers as handlers
from api.chat_service import submit_stream
from api.schemas import ChatRequest
from api.state import AppState
from core.grounding_check import GroundingVerdict
from tests.unit.test_handle_submit import (
    _gate_decision, _make_file_processor_mock, _make_orchestrator,
)
from tests.unit.test_sep09_speed_batch import ANSWER, REVISED, FALLBACK_TEXT

# Distinctive substring present ONLY in the flawed draft (REVISED differs by
# "four" vs "three"; FALLBACK_TEXT never repeats the flawed clause). Asserted
# so a future wording change to those shared fixtures fails loudly here
# instead of silently defeating this file's "no draft leaked" checks.
DRAFT_TOKEN = "three in the afternoon"
assert DRAFT_TOKEN in ANSWER and DRAFT_TOKEN not in REVISED and DRAFT_TOKEN not in FALLBACK_TEXT

# Two streaming deltas whose concatenation is exactly ANSWER (PART1 ends with
# a space, so both the enhanced route's smart_join and the agentic route's
# plain "+=" reconstruct it byte-for-byte) — the contract's "[PART1,
# PART1+PART2]" multi-chunk hold shape.
_SPLIT = ANSWER.index("the scheduled")
PART1, PART2 = ANSWER[:_SPLIT], ANSWER[_SPLIT:]
assert PART1 + PART2 == ANSWER
assert DRAFT_TOKEN not in PART1 and DRAFT_TOKEN in PART2


@pytest.fixture
def turn_setup(monkeypatch, tmp_path):
    import main  # noqa: F401 - keep cold startup imports outside async ordering checks
    import core.agentic.gate as gate
    import core.grounding_check as grounding
    monkeypatch.setattr(config, "GROUNDING_CHECK_ENABLED", True)
    # The independent personal-claim check (2026-09-15) is exercised in
    # test_personal_claim_delivery.py; keep these verifier-call counts pure.
    monkeypatch.setattr(config, "PERSONAL_CLAIM_CHECK_ENABLED", False)
    monkeypatch.setattr(config, "GROUNDING_MIN_RESPONSE_CHARS", 1)
    monkeypatch.setattr(config, "TURN_TELEMETRY_ENABLED", False)
    monkeypatch.setattr(grounding, "has_checkable_claims", lambda *a: True)
    monkeypatch.setattr(grounding, "integrate_grounding_correction", AsyncMock(return_value=REVISED))
    decision = _gate_decision()
    decision.insight_intent = None
    decision.deferred_request = None
    monkeypatch.setattr(gate, "evaluate_agentic_gate", AsyncMock(return_value=decision))
    monkeypatch.setattr(gate, "apply_intent_veto", lambda decision, *a, **kw: decision)
    monkeypatch.setattr(handlers, "file_processor", _make_file_processor_mock("Explain the sample record."))
    monkeypatch.setattr(handlers, "get_conversation_logger", MagicMock())
    monkeypatch.setattr(handlers, "_apply_action_guard", AsyncMock(return_value=""))
    monkeypatch.setattr(handlers, "_dispatch_storage", MagicMock())
    started, release = asyncio.Event(), asyncio.Event()

    async def verify(*args, **kwargs):
        started.set()
        await release.wait()
        return GroundingVerdict(
            false_claim_present=True, claim="sample time", why_false="sample mismatch",
            confidence=0.99, correction="The sample time is four in the afternoon.",
        )

    monkeypatch.setattr(grounding, "verify_grounding", verify)
    return SimpleNamespace(started=started, release=release)


async def _consume(state, events):
    async for event in submit_stream(ChatRequest(text="Explain the sample record."), state):
        events.append(event)


def _no_draft_before_complete(events):
    complete_idx = next(i for i, e in enumerate(events) if e.event == "complete")
    for e in events[:complete_idx]:
        assert DRAFT_TOKEN not in json.dumps(e.data), f"draft leaked in pre-complete {e.event}: {e.data}"
    return complete_idx


def _has_progress_text(events, text):
    return any(e.event == "progress" and text in (e.data.get("text") or "") for e in events)


# --- Failing-first: correct mode never streams the draft, both routes. -----

@pytest.mark.parametrize("agentic", [False, True])
@pytest.mark.parametrize("integrate", [False, True])
async def test_correct_mode_buffers_draft_until_reviewed(turn_setup, monkeypatch, agentic, integrate):
    """No message/thinking event carries the draft before complete; a
    fact-check progress chunk stands in; complete carries the reviewed text
    (integrated correction, or the A05a fallback when integration is off)."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "correct")
    monkeypatch.setattr(config, "GROUNDING_INTEGRATE_ENABLED", integrate)
    orch = _make_orchestrator(
        agentic_enabled=agentic, streaming_chunks=[PART1, PART2], agentic_items=[PART1, PART2],
    )
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    await asyncio.wait_for(turn_setup.started.wait(), 10)
    turn_setup.release.set()
    await asyncio.wait_for(task, 5)

    _no_draft_before_complete(events)
    assert _has_progress_text(events, handlers._GROUNDING_CHECK_PROGRESS_TEXT)
    complete = next(e for e in events if e.event == "complete")
    assert complete.data["content"] == (REVISED if integrate else FALLBACK_TEXT)
    assert DRAFT_TOKEN not in complete.data["content"]


# --- Controls: log_only and off keep today's pass-through streaming. -------

@pytest.mark.parametrize("agentic", [False, True])
async def test_log_only_streams_draft_unchanged(turn_setup, monkeypatch, agentic):
    """log_only keeps today's event sequence — the draft still streams live
    before complete, same as before this batch."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "log_only")
    orch = _make_orchestrator(
        agentic_enabled=agentic, streaming_chunks=[PART1, PART2], agentic_items=[PART1, PART2],
    )
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    try:
        await asyncio.wait_for(turn_setup.started.wait(), 10)
        await asyncio.wait_for(asyncio.shield(task), 1)
    finally:
        turn_setup.release.set()
        await asyncio.gather(task, return_exceptions=True)
        await handlers.wait_for_pending_storage(timeout=2)

    assert any(e.event in ("message", "thinking") and DRAFT_TOKEN in json.dumps(e.data) for e in events)
    assert not _has_progress_text(events, handlers._GROUNDING_CHECK_PROGRESS_TEXT)
    assert next(e for e in events if e.event == "complete").data["content"] == ANSWER


async def test_off_mode_streams_draft_unchanged(turn_setup, monkeypatch):
    """GROUNDING_CHECK_ENABLED=False is pass-through, same as log_only — no
    captured "correct" mode, no buffering, no verifier call at all."""
    monkeypatch.setattr(config, "GROUNDING_CHECK_ENABLED", False)
    orch = _make_orchestrator(streaming_chunks=[PART1, PART2])
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    await asyncio.wait_for(asyncio.create_task(_consume(state, events)), 5)
    assert not turn_setup.started.is_set()

    assert any(e.event in ("message", "thinking") and DRAFT_TOKEN in json.dumps(e.data) for e in events)
    assert next(e for e in events if e.event == "complete").data["content"] == ANSWER


# --- Captured mode: a mid-turn config flip must not change THIS turn. ------

async def test_captured_mode_survives_mid_turn_flip_to_log_only(turn_setup, monkeypatch):
    """The mode is read once at turn start (ctx.grounding_mode). Flipping
    GROUNDING_MODE to log_only after the turn started must not un-buffer it.
    (The reverse flip is already pinned by test_sep09_speed_batch.py; this is
    the direction that pin does not exercise.)"""
    monkeypatch.setattr(config, "GROUNDING_MODE", "correct")
    orch = _make_orchestrator(streaming_chunks=[PART1, PART2])
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    await asyncio.wait_for(turn_setup.started.wait(), 10)
    assert not any(e.event == "complete" for e in events)
    monkeypatch.setattr(config, "GROUNDING_MODE", "log_only")
    turn_setup.release.set()
    await asyncio.wait_for(task, 5)

    _no_draft_before_complete(events)
    assert next(e for e in events if e.event == "complete").data["content"] == REVISED


# --- Early-return flush: an error/empty reply with no final chunk ships. ---

async def test_early_return_flush_shows_error_not_draft(turn_setup, monkeypatch):
    """Correct mode, empty model response (handlers.py ~4290-4295: chunk_count
    == 0 -> yield error_msg, then return with no final/debug chunk). The
    verifier never fires (the route returns before the grounding call); the
    flush still delivers the error message, never the nonexistent draft."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "correct")
    orch = _make_orchestrator(streaming_chunks=[])
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    await asyncio.wait_for(asyncio.create_task(_consume(state, events)), 5)
    assert not turn_setup.started.is_set()

    content = next(e for e in events if e.event == "complete").data["content"]
    assert "empty response" in content.lower() or "⚠️" in content
    assert DRAFT_TOKEN not in content


# --- Gradio /admin path: iterates handle_submit directly (no chat_service). -

async def test_gradio_path_handle_submit_direct_no_draft_before_final(turn_setup, monkeypatch):
    """gui/launch.py's Gradio /admin path iterates handle_submit directly, so
    buffering must cover it too (it lives inside handle_submit itself)."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "correct")
    orch = _make_orchestrator(streaming_chunks=[PART1, PART2])
    orch._last_turn_signals = {}
    orch.active_documents = None
    chunks = []

    async def _drive():
        async for c in handlers.handle_submit("Explain the sample record.", None, [], False, orch):
            chunks.append(c)

    task = asyncio.create_task(_drive())
    await asyncio.wait_for(turn_setup.started.wait(), 10)
    turn_setup.release.set()
    await asyncio.wait_for(task, 5)

    final_idx = next(i for i, c in enumerate(chunks) if isinstance(c, dict) and "debug" in c)
    for c in chunks[:final_idx]:
        assert DRAFT_TOKEN not in json.dumps(c)
    assert chunks[final_idx]["content"] == REVISED


# --- Direct wrapper test: the one shape neither route can produce itself. --

async def test_buffer_wrapper_reraises_without_flushing_on_exception():
    """An exception from the wrapped generator (including CancelledError —
    unguarded like any other exception here) propagates WITHOUT flushing the
    held draft, and closes the inner generator via aclose() in `finally` —
    never repeat the cancel-hole shape this wrapper closes."""
    closed = []

    async def inner():
        try:
            yield {"role": "assistant", "content": PART1}
            raise RuntimeError("synthetic mid-stream failure")
        finally:
            closed.append(True)

    ctx = SimpleNamespace(grounding_mode="correct")
    seen = []
    with pytest.raises(RuntimeError, match="synthetic mid-stream failure"):
        async for chunk in handlers._buffer_grounding_draft(ctx, inner()):
            seen.append(chunk)

    assert all(DRAFT_TOKEN not in json.dumps(c) for c in seen)
    # Only the one progress announcement was yielded — nothing held flushed.
    assert seen == [{
        "role": "assistant", "content": handlers._GROUNDING_CHECK_PROGRESS_TEXT, "is_progress": True,
    }]
    assert closed == [True]  # inner's own cleanup still ran


async def test_buffer_wrapper_aclose_closes_inner_generator():
    """aclose() on the wrapper closes the inner generator too — so its own
    cleanup still runs if the wrapper is torn down mid-turn with no
    exception in play (the enhanced route's storage `finally:` block)."""
    closed = []

    async def inner():
        try:
            yield {"role": "assistant", "content": PART1}
        finally:
            closed.append(True)

    wrapped = handlers._buffer_grounding_draft(SimpleNamespace(grounding_mode="correct"), inner())
    await wrapped.__anext__()  # the "Checking facts…" progress chunk
    await wrapped.aclose()
    assert closed == [True]


# --- F1 (A05b-2 review, deferred to A05b-3): agentic fall-through must not -
# --- leak its partial draft when the route ends unhandled. -----------------
# docs/execution/generalization/A05_design.md, Revision 2026-09-13, F1;
# batches/A05b-2.md §12 "F1 (defect, deferred to A05b-3)".

AGENTIC_PARTIAL_TOKEN = "AGENTIC_PARTIAL_DRAFT_MARKER_XYZ"


async def test_agentic_unhandled_failure_does_not_leak_partial_before_enhanced(turn_setup, monkeypatch):
    """(b) FAILING FIRST: the agentic route streams partial content, then
    fails internally (ctx.handled stays False, per its documented contract),
    so the dispatcher falls through to the enhanced route. Before the fix,
    the wrapper's exhaustion flush shipped that partial, unreviewed agentic
    draft as a message event ahead of the enhanced route's reviewed
    answer."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "correct")

    async def _agentic_fails_after_partial(*args, **kwargs):
        yield AGENTIC_PARTIAL_TOKEN
        raise RuntimeError("synthetic agentic mid-stream failure")

    orch = _make_orchestrator(agentic_enabled=True, streaming_chunks=[PART1, PART2])
    orch.agentic_controller.run_agentic_search = _agentic_fails_after_partial
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    await asyncio.wait_for(turn_setup.started.wait(), 10)
    turn_setup.release.set()
    await asyncio.wait_for(task, 5)

    complete_idx = next(i for i, e in enumerate(events) if e.event == "complete")
    for e in events[:complete_idx]:
        assert AGENTIC_PARTIAL_TOKEN not in json.dumps(e.data), (
            f"F1: partial agentic draft leaked in pre-complete {e.event}: {e.data}"
        )
    assert events[complete_idx].data["content"] == REVISED


async def test_buffer_wrapper_flush_on_exhaust_gates_the_held_chunk():
    """F1 mechanism, directly on the wrapper: flush_on_exhaust() False (the
    agentic dispatch's new argument, `lambda: ctx.handled`, when the route
    ends unhandled) drops the held chunk on exhaustion instead of flushing
    it. True -- the default, and every handled early return (watchdog/
    friendly, which set ctx.handled before returning) -- still flushes
    exactly as A05b-2 shipped it."""
    def inner():
        async def _gen():
            yield {"role": "assistant", "content": AGENTIC_PARTIAL_TOKEN}
        return _gen()

    ctx = SimpleNamespace(grounding_mode="correct")

    dropped = [c async for c in handlers._buffer_grounding_draft(ctx, inner(), flush_on_exhaust=lambda: False)]
    assert dropped == [{
        "role": "assistant", "content": handlers._GROUNDING_CHECK_PROGRESS_TEXT, "is_progress": True,
    }]

    flushed = [c async for c in handlers._buffer_grounding_draft(ctx, inner())]
    assert flushed[-1] == {"role": "assistant", "content": AGENTIC_PARTIAL_TOKEN}
