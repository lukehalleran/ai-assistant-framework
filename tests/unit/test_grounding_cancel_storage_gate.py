"""A05b-3: correct-mode cancel/failure storage gate + enhanced parity guard.

docs/execution/generalization/A05_design.md, contract points "Cancel before
review completes" and "Atomicity"; batches/A05b-2.md §11 "Contract for
A05b-3" (the storage-side gap A05b-2's buffering left open).

Drives the deployed api.chat_service.submit_stream / gui.handlers.handle_submit
for the enhanced route (agentic disabled so the gate under test is isolated).
Fakes only -- no LLM, no network, no live store; `_dispatch_storage` is a
MagicMock throughout.
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

# Same distinctive-substring guard as test_grounding_buffered_delivery.py:
# fails loudly here if a future wording change to these shared fixtures
# would otherwise silently defeat this file's "no draft leaked" checks.
DRAFT_TOKEN = "three in the afternoon"
assert DRAFT_TOKEN in ANSWER and DRAFT_TOKEN not in REVISED and DRAFT_TOKEN not in FALLBACK_TEXT


@pytest.fixture
def turn_setup(monkeypatch, tmp_path):
    import main  # noqa: F401 - keep cold startup imports outside async ordering checks
    import core.agentic.gate as gate
    import core.grounding_check as grounding
    monkeypatch.setattr(config, "GROUNDING_CHECK_ENABLED", True)
    monkeypatch.setattr(config, "GROUNDING_MIN_RESPONSE_CHARS", 1)
    monkeypatch.setattr(config, "TURN_TELEMETRY_ENABLED", True)
    path = tmp_path / "turns.jsonl"
    monkeypatch.setattr(config, "TURN_TELEMETRY_PATH", str(path))
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
    storage = MagicMock()
    monkeypatch.setattr(handlers, "_dispatch_storage", storage)
    started, release = asyncio.Event(), asyncio.Event()

    async def verify(*args, **kwargs):
        started.set()
        await release.wait()
        return GroundingVerdict(
            false_claim_present=True, claim="sample time", why_false="sample mismatch",
            confidence=0.99, correction="The sample time is four in the afternoon.",
        )

    monkeypatch.setattr(grounding, "verify_grounding", verify)
    return SimpleNamespace(path=path, started=started, release=release, storage=storage)


async def _consume(state, events):
    async for event in submit_stream(ChatRequest(text="Explain the sample record."), state):
        events.append(event)


def _rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _raising_after(text):
    """Async-gen-returning callable: yields one chunk, then raises."""
    async def _gen(*args, **kwargs):
        yield text
        raise RuntimeError("synthetic streaming failure")
    return _gen


# --- (a) + receipt: FAILING FIRST on the current code. ---------------------

async def test_cancel_before_review_enhanced_gate_and_receipt(turn_setup, monkeypatch):
    """Correct mode, enhanced route: cancel the consuming task while
    verify_grounding is blocked, so handle_submit's generator is torn down
    mid-review. Today _dispatch_storage IS called with the unreviewed draft
    (A05_design.md gap #3); the gate must stop that, and the turn must still
    leave a delivery="cancelled_before_review" receipt carrying no draft
    text."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "correct")
    orch = _make_orchestrator(agentic_enabled=False, streaming_chunks=[ANSWER])
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    await asyncio.wait_for(turn_setup.started.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    turn_setup.storage.assert_not_called()
    rows = _rows(turn_setup.path)
    assert len(rows) == 1
    assert rows[0]["delivery"] == "cancelled_before_review"
    for row in rows:
        assert DRAFT_TOKEN not in json.dumps(row)


# --- Streaming-error path: correct mode must not store the partial draft. --

async def test_streaming_error_correct_mode_no_partial_storage(turn_setup, monkeypatch):
    """The orchestrator stream raises after partial content; correct mode
    must not store the unreviewed partial final_output either (same gate as
    the cancel path above, different trigger -- the except-Exception exit
    never reaches the reviewed-delivery yield either)."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "correct")
    orch = _make_orchestrator(agentic_enabled=False, streaming_chunks=[ANSWER])
    orch.response_generator.generate_streaming_response = _raising_after(ANSWER)
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    await asyncio.wait_for(asyncio.create_task(_consume(state, events)), 5)
    assert not turn_setup.started.is_set()  # never reached the verifier

    turn_setup.storage.assert_not_called()
    content = next(e for e in events if e.event == "complete").data["content"]
    assert DRAFT_TOKEN not in content


# --- Controls: unaffected modes/paths keep today's behavior. ---------------

async def test_log_only_cancel_keeps_storage_call(turn_setup, monkeypatch):
    """Control: the gate is scoped to correct mode only. In log_only, a
    cancel at the same point (background verifier blocked) leaves the
    already-completed main turn's storage call intact."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "log_only")
    orch = _make_orchestrator(agentic_enabled=False, streaming_chunks=[ANSWER])
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    await asyncio.wait_for(turn_setup.started.wait(), 10)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    turn_setup.release.set()
    await handlers.wait_for_pending_storage(timeout=2)

    turn_setup.storage.assert_called_once()


async def test_correct_mode_happy_path_stores_reviewed_text(turn_setup, monkeypatch):
    """Control: an uncancelled correct-mode turn still stores exactly the
    reviewed text once review completes -- complete == stored, no receipt
    override."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "correct")
    orch = _make_orchestrator(agentic_enabled=False, streaming_chunks=[ANSWER])
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    await asyncio.wait_for(turn_setup.started.wait(), 10)
    turn_setup.release.set()
    await asyncio.wait_for(task, 5)

    content = next(e.data["content"] for e in events if e.event == "complete")
    assert content == REVISED
    turn_setup.storage.assert_called_once()
    assert turn_setup.storage.call_args.args[2] == content
    rows = _rows(turn_setup.path)
    assert len(rows) == 1
    assert "delivery" not in rows[0]


# --- Enhanced parity guard (log-only, mirrors the agentic one). ------------

async def test_enhanced_parity_guard_warns_on_divergence(turn_setup, monkeypatch, caplog):
    """Force a display/stored divergence -- _strip_echoed_headers (storage-
    only; _resp_for_debug never passes through it) is patched to change the
    stored body -- and a warning is logged; storage still proceeds
    unchanged (log-only guard, never mutates either text)."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "log_only")
    monkeypatch.setattr(
        handlers, "_strip_echoed_headers",
        lambda text: text.replace(DRAFT_TOKEN, "an entirely different time"),
    )
    orch = _make_orchestrator(agentic_enabled=False, streaming_chunks=[ANSWER])
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    try:
        await asyncio.wait_for(turn_setup.started.wait(), 10)
        await asyncio.wait_for(asyncio.shield(task), 1)
    finally:
        turn_setup.release.set()
        with caplog.at_level("WARNING"):
            await asyncio.gather(task, return_exceptions=True)
            await handlers.wait_for_pending_storage(timeout=2)

    turn_setup.storage.assert_called_once()
    assert any("display/storage body mismatch" in r.message for r in caplog.records)


async def test_enhanced_parity_guard_silent_when_bodies_agree(turn_setup, monkeypatch, caplog):
    """Control: no warning when the displayed and stored bodies agree."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "log_only")
    orch = _make_orchestrator(agentic_enabled=False, streaming_chunks=[ANSWER])
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    try:
        await asyncio.wait_for(turn_setup.started.wait(), 10)
        await asyncio.wait_for(asyncio.shield(task), 1)
    finally:
        turn_setup.release.set()
        with caplog.at_level("WARNING"):
            await asyncio.gather(task, return_exceptions=True)
            await handlers.wait_for_pending_storage(timeout=2)

    assert not any("display/storage body mismatch" in r.message for r in caplog.records)


# --- Parent review D1: teardown AFTER the reviewed final chunk is not a -----
# --- cancel before review. --------------------------------------------------

async def test_teardown_after_reviewed_final_chunk_still_stores(turn_setup, monkeypatch):
    """A05b-3 parent review D1. The consumer receives the reviewed final
    chunk, then closes handle_submit. That throws GeneratorExit at the final
    chunk's yield. The turn must not be recorded as cancelled_before_review,
    and the reviewed text must still be stored: A05_design.md only forbids
    storing when the cancel lands BEFORE review completes."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "correct")
    orch = _make_orchestrator(agentic_enabled=False, streaming_chunks=[ANSWER])
    orch._last_turn_signals = {}
    orch.active_documents = None
    agen = handlers.handle_submit("Explain the sample record.", None, [], False, orch)

    async def _until_final():
        async for chunk in agen:
            if isinstance(chunk, dict) and "debug" in chunk:
                return chunk

    task = asyncio.create_task(_until_final())
    await asyncio.wait_for(turn_setup.started.wait(), 10)
    turn_setup.release.set()
    final = await asyncio.wait_for(task, 5)
    assert final["content"] == REVISED
    await agen.aclose()
    # Closing handle_submit only raises GeneratorExit in ITS frame: its
    # `async for` does not close the wrapped route generators, which stay
    # suspended at the reviewed final chunk's yield. The event loop's
    # async-generator finalizer then closes them. That is the real path by
    # which _run_enhanced sees GeneratorExit after review, so let it run,
    # bounded, before asserting.
    import gc
    gc.collect()
    for _ in range(100):
        if turn_setup.storage.called or any(
            row.get("delivery") == "cancelled_before_review" for row in _rows(turn_setup.path)
        ):
            break
        await asyncio.sleep(0.02)

    turn_setup.storage.assert_called_once()
    assert turn_setup.storage.call_args.args[2] == REVISED
    assert all(row.get("delivery") != "cancelled_before_review" for row in _rows(turn_setup.path))


# ============================================================================
# A05b-4: agentic-route cancel-before-review receipt.
#
# docs/execution/generalization/A05_design.md, contract point "Cancel before
# review completes (correct mode only)"; batches/A05b-3.md §12 parent review,
# D2 -- the agentic route (`_run_agentic_search`) dispatches its own storage
# and turn telemetry inline (not in a `finally:`), so a BaseException
# unwinding before its reviewed final chunk (`_final_chunk`) was yielded
# already stores nothing (correct), but left no `delivery` receipt (the
# gap this batch closes). Same fixtures/mechanics as the enhanced-route
# tests above; agentic route enabled via `agentic_enabled=True`.
# ============================================================================

async def test_agentic_cancel_before_review_gate_and_receipt(turn_setup, monkeypatch):
    """FAILING FIRST on the current code. Correct mode, agentic route:
    cancel the consuming task while verify_grounding is blocked inside
    `_apply_grounding_check_for_delivery` (called from `_run_agentic_search`
    before its final yield, same as the enhanced route). Storage must not
    be called with the unreviewed draft, and the turn must still leave a
    delivery="cancelled_before_review" receipt carrying no draft text
    (A05b-3 parent review D2)."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "correct")
    orch = _make_orchestrator(agentic_enabled=True, agentic_items=[ANSWER])
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    await asyncio.wait_for(turn_setup.started.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    turn_setup.storage.assert_not_called()
    rows = _rows(turn_setup.path)
    assert len(rows) == 1
    assert rows[0]["delivery"] == "cancelled_before_review"
    for row in rows:
        assert DRAFT_TOKEN not in json.dumps(row)


async def test_agentic_teardown_after_reviewed_final_chunk_writes_no_cancel_receipt(turn_setup, monkeypatch):
    """Agentic counterpart to the A05b-3 parent review's D1 regression guard
    (test_teardown_after_reviewed_final_chunk_still_stores), narrowed to
    what this batch actually owns. The consumer receives the reviewed final
    chunk, then closes handle_submit. Per the TEST-MECHANISM NOTE, closing
    handle_submit does not close the wrapped route generator (it stays
    suspended at the reviewed final chunk's yield); the event loop's
    async-generator finalizer closes it later -- via `_buffer_grounding_draft`'s
    own `finally: await chunks.aclose()`, which throws GeneratorExit into
    `_run_agentic_search` at that same suspended yield -- so wait, bounded,
    before asserting.

    Unlike the enhanced route (whose storage dispatch already lives inside
    a `finally:` -- A05b-3), the agentic route's storage dispatch was plain
    sequential code AFTER `yield _final_chunk`. A GeneratorExit thrown into
    a suspended-at-yield generator propagates immediately without resuming
    that sequential code, so storage used not to be dispatched on this exact
    teardown path (A05b-4 §6 / BC-45). A05b-5 closes that gap by moving the
    storage + telemetry work into a dispatched-once helper, called on the
    normal path and from `finally:` when a teardown skipped it after review
    was already delivered -- this test now asserts the positive outcome."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "correct")
    orch = _make_orchestrator(agentic_enabled=True, agentic_items=[ANSWER])
    orch._last_turn_signals = {}
    agen = handlers.handle_submit("Explain the sample record.", None, [], False, orch)

    async def _until_final():
        async for chunk in agen:
            if isinstance(chunk, dict) and "debug" in chunk:
                return chunk

    task = asyncio.create_task(_until_final())
    await asyncio.wait_for(turn_setup.started.wait(), 10)
    turn_setup.release.set()
    final = await asyncio.wait_for(task, 5)
    assert final["content"] == REVISED
    await agen.aclose()
    # Closing handle_submit only raises GeneratorExit in ITS frame: its
    # `async for` does not close the wrapped route generator, which stays
    # suspended at the reviewed final chunk's yield. The event loop's
    # async-generator finalizer then closes it. Wait, bounded, for that.
    import gc
    gc.collect()
    for _ in range(100):
        if turn_setup.storage.called or any(
            row.get("delivery") == "cancelled_before_review" for row in _rows(turn_setup.path)
        ):
            break
        await asyncio.sleep(0.02)

    # A05b-5: the reviewed answer must be stored exactly once even on this
    # teardown path (closes A05b-4 §6 / BC-45).
    turn_setup.storage.assert_called_once()
    assert turn_setup.storage.call_args.args[2] == REVISED
    rows = _rows(turn_setup.path)
    assert len(rows) == 1
    assert "delivery" not in rows[0]


async def test_log_only_agentic_cancel_keeps_storage_call(turn_setup, monkeypatch):
    """Control: mirrors test_log_only_cancel_keeps_storage_call for the
    agentic route. The receipt is scoped to correct mode only -- in
    log_only, `_apply_grounding_check_for_delivery` never awaits
    verify_grounding synchronously (it just records ctx.grounding_pending
    and returns), so the agentic turn already completed and dispatched
    storage before the (separately scheduled, background) verifier is even
    reached. A cancel at that point leaves the already-completed turn's
    storage call intact and writes no `delivery` key -- today's behaviour,
    byte-identical."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "log_only")
    orch = _make_orchestrator(agentic_enabled=True, agentic_items=[ANSWER])
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    await asyncio.wait_for(turn_setup.started.wait(), 10)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    turn_setup.release.set()
    await handlers.wait_for_pending_storage(timeout=2)

    turn_setup.storage.assert_called_once()
    assert all("delivery" not in row for row in _rows(turn_setup.path))


async def test_agentic_exception_fallback_writes_no_agentic_receipt(turn_setup, monkeypatch):
    """Control, double-receipt evidence: the agentic route streams partial
    content then fails with a plain Exception (not a cancellation) --
    ctx.handled stays False, its documented contract -- so the dispatcher
    falls through to the enhanced route (mirrors the existing F1 test,
    test_agentic_unhandled_failure_does_not_leak_partial_before_enhanced,
    in test_grounding_buffered_delivery.py). A plain Exception is caught by
    `_run_agentic_search`'s existing `except Exception` clause, never the
    new `except BaseException` clause added by this batch, so the agentic
    route writes no receipt of its own; the enhanced route completes
    normally and owns the turn's one delivery. Exactly one telemetry row,
    with no `delivery` key (not a cancellation) -- proving no double
    receipt."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "correct")

    async def _agentic_fails_after_partial(*args, **kwargs):
        yield "A05B4_AGENTIC_PARTIAL_MARKER"
        raise RuntimeError("synthetic agentic mid-stream failure")

    orch = _make_orchestrator(agentic_enabled=True, streaming_chunks=[ANSWER])
    orch.agentic_controller.run_agentic_search = _agentic_fails_after_partial
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    await asyncio.wait_for(turn_setup.started.wait(), 10)
    turn_setup.release.set()
    await asyncio.wait_for(task, 5)

    content = next(e.data["content"] for e in events if e.event == "complete")
    assert content == REVISED
    turn_setup.storage.assert_called_once()
    rows = _rows(turn_setup.path)
    assert len(rows) == 1
    assert "delivery" not in rows[0]


# ============================================================================
# A05b-5: the agentic post-review teardown must dispatch storage exactly
# once (closes the A05b-4 §6 finding / BC-45: the reviewed answer the user
# saw was never stored on this exact path).
#
# docs/execution/generalization/A05_design.md, contract points "Cancel
# before review completes" and "Atomicity"; batches/A05b-4.md §6 (finding)
# and §13 parent review; batches/A05b-3.md §6/§12 D1 (the enhanced route's
# precedent -- storage lives in `finally:`, gated on `review_delivered`).
# ============================================================================

async def test_agentic_correct_mode_happy_path_stores_reviewed_text_once(turn_setup, monkeypatch):
    """Control for the A05b-5 dispatched-once flag: an uncancelled
    correct-mode agentic turn -- the consumer keeps iterating past the
    final chunk, so the normal post-yield path runs to completion the same
    way it always has -- must still store the reviewed text exactly once,
    not twice (the new `finally:` teardown branch must not also fire)."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "correct")
    orch = _make_orchestrator(agentic_enabled=True, agentic_items=[ANSWER])
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    await asyncio.wait_for(turn_setup.started.wait(), 10)
    turn_setup.release.set()
    await asyncio.wait_for(task, 5)

    content = next(e.data["content"] for e in events if e.event == "complete")
    assert content == REVISED
    turn_setup.storage.assert_called_once()
    assert turn_setup.storage.call_args.args[2] == content
    rows = _rows(turn_setup.path)
    assert len(rows) == 1
    assert "delivery" not in rows[0]


async def test_log_only_agentic_teardown_after_final_chunk_stores_once(turn_setup, monkeypatch):
    """A05b-5: the same post-review teardown race (a GeneratorExit thrown
    into the wrapped route generator -- suspended at its final chunk's
    yield -- by the event loop's async-generator finalizer once the
    consumer stops pulling right after that chunk) is not gated on
    grounding mode: nothing between the agentic route's final yield and its
    storage dispatch checks `ctx.grounding_mode`. Verify the fix closes it
    in log_only too, and that log_only's background grounding scheduling
    (`_start_background_grounding`, reached via `_write_turn_telemetry`) is
    unaffected -- same call, now reached via the `finally:` path instead of
    the normal path."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "log_only")
    orch = _make_orchestrator(agentic_enabled=True, agentic_items=[ANSWER])
    orch._last_turn_signals = {}
    agen = handlers.handle_submit("Explain the sample record.", None, [], False, orch)

    async def _until_final():
        async for chunk in agen:
            if isinstance(chunk, dict) and "debug" in chunk:
                return chunk

    task = asyncio.create_task(_until_final())
    final = await asyncio.wait_for(task, 5)
    # log_only never integrates -- the draft is delivered unchanged (mirrors
    # test_log_only_returns_input_unchanged_with_empty_suffix).
    assert DRAFT_TOKEN in final["content"]
    turn_setup.release.set()  # unblock any background verifier task
    await agen.aclose()
    import gc
    gc.collect()
    for _ in range(100):
        if turn_setup.storage.called:
            break
        await asyncio.sleep(0.02)

    turn_setup.storage.assert_called_once()
    assert turn_setup.storage.call_args.args[2] == final["content"]
    assert all("delivery" not in row for row in _rows(turn_setup.path))


# --- Parent review D3 (A05b-5): a failure AFTER the reviewed final chunk ----
# --- must not re-dispatch storage or fall back to a second answer. ----------

async def test_agentic_post_review_failure_stores_once_and_does_not_fall_back(turn_setup, monkeypatch):
    """A05b-5 parent review D3. Correct mode, agentic route. The reviewed
    final chunk is delivered, then something in the post-review block raises
    after `_dispatch_storage` has already run (here: the agentic turn
    telemetry write).

    Two outcomes are wrong:
    - the route's `except Exception` falling back to the enhanced route,
      which gives the user a second answer and a second storage call;
    - the `finally:` teardown branch dispatching storage again.

    Correct: exactly one `complete` event and exactly one storage call."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "correct")
    real_write = handlers._write_turn_telemetry
    calls = {"agentic": 0}

    def _flaky_write(ctx, mode, *args, **kwargs):
        if mode == "agentic-search" and calls["agentic"] == 0:
            calls["agentic"] += 1
            raise RuntimeError("synthetic telemetry failure after storage dispatch")
        return real_write(ctx, mode, *args, **kwargs)

    monkeypatch.setattr(handlers, "_write_turn_telemetry", _flaky_write)
    orch = _make_orchestrator(agentic_enabled=True, agentic_items=[ANSWER], streaming_chunks=[ANSWER])
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    await asyncio.wait_for(turn_setup.started.wait(), 10)
    turn_setup.release.set()
    await asyncio.wait_for(task, 10)

    assert calls["agentic"] == 1  # the synthetic failure really fired
    completes = [e for e in events if e.event == "complete"]
    assert len(completes) == 1
    assert completes[0].data["content"] == REVISED
    assert turn_setup.storage.call_count == 1
    assert turn_setup.storage.call_args.args[2] == REVISED
