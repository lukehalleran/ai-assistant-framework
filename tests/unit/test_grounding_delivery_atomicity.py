"""A05c: correct-mode delivery atomicity matrix, both routes.

docs/execution/generalization/A05_design.md, batch row A05c and contract
point "Atomicity" (one canonical `final_text` per turn; `display_text ==
stored_text == indexed_text == final_text` after one declared normalization).
Inherits batches/A05b-1.md "Contract for A05b-2", A05b-2.md §7/§11,
A05b-3.md §6/§11/§12 (D1), A05b-4.md §11/§13, A05b-5.md §11/§13.

Drives the deployed api.chat_service.submit_stream / gui.handlers.handle_submit
for both routes. Fakes only -- no LLM, no network, no live store;
`_dispatch_storage` is a MagicMock throughout. "Corpus/Chroma" is asserted
via that mock's call arguments -- the deepest seam available without
constructing a real corpus/Chroma instance (see docs/execution/generalization/
batches/A05c.md "Limitations").

Pruned cells (see batches/A05c.md for the full rationale):
- "flagged + integrator timeout" merges into "flagged + integrator failure
  (returns None)": `integrate_grounding_correction` (core/grounding_check.py
  ~1322-1327) already catches both `asyncio.TimeoutError` and a general
  `Exception` internally and returns None in both cases -- no caller,
  including this mocked seam, can observe them differently.
- log_only x verifier-outcome is pruned to one control cell per route:
  `_apply_grounding_check_for_delivery` returns immediately in log_only
  (gui/handlers.py:3093-3096) without calling the verifier synchronously at
  all, so there is no integrate-on/off or flagged/not-flagged branching to
  reach on the delivery path.
- log_only x cancel is out of this batch's matrix (A05_design.md's
  invariant list has no delivery/cancel semantics for log_only); already
  covered by test_grounding_cancel_storage_gate.py's
  test_log_only_cancel_keeps_storage_call /
  test_log_only_agentic_cancel_keeps_storage_call.
- correct-mode cancel x verifier-outcome is pruned to one blocking point per
  route (the batch draft's instruction): the cancellation lands before the
  verifier ever returns, so which outcome it *would* have returned is moot.
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

# Same distinctive-substring guard as test_grounding_buffered_delivery.py /
# test_grounding_cancel_storage_gate.py.
DRAFT_TOKEN = "three in the afternoon"
assert DRAFT_TOKEN in ANSWER and DRAFT_TOKEN not in REVISED and DRAFT_TOKEN not in FALLBACK_TEXT

_FLAGGED_VERDICT = GroundingVerdict(
    false_claim_present=True, claim="sample time", why_false="sample mismatch",
    confidence=0.99, correction="The sample time is four in the afternoon.",
)
# Genuinely "not flagged" (false_claim_present False) -- distinct from a
# flagged-but-suppressed verdict, which is a different axis value.
_UNFLAGGED_VERDICT = GroundingVerdict(
    false_claim_present=False, claim="", why_false="", confidence=0.95, correction="",
)


def _verify_immediate(verdict, *, status):
    """A non-blocking verify_grounding replacement. Sets telemetry the same
    way the real function does for the outcome under test (verdict is not
    None -> "complete"; verdict is None -> "failed", the invalid-verdict /
    provider-error branch -- core/grounding_check.py:814-835)."""
    async def _verify(*args, **kwargs):
        telemetry = kwargs.get("telemetry")
        if telemetry is not None:
            telemetry["grounding_status"] = status
        return verdict
    return _verify


def _verify_blocking(started, release, verdict):
    """Blocks on `release` after signalling `started` -- the one blocking
    point used for the cancel-during-review cells (mirrors turn_setup's
    verify() in test_grounding_cancel_storage_gate.py)."""
    async def _verify(*args, **kwargs):
        started.set()
        await release.wait()
        return verdict
    return _verify


@pytest.fixture
def turn_setup(monkeypatch, tmp_path):
    import main  # noqa: F401 - keep cold startup imports outside async ordering checks
    import core.agentic.gate as gate
    import core.grounding_check as grounding
    monkeypatch.setattr(config, "GROUNDING_CHECK_ENABLED", True)
    monkeypatch.setattr(config, "GROUNDING_MIN_RESPONSE_CHARS", 1)
    monkeypatch.setattr(config, "GROUNDING_INTEGRATE_ENABLED", True)
    monkeypatch.setattr(config, "TURN_TELEMETRY_ENABLED", True)
    path = tmp_path / "turns.jsonl"
    monkeypatch.setattr(config, "TURN_TELEMETRY_PATH", str(path))
    monkeypatch.setattr(grounding, "has_checkable_claims", lambda *a: True)
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
    return SimpleNamespace(
        path=path, started=started, release=release, storage=storage, grounding=grounding,
    )


async def _consume(state, events):
    async for event in submit_stream(ChatRequest(text="Explain the sample record."), state):
        events.append(event)


def _rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _no_draft_before_complete(events):
    complete_idx = next(i for i, e in enumerate(events) if e.event == "complete")
    for e in events[:complete_idx]:
        assert DRAFT_TOKEN not in json.dumps(e.data), f"draft leaked in pre-complete {e.event}: {e.data}"
    return complete_idx


def _has_progress_text(events, text):
    return any(e.event == "progress" and text in (e.data.get("text") or "") for e in events)


def _make_orch(agentic):
    orch = _make_orchestrator(agentic_enabled=agentic, streaming_chunks=[ANSWER], agentic_items=[ANSWER])
    orch._last_turn_signals = {}
    return orch


# ============================================================================
# Correct mode, no cancel: route x verifier-outcome (2 x 4 = 8 cells).
# ============================================================================

@pytest.mark.parametrize("agentic", [False, True], ids=["enhanced", "agentic"])
@pytest.mark.parametrize(
    "outcome",
    ["integrator_success", "integrator_failure", "not_flagged", "verifier_failure"],
)
async def test_correct_mode_atomicity_matrix(turn_setup, monkeypatch, agentic, outcome):
    """complete == history == stored (_dispatch_storage args, normalized) ==
    debug record `response`, and the telemetry row's grounding_status /
    grounding_fallback match the outcome, for every (route, outcome) cell."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "correct")
    grounding = turn_setup.grounding
    integrate_mock = AsyncMock()
    monkeypatch.setattr(grounding, "integrate_grounding_correction", integrate_mock)

    if outcome == "integrator_success":
        monkeypatch.setattr(grounding, "verify_grounding", _verify_immediate(_FLAGGED_VERDICT, status="complete"))
        integrate_mock.return_value = REVISED
        expected = REVISED
    elif outcome == "integrator_failure":
        monkeypatch.setattr(grounding, "verify_grounding", _verify_immediate(_FLAGGED_VERDICT, status="complete"))
        integrate_mock.return_value = None
        expected = FALLBACK_TEXT
    elif outcome == "not_flagged":
        monkeypatch.setattr(grounding, "verify_grounding", _verify_immediate(_UNFLAGGED_VERDICT, status="complete"))
        expected = ANSWER
    else:  # verifier_failure
        monkeypatch.setattr(grounding, "verify_grounding", _verify_immediate(None, status="failed"))
        expected = ANSWER

    orch = _make_orch(agentic)
    state, events = AppState(orch), []
    await asyncio.wait_for(asyncio.create_task(_consume(state, events)), 5)

    _no_draft_before_complete(events)
    assert _has_progress_text(events, handlers._GROUNDING_CHECK_PROGRESS_TEXT)
    complete = next(e for e in events if e.event == "complete")
    assert complete.data["content"] == expected

    # complete == history == stored == debug record `response` (normalized
    # by the declared display-only-decoration strip -- a no-op on this
    # content, which carries no [WEB_N]/[WIKI_N] citation markers).
    assert state.session.history[-1]["content"] == expected
    stored_text = turn_setup.storage.call_args.args[2]
    assert handlers._strip_display_only_decorations(stored_text) == expected
    debug_response = state.session.debug_records[-1]["response"]
    assert handlers._strip_display_only_decorations(debug_response) == expected

    rows = _rows(turn_setup.path)
    assert len(rows) == 1
    row = rows[0]
    assert "delivery" not in row
    if outcome == "integrator_success":
        integrate_mock.assert_called_once()
        assert row.get("grounding_status") == "complete"
        assert "grounding_fallback" not in row
    elif outcome == "integrator_failure":
        integrate_mock.assert_called_once()
        assert row.get("grounding_status") == "fallback"
        assert row.get("grounding_fallback", "").startswith("standalone:")
    elif outcome == "not_flagged":
        integrate_mock.assert_not_called()
        assert row.get("grounding_status") == "complete"
        assert "grounding_fallback" not in row
        assert "grounding_corrected" not in row
    else:  # verifier_failure
        integrate_mock.assert_not_called()
        assert row.get("grounding_status") == "failed"
        assert "grounding_fallback" not in row
        assert DRAFT_TOKEN in complete.data["content"]  # unmodified, delivered once


# ============================================================================
# Correct mode, cancel during review: one blocking point per route (2 cells).
# ============================================================================

@pytest.mark.parametrize("agentic", [False, True], ids=["enhanced", "agentic"])
async def test_correct_mode_cancel_during_review_no_storage_one_receipt(turn_setup, monkeypatch, agentic):
    """No storage call; exactly one turn row with delivery ==
    "cancelled_before_review"; no draft token in any event or row; no
    assistant turn appended to session.history either."""
    monkeypatch.setattr(config, "GROUNDING_MODE", "correct")
    grounding = turn_setup.grounding
    monkeypatch.setattr(
        grounding, "verify_grounding",
        _verify_blocking(turn_setup.started, turn_setup.release, _FLAGGED_VERDICT),
    )
    orch = _make_orch(agentic)
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    await asyncio.wait_for(turn_setup.started.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    turn_setup.storage.assert_not_called()
    assert state.session.history[-1]["role"] == "user"  # no assistant entry
    rows = _rows(turn_setup.path)
    assert len(rows) == 1
    assert rows[0]["delivery"] == "cancelled_before_review"
    for row in rows:
        assert DRAFT_TOKEN not in json.dumps(row)
    assert all(DRAFT_TOKEN not in json.dumps(e.data) for e in events)


# ============================================================================
# log_only control: draft streams live; complete == history == stored ==
# debug once it lands; no delivery key (2 cells; verifier-outcome pruned --
# see module docstring).
# ============================================================================

@pytest.mark.parametrize("agentic", [False, True], ids=["enhanced", "agentic"])
async def test_log_only_streams_draft_and_atomicity_holds(turn_setup, monkeypatch, agentic):
    monkeypatch.setattr(config, "GROUNDING_MODE", "log_only")
    grounding = turn_setup.grounding
    monkeypatch.setattr(grounding, "verify_grounding", _verify_immediate(_FLAGGED_VERDICT, status="complete"))
    orch = _make_orch(agentic)
    state, events = AppState(orch), []
    await asyncio.wait_for(asyncio.create_task(_consume(state, events)), 5)
    await handlers.wait_for_pending_storage(timeout=2)

    assert any(e.event in ("message", "thinking") and DRAFT_TOKEN in json.dumps(e.data) for e in events)
    assert not _has_progress_text(events, handlers._GROUNDING_CHECK_PROGRESS_TEXT)
    complete = next(e for e in events if e.event == "complete")
    assert complete.data["content"] == ANSWER

    assert state.session.history[-1]["content"] == ANSWER
    stored_text = turn_setup.storage.call_args.args[2]
    assert handlers._strip_display_only_decorations(stored_text) == ANSWER
    debug_response = state.session.debug_records[-1]["response"]
    assert handlers._strip_display_only_decorations(debug_response) == ANSWER

    rows = _rows(turn_setup.path)
    assert len(rows) == 1
    assert "delivery" not in rows[0]
