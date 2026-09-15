"""B6 request latency: exercise deployed SSE, hooks, and idle handling."""
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
from core.grounding_check import GroundingVerdict, verify_grounding
from tests.unit.test_handle_submit import (
    _gate_decision, _make_file_processor_mock, _make_orchestrator,
)

ANSWER = "The sample record says the scheduled event starts tomorrow at three in the afternoon. Please check the calendar entry for its details."
REVISED = "The sample record says the scheduled event starts tomorrow at four in the afternoon. Please check the calendar entry for its details."
# A05b-1: integrator disabled -> build_integrated_fallback's standalone reply
# (the claim "sample time" does not locate uniquely in ANSWER), delivered
# through the revised path instead of the retired draft-plus-suffix shape.
FALLBACK_TEXT = (
    "Before answering, I found something that needs correcting: The sample "
    "time is four in the afternoon. Let me know if you'd like me to go ahead "
    "and answer."
)


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


@pytest.mark.parametrize("agentic", [False, True])
@pytest.mark.parametrize("integrate", [False, True])
async def test_log_only_completes_before_verifier_and_records_own_turn(turn_setup, monkeypatch, agentic, integrate):
    monkeypatch.setattr(config, "GROUNDING_MODE", "log_only")
    monkeypatch.setattr(config, "GROUNDING_INTEGRATE_ENABLED", integrate)
    orch = _make_orchestrator(agentic_enabled=agentic, streaming_chunks=[ANSWER], agentic_items=[ANSWER])
    orch._last_turn_signals = {"intent": "first-turn"}
    orch._last_task_timings = {"memories": 0.1234}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    try:
        await asyncio.wait_for(turn_setup.started.wait(), 10)
        await asyncio.wait_for(asyncio.shield(task), 1)
        complete = [e for e in events if e.event == "complete"]
        assert len(complete) == 1
        assert complete[0].data["content"] == ANSWER
        debug = state.session.debug_records[0]
        assert debug["response"] == ANSWER
        assert debug["grounding_status"] == "pending"
        assert not turn_setup.path.exists()
        assert turn_setup.storage.call_args.args[2] == ANSWER
        orch.escalation_tracker.record_response.assert_called_once_with(ANSWER)
        orch._last_turn_signals = {"intent": "next-turn"}
        orch._last_task_timings = {"wrong-turn": 99}
        # A live setting change cannot turn this already-shipped check into a correction.
        monkeypatch.setattr(config, "GROUNDING_MODE", "correct")
    finally:
        turn_setup.release.set()
        await asyncio.gather(task, return_exceptions=True)
        await handlers.wait_for_pending_storage(timeout=2)
        await asyncio.sleep(0)
    rows = [json.loads(line) for line in turn_setup.path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["intent"] == "first-turn"
    assert row["grounding_flagged"] is True
    assert row["grounding_mode"] == "log_only"
    assert row.get("grounding_corrected", False) is False
    assert row["task_timings"] == {"memories": 0.123}
    assert row["wall_elapsed_s"] >= 0
    assert state.session.debug_records[0]["grounding_flagged"] is True
    assert state.session.debug_records[0]["grounding_status"] == "complete"


@pytest.mark.parametrize("agentic", [False, True])
@pytest.mark.parametrize("integrate", [False, True])
async def test_correct_mode_waits_and_keeps_display_storage_equal(turn_setup, monkeypatch, agentic, integrate):
    monkeypatch.setattr(config, "GROUNDING_MODE", "correct")
    monkeypatch.setattr(config, "GROUNDING_INTEGRATE_ENABLED", integrate)
    orch = _make_orchestrator(agentic_enabled=agentic, streaming_chunks=[ANSWER], agentic_items=[ANSWER])
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    try:
        await asyncio.wait_for(turn_setup.started.wait(), 10)
        assert not any(e.event == "complete" for e in events)
        assert not task.done()
    finally:
        turn_setup.release.set()
        await asyncio.wait_for(task, 3)
    content = next(e.data["content"] for e in events if e.event == "complete")
    assert content == (REVISED if integrate else FALLBACK_TEXT)
    assert turn_setup.storage.call_args.args[2] == content
    assert state.session.debug_records[0]["response"] == content
    row = json.loads(turn_setup.path.read_text())
    assert row["grounding_corrected"] is True
    if not integrate:
        assert row["grounding_status"] == "fallback"
        assert row["grounding_fallback"] == "standalone:claim_not_located"


@pytest.mark.parametrize("outcome", ["timed_out", "failed", "cancelled"])
async def test_background_check_keeps_one_receipt_on_failure(turn_setup, monkeypatch, outcome):
    monkeypatch.setattr(config, "GROUNDING_MODE", "log_only")
    monkeypatch.setattr(config, "GROUNDING_TIMEOUT_S", 0.01)

    async def check(*args, **kwargs):
        if outcome == "failed":
            raise RuntimeError("synthetic verifier failure")
        await asyncio.Event().wait()

    monkeypatch.setattr(handlers, "_apply_grounding_check", check)
    ctx = SimpleNamespace(
        user_text="sample", telemetry={}, grounding_task=None,
        orchestrator=SimpleNamespace(_last_turn_signals={}), debug_record={},
    )
    assert await handlers._apply_grounding_check_for_delivery(ctx, ANSWER) == (None, "")
    handlers._write_turn_telemetry(ctx, "enhanced", "session", "model", len(ANSWER))
    if outcome == "cancelled":
        ctx.grounding_task.cancel()  # before the coroutine has ever run
    await asyncio.gather(ctx.grounding_task, return_exceptions=True)
    await asyncio.sleep(0)
    rows = [json.loads(line) for line in turn_setup.path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["grounding_status"] == outcome
    assert ctx.debug_record["grounding_status"] == outcome
    assert ctx.grounding_task not in handlers._pending_storage_tasks


async def test_disabled_grounding_never_schedules_work(turn_setup, monkeypatch):
    monkeypatch.setattr(config, "GROUNDING_CHECK_ENABLED", False)
    ctx = SimpleNamespace(telemetry={}, grounding_pending=None)
    assert await handlers._apply_grounding_check_for_delivery(ctx, ANSWER) == (None, "")
    assert ctx.grounding_pending is None
    assert not turn_setup.started.is_set()


@pytest.mark.parametrize("agentic", [False, True])
@pytest.mark.parametrize("mode", ["log_only", "correct"])
@pytest.mark.parametrize("outcome, expected", [
    ("timeout", "timed_out"), ("provider_error", "failed"),
    ("malformed", "failed"), ("valid", "complete"), ("demoted", "complete"),
])
async def test_real_verifier_outcomes_reach_receipt_and_debug(
    turn_setup, monkeypatch, agentic, mode, outcome, expected,
):
    """Keep the real verifier and fail-open wrapper; replace only provider I/O."""
    import core.grounding_check as grounding

    monkeypatch.setattr(grounding, "verify_grounding", verify_grounding)
    monkeypatch.setattr(config, "GROUNDING_MODE", mode)
    monkeypatch.setattr(config, "GROUNDING_TIMEOUT_S", 0.01)

    async def generate_once(*args, **kwargs):
        if outcome == "timeout":
            await asyncio.Event().wait()
        if outcome == "provider_error":
            raise RuntimeError("synthetic provider error")
        if outcome == "malformed":
            return "not a verdict"
        return json.dumps({
            "false_claim_present": outcome == "demoted",
            "claim": "sample time", "why_false": "", "confidence": 0.99,
            "correction": ANSWER if outcome == "demoted" else "",
        })

    orch = _make_orchestrator(
        agentic_enabled=agentic, streaming_chunks=[ANSWER], agentic_items=[ANSWER],
    )
    orch._last_turn_signals = {}
    orch.model_manager.generate_once = AsyncMock(side_effect=generate_once)
    state, events = AppState(orch), []
    await _consume(state, events)
    await handlers.wait_for_pending_storage(timeout=2)
    await asyncio.sleep(0)

    rows = [json.loads(line) for line in turn_setup.path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["grounding_verifier_fired"] is True
    assert row["grounding_status"] == expected
    assert state.session.debug_records[0]["grounding_status"] == expected
    assert row["grounding_verifier_elapsed_s"] >= 0
    assert row["pre_prepare_elapsed_s"] >= 0
    assert row["wall_elapsed_s"] >= row["pre_prepare_elapsed_s"]
    assert next(e.data["content"] for e in events if e.event == "complete") == ANSWER
    assert turn_setup.storage.call_args.args[2] == ANSWER
    orch.model_manager.generate_once.assert_awaited_once()


async def test_activity_is_poked_before_file_processing(monkeypatch):
    import main
    events = []
    monkeypatch.setattr(main, "update_activity_timestamp", lambda: events.append("activity"))

    async def process(*args):
        events.append("files")
        raise RuntimeError("stop at boundary")

    monkeypatch.setattr(handlers, "file_processor", SimpleNamespace(process_files_structured=process))
    monkeypatch.setattr(handlers, "get_conversation_logger", MagicMock())
    with pytest.raises(RuntimeError, match="stop at boundary"):
        async for _ in handlers.handle_submit("hello", None, [], False, SimpleNamespace()):
            pass
    assert events == ["activity", "files"]


async def test_short_turn_inflight_clears_on_cancel_and_idle_skips(monkeypatch):
    import main
    entered = asyncio.Event()

    async def inner(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
        yield {"content": "unreachable"}

    monkeypatch.setattr(handlers, "_handle_submit_inner", inner)
    gen = handlers.handle_submit("hi", None, [], False, SimpleNamespace())
    task = asyncio.create_task(anext(gen))
    await entered.wait()
    try:
        assert handlers.has_inflight_turns()
        monkeypatch.setattr(main, "_shutdown_requested", False)
        monkeypatch.setattr(main, "_last_activity_time", 0)
        monkeypatch.setattr(main, "_orchestrator_ref", object())
        shutdown = MagicMock()
        monkeypatch.setattr(main, "_run_shutdown_tasks", shutdown)
        sleeps = []

        def tick(*args):
            sleeps.append(1)
            if len(sleeps) == 2:
                main._shutdown_requested = True

        monkeypatch.setattr(main.time, "sleep", tick)
        main._idle_monitor_thread()
        shutdown.assert_not_called()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await gen.aclose()
    assert not handlers.has_inflight_turns()


async def test_image_progress_precedes_background_ingestion(monkeypatch):
    seen = []
    image = SimpleNamespace(error="", file_path="synthetic.png")
    result = SimpleNamespace(images=[image], documents=[], text_content="sample")
    monkeypatch.setattr(handlers, "file_processor", SimpleNamespace(process_files_structured=AsyncMock(return_value=result)))
    monkeypatch.setattr(handlers, "get_conversation_logger", MagicMock())

    async def persist(*args):
        seen.append("ingest")

    async def raw(ctx):
        await asyncio.sleep(0)
        yield {"content": "done"}

    monkeypatch.setattr(handlers, "_persist_uploads", persist)
    monkeypatch.setattr(handlers, "_run_raw", raw)
    orch = SimpleNamespace(active_documents=None)
    async for chunk in handlers.handle_submit("sample", None, [], True, orch):
        if chunk.get("is_progress") and "image" in chunk["content"]:
            seen.append("progress")
    await handlers.wait_for_pending_storage(timeout=2)
    assert seen == ["progress", "ingest"]


async def test_hung_turn_stops_blocking_idle_after_the_bound(monkeypatch):
    """Referee tightening: the in-flight guard is age-bounded so a hung turn
    cannot hold off the idle shutdown forever (pre-B6 behaviour for a hang)."""
    entered = asyncio.Event()

    async def inner(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
        yield {"content": "unreachable"}

    monkeypatch.setattr(handlers, "_handle_submit_inner", inner)
    gen = handlers.handle_submit("hi", None, [], False, SimpleNamespace())
    task = asyncio.create_task(anext(gen))
    await entered.wait()
    try:
        assert handlers.has_inflight_turns()
        assert handlers.has_inflight_turns(max_age_s=60)
        # Age the turn past the bound without sleeping.
        with handlers._turn_state_lock:
            for token in handlers._active_turn_starts:
                handlers._active_turn_starts[token] -= 3600
        assert handlers.has_inflight_turns()            # still accepted
        assert not handlers.has_inflight_turns(max_age_s=60)  # but no longer "activity"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await gen.aclose()
    assert not handlers.has_inflight_turns()
