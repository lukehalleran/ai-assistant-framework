"""Personal-claim support check — delivery wiring (gui.handlers, 2026-09-15).

The 2026-09-15 audit turn (docs/AUDIT_20260915_personal_event_grounding.md)
shipped "you reworked the resume enough to call it done, uploaded it" from a
contemplated upload and the assistant's own advice. The checker itself is
covered by test_personal_claim_check.py; this file drives THE deployed
handle_submit (via api.chat_service.submit_stream) and the handler helpers:

- log_only (default): the delivered text is unchanged, the receipt is deferred
  and still reaches storage provenance, the debug record and the turn row.
- correct (opt-in): delivery is buffered until the checker answers; the
  unsupported sentences are absent from BOTH display and storage.
- disabled: nothing is scheduled and no receipt is recorded.
- checker timeout / provider error / malformed output: explicit
  unavailable/failed receipts, text unchanged, never described as verified.
- storage waits for the receipt before persisting; a cancelled checker is a
  failed-open receipt, not a lost turn.

Fakes only: no LLM, no network, no live store. The scripted provider tests
transport and mechanical validation, not model accuracy.
"""
import asyncio
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import config.app_config as config
import core.personal_claim_check as pcc
import gui.handlers as handlers
from api.chat_service import submit_stream
from api.schemas import ChatRequest
from api.state import AppState
from core.personal_claim_check import PersonalClaimResult
from tests.unit.test_handle_submit import (
    _gate_decision, _make_file_processor_mock, _make_orchestrator,
)

QUERY = "No idea how it's almost 5, feel like I've thunk a total of like 3 thoughts today"
ANSWER = (
    "That's the sleep-deprivation time warp — 6.5 hours on a 3am crash doesn't feel like a day, "
    "it feels like a blur with occasional horizontal breaks. And honestly, you did think more "
    "than 3 thoughts: you reworked the resume enough to call it done, uploaded it, and had a "
    "whole tripwire-framework conversation. Fried brains just don't log those as \"thinking.\"\n\n"
    "Low bar for the rest of the evening. Nothing left today that can't survive until tomorrow."
)
# Exact spans the scripted checker reports; the omission removes their sentences.
UNSUPPORTED_SPANS = (
    "you reworked the resume enough to call it done",
    "uploaded it",
    "Nothing left today that can't survive until tomorrow",
)
SUPPORTED_SPAN = "had a whole tripwire-framework conversation"
KEPT_TEXT = "Fried brains just don't log those as \"thinking.\""

RECENT = [
    {
        "id": "c1", "timestamp": "2026-09-15T16:41:00-05:00",
        "query": "I could just upload what I have to get something up, but it needs rework.",
        "response": "You could upload it as-is; then the one task is upload-as-is.",
    },
    {
        "id": "c2", "timestamp": "2026-09-15T16:42:00-05:00",
        "query": "Actually, I did not upload it; I cancelled that plan.",
        "response": "Got it.",
    },
]


def _scripted_result(response):
    """A checked result over whichever of the known spans survive in `response`."""
    claims = [
        {"text": UNSUPPORTED_SPANS[0], "status": "insufficient", "kind": "personal_completion", "evidence": []},
        {"text": UNSUPPORTED_SPANS[1], "status": "contradicted", "kind": "personal_completion",
         "evidence": [{"source_id": "src_current_query", "quote": "thunk"}]},
        {"text": SUPPORTED_SPAN, "status": "supported", "kind": "discussion",
         "evidence": [{"source_id": "src_current_query", "quote": "3 thoughts"}]},
        {"text": UNSUPPORTED_SPANS[2], "status": "insufficient", "kind": "other", "evidence": []},
    ]
    claims = [c for c in claims if c["text"] in response]
    return PersonalClaimResult("checked", "ok", claims=claims, elapsed_s=0.2)


@pytest.fixture
def turn_setup(monkeypatch, tmp_path):
    import main  # noqa: F401 - keep cold startup imports outside async ordering checks
    import core.agentic.gate as gate
    monkeypatch.setattr(config, "GROUNDING_CHECK_ENABLED", False)
    monkeypatch.setattr(config, "PERSONAL_CLAIM_CHECK_ENABLED", True)
    monkeypatch.setattr(config, "PERSONAL_CLAIM_MODE", "log_only")
    monkeypatch.setattr(config, "TURN_TELEMETRY_ENABLED", True)
    path = tmp_path / "turns.jsonl"
    monkeypatch.setattr(config, "TURN_TELEMETRY_PATH", str(path))
    decision = _gate_decision()
    decision.insight_intent = None
    decision.deferred_request = None
    monkeypatch.setattr(gate, "evaluate_agentic_gate", AsyncMock(return_value=decision))
    monkeypatch.setattr(gate, "apply_intent_veto", lambda decision, *a, **kw: decision)
    monkeypatch.setattr(handlers, "file_processor", _make_file_processor_mock(QUERY))
    monkeypatch.setattr(handlers, "get_conversation_logger", MagicMock())
    monkeypatch.setattr(handlers, "_apply_action_guard", AsyncMock(return_value=""))
    storage = MagicMock()
    monkeypatch.setattr(handlers, "_dispatch_storage", storage)
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def audit(response, evidence, mm, **kwargs):
        calls.append((response, evidence, kwargs))
        started.set()
        await release.wait()
        return _scripted_result(response)

    # handlers imports the checker at call time, so the module attribute is
    # the patch point (import doctrine: patch-point lazies).
    monkeypatch.setattr(pcc, "audit_personal_claims", audit)
    return SimpleNamespace(path=path, started=started, release=release,
                           storage=storage, calls=calls)


async def _consume(state, events):
    async for event in submit_stream(ChatRequest(text=QUERY), state):
        events.append(event)


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# ---------------------------------------------------------------------------
# Route-level: log_only (default)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("agentic", [False, True])
async def test_log_only_delivers_unchanged_and_defers_receipt_to_every_sink(turn_setup, monkeypatch, agentic):
    orch = _make_orchestrator(agentic_enabled=agentic, streaming_chunks=[ANSWER], agentic_items=[ANSWER])
    orch._last_turn_signals = {"intent": "first-turn"}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    try:
        # Delivery completes while the checker is still waiting.
        await asyncio.wait_for(asyncio.shield(task), 5)
        complete = [e for e in events if e.event == "complete"]
        assert len(complete) == 1
        assert complete[0].data["content"] == ANSWER
        debug = state.session.debug_records[0]
        assert debug["response"] == ANSWER
        assert debug["personal_claim_status"] == "pending"
        assert debug["personal_claim_mode"] == "log_only"
        await asyncio.wait_for(turn_setup.started.wait(), 5)
        assert not turn_setup.path.exists()  # the turn row waits for the receipt
        assert turn_setup.storage.call_args.args[2] == ANSWER
        assert isinstance(turn_setup.storage.call_args.kwargs["personal_claim_task"], asyncio.Task)
        # A live setting change cannot turn this already-shipped check into a correction.
        monkeypatch.setattr(config, "PERSONAL_CLAIM_MODE", "correct")
    finally:
        turn_setup.release.set()
        await asyncio.gather(task, return_exceptions=True)
        await handlers.wait_for_pending_storage(timeout=2)
        await asyncio.sleep(0)
    rows = _rows(turn_setup.path)
    assert len(rows) == 1
    row = rows[0]
    assert row["intent"] == "first-turn"
    assert row["personal_claim_status"] == "checked"
    assert row["personal_claim_delivery"] == "unchanged"
    assert row["personal_claim_candidate_count"] == 4
    assert row["personal_claim_contradicted_count"] == 1
    assert row["personal_claim_insufficient_count"] == 2
    assert row["personal_claim_supported_count"] == 1
    provenance = turn_setup.storage.call_args.args[9]
    receipt = provenance["personal_claim_support"]
    assert receipt["status"] == "checked" and receipt["delivery"] == "unchanged"
    assert receipt["source_ids"] == ["src_current_query"]
    # The receipt carries counts and opaque IDs only -- never the claim text.
    assert "resume" not in json.dumps(receipt) and "resume" not in json.dumps(
        {k: v for k, v in row.items() if k.startswith("personal_claim_")})
    debug = state.session.debug_records[0]
    assert debug["response"] == ANSWER
    assert debug["personal_claim_status"] == "checked"
    assert debug["personal_claim_support"]["delivery"] == "unchanged"
    # Evidence handed to the checker keeps the user's own words as the user's.
    response_seen, evidence, kwargs = turn_setup.calls[0]
    assert response_seen == ANSWER
    assert evidence[0]["role"] == "user" and evidence[0]["text"] == QUERY


# ---------------------------------------------------------------------------
# Route-level: correct (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("agentic", [False, True])
async def test_correct_mode_buffers_and_omits_from_display_and_storage(turn_setup, monkeypatch, agentic):
    monkeypatch.setattr(config, "PERSONAL_CLAIM_MODE", "correct")
    orch = _make_orchestrator(agentic_enabled=agentic, streaming_chunks=[ANSWER], agentic_items=[ANSWER])
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    task = asyncio.create_task(_consume(state, events))
    try:
        await asyncio.wait_for(turn_setup.started.wait(), 10)
        # Nothing final has reached the wire while the checker is pending.
        assert not any(e.event == "complete" for e in events)
        assert not task.done()
    finally:
        turn_setup.release.set()
        await asyncio.wait_for(task, 5)
    content = next(e.data["content"] for e in events if e.event == "complete")
    checked_text = turn_setup.calls[0][0]
    assert content == pcc.omit_unsupported_claims(checked_text, _scripted_result(checked_text))
    for span in UNSUPPORTED_SPANS:
        assert span not in content
    assert "sleep-deprivation time warp" in content
    assert KEPT_TEXT in content
    # Whole sentence removed, no invented negation or replacement event.
    assert "did not" not in content and "didn't upload" not in content
    # Display == storage == debug record.
    assert turn_setup.storage.call_args.args[2] == content
    assert state.session.debug_records[0]["response"] == content
    row = _rows(turn_setup.path)[0]
    assert row["personal_claim_status"] == "checked"
    assert row["personal_claim_delivery"] == "omitted"
    assert row["personal_claim_mode"] if "personal_claim_mode" in row else True
    assert turn_setup.storage.call_args.args[9]["personal_claim_support"]["delivery"] == "omitted"
    assert turn_setup.storage.call_args.kwargs.get("personal_claim_task") is None


@pytest.mark.parametrize("agentic", [False, True])
async def test_correct_mode_with_only_supported_claims_ships_the_draft(turn_setup, monkeypatch, agentic):
    monkeypatch.setattr(config, "PERSONAL_CLAIM_MODE", "correct")
    clean = "That took real courage to send, and " + SUPPORTED_SPAN + "."

    async def audit(response, evidence, mm, **kwargs):
        turn_setup.calls.append((response, evidence, kwargs))
        return PersonalClaimResult("checked", "ok", claims=[
            {"text": SUPPORTED_SPAN, "status": "supported", "kind": "discussion",
             "evidence": [{"source_id": "src_current_query", "quote": "thunk"}]},
        ])

    monkeypatch.setattr(pcc, "audit_personal_claims", audit)
    orch = _make_orchestrator(agentic_enabled=agentic, streaming_chunks=[clean], agentic_items=[clean])
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    await asyncio.wait_for(_consume(state, events), 5)
    content = next(e.data["content"] for e in events if e.event == "complete")
    assert content == clean
    assert turn_setup.storage.call_args.args[2] == clean
    row = _rows(turn_setup.path)[0]
    assert row["personal_claim_delivery"] == "unchanged"
    assert row["personal_claim_supported_count"] == 1


# ---------------------------------------------------------------------------
# Route-level: disabled
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("agentic", [False, True])
async def test_disabled_schedules_nothing_and_records_nothing(turn_setup, monkeypatch, agentic):
    monkeypatch.setattr(config, "PERSONAL_CLAIM_CHECK_ENABLED", False)
    turn_setup.release.set()
    orch = _make_orchestrator(agentic_enabled=agentic, streaming_chunks=[ANSWER], agentic_items=[ANSWER])
    orch._last_turn_signals = {}
    state, events = AppState(orch), []
    await asyncio.wait_for(_consume(state, events), 5)
    await handlers.wait_for_pending_storage(timeout=2)
    assert turn_setup.calls == []
    assert next(e.data["content"] for e in events if e.event == "complete") == ANSWER
    assert turn_setup.storage.call_args.kwargs.get("personal_claim_task") is None
    assert "personal_claim_support" not in (turn_setup.storage.call_args.args[9] or {})
    row = _rows(turn_setup.path)[0]
    assert not any(k.startswith("personal_claim_") for k in row)
    assert not any(k.startswith("personal_claim_") for k in state.session.debug_records[0])


# ---------------------------------------------------------------------------
# Helper-level: mode capture, failures, storage wait, cancellation
# ---------------------------------------------------------------------------

def _ctx(mm=None, *, mode=None, raw_context=None, history=()):
    return SimpleNamespace(
        user_text=QUERY, telemetry={}, debug_record={},
        orchestrator=SimpleNamespace(model_manager=mm),
        raw_context=raw_context if raw_context is not None else {"recent_conversations": list(RECENT)},
        history=history, personal_claim_mode=mode,
        personal_claim_pending=None, personal_claim_task=None, personal_claim_receipt=None,
    )


async def test_captured_mode_wins_over_live_config(monkeypatch):
    """The mode is captured at ingress (SubmitContext.personal_claim_mode); a
    config flip mid-turn cannot change this turn's buffering or delivery."""
    monkeypatch.setattr(config, "PERSONAL_CLAIM_CHECK_ENABLED", True)
    monkeypatch.setattr(config, "PERSONAL_CLAIM_MODE", "correct")
    called = []

    async def audit(*a, **kw):
        called.append(a)
        return PersonalClaimResult("checked", "ok")

    monkeypatch.setattr(pcc, "audit_personal_claims", audit)
    ctx = _ctx(mm=MagicMock(), mode="log_only")
    assert await handlers._apply_personal_claim_check_for_delivery(ctx, ANSWER) is None
    assert called == []
    assert ctx.personal_claim_pending == ANSWER
    assert ctx.telemetry["personal_claim_status"] == "pending"
    # And the reverse: a turn captured as "off" runs nothing even when config says correct.
    ctx_off = _ctx(mm=MagicMock(), mode="off")
    assert await handlers._apply_personal_claim_check_for_delivery(ctx_off, ANSWER) is None
    assert called == [] and ctx_off.telemetry == {} and ctx_off.personal_claim_pending is None


class _Provider:
    """Scripted provider: parses the evidence JSON out of the deployed review
    prompt so it can cite REAL opaque source IDs, exactly as a model would."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.prompts = []

    async def generate_once(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        if self.behaviour == "timeout":
            await asyncio.sleep(5)
        if self.behaviour == "provider_error":
            raise RuntimeError("synthetic provider failure")
        if self.behaviour == "malformed":
            return "```json\n{\"claims\": []}\n```"
        evidence = json.loads(prompt.split("EVIDENCE:\n", 1)[1].split("\n\nDRAFT:\n", 1)[0])
        correction = next(r for r in evidence if r["role"] == "user" and "did not upload it" in r["text"])
        advice = next(r for r in evidence if r["role"] == "assistant" and "upload it as-is" in r["text"])
        claims = [
            {"text": UNSUPPORTED_SPANS[0], "status": "insufficient", "kind": "personal_completion", "evidence": []},
            {"text": UNSUPPORTED_SPANS[1], "status": "contradicted", "kind": "personal_completion",
             "evidence": [{"source_id": correction["source_id"], "quote": "I did not upload it"}]},
            {"text": SUPPORTED_SPAN, "status": "supported", "kind": "discussion",
             "evidence": [{"source_id": advice["source_id"], "quote": "upload it as-is"}]},
            {"text": UNSUPPORTED_SPANS[2], "status": "insufficient", "kind": "other", "evidence": []},
        ]
        if self.behaviour == "advice_as_completion":
            # The model tries to support the upload with the assistant's advice.
            claims[1] = {"text": UNSUPPORTED_SPANS[1], "status": "supported", "kind": "personal_completion",
                         "evidence": [{"source_id": advice["source_id"], "quote": "upload it as-is"}]}
        return json.dumps({"claims": claims})


async def test_exact_audit_turn_through_real_checker_in_correct_mode(monkeypatch):
    """End to end through the deployed evidence builder, review prompt, JSON
    validation and omission: corpus-shaped {query, response} records keep
    their roles, the correction is citable, and only the unsupported
    sentences are omitted."""
    monkeypatch.setattr(config, "PERSONAL_CLAIM_CHECK_ENABLED", True)
    provider = _Provider("valid")
    ctx = _ctx(mm=provider, mode="correct")
    receipt, revised = await handlers._apply_personal_claim_check(ctx, ANSWER, mode="correct")
    assert receipt["status"] == "checked" and receipt["delivery"] == "omitted"
    assert receipt["candidate_count"] == 4 and receipt["contradicted_count"] == 1
    for span in UNSUPPORTED_SPANS:
        assert span not in revised
    assert KEPT_TEXT in revised and "sleep-deprivation time warp" in revised
    # The prompt carried role-preserved evidence with the user's correction
    # as a USER row and the advice as an ASSISTANT row.
    prompt, kwargs = provider.prompts[0]
    evidence = json.loads(prompt.split("EVIDENCE:\n", 1)[1].split("\n\nDRAFT:\n", 1)[0])
    assert [r["role"] for r in evidence] == ["user", "user", "assistant", "user", "assistant"]
    assert kwargs.get("disable_reasoning") is True
    # Telemetry/debug carry the bounded receipt only.
    assert ctx.telemetry["personal_claim_status"] == "checked"
    assert ctx.debug_record["personal_claim_support"]["delivery"] == "omitted"
    assert "resume" not in json.dumps(ctx.debug_record["personal_claim_support"])


async def test_assistant_advice_cannot_support_the_upload_through_the_handler(monkeypatch):
    monkeypatch.setattr(config, "PERSONAL_CLAIM_CHECK_ENABLED", True)
    ctx = _ctx(mm=_Provider("advice_as_completion"), mode="correct")
    receipt, revised = await handlers._apply_personal_claim_check(ctx, ANSWER, mode="correct")
    assert receipt["status"] == "checked"
    assert UNSUPPORTED_SPANS[1] not in revised  # demoted to insufficient, then omitted
    assert receipt["contradicted_count"] == 0 and receipt["insufficient_count"] == 3


@pytest.mark.parametrize("behaviour,status,reason", [
    ("timeout", "unavailable", "timeout"),
    ("provider_error", "failed", "provider_error"),
    ("malformed", "failed", "invalid_json"),
])
async def test_checker_failures_are_explicit_and_fail_open(monkeypatch, behaviour, status, reason):
    monkeypatch.setattr(config, "PERSONAL_CLAIM_CHECK_ENABLED", True)
    monkeypatch.setattr(config, "PERSONAL_CLAIM_TIMEOUT_S", 0.01)
    ctx = _ctx(mm=_Provider(behaviour), mode="correct")
    receipt, revised = await handlers._apply_personal_claim_check(ctx, ANSWER, mode="correct")
    assert revised is None  # text unchanged in every failure shape
    assert receipt["status"] == status
    assert receipt["reason"] == reason
    assert receipt["delivery"] == "failed_open"
    assert receipt["candidate_count"] == 0 and receipt["supported_count"] == 0
    assert ctx.telemetry["personal_claim_status"] == status
    assert ctx.telemetry["personal_claim_delivery"] == "failed_open"
    assert ctx.debug_record["personal_claim_support"]["reason"] == reason
    # Never recorded as verified.
    assert ctx.telemetry["personal_claim_status"] != "checked"


async def test_model_manager_missing_is_unavailable_not_verified(monkeypatch):
    monkeypatch.setattr(config, "PERSONAL_CLAIM_CHECK_ENABLED", True)
    ctx = _ctx(mm=None, mode="correct")
    receipt, revised = await handlers._apply_personal_claim_check(ctx, ANSWER, mode="correct")
    assert revised is None
    assert (receipt["status"], receipt["reason"], receipt["delivery"]) == (
        "unavailable", "no_model", "failed_open")


async def test_storage_waits_for_the_deferred_receipt_before_persisting():
    orch = MagicMock()
    orch.memory_system.store_interaction = AsyncMock(return_value="mem-1")
    provenance = {"response_mode": "enhanced", "model_name": "m", "thinking_block": ""}
    gate = asyncio.Event()
    receipt = {"status": "checked", "reason": "ok", "candidate_count": 2, "supported_count": 0,
               "contradicted_count": 1, "insufficient_count": 1, "source_ids": [],
               "elapsed_s": 0.3, "delivery": "unchanged"}

    async def check():
        await gate.wait()
        provenance["personal_claim_support"] = dict(receipt)
        return receipt

    claim_task = asyncio.create_task(check())
    store = asyncio.create_task(handlers._background_store_interaction(
        orchestrator=orch, merged_input=QUERY, response_to_store=ANSWER,
        tags=["topic:general"], user_text=QUERY, final_output=ANSWER,
        personality="default", file_names=[], conversation_logger=MagicMock(),
        session_id="s", provenance=provenance, mode="enhanced",
        personal_claim_task=claim_task,
    ))
    await asyncio.sleep(0.05)
    assert orch.memory_system.store_interaction.await_count == 0
    gate.set()
    await asyncio.wait_for(store, 5)
    kwargs = orch.memory_system.store_interaction.await_args.kwargs
    assert kwargs["provenance"]["personal_claim_support"] == receipt
    assert kwargs["response"] == ANSWER


async def test_cancelled_checker_is_a_failed_open_receipt_and_storage_proceeds():
    orch = MagicMock()
    orch.memory_system.store_interaction = AsyncMock(return_value="mem-1")
    provenance = {"response_mode": "enhanced"}
    ctx = _ctx(mm=MagicMock(), mode="log_only")
    ctx.personal_claim_pending = ANSWER
    task = handlers._start_background_personal_claim(ctx, provenance)
    assert isinstance(task, asyncio.Task)
    assert handlers._start_background_personal_claim(ctx, provenance) is task  # one task per turn
    task.cancel()  # before the coroutine ever ran
    store = asyncio.create_task(handlers._background_store_interaction(
        orchestrator=orch, merged_input=QUERY, response_to_store=ANSWER,
        tags=[], user_text=QUERY, final_output=ANSWER, personality="default",
        file_names=[], conversation_logger=MagicMock(), session_id="s",
        provenance=provenance, mode="enhanced", personal_claim_task=task,
    ))
    await asyncio.wait_for(store, 5)
    await asyncio.sleep(0)
    assert orch.memory_system.store_interaction.await_count == 1
    assert ctx.telemetry["personal_claim_status"] == "failed"
    assert ctx.telemetry["personal_claim_reason"] == "cancelled"
    assert provenance["personal_claim_support"]["delivery"] == "failed_open"
    assert task not in handlers._pending_storage_tasks


def test_default_config_is_enabled_log_only():
    """Evaluation rollout default (docs/PLAN_20260915_personal_claim_support.md):
    on, log-only. Pinned so a config change is deliberate."""
    assert config.PERSONAL_CLAIM_CHECK_ENABLED is True
    assert config.PERSONAL_CLAIM_MODE == "log_only"


def test_both_routes_and_the_ingress_capture_are_wired():
    enhanced = inspect.getsource(handlers._run_enhanced)
    agentic = inspect.getsource(handlers._run_agentic_search)
    for src in (enhanced, agentic):
        assert "_apply_personal_claim_check_for_delivery(" in src
        assert "_start_background_personal_claim(" in src
        assert "personal_claim_task=" in src
    inner = inspect.getsource(handlers._handle_submit_inner)
    assert "ctx.personal_claim_mode = " in inner
    assert 'ctx.personal_claim_mode == "correct"' in inner
