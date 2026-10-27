"""Behavioural siblings for the daemon-note and final-prompt wiring pins in
``AgenticSearchController`` (BC-63, oct27c SRC2, 2026-10-27).

Pins covered (source tests are KEPT):
  test_gate_action_offer_and_calendar_shape::test_source_wires_the_fallback_block_into_run_agentic_search (note)
  test_gate_action_offer_and_calendar_shape::test_source_wires_override_into_dispatch_single_inner
  test_gate_action_offer_and_calendar_shape::test_dispatch_chokepoints_mark_note_dispatched
  test_calendar_turn_round2::test_final_prompt_stashed_in_generate

The loop is the DEPLOYED ``run_agentic_search`` (scripted model per round via
``helpers_agentic_loop``); with ``real_dispatch=True`` the real
``_dispatch_single_inner`` chokepoint and DISPATCH_TABLE routing run and only
the note handler itself is a recorder. Fixtures are asserted clean and wrapped.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.agentic.types import AgenticSearchSession, SearchDecision, _ToolResult
from tests.unit.helpers_agentic_loop import make_controller, run_scripted_loop

NOTE_QUERY = "jot down a note for this session: Rowan's lab meets Saturdays at 11,"
NOTE_QUERY_WRAPPED = "jot down a note for this\n  session: Rowan's lab meets Saturdays\n  at 11,"
NOTE_BODY = "Rowan's lab meets Saturdays at 11"

DECLINE = SearchDecision(wants_answer=True, partial_response="Noted, I will keep that in mind.")


def _note_decision(summary="elaborated claim the user never made"):
    return SearchDecision(
        wants_create_daemon_note=True, daemon_note_title="Lab schedule",
        daemon_note_category="implementation", daemon_note_summary=summary,
    )


def _record_note_handler(ctrl, monkeypatch):
    seen = []

    async def handler(decision, round_number, *rest):
        seen.append(SimpleNamespace(
            summary=decision.daemon_note_summary,
            user_requested=decision.daemon_note_user_requested,
            title=decision.daemon_note_title))
        return _ToolResult(decision=decision, round_data=None, formatted_context="",
                           start_events=[], end_events=[])

    monkeypatch.setattr(ctrl._tool_executor, "_dispatch_create_daemon_note", handler)
    return seen


@pytest.mark.parametrize("query", [NOTE_QUERY, NOTE_QUERY_WRAPPED])
async def test_declined_note_request_is_saved_deterministically_after_the_loop(monkeypatch, query):
    run = await run_scripted_loop(
        monkeypatch, query=query, rounds=[[DECLINE]], action_query_ws=" ".join(query.split()),
    )
    notes = [d for d in run.dispatched_decisions if d.wants_create_daemon_note]
    assert len(notes) == 1
    assert notes[0].daemon_note_reason == "deterministic fallback: model declined to save"
    assert notes[0].daemon_note_summary == NOTE_BODY
    assert notes[0].daemon_note_user_requested is True
    assert run.session._note_force_fallback_sent is True
    # the reuse candidate is discarded so a real synthesis runs after the fallback
    assert len(run.final_calls) == 1


@pytest.mark.parametrize("query", [NOTE_QUERY, NOTE_QUERY_WRAPPED])
async def test_model_authored_note_suppresses_the_fallback_and_is_overridden_to_the_users_words(monkeypatch, query):
    ctrl = make_controller()
    seen = _record_note_handler(ctrl, monkeypatch)
    run = await run_scripted_loop(
        monkeypatch, query=query, controller=ctrl, real_dispatch=True,
        action_query_ws=" ".join(query.split()),
        rounds=[[_note_decision()], [DECLINE]],
    )
    # exactly one note reached the handler: the model's own, rewritten to the
    # user's stated body at the dispatch chokepoint; no second fallback note
    assert len(seen) == 1
    assert seen[0].summary == NOTE_BODY
    assert seen[0].user_requested is True
    assert run.session._note_dispatched is True
    assert getattr(run.session, "_note_force_fallback_sent", False) is False


async def test_note_without_a_note_request_keeps_the_models_own_summary(monkeypatch):
    ctrl = make_controller()
    seen = _record_note_handler(ctrl, monkeypatch)
    await run_scripted_loop(
        monkeypatch, query="what did we decide about the schedule", controller=ctrl,
        real_dispatch=True, rounds=[[_note_decision("the model's own summary")], [DECLINE]],
    )
    assert [s.summary for s in seen] == ["the model's own summary"]
    assert seen[0].user_requested is False


async def test_dispatch_chokepoint_marks_note_dispatched_only_for_a_note_decision(monkeypatch):
    ctrl = make_controller()
    _record_note_handler(ctrl, monkeypatch)
    session = AgenticSearchSession(query="q", max_rounds=3)
    await ctrl._dispatch_single_inner(_note_decision(), 1, session, None, None)
    assert session._note_dispatched is True
    other = AgenticSearchSession(query="q", max_rounds=3)
    await ctrl._dispatch_single_inner(SearchDecision(is_done=True), 1, other, None, None)
    assert getattr(other, "_note_dispatched", False) is False


async def test_generate_final_response_stashes_the_exact_prompt_for_regeneration():
    ctrl = make_controller()
    sent = {}

    async def generate_async(**kw):
        sent["prompt"], sent["system"], sent["model"] = kw["prompt"], kw["system_prompt"], kw["model_name"]
        return "A complete string answer."

    recovered_calls = []

    async def generate_once(**kw):
        recovered_calls.append(kw)
        return "Final answer. " * 30

    ctrl.model_manager.generate_async = generate_async
    ctrl.model_manager.generate_once = generate_once
    session = AgenticSearchSession(query="what is the plan", max_rounds=3)
    session.accumulated_context = "gathered fact: the plan is blue"
    chunks = [c async for c in ctrl._generate_final_response(
        "what is the plan", "sys", "model-x", session)]
    assert "".join(chunks) == "A complete string answer."
    assert ctrl._last_final_prompt == sent["prompt"]
    assert "the plan is blue" in ctrl._last_final_prompt

    # regeneration replays exactly that prompt, with the directive appended
    out = await ctrl.regenerate_final_answer()
    assert out is not None and out.startswith("Final answer.")
    assert recovered_calls[0]["prompt"].startswith(sent["prompt"])
    assert recovered_calls[0]["system_prompt"] == "sys"
    assert recovered_calls[0]["model_name"] == "model-x"
    assert recovered_calls[0]["disable_reasoning"] is True


async def test_regeneration_has_nothing_to_replay_until_a_final_prompt_was_built():
    ctrl = make_controller()
    ctrl.model_manager.generate_once = AsyncMock(return_value="x" * 400)
    assert await ctrl.regenerate_final_answer() is None
    ctrl.model_manager.generate_once.assert_not_called()
