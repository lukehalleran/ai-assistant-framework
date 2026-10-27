"""Behavioural siblings for the forced-write-action wiring pins in
``AgenticSearchController.run_agentic_search`` (BC-63, oct27c SRC2, 2026-10-27).

Each source-text pin (``"x" in inspect.getsource(run_agentic_search)``) proves a
name is still spelled in the loop, not that the loop uses it. These tests drive
the DEPLOYED loop through ``helpers_agentic_loop.run_scripted_loop`` (a scripted
model per decision round, recording tool dispatch) and assert what each round
received and what was dispatched.

Pins covered (source tests are KEPT):
  test_actions_registry_calendar_time_grounding::test_wired_into_run_agentic_search
  test_sep10_calendar_forced_action_loop::test_controller_declines_and_never_reforces
  test_calendar_turn_round2::test_forced_retry_wired_in_loop
  test_calendar_turn_round2::test_xml_branch_wired_in_force_block
  test_calendar_turn_round3::test_forced_prompt_retains_iteration_context
  test_gate_action_offer_and_calendar_shape::test_source_wires_the_fallback_block_into_run_agentic_search (calendar)
  test_gate_action_offer_and_calendar_shape::test_source_wires_ground_calendar_params_by_resolution_into_run_agentic_search
  test_handlers_calendar_offer_continuation::test_forced_action_reaches_the_controller

Every request fixture is asserted clean AND soft-wrapped (workflow section 3).
"""
from datetime import datetime

import pytest

from core.agentic.types import SearchDecision
from tests.unit.helpers_agentic_loop import run_scripted_loop

Q_RESOLVABLE = ("put a recurring calendar event on my google calendar for the ABC "
                "study group, Tuesdays at 3, through Dec 4")
Q_RESOLVABLE_WRAPPED = ("put a recurring calendar event on my google\n  calendar for the ABC "
                        "study group, Tuesdays at\n  3, through Dec 4")
Q_UNRESOLVABLE = "please add the Rowan TA session to my google calendar"
Q_UNRESOLVABLE_WRAPPED = "please add the Rowan TA session\n  to my google calendar"

CONTEXT_NO_CLOCK = {"recent_conversations": [{
    "query": "when is the Rowan TA session?",
    "response": "The Rowan TA session is on Friday evening, I do not know the exact time.",
    "timestamp": "2026-09-10 18:00:00",
}]}

NOT_ACTION = SearchDecision(wants_answer=True, partial_response="Sure, one moment.")


@pytest.fixture(autouse=True)
def _clock(monkeypatch):
    import core.actions.registry as registry
    monkeypatch.setattr(registry, "_current_wall_clock", lambda: datetime(2026, 9, 10, 19, 21))


def _invented_time_proposal(start="2026-09-11T17:00:00", end="2026-09-11T18:00:00"):
    return SearchDecision(
        wants_action=True, action_type="calendar_create_event",
        action_params={"summary": "Rowan TA session", "start_time": start, "end_time": end},
    )


@pytest.mark.parametrize("query", [Q_UNRESOLVABLE, Q_UNRESOLVABLE_WRAPPED])
async def test_invented_clock_time_in_forced_round_is_declined_not_dispatched(monkeypatch, query):
    run = await run_scripted_loop(
        monkeypatch, query=query, protocol="native",
        initial_context=CONTEXT_NO_CLOCK,
        rounds=[[_invented_time_proposal()]],
    )
    assert run.rounds[0].forced_action_type == "calendar_create_event"
    assert run.dispatched_decisions == []
    assert run.session._action_force_declined is True
    assert "[ACTION NOT PROPOSED]" in run.session.accumulated_context
    assert "start_time=2026-09-11T17:00:00" in run.session.accumulated_context


@pytest.mark.parametrize("query", [Q_UNRESOLVABLE, Q_UNRESOLVABLE_WRAPPED])
async def test_declined_force_is_never_re_forced(monkeypatch, query):
    run = await run_scripted_loop(
        monkeypatch, query=query, protocol="native",
        initial_context=CONTEXT_NO_CLOCK,
        rounds=[[_invented_time_proposal()], [NOT_ACTION], [NOT_ACTION]],
    )
    assert run.session._action_force_declined is True
    # round 1 forced; no later round may force again, narrow the tools, or
    # name propose_action (the loop continues unforced after a decline)
    for later in run.rounds[1:]:
        assert later.forced_action_type is None
        assert later.tool_choice == "auto"
        assert later.tools_override is None
    assert run.dispatched_decisions == []
    assert getattr(run.session, "_action_force_retry_sent", False) is False


@pytest.mark.parametrize("query", [Q_RESOLVABLE, Q_RESOLVABLE_WRAPPED])
async def test_model_clock_time_is_replaced_by_the_requests_own_resolution(monkeypatch, query):
    # the model proposes 17:00 for a request that says "Tuesdays at 3": the
    # loop replaces it with the deterministic resolution (15:00) and DISPATCHES
    # instead of declining
    run = await run_scripted_loop(
        monkeypatch, query=query, protocol="native",
        initial_context=CONTEXT_NO_CLOCK,
        action_query_ws=" ".join(query.split()),
        rounds=[[_invented_time_proposal("2026-09-15T17:00:00", "2026-09-15T18:00:00")]],
    )
    assert len(run.dispatched_decisions) == 1
    params = run.dispatched_decisions[0].action_params
    assert params["start_time"] == "2026-09-15T15:00:00"
    assert params["end_time"] == "2026-09-15T16:00:00"
    assert getattr(run.session, "_action_force_declined", False) is False


@pytest.mark.parametrize("query", [Q_RESOLVABLE, Q_RESOLVABLE_WRAPPED])
async def test_forced_round_with_no_action_is_retried_once_with_the_force_re_armed(monkeypatch, query):
    run = await run_scripted_loop(
        monkeypatch, query=query, protocol="native",
        action_query_ws=" ".join(query.split()),
        rounds=[[NOT_ACTION], [NOT_ACTION]],
    )
    assert run.rounds[0].forced_action_type == "calendar_create_event"
    assert run.rounds[1].forced_action_type == "calendar_create_event"  # re-armed
    assert run.rounds[1].tool_choice == {"type": "function", "function": {"name": "propose_action"}}
    assert run.session._action_force_retry_sent is True
    # ...and only once: a third forced round never appears
    assert all(r.forced_action_type is None for r in run.rounds[2:])


@pytest.mark.parametrize("query", [Q_RESOLVABLE, Q_RESOLVABLE_WRAPPED])
async def test_deterministic_calendar_fallback_mints_the_proposal_after_force_and_retry(monkeypatch, query):
    run = await run_scripted_loop(
        monkeypatch, query=query, protocol="native",
        action_query_ws=" ".join(query.split()),
        rounds=[[NOT_ACTION], [NOT_ACTION]],
    )
    assert len(run.dispatched_decisions) == 1
    d = run.dispatched_decisions[0]
    assert d.action_type == "calendar_create_event"
    assert d.action_reason == "deterministic fallback: model declined to propose"
    assert d.action_params["start_time"] == "2026-09-15T15:00:00"
    assert d.action_params["recurrence"].startswith("RRULE:FREQ=WEEKLY")
    assert run.session._action_dispatched is True
    assert run.session._action_force_fallback_sent is True


@pytest.mark.parametrize("query", [Q_RESOLVABLE, Q_RESOLVABLE_WRAPPED])
async def test_xml_protocol_force_round_carries_marker_prompt_not_native_tools(monkeypatch, query):
    run = await run_scripted_loop(
        monkeypatch, query=query, protocol="xml",
        action_query_ws=" ".join(query.split()),
        rounds=[[NOT_ACTION]],
    )
    first = run.rounds[0]
    assert first.forced_action_type == "calendar_create_event"
    assert first.tools_override is None  # XML path ignores native tool narrowing
    assert "<action" in first.prompt  # the marker syntax the model can actually emit
    assert "emit the <action> marker" in first.prompt
    assert "propose_action(" not in first.prompt  # native call vocabulary stays out
    assert "Do NOT narrate" in first.system_prompt


@pytest.mark.parametrize("query", [Q_RESOLVABLE, Q_RESOLVABLE_WRAPPED])
@pytest.mark.parametrize("protocol", ["native", "xml"])
async def test_forced_round_prompt_keeps_the_iteration_context(monkeypatch, query, protocol):
    run = await run_scripted_loop(
        monkeypatch, query=query, protocol=protocol,
        action_query_ws=" ".join(query.split()),
        rounds=[[NOT_ACTION]],
    )
    forced = run.rounds[0]
    assert forced.forced_action_type == "calendar_create_event"
    iteration_marker = "What would you like to do?"  # tail of _build_iteration_prompt
    assert iteration_marker in forced.prompt
    # the action directive is APPENDED after the iteration prompt, not instead of it
    assert forced.prompt.index(iteration_marker) < forced.prompt.index("[RESOLVED FIELDS]")


@pytest.mark.parametrize("query", ["yes", "  yes\n  please  "])
async def test_gate_supplied_forced_action_reaches_the_controller_loop(monkeypatch, query):
    run = await run_scripted_loop(
        monkeypatch, query=query, protocol="native",
        forced_action="calendar_create_event",
        rounds=[[NOT_ACTION]],
    )
    assert run.rounds[0].forced_action_type == "calendar_create_event"
    assert [t["function"]["name"] for t in run.rounds[0].tools_override] == ["propose_action"]
    # an unknown value is ignored, not forced
    run2 = await run_scripted_loop(
        monkeypatch, query=query, protocol="native",
        forced_action="not_an_action",
        rounds=[[NOT_ACTION]],
    )
    assert run2.rounds[0].forced_action_type is None
