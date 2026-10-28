"""Behavioural siblings for the handler source-text tests on the agentic route.

Runs the DEPLOYED ``handle_submit`` -> ``_run_agentic_search`` (helpers_handler_turn)
with a scripted controller and asserts what the route DOES: regenerates a
narration-shaped final answer, forwards the section outcomes, aborts a degenerate
stream before storage and without falling through to the enhanced route.

Incident class: BC-63 (source-text test proves spelling only). Lane SRCH, 2026-10-27.
Source tests kept: test_calendar_turn_round2::test_handlers_wiring,
test_section_outcome_receipts::test_two_callers_read_and_forward,
test_stream_artifacts (watchdog rows).
"""
from unittest.mock import AsyncMock

import pytest

from core.agentic.controller import AgenticSearchController
from tests.unit.helpers_handler_turn import make_gate_decision, run_handler_turn
from tests.unit.test_handle_submit import _make_orchestrator

pytestmark = pytest.mark.asyncio

QUERY = "Which dates from the sample syllabus belong on my calendar?"
WRAPPED = "Which dates from the\n   sample syllabus belong\n   on my calendar?"
NARRATION = (
    "I already pulled the syllabus details earlier today, so let me grab the "
    "full text back out of memory to make sure I have every date exact before "
    "we touch the calendar."
)
RECOVERED = (
    "Here are the dates from the sample syllabus: the first quiz is on the 14th, "
    "the project proposal is due on the 21st, and the final presentation lands on "
    "the 5th of the next month. Nothing else on the syllabus carried a date."
)


def _agentic_gate():
    return make_gate_decision(should_trigger=True, modes=["tools"])


def _orch(items, **controller_attrs):
    orch = _make_orchestrator(agentic_enabled=True, agentic_items=items)
    for name, value in controller_attrs.items():
        setattr(orch.agentic_controller, name, value)
    return orch


@pytest.mark.parametrize("text", [QUERY, WRAPPED], ids=["clean", "wrapped"])
async def test_narration_shaped_final_is_regenerated_before_delivery_and_storage(monkeypatch, text):
    real = object.__new__(AgenticSearchController)
    regenerate = AsyncMock(return_value=RECOVERED)
    orch = _orch([NARRATION], narration_shaped_final=real.narration_shaped_final,
                 regenerate_final_answer=regenerate)
    turn = await run_handler_turn(monkeypatch, text, orchestrator=orch, gate=_agentic_gate())

    assert real.narration_shaped_final(NARRATION) is True  # the fixture IS narration-shaped
    regenerate.assert_awaited_once()
    assert turn.debug["mode"] == "agentic-search"
    assert turn.final["content"] == RECOVERED
    assert turn.storage_calls[0].stored_text == RECOVERED
    assert turn.ctx.telemetry["agentic_narration_recovered"] is True


async def test_a_real_answer_is_not_regenerated(monkeypatch):
    real = object.__new__(AgenticSearchController)
    regenerate = AsyncMock(return_value=RECOVERED)
    orch = _orch([RECOVERED], narration_shaped_final=real.narration_shaped_final,
                 regenerate_final_answer=regenerate)
    turn = await run_handler_turn(monkeypatch, QUERY, orchestrator=orch, gate=_agentic_gate())
    regenerate.assert_not_awaited()
    assert turn.final["content"] == RECOVERED
    assert "agentic_narration_recovered" not in turn.ctx.telemetry


async def test_failed_regeneration_keeps_the_original_answer(monkeypatch):
    real = object.__new__(AgenticSearchController)
    regenerate = AsyncMock(return_value=None)
    orch = _orch([NARRATION], narration_shaped_final=real.narration_shaped_final,
                 regenerate_final_answer=regenerate)
    turn = await run_handler_turn(monkeypatch, QUERY, orchestrator=orch, gate=_agentic_gate())
    assert turn.final["content"] == NARRATION
    assert "agentic_narration_recovered" not in turn.ctx.telemetry


async def test_agentic_debug_record_forwards_the_orchestrators_section_outcomes(monkeypatch):
    outcomes = {"memories": "ok", "wiki": "timeout-sentinel"}
    turn = await run_handler_turn(
        monkeypatch, QUERY, gate=_agentic_gate(), agentic_items=["Done."],
        orch_attrs={"_last_section_outcomes": outcomes},
    )
    assert turn.debug["mode"] == "agentic-search"
    assert turn.debug["section_outcomes"] == outcomes


DEGENERATE_LINE = "the loop repeats this exact line again and again\n"
# 500-char pieces so the 2000-char watchdog threshold is crossed mid-stream.
DEGENERATE = [DEGENERATE_LINE * 10] * 6
LEGIT = [
    " ".join(f"Sentence {i + 10 * j} covers topic {(i + 10 * j) * 7} in its own detail."
             for i in range(10)) + " "
    for j in range(8)
]


async def test_degenerate_agentic_stream_aborts_without_storage_or_fallthrough(monkeypatch):
    turn = await run_handler_turn(monkeypatch, QUERY, gate=_agentic_gate(),
                                  agentic_items=DEGENERATE)
    assert "repetitive and incoherent" in turn.final["content"]
    assert turn.storage_calls == []
    assert turn.ctx.handled is True
    assert turn.stream_calls == []  # the enhanced route never ran as a fallback
    assert turn.events == []        # no storage tracking, no turn telemetry


async def test_long_legitimate_agentic_stream_is_delivered_and_stored(monkeypatch):
    turn = await run_handler_turn(monkeypatch, QUERY, gate=_agentic_gate(), agentic_items=LEGIT)
    body = "".join(LEGIT)
    assert len(body) > 2000
    assert turn.final["content"].strip() == body.strip()
    assert len(turn.storage_calls) == 1
    assert "repetitive" not in turn.final["content"]
