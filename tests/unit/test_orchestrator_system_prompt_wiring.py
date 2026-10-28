"""Behavioural siblings for orchestrator system-prompt wiring (source-text rows).

Incident dates: 2026-09-10 (CONCERN request-shaped carve-out never reached by a
query that the orchestrator did not forward), 2026-09-11 (task directive must not
carry an intent style block), 2026-08-28 (accuracy clause rendered twice), plan
injection gate (None plan -> nothing injected). Class: BC-63 (a grep for the call
text stands in for the behaviour), BC-04 (wiring dropped at a call site).

Every test builds the prompt through THE deployed ``DaemonOrchestrator.
_build_system_prompt`` / ``build_full_prompt`` and asserts what the model would be
told. Fixture text is synthetic; queries are asserted clean AND wrapped/indented.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

from core.context_pipeline import ToneLevel
from core.escalation_tracker import EscalationTracker, ResponseStrategy
from core.grounding_check import GROUNDING_ACCURACY_CLAUSE
from core.response_planner import ResponsePlan, ResponsePlanner
from tests.unit.test_process_user_query import _make_bfp_orch, _make_context
from utils.emotional_context import EmotionalContext
from utils.need_detector import NeedType
from utils.tone_detector import CrisisLevel

REQUEST = "What time should I take it tonight to get to bed?"
VENT = "ugh i feel so stressed about everything today"


def _emotional(level, need=NeedType.NEUTRAL):
    return EmotionalContext(
        crisis_level=level, need_type=need, tone_confidence=0.9, need_confidence=0.9,
        tone_trigger="test", need_trigger="test", explanation="test",
    )


def _wrapped(text):
    words = text.split()
    half = len(words) // 2
    return "    " + " ".join(words[:half]) + "\n    " + " ".join(words[half:]) + "\n   "


def _concern_prompt(query, **orch_kw):
    orch = _make_bfp_orch(**orch_kw)
    ctx = _make_context(
        original_query=query, tone_level=ToneLevel.CONCERN,
        emotional_context=_emotional(CrisisLevel.CONCERN),
    )
    return orch._build_system_prompt(ctx)


# --------------------------------------------- query reaches the CONCERN carve-out
@pytest.mark.parametrize("query", [REQUEST, _wrapped(REQUEST)])
def test_request_shaped_query_gets_the_answer_it_carve_out(query):
    prompt = _concern_prompt(query)
    assert "The user asked a direct question" in prompt
    assert "Don't offer unsolicited advice" not in prompt


@pytest.mark.parametrize("query", [VENT, _wrapped(VENT)])
def test_vent_gets_the_default_light_support_block(query):
    prompt = _concern_prompt(query)
    assert "Don't offer unsolicited advice" in prompt
    assert "The user asked a direct question" not in prompt


# ------------------------------------------------ query reaches the intent style call
def _intent_prompt(monkeypatch, query):
    monkeypatch.setattr("config.app_config.INTENT_STYLE_INSTRUCTIONS_ENABLED", True)
    orch = _make_bfp_orch()
    ctx = _make_context(
        original_query=query,
        intent=SimpleNamespace(intent="factual_recall", confidence=0.9),
    )
    return orch._build_system_prompt(ctx)


@pytest.mark.parametrize("query", [
    "when did we last talk about the thesis?",
    "    when did we last talk\n    about the thesis?\n  ",
])
def test_question_keeps_the_intent_style_block(monkeypatch, query):
    assert "INTENT STYLE: FACTUAL RECALL" in _intent_prompt(monkeypatch, query)


@pytest.mark.parametrize("query", [
    "jot down a note for this session: office hours are on Saturdays",
    "    jot down a note for this session:\n    office hours are on Saturdays\n  ",
])
def test_task_directive_drops_the_intent_style_block(monkeypatch, query):
    assert "INTENT STYLE" not in _intent_prompt(monkeypatch, query)


# ------------------------------------------------------- accuracy clause rendered once
@pytest.mark.parametrize("query", [VENT, _wrapped(VENT)])
def test_accuracy_clause_appears_once_when_tone_and_escalation_both_carry_it(query):
    tracker = EscalationTracker()
    tracker.current_strategy = ResponseStrategy.GROUNDING_PRESENCE
    assert GROUNDING_ACCURACY_CLAUSE in tracker.get_strategy_instructions()  # control
    prompt = _concern_prompt(query, escalation_tracker=tracker)
    assert prompt.count(GROUNDING_ACCURACY_CLAUSE) == 1
    assert "GROUNDING" in prompt.upper()  # the escalation block itself still rendered


# --------------------------------------------------------- plan injection gate
async def _full_prompt_with_plan(plan):
    planner = MagicMock()
    planner.should_plan = MagicMock(return_value=True)
    planner.format_plan_injection = ResponsePlanner.format_plan_injection
    planner.create_plan = AsyncMock(return_value=plan)
    orch = _make_bfp_orch(response_planner=planner)
    _, system_prompt = await orch.build_full_prompt(_make_context())
    return system_prompt, orch


@pytest.mark.asyncio
async def test_a_supported_plan_is_injected_into_the_system_prompt():
    plan = ResponsePlan(key_points=["answer the question"], tone="direct", strategy="answer")
    system_prompt, orch = await _full_prompt_with_plan(plan)
    assert "[RESPONSE PLAN]" in system_prompt and "answer the question" in system_prompt
    assert orch._current_response_plan is plan


@pytest.mark.asyncio
async def test_a_discarded_plan_injects_nothing():
    system_prompt, orch = await _full_prompt_with_plan(None)
    assert "[RESPONSE PLAN]" not in system_prompt
    assert orch._current_response_plan is None
