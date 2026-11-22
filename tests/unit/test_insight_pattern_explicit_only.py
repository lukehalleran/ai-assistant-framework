"""Insight mode owns a pattern_temporal turn ONLY on the explicit command.

Plan v4 Fix A, option (i) (class BC-06): an implicit "how often / how many /
has my X changed" question is never taken over by insight mode at any origin
(deterministic detector, LLM pre-emption, Tier-4 classification). The existing
command ("pattern_scan ...", "run the pattern scan ...") still routes.

The deployed detector and gate are driven; only the model manager is faked.
The fake model ALWAYS answers needs_pattern_analysis=True, which a misbehaving
classifier could do for any turn.
"""
import json

import pytest

import utils.web_search_trigger as _wst
from core.agentic.gate import apply_intent_veto, evaluate_agentic_gate
from core.insight.detector import (
    _detect_deliberation_shape,
    detect_insight_request,
)


@pytest.fixture(autouse=True)
def _fresh_trigger_cache():
    # The LLM trigger decision is cached per query for the whole process
    # (utils/web_search_trigger._llm_trigger_cache) and tests/conftest.py does
    # not reset it: another test asking PREEMPT_QUESTION first would leave its
    # decision cached and the fake below would never be called.
    _wst._llm_trigger_cache.clear()
    yield
    _wst._llm_trigger_cache.clear()

# Pattern-arm shape (_PATTERN_TEMPORAL_PATTERNS "how often ... over the last month").
PATTERN_ARM_QUESTION = "How often have I gone running over the last month?"
# Routed by _detect_deliberation_shape on the base tree (implicit personal comparison).
DELIBERATION_QUESTION = "Has my sleep changed since I moved?"
# Pre-emption branch: pattern candidate (first person + operation word), detector None.
PREEMPT_QUESTION = (
    "Could my late work nights and my delivery orders be connected? "
    "Use online sources and whatever you have stored to evaluate it."
)
# Tier-4 branch: not a pattern candidate, info-seeking, detector None.
TIER4_QUESTION = "Why do the stars twinkle at night?"
EXPLICIT = "pattern_scan how often have I felt anxious this month"
# LongMemEval-shaped (ids 603deb26 and gpt4_4fc4f797).
LME_COUNT = ("How many times have I tried making a Negroni at home since my "
             "friend Emma showed me how to make it?")
LME_DATE_DIFF = ("How many days passed between the day I received feedback about my "
                 "car's suspension and the day I tested my new suspension setup?")


class AlwaysPatternModel:
    """Misbehaving classifier: flags every turn needs_pattern_analysis=True."""

    def __init__(self):
        self.calls = 0

    async def generate_once(self, *_a, **_k):
        self.calls += 1
        return json.dumps({
            "should_search": False, "confidence": 0.94, "reason": "pattern",
            "search_terms": [], "search_depth": "deep", "num_searches": 0,
            "needs_memory_search": False, "needs_knowledge_search": False,
            "needs_pattern_analysis": True, "needs_document_generation": False,
        })


def _not_pattern(intent):
    return intent is None or intent.kind != "pattern_temporal"


def _not_insight(decision):
    return "insight" not in (decision.modes or [])


def test_a_pattern_arm_question_without_command_is_not_owned():
    assert _not_pattern(detect_insight_request(PATTERN_ARM_QUESTION))


def test_b_deliberation_shape_without_command_is_not_owned():
    # Base tree routes this through _detect_deliberation_shape.
    assert _detect_deliberation_shape(DELIBERATION_QUESTION) is not None
    assert _not_pattern(detect_insight_request(DELIBERATION_QUESTION))


@pytest.mark.asyncio
async def test_c_llm_preemption_branch_does_not_own_implicit_question():
    model = AlwaysPatternModel()
    assert detect_insight_request(PREEMPT_QUESTION) is None
    decision = await evaluate_agentic_gate(PREEMPT_QUESTION, model_manager=model)
    assert model.calls >= 1, "fake never reached the LLM branch"
    assert _not_insight(decision)


@pytest.mark.asyncio
async def test_d_tier4_branch_does_not_own_implicit_question():
    model = AlwaysPatternModel()
    assert detect_insight_request(TIER4_QUESTION) is None
    decision = await evaluate_agentic_gate(TIER4_QUESTION, model_manager=model)
    assert model.calls >= 1, "fake never reached the Tier-4 call"
    assert _not_insight(decision)


@pytest.mark.asyncio
async def test_e_explicit_command_still_routes_detector_and_gate():
    intent = detect_insight_request(EXPLICIT)
    assert intent is not None and intent.kind == "pattern_temporal"
    decision = await evaluate_agentic_gate(EXPLICIT)
    assert decision.modes == ["insight"]


@pytest.mark.asyncio
async def test_f_explicit_command_twice_in_one_session():
    first = await evaluate_agentic_gate(EXPLICIT)
    second = await evaluate_agentic_gate("run the pattern scan on my sleep over the last week")
    assert first.modes == ["insight"]
    assert second.modes == ["insight"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tone", ["CrisisLevel.CONCERN", "CrisisLevel.HIGH"])
async def test_g_explicit_command_survives_elevated_tone(tone):
    decision = await evaluate_agentic_gate(EXPLICIT)
    vetoed = apply_intent_veto(decision, None, tone_level=tone, query=EXPLICIT)
    assert vetoed.modes == ["insight"]
    assert vetoed.should_trigger


@pytest.mark.parametrize("question", [LME_COUNT, LME_DATE_DIFF])
def test_h_longmemeval_shaped_questions_are_not_owned(question):
    assert _not_pattern(detect_insight_request(question))


def test_predicate_is_the_explicit_command_only():
    from core.insight.detector import is_explicit_pattern_command
    assert is_explicit_pattern_command(EXPLICIT)
    assert is_explicit_pattern_command("Please run the pattern scan on my sleep")
    assert not is_explicit_pattern_command(PATTERN_ARM_QUESTION)
    assert not is_explicit_pattern_command("")


# ---------------------------------------------------------------------------
# Request scope (review docs/REVIEW_LONGMEMEVAL_FIX_COMMITS.md, P2): the command
# counts only in the user's own request span, never inside a pasted document.
# ---------------------------------------------------------------------------

def _with_paste(head: str) -> str:
    """``head`` followed by a document-shaped paste with a command line inside it."""
    body = "\n".join(
        ["Document heading", "pattern_scan example from the manual"]
        + ["Reference entry: atmospheric observations and background material."] * 40
        + ["Short heading"] * 20
    )
    return head + "\n" + body


def test_paste_fixture_is_split_by_request_span():
    from utils.request_span import request_span
    assert request_span(_with_paste(TIER4_QUESTION)) == TIER4_QUESTION
    assert request_span(_with_paste(PREEMPT_QUESTION)) == PREEMPT_QUESTION


def test_predicate_ignores_a_command_inside_a_paste():
    from core.insight.detector import is_explicit_pattern_command
    assert not is_explicit_pattern_command(_with_paste(TIER4_QUESTION))
    assert is_explicit_pattern_command(_with_paste(EXPLICIT))


@pytest.mark.asyncio
@pytest.mark.parametrize("head", [TIER4_QUESTION, PREEMPT_QUESTION],
                         ids=["tier4", "preemption"])
async def test_command_inside_paste_does_not_own_the_turn(head):
    model = AlwaysPatternModel()
    query = _with_paste(head)
    assert detect_insight_request(query) is None
    decision = await evaluate_agentic_gate(query, model_manager=model)
    assert model.calls >= 1, "fake never reached an LLM branch"
    assert _not_insight(decision)


@pytest.mark.asyncio
async def test_command_in_request_span_before_a_paste_still_routes():
    query = _with_paste(EXPLICIT)
    intent = detect_insight_request(query)
    assert intent is not None and intent.kind == "pattern_temporal"
    decision = await evaluate_agentic_gate(query, model_manager=AlwaysPatternModel())
    assert decision.modes == ["insight"]


# ---------------------------------------------------------------------------
# The explicit route's engine keeps its full count and window (plan v4 Fix A
# tests: "the pattern engine keeps its full denominator and window"). Engine
# level, deployed run_pattern_query over a real CorpusManager on tmp_path.
# ---------------------------------------------------------------------------

def test_pattern_engine_counts_every_event_in_window_beyond_top_k(tmp_path):
    from datetime import datetime, timedelta

    from memory.corpus_manager import CorpusManager
    from memory.pattern_engine import PatternQuery, run_pattern_query

    now = datetime(2026, 8, 29, 15, 0, 0)
    entries = []

    def add(ts, query):
        entries.append({"query": query, "response": "ok", "timestamp": ts.isoformat()})

    for d in range(1, 60, 2):            # 30 mentions on 30 days inside 60 days
        add(now - timedelta(days=d), "I went for a long run by the river")
    for d in range(2, 42, 2):            # 20 other user turns inside the window
        add(now - timedelta(days=d, hours=5), "regular message about something else")
    for d in range(100, 105):            # 5 mentions outside the window
        add(now - timedelta(days=d), "I went for a long run by the river")
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(entries))
    corpus = CorpusManager(corpus_file=str(path))

    res = run_pattern_query(
        PatternQuery(dimension="topic_keyword", terms=["run"], window_days=60, now=now),
        corpus_manager=corpus)
    assert res.total == 30                    # every in-window mention, not a top-k slice
    assert res.active_days == 30
    assert res.denominator_total == 50        # every user turn in the window
    assert sum(b.count for b in res.buckets) == 30
    assert sum(b.denominator for b in res.buckets) == 50
