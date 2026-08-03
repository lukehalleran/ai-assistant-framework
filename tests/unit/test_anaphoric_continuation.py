"""
Regression tests for the 2026-07-28 anaphora-misclassification incident.

Mid long-covid conversation, the user sent "It was maybe 3 years of twice a
week then the year and a half of like 5 6 days a week" (his ILLNESS
frequency). The pipeline fresh-classified the fragment from surface keywords:
topic became "Exercise Routine", [THREAD CONTEXT] asserted a topic shift
("Follow the current query"), and the response planner — which sees only the
raw query + topic labels — produced an exercise-routine plan. The model
answered about the gym. The explicit repair on the next turn ("No I mean of
being sick I wasn't talking about working out") was classified as ANOTHER
fresh topic ("Being Sick") and misread as a new current illness.

Fixes under test:
1. query_checker.is_anaphoric_continuation() — detects pronoun-anchored
   fragments and referent corrections.
2. ContextPipeline._extract_topics() — such messages INHERIT the previous
   turn's topic instead of being fresh-classified.
3. orchestrator._thread_topic_shifted() — never asserts a topic shift on an
   anaphoric/corrective turn.
4. ResponsePlanner.create_plan() — includes the previous exchange in the
   planner prompt plus a fragment-resolution instruction.
"""

import asyncio

import pytest

from utils.query_checker import is_anaphoric_continuation


# The two live queries from the incident.
INCIDENT_FRAGMENT = (
    "It was maybe 3 years of twice and week then the year and a half "
    "of like 5 6 days a week"
)
INCIDENT_CORRECTION = "No I mean of being sick I wasn't talking about working out"


# ---------------------------------------------------------------------------
# 1. Detector
# ---------------------------------------------------------------------------

class TestIsAnaphoricContinuation:
    def test_incident_fragment_detected(self):
        assert is_anaphoric_continuation(INCIDENT_FRAGMENT)

    def test_incident_correction_detected(self):
        assert is_anaphoric_continuation(INCIDENT_CORRECTION)

    @pytest.mark.parametrize("q", [
        "That was the worst part honestly",
        "This started after the semester began",
        "They said it could last 6 months",
        "Those were the bad years",
        "I meant the withdrawal not the gym",
        "That's not what I was asking about",
        "I was referring to the illness",
        "no i didnt mean the class",
    ])
    def test_positive_cases(self, q):
        assert is_anaphoric_continuation(q)

    @pytest.mark.parametrize("q", [
        "",
        "thinking about switching to postgres",
        "What is anarchism?",
        "I slept 11 hours and feel like shit",
        "Ugh",
        "How does your memory scoring work",
        "My homework is due Sunday and I am behind on the videos",
    ])
    def test_negative_cases(self, q):
        assert not is_anaphoric_continuation(q)

    def test_long_pasted_content_not_flagged(self):
        # A long message that happens to start with "It" (e.g. pasted email)
        # is substantive enough to classify on its own.
        long_msg = "It " + "word " * 40
        assert not is_anaphoric_continuation(long_msg)


# ---------------------------------------------------------------------------
# 2. Topic inheritance in ContextPipeline._extract_topics
# ---------------------------------------------------------------------------

class _FakeTopicManager:
    def __init__(self, last_topic, fresh_topic="Exercise Routine"):
        self.last_topic = last_topic
        self.fresh_topic = fresh_topic
        self.fresh_calls = 0

    def get_primary_topic(self, text=None):
        self.fresh_calls += 1
        return self.fresh_topic


def _make_pipeline(topic_manager):
    from core.context_pipeline import ContextPipeline
    pipeline = ContextPipeline.__new__(ContextPipeline)
    pipeline.topic_manager = topic_manager
    return pipeline


class TestTopicInheritance:
    def test_anaphoric_fragment_inherits_previous_topic(self):
        tm = _FakeTopicManager(last_topic="Fear Of Long Covid")
        pipeline = _make_pipeline(tm)
        primary, topics = asyncio.run(pipeline._extract_topics(INCIDENT_FRAGMENT))
        assert primary == "Fear Of Long Covid"
        assert topics == ["Fear Of Long Covid"]
        assert tm.fresh_calls == 0  # classifier never consulted

    def test_correction_inherits_previous_topic(self):
        tm = _FakeTopicManager(last_topic="Fear Of Long Covid")
        pipeline = _make_pipeline(tm)
        primary, _ = asyncio.run(pipeline._extract_topics(INCIDENT_CORRECTION))
        assert primary == "Fear Of Long Covid"

    def test_standalone_query_fresh_classified(self):
        tm = _FakeTopicManager(last_topic="Fear Of Long Covid",
                               fresh_topic="Postgres Migration")
        pipeline = _make_pipeline(tm)
        primary, _ = asyncio.run(
            pipeline._extract_topics("thinking about switching to postgres")
        )
        assert primary == "Postgres Migration"
        assert tm.fresh_calls == 1

    def test_no_previous_topic_falls_through_to_classifier(self):
        tm = _FakeTopicManager(last_topic=None)
        pipeline = _make_pipeline(tm)
        primary, _ = asyncio.run(pipeline._extract_topics(INCIDENT_FRAGMENT))
        assert primary == "Exercise Routine"
        assert tm.fresh_calls == 1

    def test_general_previous_topic_not_inherited(self):
        # "general" carries no signal — don't inherit it.
        tm = _FakeTopicManager(last_topic="general")
        pipeline = _make_pipeline(tm)
        asyncio.run(pipeline._extract_topics(INCIDENT_FRAGMENT))
        assert tm.fresh_calls == 1


# ---------------------------------------------------------------------------
# 3. Thread-shift assertion guard
# ---------------------------------------------------------------------------

class TestThreadTopicShifted:
    def test_no_shift_asserted_on_anaphoric_fragment(self):
        from core.orchestrator import _thread_topic_shifted
        # Exactly the incident: topics diverge, but the message is anaphoric.
        assert not _thread_topic_shifted(
            "Fear Of Long Covid", "Exercise Routine", INCIDENT_FRAGMENT
        )

    def test_no_shift_asserted_on_correction(self):
        from core.orchestrator import _thread_topic_shifted
        assert not _thread_topic_shifted(
            "Fear Of Long Covid", "Being Sick", INCIDENT_CORRECTION
        )

    def test_real_shift_still_asserted(self):
        from core.orchestrator import _thread_topic_shifted
        assert _thread_topic_shifted(
            "Forearm Pain", "Postgres Migration",
            "thinking about switching to postgres",
        )

    def test_related_topics_never_shift(self):
        from core.orchestrator import _thread_topic_shifted
        assert not _thread_topic_shifted(
            "Long Covid Fears", "Fear Of Long Covid",
            "some standalone message about my covid fears",
        )


# ---------------------------------------------------------------------------
# 4. Planner grounding
# ---------------------------------------------------------------------------

class _FakeModelManager:
    def __init__(self):
        self.prompts = []

    async def generate_once(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return (
            '{"key_points": ["a"], "tone": "warm", '
            '"avoid": ["b"], "strategy": "c"}'
        )


class _PlanContext:
    """Minimal stand-in for ContextResult as read by create_plan."""
    intent = None
    tone_level = None
    topics = ["Exercise Routine"]
    thread_context = None

    def __init__(self, last_exchange=None):
        self.last_exchange = last_exchange


class TestPlannerGrounding:
    def _run_plan(self, context):
        from core.response_planner import ResponsePlanner
        mm = _FakeModelManager()
        planner = ResponsePlanner(mm)
        plan = asyncio.run(planner.create_plan(INCIDENT_FRAGMENT, context))
        assert plan is not None
        return mm.prompts[0]

    def test_previous_exchange_included_in_prompt(self):
        ctx = _PlanContext(last_exchange={
            "query": "I'm sometimes scared Casey was right and I never had long covid",
            "response": "That fear makes sense...",
        })
        prompt = self._run_plan(ctx)
        assert "Previous exchange" in prompt
        assert "never had long covid" in prompt
        # The fragment-resolution instruction is present.
        assert "do not treat it as a standalone statement" in prompt

    def test_no_exchange_block_when_history_absent(self):
        prompt = self._run_plan(_PlanContext(last_exchange=None))
        assert "Previous exchange" not in prompt

    def test_long_exchange_truncated(self):
        ctx = _PlanContext(last_exchange={
            "query": "x" * 2000,
            "response": "y" * 2000,
        })
        prompt = self._run_plan(ctx)
        assert "x" * 401 not in prompt
        assert "y" * 401 not in prompt
