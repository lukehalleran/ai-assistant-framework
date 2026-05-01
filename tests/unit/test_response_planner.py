"""
Tests for core.response_planner — ResponsePlanner, ResponsePlan, ReviewResult.

Covers:
- should_plan() bypass logic (small-talk, crisis, config)
- create_plan() success, failure, timeout, bad JSON
- review_answer() success, failure
- format_plan_injection() output format
- Pydantic model validation
"""

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.response_planner import ResponsePlan, ResponsePlanner, ReviewResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _ToneLevel(Enum):
    CRISIS = "HIGH"
    ELEVATED = "MEDIUM"
    CONCERN = "CONCERN"
    CONVERSATIONAL = "CONVERSATIONAL"


@dataclass
class _QueryAnalysis:
    is_small_talk: bool = False


@dataclass
class _IntentResult:
    intent_type: str = "GENERAL"


@dataclass
class _FakeContext:
    original_query: str = "Can you help me understand how the memory system works in detail?"
    tone_level: _ToneLevel = _ToneLevel.CONVERSATIONAL
    query_analysis: _QueryAnalysis = field(default_factory=_QueryAnalysis)
    intent: _IntentResult = field(default_factory=_IntentResult)
    topics: list = field(default_factory=lambda: ["family"])
    thread_context: dict = field(default_factory=dict)


def _make_planner(llm_response: str = "", side_effect=None):
    """Return a ResponsePlanner with a mocked model_manager."""
    mm = MagicMock()
    if side_effect:
        mm.generate_once = AsyncMock(side_effect=side_effect)
    else:
        mm.generate_once = AsyncMock(return_value=llm_response)
    return ResponsePlanner(model_manager=mm)


_VALID_PLAN_JSON = json.dumps({
    "key_points": ["Recall brother's name from memory", "Be direct"],
    "tone": "warm",
    "avoid": ["Guessing if unsure"],
    "strategy": "Retrieve factual recall from user profile and memories",
})

_VALID_REVIEW_PASS = json.dumps({
    "passes": True,
    "confidence": 0.95,
    "issues": [],
    "suggestion": "",
})

_VALID_REVIEW_FAIL = json.dumps({
    "passes": False,
    "confidence": 0.90,
    "issues": ["Did not mention the brother's name", "Too vague"],
    "suggestion": "Search memory for brother's name and state it directly",
})


# ---------------------------------------------------------------------------
# should_plan() tests
# ---------------------------------------------------------------------------


class TestShouldPlan:

    def test_normal_query_returns_true(self):
        ctx = _FakeContext()
        with patch("config.app_config.RESPONSE_PLANNING_ENABLED", True):
            with patch.dict("sys.modules", {"core.context_pipeline": MagicMock(ToneLevel=_ToneLevel)}):
                assert ResponsePlanner.should_plan(ctx) is True

    def test_small_talk_skips(self):
        ctx = _FakeContext(query_analysis=_QueryAnalysis(is_small_talk=True))
        with patch("config.app_config.RESPONSE_PLANNING_ENABLED", True):
            assert ResponsePlanner.should_plan(ctx) is False

    def test_crisis_skips(self):
        ctx = _FakeContext(tone_level=_ToneLevel.CRISIS)
        with patch("config.app_config.RESPONSE_PLANNING_ENABLED", True):
            with patch.dict("sys.modules", {"core.context_pipeline": MagicMock(ToneLevel=_ToneLevel)}):
                assert ResponsePlanner.should_plan(ctx) is False

    def test_elevated_skips(self):
        ctx = _FakeContext(tone_level=_ToneLevel.ELEVATED)
        with patch("config.app_config.RESPONSE_PLANNING_ENABLED", True):
            with patch.dict("sys.modules", {"core.context_pipeline": MagicMock(ToneLevel=_ToneLevel)}):
                assert ResponsePlanner.should_plan(ctx) is False

    def test_disabled_config_skips(self):
        ctx = _FakeContext()
        with patch("config.app_config.RESPONSE_PLANNING_ENABLED", False):
            assert ResponsePlanner.should_plan(ctx) is False


# ---------------------------------------------------------------------------
# create_plan() tests
# ---------------------------------------------------------------------------


class TestCreatePlan:

    @pytest.mark.asyncio
    async def test_returns_plan_on_valid_json(self):
        planner = _make_planner(_VALID_PLAN_JSON)
        ctx = _FakeContext()
        with patch("config.app_config.RESPONSE_PLANNING_MODEL", None), \
             patch("config.app_config.RESPONSE_PLANNING_MAX_TOKENS", 200), \
             patch("config.app_config.RESPONSE_PLANNING_TIMEOUT", 5.0):
            plan = await planner.create_plan("What is my brother's name?", ctx)

        assert plan is not None
        assert isinstance(plan, ResponsePlan)
        assert len(plan.key_points) == 2
        assert plan.tone == "warm"
        assert len(plan.avoid) == 1
        assert plan.strategy != ""
        assert plan.raw_llm_output == _VALID_PLAN_JSON

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self):
        planner = _make_planner(side_effect=RuntimeError("API down"))
        ctx = _FakeContext()
        with patch("config.app_config.RESPONSE_PLANNING_MODEL", None), \
             patch("config.app_config.RESPONSE_PLANNING_MAX_TOKENS", 200), \
             patch("config.app_config.RESPONSE_PLANNING_TIMEOUT", 5.0):
            plan = await planner.create_plan("test", ctx)
        assert plan is None

    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(self):
        planner = _make_planner("not valid json {{{")
        ctx = _FakeContext()
        with patch("config.app_config.RESPONSE_PLANNING_MODEL", None), \
             patch("config.app_config.RESPONSE_PLANNING_MAX_TOKENS", 200), \
             patch("config.app_config.RESPONSE_PLANNING_TIMEOUT", 5.0):
            plan = await planner.create_plan("test", ctx)
        assert plan is None

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        async def _slow(*args, **kwargs):
            await asyncio.sleep(10)
            return _VALID_PLAN_JSON

        mm = MagicMock()
        mm.generate_once = _slow
        planner = ResponsePlanner(model_manager=mm)
        ctx = _FakeContext()
        with patch("config.app_config.RESPONSE_PLANNING_MODEL", None), \
             patch("config.app_config.RESPONSE_PLANNING_MAX_TOKENS", 200), \
             patch("config.app_config.RESPONSE_PLANNING_TIMEOUT", 0.1):
            plan = await planner.create_plan("test", ctx)
        assert plan is None

    @pytest.mark.asyncio
    async def test_empty_response_returns_none(self):
        planner = _make_planner("")
        ctx = _FakeContext()
        with patch("config.app_config.RESPONSE_PLANNING_MODEL", None), \
             patch("config.app_config.RESPONSE_PLANNING_MAX_TOKENS", 200), \
             patch("config.app_config.RESPONSE_PLANNING_TIMEOUT", 5.0):
            plan = await planner.create_plan("test", ctx)
        assert plan is None

    @pytest.mark.asyncio
    async def test_markdown_fenced_json_parsed(self):
        fenced = f"```json\n{_VALID_PLAN_JSON}\n```"
        planner = _make_planner(fenced)
        ctx = _FakeContext()
        with patch("config.app_config.RESPONSE_PLANNING_MODEL", None), \
             patch("config.app_config.RESPONSE_PLANNING_MAX_TOKENS", 200), \
             patch("config.app_config.RESPONSE_PLANNING_TIMEOUT", 5.0):
            plan = await planner.create_plan("test", ctx)
        assert plan is not None
        assert plan.tone == "warm"


# ---------------------------------------------------------------------------
# review_answer() tests
# ---------------------------------------------------------------------------


class TestReviewAnswer:

    @pytest.mark.asyncio
    async def test_passes_review(self):
        planner = _make_planner(_VALID_REVIEW_PASS)
        plan = ResponsePlan(
            key_points=["Recall name"], tone="warm", avoid=[], strategy="direct recall"
        )
        with patch("config.app_config.RESPONSE_REVIEW_MODEL", None), \
             patch("config.app_config.RESPONSE_REVIEW_MAX_TOKENS", 200), \
             patch("config.app_config.RESPONSE_REVIEW_TIMEOUT", 5.0):
            review = await planner.review_answer(plan, "Your brother's name is Alex.", "test")

        assert review is not None
        assert review.passes is True
        assert review.confidence == 0.95
        assert review.issues == []

    @pytest.mark.asyncio
    async def test_fails_with_issues(self):
        planner = _make_planner(_VALID_REVIEW_FAIL)
        plan = ResponsePlan(
            key_points=["Recall name"], tone="warm", avoid=[], strategy="direct recall"
        )
        with patch("config.app_config.RESPONSE_REVIEW_MODEL", None), \
             patch("config.app_config.RESPONSE_REVIEW_MAX_TOKENS", 200), \
             patch("config.app_config.RESPONSE_REVIEW_TIMEOUT", 5.0):
            review = await planner.review_answer(plan, "I'm not sure about that.", "test")

        assert review is not None
        assert review.passes is False
        assert review.confidence == 0.90
        assert len(review.issues) == 2

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self):
        planner = _make_planner(side_effect=RuntimeError("API down"))
        plan = ResponsePlan(key_points=[], tone="warm", avoid=[], strategy="")
        with patch("config.app_config.RESPONSE_REVIEW_MODEL", None), \
             patch("config.app_config.RESPONSE_REVIEW_MAX_TOKENS", 200), \
             patch("config.app_config.RESPONSE_REVIEW_TIMEOUT", 5.0):
            review = await planner.review_answer(plan, "response", "query")
        assert review is None


# ---------------------------------------------------------------------------
# format_plan_injection() tests
# ---------------------------------------------------------------------------


class TestFormatPlanInjection:

    def test_format_includes_key_fields(self):
        plan = ResponsePlan(
            key_points=["Point A", "Point B"],
            tone="analytical",
            avoid=["Guessing"],
            strategy="Use factual recall",
        )
        result = ResponsePlanner.format_plan_injection(plan)
        assert "[RESPONSE PLAN]" in result
        assert "Point A" in result
        assert "Point B" in result
        assert "analytical" in result
        assert "Guessing" in result
        assert "Use factual recall" in result
        assert "Do not mention this plan" in result

    def test_format_handles_empty_fields(self):
        plan = ResponsePlan(key_points=[], tone="neutral", avoid=[], strategy="")
        result = ResponsePlanner.format_plan_injection(plan)
        assert "[RESPONSE PLAN]" in result
        assert "(none)" in result


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestModels:

    def test_response_plan_defaults(self):
        plan = ResponsePlan()
        assert plan.key_points == []
        assert plan.tone == "neutral"
        assert plan.avoid == []
        assert plan.strategy == ""
        assert plan.raw_llm_output == ""

    def test_review_result_defaults(self):
        review = ReviewResult()
        assert review.passes is True
        assert review.confidence == 0.0
        assert review.issues == []
        assert review.suggestion == ""

    def test_response_plan_serialization(self):
        plan = ResponsePlan(
            key_points=["a", "b"], tone="warm", avoid=["c"], strategy="d"
        )
        data = plan.model_dump()
        assert data["key_points"] == ["a", "b"]
        assert data["tone"] == "warm"
        restored = ResponsePlan(**data)
        assert restored == plan

    def test_review_result_serialization(self):
        review = ReviewResult(
            passes=False, confidence=0.85, issues=["x"], suggestion="y"
        )
        data = review.model_dump()
        assert data["passes"] is False
        restored = ReviewResult(**data)
        assert restored == review
