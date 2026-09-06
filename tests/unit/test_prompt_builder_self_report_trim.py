"""Self-report retrieval trim (2026-09-06).

A one-line first-person status update with no request ("I took my stimulant
at 10 AM today and I'm just resting this afternoon, feels good honestly even
though I got nothing done") built an 11K-token prompt — [RELEVANT MEMORIES]
n=14, [USER PROFILE] n=60 facts, 5 personal notes, a stale upload, full
narrative — because nothing keyed off the message's SHAPE trimmed retrieval
for it (intent classified general, no override fired).

Tests drive THE deployed functions: core.prompt.builder._apply_self_report_trim,
_distress_from_crisis_level, and core.prompt.builder.UnifiedPromptBuilder.build_prompt
itself (verifying the effective per-call args it hands the context gatherer).
"""

import pytest
from unittest.mock import MagicMock

from core.prompt.builder import (
    UnifiedPromptBuilder,
    SELF_REPORT_RETRIEVAL_TRIM,
    _apply_self_report_trim,
    _distress_from_crisis_level,
)
from core.prompt.context_gatherer import ContextGatherer
from core.prompt.token_manager import USER_PROFILE_MAX_TOKENS
from utils.query_checker import is_self_report, _is_heavy_topic_heuristic


SELF_REPORT_QUERY = (
    "I took my stimulant at 10 AM today and I'm just resting this afternoon, "
    "feels good honestly even though I got nothing done"
)
QUESTION_QUERY = "Does my history support that I have been consistent with meds?"
HEAVY_SELF_REPORT_QUERY = "I am so angry and furious right now, I want to hurt someone"


# =============================================================================
# _distress_from_crisis_level
# =============================================================================

class TestDistressFromCrisisLevel:
    @pytest.mark.parametrize("crisis", [
        "CrisisLevel.HIGH",
        "CrisisLevel.MEDIUM",
        "CrisisLevel.CONCERN",
        "crisis_support",
        "elevated_support",
        "light_support",
    ])
    def test_distress_encodings_true(self, crisis):
        assert _distress_from_crisis_level(crisis)

    @pytest.mark.parametrize("crisis", [None, "", "conversational", "CONVERSATIONAL"])
    def test_non_distress_encodings_false(self, crisis):
        # Note: "CrisisLevel.CONVERSATIONAL" is NOT included here — the
        # "CrisisLevel." prefix itself contains the substring "CRISIS", so
        # the deployed substring-match logic (unchanged from the original
        # ~1049-1056 code this helper was extracted from) already treats it
        # as distress. That pre-existing quirk is out of scope for this fix.
        assert not _distress_from_crisis_level(crisis)


# =============================================================================
# _apply_self_report_trim (pure function)
# =============================================================================

class TestApplySelfReportTrim:
    def test_basic_trim_applied(self):
        out = _apply_self_report_trim({}, SELF_REPORT_QUERY, None)
        assert out == SELF_REPORT_RETRIEVAL_TRIM

    def test_input_dict_not_mutated(self):
        ro = {}
        _apply_self_report_trim(ro, SELF_REPORT_QUERY, None)
        assert ro == {}

    def test_returns_new_dict(self):
        ro = {"max_mems": 3}
        out = _apply_self_report_trim(ro, SELF_REPORT_QUERY, None)
        assert out is not ro

    def test_intent_lower_value_kept(self):
        """An intent override already stricter than the trim is never raised."""
        ro = {"max_mems": 3}
        out = _apply_self_report_trim(ro, SELF_REPORT_QUERY, None)
        assert out["max_mems"] == 3

    def test_intent_higher_value_capped(self):
        """An intent override looser than the trim is capped down to it."""
        ro = {"max_mems": 10}
        out = _apply_self_report_trim(ro, SELF_REPORT_QUERY, None)
        assert out["max_mems"] == SELF_REPORT_RETRIEVAL_TRIM["max_mems"]

    def test_missing_key_takes_trim_value(self):
        ro = {"max_surfaced_threads": 20}  # a key the trim doesn't govern
        out = _apply_self_report_trim(ro, SELF_REPORT_QUERY, None)
        assert out["max_surfaced_threads"] == 20
        assert out["max_wiki"] == SELF_REPORT_RETRIEVAL_TRIM["max_wiki"]
        # 2026-09-06 retest: recent turns are capped (min-merge) and commits zeroed.
        assert out["max_recent"] == SELF_REPORT_RETRIEVAL_TRIM["max_recent"]
        assert out["max_git_commits"] == 0

    def test_profile_tokens_key_present(self):
        out = _apply_self_report_trim({}, SELF_REPORT_QUERY, None)
        assert out["max_profile_tokens"] == SELF_REPORT_RETRIEVAL_TRIM["max_profile_tokens"]

    def test_noop_for_question(self):
        assert not is_self_report(QUESTION_QUERY)
        out = _apply_self_report_trim({}, QUESTION_QUERY, None)
        assert out == {}

    @pytest.mark.parametrize("crisis", [
        "CrisisLevel.HIGH", "CrisisLevel.CONCERN", "crisis_support", "elevated_support",
    ])
    def test_noop_at_distress_crisis_levels(self, crisis):
        out = _apply_self_report_trim({}, SELF_REPORT_QUERY, crisis)
        assert out == {}

    def test_noop_for_heavy_topic_self_report(self):
        # Confirm the fixture is actually both self-report-shaped AND heavy,
        # per the deployed heuristics, rather than assuming it.
        assert is_self_report(HEAVY_SELF_REPORT_QUERY)
        assert _is_heavy_topic_heuristic(HEAVY_SELF_REPORT_QUERY)
        out = _apply_self_report_trim({}, HEAVY_SELF_REPORT_QUERY, None)
        assert out == {}


# =============================================================================
# Builder-level integration: build_prompt's effective per-call args
# =============================================================================
# UnifiedPromptBuilder.__init__ pulls in a real MemoryCoordinator/ModelManager
# by default, so tests construct a bare instance via __new__ (same pattern as
# tests/unit/test_light_prompt_path.py::TestBuilderRouting) and mock only
# context_gatherer (spec=ContextGatherer, so its real async methods surface
# as AsyncMock automatically) — build_prompt's single big try/except means a
# downstream hygiene/formatting failure against mock data doesn't invalidate
# the assertions below: the gather-stage calls we inspect have already
# happened and been awaited by the time any such failure occurs.

def _make_builder() -> UnifiedPromptBuilder:
    b = UnifiedPromptBuilder.__new__(UnifiedPromptBuilder)
    b.context_gatherer = MagicMock(spec=ContextGatherer)
    b.time_manager = None
    b.model_manager = None
    b.memory_coordinator = MagicMock()
    b._skill_activation_policy = None
    b._skill_fetch_multiplier = 1
    return b


@pytest.mark.asyncio
async def test_profile_tokens_trimmed_for_self_report():
    b = _make_builder()
    await b.build_prompt(SELF_REPORT_QUERY, intent_type=None)
    b.context_gatherer.get_user_profile_context.assert_awaited_once()
    _, kwargs = b.context_gatherer.get_user_profile_context.await_args
    assert kwargs["max_tokens"] == SELF_REPORT_RETRIEVAL_TRIM["max_profile_tokens"]


@pytest.mark.asyncio
async def test_profile_tokens_full_for_question():
    b = _make_builder()
    await b.build_prompt(QUESTION_QUERY, intent_type=None)
    b.context_gatherer.get_user_profile_context.assert_awaited_once()
    _, kwargs = b.context_gatherer.get_user_profile_context.await_args
    assert kwargs["max_tokens"] == USER_PROFILE_MAX_TOKENS


@pytest.mark.asyncio
async def test_uploads_and_wiki_suppressed_for_self_report():
    b = _make_builder()
    await b.build_prompt(SELF_REPORT_QUERY, intent_type=None)
    b.context_gatherer.get_user_uploads.assert_not_awaited()
    b.context_gatherer._get_wiki_content.assert_not_awaited()
    b.context_gatherer._get_semantic_chunks.assert_not_awaited()


@pytest.mark.asyncio
async def test_uploads_and_wiki_present_for_question():
    """Baseline: the same sources ARE retrieved for a non-self-report query,
    proving the suppression above is specific to the self-report shape and
    not a side effect of the mocked harness."""
    b = _make_builder()
    await b.build_prompt(QUESTION_QUERY, intent_type=None)
    b.context_gatherer.get_user_uploads.assert_awaited_once()
    b.context_gatherer._get_wiki_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_distress_self_report_keeps_full_context():
    """A self-report shape during an elevated crisis_level still gets full
    retrieval — the trim never fires on distress turns."""
    b = _make_builder()
    await b.build_prompt(SELF_REPORT_QUERY, intent_type=None, crisis_level="CrisisLevel.CONCERN")
    b.context_gatherer.get_user_profile_context.assert_awaited_once()
    _, kwargs = b.context_gatherer.get_user_profile_context.await_args
    assert kwargs["max_tokens"] == USER_PROFILE_MAX_TOKENS
    b.context_gatherer.get_user_uploads.assert_awaited_once()
