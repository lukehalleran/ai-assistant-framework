"""Fix 1.4 regression: 'give me a detailed analysis ... of what my record can
establish about X' must route to insight mode (pattern_temporal), not fall
through to the agentic gate's Tier-2 memory arm.

Root cause: _DELIBERATION_OPERATION_RE had a verb-only "analy[sz]e" (never
matched the noun "analysis") and no establish/show/say/indicate/support/
suggest/tell verb family, so a personal-record anchor ("my record") with no
other operation cue produced no deliberation match. See
core/insight/detector.py's _DELIBERATION_OPERATION_RE / _RECORD_ESTABLISHES_RE
/ _find_operation_outside for the fix and its overlap-exclusion guard (a bare
"my analysis" must not satisfy both sides of the record+operation test off
the same word — see test_zelphex_exact_request.py::test_possessive_analysis_cue_is_tight).
"""
import pytest

from core.insight.detector import detect_insight_request

LIVE_QUERY = (
    "Give me a detailed analysis in a table of what my record can "
    "establish about medication gaps."
)


class TestRecordEstablishesDetected:
    def test_live_query_routes_to_pattern_temporal(self):
        intent = detect_insight_request(LIVE_QUERY)
        assert intent is not None
        assert intent.kind == "pattern_temporal"
        assert intent.wants_document is False

    @pytest.mark.parametrize("query", [
        "What does my history actually show about my sleep?",
        "Tell me what my notes say about my sleep.",
        "Give me a breakdown of what my data indicates about my workouts.",
        "What can my record establish about my mood this month?",
        "What my corpus can establish about my energy levels?",
    ])
    def test_paraphrases_detected(self, query):
        intent = detect_insight_request(query)
        assert intent is not None, f"expected a match for: {query!r}"
        assert intent.kind == "pattern_temporal"

    @pytest.mark.parametrize("query", [
        "What does the research show about caffeine and sleep?",
        "Show me my calendar for tomorrow.",
        "My record collection is huge.",
    ])
    def test_negatives_stay_none(self, query):
        assert detect_insight_request(query) is None


class TestOverlapGuardUnaffected:
    """The new operation vocabulary must not resurrect the false positives
    the possessive-analysis tightening (Zelphex batch) already closed."""

    def test_third_party_possessive_analysis_stays_none(self):
        assert detect_insight_request(
            "I read my friend's analysis of the election and compared it "
            "to before"
        ) is None

    def test_bare_my_analysis_with_no_operation_stays_none(self):
        assert detect_insight_request(
            "The teacher said my analysis was wrong"
        ) is None
