"""
Regression tests for the 2026-09-04 doc-generation negation guard.

Live incident: a message ending "... Do not save a document for this. Plain
list in the reply is fine." matched DOCUMENT_TRIGGER_PATTERN on "save ...
document" and fired document generation ("Document saved: ...") despite the
explicit negation — the trigger pattern only checks for save-verb + doc-noun
co-occurrence, never whether the verb is negated.

Fix: knowledge/document_generator._trigger_match_is_negated (a lookback
check applied to a MATCH, not a rewrite of the tuned bounded-gap regex) is
now the single chokepoint (_non_negated_trigger_matches) used by BOTH
deterministic doc-trigger sites in that module — detect_document_intent's
existence check and the incidental-trigger placement guard. core.agentic.
gate's Tier-3 doc-gen arm and core.orchestrator both call into
detect_document_intent directly, so they inherit the fix without any
duplicated logic (verified below by grepping the call sites, not just
asserted).

Same-day companion fix: core/insight/detector.py gained one narrow cue so an
explicit "evidence sweep" over the user's own conversations routes to
insight mode instead of falling through to nothing.

These tests drive the DEPLOYED functions.
"""
import inspect

from knowledge.document_generator import detect_document_intent
from core.insight.detector import detect_insight_request


class TestNegationGuard:
    def test_live_negated_message_does_not_trigger(self):
        query = (
            "Please make a plan for the trip. Order by date. "
            "Do not save a document for this. Plain list in the reply is fine."
        )
        assert detect_document_intent(query) is None

    def test_plain_save_request_still_triggers(self):
        result = detect_document_intent(
            "Please save a document summarizing this for my therapist"
        )
        assert result is not None
        assert "topic" in result

    def test_negation_after_verb_does_not_disqualify(self):
        """A negation cue that appears AFTER the save-verb (scoping something
        else, not the request itself) must not suppress a genuine request."""
        result = detect_document_intent(
            "Write me a report on the migration plan, don't just answer in chat"
        )
        assert result is not None

    def test_no_need_to_phrasing_does_not_trigger(self):
        assert detect_document_intent(
            "no need to write a report, just tell me"
        ) is None

    def test_other_negation_cues(self):
        for query in (
            "I don't want you to write a report about this, just explain it",
            "never save a document about this conversation",
            "skip writing a report, a quick answer is enough",
            "without saving a document, can you just summarize this",
        ):
            assert detect_document_intent(query) is None, query

    def test_negation_far_from_verb_does_not_suppress(self):
        """Negation more than ~5 tokens before the save-verb is unrelated —
        it must not blanket-suppress a real, later request in the same
        message."""
        result = detect_document_intent(
            "I never really liked long meetings, anyway please write a report "
            "about the migration plan for the team"
        )
        assert result is not None


class TestDocGenTriggerSingleChokepoint:
    """The negation guard lives in ONE place; every caller must reach it
    through detect_document_intent rather than re-implementing detection."""

    def test_gate_calls_the_shared_detector(self):
        import core.agentic.gate as gate
        src = inspect.getsource(gate)
        assert "from knowledge.document_generator import detect_document_intent" in src

    def test_orchestrator_calls_the_shared_detector(self):
        import core.orchestrator as orchestrator
        src = inspect.getsource(orchestrator)
        assert "from knowledge.document_generator import detect_document_intent" in src


class TestEvidenceSweepInsightCue:
    def test_explicit_evidence_sweep_request_detected(self):
        query = (
            "Evidence sweep request. From 2026-07-15 through today, find "
            "every turn where I corrected something you said, and summarize "
            "the pattern."
        )
        intent = detect_insight_request(query)
        assert intent is not None

    def test_memory_sweep_phrasing_detected(self):
        query = "Can you do a memory sweep of our conversations for anything about my sleep?"
        intent = detect_insight_request(query)
        assert intent is not None

    def test_bystander_sweep_mention_not_detected(self):
        assert detect_insight_request("the street sweeper came by this morning") is None

    def test_bare_sweep_without_anchor_not_detected(self):
        """'sweep' with no first-person/possessive anchor and no
        evidence/memory qualifier must not qualify."""
        assert detect_insight_request(
            "sweep the floor before the guests arrive"
        ) is None
