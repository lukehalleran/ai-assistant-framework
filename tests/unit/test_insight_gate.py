"""
Insight-mode gate integration (2026-08-23).

Covers:
  * detect_insight_request routing through evaluate_agentic_gate BEFORE the
    Tier-3 doc-gen check — a personal-theme document request must route to
    insight mode, not web research (the observed generate_document misfire).
  * Ordinary doc-gen ("write a report about the roman empire") unaffected.
  * Insight decisions are veto_exempt — no tone/intent veto suppresses them.
  * Consent offer: armed only on insight-shaped statements at NON-elevated
    tone, once per session; a terse affirmation next turn yields an
    insight_assessment decision; the slot is one-shot.
"""

import pytest

import core.agentic.gate as gate
from core.agentic.gate import (
    AgenticDecision,
    apply_intent_veto,
    evaluate_agentic_gate,
    maybe_arm_insight_offer,
)
from core.insight.detector import detect_insight_request, detect_insight_statement


@pytest.fixture(autouse=True)
def _clean_slots():
    gate._DEFERRED_REQUEST_SLOT.clear()
    gate._reset_insight_offer_state()
    yield
    gate._DEFERRED_REQUEST_SLOT.clear()
    gate._reset_insight_offer_state()


class TestInsightRouting:
    @pytest.mark.asyncio
    async def test_theme_sweep_triggers_insight_mode(self):
        d = await evaluate_agentic_gate(
            user_text="gather everything I've said about sleep over time"
        )
        assert d.should_trigger is True
        assert d.modes == ["insight"]
        assert d.veto_exempt is True
        assert d.insight_intent["kind"] == "theme_sweep"
        assert "sleep" in d.insight_intent["theme"]

    @pytest.mark.asyncio
    async def test_personal_doc_beats_doc_gen(self):
        d = await evaluate_agentic_gate(
            user_text="write a summary of my pattern with casey for my therapist"
        )
        assert d.modes == ["insight"]
        assert d.insight_intent["wants_document"] is True
        assert d.doc_gen_intent is None  # never reached Tier-3

    @pytest.mark.asyncio
    async def test_ordinary_doc_gen_unaffected(self):
        d = await evaluate_agentic_gate(
            user_text="write a report about the roman empire and save it"
        )
        assert d.insight_intent is None
        assert d.doc_gen_intent is not None  # Tier-3 still owns this

    @pytest.mark.asyncio
    async def test_assessment_request_triggers(self):
        d = await evaluate_agentic_gate(
            user_text="I think my real problem is avoidance — check that "
                      "against what I've told you"
        )
        assert d.modes == ["insight"]
        assert d.insight_intent["kind"] == "insight_assessment"

    @pytest.mark.asyncio
    async def test_veto_cannot_suppress_insight_decision(self):
        d = await evaluate_agentic_gate(
            user_text="gather everything I've said about my anxiety over time"
        )
        after = apply_intent_veto(
            d, {"intent_type": "emotional_support", "confidence": 0.60},
            tone_level="MEDIUM",
            query="gather everything I've said about my anxiety over time",
        )
        assert after.should_trigger is True
        assert after.insight_intent is not None

    @pytest.mark.asyncio
    async def test_disabled_flag_falls_through(self, monkeypatch):
        import config.app_config as app_config
        monkeypatch.setattr(app_config, "INSIGHT_MODE_ENABLED", False)
        d = await evaluate_agentic_gate(
            user_text="gather everything I've said about sleep over time"
        )
        assert d.insight_intent is None


class TestInsightStatementShape:
    def test_insight_statement_detected(self):
        assert detect_insight_statement(
            "I'm starting to realize my real problem is avoidance, not anxiety"
        )
        assert detect_insight_statement(
            "maybe the real pattern is I keep repeating the same fights"
        )

    def test_vent_is_not_insight_statement(self):
        assert not detect_insight_statement("I am so unhappy with everything")
        assert not detect_insight_statement(
            "ugh today was awful and I hate all of it"
        )

    def test_question_is_not_insight_statement(self):
        assert not detect_insight_statement(
            "do you think my real problem is avoidance?"
        )

    def test_explicit_assessment_that_is_also_vent_still_triggers(self):
        # A distressed phrasing that carries an EXPLICIT assessment request
        # must still route (explicit requests always work, even mid-distress).
        intent = detect_insight_request(
            "I feel like garbage and I keep falling into the same hole — "
            "am I right that this is the same pattern as with casey"
        )
        assert intent is not None
        assert intent.kind == "insight_assessment"


class TestConsentOffer:
    STATEMENT = "I'm starting to realize my real problem is avoidance"

    def test_armed_at_conversational_tone(self):
        assert maybe_arm_insight_offer(self.STATEMENT, "CONVERSATIONAL") is True
        assert gate._INSIGHT_OFFER_SLOT["statement"] == self.STATEMENT

    def test_not_armed_at_concern(self):
        assert maybe_arm_insight_offer(self.STATEMENT, "CONCERN") is False
        assert gate._INSIGHT_OFFER_SLOT == {}

    def test_not_armed_at_acute(self):
        assert maybe_arm_insight_offer(self.STATEMENT, "MEDIUM") is False

    def test_not_armed_for_non_statement(self):
        assert maybe_arm_insight_offer("what's the weather", None) is False

    def test_session_cap_is_one(self):
        assert maybe_arm_insight_offer(self.STATEMENT, None) is True
        gate._INSIGHT_OFFER_SLOT.clear()  # simulate the turn passing
        assert maybe_arm_insight_offer(self.STATEMENT, None) is False

    def test_offer_disabled_by_config(self, monkeypatch):
        import config.app_config as app_config
        monkeypatch.setattr(app_config, "INSIGHT_OFFER_ENABLED", False)
        assert maybe_arm_insight_offer(self.STATEMENT, None) is False

    @pytest.mark.asyncio
    async def test_affirmation_yields_assessment_decision(self):
        maybe_arm_insight_offer(self.STATEMENT, "CONVERSATIONAL")
        d = await evaluate_agentic_gate(user_text="sure go ahead")
        assert d.should_trigger is True
        assert d.modes == ["insight"]
        assert d.veto_exempt is True
        assert d.insight_intent["kind"] == "insight_assessment"
        assert d.insight_intent["theme"] == self.STATEMENT
        assert gate._INSIGHT_OFFER_SLOT == {}  # consumed

    @pytest.mark.asyncio
    async def test_non_affirmation_drops_offer_permanently(self):
        maybe_arm_insight_offer(self.STATEMENT, None)
        d = await evaluate_agentic_gate(
            user_text="anyway my mom got home from her trip yesterday"
        )
        assert "insight-offer" not in (d.reason or "")
        # slot consumed and discarded; a later "sure" does nothing
        d2 = await evaluate_agentic_gate(user_text="sure")
        assert "insight-offer" not in (d2.reason or "")
        # and the session cap keeps a re-offer from arming again
        assert maybe_arm_insight_offer(self.STATEMENT, None) is False

    @pytest.mark.asyncio
    async def test_long_reply_is_not_affirmation(self):
        maybe_arm_insight_offer(self.STATEMENT, None)
        d = await evaluate_agentic_gate(
            user_text="sure but first tell me about what you found last "
                      "time we talked about this stuff"
        )
        assert "insight-offer" not in (d.reason or "")
