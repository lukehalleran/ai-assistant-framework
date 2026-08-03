"""
Regression tests for the 2026-07-25 borderline-tone incident.

Three coupled fixes:

1. `_llm_crisis_fallback` used `model_manager.generate_async` (returns a
   stream object for API models) and called `.strip()` on it — every
   borderline classification crashed and silently defaulted to
   CONVERSATIONAL. It now uses `generate_once` (non-streaming str) under a
   timeout.

2. Deterministic borderline backstop: when the LLM arbiter is unavailable
   (broken, timed out, or no model manager) and the semantic scores rank a
   distress class above conversational by the margin, the detector floors to
   CONCERN instead of failing to CONVERSATIONAL. Live miss this guards:
   "I keep thinking I am a stupid piece of shit ... I wanna cry" scored
   medium=0.390 > conversational=0.302 and was labeled CONVERSATIONAL.

3. Tone-corroborated agentic veto: emotional_support at the STM-refined
   confidence floor (0.60, below the 0.75 hard veto floor) vetoes the agentic
   gate when the tone detector independently reads CONCERN or above. Live
   miss: "When I was moaning and crying in bed my mom ignored me" ran a 22s
   agentic search loop.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.tone_detector import (
    CrisisLevel,
    TONE_CONFIG,
    _llm_crisis_fallback,
    detect_crisis_level,
)
from core.agentic.gate import AgenticDecision, _tone_is_elevated, apply_intent_veto


# The real 18:52 turn — long enough to skip the short-message path, no
# explicit crisis keyword, semantically borderline.
BORDERLINE_DISTRESS_MSG = (
    "I am back now. I keep thinking I am a stupid piece of shit. "
    "I know it's the meds but I wanna cry"
)

# Semantic scores logged for that turn (distress top, conversational last).
DISTRESS_TOP_SCORES = {
    "high": 0.365, "medium": 0.390, "concern": 0.349, "conversational": 0.302,
}
# The 17:59 turn's shape — conversational top; backstop must NOT fire.
CONVERSATIONAL_TOP_SCORES = {
    "high": 0.249, "medium": 0.321, "concern": 0.324, "conversational": 0.340,
}


def _mock_model_manager(response="CONCERN"):
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value=response)
    # A stream-returning generate_async that would crash `.strip()` — present
    # so the test fails loudly if the fallback ever regresses to calling it.
    mm.generate_async = AsyncMock(return_value=object())
    return mm


class TestLLMFallback:
    @pytest.mark.asyncio
    async def test_uses_generate_once_not_generate_async(self):
        mm = _mock_model_manager("CONCERN")
        result = await _llm_crisis_fallback("some borderline message", mm)
        assert result is not None
        level, conf = result
        assert level == CrisisLevel.CONCERN
        mm.generate_once.assert_awaited_once()
        mm.generate_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_parses_all_levels(self):
        for text, expected in [
            ("HIGH", CrisisLevel.HIGH),
            ("MEDIUM", CrisisLevel.MEDIUM),
            ("CONCERN", CrisisLevel.CONCERN),
            ("CONVERSATIONAL", CrisisLevel.CONVERSATIONAL),
        ]:
            result = await _llm_crisis_fallback("msg", _mock_model_manager(text))
            assert result is not None and result[0] == expected

    @pytest.mark.asyncio
    async def test_non_string_response_returns_none(self):
        # A regressed stream-object return must yield None, not crash.
        mm = _mock_model_manager(response=object())
        assert await _llm_crisis_fallback("msg", mm) is None

    @pytest.mark.asyncio
    async def test_exception_returns_none(self):
        mm = MagicMock()
        mm.generate_once = AsyncMock(side_effect=RuntimeError("api down"))
        assert await _llm_crisis_fallback("msg", mm) is None

    @pytest.mark.asyncio
    async def test_hung_call_times_out(self):
        async def _hang(*a, **k):
            await asyncio.sleep(30)

        mm = MagicMock()
        mm.generate_once = AsyncMock(side_effect=_hang)
        with patch.dict(TONE_CONFIG, {"llm_fallback_timeout_s": 0.05}):
            assert await _llm_crisis_fallback("msg", mm) is None


class TestBorderlineBackstop:
    @pytest.mark.asyncio
    async def test_distress_top_with_broken_arbiter_floors_to_concern(self):
        with patch(
            "utils.tone_detector._semantic_crisis_detection",
            return_value=(CrisisLevel.CONVERSATIONAL, 0.302, DISTRESS_TOP_SCORES),
        ), patch(
            "utils.tone_detector._llm_crisis_fallback",
            new=AsyncMock(return_value=None),
        ):
            analysis = await detect_crisis_level(
                BORDERLINE_DISTRESS_MSG, model_manager=MagicMock()
            )
        assert analysis.level == CrisisLevel.CONCERN
        assert analysis.trigger == "borderline_backstop"

    @pytest.mark.asyncio
    async def test_backstop_fires_without_model_manager(self):
        with patch(
            "utils.tone_detector._semantic_crisis_detection",
            return_value=(CrisisLevel.CONVERSATIONAL, 0.302, DISTRESS_TOP_SCORES),
        ):
            analysis = await detect_crisis_level(
                BORDERLINE_DISTRESS_MSG, model_manager=None
            )
        assert analysis.level == CrisisLevel.CONCERN
        assert analysis.trigger == "borderline_backstop"

    @pytest.mark.asyncio
    async def test_conversational_top_stays_conversational(self):
        with patch(
            "utils.tone_detector._semantic_crisis_detection",
            return_value=(CrisisLevel.CONVERSATIONAL, 0.340, CONVERSATIONAL_TOP_SCORES),
        ), patch(
            "utils.tone_detector._llm_crisis_fallback",
            new=AsyncMock(return_value=None),
        ):
            analysis = await detect_crisis_level(
                "Yeah. Fuuuuck. I think I just laid in bed for a while but "
                "walking to the store now and feeling a little better",
                model_manager=MagicMock(),
            )
        assert analysis.level == CrisisLevel.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_weak_absolute_score_does_not_backstop(self):
        # "That movie made me sad" shape: big margin over conversational but
        # weak absolute distress score — situational sadness, not distress.
        weak_scores = {
            "high": 0.13, "medium": 0.25, "concern": 0.34, "conversational": 0.15,
        }
        with patch(
            "utils.tone_detector._semantic_crisis_detection",
            return_value=(CrisisLevel.CONVERSATIONAL, 0.15, weak_scores),
        ):
            analysis = await detect_crisis_level(
                "That movie made me sad", model_manager=None
            )
        assert analysis.level == CrisisLevel.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_working_arbiter_wins_over_backstop(self):
        with patch(
            "utils.tone_detector._semantic_crisis_detection",
            return_value=(CrisisLevel.CONVERSATIONAL, 0.302, DISTRESS_TOP_SCORES),
        ), patch(
            "utils.tone_detector._llm_crisis_fallback",
            new=AsyncMock(return_value=(CrisisLevel.MEDIUM, 0.75)),
        ):
            analysis = await detect_crisis_level(
                BORDERLINE_DISTRESS_MSG, model_manager=MagicMock()
            )
        assert analysis.level == CrisisLevel.MEDIUM
        assert analysis.trigger == "llm_fallback"


def _triggered_decision(veto_exempt=False):
    return AgenticDecision(
        should_trigger=True,
        modes=["memory"],
        search_terms=["something"],
        matched_entities=[],
        doc_gen_intent=None,
        self_note_intent=None,
        skip_initial_search=True,
        reason="triggered: memory",
        veto_exempt=veto_exempt,
    )


class TestToneElevated:
    def test_encodings(self):
        assert _tone_is_elevated("light_support")
        assert _tone_is_elevated("elevated_support")
        assert _tone_is_elevated("crisis_support")
        assert _tone_is_elevated("CrisisLevel.CONCERN")
        assert _tone_is_elevated("CrisisLevel.HIGH")
        assert not _tone_is_elevated("conversational")
        assert not _tone_is_elevated("CrisisLevel.CONVERSATIONAL")
        assert not _tone_is_elevated(None)
        assert not _tone_is_elevated("")


class TestToneCorroboratedVeto:
    def _intent(self, intent_type="emotional_support", confidence=0.60):
        return {"intent_type": intent_type, "confidence": confidence}

    def test_emotional_support_with_elevated_tone_vetoes(self):
        d = apply_intent_veto(
            _triggered_decision(), self._intent(), tone_level="light_support"
        )
        assert d.should_trigger is False
        assert d.search_terms == []
        assert "emotional_support" in d.reason

    def test_emotional_support_without_tone_does_not_veto(self):
        d = apply_intent_veto(_triggered_decision(), self._intent(), tone_level=None)
        assert d.should_trigger is True

    def test_emotional_support_conversational_tone_does_not_veto(self):
        d = apply_intent_veto(
            _triggered_decision(), self._intent(), tone_level="conversational"
        )
        assert d.should_trigger is True

    def test_low_confidence_does_not_veto_even_with_tone(self):
        d = apply_intent_veto(
            _triggered_decision(),
            self._intent(confidence=0.40),
            tone_level="crisis_support",
        )
        assert d.should_trigger is True

    def test_veto_exempt_survives_corroborated_veto(self):
        d = apply_intent_veto(
            _triggered_decision(veto_exempt=True),
            self._intent(confidence=0.90),
            tone_level="crisis_support",
        )
        assert d.should_trigger is True

    def test_hard_veto_intents_unchanged(self):
        # Regression: the pre-existing 0.75-floor veto still works with the
        # new signature (tone omitted).
        d = apply_intent_veto(
            _triggered_decision(), self._intent("casual_social", 0.80)
        )
        assert d.should_trigger is False

    def test_hard_veto_floor_unchanged(self):
        d = apply_intent_veto(
            _triggered_decision(), self._intent("casual_social", 0.60)
        )
        assert d.should_trigger is True
