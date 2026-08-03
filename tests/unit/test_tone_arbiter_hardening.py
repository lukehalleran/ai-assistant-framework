"""
Regression tests for the 2026-08-02 tone-arbiter misses.

Three coupled fixes in utils/tone_detector.py:

1. Arbiter call hardening: `_llm_crisis_fallback` now passes
   `disable_reasoning=True` and max_tokens=16 — on reasoning models (kimi-3)
   the reasoning channel ate the 10-token budget and returned empty content,
   silently dropping the arbiter verdict ("Not everyone deserves to be alive
   like me ... I would hurt them terribly if legal" fell through to
   CONVERSATIONAL).

2. Unrecognized arbiter output returns None (arbiter failure → deterministic
   backstop applies) instead of being parsed as a CONVERSATIONAL verdict.

3. Arbiter-contradiction backstop: an arbiter CONVERSATIONAL verdict on a
   borderline where the semantic scores rank a distress class above
   conversational by the margin (and above backstop_min_score) is overridden
   to CONCERN. Live miss: "I am embarrassed for how I reacted earlier ... I am
   so unhappy" — the arbiter's literal rubric ("anxiety, worry, stress")
   labeled plain unhappiness CONVERSATIONAL.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agentic.gate import AgenticDecision, apply_intent_veto, _is_info_seeking
from utils.tone_detector import (
    CrisisLevel,
    _llm_crisis_fallback,
    detect_crisis_level,
)

# Long enough to skip the short-message fast path, no explicit crisis keyword.
BORDERLINE_DISTRESS_MSG = (
    "I am back now. I keep thinking I am a stupid piece of shit. "
    "I know it's the meds but I wanna cry"
)

# Distress-top borderline shape (medium above the 0.37 backstop floor).
DISTRESS_TOP_SCORES = {
    "high": 0.365, "medium": 0.390, "concern": 0.349, "conversational": 0.302,
}

# Weak absolute scores ("that movie made me sad" shape) — backstop must not fire.
WEAK_SCORES = {
    "high": 0.13, "medium": 0.25, "concern": 0.34, "conversational": 0.15,
}


def _mock_model_manager(response="CONCERN"):
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value=response)
    return mm


class TestArbiterCallHardening:
    @pytest.mark.asyncio
    async def test_disables_reasoning_on_arbiter_call(self):
        mm = _mock_model_manager("CONCERN")
        await _llm_crisis_fallback("msg", mm)
        kwargs = mm.generate_once.await_args.kwargs
        assert kwargs.get("disable_reasoning") is True
        assert kwargs.get("max_tokens", 0) >= 16

    @pytest.mark.asyncio
    async def test_empty_response_returns_none(self):
        assert await _llm_crisis_fallback("msg", _mock_model_manager("")) is None

    @pytest.mark.asyncio
    async def test_unrecognized_verdict_returns_none(self):
        # Reasoning fragments / refusals must be arbiter failure, not a
        # CONVERSATIONAL verdict.
        for garbage in ("The message seems", "I cannot classify", "OK"):
            assert await _llm_crisis_fallback("msg", _mock_model_manager(garbage)) is None

    @pytest.mark.asyncio
    async def test_explicit_conversational_still_parses(self):
        result = await _llm_crisis_fallback("msg", _mock_model_manager("CONVERSATIONAL"))
        assert result is not None and result[0] == CrisisLevel.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_rubric_covers_sadness_and_violent_ideation(self):
        # The prompt itself must map plain sadness/unhappiness to CONCERN and
        # violent thoughts to MEDIUM — the literal "anxiety, worry, stress"
        # rubric was the 2026-08-02 miss.
        mm = _mock_model_manager("CONCERN")
        await _llm_crisis_fallback("I am so unhappy", mm)
        prompt = mm.generate_once.await_args.args[0]
        low = prompt.lower()
        assert "unhappiness" in low or "sadness" in low
        assert "violent thoughts" in low


class TestArbiterContradictionBackstop:
    @pytest.mark.asyncio
    async def test_arbiter_conversational_overridden_on_distress_top(self):
        with patch(
            "utils.tone_detector._semantic_crisis_detection",
            return_value=(CrisisLevel.CONVERSATIONAL, 0.302, DISTRESS_TOP_SCORES),
        ), patch(
            "utils.tone_detector._llm_crisis_fallback",
            new=AsyncMock(return_value=(CrisisLevel.CONVERSATIONAL, 0.6)),
        ):
            analysis = await detect_crisis_level(
                BORDERLINE_DISTRESS_MSG, model_manager=MagicMock()
            )
        assert analysis.level == CrisisLevel.CONCERN
        assert analysis.trigger == "borderline_backstop"

    @pytest.mark.asyncio
    async def test_arbiter_conversational_stands_on_weak_scores(self):
        with patch(
            "utils.tone_detector._semantic_crisis_detection",
            return_value=(CrisisLevel.CONVERSATIONAL, 0.15, WEAK_SCORES),
        ), patch(
            "utils.tone_detector._llm_crisis_fallback",
            new=AsyncMock(return_value=(CrisisLevel.CONVERSATIONAL, 0.6)),
        ):
            analysis = await detect_crisis_level(
                "That movie we watched tonight made me kind of sad honestly, "
                "the ending was rough but it was a good night overall",
                model_manager=MagicMock(),
            )
        assert analysis.level == CrisisLevel.CONVERSATIONAL
        assert analysis.trigger == "llm_fallback"

    @pytest.mark.asyncio
    async def test_arbiter_can_still_raise(self):
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


# ---- Tone-statement veto (intent-independent) ------------------------------
# Every 2026-08-02 distress vent classified general@0.00, so the
# emotional_support-gated corroboration veto never fired; "I am embarrassed
# for how I reacted earlier … I am so unhappy" ran a 31s agentic memory loop.

VENT = "I am embarrassed for how I reacted earlier. My dad came by for a bit. I am so unhappy"


def _decision(veto_exempt=False):
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


def _intent(intent_type="general", confidence=0.0):
    return {"intent_type": intent_type, "confidence": confidence}


class TestToneStatementVeto:
    def test_live_vent_vetoed_despite_general_intent(self):
        d = apply_intent_veto(
            _decision(), _intent(), tone_level="light_support", query=VENT
        )
        assert d.should_trigger is False
        assert "tone-veto" in d.reason

    def test_question_shape_not_vetoed(self):
        d = apply_intent_veto(
            _decision(), _intent(), tone_level="light_support",
            query="What did my therapist say about this last week?",
        )
        assert d.should_trigger is True

    def test_lookup_cue_not_vetoed(self):
        d = apply_intent_veto(
            _decision(), _intent(), tone_level="light_support",
            query="I'm upset, remember when we talked about coping stuff, pull that up",
        )
        assert d.should_trigger is True

    def test_confident_retrieval_intent_wins(self):
        d = apply_intent_veto(
            _decision(), _intent("factual_recall", 0.90),
            tone_level="light_support", query="My meds got changed last month I think.",
        )
        assert d.should_trigger is True

    def test_conversational_tone_not_vetoed(self):
        d = apply_intent_veto(
            _decision(), _intent(), tone_level="conversational", query=VENT
        )
        assert d.should_trigger is True

    def test_veto_exempt_survives(self):
        d = apply_intent_veto(
            _decision(veto_exempt=True), _intent(),
            tone_level="crisis_support", query=VENT,
        )
        assert d.should_trigger is True

    def test_no_query_fails_open(self):
        d = apply_intent_veto(
            _decision(), _intent(), tone_level="light_support", query=None
        )
        assert d.should_trigger is True

    def test_is_info_seeking_shapes(self):
        assert _is_info_seeking("what's the weather")
        assert _is_info_seeking("can you check my calendar")
        assert _is_info_seeking("so was it thursday we talked?")
        assert not _is_info_seeking(VENT)
        assert not _is_info_seeking("I just feel really off today.")
