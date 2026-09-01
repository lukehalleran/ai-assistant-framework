"""2026-08-29 calendar-task turn misfires (14:29, "search for documents
related to the MGT class ... place each in the appropriate time slot on my
Google calendar").

Four live misfires, each with a deterministic fix:

1. WEB ROUTING — the word "search" made the agentic gate's Tier-1 web arm
   run the turn in web_search mode AND the web-search trigger call it an
   "explicit search request" (conf 0.80): three Tavily sub-searches / 6
   credits, one of them literally "Add dates and deadlines to Google
   Calendar". The search target is the user's OWN corpus →
   query_checker.is_personal_doc_search() suppresses the web trigger and
   routes the gate to file/doc tools.

2. CALENDAR ACTION MISS — "place each ... on my Google calendar" matched no
   CALENDAR_CREATE_EVENT intent pattern (verb "place" unknown, bare
   "calendar" not an object, 44-char verb→object span over the 40 window) —
   the explicit request produced a "Want me to?" offer instead of a
   proposal. New placement-verb pattern in ACTION_SPECS.

3. VERIFIER SOURCE-BLINDNESS — the grounding verifier flagged "MGT 6203,
   Fall 2026" at conf 0.9 ("The course date should reflect the current
   academic calendar. Please verify the correct semester...") because the
   syllabus lived in RETRIEVED documents it never saw, and the morning's
   advice-shape regex only anchored at STRING start (the "Please verify"
   opened sentence two). Fixes: per-sentence advice classification with an
   assertive-clause rescue, and a source_material channel fed from the
   agentic loop's tool rounds (covered in test_grounding_wiring.py).

4. INTEGRATION — corrections are woven into the response (bounded rewrite)
   instead of tacked on as a ⚠️ blockquote; append is the fallback
   (integrator guard tests here, wiring in test_grounding_wiring.py).
"""
import json

import pytest

from utils.query_checker import is_personal_doc_search
from utils.web_search_trigger import should_search_heuristic
from core.agentic.gate import evaluate_agentic_gate
from core.actions.registry import detect_action_intent
from core.actions.types import ActionType
from core.grounding_check import (
    GroundingVerdict,
    _is_advice_shaped,
    _parse_verdict,
    _substantive_correction_text,
    integrate_grounding_correction,
)


LIVE_QUERY = (
    "Ok here's a good task, and if it doesn't work we can fix. Okay so please "
    "search for documents related to the MGT class I am currently enrolled in. "
    "Catalog all dates and deadlines etc and please place each in the "
    "appropriate time slot on my Google calendar"
)

# The exact correction that shipped on the live turn (conf 0.90, post-restart
# code — it slipped the string-start-anchored advice regex).
LIVE_CORRECTION_2 = (
    "The course date should reflect the current academic calendar. "
    "Please verify the correct semester for MGT 6203."
)


# ===========================================================================
# 1. Personal-document search detection
# ===========================================================================

class TestIsPersonalDocSearch:

    def test_live_query_detected(self):
        assert is_personal_doc_search(LIVE_QUERY) is True

    def test_find_my_notes(self):
        assert is_personal_doc_search("find my notes on the fin aid call") is True

    def test_pull_up_saved_files(self):
        assert is_personal_doc_search(
            "can you look through the files I uploaded last week") is True

    def test_news_search_not_personal(self):
        assert is_personal_doc_search(
            "search for news about the election results") is False

    def test_explicit_web_disqualifies(self):
        assert is_personal_doc_search(
            "search the web for documents about my condition") is False
        assert is_personal_doc_search(
            "look online for notes on ISLR chapter 3, I am stuck") is False

    def test_google_calendar_does_not_disqualify(self):
        # "Google calendar"/"Google Docs" are product names, not web cues.
        assert is_personal_doc_search(LIVE_QUERY) is True

    def test_no_personal_anchor_no_fire(self):
        assert is_personal_doc_search("search for documents about the Roman Empire") is False

    def test_paste_sized_message_never_fires(self):
        paste = "please search for documents related to my class. " + ("word " * 80)
        assert is_personal_doc_search(paste) is False

    def test_empty(self):
        assert is_personal_doc_search("") is False
        assert is_personal_doc_search("   ") is False


# ===========================================================================
# 2. Web-trigger suppression (heuristic + LLM-first deterministic pre-rule)
# ===========================================================================

class TestWebTriggerSuppression:

    def test_heuristic_suppresses_live_query(self):
        d = should_search_heuristic(LIVE_QUERY)
        assert d.should_search is False
        assert "personal_doc_search" in d.matched_patterns

    @pytest.mark.asyncio
    async def test_llm_path_deterministic_no_search(self):
        from utils.web_search_trigger import analyze_for_web_search_llm

        class _MM:
            async def generate_once(self, *a, **k):  # pragma: no cover
                raise AssertionError("LLM must not be consulted")

        d = await analyze_for_web_search_llm(LIVE_QUERY, model_manager=_MM())
        assert d.should_search is False
        assert "personal_doc_search" in d.matched_patterns

    def test_real_web_search_still_fires(self):
        d = should_search_heuristic("search for the latest news on the shutdown")
        assert d.should_search is True


# ===========================================================================
# 3. Agentic gate routing — tools, not web
# ===========================================================================

class TestGateRoutesToTools:

    @pytest.mark.asyncio
    async def test_live_query_routes_tools_not_web(self):
        d = await evaluate_agentic_gate(user_text=LIVE_QUERY)
        assert d.should_trigger is True
        assert "tools" in (d.modes or [])
        assert "web_search" not in (d.modes or [])

    @pytest.mark.asyncio
    async def test_url_still_wins_web(self):
        d = await evaluate_agentic_gate(
            user_text="go to https://example.com/syllabus and check my documents I am enrolled")
        assert "web_search" in (d.modes or [])


# ===========================================================================
# 4. Calendar action intent pattern
# ===========================================================================

class TestCalendarActionIntent:

    def test_live_phrase_detected(self):
        assert detect_action_intent(LIVE_QUERY) == ActionType.CALENDAR_CREATE_EVENT

    def test_place_on_calendar_minimal(self):
        assert detect_action_intent(
            "place the deadlines on my calendar") == ActionType.CALENDAR_CREATE_EVENT

    def test_add_to_google_calendar(self):
        assert detect_action_intent(
            "add these dates to my Google calendar") == ActionType.CALENDAR_CREATE_EVENT

    def test_original_pattern_still_works(self):
        assert detect_action_intent(
            "schedule a meeting with my advisor for Tuesday") == ActionType.CALENDAR_CREATE_EVENT

    def test_bare_calendar_mention_no_verb_no_fire(self):
        assert detect_action_intent("my calendar is looking rough this week") is None

    def test_narration_no_fire(self):
        assert detect_action_intent(
            "I looked at the calendar yesterday and cried a little") is None


# ===========================================================================
# 5. Grounding: live correction #2 demoted; assertive advice survives
# ===========================================================================

class TestLiveCorrection2Demoted:

    def _verdict(self, correction, why_false=""):
        return GroundingVerdict(
            false_claim_present=True, claim="MGT 6203, Fall 2026",
            why_false=why_false, confidence=0.9, correction=correction)

    def test_sentence_two_advice_is_stripped(self):
        assert _substantive_correction_text(LIVE_CORRECTION_2) == ""

    def test_live_verdict_is_advice_shaped(self):
        assert _is_advice_shaped(self._verdict(LIVE_CORRECTION_2)) is True

    def test_parse_demotes_live_verdict(self):
        raw = json.dumps({
            "false_claim_present": True,
            "claim": "MGT 6203, Fall 2026",
            "why_false": "The semester stated may not reflect the current academic calendar.",
            "confidence": 0.9,
            "correction": LIVE_CORRECTION_2,
        })
        v = _parse_verdict(raw)
        assert v is not None
        assert v.false_claim_present is False

    def test_advice_opener_with_asserted_value_survives(self):
        # "Check ... the correct due date is Sep 20" asserts a value — the
        # assertive-clause rescue must keep it (under-demote doctrine).
        kept = _substantive_correction_text(
            "Check the syllabus again — the correct due date is Sep 20.")
        assert "Sep 20" in kept
        assert _is_advice_shaped(self._verdict(
            "Check the syllabus again — the correct due date is Sep 20.",
            why_false="The stated date is incorrect.")) is False

    def test_hedge_should_reflect_stripped(self):
        assert _substantive_correction_text(
            "The course date should reflect the current academic calendar.") == ""


# ===========================================================================
# 6. Integrator guards (unit level — wiring in test_grounding_wiring.py)
# ===========================================================================

class _ScriptedMM:
    def __init__(self, out):
        self.out = out

    async def generate_once(self, prompt, **kwargs):
        return self.out


_VERDICT = GroundingVerdict(
    false_claim_present=True,
    claim="the theory lands closer to truth",
    why_false="The theory was discredited.",
    confidence=0.95,
    correction="The theory was discredited long ago.",
)

_RESPONSE = (
    "That framing has history behind it, and honestly the theory lands closer "
    "to truth than people admit. Worth a longer conversation another day."
)


class TestIntegratorGuards:

    @pytest.mark.asyncio
    async def test_in_bounds_revision_returned(self):
        revised_text = (
            "That framing has history behind it — correction: the theory was "
            "discredited long ago, so it doesn't hold up. Worth a longer "
            "conversation another day."
        )
        out = await integrate_grounding_correction(
            _RESPONSE, _VERDICT, _ScriptedMM(revised_text))
        assert out == revised_text

    @pytest.mark.asyncio
    async def test_too_short_revision_rejected(self):
        out = await integrate_grounding_correction(
            _RESPONSE, _VERDICT, _ScriptedMM("Nope."))
        assert out is None

    @pytest.mark.asyncio
    async def test_runaway_expansion_rejected(self):
        out = await integrate_grounding_correction(
            _RESPONSE, _VERDICT, _ScriptedMM(_RESPONSE + " padding" * 30))
        assert out is None

    @pytest.mark.asyncio
    async def test_identical_output_rejected(self):
        out = await integrate_grounding_correction(
            _RESPONSE, _VERDICT, _ScriptedMM(_RESPONSE))
        assert out is None

    @pytest.mark.asyncio
    async def test_leaked_warning_idiom_rejected(self):
        out = await integrate_grounding_correction(
            _RESPONSE, _VERDICT,
            _ScriptedMM(_RESPONSE[:-30] + " \n\n> ⚠️ Correction: discredited."))
        assert out is None

    @pytest.mark.asyncio
    async def test_long_response_skipped_without_llm_call(self):
        class _Boom:
            async def generate_once(self, *a, **k):  # pragma: no cover
                raise AssertionError("must not be called")
        out = await integrate_grounding_correction(
            "x" * 5000, _VERDICT, _Boom(), max_response_chars=4000)
        assert out is None

    @pytest.mark.asyncio
    async def test_exception_returns_none(self):
        class _Err:
            async def generate_once(self, *a, **k):
                raise RuntimeError("api down")
        out = await integrate_grounding_correction(_RESPONSE, _VERDICT, _Err())
        assert out is None
