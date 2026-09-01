"""Regression tests for the 2026-09-01 live grounding-integration misfires.

Three live turns (12:44-13:02): the verifier hallucinated date corrections at
conf 0.9 twice; the integrator (1) shipped "September 6 — correction: just
needs you to confirm" — a leaked splice label plus an impossible
Sunday-as-Monday pair the year-required regexes missed, and (2) rewrote a
CORRECT Sep-9 proposal card to Sep 8, contradicting the pending-store truth;
the force prompt's ET-only timezone example produced a 13:00 ET (= noon
Central) executed event for a local "1 PM"; and an enhanced turn claimed
"Re-queuing the event" with no backend action ("queue" verbs unknown to the
action-claim guard).
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from core.grounding_check import (
    GroundingVerdict,
    integrate_grounding_correction,
    weekday_date_mismatches,
)


class _StubMM:
    def __init__(self, raw):
        self.raw = raw
        self.prompts = []

    async def generate_once(self, prompt, **k):
        self.prompts.append(prompt)
        return self.raw


VERDICT = GroundingVerdict(
    false_claim_present=True,
    claim="next Monday is September 6",
    correction="Next Monday is September 7, 2026.",
    confidence=0.9,
)


class TestWeekdayMismatchCoverage:
    def test_filler_words_and_missing_year_detected(self):
        # The exact live shape that sailed through: intervening words + no year.
        assert weekday_date_mismatches(
            "she's offering the appointment for next Monday, which is September 6"
        )

    def test_consistent_pair_with_filler_passes(self):
        # Sep 7 2026 IS a Monday — no mismatch.
        assert not weekday_date_mismatches(
            "next Monday, which is September 7, works fine"
        )

    def test_strict_shapes_still_detected(self):
        assert weekday_date_mismatches("Saturday, September 7, 2026")
        assert not weekday_date_mismatches("Monday, September 7, 2026")


class TestIntegratorVet:
    def _run(self, mm, response):
        return asyncio.run(
            integrate_grounding_correction(response, VERDICT, mm)
        )

    def test_dropped_corrected_date_rejected(self):
        # No weekday word at all — isolates the date-applied vet from the
        # mismatch vet. Correction says Sep 7; the rewrite asserts Sep 5.
        mm = _StubMM(
            "Good news: she's offering the slot on September 5 — correction: "
            "just needs you to confirm. Reply in Simple Practice today and "
            "ask her to hard-confirm it before the office gives it away."
        )
        original = (
            "Good news: she's offering the appointment slot early next week. "
            "Reply in Simple Practice saying the paperwork is complete and "
            "ask her to hard-confirm it before the office gives it away."
        )
        assert self._run(mm, original) is None  # falls back to suffix

    def test_live_mangle_shape_rejected(self):
        # The exact 2026-09-01 shipped shape: label + wrong date + no
        # corrected fact ("September 6 — correction: just needs you to
        # confirm"; correction said September 7).
        mm = _StubMM(
            "She's offering the appointment for next Monday, which is "
            "September 6 — correction: just needs you to confirm. Reply in "
            "Simple Practice and ask her to hard-confirm it later today."
        )
        original = (
            "She's offering the appointment for next Monday. Reply in Simple "
            "Practice saying the paperwork is 100% complete and ask her to "
            "hard-confirm the rescheduled slot before the end of the day."
        )
        assert self._run(mm, original) is None

    def test_proposal_card_never_rewritten(self):
        card = (
            "\n\n---\n**calendar_create_event** — **Psychiatrist appointment"
            " (rescheduled)** — 2026-09-09T13:00:00-05:00\n"
        )
        prose = (
            "Nice — forms in and the appointment locked for Wednesday, "
            "September 9 at 1:00 PM. Approve the queued event and you're set "
            "for the new psychiatrist intake next week without any more forms."
        )
        rewritten_prose = (
            "Nice — forms in and the appointment locked in for Wednesday, "
            "September 9, 2026 at 1:00 PM sharp. Approve the queued event and "
            "you're set for the new psychiatrist intake next week, forms done."
        )
        mm = _StubMM(rewritten_prose)
        card_verdict = GroundingVerdict(
            false_claim_present=True,
            claim="the appointment is Tuesday, September 8",
            correction="The appointment is Wednesday, September 9 at 1:00 PM.",
            confidence=0.9,
        )
        out = asyncio.run(
            integrate_grounding_correction(prose + card, card_verdict, mm)
        )
        assert out is not None
        # Card reattached verbatim; the integrator never saw it.
        assert out.endswith(card)
        assert "2026-09-09T13:00:00-05:00" in out
        assert all(card.strip() not in p for p in mm.prompts)

    def test_impossible_pair_in_rewrite_rejected(self):
        mm = _StubMM(
            "She's offering next Monday, which is September 6, so reply in "
            "Simple Practice saying the paperwork is complete and ask her to "
            "hard-confirm the slot before the office fills it with someone."
        )
        original = (
            "She's offering the appointment for next Monday. Reply in Simple "
            "Practice saying the paperwork is complete and ask her to "
            "hard-confirm the slot before the office fills it with someone."
        )
        assert self._run(mm, original) is None


class TestVerifierPromptRules:
    def test_calendar_arithmetic_abstain_rule_present(self):
        from core.grounding_check import _build_verifier_prompt
        prompt = _build_verifier_prompt("q", "r")
        assert "Never correct calendar arithmetic" in prompt
        assert "user explicitly stated in their own message" in prompt


class TestCalendarForcePromptTimezone:
    def test_local_timezone_is_default_not_et(self):
        from core.agentic.controller import AgenticSearchController
        from unittest.mock import MagicMock
        from core.actions.types import ActionType
        from core.actions.registry import ACTION_SPECS
        mm = MagicMock(); mm.api_models = {}
        c = AgenticSearchController(model_manager=mm, web_search_manager=MagicMock())
        spec = ACTION_SPECS[ActionType.CALENDAR_CREATE_EVENT]
        prompt = c._build_xml_action_force_prompt(
            "make a calendar event Wednesday the 9th at 1",
            ActionType.CALENDAR_CREATE_EVENT, spec)
        assert "USER'S" in prompt and "LOCAL" in prompt
        # ET appears only as the source-named-zone EXCEPTION, not the default;
        # wall-clock doctrine: offsetless local times, executor strips offsets.
        assert "NO explicit timezone" in prompt
        assert "WITHOUT a UTC offset" in prompt


class TestActionGuardQueueVerbs:
    def test_requeue_claim_detected(self):
        from core.action_claim_guard import detect_completion_claims
        claims = detect_completion_claims(
            "Good catch — Wednesday it is. Re-queuing the event with the "
            "corrected date. Approve that one and we're set."
        )
        assert claims, "queue-verb completion claim not detected"

    def test_ive_queued_detected(self):
        from core.action_claim_guard import detect_completion_claims
        assert detect_completion_claims(
            "I've queued the calendar event for Wednesday at 1 PM."
        )


class TestVerifierNoiseDemotions:
    """2026-09-01 afternoon pair (both conf 0.9): a 'correction' that merely
    RESTATES the user's own request shipped as a ⚠️ suffix twice, and the
    integrator spliced terminology pedantry ("no widely accepted historical
    period ... 'misdiagnosis era'") into an emotional-support reply — the
    verifier policing the user's own life-narrative shorthand."""

    def _verify(self, raw_verdict, response):
        import core.grounding_check as gc
        mm = _StubMM(raw_verdict)
        return asyncio.run(
            gc.verify_grounding("q", response, mm))

    def test_restatement_correction_demoted(self):
        raw = json.dumps({
            "false_claim_present": True,
            "claim": "the appointment time",
            "correction": "The appointment is scheduled for 1 PM on "
                          "September 9, 2026, as per the user's request.",
            "confidence": 0.9,
        })
        response = ("The update is queued: Psychiatrist appointment "
                    "(rescheduled), Wed Sep 9, noon → 1:00 PM. Approve it "
                    "and the move goes through.")
        assert self._verify(raw, response) is None

    def test_dates_already_in_response_demoted(self):
        raw = json.dumps({
            "false_claim_present": True,
            "claim": "x",
            "correction": "The event is on September 9 at 1 PM.",
            "confidence": 0.9,
        })
        response = "Moved to September 9 at 1 PM as requested — all set."
        assert self._verify(raw, response) is None

    def test_terminology_policing_demoted(self):
        raw = json.dumps({
            "false_claim_present": True,
            "claim": "the misdiagnosis era",
            "correction": "There is no widely accepted historical period "
                          "referred to as the 'misdiagnosis era' in medical "
                          "literature.",
            "confidence": 0.9,
        })
        response = ("Eight years means she's seen the whole arc — the "
                    "misdiagnosis era, Casey, all of it.")
        verdict = self._verify(raw, response)
        # Demoted at either layer (parse-time hedge classification or the
        # terminology-policing rule) — what matters is no correction ships.
        assert verdict is None or not (
            verdict.false_claim_present and verdict.correction.strip())

    def test_real_date_correction_survives(self):
        # A correction introducing a fact the response LACKS must survive.
        raw = json.dumps({
            "false_claim_present": True,
            "claim": "due date is September 13",
            "correction": "The correct due date is September 20.",
            "confidence": 0.9,
        })
        response = "Your first homework is due September 13, plenty of runway."
        verdict = self._verify(raw, response)
        assert verdict is not None and verdict.false_claim_present

    def test_prose_correction_untouched(self):
        # No date/time tokens and no restatement phrasing → rule not applicable.
        raw = json.dumps({
            "false_claim_present": True,
            "claim": "refrigerator mother theory lands closer to truth",
            "correction": "The refrigerator-mother theory was discredited "
                          "decades ago; autism is neurodevelopmental.",
            "confidence": 0.9,
        })
        response = "That frame honestly lands closer to truth than people admit."
        verdict = self._verify(raw, response)
        assert verdict is not None and verdict.false_claim_present
