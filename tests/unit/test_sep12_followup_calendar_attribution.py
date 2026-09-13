"""2026-09-12 follow-up: calendar-claim OWNER ATTRIBUTION + clause-scoped voice.

Source: the Sep-12 implementation-referee follow-up review (finding 1,
Codex + two cheap reviewers, /tmp/daemon_sep12_followup_review.md). The
previous batch's ``_claim_sentence_eligible`` (A22) gated card/calendar
claims on VOICE + a same/adjacent-SENTENCE referent anchor, but never
required the claim to actually be attributed to the ADDRESSED OWNER's own
calendar — so third-party/hypothetical calendar sentences with no "your"
anywhere near them ("The event is already scheduled for March.", "Their
appointment is already scheduled for Friday at 3 PM.") still returned as
claims, and a REPORTING FRAME ("The email says the meeting is already
scheduled, and it is already on your calendar for Friday at 3 PM.") vetoed
the whole sentence instead of scoping the veto to the reported CLAUSE,
losing a real, independently-owner-attributed claim in the same sentence.

This batch adds a clause-splitting layer (``_claim_clause_spans``),
per-clause voice-ok REGIONS (``_voice_ok_regions`` — a maximal run of
consecutive voice-ok clauses; the whole sentence when every clause is
voice-ok, so existing byte-identical-text assertions are unaffected), and
an OWNER-CALENDAR-SURFACE attribution grammar (S1/S2/S3 direct-surface
patterns + a D definite/anaphoric-reference pattern that only attributes
when an EARLIER voice-ok clause of the same reply already established
surface) that ``claims_calendar_state`` now requires before returning a
sentence/region as a claim. The card family (``claims_pending_card`` +
the annotator's card branch) is rescoped to voice-ok REGIONS (instead of
whole sentences) so a reported clause can no longer veto an independently
owner-attributed claim living in the same sentence.

class: BC-04
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.action_claim_guard import (
    ActionKind,
    NO_CARD_NOTICE,
    annotate_unverified_action_claim,
    claims_calendar_state,
    claims_pending_card,
)

# pytest.ini sets asyncio_mode = auto — async test functions below need no
# explicit @pytest.mark.asyncio.


class _FakeActionGuardOrchestrator:
    active_documents = None


def _make_ctx(*, user_text="x", raw_context=None):
    return SimpleNamespace(
        user_text=user_text,
        user_text_ws=user_text,
        orchestrator=_FakeActionGuardOrchestrator(),
        raw_context=raw_context or {},
    )


def _wrap(text: str, anchor: str) -> str:
    """Insert a client-style soft line-wrap ("\\n  ") right before the
    first occurrence of `anchor` in `text` — same helper as
    test_sep12_action_claim_eligibility.py, exercising the normalize_ws
    ingress chokepoint (A10)."""
    idx = text.index(anchor)
    return text[:idx] + "\n  " + text[idx:]


CAL_NOTICE = "I don't see that on your calendar"

UNRELATED = [
    {"summary": "Dentist cleaning", "start": "2026-09-15T09:00:00", "end": "2026-09-15T10:00:00"},
]  # Tuesday
MATCHING_FRIDAY = [
    {"summary": "Team sync", "start": "2026-09-18T15:00:00", "end": "2026-09-18T16:00:00"},
]  # Friday


# ===========================================================================
# (A) Not claims — clean AND wrapped.
# ===========================================================================

# All 8 rows must NOT be a calendar-state claim under the new
# owner-attribution rule (section 4/5 of the plan) — checked for every row
# below. THREE of the eight ALSO independently trip the pre-existing,
# OUT-OF-CONTRACT `detect_completion_claims`/`_COMPLETION_PATTERNS`
# pattern #5 ("scheduled" + up to 40 chars + a/an/the/your/this/that/it —
# a pattern meant for ACTIVE constructions like "saved the note" that also
# accidentally matches an unrelated determiner up to 40 chars after a
# PASSIVE "already scheduled"), which makes
# `annotate_unverified_action_claim` mark them regardless of anything this
# batch touches (verified directly against the deployed
# `detect_completion_claims` — this collision predates and is independent
# of sections 1-7). Documented residual, reported to the frontier per the
# plan's STOP condition #2 (a case cannot pass without touching code
# outside the enumerated contract) — NOT fixed here. See
# TestAAnnotateResidualOutOfContract below.
_A_NOT_CLAIMS = [
    "The event is already scheduled for March.",
    "Their appointment is already scheduled for Friday at 3 PM.",
    "Your professor's office hours are already scheduled for Friday at 3 PM.",
    "The vote is already scheduled for March on the legislative calendar.",
    "The event is already scheduled for March, so put it on your calendar.",
    "The email says the meeting is already on your calendar for Friday at 3 PM.",
    "The email says the meeting is already scheduled, and that it is already "
    "on your calendar for Friday at 3 PM.",
    "If you have an appointment on Friday at 3 PM, bring the forms.",
]
_A_NOT_CLAIMS_ANCHORS = [
    "already scheduled",
    "already scheduled",
    "already scheduled",
    "already scheduled",
    "put it",
    "already on your calendar",
    "already on your calendar",
    "bring the forms",
]

# Rows whose `annotate_unverified_action_claim` output is unaffected by the
# out-of-contract collision above — asserted unchanged, clean and wrapped.
_A_ANNOTATE_UNCHANGED_IDX = [0, 1, 2, 5, 7]
# Rows that trip the pre-existing detect_completion_claims collision.
_A_ANNOTATE_RESIDUAL_IDX = [3, 4, 6]


class TestANotClaims:
    @pytest.mark.parametrize("text", _A_NOT_CLAIMS)
    def test_clean_not_a_claim(self, text):
        assert claims_calendar_state(text) == []

    @pytest.mark.parametrize("text,anchor", list(zip(_A_NOT_CLAIMS, _A_NOT_CLAIMS_ANCHORS)))
    def test_wrapped_not_a_claim(self, text, anchor):
        wrapped = _wrap(text, anchor)
        assert claims_calendar_state(wrapped) == []

    @pytest.mark.parametrize("i", _A_ANNOTATE_UNCHANGED_IDX)
    def test_clean_annotate_unchanged(self, i):
        text = _A_NOT_CLAIMS[i]
        assert annotate_unverified_action_claim(text) == text

    @pytest.mark.parametrize("i", _A_ANNOTATE_UNCHANGED_IDX)
    def test_wrapped_annotate_unchanged(self, i):
        wrapped = _wrap(_A_NOT_CLAIMS[i], _A_NOT_CLAIMS_ANCHORS[i])
        assert annotate_unverified_action_claim(wrapped) == wrapped

    def test_first_sentence_known_limit_second_sentence_excluded(self):
        text = "Your calendar is clear on Friday. The conference is already scheduled for March."
        result = claims_calendar_state(text)
        assert "The conference is already scheduled for March." not in result


class TestAAnnotateResidualOutOfContract:
    """DOCUMENTED RESIDUAL (not fixed by this batch): the plan's spec
    calls for `annotate_unverified_action_claim` to leave these three rows
    unchanged, but `detect_completion_claims` — checked FIRST inside
    `annotate_unverified_action_claim`, and entirely outside this batch's
    contract (sections 1-7 touch only the card/calendar claim families and
    the annotator's regex-only card/calendar branches) — independently
    marks them via its pre-existing pattern #5 over-match. Verified
    directly against the deployed `detect_completion_claims` (see the
    final report). `claims_calendar_state` itself correctly returns []
    for all three (asserted in TestANotClaims above)."""

    @pytest.mark.parametrize("i", _A_ANNOTATE_RESIDUAL_IDX)
    def test_clean_annotate_marked_by_out_of_contract_completion_check(self, i):
        text = _A_NOT_CLAIMS[i]
        assert claims_calendar_state(text) == []
        assert annotate_unverified_action_claim(text).endswith(
            "[unverified action claim]"
        )

    @pytest.mark.parametrize("i", [3, 6])
    def test_wrapped_annotate_marked_by_out_of_contract_completion_check(self, i):
        wrapped = _wrap(_A_NOT_CLAIMS[i], _A_NOT_CLAIMS_ANCHORS[i])
        assert claims_calendar_state(wrapped) == []
        assert annotate_unverified_action_claim(wrapped).endswith(
            "[unverified action claim]"
        )

    def test_wrapped_index_4_residual_does_not_survive_the_wrap(self):
        """Index 4's soft-wrap anchor ("put it") happens to land the
        line-wrap's embedded newline BETWEEN "scheduled" and "it" —
        `detect_completion_claims` (unlike this batch's calendar-family
        functions) never calls `normalize_ws`, so its own
        ``_split_sentences`` treats that newline as a sentence break and
        the two tokens driving the out-of-contract pattern #5 collision
        end up in different sentences. The collision is real (see the
        clean-text test above) but does not survive THIS PARTICULAR wrap
        — documented, not asserted as a general property."""
        text = _A_NOT_CLAIMS[4]
        wrapped = _wrap(text, _A_NOT_CLAIMS_ANCHORS[4])
        assert claims_calendar_state(wrapped) == []
        assert annotate_unverified_action_claim(wrapped) == wrapped


# ===========================================================================
# (B) Claims — exact return, clean AND wrapped (wrapped result == clean result).
# ===========================================================================

class TestBClaimsExactReturn:
    def test_bare_it_on_calendar(self):
        text = "It's already on your calendar for Friday at 3 PM."
        assert claims_calendar_state(text) == [text]
        wrapped = _wrap(text, "already on your calendar")
        assert claims_calendar_state(wrapped) == [text]

    def test_comma_form_scopes_to_the_owner_attributed_clause(self):
        text = (
            "The email says the meeting is already scheduled, and it is "
            "already on your calendar for Friday at 3 PM."
        )
        expected = ["it is already on your calendar for Friday at 3 PM."]
        assert claims_calendar_state(text) == expected
        wrapped = _wrap(text, "already on your calendar")
        assert claims_calendar_state(wrapped) == expected

    def test_period_form(self):
        text = (
            "The email says the meeting is already scheduled. "
            "It is already on your calendar for Friday at 3 PM."
        )
        expected = ["It is already on your calendar for Friday at 3 PM."]
        assert claims_calendar_state(text) == expected
        wrapped = _wrap(text, "already on your calendar")
        assert claims_calendar_state(wrapped) == expected

    def test_semicolon_form(self):
        text = (
            "The email says the meeting is already scheduled; it is "
            "already on your calendar for Friday at 3 PM."
        )
        expected = ["it is already on your calendar for Friday at 3 PM."]
        assert claims_calendar_state(text) == expected
        wrapped = _wrap(text, "already on your calendar")
        assert claims_calendar_state(wrapped) == expected

    def test_dash_form(self):
        text = (
            "The email says the meeting is already scheduled — it is "
            "already on your calendar for Friday at 3 PM."
        )
        expected = ["it is already on your calendar for Friday at 3 PM."]
        assert claims_calendar_state(text) == expected
        wrapped = _wrap(text, "already on your calendar")
        assert claims_calendar_state(wrapped) == expected

    def test_i_checked_your_calendar_prefix(self):
        text = "I checked your calendar. The event is already scheduled for Friday at 3 PM."
        expected = ["The event is already scheduled for Friday at 3 PM."]
        assert claims_calendar_state(text) == expected
        wrapped = _wrap(text, "already scheduled")
        assert claims_calendar_state(wrapped) == expected

    def test_you_have_office_hours(self):
        text = "You have office hours tonight at 8 PM."
        assert claims_calendar_state(text) == [text]
        wrapped = _wrap(text, "office hours")
        assert claims_calendar_state(wrapped) == [text]

    def test_annotator_comma_form_marked(self):
        text = (
            "The email says the meeting is already scheduled, and it is "
            "already on your calendar for Friday at 3 PM."
        )
        assert annotate_unverified_action_claim(text).endswith("[unverified action claim]")

    def test_annotator_period_form_marked(self):
        text = (
            "The email says the meeting is already scheduled. "
            "It is already on your calendar for Friday at 3 PM."
        )
        assert annotate_unverified_action_claim(text).endswith("[unverified action claim]")


# ===========================================================================
# (C) Card family.
# ===========================================================================

class TestCCardFamily:
    def test_reported_frame_with_independent_owner_directive_still_fires(self):
        assert claims_pending_card(
            "The email says the card is pending, and you can approve it below."
        ) is True

    def test_reported_frame_alone_does_not_fire(self):
        assert claims_pending_card("The email says the card is pending.") is False


# ===========================================================================
# (D) Semantic channel obeys attribution.
# ===========================================================================

class TestDSemanticChannelObeysAttribution:
    def test_semantic_hit_without_attribution_does_not_fire(self, monkeypatch):
        import core.action_claim_guard as guard
        monkeypatch.setattr(
            guard, "_claim_semantic_hit", lambda s, label: label == "calendar_state"
        )
        assert claims_calendar_state("Everything for the talk is squared away.") == []

    def test_semantic_hit_with_attribution_fires(self, monkeypatch):
        import core.action_claim_guard as guard
        monkeypatch.setattr(
            guard, "_claim_semantic_hit", lambda s, label: label == "calendar_state"
        )
        text = "Everything on your calendar is squared away."
        assert claims_calendar_state(text) == [text]


# ===========================================================================
# (E) Handler outcomes via gui.handlers._apply_action_guard.
# ===========================================================================

class TestEHandlerOutcomes:
    async def _suffix(self, text, raw_context):
        from gui.handlers import _apply_action_guard
        ctx = _make_ctx(raw_context=raw_context)
        return await _apply_action_guard(
            ctx, text, executed_kinds=set(), proposed_kinds=set(), self_repair=False,
        )

    async def test_the_event_is_already_scheduled_no_notice(self):
        suffix = await self._suffix(
            "The event is already scheduled for March.", {"google_calendar": UNRELATED}
        )
        assert CAL_NOTICE not in suffix

    async def test_their_appointment_no_notice(self):
        suffix = await self._suffix(
            "Their appointment is already scheduled for Friday at 3 PM.",
            {"google_calendar": UNRELATED},
        )
        assert CAL_NOTICE not in suffix

    async def test_bare_it_on_calendar_unrelated_gets_notice_matching_none(self):
        text = "It's already on your calendar for Friday at 3 PM."
        suffix = await self._suffix(text, {"google_calendar": UNRELATED})
        assert CAL_NOTICE in suffix
        suffix2 = await self._suffix(text, {"google_calendar": MATCHING_FRIDAY})
        assert CAL_NOTICE not in suffix2

    async def test_comma_form_unrelated_and_matching(self):
        text = (
            "The email says the meeting is already scheduled, and it is "
            "already on your calendar for Friday at 3 PM."
        )
        suffix = await self._suffix(text, {"google_calendar": UNRELATED})
        assert CAL_NOTICE in suffix
        suffix2 = await self._suffix(text, {"google_calendar": MATCHING_FRIDAY})
        assert CAL_NOTICE not in suffix2

    async def test_period_form_unrelated_and_matching(self):
        text = (
            "The email says the meeting is already scheduled. "
            "It is already on your calendar for Friday at 3 PM."
        )
        suffix = await self._suffix(text, {"google_calendar": UNRELATED})
        assert CAL_NOTICE in suffix
        suffix2 = await self._suffix(text, {"google_calendar": MATCHING_FRIDAY})
        assert CAL_NOTICE not in suffix2

    async def test_wrapped_comma_form_matches_clean_outcomes(self):
        text = (
            "The email says the meeting is already scheduled, and it is "
            "already on your calendar for Friday at 3 PM."
        )
        wrapped = _wrap(text, "already on your")
        suffix = await self._suffix(wrapped, {"google_calendar": UNRELATED})
        assert CAL_NOTICE in suffix
        suffix2 = await self._suffix(wrapped, {"google_calendar": MATCHING_FRIDAY})
        assert CAL_NOTICE not in suffix2

    async def test_wrapped_period_form_matches_clean_outcomes(self):
        text = (
            "The email says the meeting is already scheduled. "
            "It is already on your calendar for Friday at 3 PM."
        )
        wrapped = _wrap(text, "scheduled.")
        suffix = await self._suffix(wrapped, {"google_calendar": UNRELATED})
        assert CAL_NOTICE in suffix
        suffix2 = await self._suffix(wrapped, {"google_calendar": MATCHING_FRIDAY})
        assert CAL_NOTICE not in suffix2

    async def test_i_checked_your_calendar_prefix_unrelated_and_matching(self):
        text = "I checked your calendar. The event is already scheduled for Friday at 3 PM."
        suffix = await self._suffix(text, {"google_calendar": UNRELATED})
        assert CAL_NOTICE in suffix
        suffix2 = await self._suffix(text, {"google_calendar": MATCHING_FRIDAY})
        assert CAL_NOTICE not in suffix2


# ===========================================================================
# (G) Referee follow-up: a numeric range is not a clause break.
# ===========================================================================

class TestGNumericRangeDashIsNotAClauseBreak:
    """The clause grammar split on every dash, so "Your appointment runs
    2–3 on Friday." became "Your appointment runs 2" + "3 on Friday": the
    calendar noun and the weekday landed in different clauses and the
    per-clause A21 rule returned no claim, where the pre-A23 whole-sentence
    rule caught it (frontier probe, 4/4 returned [] before the fix). A dash
    or spaced hyphen followed by a digit is a range; a dash before a word
    still splits (see test_dash_form above)."""

    @pytest.mark.parametrize("text", [
        "Your office hours event runs 8–9 PM on Fridays.",
        "Your office hours event runs 8 – 9 PM on Fridays.",
        "Your office hours event runs 8 - 9 PM on Fridays.",
        "Your appointment runs 2–3 on Friday.",
    ])
    def test_range_keeps_the_claim_in_one_clause(self, text):
        assert claims_calendar_state(text) == [text]

    def test_range_inside_a_reported_clause_is_still_reported(self):
        text = "The email says office hours run 8–9 PM on Fridays."
        assert claims_calendar_state(text) == []


# ===========================================================================
# (F) Red-control note: this file was run against the UNMODIFIED module
# before the fix landed and the pass/fail counts recorded in the final
# report (not asserted here — a one-time historical record).
# ===========================================================================
