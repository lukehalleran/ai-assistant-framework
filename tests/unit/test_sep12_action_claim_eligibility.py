"""2026-09-12 action-claim VOICE + REFERENT-ANCHOR eligibility (A22).

Two live over-fire classes in the composed card/calendar claim grammar
(``core/action_claim_guard.py``):

  1. A third-party "approve" ("...when asked whether Congress would need to
     approve it, said...") read as though the OWNER were being told to
     approve something — the bare ``approve\\s+(?:it|that|...)`` alternative
     had no subject check.
  2. A bare-pronoun THING+MODAL+STATE hit ("it's there if the question
     resurfaces", "that's given up", "the cleaned-up version", "a good one
     to wake up to", "all there is to it") matched ordinary English with no
     approval-surface word anywhere nearby.

``_claim_sentence_eligible`` (and its ``_claim_voice_ok``/``_claim_anchor_ok``
halves) close both: VOICE excludes reported content (a source/third party
being quoted or paraphrased) and a non-owner-directed "approve" mention;
REFERENT ANCHOR requires an approval-surface word in the same or immediately
preceding sentence for a pronoun THING (a non-pronoun THING like "card"/
"proposal" always satisfies this trivially).

All assertions below call the DEPLOYED functions directly — no
re-derivations. Replay sentences are minimal excerpts (the claim sentence
plus the sentence immediately before it) with any person/place names
replaced by generic phrasing; none of the sentences used here named a
course code or the user's own personal details.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.action_claim_guard import (
    ActionKind,
    annotate_unverified_action_claim,
    claims_calendar_state,
    claims_pending_card,
)

# pytest.ini sets asyncio_mode = auto — async test functions below need no
# explicit @pytest.mark.asyncio.


# ---------------------------------------------------------------------------
# Shared ctx fixture for the handler-level part (d) — same minimal pattern as
# tests/unit/test_sep10_probe_dump_actions.py::_make_action_guard_ctx.
# ---------------------------------------------------------------------------
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
    """Insert a client-style soft line-wrap ("\\n  ") right before the first
    occurrence of `anchor` in `text` — exercises the same normalize_ws
    ingress chokepoint claims_pending_card/claims_calendar_state already
    rely on (A10)."""
    idx = text.index(anchor)
    return text[:idx] + "\n  " + text[idx:]


# ===========================================================================
# (a) SPURIOUS replay sentences (2026-09-10/11/12 corpus rows carrying a
#     "no card to approve" / "I don't see that on your calendar" notice) —
#     each must NOT fire post-fix, with its immediately preceding sentence
#     given as context exactly as the deployed functions see it.
# ===========================================================================

_CONGRESS_PREV = "You weren't — I pulled it up and it's exactly what it sounded like."
_CONGRESS_CUR = (
    "In an exclusive interview, the official doubled down on the funding plan and, "
    "when asked whether Congress would need to approve it, said, \"Well, we think not.\""
)

_OFFICE_HOURS_PREV = (
    "If you're stuck on interpretation, you already have a list of specific "
    "questions to bring."
)
_OFFICE_HOURS_CUR = (
    "One heads-up: your calendar also has professor office hours tonight at 8 PM "
    "— probably not needed either given where you are, but it's there if the "
    "question resurfaces."
)

_REALITY_PREV = (
    "A doctor in 2026 who keeps using a practice after the evidence has turned "
    "against it — that's where negligence starts, because the information was "
    "available and they chose not to update."
)
_REALITY_CUR = (
    "The sin isn't being wrong; it's refusing to track reality when reality is "
    "right there."
)

_GIVEN_UP_PREV = (
    "I'd push back gently on \"most people,\" though — that's a big claim about "
    "what everyone believes, and my sense is more that people are disengaged or "
    "numbed than that they've concluded elections are over."
)
_GIVEN_UP_CUR = (
    "That distinction matters, because it's the difference between a country "
    "that's given up and one that just isn't paying attention yet."
)

# First sentence of its reply — no preceding sentence exists.
_CLEANED_UP_CUR = (
    "That's the cleaned-up version of the weighted franchise argument — and "
    "notice you've now fused both terms of your product from earlier: morality "
    "and information-parsing as one composite disqualifier."
)

_GOES_THROUGH_PREV = (
    "Wrinkle for your setup: different model families route through different "
    "backends, and your notes from earlier showed one of them was blocked "
    "entirely on some providers for a while — so which layer fires can even "
    "vary by which provider is used."
)
_GOES_THROUGH_CUR = (
    "If you want a hard answer for a specific blocked request, the test is "
    "cheap: replay the exact context+query through the API call with safety "
    "settings stripped, and see whether it goes through."
)

# First sentence of its reply — no preceding sentence exists.
_WAKE_UP_CUR = (
    "Ha, that tweet's a good one to wake up to — the staple small-talk bit "
    "makes the deadpan reversal land perfectly."
)

_ALL_THERE_PREV = "Good news is it sounds like it's already easing as the morning goes on."
_ALL_THERE_CUR = (
    "Water, food, and letting it burn off while you start on homework is "
    "about all there is to it."
)


class TestSpuriousReplaySentencesNoLongerFire:
    """Each row: claims_pending_card False, claims_calendar_state == []
    (except the documented office-hours residual — see its own test), and
    annotate_unverified_action_claim returns its input unchanged. Checked
    clean and with a soft line-wrap inserted mid claim-sentence."""

    @pytest.mark.parametrize("prev,cur", [
        (_CONGRESS_PREV, _CONGRESS_CUR),
        (_REALITY_PREV, _REALITY_CUR),
        (_GIVEN_UP_PREV, _GIVEN_UP_CUR),
        (None, _CLEANED_UP_CUR),
        (None, _WAKE_UP_CUR),
        (_ALL_THERE_PREV, _ALL_THERE_CUR),
    ])
    def test_card_family_spurious_rows(self, prev, cur):
        text = f"{prev} {cur}" if prev else cur
        assert claims_pending_card(text) is False
        assert claims_calendar_state(text) == []
        assert annotate_unverified_action_claim(text) == text

    @pytest.mark.parametrize("prev,cur,anchor", [
        (_CONGRESS_PREV, _CONGRESS_CUR, "approve it"),
        (_REALITY_PREV, _REALITY_CUR, "right there"),
        (_GIVEN_UP_PREV, _GIVEN_UP_CUR, "given up"),
        (None, _CLEANED_UP_CUR, "cleaned-up"),
        (None, _WAKE_UP_CUR, "wake up to"),
        (_ALL_THERE_PREV, _ALL_THERE_CUR, "there is to it"),
    ])
    def test_card_family_spurious_rows_wrapped(self, prev, cur, anchor):
        text = f"{prev} {cur}" if prev else cur
        wrapped = _wrap(text, anchor)
        assert claims_pending_card(wrapped) is False
        assert claims_calendar_state(wrapped) == []
        assert annotate_unverified_action_claim(wrapped) == wrapped

    def test_goes_through_calendar_family_spurious(self):
        text = f"{_GOES_THROUGH_PREV} {_GOES_THROUGH_CUR}"
        assert claims_calendar_state(text) == []
        assert claims_pending_card(text) is False
        assert annotate_unverified_action_claim(text) == text

    def test_goes_through_calendar_family_spurious_wrapped(self):
        text = _wrap(f"{_GOES_THROUGH_PREV} {_GOES_THROUGH_CUR}", "goes through")
        assert claims_calendar_state(text) == []
        assert claims_pending_card(text) is False
        assert annotate_unverified_action_claim(text) == text

    def test_office_hours_card_family_no_longer_fires(self):
        """Historically this exact sentence produced the CARD notice ("no
        card to approve") — the bare pronoun "it's ... there" state hit had
        no card/proposal/queue/approve anchor nearby. Post-fix it no longer
        does."""
        text = f"{_OFFICE_HOURS_PREV} {_OFFICE_HOURS_CUR}"
        assert claims_pending_card(text) is False
        assert annotate_unverified_action_claim(text) == text

    def test_office_hours_card_family_no_longer_fires_wrapped(self):
        text = _wrap(f"{_OFFICE_HOURS_PREV} {_OFFICE_HOURS_CUR}", "it's there")
        assert claims_pending_card(text) is False
        assert annotate_unverified_action_claim(text) == text

    def test_office_hours_calendar_family_residual_documented(self):
        """Escalation/residual (see the final report): the frozen spec's
        calendar-family rule (voice check + a _CALENDAR_STRONG_RE anchor in
        the same/preceding sentence) leaves this sentence ELIGIBLE for the
        calendar family, because "calendar"/"office hours" are literally IN
        it — "a TRUE read of the calendar, not a card" per the plan's own
        framing. This is NOT a regression: the historical notice on this
        row was the CARD one (now fixed above), never the calendar one, and
        the under-firing annotator (regex-only, narrower _CALENDAR_STATE_RE
        check — never the A21 entity-anchored rule) still does not
        annotate it, so no user-facing behavior changes here."""
        text = f"{_OFFICE_HOURS_PREV} {_OFFICE_HOURS_CUR}"
        assert claims_calendar_state(text) == [_OFFICE_HOURS_CUR]
        assert annotate_unverified_action_claim(text) == text


# ===========================================================================
# (b) LEGIT replay sentences — must still fire in their family and get
#     annotated. Checked clean and wrapped.
# ===========================================================================

_APPROVE_PROPOSAL = (
    "Approve the proposal and it'll replace the single event with the full series."
)
_ONCE_YOU_APPROVE = (
    "That's already captured in the recurring calendar event we queued up "
    "earlier today (running through December 12), so once you approve that "
    "card it's locked in."
)
_APPROVING_GERUND = "Approving that card puts it on your calendar."
_APPROVE_IT_CALENDAR = "Approve it and it's on your calendar."


class TestLegitReplaySentencesStillFire:
    @pytest.mark.parametrize("text,anchor", [
        (_APPROVE_PROPOSAL, "the proposal"),
        (_ONCE_YOU_APPROVE, "approve that"),
        (_APPROVING_GERUND, "puts it"),
        (_APPROVE_IT_CALENDAR, "on your calendar"),
    ])
    def test_card_family_legit_rows(self, text, anchor):
        assert claims_pending_card(text) is True
        out = annotate_unverified_action_claim(text)
        assert out.endswith("[unverified action claim]")
        wrapped = _wrap(text, anchor)
        assert claims_pending_card(wrapped) is True
        assert annotate_unverified_action_claim(wrapped).endswith(
            "[unverified action claim]"
        )

    def test_approve_it_calendar_also_fires_calendar_family(self):
        """Explicitly required by the plan: this sentence carries BOTH a
        card claim (an owner-directed "Approve it") and a calendar-state
        claim ("it's on your calendar") in one sentence."""
        assert claims_calendar_state(_APPROVE_IT_CALENDAR) == [_APPROVE_IT_CALENDAR]
        wrapped = _wrap(_APPROVE_IT_CALENDAR, "on your calendar")
        # normalize_ws collapses the soft wrap before matching, so the
        # returned sentence is the un-wrapped form (same contract as
        # claims_pending_card's own docstring: "a genuine multi-sentence
        # reply splits identically either way").
        assert claims_calendar_state(wrapped) == [_APPROVE_IT_CALENDAR]


# ===========================================================================
# (c) Synthetic cases
# ===========================================================================

class TestSyntheticVoiceCases:
    def test_third_party_approve_never_fires_or_annotates(self):
        for text in (
            "Congress would need to approve it.",
            "The board will approve that next week.",
        ):
            assert claims_pending_card(text) is False
            assert annotate_unverified_action_claim(text) == text

    def test_owner_directed_approve_fires(self):
        assert claims_pending_card("Approve it and it should land this time.") is True
        assert claims_pending_card("You can approve the card below.") is True

    def test_adjacent_sentence_anchor_satisfies_referent_check(self):
        assert claims_pending_card(
            "I submitted the proposal for the calendar event. It should be up now."
        ) is True

    def test_anchor_does_not_cross_two_sentences(self):
        assert claims_pending_card(
            "I submitted the proposal for the calendar event. Anyway, the "
            "weather is great today. It should be up now."
        ) is False


class TestSyntheticCalendarFamilyCases:
    def test_reported_content_never_a_calendar_claim(self):
        assert claims_calendar_state(
            "The email says the meeting is already scheduled."
        ) == []

    def test_genuine_calendar_state_claim_still_detected(self):
        text = "It's already on your calendar for Tuesday at 3 PM."
        assert claims_calendar_state(text) == [text]

    def test_calendar_anchor_does_not_make_a_card_claim(self):
        assert claims_pending_card(
            "Your calendar also has office hours tonight at 8 PM, and it's "
            "there if you need it."
        ) is False


class TestSyntheticOrdinaryEnglishNeverFires:
    @pytest.mark.parametrize("text", [
        "That's all there is to it.",
        "It's up to you.",
        "This is ready to eat.",
    ])
    def test_never_a_card_claim_or_annotated(self, text):
        assert claims_pending_card(text) is False
        assert annotate_unverified_action_claim(text) == text


# ===========================================================================
# GAP 1 (frontier adversarial probe, 2026-09-12) — a possessive or bare-name
# subject before a reporting verb ("Your professor says…", "My advisor
# said…", "Sam says…") was not recognized as reported content; only the
# enumerated he/she/they/it/"the <noun>" subjects were. Generalized to: a
# reporting verb whose immediately preceding word is not first-person
# ("i"/"we") frames reported content, with "that said" carved out as the
# named discourse-idiom exception.
# ===========================================================================

class TestGap1ReportedContentGeneralized:
    def test_possessive_and_name_subjects_are_reported_content(self):
        for text in (
            "Your professor says office hours are on Tuesday at 3 PM.",
            "My advisor said the event is already scheduled for Tuesday at 3 PM.",
            "Sam says it's on your calendar.",
        ):
            assert claims_calendar_state(text) == []
            assert annotate_unverified_action_claim(text) == text

    def test_that_said_idiom_is_not_reported_content(self):
        assert claims_pending_card("That said, the card is up.") is True

    def test_that_said_idiom_wrapped(self):
        wrapped = _wrap("That said, the card is up.", "the card")
        assert claims_pending_card(wrapped) is True

    def test_first_person_said_unchanged_from_prior_behavior(self):
        """"I said"/"we said" is Daemon's own voice, not reported content —
        this text does not carry a card/calendar claim shape either way, so
        the assertion documents "unchanged", not "now fires"."""
        assert claims_pending_card("I said I'd add it to your calendar.") is False
        assert claims_calendar_state("I said I'd add it to your calendar.") == []


# ===========================================================================
# GAP 2 (frontier adversarial probe, 2026-09-12) — "proposal" is ordinary
# news/legislative/business vocabulary too ("The proposal is up for a vote
# in the Senate"), so it must not self-anchor a card claim the way "card"/
# "approval"/"queue" do. It anchors only alongside a first/second-person
# reference (I/we/you/your) in the same sentence, or an owner-directed
# "approve" mention (which is always a valid anchor regardless).
# ===========================================================================

class TestGap2AmbiguousProposalAnchor:
    def test_third_party_proposal_never_anchors(self):
        for text in (
            "The proposal is up for a vote in the Senate.",
            "Their proposal is already there.",
        ):
            assert claims_pending_card(text) is False
            assert annotate_unverified_action_claim(text) == text

    def test_third_party_proposal_never_anchors_wrapped(self):
        wrapped = _wrap("The proposal is up for a vote in the Senate.", "up for")
        assert claims_pending_card(wrapped) is False
        wrapped2 = _wrap("Their proposal is already there.", "already there")
        assert claims_pending_card(wrapped2) is False

    def test_third_party_proposal_across_two_sentences_never_anchors(self):
        text = "Senate leaders are waiting for the proposal. It is ready."
        assert claims_pending_card(text) is False

    def test_third_party_proposal_across_two_sentences_wrapped(self):
        wrapped = _wrap(
            "Senate leaders are waiting for the proposal. It is ready.", "is ready"
        )
        assert claims_pending_card(wrapped) is False

    def test_first_person_proposal_still_anchors(self):
        text = "I submitted the proposal for the calendar event. It should be up now."
        assert claims_pending_card(text) is True

    def test_first_person_proposal_still_anchors_wrapped(self):
        wrapped = _wrap(
            "I submitted the proposal for the calendar event. It should be up now.",
            "should be up",
        )
        assert claims_pending_card(wrapped) is True

    def test_owner_directed_approve_still_anchors_ambiguous_proposal(self):
        """LEGIT replay row (2026-09-10T16:12) — must keep firing."""
        text = (
            "Approve the proposal and it'll replace the single event with "
            "the full series."
        )
        assert claims_pending_card(text) is True

    def test_unambiguous_card_anchor_needs_no_person_reference(self):
        assert claims_pending_card("The card is waiting below.") is True


# ===========================================================================
# (d) Handler-level: gui.handlers._apply_action_guard
# ===========================================================================

class TestApplyActionGuardEligibility:
    async def test_third_party_congress_reply_gets_no_card_notice_appended(self):
        from core.action_claim_guard import NO_CARD_NOTICE
        from gui.handlers import _apply_action_guard

        ctx = _make_ctx()
        suffix = await _apply_action_guard(
            ctx, _CONGRESS_CUR, executed_kinds=set(), proposed_kinds=set(),
            self_repair=False,
        )
        assert NO_CARD_NOTICE not in suffix

    async def test_wake_up_reply_gets_no_card_notice_appended(self):
        from core.action_claim_guard import NO_CARD_NOTICE
        from gui.handlers import _apply_action_guard

        ctx = _make_ctx()
        suffix = await _apply_action_guard(
            ctx, _WAKE_UP_CUR, executed_kinds=set(), proposed_kinds=set(),
            self_repair=False,
        )
        assert NO_CARD_NOTICE not in suffix

    async def test_owner_directed_approve_gets_no_card_notice(self):
        from core.action_claim_guard import NO_CARD_NOTICE
        from gui.handlers import _apply_action_guard

        ctx = _make_ctx()
        suffix = await _apply_action_guard(
            ctx, "Approve it and it should land this time.",
            executed_kinds=set(), proposed_kinds=set(), self_repair=False,
        )
        assert NO_CARD_NOTICE in suffix

    async def test_no_notice_when_a_calendar_card_was_proposed_this_turn(self):
        from core.action_claim_guard import NO_CARD_NOTICE
        from gui.handlers import _apply_action_guard

        ctx = _make_ctx()
        suffix = await _apply_action_guard(
            ctx, "Approve it and it should land this time.",
            executed_kinds=set(), proposed_kinds={ActionKind.CALENDAR},
            self_repair=False,
        )
        assert NO_CARD_NOTICE not in suffix


# ===========================================================================
# (e) Red-control note: run this file BEFORE the fix landed. Result recorded
# in the final report (not asserted here — this is a historical record of a
# one-time manual check, not a live regression test against unfixed code).
# ===========================================================================
