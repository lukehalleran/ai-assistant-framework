"""tests/unit/test_sep22_calendar_backstop_user_words.py

2026-09-22 turn-audit fix (lane GF B1) + adversarial review (Codex round 3).
Live incident, structure only: the calendar STATE-claim backstop in
`gui/handlers.py` compared a reply's claim ONLY against this turn's gathered
`google_calendar` events; a reply that RESTATED the user's own status report
("...sleeping til 10 when you had a noon appointment...", echoing the user's
"...then did the appointment in bed") got a false "I don't see that on your
calendar — nothing was created" notice, which was stored, audited and
re-rendered next turn.

Everything here drives deployed functions with SYNTHETIC text (no owner
vocabulary — the original live fixture tripped the privacy scan, BC-61):

(a) `action_claim_guard.claims_calendar_state` still detects the live-shaped
    region — the fix lives at the BACKSTOP, not by weakening the detector;
(b) `gui.handlers._calendar_claim_user_corroborated`: a user-authored span
    that ASSERTS the clause's own calendar noun is evidence; a request, a
    question, or a negated mention is not (review finding 1); a persisted
    calendar-state claim ("already on your calendar") needs the user to
    assert calendar state, attendance alone is not proof; explicit weekday /
    date / clause-scoped clock time must agree where both sides state them —
    and a clock time ELSEWHERE in a multi-event status message ("took meds
    at 1030 ... then did the appointment") never attaches to the noun;
(c) `gui.handlers._apply_action_guard` end-to-end: the live-shaped turn gets
    NO notice; an unrelated control claim with no user corroboration still
    gets one; a recent turn's user-authored `query` corroborates, its
    `response` never does.

class: BC-50, BC-58 (also BC-61 for the synthetic fixtures).
"""
import asyncio
from types import SimpleNamespace

import core.action_claim_guard as action_claim_guard
import gui.handlers as handlers


# Live-shaped synthetic pair: a multi-event status message whose calendar
# noun sits in its own clause, with unrelated clock times earlier.
USER_STATUS = (
    "Yeah just finished the session, walking to the store for snacks. Still waking up tbh lol "
    "I took my pills at 1030 and slept til 10 before it then did the appointment in bed"
)
RESTATING_REPLY = (
    "Doing the session from bed still counts. And honestly, sleeping til 10 when you had a noon "
    "appointment isn't a bad trade; you showed up, just horizontally. The walk will help burn off "
    "the grogginess."
)
PLAIN_REPLY = "You had a noon piano appointment on Friday, and it sounds like it went well."
PERSISTED_REPLY = "Your piano appointment Friday at noon is already on your calendar."
CONTROL_REPLY = (
    "It's also already on your calendar as a recurring weekly event "
    "(through December 12, Zoom link attached)."
)
EVENTS = [
    {"summary": "Project Review", "start": "2026-09-24T14:00:00"},
    {"summary": "Quarterly Checklist Due", "start": "2026-09-26T00:00:00", "all_day": True},
]


def _ctx(user_text="", raw_context=None):
    return SimpleNamespace(
        user_text=user_text,
        user_text_ws=user_text,
        orchestrator=SimpleNamespace(),
        raw_context=raw_context or {},
    )


def _claims(reply):
    claims = action_claim_guard.claims_calendar_state(reply)
    assert claims, reply
    return claims


def _corroborated(reply, spans):
    return any(handlers._calendar_claim_user_corroborated(c, spans) for c in _claims(reply))


class TestDetectorUnchanged:
    def test_live_shaped_region_and_persisted_claim_still_detected(self):
        assert any("noon appointment" in c.lower() for c in _claims(RESTATING_REPLY))
        assert any("piano appointment" in c.lower() for c in _claims(PERSISTED_REPLY))


class TestUserCorroboration:
    def test_live_shaped_status_report_corroborates_the_restatement(self):
        assert _corroborated(RESTATING_REPLY, [USER_STATUS]) is True

    def test_soft_wrapped_and_plain_self_reports_corroborate(self):
        assert _corroborated(PLAIN_REPLY, ["I had a noon piano appointment on Friday and it went fine."]) is True
        assert _corroborated(PLAIN_REPLY, ["I had a noon piano appointment on Friday\n  and attended it from home."]) is True
        # The noun alone, with no conflicting detail, is the user asserting the event.
        assert _corroborated(PLAIN_REPLY, ["I had an appointment on Friday."]) is True

    def test_request_question_or_negated_mention_is_not_evidence(self):
        assert _corroborated(PLAIN_REPLY, ["Can you add a noon appointment on Friday?"]) is False
        assert _corroborated(PLAIN_REPLY, ["Did I have an appointment on Friday?"]) is False
        assert _corroborated(PLAIN_REPLY, ["I don't have a noon appointment on Friday."]) is False
        assert _corroborated(PERSISTED_REPLY, ["Can you put my piano appointment Friday at noon on my calendar?"]) is False

    def test_span_sharing_only_a_non_calendar_word_does_not_corroborate(self):
        assert _corroborated(PLAIN_REPLY, ["honestly just a long day at the store"]) is False

    def test_explicit_when_details_must_agree(self):
        assert _corroborated(PLAIN_REPLY, ["I had an appointment on Monday."]) is False
        assert _corroborated(PLAIN_REPLY, ["I had an appointment Friday at 3 pm."]) is False
        assert _corroborated(
            "Your dentist appointment October 5 at noon is on your calendar.",
            ["I put my dentist appointment October 12 on my calendar."]) is False
        assert _corroborated(
            "Your dentist appointment October 5 at noon is on your calendar.",
            ["I put my dentist appointment October 5 on my calendar."]) is True

    def test_clock_time_in_another_clause_never_attaches_to_the_noun(self):
        assert _corroborated(PLAIN_REPLY, ["Took meds at 1030 then did the appointment in bed"]) is True

    def test_persisted_state_claim_needs_user_calendar_assertion(self):
        assert _corroborated(PERSISTED_REPLY, ["I attended my piano appointment Friday at noon."]) is False
        assert _corroborated(PERSISTED_REPLY, ["My piano appointment Friday at noon is on my calendar."]) is True

    def test_no_anchor_or_no_spans_is_false(self):
        assert handlers._calendar_claim_user_corroborated("", [USER_STATUS]) is False
        assert handlers._calendar_claim_user_corroborated(_claims(PLAIN_REPLY)[0], []) is False
        assert handlers._calendar_claim_user_corroborated("it's all set for Saturday", [USER_STATUS]) is False


class TestApplyActionGuardEndToEnd:
    def _guard(self, ctx, reply):
        return asyncio.run(handlers._apply_action_guard(
            ctx, reply, executed_kinds=set(), proposed_kinds=set(), self_repair=False,
        ))

    def test_live_shaped_turn_gets_no_notice_and_control_still_does(self):
        live = self._guard(_ctx(USER_STATUS, {"google_calendar": EVENTS}), RESTATING_REPLY)
        control = self._guard(_ctx("ok", {"google_calendar": EVENTS}), CONTROL_REPLY)
        assert "I don't see that on your calendar" not in live
        assert "I don't see that on your calendar" in control

    def test_recent_turn_user_query_corroborates_but_response_never_does(self):
        via_query = self._guard(_ctx("ok", {
            "google_calendar": EVENTS,
            "recent_conversations": [{"query": USER_STATUS, "response": "sure"}],
        }), RESTATING_REPLY)
        via_response = self._guard(_ctx("ok", {
            "google_calendar": EVENTS,
            "recent_conversations": [{"query": "unrelated", "response": USER_STATUS}],
        }), RESTATING_REPLY)
        assert "I don't see that on your calendar" not in via_query
        assert "I don't see that on your calendar" in via_response
