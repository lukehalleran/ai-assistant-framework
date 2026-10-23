"""2026-10-23 incident (class BC-58, BC-55): a ~230-word status update was forced
into a send_email action. Turn N-1 was an owner probe whose reply recapped the
USER's own action ("- Weeks 4-7 notes sent to Rowan with the meeting email");
turn N began with the ack "Alright." Two guards are pinned here, each proven
independently with the other neutralised:

  1. probe turns never serve as the "prior turn" of the action routes unless
     the current text is itself a probe (core.agentic.gate._prior_turns);
  2. a LONG message whose head clause is only an ack filler is not an offer
     affirmation (core.actions.registry.is_offer_affirmation).

Item 2 of the brief (first-person-only narrated_unbacked_action_type) is NOT
implemented: it contradicts pinned behaviour in
test_action_claim_guard_third_party_narration.py (see the lane handoff).
Fixture names are invented.
"""

from datetime import datetime, timezone
import textwrap

import pytest

from core.actions.registry import is_offer_affirmation, narrated_unbacked_action_type
from core.actions.types import ActionType
from core.agentic import tool_thread
from core.agentic.gate import _prior_tool_followup, _prior_turn_offer_action, _prior_turns

RECAP = "- Weeks 4-7 notes sent to Rowan with the meeting email"
RECAP_REPLY = "Here are the figures you gave me:\n" + RECAP + "\n- Quiz 3 due Friday"
OFFER_REPLY = "Want me to email Rowan the notes?"

STATUS = (
    "Alright. Yeah I just kind of relaxed this morning. I got up around 830 and made "
    "breakfast and then read for a while before lunch. I still need to respond to Tamsin "
    "and Hale about the schedule for next week and I have not done the problem set yet "
    "because the first two questions took me most of yesterday afternoon and I was pretty "
    "wiped out by the end of it so I figured I would pick it back up today after a walk."
)


class _Corpus:
    def __init__(self, entries):
        self._entries = entries  # newest-first

    def get_recent_memories(self, n=1):
        return list(self._entries[:n])


def _entry(query, response):
    return {"query": query, "response": response, "timestamp": datetime.now(timezone.utc)}


def _wrapped(text):
    return textwrap.indent(text, "    ")


PROBE_Q = "[test] quick check, what figures have I given you lately? [/test]"


class TestRecapPremise:
    def test_recap_bullet_is_the_unbacked_narration_the_incident_used(self):
        # Precondition for the guards below: base reads the bullet as an offer.
        assert narrated_unbacked_action_type(RECAP) == ActionType.SEND_EMAIL


class TestProbeNeverPriorTurnForActionRoutes:
    """Guard 1 alone: the current text IS a genuine affirmation."""

    @pytest.mark.parametrize("fmt", [lambda t: t, _wrapped])
    def test_probe_prior_turn_offer_is_not_accepted_by_real_yes(self, fmt):
        corpus = _Corpus([_entry(PROBE_Q, fmt(OFFER_REPLY))])
        assert _prior_turn_offer_action("yes please", corpus) == (None, False)

    def test_real_turn_before_the_probe_is_still_seen(self):
        corpus = _Corpus([
            _entry(PROBE_Q, RECAP_REPLY),
            _entry("can you email Rowan the notes?", OFFER_REPLY),
        ])
        action, _ = _prior_turn_offer_action("yes please", corpus)
        assert action == ActionType.SEND_EMAIL.value

    def test_real_prior_turn_offer_unchanged(self):
        corpus = _Corpus([_entry("hi", OFFER_REPLY)])
        action, _ = _prior_turn_offer_action("yes please", corpus)
        assert action == ActionType.SEND_EMAIL.value

    def test_probe_to_probe_chain_still_forces(self):
        # The probe-envelope-aware route (_prior_tool_followup judges the inner
        # text) must keep working through a probe predecessor.
        tool_thread.reset()
        tool_thread.record_read_tool_calls(
            [{"tool": "email_search", "args": {"query": "loan", "window_days": 30}}])
        offer = "Found nothing in the last 30 days. Want me to check further back?"
        corpus = _Corpus([_entry("[test] check my email [/test]", offer)])
        got = _prior_tool_followup("[test]yes please[/test]", corpus)
        assert got is not None and got["target_tool"] == "email_search"
        # ...but a NON-probe yes does not continue a probe's tool thread.
        assert _prior_tool_followup("yes please", corpus) is None
        tool_thread.reset()

    def test_helper_returns_n_real_turns_past_a_probe_run(self):
        corpus = _Corpus([
            _entry(PROBE_Q, "a"), _entry(PROBE_Q, "b"),
            _entry("real one", "c"), _entry("real two", "d"),
        ])
        got = _prior_turns(corpus, 2, "no probe here")
        assert [m["response"] for m in got] == ["c", "d"]
        got = _prior_turns(corpus, 2, "[test]x[/test]")
        assert [m["response"] for m in got] == ["a", "b"]


class TestLongAckHeadIsNotAcceptance:
    """Guard 2 alone: the prior turn is a REAL (non-probe) turn."""

    @pytest.mark.parametrize("fmt", [lambda t: t, _wrapped])
    def test_status_update_after_recap_does_not_force_action(self, fmt):
        corpus = _Corpus([_entry("what did I send?", RECAP_REPLY)])
        text = fmt(STATUS)
        assert len(text.split()) > 40
        assert _prior_turn_offer_action(text, corpus) == (None, False)

    @pytest.mark.parametrize("fmt", [lambda t: t, _wrapped])
    def test_affirmation_predicate_false_on_incident_shape(self, fmt):
        assert not is_offer_affirmation(fmt(STATUS))
        assert not is_offer_affirmation(fmt("Ok. So today I " + "did a bunch of reading and notes " * 10))

    @pytest.mark.parametrize("fmt", [lambda t: t, _wrapped])
    @pytest.mark.parametrize("text", [
        "yes",
        "sure, go ahead",
        "yeah lets do that, here are the two links: https://a.example https://b.example",
        "ok please do",
        "Yes please send it, and also " + "include the notes from week four and the quiz summary " * 6,
    ])
    def test_real_affirmations_still_true(self, fmt, text):
        assert is_offer_affirmation(fmt(text))


class TestIncidentEndToEnd:
    @pytest.mark.parametrize("fmt", [lambda t: t, _wrapped])
    def test_probe_recap_then_long_status_update(self, fmt):
        corpus = _Corpus([_entry(PROBE_Q, fmt(RECAP_REPLY))])
        assert _prior_turn_offer_action(fmt(STATUS), corpus) == (None, False)


# ---- Item 2 (referee decision): action-context shape ------------------------
UNRELATED = "what figures have I given you lately?"


class TestNarratedUnbackedNeedsAgentOrRequest:
    @pytest.mark.parametrize("fmt", [lambda t: t, _wrapped])
    def test_recap_with_unrelated_prior_is_not_an_action(self, fmt):
        assert narrated_unbacked_action_type(fmt(RECAP), prior_user_text=UNRELATED) is None

    def test_passive_claim_with_matching_request_counts(self):
        assert narrated_unbacked_action_type(
            "Email sent to the team.", prior_user_text="email Rowan the notes"
        ) == ActionType.SEND_EMAIL

    def test_passive_claim_with_unrelated_prior_does_not(self):
        assert narrated_unbacked_action_type(
            "Email sent to the team.", prior_user_text=UNRELATED) is None

    def test_first_person_claim_counts_with_unrelated_prior(self):
        assert narrated_unbacked_action_type(
            "I sent the email to Rowan.", prior_user_text=UNRELATED
        ) == ActionType.SEND_EMAIL

    def test_notice_path_unchanged(self):
        from core.action_claim_guard import NO_CARD_NOTICE
        text = "Queued up the email to Rowan." + NO_CARD_NOTICE
        assert narrated_unbacked_action_type(text, prior_user_text=UNRELATED) \
            == narrated_unbacked_action_type(text) is not None

    @pytest.mark.parametrize("text", [RECAP, "Email sent to the team.", "I sent the email to Rowan."])
    def test_none_prior_is_exactly_todays_behaviour(self, text):
        assert narrated_unbacked_action_type(text) == ActionType.SEND_EMAIL
        assert narrated_unbacked_action_type(text, prior_user_text=None) == ActionType.SEND_EMAIL

    @pytest.mark.parametrize("fmt", [lambda t: t, _wrapped])
    def test_gate_route_recap_real_prior_unrelated_query(self, fmt):
        # Items 1 and 3 neutralised: prior is a REAL turn, current is a short yes.
        corpus = _Corpus([_entry(UNRELATED, fmt(RECAP_REPLY))])
        assert _prior_turn_offer_action("yes please", corpus) == (None, False)

    def test_gate_route_real_request_prior_still_forces(self):
        corpus = _Corpus([_entry("email Rowan the notes", "Email sent to the team.")])
        action, _ = _prior_turn_offer_action("yes please", corpus)
        assert action == ActionType.SEND_EMAIL.value
