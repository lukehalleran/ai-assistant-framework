"""Write-action offers need a WRITE verb (2026-09-28, plan Y1; class: BC-04, BC-58).

Live 09-28 17:39: "do you want me to search your email for the SAVE plan
deadline notice? ... just say the word." + "yes please" resolved to
send_email (any EMAIL-kind offer clause mapped to SEND_EMAIL) and forced a
card to the owner's own inbox. A read offer must return None so the
affirmation reaches the read-tool continuation.
"""
from __future__ import annotations

import asyncio

from core.actions.registry import offer_action_type
from core.actions.types import ActionType
from core.agentic.gate import evaluate_agentic_gate

LIVE_REPLY = (
    "No search yet - do you want me to search your email for the SAVE plan "
    "deadline notice? Just say the word."
)


class _Corpus:
    def __init__(self, entries):
        self._e = entries

    def get_recent_memories(self, n=1):
        return self._e[:n]


def test_read_offer_is_not_a_write_action():
    assert offer_action_type(LIVE_REPLY) is None


def test_other_read_offers_are_not_write_actions():
    for r in ("Want me to check your inbox for the notice?",
              "Want me to look through your email for it?",
              "I can pull up your Gmail messages from Sam if you'd like."):
        assert offer_action_type(r) is None, r


def test_send_offer_is_send_email():
    assert offer_action_type("Want me to send that email to Sam?") == ActionType.SEND_EMAIL


def test_draft_reply_offer_is_send_email():
    assert offer_action_type("Want me to draft a reply to that email?") == ActionType.SEND_EMAIL


def test_verb_position_email_is_send_email():
    assert offer_action_type("Want me to email Sam about it?") == ActionType.SEND_EMAIL


def test_calendar_offers_unchanged():
    assert offer_action_type(
        "Want me to create the recurring event on your calendar?"
    ) == ActionType.CALENDAR_CREATE_EVENT
    assert offer_action_type(
        "Want me to move the meeting on your calendar to Friday?"
    ) == ActionType.CALENDAR_UPDATE_EVENT
    assert offer_action_type(
        "Want me to cancel the meeting on your calendar?"
    ) == ActionType.CALENDAR_DELETE_EVENT


def test_gate_live_shape_is_tool_continuation_not_forced_action():
    corpus = _Corpus([{"query": "q", "response": LIVE_REPLY,
                       "response_mode": "agentic-search"}])
    d = asyncio.run(evaluate_agentic_gate(
        user_text="yes please", entity_resolver=None, model_manager=None,
        corpus_manager=corpus, intent_info=None))
    assert d.forced_action is None
    assert d.tool_continuation is not None
