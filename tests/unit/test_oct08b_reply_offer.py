"""A bare "Want me to draft a reply?" offer binds to a channel noun (2026-10-08).

class: BC-04.  The offer clause is detected by a categorized grammar
(_OFFER_REPLY_RE); the action KIND still comes only from a channel noun in the
reply, so a reply offer about a forum / notification / read-only follow-up
returns no action.
"""
from __future__ import annotations

import asyncio

import pytest

from core.actions.registry import offer_action_type
from core.actions.types import ActionType
from core.agentic.gate import evaluate_agentic_gate


@pytest.mark.parametrize("reply", [
    "Sam emailed you about the deadline. Want me to draft a reply?",
    "Your email inbox has a message from Sam about the deadline. Want me to draft a reply?",
])
def test_email_reply_offer_is_send_email(reply):
    assert offer_action_type(reply) == ActionType.SEND_EMAIL


@pytest.mark.parametrize("reply", [
    "Want me to draft a reply to the forum post?",
    "Got the forum notification. Want me to draft a reply?",
    "Sam emailed. Want me to wait for a reply?",
    "Sam emailed earlier. Want me to look for his reply?",
])
def test_non_send_reply_shapes_return_no_action(reply):
    assert offer_action_type(reply) is None, reply


def test_discord_reply_offer_binds_to_discord():
    got = offer_action_type("Want me to draft a reply on Discord?")
    assert got is not None
    assert "DISCORD" in got.name


class _Corpus:
    def __init__(self, entries):
        self._e = entries

    def get_recent_memories(self, n=1):
        return self._e[:n]


@pytest.mark.parametrize("reply", [
    "Sam emailed you about the deadline. Want me to draft a reply?",
    # No third-party-narration shortcut: only the offer grammar can route this.
    "Your email inbox has a message from Sam about the deadline. Want me to draft a reply?",
])
def test_gate_yes_after_reply_offer_forces_email_send(reply):
    corpus = _Corpus([{"query": "anything new?", "response": reply,
                       "response_mode": "agentic-search"}])
    d = asyncio.run(evaluate_agentic_gate(
        user_text="yes please", entity_resolver=None, model_manager=None,
        corpus_manager=corpus, intent_info=None))
    assert d.forced_action == ActionType.SEND_EMAIL.value
