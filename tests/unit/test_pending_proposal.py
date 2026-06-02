"""Unit tests for core.pending_proposal."""

import pytest

from core.action_claim_guard import ActionKind, detect_proposals
from core.pending_proposal import (
    PendingProposal,
    PendingProposalStore,
    build_proposal_from_response,
    is_affirmation,
)

PROPOSING_RESPONSE = (
    "Here's a rough shape for it:\n"
    "Week 1 — catch up on the missed videos, then review the bad quiz.\n"
    "Week 2 — start the homework early and run the project in parallel.\n"
    "Want me to drop this 2-week plan into a daemon note so it's there tomorrow?"
)


# ---------------------------------------------------------------------------
# Affirmation detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "sure that makes sense",  # the exact production affirmation
        "sure",
        "yes",
        "yes please",
        "ok do it",
        "go ahead",
        "sounds good",
        "yeah that works",
        "ok, sure",
        "please do",
    ],
)
def test_is_affirmation_true(text):
    assert is_affirmation(text)


@pytest.mark.parametrize(
    "text",
    [
        "no",
        "no thanks",
        "not now",
        "actually no, don't",
        "maybe later",
        "wait",
        "can you instead make a calendar event",
        "sure, but actually don't save it yet",  # negation veto
        "what does that mean exactly and why would I want it saved",  # too long
        "",
        "   ",
    ],
)
def test_is_affirmation_false(text):
    assert not is_affirmation(text)


# ---------------------------------------------------------------------------
# Store lifecycle
# ---------------------------------------------------------------------------


def test_capture_and_consume_on_affirmation():
    store = PendingProposalStore(ttl_turns=2)
    store.bump_turn()  # turn 1: proposing turn
    p = PendingProposal(kind=ActionKind.NOTE, title="2-week plan", body="…")
    store.capture(p)
    assert store.peek() is not None
    assert p.created_turn == 1

    store.bump_turn()  # turn 2: affirmation turn
    got = store.consume_if_affirmed("sure that makes sense")
    assert got is not None and got.title == "2-week plan"
    # One-shot: slot cleared.
    assert store.peek() is None
    assert store.consume_if_affirmed("sure") is None


def test_no_consume_without_affirmation():
    store = PendingProposalStore()
    store.bump_turn()
    store.capture(PendingProposal(kind=ActionKind.NOTE, title="x", body="y"))
    store.bump_turn()
    assert store.consume_if_affirmed("can you also email it") is None
    # Proposal still pending (not consumed, not a yes).
    assert store.peek() is not None


def test_ttl_expiry_discards_stale_proposal():
    store = PendingProposalStore(ttl_turns=1)
    store.bump_turn()  # turn 1
    store.capture(PendingProposal(kind=ActionKind.NOTE, title="x", body="y"))
    store.bump_turn()  # turn 2 (age 1, within ttl)
    store.bump_turn()  # turn 3 (age 2, past ttl=1)
    assert store.consume_if_affirmed("sure") is None
    assert store.peek() is None  # stale proposal cleared


def test_latest_proposal_wins():
    store = PendingProposalStore()
    store.bump_turn()
    store.capture(PendingProposal(kind=ActionKind.NOTE, title="first", body="a"))
    store.capture(PendingProposal(kind=ActionKind.DOCUMENT, title="second", body="b"))
    assert store.peek().title == "second"


# ---------------------------------------------------------------------------
# Proposal construction from a proposing response
# ---------------------------------------------------------------------------


def test_build_proposal_strips_offer_tail():
    detected = detect_proposals(PROPOSING_RESPONSE)[0]
    proposal = build_proposal_from_response(
        PROPOSING_RESPONSE, detected, turn=1, session_id="s1", now=1000.0,
    )
    assert proposal.kind == ActionKind.NOTE
    # The body keeps the plan but drops the trailing "Want me to …?" offer.
    assert "Week 1" in proposal.body
    assert "Want me to drop this" not in proposal.body
    assert proposal.created_turn == 1
    assert proposal.session_id == "s1"
    assert proposal.created_at == 1000.0


def test_build_proposal_picks_category():
    text = "We decided to refactor the gate. Want me to save a note about this decision?"
    detected = detect_proposals(text)[0]
    proposal = build_proposal_from_response(text, detected, now=1.0)
    assert proposal.category == "decisions"
