"""Unit tests for core.action_claim_guard."""

import pytest

from core.action_claim_guard import (
    ActionKind,
    DetectedAction,
    EXTERNAL,
    SELF_REPAIRABLE,
    build_correction_notice,
    detect_completion_claims,
    detect_proposals,
    is_self_repairable,
    verify_claims,
)

# The exact text from the production confabulation incident.
CONFAB_CLAIM = (
    "Done — saving the 2-week plan as a note so it's waiting for you tomorrow "
    "when you sit down fresh. No need to rebuild it from scratch."
)
REAL_PROPOSAL = (
    "By the way, is that your simulation notes? Want me to drop this 2-week plan "
    "into a daemon note so it's there when you sit down tomorrow?"
)


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


def test_taxonomy_partition():
    assert SELF_REPAIRABLE == {ActionKind.NOTE, ActionKind.DOCUMENT}
    assert ActionKind.EMAIL in EXTERNAL
    assert ActionKind.CALENDAR in EXTERNAL
    assert is_self_repairable(ActionKind.NOTE)
    assert not is_self_repairable(ActionKind.EMAIL)
    assert SELF_REPAIRABLE.isdisjoint(EXTERNAL)


# ---------------------------------------------------------------------------
# Completion-claim detection
# ---------------------------------------------------------------------------


def test_detects_the_real_confab_claim():
    claims = detect_completion_claims(CONFAB_CLAIM)
    assert len(claims) == 1
    assert claims[0].kind == ActionKind.NOTE
    assert claims[0].is_proposal is False


@pytest.mark.parametrize(
    "text,kind",
    [
        ("I've saved the note for you.", ActionKind.NOTE),
        ("Done, saved your note.", ActionKind.NOTE),
        ("I made a note of your deadline.", ActionKind.NOTE),
        ("I'll save this as a note.", ActionKind.NOTE),
        ("The note has been saved.", ActionKind.NOTE),
        ("I've sent the email to your advisor.", ActionKind.EMAIL),
        ("Added that to your calendar.", ActionKind.CALENDAR),
        ("I created a document with the report.", ActionKind.DOCUMENT),
    ],
)
def test_detects_completion_claims(text, kind):
    claims = detect_completion_claims(text)
    assert any(c.kind == kind for c in claims), f"expected {kind} claim in {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "Note that your homework is due in two weeks.",
        "I noted that the deadline is tight, but let's focus on videos first.",
        "Here are some notes on the topic for you to read.",
        "Get some rest — tomorrow you start fresh.",
        "Your study plan should cover the videos and the quiz review.",
    ],
)
def test_no_false_positive_claims(text):
    assert detect_completion_claims(text) == []


def test_proposal_is_not_a_claim():
    # The offer that PRECEDED the confab must not be read as a completion claim.
    assert detect_completion_claims(REAL_PROPOSAL) == []


# ---------------------------------------------------------------------------
# Proposal detection
# ---------------------------------------------------------------------------


def test_detects_the_real_proposal():
    props = detect_proposals(REAL_PROPOSAL)
    assert len(props) == 1
    assert props[0].kind == ActionKind.NOTE
    assert props[0].is_proposal is True


@pytest.mark.parametrize(
    "text,kind",
    [
        ("Want me to save this as a note?", ActionKind.NOTE),
        ("Should I email this to your advisor?", ActionKind.EMAIL),
        ("I can add this to your calendar if you'd like.", ActionKind.CALENDAR),
        ("Would you like me to write this up as a document?", ActionKind.DOCUMENT),
    ],
)
def test_detects_proposals(text, kind):
    props = detect_proposals(text)
    assert any(p.kind == kind for p in props)


def test_completion_claim_is_not_a_proposal():
    assert detect_proposals(CONFAB_CLAIM) == []


# ---------------------------------------------------------------------------
# Verification / reconciliation
# ---------------------------------------------------------------------------


def test_note_claim_unbacked_when_nothing_executed():
    claims = detect_completion_claims(CONFAB_CLAIM)
    rec = verify_claims(claims, executed_kinds=set())
    assert rec.has_issue
    assert len(rec.repairable) == 1
    assert rec.repairable[0].kind == ActionKind.NOTE
    assert rec.external_unbacked == []


def test_note_claim_backed_when_note_executed():
    claims = detect_completion_claims(CONFAB_CLAIM)
    rec = verify_claims(claims, executed_kinds={ActionKind.NOTE})
    assert not rec.has_issue
    assert rec.repairable == []


def test_external_claim_goes_to_external_bucket():
    claims = detect_completion_claims("I've sent the email to Bob.")
    rec = verify_claims(claims, executed_kinds=set())
    assert rec.has_issue
    assert rec.external_unbacked and rec.external_unbacked[0].kind == ActionKind.EMAIL
    assert rec.repairable == []


def test_verify_dedupes_identical_claims():
    dup = DetectedAction(kind=ActionKind.NOTE, matched_text="saved the note", is_proposal=False)
    rec = verify_claims([dup, dup], executed_kinds=set())
    assert len(rec.unbacked_claims) == 1


def test_build_correction_notice():
    claims = detect_completion_claims("I've sent the email and added it to your calendar.")
    rec = verify_claims(claims, executed_kinds=set())
    notice = build_correction_notice(rec.external_unbacked)
    assert "didn't actually" in notice.lower()
    assert build_correction_notice([]) == ""
