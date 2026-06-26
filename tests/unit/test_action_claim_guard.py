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
# Third-/second-person narration is NOT an assistant self-claim
# ---------------------------------------------------------------------------

# The exact text from the production incident: Daemon, demonstrating its memory
# to another model, narrated something the *user* did. The third-person "He
# emailed his counselor" tripped the email completion-claim detector, appending a
# bogus "I didn't actually send that email. Want me to do it now?" notice.
THIRD_PERSON_INCIDENT = (
    "He emailed his counselor Morgan this afternoon and she said it should be "
    "fine — he just needs to confirm with financial aid tomorrow."
)


def test_third_person_email_narration_is_not_a_claim():
    assert detect_completion_claims(THIRD_PERSON_INCIDENT) == []


@pytest.mark.parametrize(
    "text",
    [
        "He emailed his counselor this afternoon.",
        "She sent the report to the team this morning.",
        "They created a doc for the meeting.",
        "He texted you the address.",
        "You saved your note already, nice.",
        "You've added it to your calendar yourself.",
        "They scheduled the call for Tuesday.",
    ],
)
def test_narration_of_others_actions_is_not_a_claim(text):
    assert detect_completion_claims(text) == []


def test_first_person_marker_overrides_narration_exclusion():
    # "I saved …" is a genuine self-claim even with a second-person clause nearby;
    # the exclusion must not swallow it (kind may resolve to email by keyword
    # priority, but a claim MUST be detected — it is not silently dropped).
    assert detect_completion_claims("I saved it right after you emailed me.") != []


def test_assistant_voice_email_claim_still_fires():
    # No third-person subject → the assistant's own email claim must still fire.
    claims = detect_completion_claims("I've sent the email to your advisor.")
    assert any(c.kind == ActionKind.EMAIL for c in claims)


# Production incident #2: the USER sent their own email and reported it; Daemon
# affirmed in passive/second-person ("The email's sent — you caught … you did the
# thing"). The passive "email's sent" tripped completion pattern #6 and the
# second-person actor wasn't excluded, firing a bogus "I didn't actually send that
# email" notice.
SECOND_PERSON_INCIDENT = (
    "The email's sent — you caught the address issue and fixed it. "
    "No bounce back yet is a good sign. Either way, you did the thing you set out to do at 4:30."
)


def test_second_person_passive_email_narration_is_not_a_claim():
    assert detect_completion_claims(SECOND_PERSON_INCIDENT) == []


@pytest.mark.parametrize(
    "text",
    [
        "The email's sent — you caught the address issue.",
        "So the note's saved now, you finally got it done.",
        "Either way, you did the thing you set out to do.",
    ],
)
def test_second_person_subject_narration_is_not_a_claim(text):
    assert detect_completion_claims(text) == []


def test_object_you_and_possessive_your_remain_self_claims():
    # "you" as an OBJECT and possessive "your" are NOT second-person narration —
    # the assistant is still the actor, so these must still fire.
    assert detect_completion_claims("Sent you the draft via email.") != []
    assert detect_completion_claims("Saved your note to the file.") != []


# Production incident #1: Daemon DRAFTS an email (in the user's first person,
# fenced by ---). "When I emailed her" inside the draft is the user's past action,
# not Daemon claiming it sent mail — but it tripped the first-person email pattern.
DRAFTED_EMAIL = """Here's a draft:

---

**Subject:** Need Disability Services Advisor

Dear ODS Team,

When I emailed her, I received an auto-reply. I've also contacted my TA.
I can provide medical records or a letter from my clinician.

Thank you,
Luke

---

I kept it general. Want me to fire it off via email, or you got it from here?"""


def test_drafted_email_body_is_not_a_completion_claim():
    # The fenced draft is the user's voice; its first-person verbs must not be
    # read as Daemon self-claims.
    assert detect_completion_claims(DRAFTED_EMAIL) == []


def test_first_person_claim_outside_a_draft_fence_still_fires():
    # A real self-claim that happens to sit after an unrelated --- divider must
    # NOT be swallowed (unpaired trailing rule → nothing stripped).
    text = "Here's the summary.\n\n---\n\nI've sent the email to your advisor."
    claims = detect_completion_claims(text)
    assert any(c.kind == ActionKind.EMAIL for c in claims)


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
