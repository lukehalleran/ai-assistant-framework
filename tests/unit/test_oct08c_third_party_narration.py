"""Batch 4 (2026-10-08): a NAMED third party narrating an email/message is not
the assistant's own narrated action.

"Sam emailed you about the deadline." has no pronoun subject, so the
pronoun-only exclusion in ``_is_third_party_narration`` missed it and the clause
flowed into ``narrated_unbacked_action_type`` -> SEND_EMAIL. Synthetic names
only. The "unchanged" cases were captured from the base functions before the
fix (expected values below are those captures).

class: BC-48, BC-53
"""

import pytest

from core.action_claim_guard import ActionKind, detect_completion_claims
from core.actions.registry import narrated_unbacked_action_type
from core.actions.types import ActionType


def _kinds(text):
    return [c.kind for c in detect_completion_claims(text)]


class TestNamedSubjectIsNarration:
    @pytest.mark.parametrize("text", [
        "Sam emailed you about the deadline.",
        "Sam Lee sent you the slides.",
        "Yesterday Sam emailed you.",
        "Priya emailed the team about the deadline.",
        "Today Sam sent the report.",
        "Sam has emailed you about the deadline.",
    ])
    def test_no_claim_no_action(self, text):
        assert _kinds(text) == []
        assert narrated_unbacked_action_type(text) is None


class TestFirstPersonStillClaims:
    def test_i_emailed_sam(self):
        text = "I emailed Sam about the deadline."
        assert _kinds(text) == [ActionKind.EMAIL]
        assert narrated_unbacked_action_type(text) == ActionType.SEND_EMAIL

    def test_first_person_beats_named_subject(self):
        text = "I emailed the report after Sam sent the draft."
        assert _kinds(text) == [ActionKind.EMAIL]


class TestUnchangedVersusBase:
    """Sentence-initial TitleCase words that are not subjects: behaviour is
    exactly what the base function returned."""

    @pytest.mark.parametrize("text,kinds,action", [
        ("Just emailed the invite to the team.", [ActionKind.EMAIL], ActionType.SEND_EMAIL),
        ("Already sent the email.", [ActionKind.EMAIL], ActionType.SEND_EMAIL),
        ("Quickly sent the email to Priya.", [ActionKind.EMAIL], ActionType.SEND_EMAIL),
        ("Okay, emailed the report to Sam.", [ActionKind.EMAIL], ActionType.SEND_EMAIL),
        ("Done — sent the email to Sam.", [ActionKind.EMAIL], ActionType.SEND_EMAIL),
        ("Sent Sam the email.", [ActionKind.EMAIL], ActionType.SEND_EMAIL),
        ("Then emailed the invite.", [ActionKind.EMAIL], ActionType.SEND_EMAIL),
        ("Just emailed Sam and saved the note.", [ActionKind.EMAIL], ActionType.SEND_EMAIL),
        ("Sent the email to Sam and saved a note.", [ActionKind.EMAIL], ActionType.SEND_EMAIL),
        ("Just emailed you the notes.", [], None),
        # a sentence-initial action-kind NOUN is the thing acted on, not a person
        ("Email sent to the team.", [ActionKind.EMAIL], ActionType.SEND_EMAIL),
        ("Note is saved.", [ActionKind.NOTE], None),
        ("Reminder saved to the calendar.", [ActionKind.CALENDAR], ActionType.CALENDAR_CREATE_EVENT),
    ])
    def test_unchanged(self, text, kinds, action):
        assert _kinds(text) == kinds
        assert narrated_unbacked_action_type(text) == action
