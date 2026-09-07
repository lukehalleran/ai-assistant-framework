"""B6/B7 (2026-09-06): completion identity + status for utils/completed_plan_claims.py.

Replaces the old token-overlap match (completion cue anywhere + >=2 shared
content tokens with the WHOLE plan sentence) with identity (the plan's
object HEAD noun) + status (a distinct non-completed status, or a negation/
hedge, never counts). Every scenario below has an adversarial control.

FAILED-before evidence (recorded 2026-09-06 against the pre-change module,
loaded via `git show HEAD:` at the time; the comparison is documented here
and NOT re-run — tests must never read git state, see
`tests/unit/test_no_git_state_in_tests.py`). Six scenarios showed the old
token-overlap code giving the WRONG answer where identity+status matching
gives the right one:
  - "He needs to book the appointment." / "Booked the appointment this
    morning." -> HEAD: no match (false negative -- only 1 shared content
    token with the whole plan sentence, below the old >=2 floor). New: match.
  - "He needs to get the car fixed." / "Haven't gotten the car fixed yet."
    -> HEAD: MATCHES (false positive -- the old code had no negation guard
    at all). New: no match (governed by "haven't ... yet").
  - "He needs to pay the electric bill this week." / "Got the water bill
    done this week." -> HEAD: MATCHES (false positive -- "bill"+"week"
    overlap >=2 with a DIFFERENT bill). New: no match (object-head mismatch:
    head="bill", modifier="electric" != "water").
  - "He needs to pay the electric bill." / "Paid that bill this morning."
    -> HEAD: no match (no token "electric" in the statement at all, and
    "paid"/"bill"/"this"/"morning" don't reach 2 unfiltered content-token
    overlap with the plan). New: match (determiner-anchored reference).
  - "He needs to book the appointment." / "That appointment is done." ->
    HEAD: no match (only "appointment" shared >= threshold once "that",
    "is", "done" are filtered/absent from the plan; below the old >=2
    floor). New: match (bare head containment).
  - "He needs to finish the taxes." / two same-day statements, the second
    "Got it done, feeling accomplished." -> HEAD: no match (the completing
    statement itself never names "taxes" at all -- HEAD has no lookback
    mechanism). New: match, via the immediately-previous same-day statement
    naming the head.
`TestDefectScenariosNewBehavior` below pins the NEW answers for those exact
pairs; the old answers are the historical record above.
"""

from __future__ import annotations

from datetime import date

import pytest

from utils.completed_plan_claims import (
    _clause_matches_head,
    _plan_object_head,
    completed_by_user,
    plan_sentences,
    remove_completed_plan_claims,
)


def _stmt(text: str, ts: str) -> dict:
    return {"user_text": text, "timestamp": ts}


AS_OF = date(2026, 9, 1)


# ---------------------------------------------------------------------------
# Held-out scenarios (each with an adversarial control)
# ---------------------------------------------------------------------------

class TestAppointmentIdentity:
    PLAN = "He needs to book the appointment."

    def test_completed_appointment_matches(self):
        match = completed_by_user(
            self.PLAN, [_stmt("Booked the appointment this morning.", "2026-09-05T09:00:00")],
            as_of=AS_OF,
        )
        assert match is not None
        assert match["_matched_clause"] == "Booked the appointment this morning."

    def test_unrelated_document_share_never_matches(self):
        """Verified defect #3: an unrelated statement using the word
        'completed' inside a pasted document's own title must never close a
        real-world plan."""
        match = completed_by_user(
            self.PLAN,
            [_stmt("Attached the exercise about completed tasks.", "2026-09-05T09:00:00")],
            as_of=AS_OF,
        )
        assert match is None

    def test_document_share_guard_even_when_head_word_appears(self):
        """The document-sharing guard fires independently of head overlap --
        an attached document whose OWN text happens to mention "appointment"
        must still not resolve a non-document plan."""
        match = completed_by_user(
            self.PLAN,
            [_stmt("Attached the file about the appointment scheduling process.",
                    "2026-09-05T09:00:00")],
            as_of=AS_OF,
        )
        assert match is None

    def test_determiner_anchored_reference_matches(self):
        match = completed_by_user(
            self.PLAN, [_stmt("That appointment is done.", "2026-09-05T09:00:00")],
            as_of=AS_OF,
        )
        assert match is not None

    def test_document_plan_allows_sharing_verb(self):
        """A plan that IS about a document (e.g. sending a report) is
        completed by a document-sharing verb -- the guard only blocks
        NON-document plans."""
        match = completed_by_user(
            "He needs to send the report.",
            [_stmt("Sent you the report this morning.", "2026-09-05T09:00:00")],
            as_of=AS_OF,
        )
        assert match is not None


class TestCarFixedStatus:
    PLAN = "He needs to get the car fixed."

    def test_completed_matches(self):
        match = completed_by_user(
            self.PLAN, [_stmt("Got the car fixed.", "2026-09-05T09:00:00")], as_of=AS_OF,
        )
        assert match is not None

    def test_negated_completion_never_matches(self):
        match = completed_by_user(
            self.PLAN, [_stmt("Haven't gotten the car fixed yet.", "2026-09-05T09:00:00")],
            as_of=AS_OF,
        )
        assert match is None

    def test_did_not_variant_never_matches(self):
        match = completed_by_user(
            self.PLAN, [_stmt("Did not get the car fixed yet.", "2026-09-05T09:00:00")],
            as_of=AS_OF,
        )
        assert match is None

    def test_cancelled_status_never_matches(self):
        """A distinct status ('cancelled') is never a completion, even
        though 'repair' is a plausible related word."""
        match = completed_by_user(
            self.PLAN, [_stmt("Cancelled the repair.", "2026-09-05T09:00:00")], as_of=AS_OF,
        )
        assert match is None

    def test_rescheduled_status_never_matches(self):
        match = completed_by_user(
            self.PLAN, [_stmt("Rescheduled getting the car fixed.", "2026-09-05T09:00:00")],
            as_of=AS_OF,
        )
        assert match is None


class TestObjectHeadMismatch:
    PLAN = "He needs to pay the electric bill this week."

    def test_matching_bill_completes(self):
        match = completed_by_user(
            self.PLAN, [_stmt("Got the electric bill done this week.", "2026-09-05T09:00:00")],
            as_of=AS_OF,
        )
        assert match is not None

    def test_different_bill_never_matches(self):
        """Object-head mismatch: 'water bill' shares 'bill'/'week'/'done'
        with the electric-bill plan, but is a DIFFERENT object."""
        match = completed_by_user(
            self.PLAN, [_stmt("Got the water bill done this week.", "2026-09-05T09:00:00")],
            as_of=AS_OF,
        )
        assert match is None

    def test_determiner_reference_without_repeating_modifier(self):
        match = completed_by_user(
            "He needs to pay the electric bill.",
            [_stmt("Paid that bill this morning.", "2026-09-05T09:00:00")],
            as_of=AS_OF,
        )
        assert match is not None

    def test_contraction_variant_matches(self):
        match = completed_by_user(
            "He needs to pay the electric bill.",
            [_stmt("I've paid the electric bill.", "2026-09-05T09:00:00")],
            as_of=AS_OF,
        )
        assert match is not None

    def test_reordered_clause_variant_matches(self):
        match = completed_by_user(
            "He needs to pay the electric bill.",
            [_stmt("This morning I paid the electric bill.", "2026-09-05T09:00:00")],
            as_of=AS_OF,
        )
        assert match is not None

    def test_pronoun_subject_variant_matches(self):
        match = completed_by_user(
            "He needs to pay the electric bill.",
            [_stmt("He paid that electric bill already.", "2026-09-05T09:00:00")],
            as_of=AS_OF,
        )
        assert match is not None


class TestDigitConflictGuardKept:
    """Existing digit guard (HW6 vs hw7) must survive the rewrite."""

    def test_matching_number_completes(self):
        match = completed_by_user(
            "He plans to finish HW6 by Friday.",
            [_stmt("Finished HW6 early tonight.", "2026-09-05T09:00:00")],
            as_of=AS_OF,
        )
        assert match is not None

    def test_conflicting_number_never_matches(self):
        match = completed_by_user(
            "He plans to finish HW6 by Friday.",
            [_stmt("Finished hw7 last night, turned it in early.", "2026-09-05T09:00:00")],
            as_of=AS_OF,
        )
        assert match is None


class TestRecurringPlanNeverClosed:
    def test_habitual_cue_is_not_a_plan_sentence(self):
        narrative = "He plans to go to the gym every Tuesday."
        assert plan_sentences(narrative) == []

    def test_each_week_variant_is_not_a_plan_sentence(self):
        narrative = "He is going to do laundry each week."
        assert plan_sentences(narrative) == []

    def test_non_recurring_sibling_still_a_plan_sentence(self):
        narrative = "He is going to do laundry this weekend."
        assert plan_sentences(narrative) != []

    def test_remove_completed_plan_claims_never_touches_recurring_plan(self):
        narrative = "He plans to go to the gym every Tuesday."
        statements = [_stmt("Went to the gym on Tuesday, felt great.", "2026-09-05T09:00:00")]
        revised, removed = remove_completed_plan_claims(narrative, statements, as_of=AS_OF)
        assert revised == narrative
        assert removed == []


class TestBareItPreviousStatementLookback:
    PLAN = "He needs to finish the taxes."

    def test_bare_it_resolves_via_immediately_previous_same_day_statement(self):
        statements = [
            _stmt("Thinking about the taxes today.", "2026-09-05T09:00:00"),
            _stmt("Got it done, feeling accomplished.", "2026-09-05T15:00:00"),
        ]
        match = completed_by_user(self.PLAN, statements, as_of=AS_OF)
        assert match is not None
        assert match["_matched_clause"] == "Got it done"

    def test_bare_it_without_a_prior_statement_never_matches(self):
        statements = [_stmt("Got it done, feeling accomplished.", "2026-09-05T15:00:00")]
        match = completed_by_user(self.PLAN, statements, as_of=AS_OF)
        assert match is None

    def test_bare_it_prior_statement_must_be_same_day(self):
        statements = [
            _stmt("Thinking about the taxes today.", "2026-09-03T09:00:00"),
            _stmt("Got it done, feeling accomplished.", "2026-09-05T15:00:00"),
        ]
        match = completed_by_user(self.PLAN, statements, as_of=AS_OF)
        assert match is None

    def test_bare_it_prior_statement_must_actually_name_the_head(self):
        statements = [
            _stmt("Had a quiet morning.", "2026-09-05T09:00:00"),
            _stmt("Got it done, feeling accomplished.", "2026-09-05T15:00:00"),
        ]
        match = completed_by_user(self.PLAN, statements, as_of=AS_OF)
        assert match is None


class TestCautionLineQuotesClause:
    def test_caution_line_quotes_matched_clause_not_first_160_chars(self):
        narrative = "He needs to pay the electric bill this week."
        long_preamble = (
            "Talked about the weather and the weekend plans for a while, "
            "nothing much new there, just a lot of small talk before getting "
            "into the actual update. "
        )
        statements = [_stmt(long_preamble + "Got the electric bill done this week.",
                              "2026-09-05T09:00:00")]
        revised, removed = remove_completed_plan_claims(narrative, statements, as_of=AS_OF)
        assert removed
        assert "electric bill" in revised
        # The long unrelated preamble must NOT be what the caution line quotes.
        assert "Talked about the weather" not in revised


# ---------------------------------------------------------------------------
# Direct unit coverage of the new leaf helpers
# ---------------------------------------------------------------------------

class TestPlanObjectHead:
    def test_simple_object(self):
        assert _plan_object_head("He needs to book the appointment.") == ("appointment", "")

    def test_compound_object(self):
        assert _plan_object_head("He needs to pay the electric bill.") == ("bill", "electric")

    def test_no_object_yields_empty_head(self):
        assert _plan_object_head("") == ("", "")

    def test_multi_clause_narrative_isolates_the_plan_clause(self):
        head, modifier = _plan_object_head(
            "Sent weekend hangout invites; hanging out with a friend Saturday is pending."
        )
        assert head == "friend"


class TestClauseMatchesHead:
    def test_bare_head_no_modifier(self):
        assert _clause_matches_head("that appointment is done", "appointment", "")

    def test_compound_requires_modifier_or_determiner(self):
        assert _clause_matches_head("paid the electric bill", "bill", "electric")
        assert _clause_matches_head("paid that bill", "bill", "electric")
        assert not _clause_matches_head("paid the water bill", "bill", "electric")

    def test_no_head_never_matches(self):
        assert not _clause_matches_head("paid the electric bill", "", "")


# ---------------------------------------------------------------------------
# The six documented defect scenarios, pinned to the deployed function.
# ---------------------------------------------------------------------------

class TestDefectScenariosNewBehavior:
    """The exact plan/statement pairs from the module docstring, asserted
    against the deployed ``completed_by_user`` only (the pre-change answers
    are the docstring's historical record — never re-derived from git)."""

    CASES = [
        ("He needs to book the appointment.",
         [("Booked the appointment this morning.", "2026-09-05T09:00:00")],
         True),
        ("He needs to get the car fixed.",
         [("Haven't gotten the car fixed yet.", "2026-09-05T09:00:00")],
         False),
        ("He needs to pay the electric bill this week.",
         [("Got the water bill done this week.", "2026-09-05T09:00:00")],
         False),
        ("He needs to pay the electric bill.",
         [("Paid that bill this morning.", "2026-09-05T09:00:00")],
         True),
        ("He needs to book the appointment.",
         [("That appointment is done.", "2026-09-05T09:00:00")],
         True),
    ]

    def test_documented_scenarios(self):
        for plan, stmt_specs, expected in self.CASES:
            statements = [_stmt(t, ts) for t, ts in stmt_specs]
            result = completed_by_user(plan, statements, as_of=AS_OF)
            assert (result is not None) == expected, f"wrong answer for {plan!r}"

    def test_bare_it_lookback(self):
        plan = "He needs to finish the taxes."
        statements = [
            _stmt("Thinking about the taxes today.", "2026-09-05T09:00:00"),
            _stmt("Got it done, feeling accomplished.", "2026-09-05T15:00:00"),
        ]
        assert completed_by_user(plan, statements, as_of=AS_OF) is not None
