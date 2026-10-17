"""Completion-claim detection is clause-scoped.

Incident 2026-10-17, class BC-06 (over-fire) / CM-01, A23 lineage: a reply
describing a homework PDF was annotated "[unverified action claim: ...]"
because the kind word ("reports", inside "Crystal Reports output") and the
completion cue ("queue behavior, and Q20 is the") sat in DIFFERENT clauses of
one sentence. Drives THE deployed detect_completion_claims and
annotate_unverified_action_claim.
"""
import pytest

from core.action_claim_guard import (
    annotate_unverified_action_claim as annotate,
    detect_completion_claims,
)

INCIDENT = (
    "A few of them require you to actually build and run small models: "
    "Q6/Q7/Q8 have you drag Create-Process-Dispose, toggle Seize-Delay vs "
    "Seize-Delay-Release, and observe queue behavior, and Q20 is the heaviest "
    "— build a single-server queue (EXPO(10) interarrivals, TRIA(5,8,11) "
    "service), run 100 replications of 1000 minutes, and read server "
    "utilization off the Crystal Reports output."
)
SIBLING = "Part B has you queue the jobs, and the final report is the longest — export the summary table."


def _wrapped(text: str) -> str:
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) > 48:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n   ".join(lines)


@pytest.mark.parametrize("text", [INCIDENT, SIBLING])
@pytest.mark.parametrize("form", ["clean", "wrapped"])
def test_description_of_user_tasks_is_not_a_claim(text, form):
    t = text if form == "clean" else _wrapped(text)
    assert detect_completion_claims(t) == []
    assert "[unverified action claim" not in annotate(t)


@pytest.mark.parametrize("text", [
    "Done — saved that as a note.",
    "I've created the document with your outline.",
    "Re-queuing the event with the corrected date now.",
    "The report is saved.",
])
@pytest.mark.parametrize("form", ["clean", "wrapped"])
def test_real_completion_claims_still_fire(text, form):
    t = text if form == "clean" else _wrapped(text)
    assert len(detect_completion_claims(t)) == 1
    assert "[unverified action claim" in annotate(t)


def test_ill_save_claim_still_fires():
    # Fires on the base tree too (Opus-verified); clause scoping must not lose it.
    assert len(detect_completion_claims("I'll save this as a note.")) == 1
    assert "[unverified action claim" in annotate("I'll save this as a note.")


def test_matched_text_is_whole_sentence():
    s = "Done — saved that as a note."
    assert detect_completion_claims(s)[0].matched_text == s


def test_cross_clause_unrelated_subject_documented():
    # Result: no claim. ", and I" is a clause boundary (A23 grammar), so the
    # kind word ("report") and the cue ("saved") fall in different clauses.
    # (On the base tree this is only a claim if the whole-sentence cue matches.)
    assert detect_completion_claims("The report is long, and I saved time by skipping it") == []
