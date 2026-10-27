"""F1 leftovers (measured 2026-10-26; wave 2 2026-10-27; class: BC-55, BC-58).

(a) "until I finish X" is a PROSPECTIVE temporal subordinator (like once /
    whenever): a present-tense `until` clause is not a report. The set the
    irrealis guard keys on is the module's mechanism, extended with the
    subordinators of the same category; a past-tense `until` clause still
    reports.
(b) `_is_junk_object` accepted "meds recently" at intake (only the curator
    caught it): the structural trailing-when-adverb test now runs at intake.
"""
from __future__ import annotations

import pytest

from memory.fact_extractor import _irrealis_conflict, _is_junk_object


def _wrap(t):
    return t.replace(" until ", "\n until ").replace(" I ", " \nI ", 1)


@pytest.mark.parametrize("shape", [lambda t: t, _wrap, lambda t: "    " + t])
def test_until_present_clause_is_prospective(shape):
    text = shape("I can't relax until I finish the lab report.")
    assert _irrealis_conflict(text, "finish the lab report") is True


@pytest.mark.parametrize("text", [
    "I waited until I finished the lab report.",
    "Until I finished the lab report I was stuck at home.",
])
def test_until_past_clause_still_reports(text):
    assert _irrealis_conflict(text, "the lab report") is False


@pytest.mark.parametrize("obj,rel", [("meds recently", "took"), ("ibuprofen earlier", "took"),
                                     ("  meds recently ", "took")])
def test_trailing_when_adverb_is_junk_at_intake(obj, rel):
    assert _is_junk_object(obj, rel) is True


@pytest.mark.parametrize("obj,rel", [
    ("blue jay", "likes"), ("5 mg recently", "took"),      # a dose keeps its content
    ("meds recently", "work_schedule"),                    # schedule relations exempt
    ("actuarial science", "studies"),
])
def test_content_objects_and_exempt_relations_are_kept(obj, rel):
    assert _is_junk_object(obj, rel) is False
