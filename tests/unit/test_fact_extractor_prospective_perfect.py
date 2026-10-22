"""Prospective present-perfect and fragment subordinate clauses are not facts.

Incident 2026-10-20 (class: BC-51 modality, sibling of the 2026-10-18
conditional guard): "Once I have finished X, I'll ..." / "Once I have finished
the draft, I am caught up" still stored the consequent. A present-PERFECT
clause under once / when / after is a prospective completion; a PAST perfect
("had finished") and simple past still report facts. A fragment whose ROOT verb
carries the subordinator, or a clause hung on a "User:" speaker label, is the
subordinate clause it is. Drives THE deployed ``FactExtractor.extract_facts``
(dependency parse, no phrase list). Every case is clean AND wrapped/indented.
"""
import asyncio

import pytest

from memory.fact_extractor import FactExtractor

_FE = FactExtractor(use_rebel=False)


def _facts(text):
    return [n.content for n in asyncio.run(_FE.extract_facts(text, ""))]


def _forms(text):
    wrapped = "\n".join("   " + w for w in text.replace(", ", ",\n").replace(" the ", "\n the ", 1).split("\n"))
    return [text, wrapped]


@pytest.mark.parametrize("text", [
    "Once I have finished the draft, I am caught up with the class.",
    "Once I've finished the lab I am caught up with the class.",
    "When I have finished the lab, I am caught up with the class.",
    "After I have finished the lab, I am caught up with the class.",
])
def test_prospective_perfect_consequent_is_not_a_fact(text):
    for form in _forms(text):
        assert not [f for f in _facts(form) if "caught up" in f], form


@pytest.mark.parametrize("text", [
    "Once I have finished the draft, I am caught up with the class.",
])
def test_fragment_antecedent_alone_is_not_a_fact(text):
    for form in _forms(text.split(",")[0]):
        assert _facts(form) == [], form


@pytest.mark.parametrize("text,needle", [
    # simple past / past perfect / main-clause perfect still report
    ("After I had finished the lab, I am caught up with the class.", "caught up"),
    ("Once I finished the move I live in Tampa.", "tampa"),
    ("Once I had finished the move I live in Tampa.", "tampa"),
    ("I had finished the lab once I got home and I am caught up with the class.", "caught up"),
    ("I am caught up with the class.", "caught up"),
])
def test_past_and_plain_clauses_still_yield_the_fact(text, needle):
    # indentation-only wrap: the regex arm's line-based extraction of a
    # sentence broken MID-clause is a separate, pre-existing limitation
    for form in (text, "   " + text + "\n"):
        assert any(needle in f for f in _facts(form)), form
