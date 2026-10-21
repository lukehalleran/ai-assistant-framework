"""
Conditional / irrealis clauses are not facts.

Incident 2026-10-18 14:10 (class: BC-56 adjacent, modality inversion): the user
said that IF they finished assignment 7 THEN they would be caught up; the
regex ``is`` pattern stored ``user | is | caught up with the class``. Drives
THE deployed ``FactExtractor.extract_facts`` (dependency-parse clause check,
no phrase list). Every case is asserted clean and wrapped/indented.
"""
import asyncio

import pytest

from memory.fact_extractor import FactExtractor

_FE = FactExtractor(use_rebel=False)


def _facts(text):
    return [n.content for n in asyncio.run(_FE.extract_facts(text, ""))]


def _forms(text):
    wrapped = "\n".join("   " + w for w in text.replace(" then ", "\n then ").split("\n"))
    return [text, wrapped]


@pytest.mark.parametrize("text", [
    "If I finish assignment 7 then I am caught up with the class.",
    "Once I finish assignment 7 I am caught up with the class.",
    "If I finish the lab tonight I'll be caught up with the class.",
])
def test_conditional_consequent_is_not_a_fact(text):
    for form in _forms(text):
        assert not [f for f in _facts(form) if "caught up" in f], form


@pytest.mark.parametrize("text", [
    "I would be happier if it were warmer.",
])
def test_irrealis_is_not_a_fact(text):
    for form in _forms(text):
        assert not [f for f in _facts(form) if "happier" in f], form


def test_plain_declarative_is_still_extracted():
    for form in _forms("I am caught up with the class."):
        assert any("caught up with the class" in f for f in _facts(form)), form


# Referee amendment: once / whenever are TEMPORAL subordinators. A past-tense
# clause reports facts; a present/base/future clause is prospective.
@pytest.mark.parametrize("text,needle", [
    ("Once I finished the move I live in Tampa.", "tampa"),
    ("Once I moved I lived in Tampa.", "tampa"),
])
def test_past_tense_once_clause_still_yields_the_fact(text, needle):
    for form in _forms(text):
        assert any(needle in f for f in _facts(form)), form


@pytest.mark.parametrize("text", [
    "Once I finish the lab I am caught up with the class.",
    "Once I finish the lab and hand it in I am caught up with the class.",
])
def test_prospective_once_clause_is_not_a_fact(text):
    for form in _forms(text):
        assert not [f for f in _facts(form) if "caught up" in f], form
