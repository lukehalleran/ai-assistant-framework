"""Graph-entity chokepoint junks value-less PERSON slot labels only.

Incident 2026-10-23, class: BC-75 (nearest; BC-55 sibling). #97 dropped
"mom name" as a fact OBJECT only; is_junk_entity still let a graph node of
that name through. The predicate now lives in memory.graph_utils and
is_junk_entity calls it on the lowercased name (graph labels are often
title-cased). Referee revision: the slot's owner must be a person-role noun
(graph_utils.PERSON_ROLE_NOUNS, shared with entity_resolver); without it
concept names ("Domain name", "Gettysburg Address") were junked on both paths
(the fact-object path already had that false positive on base since #97).
Accepted miss: "Rowan Email" (a proper-noun owner, no person-role noun).
"""
import pytest

import memory.entity_resolver as er
import memory.fact_extractor as fe
import memory.graph_utils as gu
from memory.graph_utils import is_junk_entity


def _wrap(text):
    return "  " + text.replace(" ", "\n  ") if " " in text else text


def _forms(text):
    return [text, _wrap(text)]


JUNK = ["mom name", "Mom Name", "dad number", "my sister's address", "boss email"]
KEPT_CONCEPTS = [
    "Gettysburg Address", "Domain name", "domain name", "Email address",
    "Pen name", "Stage Name", "Brand Name", "Code Name", "Account Number",
    "Street Address", "IP address", "Rowan Email",
]
KEPT_OTHER = [
    "Tamsin", "mom", "Rowan's birthday party", "phone number 555-0100",
    "Rowan's birthday", "new phone",
]


@pytest.mark.parametrize("name", JUNK)
@pytest.mark.parametrize("form", [0, 1])
def test_person_slot_label_is_junk_entity(name, form):
    assert is_junk_entity(_forms(name)[form]) is True


@pytest.mark.parametrize("name", [n for n in JUNK if n != "Mom Name"])
def test_person_slot_label_is_junk_fact_object(name):
    # "Mom Name": the fact-object predicate keeps its capitalised-token =
    # value rule (unchanged since #97); only the entity path lowercases.
    assert fe._is_junk_object(name, "has", "user") is True


@pytest.mark.parametrize("name", KEPT_CONCEPTS + KEPT_OTHER)
@pytest.mark.parametrize("form", [0, 1])
def test_concept_and_value_names_not_junk_entity(name, form):
    assert is_junk_entity(_forms(name)[form]) is False


@pytest.mark.parametrize("name", KEPT_CONCEPTS)
@pytest.mark.parametrize("form", [0, 1])
def test_concept_names_not_junk_fact_object(name, form):
    assert fe._is_junk_object(_forms(name)[form], "has", "user") is False
    assert gu.is_valueless_slot_label(name) is False


def test_predicate_single_definition_reexported():
    assert fe.is_valueless_slot_label is gu.is_valueless_slot_label


def test_person_role_category_single_definition():
    for noun in gu.PERSON_ROLE_NOUNS:
        assert er.extract_possessive_aliases(f"my {noun} said hi") == [(f"my {noun}", "person")]
    assert er.extract_possessive_aliases("my cat Whiskers") == [("my cat", "pet")]


def test_fact_object_path_still_junks_through_chokepoint():
    assert fe._is_junk_object("mom name lol", "has_name", "user") is True


def test_entity_check_uses_fact_extractor_nlp_hook(monkeypatch):
    assert is_junk_entity("dad number") is True
    monkeypatch.setattr(fe, "_get_nlp", lambda: None)
    assert is_junk_entity("dad number") is False
