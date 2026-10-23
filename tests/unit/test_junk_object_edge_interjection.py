"""An edge interjection no longer hides a junk object.

2026-10-15, class: BC-75. "lol" alone was junk but "yesterday lol" was not:
the interjection category (graph_utils.INTERJECTION_TOKENS) was consulted
only when EVERY token was an interjection. Leading/trailing interjection
tokens are now stripped and the remainder judged by the EXISTING rules.
Drives the deployed _is_junk_object / is_junk_entity.

Closed 2026-10-22 (class: BC-55): "mom name" is a value-less identity-attribute
slot label; `_is_junk_object` now drops it (see
test_fact_extractor_valueless_slot_label.py). The graph-node check
(`is_junk_entity`) joined on 2026-10-23 (SMALL3).
"""
import pytest

from memory.fact_extractor import _is_junk_object
from memory.graph_utils import is_junk_entity, strip_edge_interjections


@pytest.mark.parametrize("text", [
    "yesterday lol", "ugh yesterday", "last week lmaooo",
])
def test_junk_remainder_makes_whole_junk(text):
    assert _is_junk_object(text, "has_name", "user") is True
    assert is_junk_entity(text) is True


def test_with_food_haha_object_junk():
    assert _is_junk_object("with food haha", "ate", "user") is True


@pytest.mark.parametrize("text", [
    "the gym lmao", "fair lol", "sweet lmao",
    "cool beans lol", "my dentist omg", "Chicago lol", "Biscuit ugh",
    "lol cat cafe", "Lola",
])
def test_non_junk_remainder_keeps_verdict(text):
    assert _is_junk_object(text, "has_name", "user") is False
    assert is_junk_entity(text) is False


@pytest.mark.parametrize("text", ["mom name lol", "lol mom name", "mom name"])
def test_slot_label_junk_for_facts_and_entity_check(text):
    assert _is_junk_object(text, "has_name", "user") is True
    assert is_junk_entity(text) is True  # graph chokepoint joined 2026-10-23


@pytest.mark.parametrize("text", ["fair", "sweet", "cool beans"])
def test_plain_values_unchanged(text):
    assert _is_junk_object(text, "likes", "user") is False
    assert is_junk_entity(text) is False


@pytest.mark.parametrize("text", ["lol", "ugh", "lmao", "ugh lol"])
def test_all_interjection_still_junk(text):
    assert _is_junk_object(text, "likes", "user") is True
    assert is_junk_entity(text) is True


def test_strip_never_empties_or_rewrites_inner():
    assert strip_edge_interjections("lol") == "lol"
    assert strip_edge_interjections("mom name lol") == "mom name"
    assert strip_edge_interjections("a lol b") == "a lol b"
