"""Chat interjections/laughter never become graph nodes or fact objects.

class: BC-75. Drives the deployed is_junk_entity / _is_junk_object / _clean_triple.
"""
import pytest

from memory.fact_extractor import _clean_triple, _is_junk_object
from memory.graph_utils import is_interjection, is_junk_entity

JUNK = ["lol", "LOL", "lmaooo", "hahaha", "ugh lol", "ughhh", "hahahaha", "lolll"]
KEEP = ["Lola", "Hal", "lol cat cafe", "fair", "sweet", "Ohio"]


@pytest.mark.parametrize("v", JUNK)
def test_entity_junk(v):
    assert is_junk_entity(v)


@pytest.mark.parametrize("v", KEEP)
def test_entity_kept(v):
    assert not is_junk_entity(v)


@pytest.mark.parametrize("v", JUNK)
def test_object_junk(v):
    assert _is_junk_object(v, "mom_name", "user")


@pytest.mark.parametrize("v", KEEP)
def test_object_kept(v):
    assert not _is_junk_object(v, "likes", "user")


def test_is_interjection_all_tokens_rule():
    assert is_interjection("ugh lol")
    assert not is_interjection("lol cat cafe")
    assert not is_interjection("")


def test_fact_level_drop():
    assert _clean_triple("user", "mom_name", "lol") is None
    assert _clean_triple("user", "mom_name", "Lola") is not None


# Referee addition (2026-10-14): casing marks an abbreviation or a name. The
# object check used to lowercase first, so "went_to | ER" and "plays | WoW"
# would have been dropped as interjections.
@pytest.mark.parametrize("v", ["ER", "OH", "UM", "WoW", "FR"])
def test_cased_abbreviations_and_names_kept(v):
    from memory.fact_extractor import _is_junk_object
    from memory.graph_utils import is_interjection
    assert not is_interjection(v)
    assert not _is_junk_object(v, "went_to")


@pytest.mark.parametrize("v", ["LOL", "OMG", "Ugh", "Lmaooo"])
def test_cased_laughter_still_junk(v):
    from memory.graph_utils import is_interjection
    assert is_interjection(v)
