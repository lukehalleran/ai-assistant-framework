"""G1 (2026-10-02): bare generic dwelling objects are junk for living-situation
relations. Driven through THE deployed fact_extractor._is_junk_object."""
import pytest
from memory.fact_extractor import _is_junk_object


@pytest.mark.parametrize("rel", ["home_location", "lives_in", "living_with", "living_arrangement"])
@pytest.mark.parametrize("obj", ["home", "house", "the house", "my place", "a apartment", "here", "there", "My Home", "an apartment."])
def test_bare_dwelling_rejected(rel, obj):
    assert _is_junk_object(obj, rel) is True


@pytest.mark.parametrize("obj", ["mom (her house)", "Springfield, IL", "1726 Howard St",
                                 "with my mom", "a studio downtown", "my mom's house"])
def test_real_residence_accepted(obj):
    assert _is_junk_object(obj, "lives_in") is False


def test_non_living_relation_unaffected():
    assert _is_junk_object("home", "likes") is False
    assert _is_junk_object("house", "works_on") is False
