"""B4 item 2: concerned_about="tired of everything" is a transient mood (class: BC-46).

Original batch label: oct12_b4 (renamed 2026-10-16; old path in tests/RENAMES.tsv)."""
import pytest

from memory.relation_classifier import ephemeral_ttl_hours
from memory.user_profile_schema import ProfileCategory, categorize_relation


@pytest.mark.parametrize("rel", ["concerned_about", "worried_about", "stressed_about", "feels_overwhelmed"])
def test_emotional_state_relations_are_ephemeral_and_not_hobbies(rel):
    assert ephemeral_ttl_hours(rel) is not None
    assert categorize_relation(rel) is ProfileCategory.HEALTH


def test_durable_relations_unchanged():
    assert ephemeral_ttl_hours("employer") is None
    assert ephemeral_ttl_hours("anxiety_trigger") is None
    assert ephemeral_ttl_hours("chronic_condition") is None


def test_poisoned_cache_cannot_refile_mood_under_hobbies(monkeypatch):
    from memory import user_profile_schema as s

    class _Poisoned:
        def get(self, rel):
            return ProfileCategory.HOBBIES

        def set(self, rel, cat):
            pass

    monkeypatch.setattr(s, "_category_cache", _Poisoned())
    assert categorize_relation("concerned_about") is ProfileCategory.HEALTH
