"""Household-chore relations are ephemeral and never file under CAREER (2026-09-02).

Live profile: laundry_done / laundry_status / cleaned_up / cleans_up were
DURABLE facts filed under `career` — the embedding categorizer guessed CAREER
and data/category_cache.json made the guess permanent.
"""

import pytest

import memory.user_profile_schema as ups
from memory.relation_classifier import ephemeral_ttl_hours, is_ephemeral_relation
from memory.user_profile_schema import ProfileCategory, categorize_relation


@pytest.mark.parametrize("relation", [
    "laundry_done", "laundry_status", "laundry_time_left", "cleaning_up",
    "cleaned_up", "cleans_up", "dishes_done", "grocery_run", "vacuumed_apartment",
    "homework_done",
])
def test_chore_and_completion_relations_are_ephemeral(relation):
    assert is_ephemeral_relation(relation)
    assert ephemeral_ttl_hours(relation) == pytest.approx(ephemeral_ttl_hours("current_activity"))


@pytest.mark.parametrize("relation", [
    "medication_name", "lives_in", "works_at", "has_cat", "diagnosis", "chronic_condition",
])
def test_durable_relations_unchanged(relation):
    assert not is_ephemeral_relation(relation)


class _PoisonedCache:
    """Stand-in for the persisted category cache holding the live bad entry."""

    def __init__(self):
        self.d = {"laundry_done": ProfileCategory.CAREER, "cleaned_up": ProfileCategory.CAREER}

    def get(self, key):
        return self.d.get(key)

    def set(self, key, value):
        self.d[key] = value


@pytest.mark.parametrize("relation", ["laundry_done", "laundry_status", "cleaning_up", "cleaned_up", "cleans_up"])
def test_household_activity_never_career_even_with_poisoned_cache(monkeypatch, relation):
    monkeypatch.setattr(ups, "_category_cache", _PoisonedCache())
    cat = categorize_relation(relation)
    assert cat != ProfileCategory.CAREER
    assert cat == ProfileCategory.PREFERENCES


def test_real_career_relations_still_career():
    assert categorize_relation("job_title") == ProfileCategory.CAREER
    assert categorize_relation("work_schedule") == ProfileCategory.CAREER
