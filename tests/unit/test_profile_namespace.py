"""Tests for profile fact namespace canonicalization and dedup.

Covers:
- canonicalize_profile_relation() alias resolution + value-aware disambiguation
- categorize_relation() 3-layer resolution (direct, prefix, token overlap)
- add_fact() canonical-relation-aware dedup (cross-relation merge)
- Ephemeral vs snapshot TTL handling
- False-positive dedup prevention (same value, different semantics)
"""
import pytest
from unittest.mock import patch
from datetime import datetime, timedelta

from memory.user_profile_schema import (
    canonicalize_profile_relation,
    categorize_relation,
    ProfileCategory,
    SAFE_RELATION_ALIASES,
    EPHEMERAL_RELATIONS,
    SNAPSHOT_RELATIONS,
)


# ---------------------------------------------------------------------------
# canonicalize_profile_relation()
# ---------------------------------------------------------------------------

class TestCanonicalizeRelation:

    def test_safe_alias_resolves(self):
        assert canonicalize_profile_relation("pet") == "pet_name"
        assert canonicalize_profile_relation("has_pet") == "pet_name"

    def test_homework_score_aliases(self):
        assert canonicalize_profile_relation("homework_score") == "homework_grade"
        assert canonicalize_profile_relation("last_homework_score") == "homework_grade"
        assert canonicalize_profile_relation("grade") == "homework_grade"

    def test_job_value_aware_quit(self):
        """job=quit → job_status, job=bartender → occupation."""
        assert canonicalize_profile_relation("job", "quit") == "job_status"
        assert canonicalize_profile_relation("job", "left") == "job_status"
        assert canonicalize_profile_relation("job", "unemployed") == "job_status"

    def test_job_value_aware_occupation(self):
        assert canonicalize_profile_relation("job", "bartender") == "occupation"
        assert canonicalize_profile_relation("job", "software engineer") == "occupation"

    def test_quit_job_always_job_status(self):
        assert canonicalize_profile_relation("quit_job") == "job_status"
        assert canonicalize_profile_relation("quit_job", "yes") == "job_status"
        assert canonicalize_profile_relation("quit_job", "no") == "job_status"

    def test_unknown_relation_passthrough(self):
        assert canonicalize_profile_relation("some_custom_thing") == "some_custom_thing"

    def test_plural_singular_merge(self):
        assert canonicalize_profile_relation("current_feelings") == "current_feeling"

    def test_future_plan_goal_merge(self):
        assert canonicalize_profile_relation("future_plan") == "future_goal"


# ---------------------------------------------------------------------------
# categorize_relation() — 3-layer resolution
# ---------------------------------------------------------------------------

class TestCategorizeRelation:

    def test_direct_lookup(self):
        """Layer 1: exact match in RELATION_CATEGORY_MAP."""
        assert categorize_relation("occupation") == ProfileCategory.CAREER
        assert categorize_relation("pet_name") == ProfileCategory.HOBBIES
        assert categorize_relation("squat_max") == ProfileCategory.FITNESS

    def test_prefix_lookup(self):
        """Layer 2: first token prefix match."""
        assert categorize_relation("sleep_apnea_test_result") == ProfileCategory.HEALTH
        assert categorize_relation("homework_due_date") == ProfileCategory.STUDY
        assert categorize_relation("workout_focus") == ProfileCategory.FITNESS
        assert categorize_relation("project_depth") == ProfileCategory.PROJECTS

    def test_token_overlap_scoring(self):
        """Layer 3: multi-token overlap against category keyword sets."""
        # "medication_dose" — tokens {medication, dose}, medication hits HEALTH
        assert categorize_relation("medication_dose") == ProfileCategory.HEALTH
        # "financial_status" — tokens {financial, status}, financial hits FINANCE
        assert categorize_relation("financial_status") == ProfileCategory.FINANCE

    def test_resume_not_misclassified(self):
        """resume_update_deadline should NOT go to relationships (was matching 'date')."""
        assert categorize_relation("resume_update_deadline") == ProfileCategory.CAREER
        assert categorize_relation("resume_status") == ProfileCategory.CAREER

    def test_date_planned_is_relationships(self):
        """date_planned should be relationships (prefix 'date')."""
        assert categorize_relation("date_planned") == ProfileCategory.RELATIONSHIPS

    def test_default_fallback(self):
        """Truly unknown relations fall to PREFERENCES."""
        assert categorize_relation("xyzzy_foobar_baz") == ProfileCategory.PREFERENCES

    def test_boss_name_is_career(self):
        assert categorize_relation("boss_name") == ProfileCategory.CAREER

    def test_brother_name_is_relationships(self):
        assert categorize_relation("brother_name") == ProfileCategory.RELATIONSHIPS


# ---------------------------------------------------------------------------
# add_fact() — canonical dedup
# ---------------------------------------------------------------------------

class TestAddFactCanonicalDedup:

    def _make_profile(self):
        from memory.user_profile import UserProfile
        profile = UserProfile.__new__(UserProfile)
        profile.profile = {
            "name": "Test",
            "updated_at": "",
            "quick_profile": {},
            "categories": {cat.value: [] for cat in ProfileCategory},
            "raw_log": [],
        }
        profile._embedding_model = None
        profile._fact_embeddings = {}
        import threading
        profile._lock = threading.Lock()
        return profile

    def test_pet_alias_dedup(self):
        """Adding 'pet=flapjack' after 'pet_name=flapjack' should confirm, not duplicate."""
        profile = self._make_profile()
        profile.add_fact("pet_name", "flapjack", confidence=0.9)
        profile.add_fact("pet", "flapjack", confidence=0.8)  # alias

        hobbies = [f for f in profile.profile["categories"]["hobbies"]
                   if f.get("is_current", True)]
        # Should be 1 fact (confirmed), not 2
        current_pets = [f for f in hobbies if "pet" in f.get("relation", "")]
        assert len(current_pets) == 1
        assert current_pets[0]["relation"] == "pet_name"

    def test_job_quit_supersedes_quit_job_no(self):
        """job=quit and quit_job=no should both canonicalize to job_status."""
        profile = self._make_profile()
        profile.add_fact("quit_job", "no", confidence=0.8)
        profile.add_fact("job", "quit", confidence=1.0)

        career = profile.profile["categories"]["career"]
        current_job = [f for f in career
                       if f.get("is_current", True) and f.get("relation") == "job_status"]
        assert len(current_job) == 1
        assert current_job[0]["value"] == "quit"

    def test_same_value_different_semantics_preserved(self):
        """homework_grade=90 and preferred_homework_score=90 should NOT merge."""
        profile = self._make_profile()
        profile.add_fact("homework_grade", "90", confidence=0.9)
        profile.add_fact("preferred_homework_score", "90", confidence=0.8)

        study = [f for f in profile.profile["categories"]["study"]
                 if f.get("is_current", True)]
        # These have different canonical relations, both should exist
        rels = {f["relation"] for f in study}
        assert "homework_grade" in rels
        assert "preferred_homework_score" in rels


# ---------------------------------------------------------------------------
# Ephemeral vs Snapshot
# ---------------------------------------------------------------------------

class TestEphemeralSnapshot:

    def test_ephemeral_relations_defined(self):
        """Known ephemeral relations should be in the set."""
        assert "current_day" in EPHEMERAL_RELATIONS
        assert "current_time" in EPHEMERAL_RELATIONS
        assert "current_drink" in EPHEMERAL_RELATIONS
        assert "current_mood" in EPHEMERAL_RELATIONS

    def test_snapshot_relations_not_ephemeral(self):
        """Snapshot relations must NOT be in ephemeral set."""
        for rel in SNAPSHOT_RELATIONS:
            assert rel not in EPHEMERAL_RELATIONS, f"{rel} is in both SNAPSHOT and EPHEMERAL"

    def test_current_weight_is_snapshot(self):
        assert "current_weight" in SNAPSHOT_RELATIONS

    def test_current_bench_is_snapshot(self):
        assert "current_bench" in SNAPSHOT_RELATIONS

    def test_current_day_is_ephemeral(self):
        assert "current_day" in EPHEMERAL_RELATIONS


# ---------------------------------------------------------------------------
# Alias completeness
# ---------------------------------------------------------------------------

class TestAliasCompleteness:

    def test_all_aliases_have_canonical_in_map_or_self(self):
        """Every canonical target should either be itself or map to something."""
        for variant, canonical in SAFE_RELATION_ALIASES.items():
            # canonical should not itself be an alias to something else
            # (no chains: a→b→c)
            if canonical in SAFE_RELATION_ALIASES:
                assert SAFE_RELATION_ALIASES[canonical] == canonical, \
                    f"Alias chain detected: {variant} → {canonical} → {SAFE_RELATION_ALIASES[canonical]}"

    def test_job_is_not_in_safe_aliases(self):
        """'job' needs value-aware handling, should NOT be in SAFE_RELATION_ALIASES."""
        assert "job" not in SAFE_RELATION_ALIASES
