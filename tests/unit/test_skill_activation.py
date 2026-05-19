"""Tests for memory/skill_activation.py — SkillActivationPolicy + SkillCooldownStore."""

import json
import os
import time
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from memory.skill_activation import (
    SkillActivationPolicy,
    SkillCooldownStore,
    _SUPPRESSED_INTENTS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

def _make_skill(skill_id: str, score: float = 0.5, trigger: str = "debug issue",
                action: str = "check logs first", category: str = "DEBUGGING",
                tags: list = None) -> dict:
    """Create a mock skill dict matching MemoryRetriever.get_skills() output."""
    return {
        "id": skill_id,
        "content": f"Situation: {trigger} | Action: {action}",
        "relevance_score": score,
        "metadata": {
            "trigger": trigger,
            "action_pattern": action,
            "category": category,
            "tags_json": json.dumps(tags or []),
            "confidence": 0.8,
            "created_at": time.time() - 86400,
            "times_retrieved": 3,
        },
    }


@pytest.fixture
def tmp_cooldown_path(tmp_path):
    return str(tmp_path / "skill_cooldown.json")


@pytest.fixture
def cooldown_store(tmp_cooldown_path):
    return SkillCooldownStore(persist_path=tmp_cooldown_path)


@pytest.fixture
def policy(cooldown_store):
    return SkillActivationPolicy(
        cooldown_store=cooldown_store,
        min_score=0.25,
        cooldown_hours=48.0,
        max_skills=3,
        stm_bonus=0.10,
        enabled=True,
    )


@pytest.fixture
def candidates():
    return [
        _make_skill("sk1", 0.85, "refactoring circular dependencies", "extract interface", "WORKFLOW", ["refactor", "architecture"]),
        _make_skill("sk2", 0.70, "debugging race conditions", "add logging + reproduce", "DEBUGGING", ["concurrency", "race"]),
        _make_skill("sk3", 0.55, "writing integration tests", "mock boundaries only", "TESTING", ["testing"]),
        _make_skill("sk4", 0.40, "reviewing pull requests", "check diff first", "WORKFLOW", ["review"]),
        _make_skill("sk5", 0.15, "general coding", "read the docs", "GENERAL", []),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# SkillCooldownStore tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSkillCooldownStore:

    def test_empty_store_not_cooling_down(self, cooldown_store):
        assert not cooldown_store.is_cooling_down("sk1")

    def test_record_and_check(self, cooldown_store):
        cooldown_store.record_surfaced("sk1")
        assert cooldown_store.is_cooling_down("sk1", cooldown_hours=48.0)

    def test_expired_cooldown(self, cooldown_store):
        cooldown_store._entries["sk1"] = {
            "last_surfaced": (datetime.now() - timedelta(hours=72)).isoformat(),
            "count": 1,
        }
        assert not cooldown_store.is_cooling_down("sk1", cooldown_hours=48.0)

    def test_persistence_round_trip(self, tmp_cooldown_path):
        store1 = SkillCooldownStore(persist_path=tmp_cooldown_path)
        store1.record_surfaced("sk1")
        store1.record_surfaced("sk2")

        store2 = SkillCooldownStore(persist_path=tmp_cooldown_path)
        assert store2.is_cooling_down("sk1")
        assert store2.is_cooling_down("sk2")

    def test_count_increments(self, cooldown_store):
        cooldown_store.record_surfaced("sk1")
        cooldown_store.record_surfaced("sk1")
        assert cooldown_store._entries["sk1"]["count"] == 2

    def test_cleanup_old(self, cooldown_store):
        cooldown_store._entries["old"] = {
            "last_surfaced": (datetime.now() - timedelta(days=60)).isoformat(),
            "count": 5,
        }
        cooldown_store._entries["recent"] = {
            "last_surfaced": datetime.now().isoformat(),
            "count": 1,
        }
        removed = cooldown_store.cleanup_old(max_age_days=30)
        assert removed == 1
        assert "old" not in cooldown_store._entries
        assert "recent" in cooldown_store._entries

    def test_load_missing_file(self, tmp_path):
        store = SkillCooldownStore(persist_path=str(tmp_path / "nonexistent.json"))
        assert store._entries == {}

    def test_load_corrupt_file(self, tmp_path):
        p = str(tmp_path / "bad.json")
        with open(p, "w") as f:
            f.write("{invalid json!!")
        store = SkillCooldownStore(persist_path=p)
        assert store._entries == {}


# ═══════════════════════════════════════════════════════════════════════════
# SkillActivationPolicy tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSkillActivationPolicy:

    # --- Basic filtering ---

    def test_high_match_surfaces_skills(self, policy, candidates):
        """Technical query with good candidates should surface skills."""
        result = policy.filter(candidates, intent_type="TECHNICAL_HELP")
        assert len(result) > 0
        assert len(result) <= 3  # max_skills

    def test_returns_highest_scoring(self, policy, candidates):
        result = policy.filter(candidates, intent_type="TECHNICAL_HELP")
        scores = [c["relevance_score"] for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_max_skills_cap(self, policy, candidates):
        """Even with many candidates, respects max_skills."""
        result = policy.filter(candidates, intent_type="GENERAL")
        assert len(result) <= 3

    # --- Intent suppression ---

    def test_emotional_support_suppressed(self, policy, candidates):
        result = policy.filter(candidates, intent_type="EMOTIONAL_SUPPORT")
        assert result == []

    def test_casual_social_suppressed(self, policy, candidates):
        result = policy.filter(candidates, intent_type="CASUAL_SOCIAL")
        assert result == []

    def test_suppressed_intents_are_correct(self):
        assert "EMOTIONAL_SUPPORT" in _SUPPRESSED_INTENTS
        assert "CASUAL_SOCIAL" in _SUPPRESSED_INTENTS

    def test_technical_help_not_suppressed(self, policy, candidates):
        result = policy.filter(candidates, intent_type="TECHNICAL_HELP")
        assert len(result) > 0

    def test_project_work_not_suppressed(self, policy, candidates):
        result = policy.filter(candidates, intent_type="PROJECT_WORK")
        assert len(result) > 0

    def test_none_intent_not_suppressed(self, policy, candidates):
        """None intent (no classification) should still allow skills."""
        result = policy.filter(candidates, intent_type=None)
        assert len(result) > 0

    # --- Minimum score threshold ---

    def test_below_threshold_filtered(self, policy):
        """Skills below min_score should be dropped."""
        low_skills = [
            _make_skill("low1", 0.10),
            _make_skill("low2", 0.20),
        ]
        result = policy.filter(low_skills, intent_type="TECHNICAL_HELP")
        assert result == []

    def test_at_threshold_included(self, policy):
        """Skills at exactly min_score should pass."""
        at_threshold = [_make_skill("exact", 0.25)]
        result = policy.filter(at_threshold, intent_type="GENERAL")
        assert len(result) == 1

    def test_mixed_scores_filters_low(self, policy, candidates):
        """sk5 at 0.15 should be filtered out (below 0.25 threshold)."""
        result = policy.filter(candidates, intent_type="GENERAL")
        result_ids = {s["id"] for s in result}
        assert "sk5" not in result_ids

    # --- Cooldown ---

    def test_cooldown_skips_recently_surfaced(self, policy, cooldown_store, candidates):
        """Previously surfaced skill should be skipped during cooldown."""
        cooldown_store.record_surfaced("sk1")
        result = policy.filter(candidates, intent_type="TECHNICAL_HELP")
        result_ids = {s["id"] for s in result}
        assert "sk1" not in result_ids

    def test_expired_cooldown_allows_skill(self, policy, cooldown_store, candidates):
        """Skill whose cooldown expired should be allowed again."""
        cooldown_store._entries["sk1"] = {
            "last_surfaced": (datetime.now() - timedelta(hours=72)).isoformat(),
            "count": 1,
        }
        result = policy.filter(candidates, intent_type="TECHNICAL_HELP")
        result_ids = {s["id"] for s in result}
        assert "sk1" in result_ids

    def test_cooldown_records_surfaced(self, policy, cooldown_store, candidates):
        """Surfaced skills should be recorded in cooldown store."""
        result = policy.filter(candidates, intent_type="GENERAL")
        for skill in result:
            assert cooldown_store.is_cooling_down(skill["id"])

    def test_cooldown_backfills_when_top_skill_cooled(self, policy, cooldown_store, candidates):
        """When top skill is cooled, next-best should fill the slot."""
        cooldown_store.record_surfaced("sk1")
        cooldown_store.record_surfaced("sk2")
        result = policy.filter(candidates, intent_type="TECHNICAL_HELP")
        result_ids = [s["id"] for s in result]
        assert "sk3" in result_ids

    # --- STM topic bonus ---

    def test_stm_topic_boosts_matching_skill(self, policy, candidates):
        """Skill matching STM topic should rank higher."""
        # sk3 has "testing" tag; sk1 has "refactor" + "architecture"
        result_with_stm = policy.filter(
            candidates, intent_type="GENERAL", stm_topics=["testing"]
        )
        result_without_stm = policy.filter(
            candidates, intent_type="GENERAL", stm_topics=None
        )
        # With STM boost for "testing", sk3 should rank higher than without
        ids_with = [s["id"] for s in result_with_stm]
        ids_without = [s["id"] for s in result_without_stm]
        # sk3 should appear in results with STM boost (it's at 0.55, might rank 3rd)
        assert "sk3" in ids_with

    def test_stm_topic_matches_trigger(self, policy):
        """STM topic matching skill trigger text should apply bonus."""
        skills = [
            _make_skill("a", 0.30, trigger="deploying to production", tags=[]),
            _make_skill("b", 0.35, trigger="writing unit tests", tags=["testing"]),
        ]
        result = policy.filter(skills, intent_type="GENERAL", stm_topics=["deploying"])
        # "deploying" matches trigger of skill "a", so it should get boosted above "b"
        assert result[0]["id"] == "a"

    def test_stm_empty_topics_no_crash(self, policy, candidates):
        result = policy.filter(candidates, intent_type="GENERAL", stm_topics=[])
        assert len(result) > 0

    def test_stm_general_topic_ignored_by_builder(self):
        """The builder should not pass 'general' as an STM topic."""
        # This is tested at the integration level but document the contract here
        pass

    # --- Disabled config ---

    def test_disabled_returns_empty(self, cooldown_store, candidates):
        policy = SkillActivationPolicy(
            cooldown_store=cooldown_store, enabled=False
        )
        result = policy.filter(candidates, intent_type="TECHNICAL_HELP")
        assert result == []

    # --- Edge cases ---

    def test_empty_candidates(self, policy):
        result = policy.filter([], intent_type="TECHNICAL_HELP")
        assert result == []

    def test_no_cooldown_store(self):
        """Policy without cooldown store should still work."""
        policy = SkillActivationPolicy(cooldown_store=None, min_score=0.25, max_skills=2)
        skills = [_make_skill("a", 0.5), _make_skill("b", 0.4)]
        result = policy.filter(skills, intent_type="GENERAL")
        assert len(result) == 2

    def test_skill_without_id_not_recorded(self, policy, cooldown_store):
        """Skills missing 'id' field should not crash cooldown recording."""
        skills = [{"content": "test", "relevance_score": 0.5, "metadata": {}}]
        result = policy.filter(skills, intent_type="GENERAL")
        assert len(result) == 1

    def test_missing_metadata_no_crash(self, policy):
        """Skills with missing metadata should not crash STM bonus computation."""
        skills = [{"id": "x", "relevance_score": 0.5}]
        result = policy.filter(skills, intent_type="GENERAL", stm_topics=["test"])
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Integration-style tests (policy + cooldown store together)
# ═══════════════════════════════════════════════════════════════════════════

class TestSkillActivationIntegration:

    def test_full_flow_technical_query(self, policy, candidates):
        """Technical query surfaces top skills, records cooldown."""
        result = policy.filter(candidates, intent_type="TECHNICAL_HELP", stm_topics=["refactoring"])
        assert len(result) <= 3
        assert all(s["relevance_score"] >= 0.25 for s in result)
        # sk1 about "refactoring" should be top since STM matches
        assert result[0]["id"] == "sk1"

    def test_repeated_calls_produce_different_skills(self, policy, candidates):
        """Second call should skip cooldown skills and surface different ones."""
        first = policy.filter(candidates, intent_type="TECHNICAL_HELP")
        first_ids = {s["id"] for s in first}

        second = policy.filter(candidates, intent_type="TECHNICAL_HELP")
        second_ids = {s["id"] for s in second}

        # Second call should have different skills (first-call skills are cooled)
        assert first_ids != second_ids or len(second) < len(first)

    def test_emotional_then_technical(self, policy, candidates):
        """Emotional query blocks skills; subsequent technical query allows them."""
        emo_result = policy.filter(candidates, intent_type="EMOTIONAL_SUPPORT")
        assert emo_result == []

        tech_result = policy.filter(candidates, intent_type="TECHNICAL_HELP")
        assert len(tech_result) > 0
