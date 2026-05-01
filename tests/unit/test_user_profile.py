import pytest
import tempfile
import os
import time
import json
from datetime import datetime, timedelta
from memory.user_profile import UserProfile
from memory.user_profile_schema import ProfileCategory

class TestUserProfile:
    @pytest.fixture
    def temp_profile(self):
        """Create a temporary profile for testing."""
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_create_and_save(self, temp_profile):
        profile = UserProfile(temp_profile)
        profile.add_fact("name", "Luke", confidence=0.9)
        profile.save()

        # Reload and verify
        profile2 = UserProfile(temp_profile)
        assert profile2.get_quick_profile().get("name") == "Luke"

    def test_categorization(self, temp_profile):
        profile = UserProfile(temp_profile)
        profile.add_fact("squat_max", "365 lb", confidence=0.8)

        fitness_facts = profile.get_category(ProfileCategory.FITNESS)
        assert len(fitness_facts) == 1
        assert fitness_facts[0]["value"] == "365 lb"

    def test_conflict_resolution(self, temp_profile):
        """Test that same relation + different value → old fact preserved with is_current=False."""
        profile = UserProfile(temp_profile)
        profile.add_fact("bench_max", "285 lb", confidence=0.7)
        profile.add_fact("bench_max", "295 lb", confidence=0.9)  # Different value → supersede

        # Current view: only the latest
        current_facts = profile.get_category(ProfileCategory.FITNESS)
        bench_current = [f for f in current_facts if f["relation"] == "bench_max"]
        assert len(bench_current) == 1
        assert bench_current[0]["value"] == "295 lb"
        assert bench_current[0]["is_current"] is True

        # Historical view: both preserved
        all_facts = profile.get_category(ProfileCategory.FITNESS, include_historical=True)
        bench_all = [f for f in all_facts if f["relation"] == "bench_max"]
        assert len(bench_all) == 2
        old = [f for f in bench_all if f["value"] == "285 lb"][0]
        assert old["is_current"] is False

    def test_batch_add(self, temp_profile):
        profile = UserProfile(temp_profile)
        facts = [
            {"relation": "name", "value": "Luke", "confidence": 0.9},
            {"relation": "location", "value": "St. Charles, IL", "confidence": 0.85},
            {"subject": "user", "relation": "cat_name", "object": "Flapjack"},
        ]
        added = profile.add_facts_batch(facts)
        assert added == 3
        assert profile.get_fact_count() == 3

    def test_markdown_export(self, temp_profile):
        profile = UserProfile(temp_profile)
        profile.add_fact("name", "Luke")
        profile.add_fact("squat_max", "365 lb")

        md = profile.export_markdown()
        assert "# User Profile" in md
        assert "Luke" in md
        assert "365 lb" in md

    def test_context_injection(self, temp_profile):
        profile = UserProfile(temp_profile)
        profile.add_fact("name", "Luke", confidence=0.9)
        profile.add_fact("location", "St. Charles, IL", confidence=0.85)
        profile.add_fact("squat_max", "365 lb", confidence=0.8)

        context = profile.get_context_injection(max_tokens=500)
        assert "Luke" in context
        assert "St. Charles" in context
        assert "squat_max" in context

    def test_quick_profile_normalization(self, temp_profile):
        profile = UserProfile(temp_profile)
        profile.add_fact("lives_in", "Chicago", confidence=0.9)

        quick = profile.get_quick_profile()
        # "lives_in" should be normalized to "location"
        assert quick.get("location") == "Chicago"

    def test_supersedes_tracking(self, temp_profile):
        """Test that superseding a fact references the old fact's fact_id."""
        profile = UserProfile(temp_profile)
        profile.add_fact("age", "32", confidence=0.9)

        # Get the fact_id of the original fact
        all_facts = profile.get_category(ProfileCategory.IDENTITY, include_historical=True)
        age_facts = [f for f in all_facts if f["relation"] == "age"]
        assert len(age_facts) == 1
        original_id = age_facts[0]["fact_id"]

        # Now supersede with different value
        profile.add_fact("age", "33", confidence=0.95)

        all_facts = profile.get_category(ProfileCategory.IDENTITY, include_historical=True)
        age_facts = [f for f in all_facts if f["relation"] == "age"]
        assert len(age_facts) == 2

        current = [f for f in age_facts if f["is_current"]][0]
        assert current["value"] == "33"
        assert current["supersedes"] == original_id

    def test_hybrid_retrieval(self, temp_profile):
        """Test hybrid retrieval: 2/3 semantic + 1/3 recent"""
        profile = UserProfile(temp_profile)

        # Add fitness facts with different timestamps
        profile.add_fact("squat_max", "365 lb", confidence=0.9)
        time.sleep(0.01)
        profile.add_fact("bench_max", "285 lb", confidence=0.9)
        time.sleep(0.01)
        profile.add_fact("deadlift_max", "455 lb", confidence=0.9)
        time.sleep(0.01)
        profile.add_fact("ohp_max", "185 lb", confidence=0.9)

        # Query mentioning "squat" - should prioritize squat_max semantically
        query = "What's my squat progress?"
        relevant = profile.get_relevant_facts(query, ProfileCategory.FITNESS, limit=3)

        assert len(relevant) <= 3
        # Squat should be in results due to semantic match
        squat_found = any(f["relation"] == "squat_max" for f in relevant)
        assert squat_found, "Semantic match 'squat_max' should be in results"

    def test_context_injection_with_query(self, temp_profile):
        """Test context injection uses query for semantic relevance"""
        profile = UserProfile(temp_profile)
        profile.add_fact("name", "Luke", confidence=0.9)
        profile.add_fact("squat_max", "365 lb", confidence=0.9)
        profile.add_fact("bench_max", "285 lb", confidence=0.9)
        profile.add_fact("favorite_beer", "IPA", confidence=0.8)

        # Query about lifting - should prioritize fitness facts
        context = profile.get_context_injection(max_tokens=500, query="How much can I squat?")

        assert "Luke" in context  # Quick profile always included
        assert "squat" in context.lower()  # Should find squat_max semantically


class TestAppendOnlyStorage:
    """Tests for the new append-only fact storage system."""

    @pytest.fixture
    def temp_profile(self):
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_append_only_storage(self, temp_profile):
        """Same relation, different values → both stored, only latest is_current."""
        profile = UserProfile(temp_profile)
        profile.add_fact("sleep_quality", "poor", confidence=0.7)
        profile.add_fact("sleep_quality", "slept fine", confidence=0.7)

        # Current view: only latest (sleep_quality categorizes to HEALTH)
        current = profile.get_category(ProfileCategory.HEALTH)
        sleep_current = [f for f in current if f["relation"] == "sleep_quality"]
        assert len(sleep_current) == 1
        assert sleep_current[0]["value"] == "slept fine"
        assert sleep_current[0]["is_current"] is True

        # Historical view: both preserved
        all_facts = profile.get_category(ProfileCategory.HEALTH, include_historical=True)
        sleep_all = [f for f in all_facts if f["relation"] == "sleep_quality"]
        assert len(sleep_all) == 2
        old = [f for f in sleep_all if f["value"] == "poor"][0]
        assert old["is_current"] is False

    def test_confidence_boost(self, temp_profile):
        """Same (relation, value) repeated → confidence increases by 0.05."""
        profile = UserProfile(temp_profile)
        profile.add_fact("name", "Luke", confidence=0.7)
        profile.add_fact("name", "Luke", confidence=0.7)  # Confirmation
        profile.add_fact("name", "Luke", confidence=0.7)  # Another confirmation

        facts = profile.get_category(ProfileCategory.IDENTITY)
        name_facts = [f for f in facts if f["relation"] == "name"]
        assert len(name_facts) == 1  # No duplicate
        assert name_facts[0]["confidence"] == pytest.approx(0.80, abs=0.001)  # 0.7 + 0.05 + 0.05

    def test_confidence_boost_capped_at_1(self, temp_profile):
        """Confidence boost should not exceed 1.0."""
        profile = UserProfile(temp_profile)
        profile.add_fact("name", "Luke", confidence=0.98)
        profile.add_fact("name", "Luke", confidence=0.98)  # Would be 1.03 without cap

        facts = profile.get_category(ProfileCategory.IDENTITY)
        name_facts = [f for f in facts if f["relation"] == "name"]
        assert name_facts[0]["confidence"] == 1.0

    def test_fact_id_generated(self, temp_profile):
        """All facts should get a UUID fact_id."""
        profile = UserProfile(temp_profile)
        profile.add_fact("name", "Luke", confidence=0.9)

        facts = profile.get_category(ProfileCategory.IDENTITY)
        assert facts[0].get("fact_id") is not None
        assert len(facts[0]["fact_id"]) == 36  # UUID format

    def test_multiple_supersedes(self, temp_profile):
        """Multiple value changes for same relation → chain of supersedes."""
        profile = UserProfile(temp_profile)
        profile.add_fact("body_weight", "180 lb", confidence=0.8)
        profile.add_fact("body_weight", "175 lb", confidence=0.8)
        profile.add_fact("body_weight", "172 lb", confidence=0.8)

        all_facts = profile.get_category(ProfileCategory.FITNESS, include_historical=True)
        weight_facts = [f for f in all_facts if f["relation"] == "body_weight"]
        assert len(weight_facts) == 3

        current = [f for f in weight_facts if f["is_current"]]
        assert len(current) == 1
        assert current[0]["value"] == "172 lb"

        historical = [f for f in weight_facts if not f["is_current"]]
        assert len(historical) == 2


class TestComputedViews:
    """Tests for get_current_view, get_profile_at, get_fact_history."""

    @pytest.fixture
    def temp_profile(self):
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_get_current_view(self, temp_profile):
        """get_current_view returns only is_current=True facts."""
        profile = UserProfile(temp_profile)
        profile.add_fact("squat_max", "365 lb", confidence=0.9)
        profile.add_fact("squat_max", "375 lb", confidence=0.9)  # Supersedes
        profile.add_fact("bench_max", "285 lb", confidence=0.9)

        view = profile.get_current_view()
        fitness = view.get("fitness", [])

        # Should have 2 current facts: squat_max=375 + bench_max=285
        assert len(fitness) == 2
        values = {f["value"] for f in fitness}
        assert "375 lb" in values
        assert "285 lb" in values
        assert "365 lb" not in values  # Historical, not in current view

    def test_get_current_view_single_category(self, temp_profile):
        """get_current_view with specific category filters correctly."""
        profile = UserProfile(temp_profile)
        profile.add_fact("name", "Luke", confidence=0.9)
        profile.add_fact("squat_max", "365 lb", confidence=0.9)

        view = profile.get_current_view(category=ProfileCategory.FITNESS)
        assert "fitness" in view
        assert "identity" not in view

    def test_get_profile_at(self, temp_profile):
        """Point-in-time snapshot returns correct historical state."""
        profile = UserProfile(temp_profile)

        t1 = datetime(2025, 1, 10, 12, 0, 0)
        t2 = datetime(2025, 1, 27, 12, 0, 0)
        t3 = datetime(2025, 2, 7, 12, 0, 0)

        profile.add_fact("sleep_quality", "poor", confidence=0.7, timestamp=t1)
        profile.add_fact("sleep_quality", "slept fine", confidence=0.7, timestamp=t2)
        profile.add_fact("sleep_quality", "okay", confidence=0.7, timestamp=t3)

        # Snapshot at Jan 20 → should see "poor" as the latest
        snapshot = profile.get_profile_at(datetime(2025, 1, 20))
        health_facts = snapshot.get("health", [])
        sleep = [f for f in health_facts if f["relation"] == "sleep_quality"]
        assert len(sleep) == 1
        assert sleep[0]["value"] == "poor"

        # Snapshot at Feb 1 → should see "slept fine"
        snapshot = profile.get_profile_at(datetime(2025, 2, 1))
        health_facts = snapshot.get("health", [])
        sleep = [f for f in health_facts if f["relation"] == "sleep_quality"]
        assert len(sleep) == 1
        assert sleep[0]["value"] == "slept fine"

        # Snapshot at Feb 10 → should see "okay"
        snapshot = profile.get_profile_at(datetime(2025, 2, 10))
        health_facts = snapshot.get("health", [])
        sleep = [f for f in health_facts if f["relation"] == "sleep_quality"]
        assert len(sleep) == 1
        assert sleep[0]["value"] == "okay"

    def test_get_fact_history(self, temp_profile):
        """All values a relation has had, sorted chronologically."""
        profile = UserProfile(temp_profile)

        t1 = datetime(2025, 1, 1)
        t2 = datetime(2025, 2, 1)
        t3 = datetime(2025, 3, 1)

        profile.add_fact("body_weight", "180 lb", confidence=0.8, timestamp=t1)
        profile.add_fact("body_weight", "175 lb", confidence=0.8, timestamp=t2)
        profile.add_fact("body_weight", "172 lb", confidence=0.8, timestamp=t3)

        history = profile.get_fact_history("body_weight")
        assert len(history) == 3
        assert history[0]["value"] == "180 lb"  # Oldest first
        assert history[1]["value"] == "175 lb"
        assert history[2]["value"] == "172 lb"  # Most recent

    def test_get_fact_history_specific_category(self, temp_profile):
        """get_fact_history with category filter."""
        profile = UserProfile(temp_profile)
        profile.add_fact("body_weight", "180 lb", confidence=0.8)

        history = profile.get_fact_history("body_weight", ProfileCategory.FITNESS)
        assert len(history) == 1

        # Wrong category → empty
        history = profile.get_fact_history("body_weight", ProfileCategory.IDENTITY)
        assert len(history) == 0

    def test_get_fact_history_empty(self, temp_profile):
        """Non-existent relation returns empty list."""
        profile = UserProfile(temp_profile)
        history = profile.get_fact_history("nonexistent_relation")
        assert history == []


class TestBackwardCompat:
    """Tests ensuring backward compatibility."""

    @pytest.fixture
    def temp_profile(self):
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_backward_compat_get_category(self, temp_profile):
        """Default get_category returns only current facts (no historical)."""
        profile = UserProfile(temp_profile)
        profile.add_fact("squat_max", "365 lb", confidence=0.9)
        profile.add_fact("squat_max", "375 lb", confidence=0.9)

        # Default: only current
        facts = profile.get_category(ProfileCategory.FITNESS)
        assert len(facts) == 1
        assert facts[0]["value"] == "375 lb"

        # Explicit historical: both
        facts = profile.get_category(ProfileCategory.FITNESS, include_historical=True)
        assert len(facts) == 2

    def test_context_injection_unchanged_non_temporal(self, temp_profile):
        """Non-temporal queries produce same format as before (no timeline)."""
        profile = UserProfile(temp_profile)
        profile.add_fact("name", "Luke", confidence=0.9)
        profile.add_fact("squat_max", "365 lb", confidence=0.9)
        profile.add_fact("squat_max", "375 lb", confidence=0.9)

        context = profile.get_context_injection(max_tokens=500, query="How much can I squat?")
        assert "timeline" not in context.lower()

    def test_export_markdown_shows_current_only(self, temp_profile):
        """Markdown export should only show current facts."""
        profile = UserProfile(temp_profile)
        profile.add_fact("squat_max", "365 lb", confidence=0.9)
        profile.add_fact("squat_max", "375 lb", confidence=0.9)

        md = profile.export_markdown()
        assert "375 lb" in md
        # Historical fact should NOT appear in category section
        # (it may appear in raw log though)
        lines = md.split("\n")
        fitness_section = False
        raw_section = False
        for line in lines:
            if "Fitness" in line:
                fitness_section = True
                raw_section = False
            if "Recent Fact Log" in line:
                fitness_section = False
                raw_section = True
            if fitness_section and "365 lb" in line:
                pytest.fail("Historical fact '365 lb' should not appear in Fitness section")


class TestMigration:
    """Tests for schema migration 1.0 → 2.0."""

    @pytest.fixture
    def temp_profile(self):
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_migration_1_to_2(self, temp_profile):
        """v1.0 profile → migration adds fact_id + is_current correctly."""
        # Write a v1.0-style profile directly
        v1_profile = {
            "user_id": "default",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-15T00:00:00",
            "version": "1.0",
            "quick_profile": {"name": "Luke"},
            "identity": {"name": "Luke", "pronouns": "he/him"},
            "preferences": {"style": "balanced", "check_distress": True, "brief_responses": False},
            "categories": {
                "identity": [
                    {"relation": "name", "value": "Luke", "category": "identity",
                     "confidence": 0.9, "source_excerpt": "", "timestamp": "2025-01-01T00:00:00",
                     "supersedes": None},
                ],
                "fitness": [
                    {"relation": "squat_max", "value": "355 lb", "category": "fitness",
                     "confidence": 0.8, "source_excerpt": "", "timestamp": "2025-01-01T00:00:00",
                     "supersedes": None},
                    {"relation": "squat_max", "value": "365 lb", "category": "fitness",
                     "confidence": 0.9, "source_excerpt": "", "timestamp": "2025-01-10T00:00:00",
                     "supersedes": None},
                    {"relation": "bench_max", "value": "285 lb", "category": "fitness",
                     "confidence": 0.8, "source_excerpt": "", "timestamp": "2025-01-05T00:00:00",
                     "supersedes": None},
                ],
                "education": [], "career": [], "projects": [], "health": [],
                "preferences": [], "hobbies": [], "study": [], "finance": [],
                "relationships": [], "goals": [],
            },
            "raw_log": []
        }

        with open(temp_profile, 'w') as f:
            json.dump(v1_profile, f)

        # Load — should trigger migration
        profile = UserProfile(temp_profile)

        # All facts should now have fact_id
        fitness = profile.get_category(ProfileCategory.FITNESS, include_historical=True)
        for fact in fitness:
            assert "fact_id" in fact
            assert len(fact["fact_id"]) == 36  # UUID

        # squat_max: 355 lb should be historical, 365 lb should be current
        squat_facts = [f for f in fitness if f["relation"] == "squat_max"]
        assert len(squat_facts) == 2

        old = [f for f in squat_facts if f["value"] == "355 lb"][0]
        new = [f for f in squat_facts if f["value"] == "365 lb"][0]
        assert old["is_current"] is False
        assert new["is_current"] is True
        assert new["supersedes"] == old["fact_id"]

        # bench_max: only one, should be current
        bench = [f for f in fitness if f["relation"] == "bench_max"]
        assert len(bench) == 1
        assert bench[0]["is_current"] is True

        # Version should be 2.0
        assert profile.version == "2.0"

    def test_migration_fresh_profile(self, temp_profile):
        """Fresh profile starts at version 2.0 without needing migration."""
        profile = UserProfile(temp_profile)
        assert profile.version == "2.0"


class TestEphemeralPruning:
    """Tests for ephemeral fact pruning."""

    @pytest.fixture
    def temp_profile(self):
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_pruning_ephemeral(self, temp_profile):
        """Ephemeral relations pruned to max_history when category exceeds soft cap."""
        import config.app_config as cfg
        original_cap = cfg.PROFILE_CATEGORY_SOFT_CAP
        original_max = cfg.PROFILE_EPHEMERAL_MAX_HISTORY

        try:
            # Set cap low enough that pruning triggers frequently
            # With cap=7 and max_history=5, after pruning:
            # 5 historical + 1 current = 6 entries, then 1 more addition
            # brings it to 7 (not > 7), so at most 1 extra accumulates.
            cfg.PROFILE_CATEGORY_SOFT_CAP = 7
            cfg.PROFILE_EPHEMERAL_MAX_HISTORY = 5

            profile = UserProfile(temp_profile)

            # Add 40 different current_activity values (ephemeral relation)
            for i in range(40):
                profile.add_fact("current_activity", f"activity_{i}", confidence=0.7,
                                 timestamp=datetime(2025, 1, 1) + timedelta(hours=i))

            # Check final state
            all_facts = profile.get_category(ProfileCategory.HOBBIES, include_historical=True)
            current_activity = [f for f in all_facts if f["relation"] == "current_activity"]

            # Historical count should be bounded near max_history
            # May be max_history + 1 due to prune-after-append timing
            historical = [f for f in current_activity if not f.get("is_current", True)]
            assert len(historical) <= cfg.PROFILE_EPHEMERAL_MAX_HISTORY + 1

            # Without pruning, 39 historical entries would exist.
            # Pruning must have removed a significant number.
            assert len(historical) < 20, f"Pruning should have removed many entries, got {len(historical)}"

            # Current should still exist
            current = [f for f in current_activity if f.get("is_current", True)]
            assert len(current) == 1
            assert current[0]["value"] == "activity_39"

        finally:
            cfg.PROFILE_CATEGORY_SOFT_CAP = original_cap
            cfg.PROFILE_EPHEMERAL_MAX_HISTORY = original_max

    def test_stable_facts_not_pruned(self, temp_profile):
        """Non-ephemeral relations (name, squat_max) are never pruned."""
        import config.app_config as cfg
        original_cap = cfg.PROFILE_CATEGORY_SOFT_CAP

        try:
            cfg.PROFILE_CATEGORY_SOFT_CAP = 5

            profile = UserProfile(temp_profile)

            # Add many different fitness values
            for i in range(10):
                profile.add_fact("squat_max", f"{300 + i} lb", confidence=0.8,
                                 timestamp=datetime(2025, 1, 1) + timedelta(days=i))

            all_facts = profile.get_category(ProfileCategory.FITNESS, include_historical=True)
            squat_facts = [f for f in all_facts if f["relation"] == "squat_max"]
            # All 10 should be preserved (squat_max is not ephemeral)
            assert len(squat_facts) == 10

        finally:
            cfg.PROFILE_CATEGORY_SOFT_CAP = original_cap


class TestTemporalRetrieval:
    """Tests for temporal-aware retrieval and timeline display."""

    @pytest.fixture
    def temp_profile(self):
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_temporal_query_includes_history(self, temp_profile):
        """Temporal keywords trigger inclusion of historical facts."""
        profile = UserProfile(temp_profile)

        profile.add_fact("squat_max", "355 lb", confidence=0.9,
                         timestamp=datetime(2025, 1, 1))
        profile.add_fact("squat_max", "365 lb", confidence=0.9,
                         timestamp=datetime(2025, 2, 1))
        profile.add_fact("squat_max", "375 lb", confidence=0.9,
                         timestamp=datetime(2025, 3, 1))

        # Non-temporal query: only current
        regular = profile.get_relevant_facts("What's my squat?", ProfileCategory.FITNESS, limit=5)
        squat_regular = [f for f in regular if f["relation"] == "squat_max"]
        assert len(squat_regular) == 1

        # Temporal query: includes history
        temporal = profile.get_relevant_facts(
            "What's my squat progress over time?", ProfileCategory.FITNESS, limit=10
        )
        squat_temporal = [f for f in temporal if f["relation"] == "squat_max"]
        assert len(squat_temporal) >= 2  # Should include historical

    def test_timeline_in_context_injection(self, temp_profile):
        """Temporal query adds timeline to context injection."""
        profile = UserProfile(temp_profile)

        profile.add_fact("squat_max", "355 lb", confidence=0.9,
                         timestamp=datetime(2025, 1, 1))
        profile.add_fact("squat_max", "365 lb", confidence=0.9,
                         timestamp=datetime(2025, 2, 1))
        profile.add_fact("squat_max", "375 lb", confidence=0.9,
                         timestamp=datetime(2025, 3, 1))

        context = profile.get_context_injection(
            max_tokens=1000,
            query="How has my squat progress been over time?"
        )
        assert "timeline" in context.lower()
        assert "355 lb" in context
        assert "375 lb" in context

    def test_is_temporal_query(self, temp_profile):
        """Test temporal keyword detection."""
        profile = UserProfile(temp_profile)

        assert profile._is_temporal_query("What's my squat history?")
        assert profile._is_temporal_query("How has my sleep been over time?")
        assert profile._is_temporal_query("What did I used to weigh?")
        assert profile._is_temporal_query("Show my progress")
        assert not profile._is_temporal_query("What's my name?")
        assert not profile._is_temporal_query("How much can I bench?")
