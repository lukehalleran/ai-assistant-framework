import pytest
import tempfile
import os
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
        profile = UserProfile(temp_profile)
        profile.add_fact("bench_max", "275 lb", confidence=0.7)
        profile.add_fact("bench_max", "285 lb", confidence=0.8)  # Higher conf wins
        profile.add_fact("bench_max", "265 lb", confidence=0.6)  # Lower conf rejected

        facts = profile.get_category(ProfileCategory.FITNESS)
        bench_facts = [f for f in facts if f["relation"] == "bench_max"]
        assert len(bench_facts) == 1
        assert bench_facts[0]["value"] == "285 lb"

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
        profile = UserProfile(temp_profile)
        profile.add_fact("age", "32", confidence=0.9)
        profile.add_fact("age", "33", confidence=0.95)

        facts = profile.get_category(ProfileCategory.IDENTITY)
        age_facts = [f for f in facts if f["relation"] == "age"]
        assert len(age_facts) == 1
        assert age_facts[0]["value"] == "33"
        assert age_facts[0]["supersedes"] == "32"

    def test_hybrid_retrieval(self, temp_profile):
        """Test hybrid retrieval: 2/3 semantic + 1/3 recent"""
        profile = UserProfile(temp_profile)

        # Add fitness facts with different timestamps
        import time
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
