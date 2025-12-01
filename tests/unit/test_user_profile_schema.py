import pytest
from memory.user_profile_schema import (
    ProfileCategory, ProfileFact, categorize_relation, RELATION_CATEGORY_MAP
)
from datetime import datetime

class TestProfileSchema:
    def test_categorize_known_relations(self):
        assert categorize_relation("squat_max") == ProfileCategory.FITNESS
        assert categorize_relation("works_at") == ProfileCategory.CAREER
        assert categorize_relation("cat_name") == ProfileCategory.HOBBIES
        assert categorize_relation("medication") == ProfileCategory.HEALTH

    def test_categorize_heuristic_fallback(self):
        # Unknown relation but matches pattern
        assert categorize_relation("bench_1rm") == ProfileCategory.FITNESS
        assert categorize_relation("favorite_movie") == ProfileCategory.PREFERENCES

    def test_profile_fact_serialization(self):
        fact = ProfileFact(
            relation="squat_max",
            value="365 lb",
            category=ProfileCategory.FITNESS,
            confidence=0.85,
            source_excerpt="my squat is 365",
            timestamp=datetime.now()
        )
        d = fact.to_dict()
        restored = ProfileFact.from_dict(d)
        assert restored.relation == fact.relation
        assert restored.value == fact.value
        assert restored.category == fact.category

    def test_all_categories_exist(self):
        # Verify all expected categories are defined
        expected = ["identity", "education", "career", "projects", "health",
                   "fitness", "preferences", "hobbies", "study", "finance",
                   "relationships", "goals"]
        actual = [cat.value for cat in ProfileCategory]
        assert set(actual) == set(expected)

    def test_relation_map_completeness(self):
        # Ensure all mapped relations point to valid categories
        for relation, category in RELATION_CATEGORY_MAP.items():
            assert isinstance(category, ProfileCategory)
            assert category in ProfileCategory
