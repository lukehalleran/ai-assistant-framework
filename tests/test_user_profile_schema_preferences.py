"""
tests/test_user_profile_schema_preferences.py

Tests for new schema additions: ProfilePreferences, ProfileIdentity, and schema versioning.
Ensures backward compatibility with existing ProfileFact and ProfileCategory structures.
"""

import pytest
from memory.user_profile_schema import (
    SCHEMA_VERSION,
    ProfilePreferences,
    ProfileIdentity,
    ProfileFact,
    ProfileCategory,
    categorize_relation,
)
from datetime import datetime


class TestSchemaVersion:
    """Test schema version constant."""

    def test_schema_version_constant_exists(self):
        """SCHEMA_VERSION constant should exist and be a string."""
        assert SCHEMA_VERSION is not None
        assert isinstance(SCHEMA_VERSION, str)
        assert SCHEMA_VERSION == "1.0"


class TestProfilePreferences:
    """Test ProfilePreferences dataclass."""

    def test_profile_preferences_defaults(self):
        """ProfilePreferences should have sensible defaults."""
        prefs = ProfilePreferences()
        assert prefs.style == "balanced"
        assert prefs.check_distress is True
        assert prefs.brief_responses is False

    def test_profile_preferences_to_dict(self):
        """ProfilePreferences.to_dict() should serialize correctly."""
        prefs = ProfilePreferences(style="warm", check_distress=False, brief_responses=True)
        data = prefs.to_dict()

        assert data["style"] == "warm"
        assert data["check_distress"] is False
        assert data["brief_responses"] is True

    def test_profile_preferences_from_dict(self):
        """ProfilePreferences.from_dict() should deserialize correctly."""
        data = {
            "style": "direct",
            "check_distress": True,
            "brief_responses": False,
        }
        prefs = ProfilePreferences.from_dict(data)

        assert prefs.style == "direct"
        assert prefs.check_distress is True
        assert prefs.brief_responses is False

    def test_profile_preferences_from_dict_partial(self):
        """ProfilePreferences.from_dict() should handle missing fields with defaults."""
        data = {"style": "warm"}
        prefs = ProfilePreferences.from_dict(data)

        assert prefs.style == "warm"
        assert prefs.check_distress is True  # default
        assert prefs.brief_responses is False  # default

    def test_profile_preferences_roundtrip(self):
        """ProfilePreferences should survive to_dict/from_dict roundtrip."""
        original = ProfilePreferences(style="balanced", check_distress=False, brief_responses=True)
        data = original.to_dict()
        restored = ProfilePreferences.from_dict(data)

        assert restored.style == original.style
        assert restored.check_distress == original.check_distress
        assert restored.brief_responses == original.brief_responses


class TestProfileIdentity:
    """Test ProfileIdentity dataclass."""

    def test_profile_identity_defaults(self):
        """ProfileIdentity should default to empty strings."""
        identity = ProfileIdentity()
        assert identity.name == ""
        assert identity.pronouns == ""

    def test_profile_identity_to_dict(self):
        """ProfileIdentity.to_dict() should serialize correctly."""
        identity = ProfileIdentity(name="TestUser", pronouns="they/them")
        data = identity.to_dict()

        assert data["name"] == "TestUser"
        assert data["pronouns"] == "they/them"

    def test_profile_identity_from_dict(self):
        """ProfileIdentity.from_dict() should deserialize correctly."""
        data = {"name": "Alice", "pronouns": "she/her"}
        identity = ProfileIdentity.from_dict(data)

        assert identity.name == "Alice"
        assert identity.pronouns == "she/her"

    def test_profile_identity_from_dict_partial(self):
        """ProfileIdentity.from_dict() should handle missing fields with defaults."""
        data = {"name": "Bob"}
        identity = ProfileIdentity.from_dict(data)

        assert identity.name == "Bob"
        assert identity.pronouns == ""  # default

    def test_profile_identity_roundtrip(self):
        """ProfileIdentity should survive to_dict/from_dict roundtrip."""
        original = ProfileIdentity(name="Charlie", pronouns="he/him")
        data = original.to_dict()
        restored = ProfileIdentity.from_dict(data)

        assert restored.name == original.name
        assert restored.pronouns == original.pronouns


class TestBackwardCompatibility:
    """Ensure new schema doesn't break existing ProfileFact functionality."""

    def test_profile_fact_still_works(self):
        """ProfileFact should still function correctly after schema additions."""
        fact = ProfileFact(
            relation="name",
            value="Luke",
            category=ProfileCategory.IDENTITY,
            confidence=1.0,
            source_excerpt="My name is Luke",
            timestamp=datetime.now(),
            supersedes=None,
        )

        # Test to_dict
        data = fact.to_dict()
        assert data["relation"] == "name"
        assert data["value"] == "Luke"
        assert data["category"] == "identity"

        # Test from_dict
        restored = ProfileFact.from_dict(data)
        assert restored.relation == fact.relation
        assert restored.value == fact.value
        assert restored.category == fact.category

    def test_categorize_relation_still_works(self):
        """categorize_relation() should still work after schema additions."""
        assert categorize_relation("name") == ProfileCategory.IDENTITY
        assert categorize_relation("works_at") == ProfileCategory.CAREER
        assert categorize_relation("squat_max") == ProfileCategory.FITNESS
        assert categorize_relation("unknown_relation") == ProfileCategory.PREFERENCES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
