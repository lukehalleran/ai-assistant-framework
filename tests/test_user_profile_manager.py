"""
tests/test_user_profile_manager.py

Tests for UserProfile wizard methods: is_first_run(), update_preferences(),
update_identity(), get_style_modifier(), migrate_schema().

Validates onboarding wizard integration and schema versioning.
"""

import pytest
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from memory.user_profile import UserProfile
from memory.user_profile_schema import ProfilePreferences, ProfileIdentity, SCHEMA_VERSION
from memory.corpus_manager import CorpusManager


class MockCorpusManager:
    """Mock CorpusManager for testing is_first_run()."""
    def __init__(self, corpus_size: int = 0):
        self.corpus = [{"dummy": "entry"} for _ in range(corpus_size)]


@pytest.fixture
def temp_profile_path():
    """Create a temporary file path for profile testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def temp_corpus_path():
    """Create a temporary file path for corpus testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


class TestIsFirstRun:
    """Test is_first_run() logic for wizard triggering."""

    def test_is_first_run_empty_corpus_no_identity(self, temp_profile_path):
        """Empty corpus + no identity = first run (wizard should run)."""
        profile = UserProfile(profile_path=temp_profile_path)
        corpus_mgr = MockCorpusManager(corpus_size=0)

        assert profile.is_first_run(corpus_mgr) is True

    def test_is_first_run_with_identity(self, temp_profile_path):
        """User with identity = not first run (wizard should skip)."""
        profile = UserProfile(profile_path=temp_profile_path)
        profile.update_identity("TestUser", "they/them")

        corpus_mgr = MockCorpusManager(corpus_size=0)
        assert profile.is_first_run(corpus_mgr) is False

    def test_is_first_run_empty_corpus_with_identity(self, temp_profile_path):
        """Empty corpus but has identity = not first run (user cleared corpus)."""
        profile = UserProfile(profile_path=temp_profile_path)
        profile.update_identity("TestUser", "they/them")

        corpus_mgr = MockCorpusManager(corpus_size=0)
        assert profile.is_first_run(corpus_mgr) is False

    def test_is_first_run_large_corpus_no_identity(self, temp_profile_path):
        """Large corpus (>5 entries) + no identity = not first run (imported corpus)."""
        profile = UserProfile(profile_path=temp_profile_path)

        corpus_mgr = MockCorpusManager(corpus_size=10)
        assert profile.is_first_run(corpus_mgr) is False

    def test_is_first_run_small_corpus_no_identity(self, temp_profile_path):
        """Small corpus (<5 entries) + no identity = first run."""
        profile = UserProfile(profile_path=temp_profile_path)

        corpus_mgr = MockCorpusManager(corpus_size=3)
        assert profile.is_first_run(corpus_mgr) is True

    def test_is_first_run_exactly_five_entries(self, temp_profile_path):
        """Corpus with exactly 5 entries + no identity = not first run."""
        profile = UserProfile(profile_path=temp_profile_path)

        corpus_mgr = MockCorpusManager(corpus_size=5)
        assert profile.is_first_run(corpus_mgr) is False


class TestUpdatePreferences:
    """Test update_preferences() method."""

    def test_update_preferences_valid_style_warm(self, temp_profile_path):
        """Valid style 'warm' should update successfully."""
        profile = UserProfile(profile_path=temp_profile_path)
        profile.update_preferences(style="warm", check_distress=True, brief_responses=False)

        assert profile.preferences.style == "warm"
        assert profile.preferences.check_distress is True
        assert profile.preferences.brief_responses is False

    def test_update_preferences_valid_style_balanced(self, temp_profile_path):
        """Valid style 'balanced' should update successfully."""
        profile = UserProfile(profile_path=temp_profile_path)
        profile.update_preferences(style="balanced", check_distress=False, brief_responses=True)

        assert profile.preferences.style == "balanced"
        assert profile.preferences.check_distress is False
        assert profile.preferences.brief_responses is True

    def test_update_preferences_valid_style_direct(self, temp_profile_path):
        """Valid style 'direct' should update successfully."""
        profile = UserProfile(profile_path=temp_profile_path)
        profile.update_preferences(style="direct")

        assert profile.preferences.style == "direct"

    def test_update_preferences_invalid_style_raises(self, temp_profile_path):
        """Invalid style should raise ValueError."""
        profile = UserProfile(profile_path=temp_profile_path)

        with pytest.raises(ValueError, match="Invalid style.*Must be one of"):
            profile.update_preferences(style="invalid_style")

    def test_update_preferences_persists(self, temp_profile_path):
        """Preferences should persist after save."""
        profile = UserProfile(profile_path=temp_profile_path)
        profile.update_preferences(style="direct", check_distress=False, brief_responses=True)

        # Reload profile
        profile2 = UserProfile(profile_path=temp_profile_path)
        assert profile2.preferences.style == "direct"
        assert profile2.preferences.check_distress is False
        assert profile2.preferences.brief_responses is True


class TestUpdateIdentity:
    """Test update_identity() method."""

    def test_update_identity(self, temp_profile_path):
        """Update identity should set name and pronouns."""
        profile = UserProfile(profile_path=temp_profile_path)
        profile.update_identity(name="Alice", pronouns="she/her")

        assert profile.identity.name == "Alice"
        assert profile.identity.pronouns == "she/her"

    def test_update_identity_strips_whitespace(self, temp_profile_path):
        """Update identity should strip whitespace."""
        profile = UserProfile(profile_path=temp_profile_path)
        profile.update_identity(name="  Bob  ", pronouns="  he/him  ")

        assert profile.identity.name == "Bob"
        assert profile.identity.pronouns == "he/him"

    def test_update_identity_empty_strings(self, temp_profile_path):
        """Update identity should handle empty strings."""
        profile = UserProfile(profile_path=temp_profile_path)
        profile.update_identity(name="", pronouns="")

        assert profile.identity.name == ""
        assert profile.identity.pronouns == ""

    def test_update_identity_persists(self, temp_profile_path):
        """Identity should persist after save."""
        profile = UserProfile(profile_path=temp_profile_path)
        profile.update_identity(name="Charlie", pronouns="they/them")

        # Reload profile
        profile2 = UserProfile(profile_path=temp_profile_path)
        assert profile2.identity.name == "Charlie"
        assert profile2.identity.pronouns == "they/them"


class TestGetStyleModifier:
    """Test get_style_modifier() method."""

    def test_get_style_modifier_warm(self, temp_profile_path):
        """Warm style should return warm modifier."""
        profile = UserProfile(profile_path=temp_profile_path)
        profile.update_preferences(style="warm")

        modifier = profile.get_style_modifier()
        assert "WARM & SUPPORTIVE" in modifier
        assert "empathy" in modifier

    def test_get_style_modifier_balanced_empty(self, temp_profile_path):
        """Balanced style should return empty string (no modifier)."""
        profile = UserProfile(profile_path=temp_profile_path)
        profile.update_preferences(style="balanced")

        modifier = profile.get_style_modifier()
        assert modifier == ""

    def test_get_style_modifier_direct(self, temp_profile_path):
        """Direct style should return direct modifier."""
        profile = UserProfile(profile_path=temp_profile_path)
        profile.update_preferences(style="direct")

        modifier = profile.get_style_modifier()
        assert "DIRECT & CONCISE" in modifier
        assert "shorter" in modifier


class TestSaveLoadRoundtrip:
    """Test save/load roundtrip with new fields."""

    def test_save_load_roundtrip_with_new_fields(self, temp_profile_path):
        """New fields should survive save/load roundtrip."""
        # Create profile with new fields
        profile = UserProfile(profile_path=temp_profile_path)
        profile.update_identity(name="TestUser", pronouns="they/them")
        profile.update_preferences(style="direct", check_distress=False, brief_responses=True)

        # Reload from disk
        profile2 = UserProfile(profile_path=temp_profile_path)

        # Verify all fields loaded correctly
        assert profile2.version == SCHEMA_VERSION
        assert profile2.identity.name == "TestUser"
        assert profile2.identity.pronouns == "they/them"
        assert profile2.preferences.style == "direct"
        assert profile2.preferences.check_distress is False
        assert profile2.preferences.brief_responses is True

    def test_save_includes_version_in_json(self, temp_profile_path):
        """Saved JSON should include version field."""
        profile = UserProfile(profile_path=temp_profile_path)
        profile.update_identity(name="Test", pronouns="they/them")

        # Read raw JSON
        with open(temp_profile_path, 'r') as f:
            data = json.load(f)

        assert "version" in data
        assert data["version"] == SCHEMA_VERSION
        assert "identity" in data
        assert "preferences" in data


class TestMigrateSchema:
    """Test migrate_schema() method."""

    def test_migrate_schema_from_old_version(self, temp_profile_path):
        """Old profile without version should migrate to 1.0."""
        # Create old-style profile (no version, identity, preferences)
        old_profile = {
            "user_id": "default",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "quick_profile": {},
            "categories": {},
            "raw_log": []
        }

        with open(temp_profile_path, 'w') as f:
            json.dump(old_profile, f)

        # Load profile (should trigger migration)
        profile = UserProfile(profile_path=temp_profile_path)

        # Verify migration
        assert profile.version == SCHEMA_VERSION
        assert hasattr(profile, 'identity')
        assert hasattr(profile, 'preferences')
        assert profile.identity.name == ""
        assert profile.preferences.style == "balanced"

    def test_migrate_schema_preserves_existing_data(self, temp_profile_path):
        """Migration should preserve existing facts."""
        # Create old-style profile with facts
        old_profile = {
            "user_id": "default",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "quick_profile": {"name": "OldUser"},
            "categories": {
                "identity": [
                    {
                        "relation": "name",
                        "value": "OldUser",
                        "category": "identity",
                        "confidence": 1.0,
                        "source_excerpt": "",
                        "timestamp": datetime.now().isoformat(),
                        "supersedes": None
                    }
                ]
            },
            "raw_log": [{"fact": "test"}]
        }

        with open(temp_profile_path, 'w') as f:
            json.dump(old_profile, f)

        # Load profile (should trigger migration)
        profile = UserProfile(profile_path=temp_profile_path)

        # Verify migration preserved data
        assert profile.version == SCHEMA_VERSION
        assert profile.get_quick_profile()["name"] == "OldUser"

        # Verify facts were preserved
        from memory.user_profile_schema import ProfileCategory
        identity_facts = profile.get_category(ProfileCategory.IDENTITY)
        assert len(identity_facts) == 1
        assert identity_facts[0]["value"] == "OldUser"

        assert len(profile.profile["raw_log"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
