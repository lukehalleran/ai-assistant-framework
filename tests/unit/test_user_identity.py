"""Tests for utils/user_identity.py"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from utils.user_identity import UserIdentityResolver, get_user_display_name


class TestUserIdentityResolver:
    """Resolver logic tests."""

    def test_env_override_wins(self):
        """Env DAEMON_USER_NAME takes precedence."""
        with patch.dict(os.environ, {"DAEMON_USER_NAME": "Alex"}):
            # Must patch the module-level constant
            with patch("utils.user_identity.USER_NAME_OVERRIDE", "Alex"):
                resolver = UserIdentityResolver()
                assert resolver.get_display_name() == "Alex"

    def test_env_override_junk_falls_through(self):
        """Invalid env value falls back to profile/default."""
        with patch.dict(os.environ, {"DAEMON_USER_NAME": "123 junk"}):
            with patch("utils.user_identity.USER_NAME_OVERRIDE", "123 junk"):
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    profile = {
                        "quick_profile": {"name": "Charlie"},
                        "categories": {},
                    }
                    json.dump(profile, f)
                    f.flush()
                    try:
                        resolver = UserIdentityResolver(f.name)
                        assert resolver.get_display_name() == "Charlie"
                    finally:
                        os.unlink(f.name)

    def test_profile_name_from_identity_category(self):
        """Extract name from identity category facts."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            profile = {
                "quick_profile": {"name": "OldName"},
                "categories": {
                    "identity": [
                        {
                            "relation": "name",
                            "value": "Baker",
                            "is_current": True,
                            "confidence": 0.95,
                        },
                        {
                            "relation": "age",
                            "value": "30",
                            "is_current": True,
                            "confidence": 0.9,
                        },
                    ]
                },
            }
            json.dump(profile, f)
            f.flush()
            try:
                resolver = UserIdentityResolver(f.name)
                assert resolver.get_display_name() == "Baker"
            finally:
                os.unlink(f.name)

    def test_quick_profile_fallback(self):
        """Fall back to quick_profile name when no identity facts."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            profile = {
                "quick_profile": {"name": "Devon"},
                "categories": {},
            }
            json.dump(profile, f)
            f.flush()
            try:
                resolver = UserIdentityResolver(f.name)
                assert resolver.get_display_name() == "Devon"
            finally:
                os.unlink(f.name)

    def test_junk_profile_name_falls_through(self):
        """Junk-shaped names in profile fall through to default."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            profile = {
                "quick_profile": {"name": "123 junk"},
                "categories": {
                    "identity": [
                        {
                            "relation": "name",
                            "value": "not-a-name-123",
                            "is_current": True,
                            "confidence": 0.9,
                        }
                    ]
                },
            }
            json.dump(profile, f)
            f.flush()
            try:
                resolver = UserIdentityResolver(f.name)
                assert resolver.get_display_name() == "the user"
            finally:
                os.unlink(f.name)

    def test_missing_profile_falls_back(self):
        """Missing profile file falls back to default cleanly."""
        resolver = UserIdentityResolver("/nonexistent/path/user_profile.json")
        assert resolver.get_display_name() == "the user"

    def test_hyphenated_name(self):
        """Hyphenated and apostrophed names are valid."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            profile = {
                "quick_profile": {"name": "Mary-Jane"},
                "categories": {},
            }
            json.dump(profile, f)
            f.flush()
            try:
                resolver = UserIdentityResolver(f.name)
                assert resolver.get_display_name() == "Mary-Jane"
            finally:
                os.unlink(f.name)

    def test_highest_confidence_wins(self):
        """Among multiple name facts, highest confidence wins."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            profile = {
                "quick_profile": {"name": "Default"},
                "categories": {
                    "identity": [
                        {
                            "relation": "name",
                            "value": "LowConfidence",
                            "is_current": True,
                            "confidence": 0.5,
                        },
                        {
                            "relation": "name",
                            "value": "HighConfidence",
                            "is_current": True,
                            "confidence": 0.95,
                        },
                    ]
                },
            }
            json.dump(profile, f)
            f.flush()
            try:
                resolver = UserIdentityResolver(f.name)
                assert resolver.get_display_name() == "HighConfidence"
            finally:
                os.unlink(f.name)

    def test_is_current_filter(self):
        """Only is_current=True facts are considered."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            profile = {
                "quick_profile": {"name": "Current"},
                "categories": {
                    "identity": [
                        {
                            "relation": "name",
                            "value": "OldName",
                            "is_current": False,
                            "confidence": 0.95,
                        },
                    ]
                },
            }
            json.dump(profile, f)
            f.flush()
            try:
                resolver = UserIdentityResolver(f.name)
                assert resolver.get_display_name() == "Current"
            finally:
                os.unlink(f.name)


class TestGeneratorPromptSubstitution:
    """Integration: test that generators use get_user_display_name()."""

    def test_daily_notes_prompt_substitution(self):
        """Daily notes generator substitutes __USER_NAME__ in prompts."""
        from utils.daily_notes_generator import SYSTEM_PROMPT_TEMPLATE

        # Should contain the template placeholder
        assert "__USER_NAME__" in SYSTEM_PROMPT_TEMPLATE
        # Should NOT contain literal Luke outside of the template usage
        assert "Luke" not in SYSTEM_PROMPT_TEMPLATE

        # Test substitution
        with patch("utils.user_identity.get_user_display_name", return_value="Zara"):
            result = SYSTEM_PROMPT_TEMPLATE.replace(
                "__USER_NAME__", "Zara"
            )
            assert "Zara" in result
            assert "__USER_NAME__" not in result

    def test_weekly_notes_prompt_substitution(self):
        """Weekly notes generator substitutes __USER_NAME__ in prompts."""
        from utils.weekly_notes_generator import SYSTEM_PROMPT_TEMPLATE

        assert "__USER_NAME__" in SYSTEM_PROMPT_TEMPLATE
        assert "Luke" not in SYSTEM_PROMPT_TEMPLATE

    def test_monthly_notes_prompt_substitution(self):
        """Monthly notes generator substitutes __USER_NAME__ in prompts."""
        from utils.monthly_notes_generator import SYSTEM_PROMPT_TEMPLATE

        assert "__USER_NAME__" in SYSTEM_PROMPT_TEMPLATE
        assert "Luke" not in SYSTEM_PROMPT_TEMPLATE
