"""Tests for utils/timezone_resolver.py"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from utils.timezone_resolver import TimezoneResolver, get_user_timezone


class TestTimezoneResolver:
    """Resolver logic tests."""

    def test_env_override_wins(self):
        """Env DAEMON_USER_TIMEZONE takes precedence."""
        with patch.dict(os.environ, {"DAEMON_USER_TIMEZONE": "America/New_York"}):
            with patch("utils.timezone_resolver.TIMEZONE_OVERRIDE", "America/New_York"):
                resolver = TimezoneResolver()
                assert resolver.get_timezone() == "America/New_York"

    def test_env_override_invalid_falls_through(self):
        """Invalid env value falls back to profile/system/default."""
        with patch.dict(os.environ, {"DAEMON_USER_TIMEZONE": "Invalid/Zone"}):
            with patch("utils.timezone_resolver.TIMEZONE_OVERRIDE", "Invalid/Zone"):
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    profile = {
                        "quick_profile": {"timezone": "Central"},
                        "categories": {},
                    }
                    json.dump(profile, f)
                    f.flush()
                    try:
                        resolver = TimezoneResolver(f.name)
                        # Should fall back to profile value
                        assert resolver.get_timezone() == "America/Chicago"
                    finally:
                        os.unlink(f.name)

    def test_profile_central_label(self):
        """Profile timezone 'Central' resolves to America/Chicago."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            profile = {
                "quick_profile": {"timezone": "Central"},
                "categories": {},
            }
            json.dump(profile, f)
            f.flush()
            try:
                resolver = TimezoneResolver(f.name)
                assert resolver.get_timezone() == "America/Chicago"
            finally:
                os.unlink(f.name)

    def test_profile_eastern_label(self):
        """Profile timezone 'Eastern' resolves to America/New_York."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            profile = {
                "quick_profile": {"timezone": "Eastern"},
                "categories": {},
            }
            json.dump(profile, f)
            f.flush()
            try:
                resolver = TimezoneResolver(f.name)
                assert resolver.get_timezone() == "America/New_York"
            finally:
                os.unlink(f.name)

    def test_profile_iana_passthrough(self):
        """Profile timezone as IANA name is validated and passed through."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            profile = {
                "quick_profile": {"timezone": "Europe/London"},
                "categories": {},
            }
            json.dump(profile, f)
            f.flush()
            try:
                resolver = TimezoneResolver(f.name)
                assert resolver.get_timezone() == "Europe/London"
            finally:
                os.unlink(f.name)

    def test_profile_junk_value_falls_through(self):
        """Junk timezone value falls through to fallback."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            profile = {
                "quick_profile": {"timezone": "not/a/zone"},
                "categories": {},
            }
            json.dump(profile, f)
            f.flush()
            try:
                resolver = TimezoneResolver(f.name)
                # Mock system timezone to ensure fallback is used
                with patch(
                    "utils.timezone_resolver._resolve_system_timezone",
                    return_value=None,
                ):
                    assert resolver.get_timezone() == "America/Chicago"
            finally:
                os.unlink(f.name)

    def test_system_timezone_resolution(self):
        """System /etc/localtime is parsed when profile unavailable."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            profile = {"quick_profile": {}, "categories": {}}
            json.dump(profile, f)
            f.flush()
            try:
                resolver = TimezoneResolver(f.name)
                # Mock /etc/localtime resolution
                with patch(
                    "utils.timezone_resolver._resolve_system_timezone",
                    return_value="America/Los_Angeles",
                ):
                    assert resolver.get_timezone() == "America/Los_Angeles"
            finally:
                os.unlink(f.name)

    def test_missing_profile_falls_back_to_system_or_default(self):
        """Missing profile file falls back to system timezone or default."""
        from zoneinfo import ZoneInfo

        resolver = TimezoneResolver("/nonexistent/path/user_profile.json")
        result = resolver.get_timezone()
        # Should be a valid IANA zone (either system or default)
        try:
            ZoneInfo(result)
        except Exception:
            pytest.fail(f"get_timezone returned invalid IANA zone: {result}")

    def test_case_insensitive_alias(self):
        """Timezone aliases are case-insensitive."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            profile = {
                "quick_profile": {"timezone": "EASTERN"},
                "categories": {},
            }
            json.dump(profile, f)
            f.flush()
            try:
                resolver = TimezoneResolver(f.name)
                assert resolver.get_timezone() == "America/New_York"
            finally:
                os.unlink(f.name)

    def test_mtime_caching(self):
        """Resolver caches on profile mtime."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            profile = {
                "quick_profile": {"timezone": "Central"},
                "categories": {},
            }
            json.dump(profile, f)
            f.flush()
            try:
                resolver = TimezoneResolver(f.name)
                tz1 = resolver.get_timezone()
                # Should be cached
                tz2 = resolver.get_timezone()
                assert tz1 == tz2 == "America/Chicago"
            finally:
                os.unlink(f.name)

    def test_fact_timezone_is_current_filter(self):
        """Only is_current=True timezone facts are considered."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            profile = {
                "quick_profile": {},
                "categories": {
                    "identity": [
                        {
                            "relation": "timezone",
                            "value": "Mountain",
                            "is_current": False,
                            "confidence": 0.95,
                        },
                    ]
                },
            }
            json.dump(profile, f)
            f.flush()
            try:
                resolver = TimezoneResolver(f.name)
                # Mock system timezone to ensure fallback is used
                with patch(
                    "utils.timezone_resolver._resolve_system_timezone",
                    return_value=None,
                ):
                    # Should fall back to default since fact is not current
                    assert resolver.get_timezone() == "America/Chicago"
            finally:
                os.unlink(f.name)

    def test_always_returns_valid_iana(self):
        """get_timezone always returns a valid IANA zone."""
        from zoneinfo import ZoneInfo

        resolver = TimezoneResolver("/nonexistent/path")
        tz = resolver.get_timezone()
        # Should not raise
        ZoneInfo(tz)
