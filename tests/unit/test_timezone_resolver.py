"""Tests for utils/timezone_resolver.py"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.timezone_resolver import TimezoneResolver, get_user_timezone

# Captured at module import time, before any per-test monkeypatching (the
# conftest.py autouse `_sandbox_timezone_resolver` fixture pins
# utils.timezone_resolver.get_user_timezone to a "America/Chicago" lambda
# for every test) — this is the real, unpatched singleton accessor, used
# below to test the deployed function end to end (2026-09-14, A03b-2).
_REAL_GET_USER_TIMEZONE = get_user_timezone


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
        """Profile timezone 'Central' resolves to America/Chicago, taking
        priority over system (2026-09-13, A03b-1: system is never even
        consulted once the profile resolves)."""
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
                with patch("utils.timezone_resolver._resolve_system_timezone") as mock_sys:
                    mock_sys.return_value = "America/Los_Angeles"
                    assert resolver.get_timezone() == "America/Chicago"
                    mock_sys.assert_not_called()
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
        """Junk timezone value in the profile is ignored; resolution falls
        through toward system, then to unknown (None) — 2026-09-13, A03b-1:
        replaces the pre-existing 'falls through to America/Chicago'
        expectation, which encoded BC-59's silent Central default."""
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
                # Deployed-function outcome: system also unknown -> None,
                # never a guessed zone.
                with patch("utils.timezone_resolver._resolve_system_timezone", return_value=None):
                    assert resolver.get_timezone() is None
                # Paired non-triggering control: system IS known -> still used.
                with patch("utils.timezone_resolver._resolve_system_timezone",
                            return_value="America/Denver"):
                    assert resolver.get_timezone() == "America/Denver"
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

    def test_missing_profile_and_system_unknown_returns_none(self):
        """Missing profile + unknown system: no fallback, returns None
        (2026-09-13, A03b-1 / BC-59) — replaces the pre-existing 'always a
        valid IANA zone' expectation (formerly here and at the removed
        test_always_returns_valid_iana), which only happened to pass because
        resolution fell through to a real system zone or the old Central
        default; 'unknown' was never actually surfaced. Paired control: a
        missing profile file alone does not force 'unknown' — a known
        system zone is still used."""
        with patch("utils.timezone_resolver._resolve_system_timezone", return_value=None):
            resolver = TimezoneResolver("/nonexistent/path/user_profile.json")
            assert resolver.get_timezone() is None
        with patch("utils.timezone_resolver._resolve_system_timezone",
                    return_value="America/Los_Angeles"):
            resolver = TimezoneResolver("/nonexistent/path/user_profile.json")
            assert resolver.get_timezone() == "America/Los_Angeles"

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
                # Deployed-function outcome (2026-09-13, A03b-1): the
                # non-current fact is ignored and system is also unknown ->
                # None, not a guessed default (replaces the pre-existing
                # 'falls back to America/Chicago' expectation, BC-59).
                with patch("utils.timezone_resolver._resolve_system_timezone", return_value=None):
                    assert resolver.get_timezone() is None
                # Paired non-triggering control: system IS known -> still used.
                with patch("utils.timezone_resolver._resolve_system_timezone",
                            return_value="America/New_York"):
                    assert resolver.get_timezone() == "America/New_York"
            finally:
                os.unlink(f.name)


def _stub_winreg(key_name=None, open_side_effect=None):
    """Build a MagicMock standing in for the `winreg` module: OpenKey /
    QueryValueEx / CloseKey / HKEY_LOCAL_MACHINE, exactly as real `winreg`
    exposes them. Never touches a real Windows registry."""
    stub = MagicMock()
    stub.HKEY_LOCAL_MACHINE = object()
    if open_side_effect is not None:
        stub.OpenKey.side_effect = open_side_effect
    else:
        stub.OpenKey.return_value = MagicMock()
        stub.QueryValueEx.return_value = (key_name, 1)
    return stub


class TestWindowsSystemTimezone:
    """Windows registry -> IANA resolution (2026-09-14, A03b-2). A stub
    `winreg` module is injected into sys.modules and `sys.platform` is
    patched to "win32"; nothing here reads a real Windows registry or
    depends on the host OS."""

    def test_windows_registry_timezone_resolves_via_table(self, monkeypatch):
        """FAILING FIRST on the unedited source (2026-09-14, A03b-2): today
        `_resolve_system_timezone` has no Windows branch at all, so with
        sys.platform=="win32" and /etc/localtime made absent it falls
        straight through to None regardless of the winreg stub. After the
        fix it reads the stubbed 'Mountain Standard Time' key and maps it
        through WINDOWS_TO_IANA to 'America/Denver'."""
        monkeypatch.setitem(sys.modules, "winreg", _stub_winreg("Mountain Standard Time"))
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(Path, "exists", lambda self: False)

        from utils.timezone_resolver import _resolve_system_timezone
        assert _resolve_system_timezone() == "America/Denver"

    def test_unknown_key_returns_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "winreg", _stub_winreg("Nonexistent Standard Time"))
        monkeypatch.setattr(sys, "platform", "win32")

        from utils.timezone_resolver import _resolve_system_timezone
        assert _resolve_system_timezone() is None

    def test_registry_oserror_returns_none(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "winreg",
            _stub_winreg(open_side_effect=OSError("registry unavailable")),
        )
        monkeypatch.setattr(sys, "platform", "win32")

        from utils.timezone_resolver import _resolve_system_timezone
        assert _resolve_system_timezone() is None

    def test_mapped_zone_rejected_by_validator_returns_none(self, monkeypatch):
        """A mapped IANA value that _is_valid_iana_zone rejects (e.g.
        missing tzdata on the target machine) must not be returned."""
        import utils.timezone_resolver as tzmod

        monkeypatch.setitem(sys.modules, "winreg", _stub_winreg("Mountain Standard Time"))
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(tzmod, "_is_valid_iana_zone", lambda z: False)

        assert tzmod._resolve_system_timezone() is None

    def test_get_user_timezone_end_to_end_windows(self, monkeypatch):
        """No override, no profile, Windows stub -> 'America/Denver', driven
        through the real get_user_timezone() singleton. Clears the resolver
        singleton the way the rest of this file does, and undoes conftest's
        autouse Chicago pin with this test-local monkeypatch."""
        import utils.timezone_resolver as tzmod

        monkeypatch.delenv("DAEMON_USER_TIMEZONE", raising=False)
        monkeypatch.setattr(tzmod, "TIMEZONE_OVERRIDE", "")
        monkeypatch.setattr(tzmod, "get_user_timezone", _REAL_GET_USER_TIMEZONE)
        monkeypatch.setattr(tzmod, "_resolver", None)  # fresh singleton per test
        monkeypatch.setattr(tzmod.TimezoneResolver, "_from_profile", lambda self: None)
        monkeypatch.setitem(sys.modules, "winreg", _stub_winreg("Mountain Standard Time"))
        monkeypatch.setattr(sys, "platform", "win32")

        assert tzmod.get_user_timezone() == "America/Denver"

    def test_non_windows_never_touches_winreg_stub(self, monkeypatch):
        """Non-Windows control: sys.platform 'linux' uses /etc/localtime and
        never imports or calls the injected winreg stub."""
        stub_winreg = _stub_winreg("Mountain Standard Time")
        monkeypatch.setitem(sys.modules, "winreg", stub_winreg)
        monkeypatch.setattr(sys, "platform", "linux")

        from utils.timezone_resolver import _resolve_system_timezone
        _resolve_system_timezone()
        stub_winreg.OpenKey.assert_not_called()
        stub_winreg.QueryValueEx.assert_not_called()
