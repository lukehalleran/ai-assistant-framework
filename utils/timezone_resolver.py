"""
User timezone resolution for calendar and scheduling operations (2026-09-01).

Mirror of utils/location_resolver.py for the user's TIMEZONE. Motivation:
replacing the hardcoded America/Chicago default in calendar operations with
a dynamic resolver that checks environment, profile, system timezone, and
provides sensible fallbacks.

Resolution order:
  1. `DAEMON_USER_TIMEZONE` env override (settable via config.local.yaml —
     owner PII never enters source). Must be a valid IANA timezone.
  2. User profile — timezone-shaped facts (profile quick_profile.timezone
     mapped via common labels: eastern/central/mountain/pacific →
     America/New_York / America/Chicago / America/Denver / America/Los_Angeles),
     or direct IANA names (validated).
  3. System /etc/localtime symlink parsed to IANA zone name.
  4. Final fallback to America/Chicago (UTC-6).
"""

import os
import re
import threading
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo, available_timezones

from utils.logging_utils import get_logger

logger = get_logger("timezone_resolver")

TIMEZONE_OVERRIDE = os.getenv("DAEMON_USER_TIMEZONE", "").strip()

_DEFAULT_PROFILE_PATH = os.path.join("data", "user_profile.json")
_FALLBACK_TIMEZONE = "America/Chicago"

# Map common short timezone labels to IANA names (case-insensitive).
_TIMEZONE_ALIASES = {
    "eastern": "America/New_York",
    "central": "America/Chicago",
    "mountain": "America/Denver",
    "pacific": "America/Los_Angeles",
    "est": "America/New_York",
    "cst": "America/Chicago",
    "mst": "America/Denver",
    "pst": "America/Los_Angeles",
    "edt": "America/New_York",
    "cdt": "America/Chicago",
    "mdt": "America/Denver",
    "pdt": "America/Los_Angeles",
}

_VALID_ZONES = available_timezones()


def _is_valid_iana_zone(zone: str) -> bool:
    """Check if a string is a valid IANA timezone name."""
    try:
        ZoneInfo(zone)
        return True
    except Exception:
        return False


def _resolve_system_timezone() -> Optional[str]:
    """
    Parse /etc/localtime symlink to derive system IANA timezone.
    Returns None if unable to determine.
    """
    try:
        localtime_path = Path("/etc/localtime")
        if not localtime_path.exists():
            return None
        # Resolve symlink and extract the path after 'zoneinfo/'
        real_path = localtime_path.resolve()
        path_str = str(real_path)
        if "zoneinfo/" in path_str:
            zone = path_str.split("zoneinfo/", 1)[-1]
            if _is_valid_iana_zone(zone):
                return zone
    except Exception as e:
        logger.debug(f"[Timezone] Failed to read system timezone: {e}")
    return None


class TimezoneResolver:
    """Profile-backed timezone lookup with mtime caching. Never blocks."""

    def __init__(self, profile_path: Optional[str] = None):
        self.profile_path = profile_path or _DEFAULT_PROFILE_PATH
        self._cached: Optional[str] = None
        self._mtime: Optional[float] = None
        self._lock = threading.Lock()

    def get_timezone(self) -> str:
        """
        Best currently-known IANA timezone for the user, or fallback.
        Always returns a valid IANA zone string.
        """
        # Environment override takes precedence
        if TIMEZONE_OVERRIDE:
            if _is_valid_iana_zone(TIMEZONE_OVERRIDE):
                return TIMEZONE_OVERRIDE
            logger.debug(
                f"[Timezone] env override {TIMEZONE_OVERRIDE!r} is not a valid IANA zone"
            )

        # Profile-backed resolution
        tz = self._from_profile()
        if tz:
            return tz

        # System timezone
        sys_tz = _resolve_system_timezone()
        if sys_tz:
            return sys_tz

        # Fallback
        return _FALLBACK_TIMEZONE

    # ------------------------------------------------------------------

    def _from_profile(self) -> Optional[str]:
        try:
            mtime = os.path.getmtime(self.profile_path)
        except OSError:
            return None
        with self._lock:
            if self._mtime == mtime:
                return self._cached
            value = None
            try:
                import json
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    profile = json.load(f)
                value = self._extract(profile)
            except Exception as e:
                logger.debug(f"[Timezone] profile read failed: {e}")
            self._cached = value
            self._mtime = mtime
            return value

    @staticmethod
    def _extract(profile: dict) -> Optional[str]:
        """Extract timezone from profile, checking quick_profile and facts."""
        # Try quick_profile first
        quick_tz = (profile.get("quick_profile", {}) or {}).get("timezone", "").strip()
        if quick_tz:
            tz = TimezoneResolver._resolve_tz_value(quick_tz)
            if tz:
                return tz

        # Try identity facts (relation="timezone")
        categories = profile.get("categories", {}) or {}
        candidates = []
        for facts in categories.values():
            if not isinstance(facts, list):
                continue
            for fact in facts:
                if not isinstance(fact, dict) or not fact.get("is_current", False):
                    continue
                rel = str(fact.get("relation", "")).strip().lower()
                if rel != "timezone":
                    continue
                val = str(fact.get("value", "")).strip()
                tz = TimezoneResolver._resolve_tz_value(val)
                if tz:
                    conf = float(fact.get("confidence", 0.0) or 0.0)
                    candidates.append((-conf, tz))

        if candidates:
            candidates.sort()
            return candidates[0][1]

        return None

    @staticmethod
    def _resolve_tz_value(value: str) -> Optional[str]:
        """
        Resolve a timezone value (alias or IANA name) to a valid IANA zone.
        Returns None if value is invalid.
        """
        value = value.strip()
        if not value:
            return None

        # Check if it's already an IANA zone
        if _is_valid_iana_zone(value):
            return value

        # Try case-insensitive alias lookup
        alias = _TIMEZONE_ALIASES.get(value.lower())
        if alias:
            return alias

        return None


_resolver: Optional[TimezoneResolver] = None
_resolver_lock = threading.Lock()


def get_user_timezone() -> str:
    """
    Best currently-known IANA timezone for the user, or America/Chicago fallback.
    Always returns a valid IANA zone string suitable for Calendar API calls.
    """
    global _resolver
    if _resolver is None:
        with _resolver_lock:
            if _resolver is None:
                _resolver = TimezoneResolver()
    return _resolver.get_timezone()
