"""
User timezone resolution for calendar and scheduling operations (2026-09-01;
contract revised 2026-09-13, A03b-1 / F02 / BC-59 / BC-47).

Mirror of utils/location_resolver.py for the user's TIMEZONE. Motivation:
replacing the hardcoded America/Chicago default in calendar operations with
a dynamic resolver that checks environment, profile and system timezone —
and, when NONE of those resolve, says so instead of guessing.

Resolution order:
  1. `DAEMON_USER_TIMEZONE` env override (settable via config.local.yaml —
     owner PII never enters source). Must be a valid IANA timezone.
  2. User profile — timezone-shaped facts (profile quick_profile.timezone
     mapped via common labels: eastern/central/mountain/pacific →
     America/New_York / America/Chicago / America/Denver / America/Los_Angeles),
     or direct IANA names (validated).
  3. System timezone (2026-09-14, A03b-2): on Windows (`sys.platform ==
     "win32"`), the registry value HKLM
     SYSTEM\\CurrentControlSet\\Control\\TimeZoneInformation\\TimeZoneKeyName
     is read (via `winreg`, imported lazily — the module never imports it on
     a platform where it does not exist) and mapped through the generated
     `utils.windows_timezones.WINDOWS_TO_IANA` table; elsewhere, the
     /etc/localtime symlink is parsed to an IANA zone name. Both paths
     validate the candidate with `_is_valid_iana_zone` before returning it.
  4. Unknown: returns None. There is no silent fallback to a fixed zone
     (Central or otherwise, BC-59) — callers that need a zone to act (e.g.
     calendar event creation) must refuse and ask the user rather than
     guess (BC-47: "unknown" and "a real answer" must never share a shape).
     On Windows, an unrecognized registry key, a registry read error, a
     missing tzdata package, or a mapped value that fails IANA validation
     all resolve to None the same way — never a guessed zone, and never
     UTC as a silent substitute.
"""

import os
import re
import sys
import threading
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo, available_timezones

from utils.bootstrap import get_user_profile_path
from utils.logging_utils import get_logger
from utils.windows_timezones import WINDOWS_TO_IANA

logger = get_logger("timezone_resolver")

TIMEZONE_OVERRIDE = os.getenv("DAEMON_USER_TIMEZONE", "").strip()

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


def _resolve_windows_registry_timezone() -> Optional[str]:
    """
    Windows-only (2026-09-14, A03b-2): read HKLM
    SYSTEM\\CurrentControlSet\\Control\\TimeZoneInformation\\TimeZoneKeyName
    via `winreg` (imported lazily here — it does not exist off Windows) and
    map it through the generated `WINDOWS_TO_IANA` table. Returns None on an
    unknown key, a registry read error, or a mapped value that
    `_is_valid_iana_zone` rejects (e.g. missing tzdata) — never a guess
    (BC-59/BC-47). The debug log carries only the registry key name, never
    any other registry value.
    """
    try:
        import winreg
    except ImportError:
        return None
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\TimeZoneInformation",
        )
        try:
            key_name, _ = winreg.QueryValueEx(key, "TimeZoneKeyName")
        finally:
            winreg.CloseKey(key)
    except Exception as e:
        logger.debug(f"[Timezone] Windows registry read failed: {e}")
        return None
    zone = WINDOWS_TO_IANA.get(key_name)
    if zone and _is_valid_iana_zone(zone):
        return zone
    logger.debug(f"[Timezone] Windows key {key_name!r} has no valid IANA mapping")
    return None


def _resolve_system_timezone() -> Optional[str]:
    """
    Derive the system IANA timezone. On Windows (`sys.platform == "win32"`),
    delegate to `_resolve_windows_registry_timezone`. Elsewhere, parse the
    /etc/localtime symlink (this also covers macOS
    /var/db/timezone/zoneinfo/...). Returns None if unable to determine —
    never a guessed zone.
    """
    if sys.platform == "win32":
        return _resolve_windows_registry_timezone()
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
        self.profile_path = profile_path or get_user_profile_path()
        self._cached: Optional[str] = None
        self._mtime: Optional[float] = None
        self._lock = threading.Lock()

    def get_timezone(self) -> Optional[str]:
        """
        Best currently-known IANA timezone for the user: env override ->
        profile -> system /etc/localtime. Returns None when none of those
        resolve (BC-59: never silently choose Central or any other zone) —
        callers that need a zone to act must refuse and ask, not guess.
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

        # Unknown: no fallback. See module docstring (BC-59/BC-47).
        return None

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


def get_user_timezone() -> Optional[str]:
    """
    Best currently-known IANA timezone for the user (env override -> profile
    -> system), or None when it cannot be determined. None is not an error —
    it means "ask the user" (BC-59/BC-47); it is never coerced to Central,
    UTC, or any other guessed zone.
    """
    global _resolver
    if _resolver is None:
        with _resolver_lock:
            if _resolver is None:
                _resolver = TimezoneResolver()
    return _resolver.get_timezone()


def resolve_event_timezone(explicit: Optional[str] = None) -> Optional[str]:
    """Zone to use for a TIMED calendar event: an explicit zone (IANA name or
    known alias) wins outright, even when the resolver itself is unknown;
    with no explicit zone, the best currently-known user zone. None means
    unknown — the calendar executors and proposal-time validation must refuse
    rather than guess (BC-59/BC-47/BC-46), never silently default to Central
    or UTC.

    A stated but unrecognized zone resolves to None, never to the user's own
    zone: substituting it would schedule the event at the wrong wall-clock
    time (parent review, A03b-1).
    """
    explicit = (explicit or "").strip()
    if explicit:
        return TimezoneResolver._resolve_tz_value(explicit)
    return get_user_timezone()
