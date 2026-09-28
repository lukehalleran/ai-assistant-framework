"""
# core/actions/google_calendar.py

Module Contract
- Purpose: Fetch upcoming Google Calendar events for prompt injection.
- Public interface:
  - fetch_upcoming_events(max_events, lookahead_days) -> List[Dict]
  - unavailable_reason() -> Optional[str]
  - clear_cache() -> None
- Dependencies: httpx, core.actions.google_auth
- Side effects: HTTP GET to Google Calendar API (read-only, calendar.readonly scope).
  Returns minimal event fields only: summary, start, end, all_day, location.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

from utils.logging_utils import get_logger

logger = get_logger("google_calendar")

# Module-level cache
_cache: Optional[List[Dict]] = None
_cache_ts: float = 0.0
_CACHE_TTL_SECONDS = 300  # 5 minutes

# 2026-09-27 (BC-47, BC-58): the last reason this PROCESS's calendar fetch
# could not run — same shape as core/email/gmail_provider.py's
# `_LAST_FAILURE` (E1). `fetch_upcoming_events()` itself still returns `[]`
# on any failure (unchanged contract — callers that only check truthiness
# are unaffected); `unavailable_reason()` is the SEPARATE hook a consumer
# reads to tell "no upcoming events" apart from "the fetch never ran".
_LAST_FAILURE: Optional[str] = None


def _record_failure(reason: str) -> None:
    global _LAST_FAILURE
    _LAST_FAILURE = reason


def _clear_failure() -> None:
    global _LAST_FAILURE
    _LAST_FAILURE = None


def _auth_failure_reason(auth) -> str:
    """`auth.auth_failure` when it is a genuine non-empty string, else a
    generic fallback — defends against a partially-stubbed test double
    whose unset attribute reads back as a truthy Mock, not the
    `Optional[str]` the real property returns. 2026-09-27 (BC-47, BC-58)."""
    reason = getattr(auth, "auth_failure", None)
    if isinstance(reason, str) and reason:
        return reason
    return "Google Calendar token refresh failed (unknown reason)"


def unavailable_reason() -> Optional[str]:
    """Why the last Google Calendar fetch could not run, or None.

    Auth-singleton state (`auth.auth_failure`) takes precedence over this
    process's own last transport failure — same precedence as
    core/email/gmail_provider.py's `unavailable_reason()`.
    2026-09-27 (BC-47, BC-58)."""
    from core.actions.google_auth import get_google_auth  # lazy import: cycle

    auth = get_google_auth()
    auth_reason = getattr(auth, "auth_failure", None) if auth is not None else None
    if isinstance(auth_reason, str) and auth_reason:
        return auth_reason
    return _LAST_FAILURE


async def fetch_upcoming_events(
    max_events: int = 10,
    lookahead_days: int = 7,
) -> List[Dict]:
    """Fetch upcoming events from Google Calendar.

    Returns list of dicts with keys:
        summary (str): Event title.
        start (str): ISO datetime or date string.
        end (str): ISO datetime or date string.
        all_day (bool): True if all-day event.
        location (str): Event location, empty string if not set.

    Returns empty list if disabled, unconfigured, unauthenticated,
    token refresh fails, or the API errors.
    """
    global _cache, _cache_ts

    # Return cached if fresh
    if _cache is not None and (time.time() - _cache_ts) < _CACHE_TTL_SECONDS:
        return _cache[:max_events]

    try:
        from config.app_config import GOOGLE_CALENDAR_ENABLED  # lazy import: live-config
    except ImportError:
        return []

    if not GOOGLE_CALENDAR_ENABLED:
        return []

    from core.actions.google_auth import get_google_auth  # lazy import: cycle

    auth = get_google_auth()
    if auth is None or not auth.is_authenticated:
        return []

    creds = auth.get_credentials()
    if not creds:
        logger.warning("[GoogleCalendar] Token refresh failed")
        # 2026-09-27 (BC-47, BC-58): a failed refresh is a FAILURE, not
        # silence — `auth.auth_failure` already carries the specific
        # reason (permanent vs transient); follow the E1 shape.
        _record_failure(_auth_failure_reason(auth))
        return []

    # Build time window
    now = datetime.now(timezone.utc)
    time_min = now.isoformat()
    time_max = (now + timedelta(days=lookahead_days)).isoformat()

    try:
        import httpx  # lazy import: patch-point (tests/unit/test_audit0831_fixes.py:653)

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                headers={"Authorization": f"Bearer {creds.token}"},
                params={
                    "timeMin": time_min,
                    "timeMax": time_max,
                    "singleEvents": "true",
                    "orderBy": "startTime",
                    "maxResults": str(max_events),
                    "fields": "items(summary,start,end,location)",
                },
                timeout=15.0,
            )

        if resp.status_code != 200:
            logger.warning(f"[GoogleCalendar] API error: HTTP {resp.status_code}")
            _record_failure(f"Google Calendar API error: HTTP {resp.status_code}")
            return []

        # A 200 means the call itself succeeded — clear any stale transport
        # failure before we even know how many events came back.
        _clear_failure()

        data = resp.json()
        items = data.get("items", [])

        events = []
        for item in items:
            start_obj = item.get("start", {})
            end_obj = item.get("end", {})

            # All-day events use "date", timed events use "dateTime"
            all_day = "date" in start_obj and "dateTime" not in start_obj
            start_str = start_obj.get("dateTime") or start_obj.get("date", "")
            end_str = end_obj.get("dateTime") or end_obj.get("date", "")

            events.append({
                "summary": item.get("summary", "Untitled"),
                "start": start_str,
                "end": end_str,
                "all_day": all_day,
                "location": item.get("location", ""),
            })

        _cache = events
        _cache_ts = time.time()
        logger.info(f"[GoogleCalendar] Fetched {len(events)} upcoming events")
        return events[:max_events]

    except Exception as e:
        logger.warning(f"[GoogleCalendar] Fetch failed: {e}")
        _record_failure(f"Google Calendar fetch failed: {e}")
        return []


def clear_cache():
    """Clear the event cache (call on session start)."""
    global _cache, _cache_ts
    _cache = None
    _cache_ts = 0.0
    _clear_failure()
