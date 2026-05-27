"""
# core/actions/google_calendar.py

Module Contract
- Purpose: Fetch upcoming Google Calendar events for prompt injection.
- Public interface:
  - fetch_upcoming_events(max_events, lookahead_days) -> List[Dict]
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
        from config.app_config import GOOGLE_CALENDAR_ENABLED
    except ImportError:
        return []

    if not GOOGLE_CALENDAR_ENABLED:
        return []

    from core.actions.google_auth import get_google_auth

    auth = get_google_auth()
    if auth is None or not auth.is_authenticated:
        return []

    creds = auth.get_credentials()
    if not creds:
        logger.warning("[GoogleCalendar] Token refresh failed")
        return []

    # Build time window
    now = datetime.now(timezone.utc)
    time_min = now.isoformat()
    time_max = (now + timedelta(days=lookahead_days)).isoformat()

    try:
        import httpx

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
            return []

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
        return []


def clear_cache():
    """Clear the event cache (call on session start)."""
    global _cache, _cache_ts
    _cache = None
    _cache_ts = 0.0
