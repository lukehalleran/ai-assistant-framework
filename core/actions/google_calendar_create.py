"""
# core/actions/google_calendar_create.py

Module Contract
- Purpose: Create Google Calendar events via the Calendar API.
- Public interface:
  - create_calendar_event(proposal: ActionProposal) -> ActionResult
- Dependencies: httpx, core.actions.google_auth
- Side effects: Creates a calendar event via Google Calendar API (write action).
  Requires calendar.events scope. Blocks if scope not granted.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from core.actions.types import ActionProposal, ActionResult

logger = logging.getLogger("actions_calendar_create")

CALENDAR_EVENTS_SCOPE = "https://www.googleapis.com/auth/calendar.events"


def _event_key(calendar_id: str, summary: str, start_str: str) -> Tuple[str, str, str]:
    """Use the same date identity for all-day and timed calendar events."""
    return calendar_id, summary.casefold().strip(), str(start_str)[:10]


async def _fetch_existing_event_keys(
    token: str, calendar_ids: Set[str], time_min: str, time_max: str
) -> Optional[Set[Tuple[str, str, str]]]:
    """Read the bounded calendar windows, returning None if checking failed."""
    try:
        import httpx
        keys: Set[Tuple[str, str, str]] = set()
        async with httpx.AsyncClient(timeout=15.0) as client:
            for calendar_id in calendar_ids:
                response = await client.get(
                    f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events",
                    headers={"Authorization": f"Bearer {token}"},
                    params={
                        "timeMin": time_min, "timeMax": time_max,
                        "singleEvents": "true", "maxResults": 250,
                        "fields": "items(summary,start)",
                    },
                )
                if response.status_code != 200:
                    return None
                for item in (response.json() or {}).get("items", []):
                    start = item.get("start", {}) or {}
                    date_str = start.get("date") or start.get("dateTime", "")
                    keys.add(_event_key(calendar_id, item.get("summary", ""), date_str))
        return keys
    except Exception as exc:
        logger.warning(f"[CalendarCreate] Duplicate pre-flight unavailable: {exc}")
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _event_items(params: Dict[str, Any], max_events: int) -> Tuple[List[Dict[str, Any]], str]:
    """Normalize one event or an events[] batch and validate everything first."""
    raw_items = params.get("events")
    if raw_items is None:
        raw_items = [params]
    if not isinstance(raw_items, list) or not raw_items:
        return [], "The calendar batch must contain at least one event."
    if len(raw_items) > max_events:
        return [], f"Calendar batch has {len(raw_items)} events; the maximum is {max_events}."

    # Batch-wide defaults are allowed, while an item can override any of
    # them. In particular, native tool callers may naturally emit one
    # top-level all_day=true alongside seven date-only event objects.
    inherited = {
        key: params.get(key)
        for key in (
            "calendar_id", "time_zone", "all_day", "description", "location",
        )
        if key in params and params.get(key) not in (None, "")
    }
    items: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_items, start=1):
        if not isinstance(raw, dict):
            return [], f"Calendar event {index} is not an object."
        event = dict(inherited)
        event.update(raw)
        missing = [
            field for field in ("summary", "start_time", "end_time")
            if not str(event.get(field, "") or "").strip()
        ]
        if missing:
            return [], (
                f"Calendar event {index} is missing required parameter(s): "
                + ", ".join(missing) + "."
            )

        # Reject an impossible/non-positive interval before creating anything.
        try:
            if _truthy(event.get("all_day")):
                start = datetime.strptime(str(event["start_time"]), "%Y-%m-%d")
                end = datetime.strptime(str(event["end_time"]), "%Y-%m-%d")
            else:
                start_raw = str(event["start_time"])
                end_raw = str(event["end_time"])
                # datetime.fromisoformat accepts a bare YYYY-MM-DD as
                # midnight. Treating an omitted all_day flag that way would
                # silently create a timed midnight event instead of the
                # selected all-day entry.
                if not any(separator in start_raw for separator in ("T", " ")):
                    raise ValueError("timed start lacks a time component")
                if not any(separator in end_raw for separator in ("T", " ")):
                    raise ValueError("timed end lacks a time component")
                start = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
                end = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
            if end <= start:
                return [], f"Calendar event {index} must end after it starts."
        except (ValueError, TypeError):
            return [], f"Calendar event {index} has an invalid ISO 8601 start/end time."

        items.append(event)
    return items, ""


def _event_body(event: Dict[str, Any]) -> Dict[str, Any]:
    if _truthy(event.get("all_day")):
        body: Dict[str, Any] = {
            "summary": event["summary"],
            "start": {"date": event["start_time"]},
            "end": {"date": event["end_time"]},
        }
        if event.get("description"):
            body["description"] = event["description"]
        if event.get("location"):
            body["location"] = event["location"]
        return body

    time_zone = event.get("time_zone", "America/Chicago")
    body: Dict[str, Any] = {
        "summary": event["summary"],
        "start": {"dateTime": event["start_time"], "timeZone": time_zone},
        "end": {"dateTime": event["end_time"], "timeZone": time_zone},
    }
    if event.get("description"):
        body["description"] = event["description"]
    if event.get("location"):
        body["location"] = event["location"]
    return body


async def create_calendar_event(proposal: ActionProposal) -> ActionResult:
    """Create one Google Calendar event or an approved events[] batch.

    Expects proposal.params to contain:
        - summary (str): Event title.
        - start_time (str): ISO 8601 datetime for event start.
        - end_time (str): ISO 8601 datetime for event end.
        - description (str, optional): Event description.
        - calendar_id (str, optional): Calendar ID, defaults to "primary".
        - time_zone (str, optional): IANA timezone, defaults to "America/Chicago".
        - location (str, optional): Event location.
        - events (list[dict], optional): Multiple events under one approval;
          every item has the same required/optional fields above.
        - all_day (bool, optional): Use Google date fields; start_time and
          end_time are YYYY-MM-DD and end_time is exclusive (normally next day).
    """
    from config.app_config import GOOGLE_CALENDAR_ENABLED

    if not GOOGLE_CALENDAR_ENABLED:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="Google Calendar is not enabled in config.",
        )

    from core.actions.google_auth import get_google_auth

    auth = get_google_auth()
    if auth is None:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="Google OAuth not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.",
        )

    if not auth.is_authenticated:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="Google OAuth not authenticated. Run 'python -m core.actions.google_auth' to sign in.",
        )

    # Check for calendar write scope
    if not auth.has_scope(CALENDAR_EVENTS_SCOPE):
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="Google Calendar write scope not granted. "
                    "Re-run 'python -m core.actions.google_auth' to approve calendar event creation.",
        )

    creds = auth.get_credentials()
    if not creds:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="Google token refresh failed.",
        )

    try:
        from config.app_config import GOOGLE_CALENDAR_MAX_EVENTS
        max_events = int(GOOGLE_CALENDAR_MAX_EVENTS)
    except (ImportError, TypeError, ValueError):
        max_events = 10
    events, validation_error = _event_items(proposal.params, max_events)
    if validation_error:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message=validation_error,
        )

    # Check the live calendar before creating approved events.  The window is
    # deliberately wider than the event dates to catch timezone representations.
    event_dates = [str(event["start_time"])[:10] for event in events]
    try:
        parsed_dates = [datetime.strptime(value, "%Y-%m-%d") for value in event_dates]
        time_min = (min(parsed_dates) - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
        time_max = (max(parsed_dates) + timedelta(days=2)).strftime("%Y-%m-%dT00:00:00Z")
    except ValueError:
        time_min = time_max = ""
    calendar_ids = {str(event.get("calendar_id", "primary")) for event in events}
    existing = await _fetch_existing_event_keys(
        str(creds.token), calendar_ids, time_min, time_max
    )
    skipped = [
        event for event in events
        if existing is not None and _event_key(
            str(event.get("calendar_id", "primary")),
            str(event["summary"]), str(event["start_time"]),
        ) in existing
    ]
    to_create = [event for event in events if event not in skipped]
    unavailable_note = ""
    if existing is None:
        to_create = events
        unavailable_note = " (note: duplicate check unavailable — created without it)"
    skipped_text = "; ".join(
        f"{event['summary']} ({str(event['start_time'])[:10]})" for event in skipped
    )
    if not to_create:
        return ActionResult(
            action_id=proposal.action_id,
            success=True,
            message=(
                f"All {len(events)} event(s) already exist on your calendar — "
                f"nothing created. Skipped: {skipped_text}"
            ),
        )

    try:
        import httpx

        created: List[Tuple[str, str]] = []
        failures: List[Tuple[str, str]] = []
        async with httpx.AsyncClient() as client:
            for event in to_create:
                summary = str(event["summary"])
                calendar_id = event.get("calendar_id", "primary")
                try:
                    resp = await client.post(
                        f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events",
                        headers={"Authorization": f"Bearer {creds.token}"},
                        json=_event_body(event),
                        timeout=15.0,
                    )
                    if resp.status_code in (200, 201):
                        data = resp.json()
                        event_link = data.get("htmlLink", "")
                        event_id = data.get("id", "")
                        created.append((summary, event_link))
                        logger.info(
                            f"[CalendarCreate] Event created: {summary} ({event_id})"
                        )
                        continue

                    try:
                        error_body = str(resp.text)[:300]
                    except Exception:
                        error_body = ""
                    detail = f"HTTP {resp.status_code}"
                    if error_body:
                        detail += f" — {error_body[:200]}"
                    failures.append((summary, detail))
                    logger.warning(
                        f"[CalendarCreate] API error for {summary}: {detail}"
                    )
                except Exception as event_error:
                    failures.append((summary, str(event_error)))
                    logger.error(
                        f"[CalendarCreate] Failed to create {summary}: {event_error}"
                    )

        if not failures:
            if created:
                try:
                    from core.actions.google_calendar import clear_cache
                    clear_cache()
                except Exception as exc:
                    logger.debug(f"[CalendarCreate] Could not clear calendar cache: {exc}")
            if len(created) == 1:
                summary, event_link = created[0]
                msg = f"Calendar event created: {summary}{unavailable_note}"
                if event_link:
                    msg += f"\n{event_link}"
            else:
                titles = "; ".join(title for title, _ in created)
                msg = f"Created all {len(created)} calendar events: {titles}{unavailable_note}"
            if skipped:
                msg += f" Skipped: {skipped_text}"
            return ActionResult(
                action_id=proposal.action_id,
                success=True,
                message=msg,
            )

        failed_text = "; ".join(f"{title} ({detail})" for title, detail in failures)
        if created:
            created_text = "; ".join(title for title, _ in created)
            message = (
                f"Created {len(created)} of {len(events)} calendar events: {created_text}. "
                f"{('Skipped: ' + skipped_text + '. ') if skipped else ''}"
                f"Failed: {failed_text}. The successful events already exist; do not retry "
                "the entire batch or they may be duplicated."
            )
        elif len(events) == 1:
            detail = failures[0][1]
            message = (
                f"Google Calendar API error: {detail}"
                if detail.startswith("HTTP ") else
                f"Calendar event creation failed: {detail}"
            )
        else:
            message = f"No calendar events were created. Failed: {failed_text}"
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message=message,
        )

    except Exception as e:
        logger.error(f"[CalendarCreate] Failed: {e}")
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message=f"Calendar event creation failed: {e}",
        )
