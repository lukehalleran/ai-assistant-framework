"""
# core/actions/google_calendar_modify.py

Module Contract
- Purpose: Update or delete existing Google Calendar events via the Calendar API.
- Public interface:
  - update_calendar_event(proposal: ActionProposal) -> ActionResult
  - delete_calendar_event(proposal: ActionProposal) -> ActionResult
- Dependencies: httpx, core.actions.google_auth
- Side effects: Modifies/deletes a calendar event (write action). Requires
  calendar.events scope.

Resolution doctrine (never-wrong > always-active): the target event is
identified by summary + date (or an explicit event_id, verified by GET).
EXACTLY one live match is required — zero or multiple matches fail honestly
with the candidates listed, and nothing is touched. Delete is irreversible,
so ambiguity never resolves by guessing.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from core.actions.types import ActionProposal, ActionResult
from core.actions.google_calendar_create import CALENDAR_EVENTS_SCOPE, wall_clock_time

logger = logging.getLogger("actions_calendar_modify")

_API_BASE = "https://www.googleapis.com/calendar/v3/calendars"
_DEFAULT_TZ = "America/Chicago"


def _prereq_error(proposal: ActionProposal):
    """Shared config/auth/scope gating; returns (creds, ActionResult|None)."""
    from config.app_config import GOOGLE_CALENDAR_ENABLED

    def fail(msg):
        return None, ActionResult(
            action_id=proposal.action_id, success=False, message=msg)

    if not GOOGLE_CALENDAR_ENABLED:
        return fail("Google Calendar is not enabled in config.")
    from core.actions.google_auth import get_google_auth
    auth = get_google_auth()
    if auth is None:
        return fail("Google OAuth not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.")
    if not auth.is_authenticated:
        return fail("Google OAuth not authenticated. Run 'python -m core.actions.google_auth' to sign in.")
    if not auth.has_scope(CALENDAR_EVENTS_SCOPE):
        return fail("Google Calendar write scope not granted. "
                    "Re-run 'python -m core.actions.google_auth' to approve calendar changes.")
    creds = auth.get_credentials()
    if not creds:
        return fail("Google token refresh failed.")
    return creds, None


def _start_repr(item: Dict[str, Any]) -> str:
    start = item.get("start", {}) or {}
    return str(start.get("date") or start.get("dateTime") or "")


async def _resolve_event(
    client, token: str, calendar_id: str,
    summary: str, date: str, event_id: str = "",
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return (event, "") on an exact single match, else (None, reason).

    With an event_id the event is fetched directly (verifying it exists).
    Otherwise events whose LOCAL start date equals `date` are matched by
    case-insensitive summary equality first, containment second.
    """
    headers = {"Authorization": f"Bearer {token}"}
    if event_id:
        resp = await client.get(
            f"{_API_BASE}/{calendar_id}/events/{event_id}", headers=headers)
        if resp.status_code != 200:
            return None, f"Event id {event_id!r} not found (HTTP {resp.status_code})."
        return resp.json(), ""

    try:
        day = datetime.strptime(str(date)[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None, f"date must be YYYY-MM-DD (got {date!r})."
    # Generous UTC query window; the exact-day filter below uses each item's
    # own LOCAL start date, so offsets can't push a match out of the window.
    time_min = (day - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    time_max = (day + timedelta(days=2)).strftime("%Y-%m-%dT00:00:00Z")
    resp = await client.get(
        f"{_API_BASE}/{calendar_id}/events", headers=headers,
        params={"timeMin": time_min, "timeMax": time_max,
                "singleEvents": "true", "maxResults": 250,
                "fields": "items(id,summary,start,end)"},
    )
    if resp.status_code != 200:
        return None, f"Could not list events to resolve the target (HTTP {resp.status_code})."
    day_str = day.strftime("%Y-%m-%d")
    on_day = [
        item for item in (resp.json() or {}).get("items", [])
        if _start_repr(item)[:10] == day_str
    ]
    want = (summary or "").casefold().strip()
    exact = [i for i in on_day if (i.get("summary") or "").casefold().strip() == want]
    matches = exact or [
        i for i in on_day if want and want in (i.get("summary") or "").casefold()
    ]
    if len(matches) == 1:
        return matches[0], ""
    if not matches:
        others = "; ".join(
            f"{i.get('summary', '?')} ({_start_repr(i)})" for i in on_day[:5]
        ) or "none"
        return None, (f"No event titled {summary!r} on {day_str}. "
                      f"Events that day: {others}.")
    listing = "; ".join(
        f"{i.get('summary', '?')} ({_start_repr(i)})" for i in matches[:5])
    return None, (f"{len(matches)} events match {summary!r} on {day_str} — "
                  f"refusing to guess. Candidates: {listing}.")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


async def update_calendar_event(proposal: ActionProposal) -> ActionResult:
    """Update one existing event, resolved by summary + date (or event_id).

    Change fields (at least one required): new_summary, new_start_time,
    new_end_time (start/end must be supplied TOGETHER), new_description,
    new_location. all_day + time_zone qualify the new times.
    """
    creds, err = _prereq_error(proposal)
    if err:
        return err
    p = proposal.params or {}
    calendar_id = p.get("calendar_id") or "primary"
    changes: Dict[str, Any] = {}
    if p.get("new_summary"):
        changes["summary"] = str(p["new_summary"])
    if p.get("new_description"):
        changes["description"] = str(p["new_description"])
    if p.get("new_location"):
        changes["location"] = str(p["new_location"])
    new_start, new_end = p.get("new_start_time"), p.get("new_end_time")
    if bool(new_start) != bool(new_end):
        return ActionResult(
            action_id=proposal.action_id, success=False,
            message="new_start_time and new_end_time must be supplied together "
                    "(a lone start could invert the event window).")
    if new_start and new_end:
        if _truthy(p.get("all_day")):
            changes["start"] = {"date": str(new_start)[:10]}
            changes["end"] = {"date": str(new_end)[:10]}
        else:
            tz = p.get("time_zone") or _DEFAULT_TZ
            changes["start"] = {"dateTime": wall_clock_time(str(new_start)), "timeZone": tz}
            changes["end"] = {"dateTime": wall_clock_time(str(new_end)), "timeZone": tz}
    if not changes:
        return ActionResult(
            action_id=proposal.action_id, success=False,
            message="No change fields supplied (new_summary/new_start_time+"
                    "new_end_time/new_description/new_location).")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            event, reason = await _resolve_event(
                client, str(creds.token), calendar_id,
                p.get("summary", ""), p.get("date", ""), p.get("event_id", ""))
            if event is None:
                return ActionResult(
                    action_id=proposal.action_id, success=False, message=reason)
            resp = await client.patch(
                f"{_API_BASE}/{calendar_id}/events/{event['id']}",
                headers={"Authorization": f"Bearer {creds.token}"},
                json=changes,
            )
            if resp.status_code != 200:
                return ActionResult(
                    action_id=proposal.action_id, success=False,
                    message=f"Calendar update failed (HTTP {resp.status_code}): "
                            f"{resp.text[:200]}")
            updated = resp.json() or {}
            return ActionResult(
                action_id=proposal.action_id, success=True,
                message=f"Updated event: {updated.get('summary', event.get('summary', '?'))} "
                        f"→ now {_start_repr(updated) or 'unchanged time'}\n"
                        f"{updated.get('htmlLink', '')}".rstrip())
    except Exception as exc:
        logger.error(f"[CalendarModify] Update failed: {exc}")
        return ActionResult(
            action_id=proposal.action_id, success=False,
            message=f"Calendar update failed: {exc}")


async def delete_calendar_event(proposal: ActionProposal) -> ActionResult:
    """Delete one existing event, resolved by summary + date (or event_id).

    Irreversible — executes only on an exact single match; the result message
    names exactly what was deleted.
    """
    creds, err = _prereq_error(proposal)
    if err:
        return err
    p = proposal.params or {}
    calendar_id = p.get("calendar_id") or "primary"
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            event, reason = await _resolve_event(
                client, str(creds.token), calendar_id,
                p.get("summary", ""), p.get("date", ""), p.get("event_id", ""))
            if event is None:
                return ActionResult(
                    action_id=proposal.action_id, success=False, message=reason)
            resp = await client.delete(
                f"{_API_BASE}/{calendar_id}/events/{event['id']}",
                headers={"Authorization": f"Bearer {creds.token}"},
            )
            if resp.status_code in (200, 204):
                return ActionResult(
                    action_id=proposal.action_id, success=True,
                    message=f"Deleted event: {event.get('summary', '?')} "
                            f"({_start_repr(event)}) from {calendar_id}.")
            if resp.status_code == 410:
                return ActionResult(
                    action_id=proposal.action_id, success=True,
                    message=f"Event {event.get('summary', '?')} was already deleted.")
            return ActionResult(
                action_id=proposal.action_id, success=False,
                message=f"Calendar delete failed (HTTP {resp.status_code}): "
                        f"{resp.text[:200]}")
    except Exception as exc:
        logger.error(f"[CalendarModify] Delete failed: {exc}")
        return ActionResult(
            action_id=proposal.action_id, success=False,
            message=f"Calendar delete failed: {exc}")
