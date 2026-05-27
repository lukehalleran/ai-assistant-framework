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
from typing import Optional

from core.actions.types import ActionProposal, ActionResult

logger = logging.getLogger("actions_calendar_create")

CALENDAR_EVENTS_SCOPE = "https://www.googleapis.com/auth/calendar.events"


async def create_calendar_event(proposal: ActionProposal) -> ActionResult:
    """Create a Google Calendar event.

    Expects proposal.params to contain:
        - summary (str): Event title.
        - start_time (str): ISO 8601 datetime for event start.
        - end_time (str): ISO 8601 datetime for event end.
        - description (str, optional): Event description.
        - calendar_id (str, optional): Calendar ID, defaults to "primary".
        - time_zone (str, optional): IANA timezone, defaults to "America/Chicago".
        - location (str, optional): Event location.
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

    # Validate required params
    summary = proposal.params.get("summary", "")
    start_time = proposal.params.get("start_time", "")
    end_time = proposal.params.get("end_time", "")

    if not summary:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="Missing required parameter: summary (event title).",
        )

    if not start_time or not end_time:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="Missing required parameters: start_time and end_time.",
        )

    calendar_id = proposal.params.get("calendar_id", "primary")
    time_zone = proposal.params.get("time_zone", "America/Chicago")
    description = proposal.params.get("description", "")
    location = proposal.params.get("location", "")

    # Build event body
    event_body: dict = {
        "summary": summary,
        "start": {"dateTime": start_time, "timeZone": time_zone},
        "end": {"dateTime": end_time, "timeZone": time_zone},
    }
    if description:
        event_body["description"] = description
    if location:
        event_body["location"] = location

    try:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events",
                headers={"Authorization": f"Bearer {creds.token}"},
                json=event_body,
                timeout=15.0,
            )

        if resp.status_code == 200:
            data = resp.json()
            event_link = data.get("htmlLink", "")
            event_id = data.get("id", "")
            logger.info(f"[CalendarCreate] Event created: {summary} ({event_id})")
            msg = f"Calendar event created: {summary}"
            if event_link:
                msg += f"\n{event_link}"
            return ActionResult(
                action_id=proposal.action_id,
                success=True,
                message=msg,
            )

        error_body = ""
        try:
            error_body = resp.text[:300]
        except Exception:
            pass
        logger.warning(f"[CalendarCreate] API error: HTTP {resp.status_code} — {error_body}")
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message=f"Google Calendar API error: HTTP {resp.status_code} — {error_body[:200]}",
        )

    except Exception as e:
        logger.error(f"[CalendarCreate] Failed: {e}")
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message=f"Calendar event creation failed: {e}",
        )
