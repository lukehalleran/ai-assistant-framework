"""
# core/email/gmail_provider.py

Module Contract
- Purpose: Gmail provider adapter implementing EmailProvider protocol
  (2026-09-01, docs/EMAIL_INTEGRATION_DESIGN.md).
  Metadata-only read access to Gmail messages (headers + snippet).
- Public interface:
  - GmailProvider: async EmailProvider adapter conforming to core/email/provider.py
- Dependencies: httpx, core.actions.google_auth
- Side effects: HTTP GET to Gmail API (read-only, gmail.readonly scope).
  Fetches message metadata headers (From/To/Subject/Date) only — never full bodies.
  Results NOT persisted to chroma/corpus; in-memory TTL cache lives in service.py.
  All messages sorted newest-first.
"""

import asyncio
from email.utils import parsedate_to_datetime
from typing import List, Optional

from core.email.provider import EmailMessage
from utils.logging_utils import get_logger

logger = get_logger("gmail_provider")

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
_MAX_CONCURRENT_FETCHES = 5


class GmailProvider:
    """Gmail provider adapter — fetches message metadata via Gmail API v1."""

    name: str = "gmail"

    def is_configured(self) -> bool:
        """Cheap check: config flag + auth token file present."""
        try:
            from config.app_config import EMAIL_INTEGRATION_ENABLED, EMAIL_GMAIL_ENABLED
        except ImportError:
            return False

        if not (EMAIL_INTEGRATION_ENABLED and EMAIL_GMAIL_ENABLED):
            return False

        from core.actions.google_auth import get_google_auth

        auth = get_google_auth()
        if auth is None or not auth.is_authenticated:
            return False

        return bool(auth.has_scope(GMAIL_READONLY_SCOPE))

    async def health(self) -> dict:
        """Check availability: config + token + scope (no network call)."""
        try:
            from config.app_config import EMAIL_INTEGRATION_ENABLED, EMAIL_GMAIL_ENABLED
        except ImportError:
            return {
                "available": False,
                "detail": "config import failed",
            }

        if not EMAIL_INTEGRATION_ENABLED:
            return {
                "available": False,
                "detail": "EMAIL_INTEGRATION_ENABLED=False",
            }

        if not EMAIL_GMAIL_ENABLED:
            return {
                "available": False,
                "detail": "EMAIL_GMAIL_ENABLED=False",
            }

        from core.actions.google_auth import get_google_auth

        auth = get_google_auth()
        if auth is None:
            return {
                "available": False,
                "detail": "Google auth not configured",
            }

        if not auth.is_authenticated:
            return {
                "available": False,
                "detail": "No Gmail token file",
            }

        if not auth.has_scope(GMAIL_READONLY_SCOPE):
            return {
                "available": False,
                "detail": (
                    "gmail.readonly scope not granted; "
                    "run: python -m core.actions.google_auth"
                ),
            }

        return {
            "available": True,
            "detail": "Gmail configured and authenticated",
        }

    async def search(
        self,
        query: str,
        *,
        window_days: int = 30,
        limit: int = 20,
    ) -> List[EmailMessage]:
        """Search Gmail for messages matching query within the window, newest first.

        Args:
            query: Search terms (sanitized to avoid breaking Gmail q syntax).
            window_days: Only include messages newer than this many days.
            limit: Max results to return.

        Returns:
            List of EmailMessage objects, sorted newest-first. Empty on any error.
        """
        from core.actions.google_auth import get_google_auth

        auth = get_google_auth()
        if auth is None or not auth.is_authenticated:
            logger.debug("[Gmail] Not authenticated")
            return []

        creds = auth.get_credentials()
        if creds is None:
            logger.warning("[Gmail] Token refresh failed")
            return []

        # Sanitize query: strip problematic characters and collapse whitespace
        safe_query = (query or "").strip()
        safe_query = " ".join(safe_query.split())
        if not safe_query:
            safe_query = "*"

        gmail_query = f'{safe_query} newer_than:{window_days}d'

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                list_resp = await client.get(
                    "https://gmail.googleapis.com/gmail/v1/users/me/messages",
                    headers={"Authorization": f"Bearer {creds.token}"},
                    params={
                        "q": gmail_query,
                        "maxResults": str(limit),
                    },
                    timeout=15.0,
                )

            if list_resp.status_code != 200:
                err_body = (
                    list_resp.text[:500] if list_resp.text else "(no body)"
                )
                logger.warning(
                    f"[Gmail] search list error: "
                    f"HTTP {list_resp.status_code} — {err_body}"
                )
                return []

            data = list_resp.json()
            messages = data.get("messages", [])

            if not messages:
                return []

            # Fetch full metadata for each message (with concurrency limit)
            sem = asyncio.Semaphore(_MAX_CONCURRENT_FETCHES)
            fetch_results = await asyncio.gather(
                *[
                    self._fetch_message_metadata(creds.token, msg["id"], sem)
                    for msg in messages
                ],
                return_exceptions=True,
            )

            results = [r for r in fetch_results if r is not None]
            # Newest first (Gmail search already returns newest-first, but re-sort to be safe)
            results.sort(key=lambda m: m.date or "", reverse=True)
            return results

        except Exception as e:
            logger.warning(f"[Gmail] search failed: {e}")
            return []

    async def recent(
        self,
        *,
        window_days: int = 7,
        limit: int = 25,
    ) -> List[EmailMessage]:
        """Fetch recent messages in the window (no query), newest first.

        Args:
            window_days: Only include messages newer than this many days.
            limit: Max results to return.

        Returns:
            List of EmailMessage objects, sorted newest-first. Empty on any error.
        """
        from core.actions.google_auth import get_google_auth

        auth = get_google_auth()
        if auth is None or not auth.is_authenticated:
            logger.debug("[Gmail] Not authenticated")
            return []

        creds = auth.get_credentials()
        if creds is None:
            logger.warning("[Gmail] Token refresh failed")
            return []

        # Query: newer_than + exclude low-signal categories
        gmail_query = f"newer_than:{window_days}d -category:promotions -category:social"

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                list_resp = await client.get(
                    "https://gmail.googleapis.com/gmail/v1/users/me/messages",
                    headers={"Authorization": f"Bearer {creds.token}"},
                    params={
                        "q": gmail_query,
                        "maxResults": str(limit),
                    },
                    timeout=15.0,
                )

            if list_resp.status_code != 200:
                err_body = (
                    list_resp.text[:500] if list_resp.text else "(no body)"
                )
                logger.warning(
                    f"[Gmail] recent list error: "
                    f"HTTP {list_resp.status_code} — {err_body}"
                )
                return []

            data = list_resp.json()
            messages = data.get("messages", [])

            if not messages:
                return []

            # Fetch full metadata for each message (with concurrency limit)
            sem = asyncio.Semaphore(_MAX_CONCURRENT_FETCHES)
            fetch_results = await asyncio.gather(
                *[
                    self._fetch_message_metadata(creds.token, msg["id"], sem)
                    for msg in messages
                ],
                return_exceptions=True,
            )

            results = [r for r in fetch_results if r is not None]
            # Newest first
            results.sort(key=lambda m: m.date or "", reverse=True)
            return results

        except Exception as e:
            logger.warning(f"[Gmail] recent failed: {e}")
            return []

    async def _fetch_message_metadata(
        self,
        token: str,
        message_id: str,
        sem: asyncio.Semaphore,
    ) -> Optional[EmailMessage]:
        """Fetch metadata for a single message.

        Returns EmailMessage or None on any error.
        """
        async with sem:
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}",
                        headers={"Authorization": f"Bearer {token}"},
                        params={
                            "format": "metadata",
                            "metadataHeaders": ["From", "To", "Subject", "Date"],
                        },
                        timeout=10.0,
                    )

                if resp.status_code != 200:
                    return None

                data = resp.json()
                payload = data.get("payload", {})
                headers = payload.get("headers", [])

                # Parse headers
                sender = ""
                to = ""
                subject = ""
                date_str = ""

                for h in headers:
                    name = (h.get("name", "") or "").lower()
                    value = h.get("value", "") or ""
                    if name == "from":
                        sender = value
                    elif name == "to":
                        to = value
                    elif name == "subject":
                        subject = value
                    elif name == "date":
                        date_str = value

                # Parse date to ISO 8601
                date_iso = ""
                if date_str:
                    try:
                        dt = parsedate_to_datetime(date_str)
                        date_iso = dt.isoformat()
                    except Exception:
                        pass

                # Gmail snippet and threadId are at message top level
                snippet = data.get("snippet", "")
                thread_id = data.get("threadId", "")

                # Check unread status via labelIds
                unread = False
                label_ids = data.get("labelIds", [])
                if "UNREAD" in label_ids:
                    unread = True

                # Web link
                web_link = f"https://mail.google.com/mail/u/0/#inbox/{message_id}"

                return EmailMessage(
                    provider="gmail",
                    message_id=message_id,
                    thread_id=thread_id,
                    sender=sender,
                    to=to,
                    subject=subject,
                    snippet=snippet,
                    date=date_iso,
                    unread=unread,
                    web_link=web_link,
                )

            except Exception as e:
                logger.debug(f"[Gmail] _fetch_message_metadata({message_id}) failed: {e}")
                return None
