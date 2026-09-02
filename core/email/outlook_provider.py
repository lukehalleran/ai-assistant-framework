"""
# core/email/outlook_provider.py

Module Contract
- Purpose: Outlook/Microsoft 365 provider adapter implementing EmailProvider
  protocol (2026-09-01, docs/EMAIL_INTEGRATION_DESIGN.md).
  Metadata-only read access to Outlook messages via Microsoft Graph API.
- Public interface:
  - OutlookProvider: async EmailProvider adapter conforming to core/email/provider.py
- Dependencies: httpx, core/email/outlook_auth
- Side effects: HTTP GET to Microsoft Graph (read-only, Mail.Read scope).
  Fetches message metadata only (from/to/subject/date headers via $select).
  Results NOT persisted to chroma/corpus; in-memory TTL cache lives in service.py.
  All messages sorted newest-first.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from core.email.provider import EmailMessage
from utils.logging_utils import get_logger

logger = get_logger("outlook_provider")


class OutlookProvider:
    """Outlook provider adapter — fetches message metadata via Microsoft Graph API."""

    name: str = "outlook"

    def is_configured(self) -> bool:
        """Cheap check: config flag + auth token file present."""
        try:
            from config.app_config import (
                EMAIL_INTEGRATION_ENABLED,
                EMAIL_OUTLOOK_ENABLED,
            )
        except ImportError:
            return False

        if not (EMAIL_INTEGRATION_ENABLED and EMAIL_OUTLOOK_ENABLED):
            return False

        from core.email.outlook_auth import get_outlook_auth

        auth = get_outlook_auth()
        if auth is None or not auth.token_exists:
            return False

        return True

    async def health(self) -> dict:
        """Check availability: config + token (no network call)."""
        try:
            from config.app_config import (
                EMAIL_INTEGRATION_ENABLED,
                EMAIL_OUTLOOK_ENABLED,
            )
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

        if not EMAIL_OUTLOOK_ENABLED:
            return {
                "available": False,
                "detail": "EMAIL_OUTLOOK_ENABLED=False",
            }

        from core.email.outlook_auth import get_outlook_auth

        auth = get_outlook_auth()
        if auth is None:
            return {
                "available": False,
                "detail": "Outlook auth not configured",
            }

        if not auth.token_exists:
            return {
                "available": False,
                "detail": "No Outlook token file; run: python scripts/auth_outlook.py",
            }

        if not auth.has_refresh_token:
            return {
                "available": False,
                "detail": (
                    "No refresh_token; re-authenticate: "
                    "python scripts/auth_outlook.py"
                ),
            }

        return {
            "available": True,
            "detail": "Outlook configured and authenticated",
        }

    async def search(
        self,
        query: str,
        *,
        window_days: int = 30,
        limit: int = 20,
    ) -> List[EmailMessage]:
        """Search Outlook for messages matching query within the window, newest first.

        Args:
            query: Search terms (uses Graph $search parameter).
            window_days: Only include messages newer than this many days.
            limit: Max results to return.

        Returns:
            List of EmailMessage objects, sorted newest-first. Empty on any error.
        """
        from core.email.outlook_auth import get_outlook_auth

        auth = get_outlook_auth()
        if auth is None:
            logger.debug("[Outlook] Not configured")
            return []

        token = auth.get_access_token()
        if token is None:
            logger.warning("[Outlook] Could not get access token")
            return []

        # Sanitize query for Graph $search
        safe_query = (query or "").strip()
        safe_query = " ".join(safe_query.split())
        if not safe_query:
            safe_query = "*"

        # Calculate cutoff date
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=window_days)

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                # Note: Graph $search cannot combine with $filter/$orderby, so we
                # search, then post-filter by date and sort client-side
                resp = await client.get(
                    "https://graph.microsoft.com/v1.0/me/messages",
                    headers={"Authorization": f"Bearer {token}"},
                    params={
                        "$search": f'"{safe_query}"',
                        "$select": (
                            "id,conversationId,subject,from,toRecipients,"
                            "receivedDateTime,bodyPreview,isRead,webLink"
                        ),
                        "$top": str(limit),
                    },
                    timeout=15.0,
                )

            if resp.status_code != 200:
                err_body = resp.text[:500] if resp.text else "(no body)"
                logger.warning(
                    f"[Outlook] search error: HTTP {resp.status_code} — {err_body}"
                )
                return []

            data = resp.json()
            messages = data.get("value", [])

            # Convert and filter by date
            results = []
            for msg in messages:
                received = msg.get("receivedDateTime", "")
                try:
                    msg_date = datetime.fromisoformat(received.replace("Z", "+00:00"))
                    if msg_date < cutoff:
                        continue  # Outside window
                except Exception:
                    pass  # Skip on parse error

                email_msg = self._parse_message(msg)
                if email_msg:
                    results.append(email_msg)

            # Newest first
            results.sort(key=lambda m: m.date or "", reverse=True)
            return results[:limit]

        except Exception as e:
            logger.warning(f"[Outlook] search failed: {e}")
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
        from core.email.outlook_auth import get_outlook_auth

        auth = get_outlook_auth()
        if auth is None:
            logger.debug("[Outlook] Not configured")
            return []

        token = auth.get_access_token()
        if token is None:
            logger.warning("[Outlook] Could not get access token")
            return []

        # Calculate cutoff date
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=window_days)
        cutoff_iso = cutoff.isoformat().replace("+00:00", "Z")

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://graph.microsoft.com/v1.0/me/messages",
                    headers={"Authorization": f"Bearer {token}"},
                    params={
                        "$select": (
                            "id,conversationId,subject,from,toRecipients,"
                            "receivedDateTime,bodyPreview,isRead,webLink"
                        ),
                        "$top": str(limit),
                        "$orderby": "receivedDateTime desc",
                        "$filter": f"receivedDateTime ge {cutoff_iso}",
                    },
                    timeout=15.0,
                )

            if resp.status_code != 200:
                err_body = resp.text[:500] if resp.text else "(no body)"
                logger.warning(
                    f"[Outlook] recent error: HTTP {resp.status_code} — {err_body}"
                )
                return []

            data = resp.json()
            messages = data.get("value", [])

            results = []
            for msg in messages:
                email_msg = self._parse_message(msg)
                if email_msg:
                    results.append(email_msg)

            # Already ordered by Graph, but re-sort to be safe
            results.sort(key=lambda m: m.date or "", reverse=True)
            return results

        except Exception as e:
            logger.warning(f"[Outlook] recent failed: {e}")
            return []

    def _parse_message(self, msg: dict) -> Optional[EmailMessage]:
        """Parse a Microsoft Graph message object into an EmailMessage.

        Returns EmailMessage or None on parse error.
        """
        try:
            message_id = msg.get("id", "")
            if not message_id:
                return None

            # Parse sender
            sender = ""
            from_obj = msg.get("from", {})
            if from_obj:
                email_address = from_obj.get("emailAddress", {})
                name = email_address.get("name", "")
                address = email_address.get("address", "")
                if name and address:
                    sender = f"{name} <{address}>"
                elif address:
                    sender = address

            # Parse recipients
            to = ""
            to_recipients = msg.get("toRecipients", [])
            if to_recipients:
                to_addrs = []
                for recipient in to_recipients:
                    email_address = recipient.get("emailAddress", {})
                    address = email_address.get("address", "")
                    if address:
                        to_addrs.append(address)
                to = ", ".join(to_addrs)

            subject = msg.get("subject", "")
            snippet = msg.get("bodyPreview", "")
            date_str = msg.get("receivedDateTime", "")
            unread = not msg.get("isRead", True)
            web_link = msg.get("webLink", "")
            conversation_id = msg.get("conversationId", "")

            # Normalize date to ISO 8601
            date_iso = ""
            if date_str:
                try:
                    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    date_iso = dt.isoformat()
                except Exception:
                    pass

            return EmailMessage(
                provider="outlook",
                message_id=message_id,
                thread_id=conversation_id,
                sender=sender,
                to=to,
                subject=subject,
                snippet=snippet,
                date=date_iso,
                unread=unread,
                web_link=web_link,
            )

        except Exception as e:
            logger.debug(f"[Outlook] _parse_message failed: {e}")
            return None
