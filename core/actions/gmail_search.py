"""
# core/actions/gmail_search.py

Module Contract
- Purpose: Search Gmail message headers to find email addresses for contact resolution.
  Used as a fallback when Google Contacts returns no matches.
- Public interface:
  - search_gmail_contacts(query, max_results) -> List[Dict]
  - clear_cache() -> None
- Dependencies: httpx, email.utils, core.actions.google_auth
- Side effects: HTTP GET to Gmail API (read-only, gmail.readonly scope).
  Fetches message metadata headers (From/To) only — never reads message bodies.
  Returns minimal fields: name, email, direction, source.
"""

import asyncio
import time
from email.utils import parseaddr
from typing import Dict, List, Optional, Tuple

from utils.logging_utils import get_logger

logger = get_logger("gmail_search")

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"

# Module-level cache: normalized_query -> results
_cache: Dict[str, List[Dict]] = {}
_cache_ts: Dict[str, float] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes

# Limit concurrent message fetches to avoid rate limiting
_MAX_CONCURRENT_FETCHES = 5


async def search_gmail_contacts(
    query: str,
    max_results: int = 10,
) -> List[Dict]:
    """Search Gmail message headers for email addresses matching a name.

    Searches for messages where the query appears in From or To headers,
    then extracts and deduplicates the email addresses found.

    Returns list of dicts with keys:
        name (str): Sender/recipient display name.
        email (str): Email address.
        direction (str): "from" or "to".
        source (str): Always "gmail_headers".

    Returns empty list if disabled, unconfigured, unauthenticated,
    missing scope, API error, or no results.
    """
    try:
        from config.app_config import GOOGLE_GMAIL_SEARCH_ENABLED
    except ImportError:
        return []

    if not GOOGLE_GMAIL_SEARCH_ENABLED:
        return []

    normalized = query.strip().lower()
    if normalized in _cache and (time.time() - _cache_ts.get(normalized, 0)) < _CACHE_TTL_SECONDS:
        return _cache[normalized][:max_results]

    from core.actions.google_auth import get_google_auth

    auth = get_google_auth()
    if auth is None or not auth.is_authenticated:
        return []

    if not auth.has_scope(GMAIL_READONLY_SCOPE):
        logger.warning("[GmailSearch] Missing scope: gmail.readonly")
        return []

    creds = auth.get_credentials()
    if not creds:
        logger.warning("[GmailSearch] Token refresh failed")
        return []

    try:
        import httpx

        # Search for messages matching the query
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://gmail.googleapis.com/gmail/v1/users/me/messages",
                headers={"Authorization": f"Bearer {creds.token}"},
                params={
                    "q": query,
                    "maxResults": str(max_results),
                },
                timeout=15.0,
            )

        if resp.status_code != 200:
            try:
                err_body = resp.text[:500]
            except Exception:
                err_body = "(no body)"
            logger.warning(f"[GmailSearch] API error: HTTP {resp.status_code} — {err_body}")
            return []

        data = resp.json()
        messages = data.get("messages", [])

        if not messages:
            return []

        # Fetch headers for each message (with concurrency limit)
        sem = asyncio.Semaphore(_MAX_CONCURRENT_FETCHES)
        headers_list = await asyncio.gather(
            *[_fetch_message_headers(creds.token, msg["id"], sem) for msg in messages],
            return_exceptions=True,
        )

        # Get the user's own email to exclude from results
        _own_email = ""
        try:
            from config.app_config import INTERNET_ACTIONS_SMTP_FROM, INTERNET_ACTIONS_SMTP_USER
            _own_email = (INTERNET_ACTIONS_SMTP_FROM or INTERNET_ACTIONS_SMTP_USER or "").lower()
        except ImportError:
            pass

        # Parse and deduplicate
        seen_emails: Dict[str, Dict] = {}  # email -> best result dict
        for headers in headers_list:
            if isinstance(headers, Exception) or not headers:
                continue
            for direction, raw_header in headers:
                name, email = parseaddr(raw_header)
                if not email or "@" not in email:
                    continue
                email_lower = email.lower()
                # Skip the user's own email address
                if _own_email and email_lower == _own_email:
                    continue
                if email_lower not in seen_emails:
                    seen_emails[email_lower] = {
                        "name": name or email,
                        "email": email,
                        "direction": direction,
                        "source": "gmail_headers",
                    }

        results = list(seen_emails.values())

        # Cache
        _cache[normalized] = results
        _cache_ts[normalized] = time.time()

        logger.info(f"[GmailSearch] Found {len(results)} contacts for '{query}'")
        return results[:max_results]

    except Exception as e:
        logger.warning(f"[GmailSearch] Search failed: {e}")
        return []


async def _fetch_message_headers(
    token: str,
    message_id: str,
    sem: asyncio.Semaphore,
) -> List[Tuple[str, str]]:
    """Fetch From and To headers for a single message.

    Returns list of (direction, header_value) tuples.
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
                        "metadataHeaders": ["From", "To"],
                    },
                    timeout=10.0,
                )

            if resp.status_code != 200:
                return []

            data = resp.json()
            payload = data.get("payload", {})
            headers = payload.get("headers", [])

            results = []
            for h in headers:
                name = h.get("name", "").lower()
                value = h.get("value", "")
                if name == "from":
                    results.append(("from", value))
                elif name == "to":
                    # To header can have multiple addresses
                    for addr in value.split(","):
                        results.append(("to", addr.strip()))

            return results

        except Exception:
            return []


def clear_cache() -> None:
    """Clear the Gmail search cache."""
    _cache.clear()
    _cache_ts.clear()
