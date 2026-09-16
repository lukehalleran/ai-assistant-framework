"""
# core/actions/google_contacts.py

Module Contract
- Purpose: Search Google Contacts (People API) by name for recipient resolution.
- Public interface:
  - search_contacts(query, max_results) -> List[Dict]: Search saved contacts.
  - search_other_contacts(query, max_results) -> List[Dict]: Search other/auto contacts.
  - resolve_contact(name, max_results) -> List[Dict]: Combined: saved first, then other.
  - clear_cache() -> None
- Dependencies: httpx, core.actions.google_auth
- Side effects: HTTP GET to Google People API (read-only).
  Scopes: contacts.readonly (saved), contacts.other.readonly (other).
  Returns minimal contact fields only: name, email, source.
"""

import time
from typing import Dict, List, Optional, Tuple

from utils.logging_utils import get_logger

logger = get_logger("google_contacts")

# Scopes
CONTACTS_READONLY_SCOPE = "https://www.googleapis.com/auth/contacts.readonly"
OTHER_CONTACTS_READONLY_SCOPE = "https://www.googleapis.com/auth/contacts.other.readonly"

# Module-level cache: (endpoint, normalized_query) -> results
_cache: Dict[Tuple[str, str], List[Dict]] = {}
_cache_ts: Dict[Tuple[str, str], float] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes

# Warmup flag — People API recommends an initial empty-query request
_warmed_up: bool = False

# Last observed API error (2026-09-08, B6 operational fix): a bare
# "HTTP 403" told the owner nothing about WHY (API not enabled for the
# project vs. insufficient OAuth scope vs. quota). Set on any non-200
# response; read by get_runtime_action_health() so the distinction is
# visible without reading logs. Diagnostic only — never gates behavior,
# and no auth/retry logic changes with it.
_last_error: Optional[str] = None

# Google error message bodies can be long HTML-flavored prose; keep the
# warning/health line readable.
_ERROR_MESSAGE_MAX_CHARS = 160


def get_last_error() -> Optional[str]:
    """The most recent Google Contacts API error string (endpoint + Google's
    own status/message), or None if no call has failed yet this process."""
    return _last_error


async def search_contacts(
    query: str,
    max_results: int = 10,
) -> List[Dict]:
    """Search saved Google Contacts by name.

    Returns list of dicts with keys:
        name (str): Contact display name.
        email (str): Primary email address.
        source (str): Always "contacts".

    Returns empty list if disabled, unconfigured, unauthenticated,
    missing scope, API error, or no results.
    """
    try:
        from config.app_config import GOOGLE_CONTACTS_ENABLED  # lazy import: live-config
    except ImportError:
        return []

    if not GOOGLE_CONTACTS_ENABLED:
        return []

    return await _search_api(
        query=query,
        max_results=max_results,
        endpoint="saved",
        scope=CONTACTS_READONLY_SCOPE,
        source_label="contacts",
    )


async def search_other_contacts(
    query: str,
    max_results: int = 10,
) -> List[Dict]:
    """Search Google 'Other Contacts' (auto-saved from interactions).

    Returns list of dicts with keys:
        name (str): Contact display name.
        email (str): Primary email address.
        source (str): Always "other_contacts".

    Returns empty list if disabled, unconfigured, unauthenticated,
    missing scope, API error, or no results.
    """
    try:
        from config.app_config import GOOGLE_OTHER_CONTACTS_ENABLED  # lazy import: live-config
    except ImportError:
        return []

    if not GOOGLE_OTHER_CONTACTS_ENABLED:
        return []

    return await _search_api(
        query=query,
        max_results=max_results,
        endpoint="other",
        scope=OTHER_CONTACTS_READONLY_SCOPE,
        source_label="other_contacts",
    )


async def resolve_contact(
    name: str,
    max_results: int = 10,
) -> List[Dict]:
    """Combined contact resolution: saved contacts first, then other contacts,
    then Gmail header search as fallback.

    Returns deduplicated results with saved contacts listed first.
    """
    results: List[Dict] = []
    seen_emails: set = set()

    # Priority 1: saved contacts
    saved = await search_contacts(name, max_results=max_results)
    for r in saved:
        email_lower = r["email"].lower()
        if email_lower not in seen_emails:
            seen_emails.add(email_lower)
            results.append(r)

    # Priority 2: other contacts (only if saved didn't fill the budget)
    if len(results) < max_results:
        other = await search_other_contacts(name, max_results=max_results)
        for r in other:
            email_lower = r["email"].lower()
            if email_lower not in seen_emails:
                seen_emails.add(email_lower)
                results.append(r)
                if len(results) >= max_results:
                    break

    # Priority 3: Gmail header search (fallback when contacts return nothing)
    if not results:
        try:
            from core.actions.gmail_search import search_gmail_contacts  # lazy import: cycle
            gmail = await search_gmail_contacts(name, max_results=max_results)
            for r in gmail:
                email_lower = r["email"].lower()
                if email_lower not in seen_emails:
                    seen_emails.add(email_lower)
                    results.append(r)
                    if len(results) >= max_results:
                        break
        except Exception as e:
            logger.debug(f"[GoogleContacts] Gmail search fallback failed: {e}")

    return results


def clear_cache() -> None:
    """Clear the contacts cache."""
    global _cache, _cache_ts, _warmed_up, _last_error
    _cache.clear()
    _cache_ts.clear()
    _warmed_up = False
    _last_error = None


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

async def _warmup(auth) -> None:
    """Send an initial empty-query request as recommended by People API docs."""
    global _warmed_up
    if _warmed_up:
        return

    creds = auth.get_credentials()
    if not creds:
        return

    try:
        import httpx  # lazy import: patch-point (tests/unit/test_audit0831_fixes.py:653)

        async with httpx.AsyncClient() as client:
            await client.get(
                "https://people.googleapis.com/v1/people:searchContacts",
                headers={"Authorization": f"Bearer {creds.token}"},
                params={"query": "", "readMask": "names", "pageSize": "1"},
                timeout=10.0,
            )
        _warmed_up = True
        logger.debug("[GoogleContacts] Warmup request sent")
    except Exception as e:
        logger.debug(f"[GoogleContacts] Warmup failed (non-fatal): {e}")
        _warmed_up = True  # Don't retry on failure


async def _search_api(
    query: str,
    max_results: int,
    endpoint: str,
    scope: str,
    source_label: str,
) -> List[Dict]:
    """Shared implementation for saved and other contacts search."""
    normalized = query.strip().lower()
    cache_key = (endpoint, normalized)

    # Return cached if fresh
    if cache_key in _cache and (time.time() - _cache_ts.get(cache_key, 0)) < _CACHE_TTL_SECONDS:
        return _cache[cache_key][:max_results]

    from core.actions.google_auth import get_google_auth  # lazy import: cycle

    auth = get_google_auth()
    if auth is None or not auth.is_authenticated:
        return []

    if not auth.has_scope(scope):
        logger.warning(f"[GoogleContacts] Missing scope: {scope}")
        return []

    creds = auth.get_credentials()
    if not creds:
        logger.warning("[GoogleContacts] Token refresh failed")
        return []

    # Warmup on first real call
    await _warmup(auth)

    try:
        import httpx  # lazy import: patch-point (tests/unit/test_audit0831_fixes.py:653)

        if endpoint == "saved":
            url = "https://people.googleapis.com/v1/people:searchContacts"
            params = {
                "query": query,
                "readMask": "names,emailAddresses",
                "pageSize": str(max_results),
            }
        else:
            url = "https://people.googleapis.com/v1/otherContacts:search"
            params = {
                "query": query,
                "readMask": "names,emailAddresses",
                "pageSize": str(max_results),
            }

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {creds.token}"},
                params=params,
                timeout=15.0,
            )

        if resp.status_code != 200:
            global _last_error
            detail = _describe_api_error(resp)
            _last_error = f"HTTP {resp.status_code} ({endpoint}) — {detail}"
            logger.warning(f"[GoogleContacts] API error: HTTP {resp.status_code} ({endpoint}) — {detail}")
            return []

        data = resp.json()
        results = _parse_contacts_response(data, source_label)

        # Cache results
        _cache[cache_key] = results
        _cache_ts[cache_key] = time.time()
        _last_error = None

        logger.info(f"[GoogleContacts] Found {len(results)} results for '{query}' ({endpoint})")
        return results[:max_results]

    except Exception as e:
        logger.warning(f"[GoogleContacts] Search failed ({endpoint}): {e}")
        return []


def _describe_api_error(resp) -> str:
    """Defensively parse a non-200 People API response body's
    ``{"error": {"status": ..., "message": ...}}`` shape into a short
    "STATUS: message" string (message truncated). Never raises — an
    unparseable/empty body degrades to a generic description rather than
    losing the HTTP status the caller already has."""
    try:
        body = resp.json()
    except Exception:
        return "no parseable error body"
    error = body.get("error", {}) if isinstance(body, dict) else {}
    if not isinstance(error, dict):
        return "no parseable error body"
    status = str(error.get("status") or "").strip()
    message = str(error.get("message") or "").strip()[:_ERROR_MESSAGE_MAX_CHARS]
    if status and message:
        return f"{status}: {message}"
    return status or message or "no error details in response body"


def _parse_contacts_response(data: dict, source_label: str) -> List[Dict]:
    """Parse People API search response into minimal contact dicts."""
    contacts = []

    # searchContacts returns {"results": [{"person": {...}}]}
    # otherContacts:search returns {"results": [{"person": {...}}]}
    results = data.get("results", [])

    for item in results:
        person = item.get("person", {})

        # Get display name
        names = person.get("names", [])
        display_name = names[0].get("displayName", "") if names else ""

        # Get primary email
        emails = person.get("emailAddresses", [])
        if not emails:
            continue  # Skip contacts without email

        email = emails[0].get("value", "")
        if not email:
            continue

        contacts.append({
            "name": display_name or email,
            "email": email,
            "source": source_label,
        })

    return contacts
