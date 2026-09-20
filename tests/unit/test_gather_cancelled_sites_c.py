"""Regression: three more gather(return_exceptions=True) sites routed through
utils.async_results (2026-09-19, class BCP-0916-T2, batch L1b).

Before this batch each site unpacked/filtered gather results with
``isinstance(x, Exception)`` (or a bare tuple unpack), which is False/fatal
for an ``asyncio.CancelledError`` — a ``BaseException``, not an ``Exception``.
Each test here reproduces one site's exact old failure mode and asserts the
survivor result is used instead of the whole call blowing up.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.actions import gmail_search
from core.email.provider import EmailMessage
from core.email.service import EmailService
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore


# ---------------------------------------------------------------------------
# B1 — memory/storage/multi_collection_chroma_store.py: query_multiple_collections
# ---------------------------------------------------------------------------

class _FakeCollection:
    """Minimal stand-in so _get_collection()/query_single_collection() never
    touch a real Chroma client."""

    def __init__(self, n=1):
        self._n = n

    def count(self):
        return self._n


def _store_without_real_client():
    # Skip __init__ so no real Chroma client / SentenceTransformer is built.
    store = object.__new__(MultiCollectionChromaStore)
    store.collections = {"a": _FakeCollection(), "b": _FakeCollection()}
    return store


def _fake_query_collection_cancel_a(collection_name, query_text, n_results, query_embedding=None):
    if collection_name == "a":
        raise asyncio.CancelledError()
    return [{"id": "x"}]


@pytest.mark.asyncio
async def test_query_multiple_collections_skips_cancelled_collection():
    """Before this fix: ``for collection_name, results in await asyncio.gather(...)``
    UNPACKS every slot before the isinstance check runs — a CancelledError
    instance in a slot raised ``TypeError: cannot unpack non-iterable
    CancelledError object`` and killed the whole batch query, not just the
    one cancelled collection.
    """
    store = _store_without_real_client()
    store.query_collection = _fake_query_collection_cancel_a

    result = await store.query_multiple_collections(["a", "b"], "", n_results=5)

    assert result == {"a": [], "b": [{"id": "x"}]}


# ---------------------------------------------------------------------------
# B2 — core/actions/gmail_search.py: search_gmail_contacts
# ---------------------------------------------------------------------------

def _mock_google_auth():
    mock_creds = MagicMock()
    mock_creds.token = "t"
    mock_auth = MagicMock()
    mock_auth.is_authenticated = True
    mock_auth.get_credentials.return_value = mock_creds
    mock_auth.has_scope.return_value = True
    return mock_auth


def _list_resp(messages):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"messages": messages}
    return resp


async def _fake_fetch_headers_survivor_only(token, message_id, sem):
    if message_id == "msg1":
        raise asyncio.CancelledError()
    return [("from", "Pat Example <pat@example.com>")]


@pytest.mark.asyncio
async def test_gmail_contacts_skips_cancelled_header_fetch(monkeypatch):
    """Before this fix: ``if isinstance(headers, Exception) or not headers``
    is False for a CancelledError (a BaseException, not an Exception), so the
    loop fell through to ``for direction, raw_header in headers`` and raised
    ``TypeError: 'CancelledError' object is not iterable`` — caught by the
    outer ``except Exception`` and silently returning [] instead of the one
    survivor contact.
    """
    gmail_search.clear_cache()
    monkeypatch.setattr(gmail_search, "_fetch_message_headers", _fake_fetch_headers_survivor_only)

    with patch("config.app_config.GOOGLE_GMAIL_SEARCH_ENABLED", True), \
         patch("core.actions.google_auth.get_google_auth", return_value=_mock_google_auth()), \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock,
               return_value=_list_resp([{"id": "msg1"}, {"id": "msg2"}])):
        result = await gmail_search.search_gmail_contacts("Pat")

    assert len(result) == 1
    assert result[0]["email"] == "pat@example.com"


# ---------------------------------------------------------------------------
# B3 — core/email/service.py: EmailService._fan_out
# ---------------------------------------------------------------------------

async def _raise_cancelled():
    raise asyncio.CancelledError()


async def _return_survivor(msg):
    return [msg]


@pytest.mark.asyncio
async def test_email_service_fan_out_skips_cancelled_provider():
    """Before this fix: ``if isinstance(res, Exception)`` is False for a
    CancelledError (a BaseException, not an Exception), so the cancelled
    provider's CancelledError object reached ``merged.extend(res or [])``
    and raised ``TypeError: 'CancelledError' object is not iterable``.
    """
    survivor_msg = EmailMessage(
        provider="gmail",
        message_id="msg2",
        subject="Survivor",
        date="2026-09-01T12:00:00+00:00",
    )
    fake_a = MagicMock()
    fake_a.name = "cancelled_provider"
    fake_b = MagicMock()
    fake_b.name = "gmail"

    service = EmailService(providers=[fake_a, fake_b], cache_ttl_seconds=300)
    merged = await service._fan_out([_raise_cancelled(), _return_survivor(survivor_msg)])

    assert merged == [survivor_msg]
