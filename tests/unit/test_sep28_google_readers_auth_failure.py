"""tests/unit/test_sep28_google_readers_auth_failure.py

Batch Y3 (2026-09-28): the Google sibling readers (calendar, Gmail contact
search, People-API contacts) distinguish "the lookup could not run" from
"nothing matched" through the REAL GoogleAuthManager (RefreshError
invalid_grant) — not a mocked auth.  A genuine empty 200 stays empty-ok.
Also pins the two gaps PR #40 left: missing scope, and a contacts transport
exception, previously read as an empty result.

class: BC-47, BC-58
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.auth.exceptions import RefreshError

import core.actions.google_calendar as cal
import core.actions.gmail_search as gms
import core.actions.google_contacts as gct
from core.actions.google_auth import GoogleAuthManager

GMAIL = "https://www.googleapis.com/auth/gmail.readonly"
CONTACTS = "https://www.googleapis.com/auth/contacts.readonly"


@pytest.fixture(autouse=True)
def _reset():
    for m in (cal, gms, gct):
        m.clear_cache()
    yield
    for m in (cal, gms, gct):
        m.clear_cache()


def _revoked_auth(tmp_path, monkeypatch, scopes=(GMAIL, CONTACTS)):
    p = tmp_path / "token.json"
    p.write_text("{}")
    auth = GoogleAuthManager(client_id="c", client_secret="s", token_path=str(p))
    creds = MagicMock()
    creds.expired = True
    creds.valid = False
    creds.refresh_token = "r"
    creds.scopes = list(scopes)
    creds.refresh.side_effect = RefreshError(
        "invalid_grant", {"error": "invalid_grant"})
    monkeypatch.setattr(auth, "_load_token", lambda: creds)
    monkeypatch.setattr("core.actions.google_auth.get_google_auth", lambda: auth)
    return auth


def _healthy_auth(monkeypatch, scopes=(GMAIL, CONTACTS)):
    auth = MagicMock()
    auth.is_authenticated = True
    auth.auth_failure = None
    auth.has_scope.side_effect = lambda s: s in scopes
    creds = MagicMock()
    creds.token = "t"
    auth.get_credentials.return_value = creds
    monkeypatch.setattr("core.actions.google_auth.get_google_auth", lambda: auth)
    return auth


def _http(monkeypatch, status=200, body=None, exc=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body if body is not None else {}
    resp.text = ""
    client = MagicMock()
    client.get = AsyncMock(side_effect=exc) if exc else AsyncMock(return_value=resp)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: cm)


def run(c):
    return asyncio.run(c)


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    for n in ("GOOGLE_CALENDAR_ENABLED", "GOOGLE_GMAIL_SEARCH_ENABLED",
              "GOOGLE_CONTACTS_ENABLED"):
        monkeypatch.setattr(f"config.app_config.{n}", True, raising=False)


class TestRevokedGrantRealAuthManager:
    def test_calendar(self, tmp_path, monkeypatch):
        _revoked_auth(tmp_path, monkeypatch)
        assert run(cal.fetch_upcoming_events()) == []
        assert "scripts/reauth_google.py" in cal.unavailable_reason()

    def test_gmail_search(self, tmp_path, monkeypatch):
        _revoked_auth(tmp_path, monkeypatch)
        assert run(gms.search_gmail_contacts("bob")) == []
        assert "scripts/reauth_google.py" in gms.unavailable_reason()

    def test_contacts(self, tmp_path, monkeypatch):
        _revoked_auth(tmp_path, monkeypatch)
        assert run(gct.search_contacts("bob")) == []
        assert "scripts/reauth_google.py" in gct.unavailable_reason()

    def test_resolve_contact_chain(self, tmp_path, monkeypatch):
        _revoked_auth(tmp_path, monkeypatch)
        assert run(gct.resolve_contact("bob")) == []
        assert "scripts/reauth_google.py" in gct.unavailable_reason()
        assert "scripts/reauth_google.py" in gms.unavailable_reason()


class TestGenuineEmptyStaysOk:
    def test_calendar(self, monkeypatch):
        _healthy_auth(monkeypatch)
        _http(monkeypatch, body={"items": []})
        assert run(cal.fetch_upcoming_events()) == []
        assert cal.unavailable_reason() is None

    def test_gmail(self, monkeypatch):
        _healthy_auth(monkeypatch)
        _http(monkeypatch, body={})
        assert run(gms.search_gmail_contacts("bob")) == []
        assert gms.unavailable_reason() is None

    def test_contacts(self, monkeypatch):
        _healthy_auth(monkeypatch)
        _http(monkeypatch, body={"results": []})
        assert run(gct.search_contacts("bob")) == []
        assert gct.unavailable_reason() is None


class TestPr40Gaps:
    def test_gmail_missing_scope_is_unavailable(self, monkeypatch):
        _healthy_auth(monkeypatch, scopes=())
        assert run(gms.search_gmail_contacts("bob")) == []
        assert "gmail.readonly" in (gms.unavailable_reason() or "")

    def test_contacts_missing_scope_is_unavailable(self, monkeypatch):
        _healthy_auth(monkeypatch, scopes=())
        assert run(gct.search_contacts("bob")) == []
        assert "scope" in (gct.unavailable_reason() or "")

    def test_contacts_transport_exception_is_unavailable(self, monkeypatch):
        _healthy_auth(monkeypatch)
        _http(monkeypatch, exc=RuntimeError("boom"))
        assert run(gct.search_contacts("bob")) == []
        assert gct.unavailable_reason() is not None
