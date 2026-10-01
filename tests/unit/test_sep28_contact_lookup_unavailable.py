"""2026-09-28 (BC-47): an empty contact result under a dead Google login must
read as "lookup unavailable", never "no such contact" — both consumers of
resolve_contact (agentic lookup_contact tool, send-email recipient resolution)."""

import pytest

import core.actions.email as email_mod
import core.actions.gmail_search as gmail_search
import core.actions.google_contacts as google_contacts
from core.agentic.tools import ToolExecutor

REASON = "Gmail/Google authorization expired or was revoked — re-authorize"


async def _no_matches(name, max_results=10):
    return []


@pytest.fixture
def dead_login(monkeypatch):
    monkeypatch.setattr(google_contacts, "resolve_contact", _no_matches)
    monkeypatch.setattr(google_contacts, "unavailable_reason", lambda: REASON)
    monkeypatch.setattr(gmail_search, "unavailable_reason", lambda: None)


@pytest.fixture
def healthy_login(monkeypatch):
    monkeypatch.setattr(google_contacts, "resolve_contact", _no_matches)
    monkeypatch.setattr(google_contacts, "unavailable_reason", lambda: None)
    monkeypatch.setattr(gmail_search, "unavailable_reason", lambda: None)
    # 2026-09-30: a healthy login is also an AUTHENTICATED one (D1d checks it
    # before calling an empty lookup "no contacts found").
    import core.actions.google_auth as google_auth
    monkeypatch.setattr(google_auth, "get_google_auth",
                        lambda: type("_Auth", (), {"is_authenticated": True})())


@pytest.mark.asyncio
async def test_lookup_contact_reports_dead_login(dead_login):
    ex = ToolExecutor.__new__(ToolExecutor)
    out = await ToolExecutor._execute_lookup_contact(ex, "Sam")
    assert "CONTACT LOOKUP FAILED" in out
    assert "No contacts found" not in out


@pytest.mark.asyncio
async def test_lookup_contact_genuine_empty_stays_empty(healthy_login):
    ex = ToolExecutor.__new__(ToolExecutor)
    out = await ToolExecutor._execute_lookup_contact(ex, "Sam")
    assert out == "[No contacts found matching 'Sam']"


@pytest.mark.asyncio
async def test_resolve_recipient_reports_dead_login(dead_login):
    addr, msg = await email_mod._resolve_recipient("Sam")
    assert addr is None
    assert "unavailable" in msg
    assert "No matches found" not in msg


@pytest.mark.asyncio
async def test_resolve_recipient_genuine_empty(healthy_login):
    addr, msg = await email_mod._resolve_recipient("Sam")
    assert addr is None
    assert "No matches found in Google Contacts" in msg
