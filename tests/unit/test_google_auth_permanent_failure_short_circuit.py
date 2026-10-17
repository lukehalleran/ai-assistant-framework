"""GoogleAuthManager.get_credentials short-circuits a PERMANENT refresh failure,
and the per-turn calendar reader calls it off the event loop.

2026-10-17, classes BC-41 (blocking call on the loop), BC-47 (failure != empty).

Live: `[GoogleAuth] Token refresh failed: ('invalid_grant: Token has been
expired or revoked.' ...)` logged on EVERY turn, 2.5 s of loop stall each.
Synthetic token file in tmp_path only (never data/google_token.json).
"""
import asyncio
import json
import os
import threading
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("google.oauth2.credentials")

from core.actions.google_auth import GoogleAuthManager


class _FakeCreds:
    def __init__(self, exc=None):
        self.expired = True
        self.valid = False
        self.refresh_token = "r"
        self.token = "t"
        self.token_uri = "u"
        self.client_id = "c"
        self.client_secret = "s"
        self.scopes = []
        self.expiry = None
        self.exc = exc
        self.refresh_calls = 0

    def refresh(self, _request):
        self.refresh_calls += 1
        if self.exc is not None:
            raise self.exc
        self.expired = False
        self.valid = True


def _mgr(tmp_path, creds):
    token = tmp_path / "token.json"
    token.write_text(json.dumps({"token": "x"}))
    mgr = GoogleAuthManager(client_id="c", client_secret="s", token_path=str(token))
    mgr._load_token = lambda: creds
    return mgr, token


def _invalid_grant():
    return Exception("invalid_grant: Token has been expired or revoked.",
                     {"error": "invalid_grant"})


def test_permanent_failure_refreshes_once_then_short_circuits(tmp_path):
    creds = _FakeCreds(_invalid_grant())
    mgr, _ = _mgr(tmp_path, creds)
    with patch("google.auth.transport.requests.Request"):
        assert mgr.get_credentials() is None
        assert creds.refresh_calls == 1
        assert mgr.get_credentials() is None
        assert mgr.get_credentials() is None
    assert creds.refresh_calls == 1, "permanent failure must not re-hit the network"
    assert mgr.auth_failure is not None


def test_reauth_new_mtime_refreshes_again(tmp_path):
    creds = _FakeCreds(_invalid_grant())
    mgr, token = _mgr(tmp_path, creds)
    with patch("google.auth.transport.requests.Request"):
        assert mgr.get_credentials() is None
        assert creds.refresh_calls == 1
        st = token.stat()
        os.utime(token, (st.st_atime + 10, st.st_mtime + 10))  # re-auth rewrote the file
        assert mgr.get_credentials() is None
    assert creds.refresh_calls == 2


def test_transient_failure_is_retried(tmp_path):
    creds = _FakeCreds(ConnectionError("network down"))
    mgr, _ = _mgr(tmp_path, creds)
    with patch("google.auth.transport.requests.Request"):
        assert mgr.get_credentials() is None
        assert mgr.get_credentials() is None
    assert creds.refresh_calls == 2
    assert "ConnectionError" in (mgr.auth_failure or "")


@pytest.mark.asyncio
async def test_fetch_upcoming_events_gets_credentials_off_the_loop():
    import core.actions.google_calendar as cal

    cal.clear_cache()
    seen = {}

    def fake_get_credentials():
        seen["thread"] = threading.get_ident()
        return None

    auth = MagicMock()
    auth.is_authenticated = True
    auth.get_credentials = fake_get_credentials
    auth.auth_failure = "revoked"

    loop_thread = threading.get_ident()
    with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
         patch("core.actions.google_auth.get_google_auth", return_value=auth):
        assert await cal.fetch_upcoming_events() == []
    assert seen["thread"] != loop_thread
    cal.clear_cache()
