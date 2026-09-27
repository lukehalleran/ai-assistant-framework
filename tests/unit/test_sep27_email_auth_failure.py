"""2026-09-27 (BC-47, BC-78, BC-69, BC-71): a failed email search must never
render the same way as a genuinely empty inbox.

Live incident: the Gmail OAuth token was revoked (~09-23); every downstream
reader treated `get_credentials() -> None` as "no credentials" and reported a
silent empty result — five replies said "searched Gmail, came up empty" and
one guessed the mail must be in Outlook, although Gmail's own search never
ran after the revoke. This file drives the DEPLOYED functions end to end:
`GoogleAuthManager.get_credentials`/`auth_failure`, `GmailProvider`/
`OutlookProvider.unavailable_reason`, `registry.provider_coverage`/
`coverage_note`, `EmailService` caching, and `ToolExecutor._execute_email_search`.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.auth.exceptions import RefreshError

from core.actions.google_auth import GoogleAuthManager


def _fake_creds(expired: bool = True, refresh_token: str = "rt"):
    """A stand-in for `google.oauth2.credentials.Credentials` — only the
    attributes `get_credentials()` actually reads."""
    creds = MagicMock()
    creds.expired = expired
    creds.refresh_token = refresh_token
    creds.valid = False
    return creds


class TestGoogleAuthManagerRefreshFailure:
    """Acceptance (a): RefreshError(invalid_grant) -> auth_failure set,
    cleared after the token file's mtime changes."""

    def test_invalid_grant_is_permanent_and_names_reauth(self, tmp_path, monkeypatch):
        token_path = tmp_path / "token.json"
        token_path.write_text("{}")

        auth = GoogleAuthManager(
            client_id="cid", client_secret="csec", token_path=str(token_path))

        creds = _fake_creds()
        error = RefreshError(
            "invalid_grant: Token has been expired or revoked.",
            {"error": "invalid_grant", "error_description": "revoked"},
        )
        creds.refresh.side_effect = error
        monkeypatch.setattr(auth, "_load_token", lambda: creds)

        assert auth.get_credentials() is None
        reason = auth.auth_failure
        assert reason is not None
        assert "revoked" in reason
        assert "scripts/reauth_google.py" in reason

    def test_transient_refresh_error_is_generic_not_permanent(self, tmp_path, monkeypatch):
        token_path = tmp_path / "token.json"
        token_path.write_text("{}")

        auth = GoogleAuthManager(
            client_id="cid", client_secret="csec", token_path=str(token_path))

        creds = _fake_creds()
        # A 5xx-shaped RefreshError with a DIFFERENT error code must not be
        # read as a revoked grant — equality, never a substring match.
        error = RefreshError("server_error", {"error": "server_error"})
        creds.refresh.side_effect = error
        monkeypatch.setattr(auth, "_load_token", lambda: creds)

        assert auth.get_credentials() is None
        reason = auth.auth_failure
        assert reason is not None
        assert "RefreshError" in reason
        assert "reauth_google.py" not in reason

    def test_auth_failure_clears_once_token_file_mtime_changes(self, tmp_path, monkeypatch):
        token_path = tmp_path / "token.json"
        token_path.write_text("{}")

        auth = GoogleAuthManager(
            client_id="cid", client_secret="csec", token_path=str(token_path))

        creds = _fake_creds()
        error = RefreshError("invalid_grant", {"error": "invalid_grant"})
        creds.refresh.side_effect = error
        monkeypatch.setattr(auth, "_load_token", lambda: creds)

        assert auth.get_credentials() is None
        assert auth.auth_failure is not None

        # A reauth rewrites the token file — force a distinct mtime rather
        # than sleeping (filesystem mtime resolution varies).
        recorded_mtime = token_path.stat().st_mtime
        os.utime(token_path, (recorded_mtime + 5, recorded_mtime + 5))

        assert auth.auth_failure is None

    def test_successful_refresh_clears_a_prior_failure(self, tmp_path, monkeypatch):
        token_path = tmp_path / "token.json"
        token_path.write_text("{}")

        auth = GoogleAuthManager(
            client_id="cid", client_secret="csec", token_path=str(token_path))

        failing_creds = _fake_creds()
        failing_creds.refresh.side_effect = RefreshError(
            "invalid_grant", {"error": "invalid_grant"})
        monkeypatch.setattr(auth, "_load_token", lambda: failing_creds)
        assert auth.get_credentials() is None
        assert auth.auth_failure is not None

        # Next call: a healthy, refreshable credential succeeds and saves —
        # `_save_token` must clear the recorded failure immediately (not
        # only via the mtime check on the next read).
        ok_creds = _fake_creds()
        monkeypatch.setattr(auth, "_load_token", lambda: ok_creds)
        monkeypatch.setattr(auth, "_save_token", lambda creds: None)
        result = auth.get_credentials()

        assert result is ok_creds
        assert auth.auth_failure is None

    def test_is_authenticated_semantics_unchanged(self, tmp_path):
        """No token file at all: is_authenticated is False, auth_failure is
        None (a missing token is not a refresh FAILURE, it's not-configured —
        that distinction is registry.provider_coverage's 'unconnected')."""
        token_path = tmp_path / "missing.json"
        auth = GoogleAuthManager(
            client_id="cid", client_secret="csec", token_path=str(token_path))
        assert auth.is_authenticated is False
        assert auth.auth_failure is None


class TestGmailProviderUnavailableReason:
    """Acceptance (b): provider search with creds None -> [] AND
    unavailable_reason() set."""

    @pytest.mark.asyncio
    async def test_search_creds_none_records_auth_reason(self, monkeypatch):
        import core.email.gmail_provider as gp

        gp._clear_failure()
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.get_credentials.return_value = None
        mock_auth.auth_failure = (
            "Gmail/Google authorization expired or was revoked — "
            "re-authorize with: python scripts/reauth_google.py"
        )
        monkeypatch.setattr(
            "core.actions.google_auth.get_google_auth", lambda: mock_auth)

        provider = gp.GmailProvider()
        result = await provider.search("test")

        assert result == []
        assert provider.unavailable_reason() == mock_auth.auth_failure

    @pytest.mark.asyncio
    async def test_a_fresh_provider_instance_still_sees_the_failure(self, monkeypatch):
        """The registry builds a FRESH GmailProvider() per call — the
        failure must live on the auth singleton, not on `self`."""
        import core.email.gmail_provider as gp

        gp._clear_failure()
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.get_credentials.return_value = None
        mock_auth.auth_failure = "Gmail/Google authorization expired or was revoked"
        monkeypatch.setattr(
            "core.actions.google_auth.get_google_auth", lambda: mock_auth)

        first = gp.GmailProvider()
        await first.search("test")

        second = gp.GmailProvider()
        assert second.unavailable_reason() == mock_auth.auth_failure

    @pytest.mark.asyncio
    async def test_non_200_records_transport_failure_cleared_on_success(self, monkeypatch):
        import core.email.gmail_provider as gp
        import httpx

        gp._clear_failure()
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_creds = MagicMock()
        mock_creds.token = "tok"
        mock_auth.get_credentials.return_value = mock_creds
        mock_auth.auth_failure = None
        monkeypatch.setattr(
            "core.actions.google_auth.get_google_auth", lambda: mock_auth)

        bad_resp = MagicMock()
        bad_resp.status_code = 500
        bad_resp.text = "boom"
        monkeypatch.setattr(httpx.AsyncClient, "get", AsyncMock(return_value=bad_resp))

        provider = gp.GmailProvider()
        result = await provider.search("test")
        assert result == []
        assert provider.unavailable_reason() is not None
        assert "HTTP 500" in provider.unavailable_reason()

        # A subsequent 200 clears the transport failure.
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"messages": []}
        monkeypatch.setattr(httpx.AsyncClient, "get", AsyncMock(return_value=ok_resp))
        result = await provider.search("test")
        assert result == []
        assert provider.unavailable_reason() is None


class TestOutlookProviderUnavailableReason:
    """Same shape as Gmail's, module-level, no auth-singleton reason."""

    @pytest.mark.asyncio
    async def test_search_token_none_records_failure(self, monkeypatch):
        import core.email.outlook_provider as op

        op._clear_failure()
        mock_auth = MagicMock()
        mock_auth.get_access_token.return_value = None
        monkeypatch.setattr(
            "core.email.outlook_auth.get_outlook_auth", lambda: mock_auth)

        provider = op.OutlookProvider()
        result = await provider.search("test")

        assert result == []
        reason = provider.unavailable_reason()
        assert reason is not None
        assert "auth_outlook.py" in reason

    @pytest.mark.asyncio
    async def test_health_reports_failure_when_token_exists_but_search_broke(self, monkeypatch):
        import core.email.outlook_provider as op

        op._record_failure("Outlook API error: HTTP 401")
        mock_auth = MagicMock()
        mock_auth.token_exists = True
        mock_auth.has_refresh_token = True
        monkeypatch.setattr(
            "core.email.outlook_auth.get_outlook_auth", lambda: mock_auth)
        monkeypatch.setattr("config.app_config.EMAIL_INTEGRATION_ENABLED", True)
        monkeypatch.setattr("config.app_config.EMAIL_OUTLOOK_ENABLED", True)

        provider = op.OutlookProvider()
        result = await provider.health()

        assert result["available"] is False
        assert "HTTP 401" in result["detail"]
        op._clear_failure()


class TestProviderCoverageFailedKey:
    """Acceptance (c): provider_coverage() lists gmail under `failed`,
    coverage_note() contains 'FAILED'; the two-key shape is preserved when
    nothing failed (existing consumers assert `set(cov) == {...}`)."""

    def test_configured_but_failing_provider_goes_to_failed_not_searched(self, monkeypatch):
        import core.email.registry as reg

        class _FailingGmail:
            name = "gmail"

            def is_configured(self):
                return True

            def unavailable_reason(self):
                return (
                    "Gmail/Google authorization expired or was revoked — "
                    "re-authorize with: python scripts/reauth_google.py"
                )

        monkeypatch.setattr(reg, "PROVIDERS", {
            "gmail": {"factory": lambda: _FailingGmail(), "enabled": lambda: True},
            "outlook": {"factory": lambda: None, "enabled": lambda: False},
        })

        cov = reg.provider_coverage()
        assert cov["searched"] == []
        assert "gmail" in cov["failed"]
        assert "revoked" in cov["failed"]["gmail"]

        note = reg.coverage_note()
        assert "FAILED" in note
        assert "Gmail" in note

    def test_shape_unchanged_when_nothing_failed(self, monkeypatch):
        import core.email.registry as reg

        class _HealthyGmail:
            name = "gmail"

            def is_configured(self):
                return True

            def unavailable_reason(self):
                return None

        monkeypatch.setattr(reg, "PROVIDERS", {
            "gmail": {"factory": lambda: _HealthyGmail(), "enabled": lambda: True},
        })

        cov = reg.provider_coverage()
        assert set(cov) == {"searched", "unconnected"}
        assert cov["searched"] == ["gmail"]

    def test_provider_missing_unavailable_reason_hook_is_never_failed(self, monkeypatch):
        """A provider that doesn't implement the (optional) hook at all —
        e.g. a future third provider that hasn't added it yet — must still
        count as searched, never crash `getattr`."""
        import core.email.registry as reg

        class _NoHookProvider:
            name = "gmail"

            def is_configured(self):
                return True

        monkeypatch.setattr(reg, "PROVIDERS", {
            "gmail": {"factory": lambda: _NoHookProvider(), "enabled": lambda: True},
        })

        cov = reg.provider_coverage()
        assert cov["searched"] == ["gmail"]
        assert "failed" not in cov


class TestEmailServiceDoesNotCacheFailure:
    """Acceptance (d): a failed fan-out is never cached — the second call
    re-invokes the provider."""

    def test_search_not_cached_when_provider_reports_failure(self):
        from core.email.service import EmailService

        class _FailingProvider:
            name = "gmail"

            def __init__(self):
                self.calls = 0

            async def search(self, query, *, window_days=30, limit=20):
                self.calls += 1
                return []

            def unavailable_reason(self):
                return "Gmail/Google authorization expired or was revoked"

        provider = _FailingProvider()
        service = EmailService(providers=[provider], cache_ttl_seconds=300)

        asyncio.run(service.search("test"))
        asyncio.run(service.search("test"))

        assert provider.calls == 2

    def test_recent_not_cached_when_provider_reports_failure(self):
        from core.email.service import EmailService

        class _FailingProvider:
            name = "gmail"

            def __init__(self):
                self.calls = 0

            async def recent(self, *, window_days=7, limit=25):
                self.calls += 1
                return []

            def unavailable_reason(self):
                return "Gmail/Google authorization expired or was revoked"

        provider = _FailingProvider()
        service = EmailService(providers=[provider], cache_ttl_seconds=300)

        asyncio.run(service.recent())
        asyncio.run(service.recent())

        assert provider.calls == 2

    def test_healthy_fan_out_still_caches(self):
        """Regression guard: the non-failure path (existing behavior) must
        still cache — only a failure bypasses it."""
        from core.email.service import EmailService

        class _HealthyProvider:
            name = "gmail"

            def __init__(self):
                self.calls = 0

            async def search(self, query, *, window_days=30, limit=20):
                self.calls += 1
                return []

            def unavailable_reason(self):
                return None

        provider = _HealthyProvider()
        service = EmailService(providers=[provider], cache_ttl_seconds=300)

        asyncio.run(service.search("test"))
        asyncio.run(service.search("test"))

        assert provider.calls == 1


class TestExecuteEmailSearchFailedText:
    """Acceptance (e): _execute_email_search returns the FAILED text, never
    'No emails found', for the auth-failure case, and still returns
    'No emails found' for a genuine empty 200."""

    def _run(self, monkeypatch, cov, coverage_note_text):
        import core.email.service as svc
        import core.email.registry as reg
        from core.agentic.tools import ToolExecutor

        class _FakeService:
            providers = ["gmail"]

            async def search(self, *a, **k):
                return []

            async def recent(self, *a, **k):
                return []

        monkeypatch.setattr(svc, "get_email_service", lambda: _FakeService())
        monkeypatch.setattr(reg, "provider_coverage", lambda: cov)
        monkeypatch.setattr(reg, "coverage_note", lambda: coverage_note_text)

        ex = ToolExecutor.__new__(ToolExecutor)
        return asyncio.run(
            ToolExecutor._execute_email_search(ex, "Aidvantage. Search that", 60))

    def test_auth_failure_never_reads_as_empty_inbox(self, monkeypatch):
        out = self._run(
            monkeypatch,
            cov={
                "searched": [],
                "unconnected": {},
                "failed": {
                    "gmail": (
                        "Gmail/Google authorization expired or was revoked — "
                        "re-authorize with: python scripts/reauth_google.py"
                    ),
                },
            },
            coverage_note_text="Gmail search FAILED: revoked.",
        )
        assert "EMAIL SEARCH FAILED" in out
        assert "No emails found" not in out
        assert "another account" in out
        assert "empty inbox" in out

    def test_genuine_empty_200_still_says_no_emails_found(self, monkeypatch):
        out = self._run(
            monkeypatch,
            cov={"searched": ["gmail"], "unconnected": {}},
            coverage_note_text="Searched: Gmail.",
        )
        assert "No emails found" in out
        assert "EMAIL SEARCH FAILED" not in out

    def test_nothing_connected_at_all_is_not_reported_as_a_failure(self, monkeypatch):
        """Zero providers ever configured (fresh install) is 'unconnected',
        not a FAILURE — nothing actually broke."""
        out = self._run(
            monkeypatch,
            cov={"searched": [], "unconnected": {"gmail": "not connected", "outlook": "disabled"}},
            coverage_note_text="No email accounts connected. Gmail not connected; Outlook disabled.",
        )
        assert "No emails found" in out
        assert "EMAIL SEARCH FAILED" not in out
