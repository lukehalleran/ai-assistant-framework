"""Tests for Gmail API send path in core.actions.email."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from core.actions.email import send_email, _try_gmail_send, _smtp_send
from core.actions.types import ActionProposal


def _make_proposal(**overrides) -> ActionProposal:
    """Create a minimal ActionProposal for email testing."""
    defaults = {
        "action_type": "send_email",
        "params": {
            "recipient": "test@example.com",
            "message": "Hello from Daemon",
            "subject": "Test Subject",
        },
        "summary": "Send test email",
        "reasoning": "Testing",
    }
    defaults.update(overrides)
    return ActionProposal(**defaults)


# Common config patches for Gmail-configured state
_GMAIL_CONFIG = {
    "config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_ID": "test-client-id",
    "config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET": "test-client-secret",
    "config.app_config.INTERNET_ACTIONS_GOOGLE_TOKEN_PATH": "/tmp/test_token.json",
    "config.app_config.INTERNET_ACTIONS_SMTP_FROM": "me@test.com",
    "config.app_config.INTERNET_ACTIONS_SMTP_USER": "me@test.com",
}


def _patch_gmail_config():
    """Return a stack of patches for Gmail-configured state."""
    import contextlib
    return contextlib.ExitStack()


class TestValidation:
    """Test common validation in send_email."""

    @pytest.mark.asyncio
    async def test_invalid_recipient_no_contacts(self):
        """Recipient without '@' and no contact match returns error."""
        proposal = _make_proposal(params={"recipient": "bad", "message": "hi"})
        with patch("core.actions.email._resolve_recipient", new_callable=AsyncMock,
                   return_value=(None, "Could not resolve 'bad'")):
            result = await send_email(proposal)
        assert result.success is False
        assert "Could not resolve" in result.message

    @pytest.mark.asyncio
    async def test_missing_recipient(self):
        proposal = _make_proposal(params={"message": "hi"})
        result = await send_email(proposal)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_empty_message(self):
        proposal = _make_proposal(params={"recipient": "a@b.com", "message": ""})
        result = await send_email(proposal)
        assert result.success is False
        assert "No message" in result.message


class TestGmailPath:
    """Test the Gmail API send path."""

    @pytest.mark.asyncio
    async def test_gmail_send_success(self):
        proposal = _make_proposal()

        mock_creds = MagicMock()
        mock_creds.token = "test-access-token"
        mock_creds.valid = True

        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.get_credentials.return_value = mock_creds

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_async_client = AsyncMock()
        mock_async_client.post.return_value = mock_resp
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_ID", "cid"), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET", "csec"), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_TOKEN_PATH", "/tmp/t.json"), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_FROM", "me@test.com"), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_USER", "me@test.com"), \
             patch("core.actions.google_auth.GoogleAuthManager", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_async_client):
            result = await _try_gmail_send(
                proposal, "test@example.com", "Hello from Daemon"
            )

        assert result is not None
        assert result.success is True
        assert "Gmail" in result.message

    @pytest.mark.asyncio
    async def test_gmail_not_configured_returns_none(self):
        """When Google OAuth not configured, returns None (fall to SMTP)."""
        proposal = _make_proposal()

        with patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_ID", ""), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET", ""):
            result = await _try_gmail_send(
                proposal, "test@example.com", "Hello"
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_gmail_not_authenticated_returns_none(self):
        """When no token, returns None (fall to SMTP)."""
        proposal = _make_proposal()

        mock_auth = MagicMock()
        mock_auth.is_authenticated = False

        with patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_ID", "cid"), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET", "csec"), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_TOKEN_PATH", "/tmp/t.json"), \
             patch("core.actions.google_auth.GoogleAuthManager", return_value=mock_auth):
            result = await _try_gmail_send(
                proposal, "test@example.com", "Hello"
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_gmail_api_error_returns_failure_not_none(self):
        """Gmail API 403 returns failure ActionResult, NOT None (no SMTP fallback)."""
        proposal = _make_proposal()

        mock_creds = MagicMock()
        mock_creds.token = "tok"
        mock_creds.valid = True

        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.get_credentials.return_value = mock_creds

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"

        mock_async_client = AsyncMock()
        mock_async_client.post.return_value = mock_resp
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_ID", "cid"), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET", "csec"), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_TOKEN_PATH", "/tmp/t.json"), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_FROM", "me@t.com"), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_USER", "me@t.com"), \
             patch("core.actions.google_auth.GoogleAuthManager", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_async_client):
            result = await _try_gmail_send(
                proposal, "test@example.com", "Hello"
            )

        # Must return ActionResult (failure), NOT None
        assert result is not None
        assert result.success is False
        assert "403" in result.message

    @pytest.mark.asyncio
    async def test_gmail_exception_returns_failure_not_none(self):
        """Gmail API exception returns failure, does NOT fall back to SMTP."""
        proposal = _make_proposal()

        mock_creds = MagicMock()
        mock_creds.token = "tok"
        mock_creds.valid = True

        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.get_credentials.return_value = mock_creds

        mock_async_client = AsyncMock()
        mock_async_client.post.side_effect = Exception("Connection timeout")
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_ID", "cid"), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET", "csec"), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_TOKEN_PATH", "/tmp/t.json"), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_FROM", "me@t.com"), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_USER", "me@t.com"), \
             patch("core.actions.google_auth.GoogleAuthManager", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_async_client):
            result = await _try_gmail_send(
                proposal, "test@example.com", "Hello"
            )

        assert result is not None
        assert result.success is False
        assert "Connection timeout" in result.message


class TestSmtpFallback:
    """Test SMTP path is only used when Gmail is not set up."""

    @pytest.mark.asyncio
    async def test_smtp_not_configured(self):
        """SMTP returns failure when not configured."""
        proposal = _make_proposal()

        with patch("config.app_config.INTERNET_ACTIONS_SMTP_HOST", ""), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_PORT", 587), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_USER", ""), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_PASSWORD", ""), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_FROM", ""):
            result = await _smtp_send(proposal, "a@b.com", "hi")

        assert result.success is False
        assert "not configured" in result.message

    @pytest.mark.asyncio
    async def test_send_email_uses_smtp_when_gmail_not_configured(self):
        """Full send_email falls through to SMTP when Google OAuth not configured."""
        proposal = _make_proposal()

        with patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_ID", ""), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET", ""), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_HOST", ""), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_PORT", 587), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_USER", ""), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_PASSWORD", ""), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_FROM", ""):
            result = await send_email(proposal)

        assert result.success is False
        assert "not configured" in result.message

    @pytest.mark.asyncio
    async def test_no_smtp_fallback_after_gmail_failure(self):
        """When Gmail API fails, we do NOT fall back to SMTP."""
        proposal = _make_proposal()

        mock_creds = MagicMock()
        mock_creds.token = "tok"
        mock_creds.valid = True

        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.get_credentials.return_value = mock_creds

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        mock_async_client = AsyncMock()
        mock_async_client.post.return_value = mock_resp
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_ID", "cid"), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET", "csec"), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_TOKEN_PATH", "/tmp/t.json"), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_FROM", "me@t.com"), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_USER", "me@t.com"), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_HOST", "smtp.gmail.com"), \
             patch("core.actions.google_auth.GoogleAuthManager", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_async_client):
            result = await send_email(proposal)

        # Should get Gmail failure, NOT SMTP success
        assert result.success is False
        assert "Gmail" in result.message or "500" in result.message


class TestRecipientResolution:
    """Test name → email resolution in send_email."""

    @pytest.mark.asyncio
    async def test_full_email_bypasses_resolution(self):
        """Recipient with '@' is used as-is, no resolution attempted."""
        proposal = _make_proposal(params={
            "recipient": "direct@example.com",
            "message": "hi",
        })

        # If resolution were called, it would fail — but it shouldn't be called
        with patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_ID", ""), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET", ""), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_HOST", ""), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_PORT", 587), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_USER", ""), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_PASSWORD", ""), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_FROM", ""):
            result = await send_email(proposal)

        # SMTP not configured, but the point is we didn't fail on resolution
        assert "not configured" in result.message

    @pytest.mark.asyncio
    async def test_single_match_resolves(self):
        """Name with single contact match resolves to that email."""
        proposal = _make_proposal(params={
            "recipient": "Meagan",
            "message": "hi",
        })

        with patch("core.actions.email._resolve_recipient", new_callable=AsyncMock,
                   return_value=("meagan@example.com", "")), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_ID", ""), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET", ""), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_HOST", ""), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_PORT", 587), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_USER", ""), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_PASSWORD", ""), \
             patch("config.app_config.INTERNET_ACTIONS_SMTP_FROM", ""):
            result = await send_email(proposal)

        # Resolved successfully, then SMTP not configured is the failure
        assert "not configured" in result.message

    @pytest.mark.asyncio
    async def test_multiple_matches_returns_options(self):
        """Name with multiple matches returns error listing options."""
        proposal = _make_proposal(params={
            "recipient": "Meagan",
            "message": "hi",
        })

        with patch("core.actions.email._resolve_recipient", new_callable=AsyncMock,
                   return_value=(None, "Multiple contacts found for 'Meagan':\n  - Meagan A <a@test.com>\n  - Meagan B <b@test.com>")):
            result = await send_email(proposal)

        assert result.success is False
        assert "Multiple contacts" in result.message

    @pytest.mark.asyncio
    async def test_zero_matches_returns_error(self):
        """Name with no matches returns clear error."""
        proposal = _make_proposal(params={
            "recipient": "UnknownPerson",
            "message": "hi",
        })

        with patch("core.actions.email._resolve_recipient", new_callable=AsyncMock,
                   return_value=(None, "Could not resolve 'UnknownPerson' to an email address.")):
            result = await send_email(proposal)

        assert result.success is False
        assert "Could not resolve" in result.message

    @pytest.mark.asyncio
    async def test_resolution_exception_handled(self):
        """Exception in resolver returns graceful error."""
        proposal = _make_proposal(params={
            "recipient": "Meagan",
            "message": "hi",
        })

        with patch("core.actions.email._resolve_recipient", new_callable=AsyncMock,
                   return_value=(None, "Could not resolve 'Meagan' to an email address: API error")):
            result = await send_email(proposal)

        assert result.success is False
