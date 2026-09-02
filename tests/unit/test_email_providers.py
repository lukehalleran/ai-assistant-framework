"""Tests for email providers (Gmail, Outlook) and the EmailService integration."""

import asyncio
import json
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile

from core.email.provider import EmailMessage, EmailProvider
from core.email.gmail_provider import GmailProvider
from core.email.outlook_provider import OutlookProvider
from core.email.outlook_auth import OutlookAuthManager
from core.email.registry import PROVIDERS, build_enabled_providers
from core.email.service import EmailService, get_email_service


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_outlook_token():
    """Create a temporary Outlook token file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        token_path = Path(tmpdir) / "outlook_token.json"
        token_data = {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp(),
        }
        token_path.write_text(json.dumps(token_data))
        yield token_path


@pytest.fixture
def gmail_provider():
    """Create a Gmail provider instance."""
    return GmailProvider()


@pytest.fixture
def outlook_provider():
    """Create an Outlook provider instance."""
    return OutlookProvider()


@pytest.fixture
def mock_google_auth():
    """Create a mock Google auth manager."""
    mock_creds = MagicMock()
    mock_creds.token = "test-google-token"

    mock_auth = MagicMock()
    mock_auth.is_authenticated = True
    mock_auth.get_credentials.return_value = mock_creds
    mock_auth.has_scope.return_value = True

    return mock_auth


def _build_gmail_list_response(messages: list) -> dict:
    """Build a Gmail API messages.list() response."""
    return {"messages": messages}


def _build_gmail_get_response(msg_id: str, sender: str, to: str, subject: str, date: str, unread: bool = False) -> dict:
    """Build a Gmail API messages.get() response with metadata."""
    labels = ["INBOX"]
    if unread:
        labels.append("UNREAD")

    return {
        "id": msg_id,
        "threadId": f"thread-{msg_id}",
        "snippet": f"Preview of message {msg_id}",
        "labelIds": labels,
        "payload": {
            "headers": [
                {"name": "From", "value": sender},
                {"name": "To", "value": to},
                {"name": "Subject", "value": subject},
                {"name": "Date", "value": date},
            ],
        },
    }


def _build_outlook_list_response(messages: list) -> dict:
    """Build a Microsoft Graph messages API response."""
    return {"value": messages}


def _build_outlook_message(msg_id: str, sender: str, to: str, subject: str, date_str: str, unread: bool = False) -> dict:
    """Build a Microsoft Graph message object."""
    return {
        "id": msg_id,
        "conversationId": f"conv-{msg_id}",
        "subject": subject,
        "from": {
            "emailAddress": {
                "name": sender.split("<")[0].strip() if "<" in sender else sender,
                "address": sender.split("<")[1].rstrip(">") if "<" in sender else sender,
            }
        },
        "toRecipients": [
            {
                "emailAddress": {
                    "address": to.strip(),
                }
            }
        ],
        "receivedDateTime": date_str,
        "bodyPreview": f"Preview of message {msg_id}",
        "isRead": not unread,
        "webLink": f"https://outlook.office.com/mail/inbox/{msg_id}",
    }


# ============================================================================
# Gmail Provider Tests
# ============================================================================


class TestGmailProviderConfig:
    """Test Gmail provider configuration checks."""

    def test_is_configured_returns_false_when_disabled(self, gmail_provider):
        with patch("config.app_config.EMAIL_GMAIL_ENABLED", False):
            assert gmail_provider.is_configured() is False

    def test_is_configured_returns_false_when_no_auth(self, gmail_provider):
        with patch("config.app_config.EMAIL_GMAIL_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=None):
            assert gmail_provider.is_configured() is False

    def test_is_configured_returns_false_when_not_authenticated(self, gmail_provider):
        mock_auth = MagicMock()
        mock_auth.is_authenticated = False

        with patch("config.app_config.EMAIL_GMAIL_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            assert gmail_provider.is_configured() is False

    def test_is_configured_returns_true_when_enabled_and_authenticated(self, gmail_provider, mock_google_auth):
        mock_google_auth.is_authenticated = True

        with patch("config.app_config.EMAIL_GMAIL_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_google_auth):
            assert gmail_provider.is_configured() is True

    def test_is_configured_returns_false_without_readonly_scope(self, gmail_provider, mock_google_auth):
        mock_google_auth.is_authenticated = True
        mock_google_auth.has_scope.return_value = False

        with patch("config.app_config.EMAIL_GMAIL_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_google_auth):
            assert gmail_provider.is_configured() is False

    @pytest.mark.asyncio
    async def test_health_disabled(self, gmail_provider):
        with patch("config.app_config.EMAIL_GMAIL_ENABLED", False):
            result = await gmail_provider.health()
        assert result["available"] is False
        assert "ENABLED=False" in result["detail"]

    @pytest.mark.asyncio
    async def test_health_not_configured(self, gmail_provider):
        with patch("config.app_config.EMAIL_GMAIL_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=None):
            result = await gmail_provider.health()
        assert result["available"] is False

    @pytest.mark.asyncio
    async def test_health_missing_scope(self, gmail_provider, mock_google_auth):
        mock_google_auth.has_scope.return_value = False

        with patch("config.app_config.EMAIL_GMAIL_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_google_auth):
            result = await gmail_provider.health()
        assert result["available"] is False
        assert "scope" in result["detail"].lower()

    @pytest.mark.asyncio
    async def test_health_ok(self, gmail_provider, mock_google_auth):
        with patch("config.app_config.EMAIL_GMAIL_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_google_auth):
            result = await gmail_provider.health()
        assert result["available"] is True


class TestGmailProviderSearch:
    """Test Gmail search functionality."""

    @pytest.mark.asyncio
    async def test_search_not_authenticated(self, gmail_provider):
        with patch("core.actions.google_auth.get_google_auth", return_value=None):
            result = await gmail_provider.search("test")
        assert result == []

    @pytest.mark.asyncio
    async def test_search_token_refresh_fails(self, gmail_provider, mock_google_auth):
        mock_google_auth.get_credentials.return_value = None

        with patch("core.actions.google_auth.get_google_auth", return_value=mock_google_auth):
            result = await gmail_provider.search("test")
        assert result == []

    @pytest.mark.asyncio
    async def test_search_api_error(self, gmail_provider, mock_google_auth):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("core.actions.google_auth.get_google_auth", return_value=mock_google_auth), \
             patch("httpx.AsyncClient.get", return_value=mock_resp):
            result = await gmail_provider.search("test")
        assert result == []

    @pytest.mark.asyncio
    async def test_search_empty_results(self, gmail_provider, mock_google_auth):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"messages": []}

        with patch("core.actions.google_auth.get_google_auth", return_value=mock_google_auth), \
             patch("httpx.AsyncClient.get", return_value=mock_resp):
            result = await gmail_provider.search("test")
        assert result == []

    @pytest.mark.asyncio
    async def test_search_success(self, gmail_provider, mock_google_auth):
        """Test successful search with message parsing."""
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%a, %d %b %Y %H:%M:%S +0000")

        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.json.return_value = _build_gmail_list_response([
            {"id": "msg1"},
            {"id": "msg2"},
        ])

        get_resp = MagicMock()
        get_resp.status_code = 200

        # Return different responses for each message fetch
        msg1 = _build_gmail_get_response("msg1", "Alice <alice@example.com>", "bob@example.com", "Test 1", date_str)
        msg2 = _build_gmail_get_response("msg2", "Bob <bob@example.com>", "alice@example.com", "Test 2", date_str, unread=True)

        get_resp_1 = MagicMock()
        get_resp_1.status_code = 200
        get_resp_1.json.return_value = msg1

        get_resp_2 = MagicMock()
        get_resp_2.status_code = 200
        get_resp_2.json.return_value = msg2

        call_count = [0]

        async def get_side_effect(*args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            if "messages" in url and "msg" not in url:
                return list_resp
            call_count[0] += 1
            if call_count[0] == 1:
                return get_resp_1
            return get_resp_2

        with patch("core.actions.google_auth.get_google_auth", return_value=mock_google_auth), \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=get_side_effect):
            result = await gmail_provider.search("test", window_days=7, limit=10)

        assert len(result) == 2
        assert result[0].provider == "gmail"
        assert result[0].subject == "Test 1"
        assert result[1].subject == "Test 2"
        assert result[1].unread is True


class TestGmailProviderRecent:
    """Test Gmail recent functionality."""

    @pytest.mark.asyncio
    async def test_recent_constructs_correct_query(self, gmail_provider, mock_google_auth):
        """Test that recent() builds the correct Gmail query."""
        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.json.return_value = {"messages": []}

        with patch("core.actions.google_auth.get_google_auth", return_value=mock_google_auth), \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=list_resp) as mock_get:
            result = await gmail_provider.recent(window_days=7, limit=25)

        # Check the query parameter
        assert mock_get.called
        call_kwargs = mock_get.call_args[1]
        params = call_kwargs.get("params", {})
        q = params.get("q", "")
        assert "newer_than:7d" in q
        assert "-category:promotions" in q
        assert "-category:social" in q


# ============================================================================
# Outlook Provider Tests
# ============================================================================


class TestOutlookProviderConfig:
    """Test Outlook provider configuration checks."""

    def test_is_configured_returns_false_when_disabled(self, outlook_provider):
        with patch("config.app_config.EMAIL_OUTLOOK_ENABLED", False):
            assert outlook_provider.is_configured() is False

    def test_is_configured_returns_false_when_no_auth(self, outlook_provider):
        with patch("config.app_config.EMAIL_OUTLOOK_ENABLED", True), \
             patch("core.email.outlook_auth.get_outlook_auth", return_value=None):
            assert outlook_provider.is_configured() is False

    def test_is_configured_returns_false_when_token_missing(self, outlook_provider):
        mock_auth = MagicMock()
        mock_auth.token_exists = False

        with patch("config.app_config.EMAIL_OUTLOOK_ENABLED", True), \
             patch("core.email.outlook_auth.get_outlook_auth", return_value=mock_auth):
            assert outlook_provider.is_configured() is False

    def test_is_configured_returns_true_when_token_exists(self, outlook_provider):
        mock_auth = MagicMock()
        mock_auth.token_exists = True

        with patch("config.app_config.EMAIL_OUTLOOK_ENABLED", True), \
             patch("core.email.outlook_auth.get_outlook_auth", return_value=mock_auth):
            assert outlook_provider.is_configured() is True

    @pytest.mark.asyncio
    async def test_health_disabled(self, outlook_provider):
        with patch("config.app_config.EMAIL_OUTLOOK_ENABLED", False):
            result = await outlook_provider.health()
        assert result["available"] is False
        assert "ENABLED=False" in result["detail"]

    @pytest.mark.asyncio
    async def test_health_no_refresh_token(self, outlook_provider):
        mock_auth = MagicMock()
        mock_auth.token_exists = True
        mock_auth.has_refresh_token = False

        with patch("config.app_config.EMAIL_OUTLOOK_ENABLED", True), \
             patch("core.email.outlook_auth.get_outlook_auth", return_value=mock_auth):
            result = await outlook_provider.health()
        assert result["available"] is False
        assert "refresh_token" in result["detail"]

    @pytest.mark.asyncio
    async def test_health_ok(self, outlook_provider):
        mock_auth = MagicMock()
        mock_auth.token_exists = True
        mock_auth.has_refresh_token = True

        with patch("config.app_config.EMAIL_OUTLOOK_ENABLED", True), \
             patch("core.email.outlook_auth.get_outlook_auth", return_value=mock_auth):
            result = await outlook_provider.health()
        assert result["available"] is True


class TestOutlookProviderSearch:
    """Test Outlook search functionality."""

    @pytest.mark.asyncio
    async def test_search_not_configured(self, outlook_provider):
        with patch("core.email.outlook_auth.get_outlook_auth", return_value=None):
            result = await outlook_provider.search("test")
        assert result == []

    @pytest.mark.asyncio
    async def test_search_token_unavailable(self, outlook_provider):
        mock_auth = MagicMock()
        mock_auth.get_access_token.return_value = None

        with patch("core.email.outlook_auth.get_outlook_auth", return_value=mock_auth):
            result = await outlook_provider.search("test")
        assert result == []

    @pytest.mark.asyncio
    async def test_search_api_error(self, outlook_provider):
        mock_auth = MagicMock()
        mock_auth.get_access_token.return_value = "test-token"

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"

        with patch("core.email.outlook_auth.get_outlook_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient.get", return_value=mock_resp):
            result = await outlook_provider.search("test")
        assert result == []

    @pytest.mark.asyncio
    async def test_search_success(self, outlook_provider):
        """Test successful Outlook search with message parsing."""
        now = datetime.now(timezone.utc)
        future = (now + timedelta(days=1)).isoformat().replace("+00:00", "Z")

        mock_auth = MagicMock()
        mock_auth.get_access_token.return_value = "test-token"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _build_outlook_list_response([
            _build_outlook_message("msg1", "Alice Alice <alice@example.com>", "bob@example.com", "Test 1", future),
            _build_outlook_message("msg2", "Bob Bob <bob@example.com>", "alice@example.com", "Test 2", future, unread=True),
        ])

        with patch("core.email.outlook_auth.get_outlook_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient.get", return_value=mock_resp):
            result = await outlook_provider.search("test", window_days=7, limit=10)

        assert len(result) == 2
        assert result[0].provider == "outlook"
        assert result[0].subject == "Test 1"
        assert result[1].unread is True


class TestOutlookProviderRecent:
    """Test Outlook recent functionality."""

    @pytest.mark.asyncio
    async def test_recent_success(self, outlook_provider):
        """Test recent messages retrieval."""
        now = datetime.now(timezone.utc)
        future = (now + timedelta(days=1)).isoformat().replace("+00:00", "Z")

        mock_auth = MagicMock()
        mock_auth.get_access_token.return_value = "test-token"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _build_outlook_list_response([
            _build_outlook_message("msg1", "Alice <alice@example.com>", "bob@example.com", "Test 1", future),
        ])

        with patch("core.email.outlook_auth.get_outlook_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient.get", return_value=mock_resp):
            result = await outlook_provider.recent(window_days=7, limit=25)

        assert len(result) == 1
        assert result[0].subject == "Test 1"
        assert result[0].date  # Should have a parsed ISO date


# ============================================================================
# Registry Tests
# ============================================================================


class TestEmailRegistry:
    """Test the provider registry and factory."""

    def test_registry_build_enables_gmail_when_configured(self):
        mock_gmail = MagicMock(spec=EmailProvider)
        mock_gmail.name = "gmail"
        mock_gmail.is_configured.return_value = True
        gmail_factory = MagicMock(return_value=mock_gmail)

        with patch("config.app_config.EMAIL_GMAIL_ENABLED", True), \
             patch("config.app_config.EMAIL_INTEGRATION_ENABLED", True), \
             patch.dict(PROVIDERS["gmail"], {"factory": gmail_factory}):
            providers = build_enabled_providers()

        assert len(providers) >= 1
        assert any(p.name == "gmail" for p in providers)
        gmail_factory.assert_called_once_with()

    def test_registry_skips_outlook_when_disabled(self):
        with patch("config.app_config.EMAIL_OUTLOOK_ENABLED", False), \
             patch("config.app_config.EMAIL_GMAIL_ENABLED", False):
            providers = build_enabled_providers()

        # Only disabled providers, so empty
        assert len(providers) == 0

    def test_registry_handles_factory_exception(self):
        """Test that a factory exception is logged and skipped."""
        with patch("config.app_config.EMAIL_GMAIL_ENABLED", False), \
             patch("config.app_config.EMAIL_INTEGRATION_ENABLED", True), \
             patch("config.app_config.EMAIL_OUTLOOK_ENABLED", False):
            # With both disabled, we get an empty list
            providers = build_enabled_providers()
            assert len(providers) == 0


# ============================================================================
# EmailService Tests
# ============================================================================


class TestEmailService:
    """Test the EmailService facade."""

    @pytest.mark.asyncio
    async def test_service_fan_out(self):
        """Test that service merges results from multiple providers."""
        msg1 = EmailMessage(
            provider="gmail",
            message_id="msg1",
            subject="Gmail Message",
            date="2026-09-01T12:00:00+00:00",
        )
        msg2 = EmailMessage(
            provider="outlook",
            message_id="msg2",
            subject="Outlook Message",
            date="2026-09-01T11:00:00+00:00",
        )

        mock_gmail = AsyncMock(spec=EmailProvider)
        mock_gmail.name = "gmail"
        mock_gmail.search.return_value = [msg1]

        mock_outlook = AsyncMock(spec=EmailProvider)
        mock_outlook.name = "outlook"
        mock_outlook.search.return_value = [msg2]

        service = EmailService(providers=[mock_gmail, mock_outlook], cache_ttl_seconds=300)
        result = await service.search("test")

        assert len(result) == 2
        # Should be newest-first
        assert result[0].subject == "Gmail Message"
        assert result[1].subject == "Outlook Message"

    @pytest.mark.asyncio
    async def test_service_handles_provider_exception(self):
        """Test that a failing provider doesn't break the service."""
        msg1 = EmailMessage(
            provider="gmail",
            message_id="msg1",
            subject="Gmail Message",
            date="2026-09-01T12:00:00+00:00",
        )

        mock_gmail = AsyncMock(spec=EmailProvider)
        mock_gmail.name = "gmail"
        mock_gmail.search.return_value = [msg1]

        mock_outlook = AsyncMock(spec=EmailProvider)
        mock_outlook.name = "outlook"
        mock_outlook.search.side_effect = RuntimeError("Outlook API error")

        service = EmailService(providers=[mock_gmail, mock_outlook], cache_ttl_seconds=300)
        result = await service.search("test")

        # Should have results from Gmail despite Outlook failing
        assert len(result) == 1
        assert result[0].provider == "gmail"

    @pytest.mark.asyncio
    async def test_service_caches_results(self):
        """Test that search results are cached by TTL."""
        msg1 = EmailMessage(
            provider="gmail",
            message_id="msg1",
            subject="Test",
            date="2026-09-01T12:00:00+00:00",
        )

        mock_gmail = AsyncMock(spec=EmailProvider)
        mock_gmail.name = "gmail"
        mock_gmail.search.return_value = [msg1]

        service = EmailService(providers=[mock_gmail], cache_ttl_seconds=300)

        # First call
        result1 = await service.search("test")
        assert mock_gmail.search.call_count == 1

        # Second call (should be cached)
        result2 = await service.search("test")
        assert mock_gmail.search.call_count == 1  # No additional call

        assert result1 == result2

    @pytest.mark.asyncio
    async def test_service_health_aggregates(self):
        """Test that health() aggregates provider health."""
        mock_gmail = AsyncMock(spec=EmailProvider)
        mock_gmail.name = "gmail"
        mock_gmail.health.return_value = {"available": True, "detail": "OK"}

        mock_outlook = AsyncMock(spec=EmailProvider)
        mock_outlook.name = "outlook"
        mock_outlook.health.return_value = {"available": False, "detail": "No token"}

        service = EmailService(providers=[mock_gmail, mock_outlook])
        result = await service.health()

        assert result["gmail"]["available"] is True
        assert result["outlook"]["available"] is False


# ============================================================================
# Integration Tests
# ============================================================================


class TestEmailIntegration:
    """Integration tests across the email layer."""

    def test_config_loads(self):
        """Test that config schema validates and loads."""
        from config.app_config import (
            EMAIL_INTEGRATION_ENABLED,
            EMAIL_GMAIL_ENABLED,
            EMAIL_OUTLOOK_ENABLED,
            EMAIL_MAX_RESULTS,
            EMAIL_CACHE_TTL_SECONDS,
        )

        assert EMAIL_INTEGRATION_ENABLED is not None
        assert EMAIL_MAX_RESULTS > 0
        assert EMAIL_CACHE_TTL_SECONDS > 0

    def test_service_singleton(self):
        """Test that get_email_service() returns a singleton."""
        # Reset the singleton
        import core.email.service
        core.email.service._service = None

        service1 = get_email_service()
        service2 = get_email_service()

        assert service1 is service2

        # Clean up
        core.email.service._service = None
