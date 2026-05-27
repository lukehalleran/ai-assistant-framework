"""Tests for Google Calendar event creation."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from core.actions.google_calendar_create import create_calendar_event
from core.actions.types import ActionProposal, ActionType, CONFIRMATION_REQUIRED


def _make_proposal(**overrides) -> ActionProposal:
    """Create a minimal calendar event proposal."""
    defaults = {
        "action_type": "calendar_create_event",
        "params": {
            "summary": "Team Meeting",
            "start_time": "2026-05-28T14:00:00-05:00",
            "end_time": "2026-05-28T15:00:00-05:00",
            "time_zone": "America/Chicago",
            "description": "Weekly sync",
            "calendar_id": "primary",
        },
        "summary": "Create calendar event: Team Meeting",
        "reasoning": "User asked to schedule a meeting",
    }
    defaults.update(overrides)
    return ActionProposal(**defaults)


class TestConfirmationRequired:
    """Calendar create must go through user confirmation."""

    def test_calendar_create_in_confirmation_required(self):
        assert ActionType.CALENDAR_CREATE_EVENT in CONFIRMATION_REQUIRED


class TestCalendarCreateValidation:
    """Test param validation."""

    @pytest.mark.asyncio
    async def test_disabled_returns_failure(self):
        proposal = _make_proposal()
        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", False):
            result = await create_calendar_event(proposal)
        assert result.success is False
        assert "not enabled" in result.message

    @pytest.mark.asyncio
    async def test_no_auth_configured(self):
        proposal = _make_proposal()
        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=None):
            result = await create_calendar_event(proposal)
        assert result.success is False
        assert "not configured" in result.message

    @pytest.mark.asyncio
    async def test_not_authenticated(self):
        proposal = _make_proposal()
        mock_auth = MagicMock()
        mock_auth.is_authenticated = False

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            result = await create_calendar_event(proposal)
        assert result.success is False
        assert "not authenticated" in result.message

    @pytest.mark.asyncio
    async def test_missing_write_scope(self):
        proposal = _make_proposal()
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.has_scope.return_value = False

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            result = await create_calendar_event(proposal)
        assert result.success is False
        assert "write scope not granted" in result.message
        assert "python -m core.actions.google_auth" in result.message

    @pytest.mark.asyncio
    async def test_missing_summary(self):
        proposal = _make_proposal(params={
            "start_time": "2026-05-28T14:00:00-05:00",
            "end_time": "2026-05-28T15:00:00-05:00",
        })
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.has_scope.return_value = True
        mock_auth.get_credentials.return_value = MagicMock(token="tok")

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            result = await create_calendar_event(proposal)
        assert result.success is False
        assert "summary" in result.message

    @pytest.mark.asyncio
    async def test_missing_times(self):
        proposal = _make_proposal(params={"summary": "Test"})
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.has_scope.return_value = True
        mock_auth.get_credentials.return_value = MagicMock(token="tok")

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            result = await create_calendar_event(proposal)
        assert result.success is False
        assert "start_time" in result.message


class TestCalendarCreateExecution:
    """Test API call and response handling."""

    def _setup_auth(self):
        mock_creds = MagicMock()
        mock_creds.token = "test-access-token"
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.has_scope.return_value = True
        mock_auth.get_credentials.return_value = mock_creds
        return mock_auth

    @pytest.mark.asyncio
    async def test_success(self):
        proposal = _make_proposal()
        mock_auth = self._setup_auth()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "event123",
            "htmlLink": "https://calendar.google.com/event?eid=event123",
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await create_calendar_event(proposal)

        assert result.success is True
        assert "Team Meeting" in result.message
        assert "calendar.google.com" in result.message

    @pytest.mark.asyncio
    async def test_api_error(self):
        proposal = _make_proposal()
        mock_auth = self._setup_auth()

        mock_resp = MagicMock()
        mock_resp.status_code = 403

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await create_calendar_event(proposal)

        assert result.success is False
        assert "403" in result.message

    @pytest.mark.asyncio
    async def test_exception(self):
        proposal = _make_proposal()
        mock_auth = self._setup_auth()

        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("Connection refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await create_calendar_event(proposal)

        assert result.success is False
        assert "Connection refused" in result.message

    @pytest.mark.asyncio
    async def test_optional_fields_in_body(self):
        """Description and location included in API call when provided."""
        proposal = _make_proposal(params={
            "summary": "Dentist",
            "start_time": "2026-05-29T14:00:00-05:00",
            "end_time": "2026-05-29T15:00:00-05:00",
            "description": "Annual checkup",
            "location": "123 Main St",
        })
        mock_auth = self._setup_auth()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "e1", "htmlLink": ""}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await create_calendar_event(proposal)

        assert result.success is True
        # Verify the body sent to the API
        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["description"] == "Annual checkup"
        assert body["location"] == "123 Main St"

    @pytest.mark.asyncio
    async def test_token_refresh_failure(self):
        proposal = _make_proposal()
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.has_scope.return_value = True
        mock_auth.get_credentials.return_value = None  # refresh failed

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            result = await create_calendar_event(proposal)

        assert result.success is False
        assert "refresh failed" in result.message
