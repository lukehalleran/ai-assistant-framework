"""Tests for core.actions.google_calendar — Google Calendar event fetching."""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from core.actions.google_calendar import fetch_upcoming_events, clear_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear module cache before each test."""
    clear_cache()
    yield
    clear_cache()


class TestFetchConfig:
    """Test configuration gating."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self):
        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", False), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_ID", "cid"), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET", "csec"), \
             patch("config.app_config.INTERNET_ACTIONS_GOOGLE_TOKEN_PATH", "/tmp/t.json"):
            result = await fetch_upcoming_events()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_client_id(self):
        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=None):
            result = await fetch_upcoming_events()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_not_authenticated(self):
        mock_auth = MagicMock()
        mock_auth.is_authenticated = False

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            result = await fetch_upcoming_events()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_refresh_fails(self):
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.get_credentials.return_value = None

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            result = await fetch_upcoming_events()
        assert result == []


class TestFetchEvents:
    """Test event fetching and parsing."""

    def _mock_calendar_response(self, items):
        """Build a mock httpx response with calendar items."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"items": items}
        return mock_resp

    def _setup_auth_mock(self):
        """Return a mock auth manager that's authenticated."""
        mock_creds = MagicMock()
        mock_creds.token = "test-token"
        mock_creds.valid = True

        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.get_credentials.return_value = mock_creds
        return mock_auth, mock_creds

    @pytest.mark.asyncio
    async def test_parses_timed_event(self):
        mock_auth, _ = self._setup_auth_mock()
        resp = self._mock_calendar_response([
            {
                "summary": "Team Meeting",
                "start": {"dateTime": "2026-05-27T14:00:00-04:00"},
                "end": {"dateTime": "2026-05-27T15:00:00-04:00"},
                "location": "Conference Room A",
            }
        ])

        mock_client = AsyncMock()
        mock_client.get.return_value = resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await fetch_upcoming_events()

        assert len(result) == 1
        evt = result[0]
        assert evt["summary"] == "Team Meeting"
        assert evt["start"] == "2026-05-27T14:00:00-04:00"
        assert evt["end"] == "2026-05-27T15:00:00-04:00"
        assert evt["all_day"] is False
        assert evt["location"] == "Conference Room A"

    @pytest.mark.asyncio
    async def test_parses_all_day_event(self):
        mock_auth, _ = self._setup_auth_mock()
        resp = self._mock_calendar_response([
            {
                "summary": "Holiday",
                "start": {"date": "2026-05-28"},
                "end": {"date": "2026-05-29"},
            }
        ])

        mock_client = AsyncMock()
        mock_client.get.return_value = resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await fetch_upcoming_events()

        assert len(result) == 1
        evt = result[0]
        assert evt["all_day"] is True
        assert evt["start"] == "2026-05-28"
        assert evt["location"] == ""

    @pytest.mark.asyncio
    async def test_missing_summary_defaults_to_untitled(self):
        mock_auth, _ = self._setup_auth_mock()
        resp = self._mock_calendar_response([
            {
                "start": {"dateTime": "2026-05-27T10:00:00Z"},
                "end": {"dateTime": "2026-05-27T11:00:00Z"},
            }
        ])

        mock_client = AsyncMock()
        mock_client.get.return_value = resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await fetch_upcoming_events()

        assert result[0]["summary"] == "Untitled"

    @pytest.mark.asyncio
    async def test_api_error_returns_empty(self):
        mock_auth, _ = self._setup_auth_mock()
        mock_resp = MagicMock()
        mock_resp.status_code = 403

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await fetch_upcoming_events()

        assert result == []

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self):
        mock_auth, _ = self._setup_auth_mock()

        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Network error")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await fetch_upcoming_events()

        assert result == []

    @pytest.mark.asyncio
    async def test_respects_max_events(self):
        mock_auth, _ = self._setup_auth_mock()
        items = [
            {"summary": f"Event {i}", "start": {"date": f"2026-06-{i:02d}"}, "end": {"date": f"2026-06-{i+1:02d}"}}
            for i in range(1, 11)
        ]
        resp = self._mock_calendar_response(items)

        mock_client = AsyncMock()
        mock_client.get.return_value = resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await fetch_upcoming_events(max_events=3)

        assert len(result) == 3


class TestCache:
    """Test caching behavior."""

    @pytest.mark.asyncio
    async def test_cache_prevents_duplicate_calls(self):
        mock_auth, _ = TestFetchEvents()._setup_auth_mock()
        resp = TestFetchEvents()._mock_calendar_response([
            {"summary": "Cached Event", "start": {"date": "2026-06-01"}, "end": {"date": "2026-06-02"}}
        ])

        mock_client = AsyncMock()
        mock_client.get.return_value = resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result1 = await fetch_upcoming_events()
            result2 = await fetch_upcoming_events()

        assert result1 == result2
        # httpx.AsyncClient should only be instantiated once (second call uses cache)
        assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_clear_cache_forces_fresh_fetch(self):
        mock_auth, _ = TestFetchEvents()._setup_auth_mock()
        resp = TestFetchEvents()._mock_calendar_response([
            {"summary": "Event", "start": {"date": "2026-06-01"}, "end": {"date": "2026-06-02"}}
        ])

        mock_client = AsyncMock()
        mock_client.get.return_value = resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            await fetch_upcoming_events()
            clear_cache()
            await fetch_upcoming_events()

        assert mock_client.get.call_count == 2
