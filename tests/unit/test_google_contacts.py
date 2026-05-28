"""Tests for core.actions.google_contacts — Google People API contact lookup."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from core.actions.google_contacts import (
    search_contacts,
    search_other_contacts,
    resolve_contact,
    clear_cache,
    _parse_contacts_response,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear module cache before each test."""
    clear_cache()
    yield
    clear_cache()


def _mock_auth(has_scope_val=True, creds_token="test-token"):
    """Create a mock GoogleAuthManager."""
    mock = MagicMock()
    mock.is_authenticated = True
    mock.has_scope.return_value = has_scope_val
    mock_creds = MagicMock()
    mock_creds.token = creds_token
    mock.get_credentials.return_value = mock_creds
    return mock


def _mock_httpx_client(response_json, status_code=200):
    """Create a mock httpx.AsyncClient with preset response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = response_json

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# Sample People API responses
SINGLE_CONTACT_RESPONSE = {
    "results": [
        {
            "person": {
                "names": [{"displayName": "Meagan Smith"}],
                "emailAddresses": [{"value": "meagan@example.com"}],
            }
        }
    ]
}

MULTIPLE_CONTACTS_RESPONSE = {
    "results": [
        {
            "person": {
                "names": [{"displayName": "Meagan Smith"}],
                "emailAddresses": [{"value": "meagan@example.com"}],
            }
        },
        {
            "person": {
                "names": [{"displayName": "Meagan Jones"}],
                "emailAddresses": [{"value": "meagan.jones@work.com"}],
            }
        },
    ]
}

NO_EMAIL_CONTACT_RESPONSE = {
    "results": [
        {
            "person": {
                "names": [{"displayName": "Meagan NoEmail"}],
                "emailAddresses": [],
            }
        }
    ]
}

EMPTY_RESPONSE = {"results": []}


class TestContactsConfig:
    """Test configuration gating for search_contacts."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self):
        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", False):
            result = await search_contacts("Meagan")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_auth(self):
        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=None):
            result = await search_contacts("Meagan")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_not_authenticated(self):
        mock_auth = MagicMock()
        mock_auth.is_authenticated = False

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            result = await search_contacts("Meagan")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_scope_missing(self):
        mock_auth = _mock_auth(has_scope_val=False)

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            result = await search_contacts("Meagan")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_refresh_fails(self):
        mock_auth = _mock_auth()
        mock_auth.get_credentials.return_value = None

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            result = await search_contacts("Meagan")
        assert result == []


class TestOtherContactsConfig:
    """Test configuration gating for search_other_contacts."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self):
        with patch("config.app_config.GOOGLE_OTHER_CONTACTS_ENABLED", False):
            result = await search_other_contacts("Meagan")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_auth(self):
        with patch("config.app_config.GOOGLE_OTHER_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=None):
            result = await search_other_contacts("Meagan")
        assert result == []


class TestSearchContacts:
    """Test saved contacts search."""

    @pytest.mark.asyncio
    async def test_single_contact(self):
        mock_auth = _mock_auth()
        mock_client = _mock_httpx_client(SINGLE_CONTACT_RESPONSE)

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await search_contacts("Meagan")

        assert len(result) == 1
        assert result[0]["name"] == "Meagan Smith"
        assert result[0]["email"] == "meagan@example.com"
        assert result[0]["source"] == "contacts"

    @pytest.mark.asyncio
    async def test_multiple_contacts(self):
        mock_auth = _mock_auth()
        mock_client = _mock_httpx_client(MULTIPLE_CONTACTS_RESPONSE)

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await search_contacts("Meagan")

        assert len(result) == 2
        assert result[0]["email"] == "meagan@example.com"
        assert result[1]["email"] == "meagan.jones@work.com"

    @pytest.mark.asyncio
    async def test_skips_contact_without_email(self):
        mock_auth = _mock_auth()
        mock_client = _mock_httpx_client(NO_EMAIL_CONTACT_RESPONSE)

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await search_contacts("Meagan")

        assert result == []

    @pytest.mark.asyncio
    async def test_empty_results(self):
        mock_auth = _mock_auth()
        mock_client = _mock_httpx_client(EMPTY_RESPONSE)

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await search_contacts("Nobody")

        assert result == []

    @pytest.mark.asyncio
    async def test_api_error_returns_empty(self):
        mock_auth = _mock_auth()
        mock_client = _mock_httpx_client({}, status_code=403)

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await search_contacts("Meagan")

        assert result == []

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self):
        mock_auth = _mock_auth()
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Network error")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await search_contacts("Meagan")

        assert result == []


class TestSearchOtherContacts:
    """Test other contacts search."""

    @pytest.mark.asyncio
    async def test_other_contacts_source_label(self):
        mock_auth = _mock_auth()
        mock_client = _mock_httpx_client(SINGLE_CONTACT_RESPONSE)

        with patch("config.app_config.GOOGLE_OTHER_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await search_other_contacts("Meagan")

        assert len(result) == 1
        assert result[0]["source"] == "other_contacts"


class TestResolveContact:
    """Test combined contact resolution."""

    @pytest.mark.asyncio
    async def test_saved_contacts_first(self):
        """Saved contacts appear before other contacts."""
        saved_resp = {
            "results": [
                {"person": {
                    "names": [{"displayName": "Meagan Saved"}],
                    "emailAddresses": [{"value": "meagan@saved.com"}],
                }}
            ]
        }
        other_resp = {
            "results": [
                {"person": {
                    "names": [{"displayName": "Meagan Other"}],
                    "emailAddresses": [{"value": "meagan@other.com"}],
                }}
            ]
        }

        mock_auth = _mock_auth()

        # First call: saved contacts, second call: warmup, third: other contacts
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        saved_mock_resp = MagicMock()
        saved_mock_resp.status_code = 200
        saved_mock_resp.json.return_value = saved_resp

        other_mock_resp = MagicMock()
        other_mock_resp.status_code = 200
        other_mock_resp.json.return_value = other_resp

        # The get calls: warmup, saved, warmup(cached), other
        mock_client.get.side_effect = [saved_mock_resp, saved_mock_resp, other_mock_resp]

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("config.app_config.GOOGLE_OTHER_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await resolve_contact("Meagan")

        assert len(result) == 2
        assert result[0]["source"] == "contacts"
        assert result[0]["email"] == "meagan@saved.com"
        assert result[1]["source"] == "other_contacts"

    @pytest.mark.asyncio
    async def test_deduplicates_by_email(self):
        """Same email from saved and other contacts is deduplicated."""
        same_resp = {
            "results": [
                {"person": {
                    "names": [{"displayName": "Meagan"}],
                    "emailAddresses": [{"value": "meagan@same.com"}],
                }}
            ]
        }
        mock_auth = _mock_auth()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = same_resp
        mock_client.get.return_value = mock_resp

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("config.app_config.GOOGLE_OTHER_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await resolve_contact("Meagan")

        # Should be deduplicated to 1
        assert len(result) == 1
        assert result[0]["email"] == "meagan@same.com"

    @pytest.mark.asyncio
    async def test_other_contacts_when_saved_empty(self):
        """Falls back to other contacts when saved contacts return nothing."""
        mock_auth = _mock_auth()

        empty_resp = MagicMock()
        empty_resp.status_code = 200
        empty_resp.json.return_value = EMPTY_RESPONSE

        other_resp = MagicMock()
        other_resp.status_code = 200
        other_resp.json.return_value = SINGLE_CONTACT_RESPONSE

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get.side_effect = [empty_resp, empty_resp, other_resp]

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("config.app_config.GOOGLE_OTHER_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await resolve_contact("Meagan")

        assert len(result) == 1
        assert result[0]["source"] == "other_contacts"

    @pytest.mark.asyncio
    async def test_zero_matches_returns_empty(self):
        """Returns empty list when no matches anywhere."""
        mock_auth = _mock_auth()

        empty_resp = MagicMock()
        empty_resp.status_code = 200
        empty_resp.json.return_value = EMPTY_RESPONSE

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get.return_value = empty_resp

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("config.app_config.GOOGLE_OTHER_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await resolve_contact("Nobody")

        assert result == []


class TestCache:
    """Test caching behavior."""

    @pytest.mark.asyncio
    async def test_cache_prevents_duplicate_api_calls(self):
        mock_auth = _mock_auth()
        mock_client = _mock_httpx_client(SINGLE_CONTACT_RESPONSE)

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result1 = await search_contacts("Meagan")
            result2 = await search_contacts("Meagan")

        assert result1 == result2
        # warmup + first real call = 2 calls on first invocation, 0 on second (cached)
        assert mock_client.get.call_count == 2  # warmup + search

    @pytest.mark.asyncio
    async def test_clear_cache_forces_fresh_fetch(self):
        mock_auth = _mock_auth()
        mock_client = _mock_httpx_client(SINGLE_CONTACT_RESPONSE)

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            await search_contacts("Meagan")
            clear_cache()
            await search_contacts("Meagan")

        # warmup + search + warmup + search = 4 calls
        assert mock_client.get.call_count == 4

    @pytest.mark.asyncio
    async def test_different_queries_cached_separately(self):
        mock_auth = _mock_auth()
        mock_client = _mock_httpx_client(SINGLE_CONTACT_RESPONSE)

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            await search_contacts("Meagan")
            await search_contacts("John")

        # warmup(cached after first) + search_meagan + search_john = 3 calls
        assert mock_client.get.call_count == 3


class TestParseResponse:
    """Test response parsing."""

    def test_parses_full_contact(self):
        result = _parse_contacts_response(SINGLE_CONTACT_RESPONSE, "contacts")
        assert len(result) == 1
        assert result[0] == {"name": "Meagan Smith", "email": "meagan@example.com", "source": "contacts"}

    def test_skips_contact_without_email(self):
        result = _parse_contacts_response(NO_EMAIL_CONTACT_RESPONSE, "contacts")
        assert result == []

    def test_uses_email_as_name_when_name_missing(self):
        response = {
            "results": [
                {"person": {
                    "names": [],
                    "emailAddresses": [{"value": "anon@test.com"}],
                }}
            ]
        }
        result = _parse_contacts_response(response, "contacts")
        assert len(result) == 1
        assert result[0]["name"] == "anon@test.com"

    def test_empty_results(self):
        result = _parse_contacts_response(EMPTY_RESPONSE, "contacts")
        assert result == []

    def test_missing_results_key(self):
        result = _parse_contacts_response({}, "contacts")
        assert result == []

    def test_contact_with_empty_email_value(self):
        response = {
            "results": [
                {"person": {
                    "names": [{"displayName": "No Email"}],
                    "emailAddresses": [{"value": ""}],
                }}
            ]
        }
        result = _parse_contacts_response(response, "contacts")
        assert result == []
