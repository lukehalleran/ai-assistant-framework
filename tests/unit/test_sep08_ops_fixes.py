"""tests/unit/test_sep08_ops_fixes.py

B6 (2026-09-08, operational fixes) unit tests:
1. N3 — enhanced-path content-channel thinking-only recovery: when the
   entire streamed response was a `<thinking>...</thinking>` block with
   nothing after it, ONE recovery call via
   `model_manager.generate_once(disable_reasoning=True)` runs before giving
   up (the same contract as `ResponseGenerator._recover_reasoning_only`).
2. Google Contacts 403 handling: the API error's `status`/`message` are
   captured and surfaced through `get_runtime_action_health()` so the owner
   can tell "API not enabled" from "insufficient scope" without reading logs.

No LLM/network/store access — everything mocked. Synthetic data only.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.actions.google_contacts import clear_cache, get_last_error, search_contacts
from core.actions.registry import get_runtime_action_health
from tests.unit.test_google_contacts import _mock_auth, _mock_httpx_client
from tests.unit.test_handle_submit import (
    _debug_record,
    _final_content,
    _make_orchestrator,
    _run_submit,
)

# ---------------------------------------------------------------------------
# N3 — content-channel thinking-only recovery
# ---------------------------------------------------------------------------

# A closed thinking block wrapping an EMPTY nested <reasoning> tag: real
# leak shape (module docstring, 2026-07-03 — some OpenRouter models dump
# chain-of-thought in literal <reasoning> tags) that satisfies
# ResponseParser.is_empty_thinking_shell() so the EARLIER "stuck thinking"
# per-chunk recovery (gui/handlers.py, unrelated to this fix) finds nothing
# tag-stripped to recover and leaves final_output untouched, letting this
# batch's new POST-stream recovery (parse_thinking_block finds a non-empty
# thinking part and an empty final answer) actually run.
_THINKING_ONLY_CHUNK = "<thinking><reasoning></reasoning></thinking>"


class TestThinkingOnlyStreamRecovery:
    @pytest.mark.asyncio
    async def test_recovery_success_uses_recovered_answer_and_stores_once(self):
        orch = _make_orchestrator(streaming_chunks=[_THINKING_ONLY_CHUNK])
        orch.model_manager.generate_once = AsyncMock(return_value="Recovered answer")

        with patch("gui.handlers._dispatch_storage") as mock_dispatch:
            results = await _run_submit("please walk me through the plan", orch)

        content = _final_content(results)
        assert content == "Recovered answer"
        orch.model_manager.generate_once.assert_awaited_once()
        _, call_kwargs = orch.model_manager.generate_once.call_args
        assert call_kwargs.get("disable_reasoning") is True
        assert mock_dispatch.call_count == 1

    @pytest.mark.asyncio
    async def test_recovery_failure_shows_empty_notice_and_skips_storage(self):
        orch = _make_orchestrator(streaming_chunks=[_THINKING_ONLY_CHUNK])
        orch.model_manager.generate_once = AsyncMock(return_value="")

        with patch("gui.handlers._dispatch_storage") as mock_dispatch:
            results = await _run_submit("please walk me through the plan", orch)

        content = _final_content(results)
        assert "Empty Response" in content

        debug = _debug_record(results)
        assert debug is not None
        assert debug.get("answer_call") == "empty"

        assert mock_dispatch.call_count == 0

    @pytest.mark.asyncio
    async def test_recovery_call_exception_is_non_fatal(self):
        """generate_once raising must degrade to the empty-notice path, not
        crash the turn."""
        orch = _make_orchestrator(streaming_chunks=[_THINKING_ONLY_CHUNK])
        orch.model_manager.generate_once = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("gui.handlers._dispatch_storage") as mock_dispatch:
            results = await _run_submit("please walk me through the plan", orch)

        content = _final_content(results)
        assert "Empty Response" in content
        assert mock_dispatch.call_count == 0


# ---------------------------------------------------------------------------
# Google Contacts 403 handling
# ---------------------------------------------------------------------------

_PERMISSION_DENIED_BODY = {
    "error": {
        "code": 403,
        "message": (
            "People API has not been used in project 123456789 before or it "
            "is disabled. Enable it by visiting "
            "https://console.developers.google.com/apis/api/people.googleapis.com/overview"
        ),
        "status": "PERMISSION_DENIED",
    }
}


@pytest.fixture(autouse=True)
def _clear_contacts_state():
    clear_cache()
    yield
    clear_cache()


class TestContactsErrorDetail:
    @pytest.mark.asyncio
    async def test_403_captures_status_and_message(self):
        mock_auth = _mock_auth()
        mock_client = _mock_httpx_client(_PERMISSION_DENIED_BODY, status_code=403)

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await search_contacts("Meagan")

        assert result == []
        err = get_last_error()
        assert err is not None
        assert "403" in err
        assert "PERMISSION_DENIED" in err
        assert "People API has not been used" in err

    @pytest.mark.asyncio
    async def test_message_is_truncated(self):
        long_message = "X" * 500
        body = {"error": {"status": "PERMISSION_DENIED", "message": long_message}}
        mock_auth = _mock_auth()
        mock_client = _mock_httpx_client(body, status_code=403)

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            await search_contacts("Meagan")

        err = get_last_error()
        assert err is not None
        # 160-char message cap keeps the health/log line readable.
        assert err.count("X") <= 160

    @pytest.mark.asyncio
    async def test_unparseable_body_degrades_gracefully(self):
        mock_auth = _mock_auth()
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.json.side_effect = ValueError("not json")
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await search_contacts("Meagan")

        assert result == []
        assert get_last_error() is not None  # still records SOMETHING, never raises

    @pytest.mark.asyncio
    async def test_success_clears_previous_error(self):
        mock_auth = _mock_auth()
        bad_client = _mock_httpx_client(_PERMISSION_DENIED_BODY, status_code=403)
        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=bad_client):
            await search_contacts("Meagan")
        assert get_last_error() is not None

        good_client = _mock_httpx_client({"results": []}, status_code=200)
        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=good_client):
            await search_contacts("Meagan")
        assert get_last_error() is None

    def test_no_error_yet_get_last_error_is_none(self):
        assert get_last_error() is None


class TestRuntimeActionHealthContacts:
    def test_health_silent_when_no_contacts_error(self):
        with patch("config.app_config.INTERNET_ACTIONS_ENABLED", True), \
             patch("config.app_config.GOOGLE_CALENDAR_ENABLED", False):
            health = get_runtime_action_health()
        assert "lookup_contact" not in health

    @pytest.mark.asyncio
    async def test_health_shows_degraded_after_403(self):
        mock_auth = _mock_auth()
        mock_client = _mock_httpx_client(_PERMISSION_DENIED_BODY, status_code=403)
        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth), \
             patch("httpx.AsyncClient", return_value=mock_client):
            await search_contacts("Meagan")

        with patch("config.app_config.INTERNET_ACTIONS_ENABLED", True), \
             patch("config.app_config.GOOGLE_CALENDAR_ENABLED", False):
            health = get_runtime_action_health()

        assert "lookup_contact backend: DEGRADED" in health
        assert "PERMISSION_DENIED" in health
