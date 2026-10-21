"""A disabled People API is remembered for the process; transient errors are not.

Incident 2026-10-18..20: every passive email fire logged two
`[GoogleContacts] API error: HTTP 403 ... People API has not been used in
project ... or it is disabled` lines, repeated within one process.

Class: BC-69 (silent operations failure: a permanent config fact re-paid per call).
"""

import asyncio
import logging
from unittest.mock import patch

import pytest

import core.actions.google_contacts as gc
from tests.unit.test_google_contacts import _mock_auth, _mock_httpx_client

DISABLED_BODY = {"error": {
    "code": 403, "status": "PERMISSION_DENIED",
    "message": "People API has not been used in project 1234 before or it is disabled.",
    "details": [{"reason": "SERVICE_DISABLED"}],
}}


@pytest.fixture(autouse=True)
def _clean():
    gc.clear_cache()
    yield
    gc.clear_cache()


def _search(client, caplog):
    with patch("core.actions.google_auth.get_google_auth", return_value=_mock_auth()), \
         patch("httpx.AsyncClient", return_value=client), \
         patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True):
        with caplog.at_level(logging.DEBUG):
            return asyncio.run(gc.search_contacts("Rowan"))


def _real_calls(client):
    # the warmup request is a separate GET; count searchContacts queries only
    return [c for c in client.get.call_args_list
            if c.kwargs.get("params", {}).get("query") == "Rowan"]


def test_disabled_api_403_is_requested_once_and_warned_once(caplog):
    client = _mock_httpx_client(DISABLED_BODY, status_code=403)
    assert _search(client, caplog) == []
    assert _search(client, caplog) == []
    assert len(_real_calls(client)) == 1
    warns = [r for r in caplog.records if r.levelno == logging.WARNING
             and "GoogleContacts" in r.getMessage()]
    assert len(warns) == 1
    assert "GOOGLE_CONTACTS_ENABLED" in warns[0].getMessage()


def test_status_alone_without_details_is_also_permanent(caplog):
    body = {"error": {"status": "PERMISSION_DENIED", "message": "denied"}}
    client = _mock_httpx_client(body, status_code=403)
    _search(client, caplog)
    _search(client, caplog)
    assert len(_real_calls(client)) == 1


def test_prose_only_403_is_not_remembered(caplog):
    body = {"error": {"message": "People API has not been used ... or it is disabled"}}
    client = _mock_httpx_client(body, status_code=403)
    _search(client, caplog)
    _search(client, caplog)
    assert len(_real_calls(client)) == 2


def test_503_is_retried_on_next_call(caplog):
    client = _mock_httpx_client({"error": {"status": "UNAVAILABLE", "message": "try later"}},
                                status_code=503)
    _search(client, caplog)
    _search(client, caplog)
    assert len(_real_calls(client)) == 2
