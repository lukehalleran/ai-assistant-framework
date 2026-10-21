"""Calendar update/delete resolve credentials off the event loop (BC-41).

Audit 2026-10-20 (class BC-41): ``_prereq_error`` calls
``GoogleAuth.get_credentials`` (a token refresh is a network round trip) and was
invoked inline from the async update/delete actions -- the one remaining
google_auth caller on the loop.  Drives the deployed actions; only the auth
object is a fake that records the calling thread.
"""
import threading
from types import SimpleNamespace

import pytest

import core.actions.google_calendar_modify as mod
from core.actions.types import ActionProposal, ActionType


@pytest.mark.asyncio
@pytest.mark.parametrize("fn, atype", [
    ("update_calendar_event", ActionType.CALENDAR_UPDATE_EVENT),
    ("delete_calendar_event", ActionType.CALENDAR_DELETE_EVENT),
])
async def test_get_credentials_runs_off_loop(monkeypatch, fn, atype):
    seen = []

    class _Auth:
        is_authenticated = True

        def has_scope(self, scope):
            return True

        def get_credentials(self):
            seen.append(threading.get_ident())
            return SimpleNamespace(token="synthetic-token")

    async def _no_match(*args, **kwargs):
        return None, "no match"

    monkeypatch.setattr("config.app_config.GOOGLE_CALENDAR_ENABLED", True)
    monkeypatch.setattr(mod.google_auth, "get_google_auth", lambda: _Auth())
    monkeypatch.setattr(mod, "_resolve_event", _no_match)
    proposal = ActionProposal(
        action_type=atype,
        params={"summary": "Pottery", "date": "2026-11-02", "new_summary": "Kiln"},
    )
    res = await getattr(mod, fn)(proposal)
    assert res.success is False and "no match" in res.message
    assert seen and seen[0] != threading.get_ident()
