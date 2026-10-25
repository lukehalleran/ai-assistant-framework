"""Google auth checks in the turn path must not run on the event loop.

Incident 2026-10-24 20:41 (first turn after a restart): a 5075 ms LoopStall at
google_auth._load_token, reached through fetch_upcoming_events ->
auth.is_authenticated (disk read + google-auth credential parse, synchronous,
ON the loop). Class: BC-41 (blocking call on the event loop), BC-72 (latency).

The assertion is an ORDERING one (BC-101), never a wall-clock bound: the fake
auth's ``is_authenticated`` blocks until a concurrently scheduled coroutine
signals it. If the property is evaluated on the loop thread, that coroutine
cannot run and the wait times out, so ``ran_concurrently`` is False.
"""
import asyncio
import threading

import pytest


class _BlockingAuth:
    """Fake GoogleAuthManager whose token reads block until the loop is free."""

    def __init__(self):
        self.loop_was_free = threading.Event()
        self.ran_concurrently = None
        self.scope_checked_concurrently = None

    @property
    def is_authenticated(self):
        # Bounded wait: a loop-thread call can never be released by the ticker.
        self.ran_concurrently = self.loop_was_free.wait(timeout=2.0)
        return False

    def has_scope(self, scope):
        self.scope_checked_concurrently = threading.current_thread() is not threading.main_thread()
        return False


async def _ticker(auth):
    # Needs one loop iteration to run; only possible if the loop is not blocked.
    await asyncio.sleep(0)
    auth.loop_was_free.set()


@pytest.fixture
def auth(monkeypatch):
    import core.actions.google_auth as ga

    fake = _BlockingAuth()
    monkeypatch.setattr(ga, "get_google_auth", lambda: fake)
    return fake


@pytest.mark.asyncio
async def test_fetch_upcoming_events_auth_check_is_off_loop(auth, monkeypatch):
    import config.app_config as app_config
    import core.actions.google_calendar as gc

    monkeypatch.setattr(app_config, "GOOGLE_CALENDAR_ENABLED", True)
    monkeypatch.setattr(gc, "_cache", None)
    result, _ = await asyncio.gather(gc.fetch_upcoming_events(), _ticker(auth))
    assert result == []
    assert auth.ran_concurrently is True


@pytest.mark.asyncio
async def test_search_other_contacts_auth_check_is_off_loop(auth):
    import core.actions.google_contacts as contacts

    contacts.clear_cache()
    result, _ = await asyncio.gather(
        contacts.search_other_contacts("Rowan"), _ticker(auth))
    assert result == []
    assert auth.ran_concurrently is True


@pytest.mark.asyncio
async def test_search_contacts_scope_check_is_off_loop(auth, monkeypatch):
    import core.actions.google_contacts as contacts

    contacts.clear_cache()
    import core.actions.google_auth as ga

    class _Authed(_BlockingAuth):
        is_authenticated = True  # reach has_scope

    authed = _Authed()
    monkeypatch.setattr(ga, "get_google_auth", lambda: authed)
    await contacts.search_contacts("Tamsin")
    auth = authed
    assert auth.scope_checked_concurrently is True


@pytest.mark.asyncio
async def test_search_gmail_contacts_auth_check_is_off_loop(auth, monkeypatch):
    import config.app_config as app_config
    import core.actions.gmail_search as gs

    monkeypatch.setattr(app_config, "GOOGLE_GMAIL_SEARCH_ENABLED", True)
    gs.clear_cache()
    result, _ = await asyncio.gather(gs.search_gmail_contacts("Rowan"), _ticker(auth))
    assert result == []
    assert auth.ran_concurrently is True
