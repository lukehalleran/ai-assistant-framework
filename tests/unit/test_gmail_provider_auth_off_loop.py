"""Gmail provider: the auth check runs off the event loop; main()'s launch branch table.

Incident date 2026-10-28 (wave-5 lane SM2). class: BC-41 (blocking call on the
loop: ``auth.is_authenticated`` parses the token file and imports google-auth),
BC-63 (a source-text test was the only coverage of main()'s branch decision).
"""

import asyncio
import itertools
import time

import pytest

from core.email.gmail_provider import GmailProvider


class _SlowAuth:
    """is_authenticated blocks 0.3 s (token-file parse stand-in)."""

    def __init__(self, delay=0.3, authed=False):
        self.delay = delay
        self.authed = authed
        self.calls = 0

    @property
    def is_authenticated(self):
        self.calls += 1
        time.sleep(self.delay)
        return self.authed

    def has_scope(self, scope):
        return True

    def get_credentials(self):
        return None


@pytest.fixture
def slow_auth(monkeypatch):
    auth = _SlowAuth()
    import core.actions.google_auth as ga
    monkeypatch.setattr(ga, "get_google_auth", lambda: auth)
    import config.app_config as cfg
    monkeypatch.setattr(cfg, "EMAIL_INTEGRATION_ENABLED", True)
    monkeypatch.setattr(cfg, "EMAIL_GMAIL_ENABLED", True)
    return auth


async def _max_tick_lateness(coro):
    late = []

    async def ticker():
        while True:
            t = time.monotonic()
            await asyncio.sleep(0.02)
            late.append(time.monotonic() - t - 0.02)

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0.05)
    result = await coro
    await asyncio.sleep(0.05)  # let an overdue tick record its lateness
    task.cancel()
    return result, max(late)


class TestAuthCheckOffLoop:
    @pytest.mark.asyncio
    async def test_health_keeps_loop_ticking(self, slow_auth):
        res, late = await _max_tick_lateness(GmailProvider().health())
        assert slow_auth.calls == 1
        assert res == {"available": False, "detail": "No Gmail token file"}
        assert late < 0.15, late

    @pytest.mark.asyncio
    async def test_search_keeps_loop_ticking(self, slow_auth):
        res, late = await _max_tick_lateness(GmailProvider().search("rowan"))
        assert slow_auth.calls == 1 and res == []
        assert late < 0.15, late

    @pytest.mark.asyncio
    async def test_recent_keeps_loop_ticking(self, slow_auth):
        res, late = await _max_tick_lateness(GmailProvider().recent())
        assert slow_auth.calls == 1 and res == []
        assert late < 0.15, late


class TestLaunchBranchTable:
    def test_branch_over_every_input_combination(self):
        import main
        for legacy, force, needs in itertools.product([False, True], repeat=3):
            assert main.use_legacy_gradio_path(legacy, force, needs) is (legacy or force)
