"""B3 (2026-10-12): POST /api/shutdown, /api/heartbeat and the packaged-mode
client watchdog -- driven through the deployed create_app and the deployed
``watchdog_tick`` (fake clock, no sleeps). class: BC-58."""

import asyncio
import os
import signal

import httpx
import pytest

from api.lifecycle import LifecycleState, watchdog_tick
from config import app_config
from tests.unit.api_launch_auth_client import (
    TEST_LAUNCH_SECRET,
    TEST_ORIGIN,
    authed_client,
    make_test_app,
)
from tests.unit.helpers_orchestrator import _make_orchestrator
from api.launch_auth import LAUNCH_TOKEN_HEADER


@pytest.fixture
def kills(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: calls.append((pid, sig)))
    return calls


def _app(inflight=False):
    app = make_test_app(_make_orchestrator(), start_background=False)
    app.state.lifecycle = LifecycleState(inflight_fn=lambda: inflight, signal_delay_s=0.01)
    return app


class TestShutdownRoute:
    @pytest.mark.asyncio
    async def test_token_required(self, kills):
        app = _app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=TEST_ORIGIN,
                                     headers={"Origin": TEST_ORIGIN}) as c:
            assert (await c.post("/api/shutdown")).status_code == 401
            bad = await c.post("/api/shutdown", headers={LAUNCH_TOKEN_HEADER: "wrong"})
            assert bad.status_code == 401
        await asyncio.sleep(0.05)
        assert kills == []

    @pytest.mark.asyncio
    async def test_non_loopback_peer_refused(self, kills):
        app = _app()
        transport = httpx.ASGITransport(app=app, client=("100.64.0.9", 4444))
        async with httpx.AsyncClient(transport=transport, base_url=TEST_ORIGIN,
                                     headers={LAUNCH_TOKEN_HEADER: TEST_LAUNCH_SECRET,
                                              "Origin": TEST_ORIGIN}) as c:
            resp = await c.post("/api/shutdown")
        assert resp.status_code == 403
        await asyncio.sleep(0.05)
        assert kills == []
        assert app.state.lifecycle.shutting_down is False

    @pytest.mark.asyncio
    async def test_accepts_and_sends_exactly_one_sigint_idempotent(self, kills):
        app = _app()
        async with authed_client(app) as c:
            first = await c.post("/api/shutdown")
            second = await c.post("/api/shutdown")
        assert first.status_code == 202 and first.json() == {"status": "shutting_down"}
        assert second.status_code == 202 and second.json()["status"] == "already_shutting_down"
        await asyncio.sleep(0.1)
        assert kills == [(os.getpid(), signal.SIGINT)]

    @pytest.mark.asyncio
    async def test_inflight_turn_is_reported_but_still_accepted(self, kills):
        app = _app(inflight=True)
        async with authed_client(app) as c:
            resp = await c.post("/api/shutdown")
        assert resp.status_code == 202
        assert resp.json() == {"status": "shutting_down", "turn_in_flight": True}
        await asyncio.sleep(0.1)
        assert len(kills) == 1


class TestHeartbeatRoute:
    @pytest.mark.asyncio
    async def test_heartbeat_requires_token_and_arms(self):
        app = _app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=TEST_ORIGIN,
                                     headers={"Origin": TEST_ORIGIN}) as c:
            assert (await c.post("/api/heartbeat")).status_code == 401
        assert app.state.lifecycle.last_heartbeat is None
        async with authed_client(app) as c:
            assert (await c.post("/api/heartbeat")).status_code == 200
        assert app.state.lifecycle.last_heartbeat is not None


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(app_config, "APP_LIFECYCLE_EXIT_WHEN_CLIENTS_GONE", True)
    monkeypatch.setattr(app_config, "APP_LIFECYCLE_CLIENT_GRACE_S", 120.0)


class TestWatchdog:
    @pytest.mark.asyncio
    async def test_not_armed_before_first_heartbeat(self, enabled, kills):
        clock = _Clock()
        st = LifecycleState(clock=clock, inflight_fn=lambda: False, signal_delay_s=0.01)
        clock.t += 10_000
        assert watchdog_tick(st) is False
        await asyncio.sleep(0.05)
        assert kills == [] and not st.shutting_down

    @pytest.mark.asyncio
    async def test_fires_once_after_grace(self, enabled, kills):
        clock = _Clock()
        st = LifecycleState(clock=clock, inflight_fn=lambda: False, signal_delay_s=0.01)
        st.heartbeat()
        clock.t += 100
        assert watchdog_tick(st) is False  # inside grace
        st.heartbeat()  # reconnect refreshes
        clock.t += 119
        assert watchdog_tick(st) is False
        clock.t += 2
        assert watchdog_tick(st) is True
        assert watchdog_tick(st) is False  # already shutting down
        await asyncio.sleep(0.05)
        assert kills == [(os.getpid(), signal.SIGINT)]

    @pytest.mark.asyncio
    async def test_never_fires_with_turn_in_flight(self, enabled, kills):
        clock = _Clock()
        st = LifecycleState(clock=clock, inflight_fn=lambda: True, signal_delay_s=0.01)
        st.heartbeat()
        clock.t += 10_000
        assert watchdog_tick(st) is False
        await asyncio.sleep(0.05)
        assert kills == []

    @pytest.mark.asyncio
    async def test_inflight_check_failure_counts_as_busy(self, enabled, kills):
        clock = _Clock()

        def boom():
            raise RuntimeError("x")

        st = LifecycleState(clock=clock, inflight_fn=boom, signal_delay_s=0.01)
        st.heartbeat()
        clock.t += 10_000
        assert watchdog_tick(st) is False

    @pytest.mark.asyncio
    async def test_disabled_in_dev_mode(self, monkeypatch, kills):
        monkeypatch.setattr(app_config, "APP_LIFECYCLE_EXIT_WHEN_CLIENTS_GONE", False)
        clock = _Clock()
        st = LifecycleState(clock=clock, inflight_fn=lambda: False, signal_delay_s=0.01)
        st.heartbeat()
        clock.t += 10_000
        assert watchdog_tick(st) is False
        await asyncio.sleep(0.05)
        assert kills == []


def test_config_default_is_auto_off_in_dev():
    # The shipped YAML leaves exit_when_clients_gone null: off unless frozen.
    import sys
    assert app_config.APP_LIFECYCLE_EXIT_WHEN_CLIENTS_GONE == bool(getattr(sys, "frozen", False)) or \
        app_config.APP_LIFECYCLE_CFG.get("exit_when_clients_gone") is not None
    assert app_config.APP_LIFECYCLE_CLIENT_GRACE_S > 0
