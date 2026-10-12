"""Lane PL (2026-10-12): Quit only for loopback peers (heartbeat quit_allowed),
one .env authority (class: BC-83), and /api/setup 422 bodies that never echo
the submitted value. Driven through the deployed create_app / resolve_env_path."""

import httpx
import pytest

from api.launch_auth import LAUNCH_TOKEN_HEADER
from gui import setup_core
from tests.unit.api_launch_auth_client import (
    TEST_LAUNCH_SECRET,
    TEST_ORIGIN,
    authed_client,
    make_test_app,
)
from tests.unit.helpers_orchestrator import _make_orchestrator

SECRET = "sk-or-v1-SECRETSECRETSECRET"


def _app():
    return make_test_app(_make_orchestrator(), start_background=False)


def _client_from(app, peer):
    transport = httpx.ASGITransport(app=app, client=peer)
    return httpx.AsyncClient(
        transport=transport, base_url=TEST_ORIGIN,
        headers={LAUNCH_TOKEN_HEADER: TEST_LAUNCH_SECRET, "Origin": TEST_ORIGIN},
    )


class TestHeartbeatQuitAllowed:
    @pytest.mark.asyncio
    async def test_loopback_peer_may_quit(self):
        async with authed_client(_app()) as c:
            r = await c.post("/api/heartbeat")
        assert r.status_code == 200
        assert r.json() == {"ok": True, "quit_allowed": True}

    @pytest.mark.asyncio
    async def test_non_loopback_peer_may_not(self):
        app = _app()
        async with _client_from(app, ("100.64.0.9", 4444)) as c:
            r = await c.post("/api/heartbeat")
        assert r.status_code == 200
        assert r.json() == {"ok": True, "quit_allowed": False}
        assert app.state.lifecycle.last_heartbeat is not None  # still counts as liveness


class TestEnvAuthority:
    def test_frozen_uses_user_data_dir(self, monkeypatch, tmp_path):
        import utils.bootstrap as bootstrap
        monkeypatch.delenv("DAEMON_DATA_DIR", raising=False)
        monkeypatch.setattr(bootstrap, "IS_FROZEN", True)
        monkeypatch.setattr(bootstrap, "get_user_data_dir", lambda: str(tmp_path))
        assert bootstrap.resolve_env_path() == tmp_path / ".env"

    def test_data_dir_override_on_a_source_checkout(self, monkeypatch, tmp_path):
        import utils.bootstrap as bootstrap
        monkeypatch.setattr(bootstrap, "IS_FROZEN", False)
        monkeypatch.setenv("DAEMON_DATA_DIR", str(tmp_path))
        assert bootstrap.resolve_env_path() == tmp_path / ".env"

    def test_plain_checkout_uses_repo_root(self, monkeypatch):
        import utils.bootstrap as bootstrap
        from pathlib import Path
        monkeypatch.setattr(bootstrap, "IS_FROZEN", False)
        monkeypatch.delenv("DAEMON_DATA_DIR", raising=False)
        assert bootstrap.resolve_env_path() == Path(bootstrap.get_app_dir()) / ".env"

    def test_setup_core_delegates_to_bootstrap(self, monkeypatch, tmp_path):
        import utils.bootstrap as bootstrap
        sentinel = tmp_path / "elsewhere" / ".env"
        monkeypatch.setattr(bootstrap, "resolve_env_path", lambda: sentinel)
        assert setup_core.resolve_env_path() == sentinel

    def test_main_loads_the_same_file_setup_writes(self):
        """main.py must load through bootstrap.resolve_env_path (no bare load_dotenv())."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[2] / "main.py").read_text()
        assert "load_dotenv(resolve_env_path(), override=True)" in src
        assert "load_dotenv(override=True)" not in src


class TestSetupValidationErrorsDoNotEcho:
    @pytest.mark.asyncio
    async def test_wrong_typed_setup_body_does_not_reflect_the_secret(self):
        async with authed_client(_app()) as c:
            r = await c.post("/api/setup/openrouter", json={"key": 12345, "extra": SECRET})
            r2 = await c.post("/api/setup/openrouter", json={"key": {"nested": SECRET}})
            r3 = await c.post(
                "/api/setup/openrouter", content=('{"key": "' + SECRET),
                headers={"Content-Type": "application/json"},
            )
        for resp in (r, r2, r3):
            assert resp.status_code == 422
            assert "SECRETSECRET" not in resp.text
            assert all("input" not in e and "ctx" not in e for e in resp.json()["detail"])
        assert r.json()["detail"][0]["loc"] == ["body", "key"]  # still useful

    @pytest.mark.asyncio
    async def test_other_routes_keep_the_default_422_with_input(self):
        async with authed_client(_app()) as c:
            r = await c.put("/api/models/active", json={"name": 5})
        assert r.status_code == 422
        assert any("input" in e for e in r.json()["detail"])
