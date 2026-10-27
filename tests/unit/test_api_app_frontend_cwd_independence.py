"""Behavioural sibling for ``test_api_app_resolves_spa_without_cwd`` in
tests/unit/test_bootstrap_frozen_paths.py (BC-63: it read api/app.py as text
and grepped for ``resolve_bundled_path(app_config.FRONTEND_DIST_DIR)``).

Incident: the SPA bundle was located relative to the launch cwd, so a launch
from any other directory (or a frozen build) served no frontend. Bug classes:
BC-63, BC-62. THE deployed ``create_app`` is driven with a relative
``frontend_dist_dir``; the repo root (dev) / ``_internal`` (frozen) is faked
under tmp_path and the process cwd is a DIFFERENT directory.
"""
import os

import pytest

from config import app_config
from tests.unit.api_launch_auth_client import authed_client as _client
from tests.unit.api_launch_auth_client import make_test_app as create_app
from tests.unit.helpers_orchestrator import _make_orchestrator
import utils.bootstrap as bootstrap
from api.app import mount_admin_and_frontend

MARK = "<title>fixture-spa-shell</title>"


def _app(monkeypatch):
    """The deployed create_app + mount_admin_and_frontend; the Gradio admin
    build is made to fail (it is caught and logged by the mount itself) so the
    test needs no Gradio UI, only the frontend half of the mount."""
    import gui.launch as launch

    def _no_admin(orchestrator):
        raise RuntimeError("admin UI not built in this test")

    monkeypatch.setattr(launch, "build_demo", _no_admin)
    orch = _make_orchestrator()
    return mount_admin_and_frontend(create_app(orch, start_background=False), orch)


def _spa(root):
    dist = root / "web" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text(f"<html><head>{MARK}</head><body></body></html>",
                                     encoding="utf-8")


@pytest.mark.asyncio
async def test_dev_launch_from_another_cwd_still_serves_the_shell(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    _spa(root)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(bootstrap, "IS_FROZEN", False)
    monkeypatch.setattr(bootstrap, "get_app_dir", lambda: str(root))
    monkeypatch.setattr(app_config, "FRONTEND_DIST_DIR", "web/dist")
    monkeypatch.setattr(app_config, "API_SERVE_FRONTEND", True)
    monkeypatch.chdir(elsewhere)
    app = _app(monkeypatch)
    async with _client(app) as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert MARK in resp.text
    assert 'name="daemon-launch-token"' in resp.text


@pytest.mark.asyncio
async def test_frozen_build_serves_the_shell_from_internal(monkeypatch, tmp_path):
    app_dir = tmp_path / "frozen"
    _spa(app_dir / "_internal")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(bootstrap, "IS_FROZEN", True)
    monkeypatch.setattr(bootstrap, "get_app_dir", lambda: str(app_dir))
    monkeypatch.setattr(app_config, "FRONTEND_DIST_DIR", "web/dist")
    monkeypatch.setattr(app_config, "API_SERVE_FRONTEND", True)
    monkeypatch.chdir(elsewhere)
    app = _app(monkeypatch)
    async with _client(app) as client:
        resp = await client.get("/")
    assert resp.status_code == 200 and MARK in resp.text


@pytest.mark.asyncio
async def test_missing_bundle_serves_no_shell(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "IS_FROZEN", False)
    monkeypatch.setattr(bootstrap, "get_app_dir", lambda: str(tmp_path / "nothing"))
    monkeypatch.setattr(app_config, "FRONTEND_DIST_DIR", "web/dist")
    monkeypatch.setattr(app_config, "API_SERVE_FRONTEND", True)
    monkeypatch.chdir(tmp_path)
    app = _app(monkeypatch)
    async with _client(app) as client:
        resp = await client.get("/")
    assert MARK not in resp.text
