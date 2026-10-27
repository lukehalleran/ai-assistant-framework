"""reauth_google --port N: fixed OAuth callback port + SSH forward guidance.

Incident date: 2026-10-23 (#101 audit follow-up; implemented 2026-10-26, lane H1).
Class: BC-58 (a headless host cannot complete the consent redirect with a random
callback port, so the documented re-auth path was unusable over SSH).

Drives the deployed GoogleAuthManager.authenticate and the deployed script's
argument parser / main with a FAKE flow object (no network, no browser).
"""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("google.oauth2.credentials")

from core.actions.google_auth import GoogleAuthManager, SCOPES

REPO = Path(__file__).resolve().parents[2]


def _creds():
    c = MagicMock()
    c.token, c.refresh_token = "t", "r"
    c.token_uri = "https://oauth2.googleapis.com/token"
    c.client_id, c.client_secret = "id", "secret"
    c.scopes, c.expiry = set(SCOPES), None
    return c


class _Flow:
    def __init__(self, browserless=False):
        self.calls = []
        self.browserless = browserless

    def run_local_server(self, **kw):
        self.calls.append(kw)
        if self.browserless and kw.get("open_browser", True):
            raise RuntimeError("could not locate runnable browser")
        return _creds()


def _install_fake_flow(monkeypatch, flow):
    fake = types.ModuleType("google_auth_oauthlib.flow")
    fake.InstalledAppFlow = types.SimpleNamespace(from_client_config=lambda cfg, scopes: flow)
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib", types.ModuleType("google_auth_oauthlib"))
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", fake)


def _mgr(tmp_path):
    return GoogleAuthManager(client_id="id", client_secret="secret", token_path=str(tmp_path / "tok.json"))


@pytest.mark.asyncio
async def test_default_port_is_unchanged_random(monkeypatch, tmp_path):
    flow = _Flow()
    _install_fake_flow(monkeypatch, flow)
    assert await _mgr(tmp_path).authenticate() is True
    assert flow.calls == [{"port": 0}]


@pytest.mark.asyncio
async def test_port_is_passed_to_run_local_server(monkeypatch, tmp_path):
    flow = _Flow()
    _install_fake_flow(monkeypatch, flow)
    assert await _mgr(tmp_path).authenticate(port=8765) is True
    assert flow.calls == [{"port": 8765}]


@pytest.mark.asyncio
async def test_no_browser_prints_ssh_forward_with_the_port(monkeypatch, tmp_path, capsys):
    flow = _Flow(browserless=True)
    _install_fake_flow(monkeypatch, flow)
    assert await _mgr(tmp_path).authenticate(port=8765) is True
    out = capsys.readouterr().out
    assert "ssh -L 8765:localhost:8765 <host>" in out
    assert flow.calls[0] == {"port": 8765}
    assert flow.calls[1]["port"] == 8765 and flow.calls[1]["open_browser"] is False
    assert flow.calls[1]["authorization_prompt_message"] == "{url}\n"  # the URL is printed by the flow


@pytest.mark.asyncio
async def test_no_browser_without_port_tells_you_to_pick_one(monkeypatch, tmp_path, capsys):
    flow = _Flow(browserless=True)
    _install_fake_flow(monkeypatch, flow)
    assert await _mgr(tmp_path).authenticate() is True
    out = capsys.readouterr().out
    assert "--port" in out and "ssh -L" in out


def _load_script():
    spec = importlib.util.spec_from_file_location("reauth_google_under_test", REPO / "scripts" / "reauth_google.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_parses_port_and_defaults_to_random():
    mod = _load_script()
    assert mod._parse_args([]).port == 0
    assert mod._parse_args(["--port", "8765"]).port == 8765
    with pytest.raises(SystemExit):
        mod._parse_args(["--port", "70000"])


@pytest.mark.asyncio
async def test_script_main_forwards_port_to_authenticate(monkeypatch, tmp_path):
    mod = _load_script()
    seen = {}

    class _Auth:
        is_configured = True
        _token_path = str(tmp_path / "none.json")
        _scopes = []

        async def authenticate(self, port=0):
            seen["port"] = port
            return True

        def get_credentials(self):
            return None

    import core.actions.google_auth as ga
    monkeypatch.setattr(ga, "get_google_auth", lambda: _Auth())
    assert await mod.main(8765) == 0
    assert seen["port"] == 8765
