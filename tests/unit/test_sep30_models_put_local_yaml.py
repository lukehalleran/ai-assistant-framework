"""2026-09-30 (D2, BC-26/BC-37): PUT /api/models/active persists to config.local.yaml;
snapshot fallbacks match shipped defaults; snapshot carries availability flags."""
import asyncio
from types import SimpleNamespace

import pytest
import yaml

from api.routes import models as models_route
from api.schemas import ActiveModelRequest
from gui import settings_core

BASE = "# committed\nmodels:\n  active: m1\nweb_search:\n  enabled: true\n"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.yaml").write_text(BASE)
    monkeypatch.chdir(tmp_path)
    return tmp_path / "config"


class _MM:
    api_models = {"m1": 1, "m2": 2}
    models = {}

    def __init__(self):
        self.active = "m1"

    def switch_model(self, n):
        self.active = n

    def get_active_model_name(self):
        return self.active


def test_put_active_writes_local_not_base(cfg):
    mm = _MM()
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        daemon=SimpleNamespace(orchestrator=SimpleNamespace(model_manager=mm)))))
    resp = asyncio.run(models_route.set_active_model(ActiveModelRequest(name="m2"), req))
    assert resp.active == "m2" and mm.active == "m2"
    assert (cfg / "config.yaml").read_text() == BASE
    assert yaml.safe_load((cfg / "config.local.yaml").read_text()) == {"models": {"active": "m2"}}


def test_snapshot_fallbacks_match_shipped_defaults(cfg):
    (cfg / "config.yaml").write_text("{}\n")
    snap = settings_core.get_settings_snapshot(SimpleNamespace())
    assert snap["proposals"]["enabled"] is False
    assert snap["tokens"]["best_of_max_tokens"] == 8816
    assert snap["tokens"]["judge_max_tokens"] == 80


def test_availability_flags(cfg, monkeypatch, tmp_path):
    import config.app_config as ac
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(ac, "WEB_SEARCH_API_KEY", "k")
    monkeypatch.setattr(ac, "OBSIDIAN_ENABLED", True)
    monkeypatch.setattr(ac, "OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setattr(ac, "DAEMON_MODE", "dev")
    assert settings_core.get_settings_snapshot(SimpleNamespace())["availability"] == {
        "web_search_key": True, "vault": True, "dev_mode": True}
    monkeypatch.setattr(ac, "WEB_SEARCH_API_KEY", "")
    monkeypatch.setattr(ac, "OBSIDIAN_VAULT_PATH", str(tmp_path / "nope"))
    monkeypatch.setattr(ac, "DAEMON_MODE", "user")
    assert settings_core.get_settings_snapshot(SimpleNamespace())["availability"] == {
        "web_search_key": False, "vault": False, "dev_mode": False}
