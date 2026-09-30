"""2026-09-28 (BC-26/BC-37): Settings saves go to config.local.yaml, never config.yaml."""
import yaml
import pytest

from gui import settings_core

BASE = "# committed, commented\nweb_search:\n  enabled: true   # keep me\n  daily_credit_limit: 100\nmodels:\n  active: m1\n"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.yaml").write_text(BASE)
    monkeypatch.chdir(tmp_path)
    return tmp_path / "config"


def _set_limit(d):
    d.setdefault("web_search", {})["daily_credit_limit"] = 42


def test_base_config_byte_identical_and_value_in_local(cfg):
    ok, err = settings_core.save_settings(_set_limit)
    assert ok, err
    assert (cfg / "config.yaml").read_text() == BASE
    local = yaml.safe_load((cfg / "config.local.yaml").read_text())
    assert local == {"web_search": {"daily_credit_limit": 42}}


def test_existing_local_keys_survive_and_reload_shows_saved(cfg):
    (cfg / "config.local.yaml").write_text("smtp_user: me\nweb_search:\n  enabled: false\n")
    ok, _ = settings_core.save_settings(_set_limit)
    assert ok
    local = yaml.safe_load((cfg / "config.local.yaml").read_text())
    assert local["smtp_user"] == "me"
    assert local["web_search"] == {"enabled": False, "daily_credit_limit": 42}
    view = settings_core.load_settings()
    assert view["web_search"]["daily_credit_limit"] == 42
    assert view["web_search"]["enabled"] is False
    assert view["models"]["active"] == "m1"


def test_corrupt_local_is_not_overwritten(cfg):
    (cfg / "config.local.yaml").write_text("a: [unclosed\n")
    ok, err = settings_core.save_settings(_set_limit)
    assert not ok and err
    assert (cfg / "config.local.yaml").read_text() == "a: [unclosed\n"
    assert (cfg / "config.yaml").read_text() == BASE
