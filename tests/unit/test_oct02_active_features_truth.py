"""R2 (2026-10-02): [ACTIVE FEATURES] reports USABLE, not merely configured;
the narrative notes path follows utils.notes_common.daily_notes_base()."""
from unittest.mock import MagicMock

import pytest

from core.prompt.formatter import PromptFormatter
from memory.memory_consolidator import MemoryConsolidator


def _inv(monkeypatch, tmp_path, *, web_key, vault_exists, obsidian=True, web=True):
    cfg = "config.app_config."
    monkeypatch.setattr(cfg + "WEB_SEARCH_ENABLED", web)
    monkeypatch.setattr(cfg + "WEB_SEARCH_API_KEY", "k" if web_key else "")
    monkeypatch.setattr(cfg + "OBSIDIAN_ENABLED", obsidian)
    vault = tmp_path / "vault"
    if vault_exists:
        vault.mkdir()
    monkeypatch.setattr(cfg + "OBSIDIAN_VAULT_PATH", str(vault))
    f = PromptFormatter(token_manager=MagicMock(), time_manager=None)
    return f._build_feature_inventory({})


def test_web_on_without_key_is_off_not_configured(monkeypatch, tmp_path):
    assert "web_search=OFF(not configured)" in _inv(monkeypatch, tmp_path, web_key=False, vault_exists=True)


def test_obsidian_enabled_without_vault_is_off(monkeypatch, tmp_path):
    out = _inv(monkeypatch, tmp_path, web_key=True, vault_exists=False)
    assert "obsidian=OFF(no vault)" in out


def test_usable_features_still_on(monkeypatch, tmp_path):
    out = _inv(monkeypatch, tmp_path, web_key=True, vault_exists=True)
    assert "obsidian=ON" in out and "web_search=ON" in out


def _consolidator():
    return MemoryConsolidator.__new__(MemoryConsolidator)


@pytest.mark.parametrize("obsidian", [True, False])
def test_notes_path_follows_daily_notes_base(monkeypatch, tmp_path, obsidian):
    cfg = "config.app_config."
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(cfg + "OBSIDIAN_ENABLED", obsidian)
    monkeypatch.setattr(cfg + "OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setattr(cfg + "DAILY_NOTES_FOLDER", "Daily")
    monkeypatch.setattr(cfg + "CORPUS_FILE", str(tmp_path / "data" / "corpus.json"))
    expected = (vault / "Daily") if obsidian else (tmp_path / "data" / "notes" / "Daily")
    assert _consolidator()._get_obsidian_notes_path() is None  # missing -> None
    expected.mkdir(parents=True)
    assert _consolidator()._get_obsidian_notes_path() == str(expected)
