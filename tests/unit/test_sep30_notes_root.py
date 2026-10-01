"""D3: notes location resolver + committed generic defaults (class: BC-71, BC-82, BC-37)."""
from datetime import date
from pathlib import Path

import yaml

from config import app_config
from utils import notes_common
from utils.daily_notes_generator import DailyNotesGenerator, get_daily_note_path
from utils.monthly_notes_generator import MonthlyNotesGenerator
from utils.weekly_notes_generator import WeeklyNotesGenerator

REPO = Path(__file__).resolve().parents[2]


def _cfg(monkeypatch, *, enabled, vault, corpus, folder="Daily Notes"):
    monkeypatch.setattr(app_config, "OBSIDIAN_ENABLED", enabled)
    monkeypatch.setattr(app_config, "OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setattr(app_config, "CORPUS_FILE", str(corpus))
    monkeypatch.setattr(app_config, "DAILY_NOTES_FOLDER", folder)


def test_resolver_uses_vault_when_enabled_and_present(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    _cfg(monkeypatch, enabled=True, vault=vault, corpus=tmp_path / "data" / "c.json")
    assert notes_common.daily_notes_base() == vault / "Daily Notes"


def test_resolver_falls_back_to_data_root(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    corpus = tmp_path / "data" / "c.json"
    _cfg(monkeypatch, enabled=False, vault=vault, corpus=corpus)
    assert notes_common.daily_notes_base() == tmp_path / "data" / "notes" / "Daily Notes"
    # enabled but vault missing -> also data root
    _cfg(monkeypatch, enabled=True, vault=tmp_path / "nope", corpus=corpus)
    assert notes_common.daily_notes_base() == tmp_path / "data" / "notes" / "Daily Notes"


def test_relative_corpus_resolves_against_repo_root(tmp_path, monkeypatch):
    _cfg(monkeypatch, enabled=False, vault=tmp_path, corpus="./data/corpus_x.json")
    monkeypatch.chdir(tmp_path)
    assert notes_common.daily_notes_base() == REPO / "data" / "notes" / "Daily Notes"


def test_generators_share_the_resolver_base(tmp_path, monkeypatch):
    _cfg(monkeypatch, enabled=False, vault=tmp_path / "v", corpus=tmp_path / "data" / "c.json")
    base = tmp_path / "data" / "notes" / "Daily Notes"
    assert DailyNotesGenerator().output_dir == base
    assert WeeklyNotesGenerator().output_dir == base
    assert MonthlyNotesGenerator().output_dir == base


def test_explicit_vault_override_still_wins(tmp_path, monkeypatch):
    _cfg(monkeypatch, enabled=False, vault=tmp_path / "v", corpus=tmp_path / "data" / "c.json")
    ov = tmp_path / "other"
    for cls in (DailyNotesGenerator, WeeklyNotesGenerator, MonthlyNotesGenerator):
        assert cls(vault_path=str(ov)).output_dir == ov / "Daily Notes"


def test_write_and_read_land_in_same_base(tmp_path, monkeypatch):
    _cfg(monkeypatch, enabled=False, vault=tmp_path / "v", corpus=tmp_path / "data" / "c.json")
    gen = DailyNotesGenerator()
    d = date(2026, 9, 29)
    path = gen._get_note_path(d) if hasattr(gen, "_get_note_path") else None
    if path is None:
        path = gen.output_dir / f"{d.month} {d.day} {d.strftime('%y')} Daily Note.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("note", encoding="utf-8")
    assert str(path).startswith(str(tmp_path / "data" / "notes"))
    assert get_daily_note_path(d) == path


def test_committed_defaults():
    cfg = yaml.safe_load((REPO / "config" / "config.yaml").read_text())
    assert cfg["obsidian"]["enabled"] is False
    assert cfg["obsidian"]["vault_path"] == ""
    assert cfg["daily_notes"]["folder"] == "Daily Notes"
    assert cfg["file_access"]["approved_folders"] == ["."]
    assert cfg["location"]["ip_lookup_enabled"] is False
    prompt = cfg["prompts"]["default_system_prompt"]
    assert "brutally" not in prompt and "dark humor" not in prompt
    assert prompt.startswith("You are Daemon, a thoughtful, honest AI assistant")
