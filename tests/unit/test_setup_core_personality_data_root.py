"""A05 (2026-10-23): the custom personality lives under the user data root, is
resolved at call time, and a failed style write fails Setup.

class: BC-83 (one data-root authority), BC-30, BC-47 (swallowed failure), BC-82.
"""
from types import SimpleNamespace

import config.app_config as cfg
from gui import setup_core


def _orch():
    prefs = SimpleNamespace(style="balanced")

    class _Profile:
        identity = SimpleNamespace(name="", pronouns="")
        preferences = prefs

        def update_identity(self, name, pronouns):
            self.identity.name = name

        def update_preferences(self, style):
            prefs.style = style

    return SimpleNamespace(user_profile=_Profile())


def test_two_data_roots_keep_separate_custom_personalities(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    monkeypatch.setenv("DAEMON_DATA_DIR", str(a))
    assert setup_core.save_profile(_orch(), "Rowan", None, "warm")["ok"] is True
    monkeypatch.setenv("DAEMON_DATA_DIR", str(b))
    assert setup_core.save_profile(_orch(), "Tamsin", None, "direct")["ok"] is True

    pa, pb = a / "custom_personality.txt", b / "custom_personality.txt"
    assert pa.exists() and pb.exists()
    assert "Warm" in pa.read_text(encoding="utf-8") or "warm" in pa.read_text(encoding="utf-8").lower()
    assert "Direct" in pb.read_text(encoding="utf-8")
    assert pa.read_text(encoding="utf-8") != pb.read_text(encoding="utf-8")
    # The installed tree was never written.
    assert not (cfg.Path(cfg.__file__).parent / "prompts" / "custom_personality.txt").exists()


def test_prompt_reader_picks_custom_file_from_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DAEMON_DATA_DIR", str(tmp_path))
    (tmp_path / "custom_personality.txt").write_text("SYNTHETIC CUSTOM VOICE", encoding="utf-8")
    assert cfg.load_personality_text() == "SYNTHETIC CUSTOM VOICE"
    assert cfg.PERSONALITY_CUSTOM_PATH == str(tmp_path / "custom_personality.txt")
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("DAEMON_DATA_DIR", str(other))
    assert cfg.load_personality_text() != "SYNTHETIC CUSTOM VOICE"  # falls back to the bundled default


def test_unwritable_target_fails_save_profile(tmp_path, monkeypatch):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("DAEMON_DATA_DIR", str(blocker))  # a file where the data root should be
    res = setup_core.save_profile(_orch(), "Rowan", None, "warm")
    assert res["ok"] is False and res["error"]


def test_balanced_removes_the_custom_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DAEMON_DATA_DIR", str(tmp_path))
    assert setup_core.save_profile(_orch(), "Rowan", None, "warm")["ok"]
    assert (tmp_path / "custom_personality.txt").exists()
    assert setup_core.save_profile(_orch(), "Rowan", None, "balanced")["ok"]
    assert not (tmp_path / "custom_personality.txt").exists()
