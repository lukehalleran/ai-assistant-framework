"""One data-root authority (2026-10-08, class: BC-83, BC-10).

`utils.bootstrap.get_user_data_dir` computed the dev data dir from __file__
with no override while config/app_config.py honoured DAEMON_DATA_DIR for its
own stores, so the profile path / ensure_directories could disagree with the
stores. Drives the deployed resolver; monkeypatch + tmp_path only, and
ensure_directories is never called.
"""
import os

import pytest

import utils.bootstrap as bs


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("DAEMON_DATA_DIR", raising=False)
    monkeypatch.delenv("USER_PROFILE_PATH", raising=False)


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_unset_or_blank_uses_repo_data_dir(monkeypatch, blank):
    if blank is not None:
        monkeypatch.setenv("DAEMON_DATA_DIR", blank)
    monkeypatch.setattr(bs, "IS_FROZEN", False)
    assert bs.get_user_data_dir() == os.path.join(bs.get_app_dir(), "data")


def test_override_wins_in_dev_mode_as_abspath(monkeypatch, tmp_path):
    monkeypatch.setattr(bs, "IS_FROZEN", False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DAEMON_DATA_DIR", "  rel_store  ")
    assert bs.get_user_data_dir() == str(tmp_path / "rel_store")


def test_override_expands_user(monkeypatch, tmp_path):
    monkeypatch.setattr(bs, "IS_FROZEN", False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DAEMON_DATA_DIR", "~/store")
    assert bs.get_user_data_dir() == str(tmp_path / "store")


def test_override_wins_in_frozen_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(bs, "IS_FROZEN", True)
    monkeypatch.setattr(bs, "IS_WINDOWS", False)
    monkeypatch.setattr(bs, "IS_MACOS", False)
    monkeypatch.setenv("DAEMON_DATA_DIR", str(tmp_path / "frozen_store"))
    assert bs.get_user_data_dir() == str(tmp_path / "frozen_store")


def test_frozen_without_override_keeps_platform_dir(monkeypatch):
    monkeypatch.setattr(bs, "IS_FROZEN", True)
    monkeypatch.setattr(bs, "IS_WINDOWS", False)
    monkeypatch.setattr(bs, "IS_MACOS", False)
    assert bs.get_user_data_dir() == os.path.expanduser("~/.daemon")


def test_profile_path_follows_override_when_profile_env_unset(monkeypatch, tmp_path):
    monkeypatch.setattr(bs, "IS_FROZEN", False)
    monkeypatch.chdir(tmp_path)  # no cwd-relative legacy profile can interfere
    store = tmp_path / "store"
    store.mkdir()
    (store / "user_profile.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("DAEMON_DATA_DIR", str(store))
    assert bs.get_user_profile_path() == str(store / "user_profile.json")


def test_explicit_profile_env_still_beats_override(monkeypatch, tmp_path):
    monkeypatch.setenv("DAEMON_DATA_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("USER_PROFILE_PATH", str(tmp_path / "elsewhere.json"))
    assert bs.get_user_profile_path() == str(tmp_path / "elsewhere.json")
