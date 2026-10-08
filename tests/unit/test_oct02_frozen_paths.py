"""Frozen-build path resolution (2026-10-02, packaging P1).

Drives THE deployed bootstrap helpers with sys.frozen state simulated by
patching the module constants (IS_FROZEN is computed at import). Dev paths
must be unchanged.
"""
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import utils.bootstrap as bs

REPO_ROOT = Path(__file__).resolve().parents[2]

STORE_ENV = (
    "DAEMON_DATA_DIR", "NARRATIVE_CONTEXT_PATH", "FILE_UPLOAD_IMAGE_DIR",
    "TURN_TELEMETRY_PATH", "PENDING_ACTIONS_STORE_PATH", "INTERNET_ACTIONS_AUDIT_LOG",
    "GOOGLE_TOKEN_PATH", "WIKIDATA_PERSIST_PATH", "VISUAL_MEMORY_INDEX_PATH",
    "VISUAL_MEMORY_META_PATH", "VISUAL_MEMORY_ENABLED",
)


_OTHER_ENV = (
    "CORPUS_FILE", "CHROMA_PATH", "USER_PROFILE_PATH", "LOG_DIR", "CONVERSATION_LOG_DIR",
    "KNOWLEDGE_GRAPH_PERSIST_PATH", "KNOWLEDGE_GRAPH_ALIASES_PATH", "STALENESS_INDEX_PATH",
    "SURFACING_HISTORY_PATH", "WEB_SEARCH_CREDITS_PATH", "SYSTEM_PROMPT_PATH", "CONFIG_PATH",
    "WIKI_FAISS_PATH", "WIKI_DUMP_PATH", "SEM_INDEX_PATH", "WIKI_ENABLED", "PROMPT_MAX_WIKI",
    "HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "DAEMON_EXTERNAL_DATA",
)


@pytest.fixture(autouse=True)
def _restore_environ():
    """bs.setup_environment() in frozen mode writes ~25 path variables straight
    into os.environ via setdefault; monkeypatch.delenv on an ABSENT key records
    nothing to restore, so they leaked into every later test. A leaked
    DAEMON_DATA_DIR broke test_profile_path_authority once get_user_data_dir
    began honouring it (2026-10-08, class: BC-37). Snapshot and restore."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def frozen_tree(tmp_path, monkeypatch):
    app = tmp_path / "Daemon"
    (app / "_internal" / "web" / "dist").mkdir(parents=True)
    (app / "_internal" / "web" / "dist" / "index.html").write_text("<html></html>")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(bs, "IS_FROZEN", True)
    monkeypatch.setattr(bs, "IS_WINDOWS", False)
    monkeypatch.setattr(bs, "IS_MACOS", False)
    monkeypatch.setattr(sys, "executable", str(app / "Daemon"))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("APPDATA", raising=False)
    # every key setup_environment() may setdefault must be registered for restore
    for k in STORE_ENV + _OTHER_ENV:
        monkeypatch.delenv(k, raising=False)
    return app, home


def test_frozen_resolves_spa_into_internal(frozen_tree):
    app, _ = frozen_tree
    assert bs.resolve_bundled_path("web/dist") == str(app / "_internal" / "web" / "dist")


def test_absolute_path_passes_through(frozen_tree, tmp_path):
    assert bs.resolve_bundled_path(str(tmp_path)) == str(tmp_path)


def test_dev_resolves_against_repo_root_not_cwd(monkeypatch, tmp_path):
    monkeypatch.setattr(bs, "IS_FROZEN", False)
    monkeypatch.chdir(tmp_path)
    assert bs.resolve_bundled_path("web/dist") == str(REPO_ROOT / "web" / "dist")


def test_frozen_stores_land_in_user_data_dir(frozen_tree, monkeypatch):
    _, home = frozen_tree
    user_dir = bs.setup_environment()
    assert user_dir == str(home / ".daemon")
    for k in STORE_ENV:
        if k == "VISUAL_MEMORY_ENABLED":
            continue
        val = os.environ[k]
        assert val.startswith(user_dir), (k, val)


def test_frozen_without_clip_weights_disables_visual_memory(frozen_tree):
    bs.setup_environment()
    assert os.environ["VISUAL_MEMORY_ENABLED"] == "0"


def test_explicit_user_env_beats_clip_default(frozen_tree, monkeypatch):
    monkeypatch.setenv("VISUAL_MEMORY_ENABLED", "1")
    bs.setup_environment()
    assert os.environ["VISUAL_MEMORY_ENABLED"] == "1"


def test_bundled_clip_weights_keep_visual_memory_default(frozen_tree):
    app, _ = frozen_tree
    (app / "_internal" / "models" / "models--timm--vit_base_patch32_clip_224.openai").mkdir(parents=True)
    bs.setup_environment()
    assert "VISUAL_MEMORY_ENABLED" not in os.environ


def test_dev_setup_environment_sets_no_store_env(monkeypatch, tmp_path):
    for k in STORE_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(bs, "IS_FROZEN", False)
    monkeypatch.setattr(bs, "ensure_directories", lambda: str(tmp_path))
    for k in STORE_ENV:
        monkeypatch.delenv(k, raising=False)
    bs.setup_environment()
    assert not [k for k in STORE_ENV if k in os.environ]


def _app_config_values(env_extra):
    """Import the deployed app_config in a clean child and print path constants."""
    code = (
        "import config.app_config as c;"
        "print(c.FRONTEND_DIST_DIR);print(c.TURN_TELEMETRY_PATH);"
        "print(c.PENDING_ACTIONS_STORE_PATH);print(c.INTERNET_ACTIONS_GOOGLE_TOKEN_PATH);"
        "print(c.VISUAL_MEMORY_INDEX_PATH);print(c.NARRATIVE_CONTEXT_PATH);print(c.DEFAULT_DATA_DIR)"
    )
    env = {k: v for k, v in os.environ.items() if k not in STORE_ENV and k != "PYTHONPATH"}
    env.update(env_extra)
    env["DAEMON_TEST_MODE"] = "1"
    out = subprocess.run([sys.executable, "-s", "-c", code], cwd=str(REPO_ROOT), env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-800:]
    return out.stdout.strip().splitlines()[-7:]


def test_app_config_dev_defaults_unchanged():
    vals = _app_config_values({})
    assert vals[0] == "web/dist"
    assert Path(vals[1]).parts == ("logs", "turn_records.jsonl")
    assert Path(vals[2]).parts == ("data", "pending_actions.json")
    assert Path(vals[3]).parts == ("data", "google_token.json")
    assert Path(vals[4]).parts == ("data", "clip_index.faiss")
    assert vals[6] == "./data"


def test_app_config_honours_store_env(tmp_path):
    vals = _app_config_values({
        "TURN_TELEMETRY_PATH": str(tmp_path / "t.jsonl"),
        "PENDING_ACTIONS_STORE_PATH": str(tmp_path / "p.json"),
        "GOOGLE_TOKEN_PATH": str(tmp_path / "g.json"),
        "VISUAL_MEMORY_INDEX_PATH": str(tmp_path / "c.faiss"),
        "NARRATIVE_CONTEXT_PATH": str(tmp_path / "n.txt"),
        "DAEMON_DATA_DIR": str(tmp_path / "d"),
        "CHROMA_PATH": str(tmp_path / "chroma"),
    })
    assert vals[1:6] == [str(tmp_path / n) for n in ("t.jsonl", "p.json", "g.json", "c.faiss", "n.txt")]
    assert vals[6] == str(tmp_path / "d")
    assert (tmp_path / "chroma").is_dir()


def test_api_app_resolves_spa_without_cwd(monkeypatch, tmp_path):
    """api/app.py must read the live attr and resolve it against the repo root."""
    src = (REPO_ROOT / "api" / "app.py").read_text()
    assert "resolve_bundled_path(app_config.FRONTEND_DIST_DIR)" in src


def test_main_has_no_appdata_assumption():
    assert "APPDATA" not in (REPO_ROOT / "main.py").read_text()
