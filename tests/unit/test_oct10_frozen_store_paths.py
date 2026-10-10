"""Frozen app: every store under the user-data dir; headless splash (2026-10-10).

P3 smoke finding F4 (class: BC-83, BC-16): ``setup_environment()`` set
DAEMON_DATA_DIR but nine stores still resolved against the launch cwd
(``data/backups/``, last_query_time / last_session_time / active_days,
tone_state, curation queue + audit journal, the debug log and conversation
logs). F2 (class: BC-47): ``close_splash``/``update_splash`` caught only
ImportError, so a headless frozen run (pyi_splash raises RuntimeError) aborted
startup.

Frozen state is simulated by patching the module constants (IS_FROZEN is
computed at import), the same way test_oct02_frozen_paths does. Import-time
constants (tone state, curation queue/journal, BACKUP_DIR) are read through the
deployed modules in a clean child interpreter launched from a scratch cwd
carrying the env that ``setup_environment()`` produced; call-time accessors
(TimeManager, ConversationLogger, configure_logging) run in-process.
"""
import json
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import utils.bootstrap as bs

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_PYTHON = sys.executable  # fixtures patch sys.executable

# Env names setup_environment() may set that the child must NOT inherit (they
# point into the fake install tree, which has no config/ or models).
_FAKE_TREE_ENV = {
    "CONFIG_PATH", "SYSTEM_PROMPT_PATH", "HF_HOME", "TRANSFORMERS_CACHE",
    "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TIKTOKEN_CACHE_DIR",
}

_CHILD_CODE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import config.app_config as c
import core.context_pipeline as cp
import memory.curation.engine as ce
import memory.curation.journal as cj
import utils.logging_utils as lu
print(json.dumps({
    "backup_dir": c.BACKUP_DIR,
    "tone": cp.ContextPipeline._TONE_STATE_PATH,
    "queue_prod": ce._PROD_QUEUE_PATH,
    "queue_resolved": ce.resolve_queue_path(),
    "journal_prod": cj._PROD_JOURNAL_PATH,
    "journal_resolved": cj.resolve_journal_path(),
}))
"""

# Round 2: remaining runtime stores, read through the deployed modules.
_CHILD_CODE_R2 = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import utils.narrative_staleness as ns
import utils.adaptive_exemplars as ae
import memory.learned_relations as lr
from core.email.outlook_auth import OutlookAuthManager
import utils.log_rotation as lrot
import utils.backup_manager as bm
seen = {}
lrot.rotate_if_large = lambda p, *a, **k: seen.setdefault("rotate", []).append(p) or False
lrot.archive_if_large = lambda p, *a, **k: seen.setdefault("archive", []).append(p) or False
lrot.maintain_debug_archives = lambda d, **k: seen.setdefault("debug_dir", d) and {"compressed": 0, "pruned": 0}
lrot.run_startup_log_maintenance()
# pending-actions fallback (used when app_config cannot be imported)
sys.modules["config.app_config"] = None
sys.modules.pop("core.actions.types", None)
import core.actions.types as at
print(json.dumps({
    "narrative": ns._flag_path(),
    "adaptive": ae._STORE_PATH,
    "learned": lr._STORE_PATH,
    "outlook": str(OutlookAuthManager("x")._token_path),
    "rotate": seen["rotate"], "archive": seen["archive"], "debug_dir": seen["debug_dir"],
    "pending_fallback": at._CFG_STORE_PATH,
}))
"""

_CHILD_CODE_TARGETS = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import utils.backup_manager as bm
print(json.dumps(bm.backup_targets(existing_only=False)))
"""


@pytest.fixture(autouse=True)
def _restore_environ():
    """setup_environment() writes ~25 variables straight into os.environ."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def root_logger_restored():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield root
    for h in list(root.handlers):
        if h not in handlers:
            try:
                h.close()
            except Exception:  # degrades: test cleanup only
                pass
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.fixture
def frozen_home(tmp_path, monkeypatch):
    """Frozen state + HOME=tmp, after setup_environment(); returns user_dir."""
    app = tmp_path / "Daemon"
    app.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(bs, "IS_FROZEN", True)
    monkeypatch.setattr(bs, "IS_WINDOWS", False)
    monkeypatch.setattr(bs, "IS_MACOS", False)
    monkeypatch.setattr(sys, "executable", str(app / "Daemon"))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("APPDATA", raising=False)
    for k in ("DAEMON_DATA_DIR", "LOG_DIR", "CONVERSATION_LOG_DIR", "USER_PROFILE_PATH",
              "CORPUS_FILE", "CHROMA_PATH", "DAEMON_BACKUP_DIR", "DAEMON_EXTERNAL_DATA"):
        monkeypatch.delenv(k, raising=False)
    before = dict(os.environ)
    user_dir = bs.setup_environment()
    added = {k: v for k, v in os.environ.items()
             if before.get(k) != v and k not in _FAKE_TREE_ENV}
    return types.SimpleNamespace(user_dir=user_dir, home=home, env=added, tmp=tmp_path)


def _child(env_extra, cwd, code=None):
    env = {k: v for k, v in os.environ.items()
           if k not in {"PYTHONPATH", "DAEMON_DATA_DIR", "LOG_DIR", "CONVERSATION_LOG_DIR",
                        "CORPUS_FILE", "CHROMA_PATH", "DAEMON_BACKUP_DIR"}}
    env.update(env_extra)
    env["DAEMON_TEST_MODE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    out = subprocess.run([REAL_PYTHON, "-s", "-c", code or _CHILD_CODE, str(REPO_ROOT)],
                         cwd=str(cwd), env=env, capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-1200:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _dev(name):
    """Today's repo-relative store path (the cwd-relative value of a dev launch)."""
    return os.path.join("data", name)


def _under(path, root):
    return os.path.commonpath([os.path.abspath(path), str(root)]) == str(root)


# ---------------------------------------------------------------- F4: frozen

def test_frozen_import_time_stores_resolve_under_user_dir(frozen_home):
    launch = frozen_home.tmp / "launch_cwd"
    launch.mkdir()
    got = _child(frozen_home.env, launch)
    user_dir = frozen_home.user_dir
    assert got["backup_dir"] == os.path.join(user_dir, "backups")
    assert got["tone"] == os.path.join(user_dir, "tone_state.json")
    assert got["queue_prod"] == os.path.join(user_dir, "curation_queue.json")
    assert got["journal_prod"] == os.path.join(user_dir, "logs", "curation_audit.jsonl")
    # test-mode redirects must also stay inside the user dir, never the cwd
    assert _under(got["queue_resolved"], user_dir)
    assert _under(got["journal_resolved"], user_dir)
    # nothing leaked into the launch cwd while importing the modules
    assert list(launch.iterdir()) == []


def test_frozen_remaining_runtime_stores_resolve_under_user_dir(frozen_home):
    launch = frozen_home.tmp / "launch_cwd2"
    launch.mkdir()
    got = _child(frozen_home.env, launch, _CHILD_CODE_R2)
    u = frozen_home.user_dir
    assert got["narrative"] == os.path.join(u, "narrative_stale.json")
    assert got["adaptive"] == os.path.join(u, "adaptive_exemplars.json")
    assert got["learned"] == os.path.join(u, "learned_relations.json")
    assert got["outlook"] == os.path.join(u, "outlook_token.json")
    assert got["pending_fallback"] == os.path.join(u, "pending_actions.json")
    assert os.path.join(u, "logs", "daily_notes.log") in got["rotate"]
    assert got["archive"] == [os.path.join(u, "logs", "actions_audit.jsonl")]
    assert got["debug_dir"] == os.path.join(u, "logs")
    assert list(launch.iterdir()) == []


def test_frozen_every_backup_target_is_under_user_dir(frozen_home):
    launch = frozen_home.tmp / "launch_cwd3"
    launch.mkdir()
    targets = _child(frozen_home.env, launch, _CHILD_CODE_TARGETS)
    assert targets, "backup_targets returned nothing"
    outside = [t for t in targets if not _under(t, frozen_home.user_dir)]
    assert outside == []
    assert list(launch.iterdir()) == []


def test_dev_remaining_runtime_stores_unchanged():
    got = _child({}, REPO_ROOT, _CHILD_CODE_R2)
    assert got["narrative"] == _dev("narrative_stale.json")
    assert got["adaptive"] == _dev("adaptive_exemplars.json")
    assert got["learned"] == _dev("learned_relations.json")
    assert got["outlook"] == _dev("outlook_token.json")
    assert got["pending_fallback"] == _dev("pending_actions.json")
    assert os.path.join("logs", "daily_notes.log") in got["rotate"]
    assert got["archive"] == [os.path.join("logs", "actions_audit.jsonl")]
    assert got["debug_dir"] == "."


def test_launch_log_fallback_goes_through_store_path():
    # get_app_log_path is a closure inside the Gradio builder (gradio import is
    # too heavy to drive here), so pin the wiring at the source.
    src = (REPO_ROOT / "gui" / "launch.py").read_text(encoding="utf-8")
    assert "os.path.abspath(store_path('daemon_debug.log', 'logs/daemon_debug.log'))" in src


def test_frozen_time_manager_stores_resolve_under_user_dir(frozen_home):
    from utils.time_manager import TimeManager
    tm = TimeManager()
    assert tm.time_file == os.path.join(frozen_home.user_dir, "last_query_time.json")
    assert tm.active_days_file == os.path.join(frozen_home.user_dir, "active_days.json")
    assert tm.session_file == os.path.join(frozen_home.user_dir, "last_session_time.json")


def test_frozen_conversation_logger_dir_is_user_conversation_logs(frozen_home, monkeypatch):
    monkeypatch.chdir(frozen_home.tmp)
    from utils.conversation_logger import ConversationLogger
    lg = ConversationLogger()
    assert str(lg.log_dir) == os.path.join(frozen_home.user_dir, "conversation_logs")
    assert str(lg.log_dir) == os.environ["CONVERSATION_LOG_DIR"]
    assert not (frozen_home.tmp / "conversation_logs").exists()


def test_frozen_default_debug_log_is_under_user_logs(frozen_home, monkeypatch, root_logger_restored):
    monkeypatch.delenv("DAEMON_TEST_MODE", raising=False)
    monkeypatch.chdir(frozen_home.tmp)
    from utils.logging_utils import configure_logging
    configure_logging()
    files = [h.baseFilename for h in root_logger_restored.handlers
             if isinstance(h, logging.FileHandler)]
    assert files == [os.path.join(frozen_home.user_dir, "logs", "daemon_debug.log")]
    assert not (frozen_home.tmp / "daemon_debug.log").exists()


def test_frozen_backup_targets_follow_data_root(frozen_home, monkeypatch):
    import utils.backup_manager as bm
    root = Path(frozen_home.user_dir)
    (root / "tone_state.json").write_text("{}")
    (root / "curation_queue.json").write_text("{}")
    monkeypatch.chdir(frozen_home.tmp)
    targets = bm.backup_targets(existing_only=True)
    assert str(root / "tone_state.json") in targets
    assert str(root / "curation_queue.json") in targets


def test_store_path_absolute_and_explicit_paths_win(frozen_home, tmp_path):
    assert bs.store_path(str(tmp_path / "x.json")) == str(tmp_path / "x.json")
    assert bs.store_path("") == ""


def test_data_dir_override_applies_in_dev_mode_too(monkeypatch, tmp_path):
    monkeypatch.setattr(bs, "IS_FROZEN", False)
    monkeypatch.setenv("DAEMON_DATA_DIR", str(tmp_path / "root"))
    assert bs.store_path(os.path.join("data", "a.json")) == str(tmp_path / "root" / "a.json")
    assert bs.store_path(os.path.join("logs", "b.jsonl")) == str(tmp_path / "root" / "logs" / "b.jsonl")
    assert bs.store_path(os.path.join(".", "data", "backups")) == str(tmp_path / "root" / "backups")


# ---------------------------------------------------------------- dev defaults

def test_dev_import_time_defaults_unchanged(tmp_path):
    got = _child({}, REPO_ROOT)
    assert got["backup_dir"] == _dev("backups")
    assert got["tone"] == _dev("tone_state.json")
    assert got["queue_prod"] == _dev("curation_queue.json")
    assert got["queue_resolved"] == _dev("test_curation_queue.json")  # DAEMON_TEST_MODE child
    assert got["journal_prod"] == os.path.join("logs", "curation_audit.jsonl")
    assert got["journal_resolved"] == os.path.join("logs", "test_curation_audit.jsonl")
    for key in ("backup_dir", "tone", "queue_prod", "journal_prod"):
        assert os.path.abspath(got[key]) == str(REPO_ROOT / got[key])


def test_dev_call_time_defaults_unchanged(monkeypatch, tmp_path, root_logger_restored):
    monkeypatch.setattr(bs, "IS_FROZEN", False)
    monkeypatch.delenv("DAEMON_DATA_DIR", raising=False)
    monkeypatch.delenv("DAEMON_TEST_MODE", raising=False)
    monkeypatch.chdir(tmp_path)  # relative == cwd-relative, exactly as today
    from utils.time_manager import TimeManager
    from utils.conversation_logger import ConversationLogger
    from utils.logging_utils import configure_logging
    tm = TimeManager()
    assert (tm.time_file, tm.active_days_file, tm.session_file) == (
        _dev("last_query_time.json"), _dev("active_days.json"), _dev("last_session_time.json"))
    assert str(ConversationLogger().log_dir) == "conversation_logs"
    configure_logging()
    files = [h.baseFilename for h in root_logger_restored.handlers
             if isinstance(h, logging.FileHandler)]
    assert files == [str(tmp_path / "daemon_debug.log")]


# ---------------------------------------------------------------- F2: splash

@pytest.mark.parametrize("fn,args", [("close_splash", ()), ("update_splash", ("loading",))])
def test_splash_runtime_error_does_not_abort_startup(monkeypatch, fn, args):
    def boom(*_a, **_k):
        raise RuntimeError("This module is not initialized")

    fake = types.SimpleNamespace(close=boom, update_text=boom)
    monkeypatch.setitem(sys.modules, "pyi_splash", fake)
    monkeypatch.setattr(bs, "IS_FROZEN", True)
    getattr(bs, fn)(*args)  # must not raise
