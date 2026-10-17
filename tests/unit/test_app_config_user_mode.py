"""B1 (2026-10-12): user mode hides every developer surface.
class: BC-83, BC-58, BC-16. A frozen beta build greeted a fresh profile with the
OWNER's untracked repo files and offered file_access/git_stats/github tools.

Original batch label: oct12_b1 (renamed 2026-10-16; old path in tests/RENAMES.tsv)."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.agentic.formatters import AgenticFormatter
from core.agentic.tools import ToolExecutor
from core.prompt.context_gatherer import ContextGatherer
from core.prompt.formatter import PromptFormatter

REPO = Path(__file__).resolve().parents[2]
DEV_CONSTANTS = (
    "GIT_MEMORY_ENABLED", "SESSION_DIFF_ENABLED", "FILE_ACCESS_ENABLED",
    "GIT_STATS_ENABLED", "GITHUB_API_ENABLED", "INTERNET_ACTIONS_GITHUB_WRITE_ENABLED",
)


def _user_mode(monkeypatch):
    for name in DEV_CONSTANTS:
        monkeypatch.setattr(f"config.app_config.{name}", False)


def _dev_mode(monkeypatch):
    for name in DEV_CONSTANTS:
        monkeypatch.setattr(f"config.app_config.{name}", True)
    monkeypatch.setattr("config.app_config.INTERNET_ACTIONS_ENABLED", True)


def _executor():
    # Managers are HELD (as after a mid-process toggle) — the live flag must win.
    return ToolExecutor(
        model_manager=MagicMock(), web_search_manager=MagicMock(),
        formatter=AgenticFormatter(),
        file_access_manager=MagicMock(), git_stats_manager=MagicMock(),
        github_manager=MagicMock(),
    )


def _gatherer():
    g = ContextGatherer.__new__(ContextGatherer)
    g.memory_coordinator = SimpleNamespace(chroma_store=None)
    g.memory_id_map = {}
    return g


def test_tool_health_same_instance_before_after(monkeypatch):
    ex = _executor()
    _dev_mode(monkeypatch)
    on = ex.get_tool_health()
    assert "file_access: AVAILABLE" in on and "git_stats: AVAILABLE" in on
    assert "github: AVAILABLE" in on and "github_create_issue" in on
    _user_mode(monkeypatch)
    off = ex.get_tool_health()
    for needle in ("file_access", "git_stats", "github"):
        assert needle not in off, off


@pytest.mark.asyncio
async def test_codebase_changes_empty_and_no_git_when_off(monkeypatch):
    _user_mode(monkeypatch)
    monkeypatch.setattr("config.app_config.SESSION_DIFF_ENABLED", False)
    run = MagicMock()
    monkeypatch.setattr("core.prompt.gatherer_knowledge.subprocess.run", run)
    from datetime import datetime
    assert await _gatherer().get_codebase_changes(datetime.now()) == {}
    run.assert_not_called()


@pytest.mark.asyncio
async def test_codebase_changes_never_uses_launch_cwd(monkeypatch, tmp_path):
    """Enabled, but the module's own location is not a git work tree (frozen)."""
    monkeypatch.setattr("config.app_config.SESSION_DIFF_ENABLED", True)
    monkeypatch.setattr("core.prompt.gatherer_knowledge.sys.frozen", True, raising=False)
    run = MagicMock(return_value=SimpleNamespace(returncode=0, stdout="/owner/repo\n"))
    monkeypatch.setattr("core.prompt.gatherer_knowledge.subprocess.run", run)
    monkeypatch.chdir(REPO)  # launch CWD is a git repo; must be ignored
    from datetime import datetime
    assert await _gatherer().get_codebase_changes(datetime.now()) == {}
    run.assert_not_called()


@pytest.mark.asyncio
async def test_codebase_changes_dev_pins_module_repo_root(monkeypatch, tmp_path):
    monkeypatch.setattr("config.app_config.SESSION_DIFF_ENABLED", True)
    monkeypatch.delattr("core.prompt.gatherer_knowledge.sys.frozen", raising=False)
    run = MagicMock(return_value=SimpleNamespace(returncode=0, stdout=""))
    monkeypatch.setattr("core.prompt.gatherer_knowledge.subprocess.run", run)
    monkeypatch.chdir(tmp_path)  # hostile launch CWD
    from datetime import datetime
    await _gatherer().get_codebase_changes(datetime.now())
    assert run.call_count >= 1
    for call in run.call_args_list:
        assert call.args[0][:2] != ["git", "rev-parse"]
        assert Path(call.kwargs["cwd"]).resolve() == REPO


@pytest.mark.asyncio
async def test_git_commits_empty_when_off_same_instance(monkeypatch):
    g = _gatherer()
    run = MagicMock(return_value=SimpleNamespace(returncode=0, stdout=""))
    monkeypatch.setattr("knowledge.git_memory.subprocess.run", run)
    monkeypatch.setattr("config.app_config.GIT_MEMORY_ENABLED", False)
    assert await g.get_git_commits("what changed in the repo") == []
    run.assert_not_called()


def test_active_features_reports_git_off(monkeypatch):
    _user_mode(monkeypatch)
    f = PromptFormatter(token_manager=MagicMock(), time_manager=None)
    out = f._build_feature_inventory({})
    assert "git_commits=OFF" in out and "git_commits=ON" not in out


def test_github_write_actions_not_enabled(monkeypatch):
    from core.actions.registry import ACTION_SPECS, enabled_action_types, get_runtime_action_health
    _dev_mode(monkeypatch)
    assert {a.value for a in enabled_action_types()} >= {"github_create_issue", "github_comment_pr"}
    _user_mode(monkeypatch)
    assert not {a.value for a in enabled_action_types()} & {"github_create_issue", "github_comment_pr"}
    assert "github" not in get_runtime_action_health().lower()


def test_git_stats_manager_never_resolves_from_cwd(monkeypatch, tmp_path):
    from core import git_stats_manager as gsm
    run = MagicMock(return_value=SimpleNamespace(returncode=0, stdout=str(tmp_path)))
    monkeypatch.setattr(gsm.subprocess, "run", run)
    monkeypatch.chdir(tmp_path)
    mgr = gsm.GitStatsManager()
    mgr.is_available()
    assert Path(run.call_args.kwargs["cwd"]).resolve() == REPO
    # Not a work tree at the module location -> unavailable, git never runs.
    run.reset_mock()
    monkeypatch.setattr(gsm.Path, "exists", lambda self: False)
    assert gsm.GitStatsManager().is_available() is False
    run.assert_not_called()


def _import_config(**env):
    e = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    e.update(DAEMON_TEST_MODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", **env)
    code = ("import config.app_config as c;"
            "print(','.join(str(getattr(c,n)) for n in %r), c.OBSIDIAN_ENABLED, repr(c.OBSIDIAN_VAULT_PATH))"
            % (DEV_CONSTANTS,))
    return subprocess.run([sys.executable, "-s", "-c", code], cwd=REPO, env=e,
                          capture_output=True, text=True, timeout=120)


def test_import_in_user_mode_forces_every_dev_constant_off():
    # Even explicit env enables must not resurrect them in user mode.
    r = _import_config(DAEMON_MODE="user", GIT_MEMORY_ENABLED="1", FILE_ACCESS_ENABLED="1",
                       GIT_STATS_ENABLED="1", GITHUB_API_ENABLED="1", SESSION_DIFF_ENABLED="1")
    assert r.returncode == 0, r.stderr[-800:]
    last = r.stdout.strip().splitlines()[-1]
    assert last.startswith("False,False,False,False,False,False"), last
    assert last.endswith("False ''"), last  # no guessed vault, indexer off


def test_dev_mode_keeps_dev_default_vault():
    r = _import_config(DAEMON_MODE="dev")
    assert r.returncode == 0, r.stderr[-800:]
    last = r.stdout.strip().splitlines()[-1]
    assert last.startswith("True,True,True,True,True,"), last  # dev surfaces keep their defaults
    assert "Documents/Notes" in last, last


def test_notes_base_never_uses_cwd_with_empty_vault(monkeypatch, tmp_path):
    from utils.notes_common import daily_notes_base
    monkeypatch.setattr("config.app_config.OBSIDIAN_ENABLED", True)
    monkeypatch.setattr("config.app_config.OBSIDIAN_VAULT_PATH", "")
    monkeypatch.setattr("config.app_config.DAILY_NOTES_FOLDER", "Daily")
    monkeypatch.setattr("config.app_config.CORPUS_FILE", str(tmp_path / "data" / "corpus.json"))
    monkeypatch.chdir(tmp_path)
    assert daily_notes_base() == tmp_path / "data" / "notes" / "Daily"
