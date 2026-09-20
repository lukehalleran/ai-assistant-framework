"""Tests for scripts/safe_cmd.sh — drives the DEPLOYED wrapper script end to
end (real subprocess, real classifier import), never a reimplementation of
its shell logic.

Regression for the bug shipped 2026-05-21 and fixed 2026-09-19: `set -euo
pipefail` made `classify_cmd`'s
    result=$(python3 -c "..." "$args_json" 2>&1)
kill the whole script the instant the classifier process exited nonzero
(protected=1, always=2) — BEFORE `CLASSIFY_EXIT=$?` and the BLOCKED /
PERMANENTLY BLOCKED messages could run. Every protected-path command was
silently refused, with or without the documented ALLOW_DESTRUCTIVE_OPS=1 env
var or the one-shot `.agent_allow_destructive_once` lockfile — the wrapper
was fail-closed but its own documented unlock mechanisms never executed.

class: BC-63 (a check that never exercised the deployed wrapper), BC-58
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT_DEFAULT = Path(__file__).resolve().parents[2] / "scripts" / "safe_cmd.sh"
GUARD_SOURCE = Path(__file__).resolve().parents[2] / "utils" / "shell_cmd_guard.py"

pytestmark = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("git") and shutil.which("python3")),
    reason="bash, git and python3 are all required to drive the deployed safe_cmd.sh",
)


def _script() -> Path:
    """The script under test.

    SAFE_CMD_SCRIPT is an override path used ONLY to gather failed-before
    evidence against a different checkout (e.g. the pre-fix baseline clone);
    it is never set by the tests themselves.
    """
    override = os.environ.get("SAFE_CMD_SCRIPT")
    return Path(override) if override else SCRIPT_DEFAULT


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A throwaway git repo under tmp_path carrying a real copy of the deployed
    classifier, so the script's `from utils.shell_cmd_guard import
    classify_shell_cmd` resolves against the real deployed logic (not a stub).
    Never touches anything outside tmp_path.
    """
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for d in ("config", "data", "tmp_work", "utils"):
        (tmp_path / d).mkdir()
    (tmp_path / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "utils" / "shell_cmd_guard.py").write_text(
        GUARD_SOURCE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "config" / "config.yaml").write_text("key: value\n", encoding="utf-8")
    (tmp_path / "data" / "corpus_v4.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tmp_work" / "scratch.txt").write_text("scratch\n", encoding="utf-8")
    return tmp_path


def _run(sandbox: Path, args: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    # Built with os.pathsep.join rather than one colon-joined literal: a single
    # quoted string shaped like two slash-paths separated by a colon
    # false-positives the project's git-ref-blob-read guard
    # (test_no_git_state_in_tests.py), which flags any quoted text shaped that
    # way as a possible git ref-qualified blob path.
    search_path = os.pathsep.join(("/usr/bin", "/bin"))
    env = {"PATH": search_path, "HOME": str(sandbox), **(env_extra or {})}
    return subprocess.run(
        ["bash", str(_script()), *args],
        cwd=sandbox,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


class TestSafeCmdWrapperDeployed:
    def test_locked_protected_path_is_blocked(self, sandbox: Path):
        target = sandbox / "config" / "config.yaml"
        result = _run(sandbox, ["rm", "config/config.yaml"])
        assert result.returncode == 1
        assert "BLOCKED destructive command" in result.stdout
        assert target.exists()

    def test_env_unlock_allows_the_command(self, sandbox: Path):
        target = sandbox / "config" / "config.yaml"
        result = _run(sandbox, ["rm", "config/config.yaml"], env_extra={"ALLOW_DESTRUCTIVE_OPS": "1"})
        assert result.returncode == 0
        assert "explicit unlock" in result.stdout
        assert not target.exists()

    def test_lockfile_unlock_is_consumed(self, sandbox: Path):
        target = sandbox / "config" / "config.yaml"
        lockfile = sandbox / ".agent_allow_destructive_once"
        lockfile.touch()
        result = _run(sandbox, ["rm", "config/config.yaml"])
        assert result.returncode == 0
        assert not target.exists()
        assert not lockfile.exists()
        assert "lockfile consumed" in result.stdout

    def test_always_blocked_target_ignores_unlock(self, sandbox: Path):
        marker = sandbox / "config" / "config.yaml"
        result = _run(sandbox, ["rm", "-rf", "."], env_extra={"ALLOW_DESTRUCTIVE_OPS": "1"})
        assert result.returncode == 1
        assert "PERMANENTLY BLOCKED" in result.stdout
        assert marker.exists()

    def test_safe_command_passes_through_silently(self, sandbox: Path):
        target = sandbox / "tmp_work" / "scratch.txt"
        result = _run(sandbox, ["rm", "tmp_work/scratch.txt"])
        assert result.returncode == 0
        assert not target.exists()
        assert "[safe-cmd]" not in result.stdout
