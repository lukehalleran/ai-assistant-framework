"""Tests for utils.destructive_op_guard — git command classification and unlock."""

from pathlib import Path

import pytest

from utils.destructive_op_guard import (
    classify_git_args,
    is_destructive_git_args,
    unlock_allowed,
)


# ============================================================================
# Safe commands
# ============================================================================

class TestSafeCommands:
    @pytest.mark.parametrize("args", [
        ["status"],
        ["status", "--short"],
        ["diff"],
        ["diff", "--cached"],
        ["log", "--oneline", "-5"],
        ["show", "HEAD"],
        ["grep", "pattern"],
        ["ls-files"],
        ["rev-parse", "HEAD"],
        ["branch", "--show-current"],
        ["branch", "-a"],
        ["add", "file.py"],
        ["commit", "-m", "msg"],
        ["stash"],
        ["stash", "pop"],
        ["fetch"],
        ["remote", "-v"],
        ["checkout", "-b", "new-branch"],
        ["checkout", "main"],
        ["switch", "main"],
        ["switch", "-c", "new-branch"],
        ["reset", "HEAD~1"],  # soft reset is safe
        ["branch", "-d", "old-branch"],  # lowercase -d is safe
    ])
    def test_safe_commands_allowed(self, args):
        result = classify_git_args(args)
        assert result["destructive"] is False, f"Expected safe: git {' '.join(args)}"
        assert result["reason"] is None

    def test_empty_args_safe(self):
        result = classify_git_args([])
        assert result["destructive"] is False
        assert result["subcmd"] is None


# ============================================================================
# Destructive commands
# ============================================================================

class TestDestructiveCommands:
    @pytest.mark.parametrize("args,expected_reason_fragment", [
        (["restore", "."], "restore"),
        (["restore", "--staged", "file.py"], "restore"),
        (["clean", "-fd"], "clean"),
        (["clean"], "clean"),
        (["push"], "push"),
        (["push", "origin", "main"], "push"),
        (["push", "--force"], "push"),
        (["push", "-u", "origin", "feature"], "push"),
        (["reset", "--hard"], "reset --hard"),
        (["reset", "--hard", "HEAD~3"], "reset --hard"),
        (["reset", "--merge"], "reset --merge"),
        (["reset", "--keep"], "reset --keep"),
        (["checkout", "--", "file.py"], "checkout --"),
        (["checkout", "--", "."], "checkout --"),
        (["switch", "-C", "branch"], "switch -C"),
        (["branch", "-D", "old-branch"], "branch -D"),
    ])
    def test_destructive_commands_blocked(self, args, expected_reason_fragment):
        result = classify_git_args(args)
        assert result["destructive"] is True, f"Expected destructive: git {' '.join(args)}"
        assert expected_reason_fragment in result["reason"]

    def test_is_destructive_helper(self):
        assert is_destructive_git_args(["restore", "."]) is True
        assert is_destructive_git_args(["status"]) is False


# ============================================================================
# checkout edge cases
# ============================================================================

class TestCheckoutEdgeCases:
    def test_checkout_branch_safe(self):
        assert is_destructive_git_args(["checkout", "main"]) is False

    def test_checkout_new_branch_safe(self):
        assert is_destructive_git_args(["checkout", "-b", "feature"]) is False

    def test_checkout_dash_dash_destructive(self):
        assert is_destructive_git_args(["checkout", "--", "file.py"]) is True

    def test_checkout_B_flag_safe(self):
        assert is_destructive_git_args(["checkout", "-B", "branch"]) is False


# ============================================================================
# Unlock
# ============================================================================

class TestUnlock:
    def test_no_unlock_by_default(self, tmp_path):
        assert unlock_allowed(env={}, root=tmp_path) is False

    def test_env_var_unlock(self, tmp_path):
        assert unlock_allowed(env={"ALLOW_DESTRUCTIVE_OPS": "1"}, root=tmp_path) is True

    def test_env_var_wrong_value(self, tmp_path):
        assert unlock_allowed(env={"ALLOW_DESTRUCTIVE_OPS": "yes"}, root=tmp_path) is False

    def test_lockfile_unlock(self, tmp_path):
        lockfile = tmp_path / ".agent_allow_destructive_once"
        lockfile.touch()
        assert unlock_allowed(env={}, root=tmp_path) is True

    def test_lockfile_absent(self, tmp_path):
        assert unlock_allowed(env={}, root=tmp_path) is False

    def test_no_root_no_lockfile_check(self):
        # With root=None, only env var is checked
        assert unlock_allowed(env={}, root=None) is False
        assert unlock_allowed(env={"ALLOW_DESTRUCTIVE_OPS": "1"}, root=None) is True

    def test_either_unlock_works(self, tmp_path):
        lockfile = tmp_path / ".agent_allow_destructive_once"
        lockfile.touch()
        # Both present — should still be True
        assert unlock_allowed(env={"ALLOW_DESTRUCTIVE_OPS": "1"}, root=tmp_path) is True
