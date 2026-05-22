"""
Tests for agent_guard.py — standalone agent safety guard.

Module Contract
- Purpose: Verify all three guard layers (git classifier, shell classifier,
  Python FS monkey-patching) work correctly in isolation and together.
- Inputs: Synthetic repo structures via tmp_path, parameterized command lists.
- Outputs: Pass/fail assertions on classification dicts, PermissionError raises,
  and function pass-throughs.
- Dependencies: pytest, standard library only. No external fixtures or conftest.
"""

import asyncio
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent_guard


# ============================================================================
# Test-local constants (equivalent to the user's config)
# ============================================================================

TEST_PROTECTED_DIRS = frozenset({
    "data", "config", ".git", "src", "core", "tests",
})

TEST_PROTECTED_FILES = frozenset({
    "main.py", ".env", "requirements.txt",
})


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def repo(tmp_path):
    """Create a synthetic repo structure for path resolution tests."""
    for d in ("data", "config", "src", "core", "tests", ".git", "build", "tmp_work"):
        (tmp_path / d).mkdir()
    for f in ("main.py", ".env", "requirements.txt", "scratch.log", "tempfile.txt"):
        (tmp_path / f).write_text("content", encoding="utf-8")
    (tmp_path / "data" / "important.json").write_text("{}", encoding="utf-8")
    (tmp_path / "config" / "settings.yaml").write_text("key: val", encoding="utf-8")
    (tmp_path / "src" / "app.py").write_text("# app", encoding="utf-8")
    (tmp_path / "build" / "artifact.o").write_text("", encoding="utf-8")
    return tmp_path


@pytest.fixture
def guarded(repo):
    """Activate guard with test config, yield, then deactivate."""
    agent_guard.activate(
        repo_root=repo,
        protected_dirs=TEST_PROTECTED_DIRS,
        protected_files=TEST_PROTECTED_FILES,
    )
    yield repo
    agent_guard.deactivate()


@pytest.fixture
def guarded_no_agent(guarded):
    """Guard is active but agent_mode is OFF (default state)."""
    return guarded


# ############################################################################
#
#  Git command classifier
#
# ############################################################################

class TestGitClassifier:
    """Tests for classify_git_command() and is_destructive_git()."""

    @pytest.mark.parametrize("args", [
        ["status"], ["diff"], ["log", "--oneline"], ["show", "HEAD"],
        ["add", "."], ["commit", "-m", "msg"], ["stash"],
        ["fetch", "origin"], ["pull"], ["branch"], ["branch", "-a"],
        ["remote", "-v"], ["tag", "v1.0"],
    ])
    def test_safe_commands(self, args):
        result = agent_guard.classify_git_command(args)
        assert result["destructive"] is False

    def test_empty_args_safe(self):
        result = agent_guard.classify_git_command([])
        assert result["destructive"] is False
        assert result["subcmd"] is None

    @pytest.mark.parametrize("args", [
        ["push"], ["push", "origin", "main"],
        ["restore", "."], ["restore", "--staged", "file.py"],
        ["clean", "-fd"],
        ["reset", "--hard"], ["reset", "--hard", "HEAD~1"],
        ["branch", "-D", "old-branch"],
        ["switch", "-C", "new-branch"],
    ])
    def test_destructive_commands(self, args):
        result = agent_guard.classify_git_command(args)
        assert result["destructive"] is True
        assert result["reason"] is not None

    def test_is_destructive_helper(self):
        assert agent_guard.is_destructive_git(["push"]) is True
        assert agent_guard.is_destructive_git(["status"]) is False

    # --- checkout edge cases ---

    def test_checkout_branch_safe(self):
        assert agent_guard.classify_git_command(["checkout", "main"])["destructive"] is False

    def test_checkout_new_branch_safe(self):
        assert agent_guard.classify_git_command(["checkout", "-b", "feat"])["destructive"] is False

    def test_checkout_B_flag_safe(self):
        assert agent_guard.classify_git_command(["checkout", "-B", "feat"])["destructive"] is False

    def test_checkout_dash_dash_destructive(self):
        result = agent_guard.classify_git_command(["checkout", "--", "file.py"])
        assert result["destructive"] is True

    # --- reset without destructive flags is safe ---

    def test_reset_soft_safe(self):
        assert agent_guard.classify_git_command(["reset", "--soft", "HEAD~1"])["destructive"] is False

    def test_reset_no_flags_safe(self):
        assert agent_guard.classify_git_command(["reset", "HEAD~1"])["destructive"] is False


# ############################################################################
#
#  Unlock check
#
# ############################################################################

class TestUnlock:

    def test_no_unlock_by_default(self):
        assert agent_guard.unlock_allowed(env={}) is False

    def test_env_var_unlock(self):
        assert agent_guard.unlock_allowed(env={"ALLOW_DESTRUCTIVE_OPS": "1"}) is True

    def test_env_var_wrong_value(self):
        assert agent_guard.unlock_allowed(env={"ALLOW_DESTRUCTIVE_OPS": "yes"}) is False

    def test_lockfile_unlock(self, tmp_path):
        lockfile = tmp_path / ".agent_allow_destructive_once"
        lockfile.touch()
        assert agent_guard.unlock_allowed(env={}, root=tmp_path) is True

    def test_lockfile_absent(self, tmp_path):
        assert agent_guard.unlock_allowed(env={}, root=tmp_path) is False

    def test_no_root_no_lockfile_check(self):
        assert agent_guard.unlock_allowed(env={}, root=None) is False

    def test_custom_lockfile_name(self, tmp_path):
        (tmp_path / "custom_lock").touch()
        assert agent_guard.unlock_allowed(env={}, root=tmp_path, lockfile_name="custom_lock") is True


# ############################################################################
#
#  Shell command classifier
#
# ############################################################################

class TestShellClassifierSafe:
    """Commands that should be classified as safe."""

    @pytest.mark.parametrize("args", [
        ["ls", "-la"], ["cat", "file.txt"], ["echo", "hello"],
        ["cp", "a.txt", "b.txt"], ["mkdir", "newdir"], ["touch", "file"],
        ["grep", "pattern", "file"], ["python", "script.py"],
        ["git", "status"], ["curl", "http://example.com"],
    ])
    def test_safe_commands(self, args, repo):
        result = agent_guard.classify_shell_command(
            args, repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is False

    def test_empty_args(self):
        assert agent_guard.classify_shell_command([])["destructive"] is False

    def test_rm_non_protected(self, repo):
        result = agent_guard.classify_shell_command(
            ["rm", "scratch.log"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is False

    def test_rm_rf_non_protected_dir(self, repo):
        result = agent_guard.classify_shell_command(
            ["rm", "-rf", "build"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is False

    def test_mv_non_protected(self, repo):
        result = agent_guard.classify_shell_command(
            ["mv", "scratch.log", "renamed.log"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is False

    def test_chmod_safe_mode_on_protected(self, repo):
        result = agent_guard.classify_shell_command(
            ["chmod", "755", "main.py"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is False

    def test_find_without_delete(self, repo):
        result = agent_guard.classify_shell_command(
            ["find", "data", "-name", "*.json"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is False


class TestShellClassifierAlwaysBlocked:
    """Catastrophic targets that are always blocked."""

    @pytest.mark.parametrize("target", [".", "..", "/", "~", "*"])
    def test_rm_rf_always_blocked(self, target, repo):
        result = agent_guard.classify_shell_command(
            ["rm", "-rf", target], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True
        assert result["severity"] == "always"

    def test_rmdir_dot(self, repo):
        result = agent_guard.classify_shell_command(
            ["rmdir", "."], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True

    def test_mv_dot(self, repo):
        result = agent_guard.classify_shell_command(
            ["mv", ".", "/tmp/"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True

    def test_find_delete_on_slash(self, repo):
        result = agent_guard.classify_shell_command(
            ["find", "/", "-delete"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True


class TestShellClassifierProtected:
    """Protected paths that are blocked but unlockable."""

    @pytest.mark.parametrize("target", ["data", "config", "src", "core", "tests", ".git"])
    def test_rm_rf_protected_dirs(self, target, repo):
        result = agent_guard.classify_shell_command(
            ["rm", "-rf", target], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_rm_protected_file(self, repo):
        result = agent_guard.classify_shell_command(
            ["rm", "main.py"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True

    def test_rm_env_file(self, repo):
        result = agent_guard.classify_shell_command(
            ["rm", ".env"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True

    def test_rm_nested_protected(self, repo):
        result = agent_guard.classify_shell_command(
            ["rm", "data/important.json"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True

    def test_mv_protected_dir(self, repo):
        result = agent_guard.classify_shell_command(
            ["mv", "config", "/tmp/backup"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True

    def test_mv_protected_file(self, repo):
        result = agent_guard.classify_shell_command(
            ["mv", "main.py", "old_main.py"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True

    def test_rmdir_protected(self, repo):
        result = agent_guard.classify_shell_command(
            ["rmdir", "config"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True

    def test_chmod_000_protected(self, repo):
        result = agent_guard.classify_shell_command(
            ["chmod", "000", "main.py"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True

    def test_chmod_recursive_protected(self, repo):
        result = agent_guard.classify_shell_command(
            ["chmod", "-R", "755", "data"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True

    def test_truncate_protected(self, repo):
        result = agent_guard.classify_shell_command(
            ["truncate", "-s", "0", "main.py"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True

    def test_find_delete_protected(self, repo):
        result = agent_guard.classify_shell_command(
            ["find", "data", "-name", "*.tmp", "-delete"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True

    def test_find_exec_rm_protected(self, repo):
        result = agent_guard.classify_shell_command(
            ["find", "data", "-exec", "rm", "{}", ";"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True

    def test_rm_long_form_flags(self, repo):
        result = agent_guard.classify_shell_command(
            ["rm", "--recursive", "--force", "data"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True

    def test_rm_path_traversal(self, repo):
        result = agent_guard.classify_shell_command(
            ["rm", "-rf", "data/../config"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True


class TestShellHelpers:
    """Tests for path resolution and flag parsing internals."""

    def test_resolve_relative(self, repo):
        result = agent_guard._resolve_target("data", repo)
        assert result == "data"

    def test_resolve_nested(self, repo):
        result = agent_guard._resolve_target("data/subdir/file.txt", repo)
        assert result == "data/subdir/file.txt"

    def test_resolve_absolute_inside(self, repo):
        result = agent_guard._resolve_target(str(repo / "data"), repo)
        assert result == "data"

    def test_resolve_absolute_outside(self, repo):
        result = agent_guard._resolve_target("/etc/passwd", repo)
        assert result is None

    def test_resolve_tilde(self, repo):
        result = agent_guard._resolve_target("~/stuff", repo)
        assert result is None

    def test_resolve_traversal_in_repo(self, repo):
        result = agent_guard._resolve_target("data/../config", repo)
        assert result == "config"

    def test_resolve_traversal_escapes(self, repo):
        result = agent_guard._resolve_target("../../etc/passwd", repo)
        assert result is None

    def test_resolve_empty(self, repo):
        result = agent_guard._resolve_target("", repo)
        assert result is None

    def test_resolve_dash_dash(self, repo):
        result = agent_guard._resolve_target("--", repo)
        assert result is None

    def test_is_protected_dir(self):
        assert agent_guard._is_protected("data", TEST_PROTECTED_DIRS, TEST_PROTECTED_FILES) is True

    def test_is_protected_subpath(self):
        assert agent_guard._is_protected("data/file.json", TEST_PROTECTED_DIRS, TEST_PROTECTED_FILES) is True

    def test_is_protected_file(self):
        assert agent_guard._is_protected("main.py", TEST_PROTECTED_DIRS, TEST_PROTECTED_FILES) is True

    def test_not_protected(self):
        assert agent_guard._is_protected("build", TEST_PROTECTED_DIRS, TEST_PROTECTED_FILES) is False

    def test_parse_rm_flags_rf(self):
        assert agent_guard._parse_rm_flags(["-rf"]) == (True, True)

    def test_parse_rm_flags_separate(self):
        assert agent_guard._parse_rm_flags(["-r", "-f"]) == (True, True)

    def test_parse_rm_flags_long(self):
        assert agent_guard._parse_rm_flags(["--recursive", "--force"]) == (True, True)

    def test_parse_rm_flags_none(self):
        assert agent_guard._parse_rm_flags([]) == (False, False)

    def test_parse_rm_flags_uppercase_R(self):
        r, f = agent_guard._parse_rm_flags(["-Rf"])
        assert r is True


class TestShellEdgeCases:
    """Edge cases for shell command classification."""

    def test_absolute_path_to_binary(self, repo):
        result = agent_guard.classify_shell_command(
            ["/usr/bin/rm", "-rf", "data"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is True

    def test_rm_no_targets(self, repo):
        result = agent_guard.classify_shell_command(
            ["rm"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is False

    def test_mv_single_arg(self, repo):
        result = agent_guard.classify_shell_command(
            ["mv", "data"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is False

    def test_rm_outside_repo(self, repo):
        result = agent_guard.classify_shell_command(
            ["rm", "-rf", "/tmp/random"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is False

    def test_unknown_command_safe(self, repo):
        result = agent_guard.classify_shell_command(
            ["wget", "http://example.com"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        )
        assert result["destructive"] is False

    def test_is_destructive_shell_convenience(self, repo):
        assert agent_guard.is_destructive_shell(
            ["rm", "-rf", "data"], repo_root=repo,
            protected_dirs=TEST_PROTECTED_DIRS,
            protected_files=TEST_PROTECTED_FILES,
        ) is True

    def test_uses_stored_config_after_activate(self, guarded):
        """classify_shell_command() falls back to stored config."""
        result = agent_guard.classify_shell_command(["rm", "main.py"])
        assert result["destructive"] is True


# ############################################################################
#
#  Python filesystem guard
#
# ############################################################################

class TestFSGuardActivation:

    def test_activate_sets_active(self, repo):
        agent_guard.activate(repo_root=repo, protected_dirs={"data"})
        assert agent_guard.is_active() is True
        agent_guard.deactivate()

    def test_deactivate_clears_active(self, guarded):
        agent_guard.deactivate()
        assert agent_guard.is_active() is False

    def test_activate_idempotent(self, repo):
        agent_guard.activate(repo_root=repo, protected_dirs={"data"})
        agent_guard.activate(repo_root=repo, protected_dirs={"data"})  # no-op
        assert agent_guard.is_active() is True
        agent_guard.deactivate()

    def test_deactivate_idempotent(self, repo):
        agent_guard.activate(repo_root=repo)
        agent_guard.deactivate()
        agent_guard.deactivate()  # no-op
        assert agent_guard.is_active() is False

    def test_deactivate_restores_originals(self, repo):
        orig_remove = os.remove
        agent_guard.activate(repo_root=repo)
        assert os.remove is not orig_remove
        agent_guard.deactivate()
        assert os.remove is orig_remove


class TestFSGuardAgentModeOff:
    """With guard active but agent_mode OFF, everything passes through."""

    def test_remove_protected_allowed(self, guarded_no_agent):
        """Protected file removal allowed when NOT in agent mode."""
        target = guarded_no_agent / "data" / "important.json"
        os.remove(target)
        assert not target.exists()

    def test_rmtree_protected_allowed(self, guarded_no_agent):
        target = guarded_no_agent / "src"
        shutil.rmtree(target)
        assert not target.exists()

    def test_rename_protected_allowed(self, guarded_no_agent):
        src = guarded_no_agent / "main.py"
        dst = guarded_no_agent / "main_old.py"
        os.rename(src, dst)
        assert dst.exists()


class TestFSGuardBlocked:
    """Operations that should be blocked in agent_mode."""

    def test_os_remove_protected_file(self, guarded):
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="protected"):
                os.remove(guarded / "data" / "important.json")

    def test_os_unlink_protected_file(self, guarded):
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="protected"):
                os.unlink(guarded / "config" / "settings.yaml")

    def test_os_rmdir_protected_dir(self, guarded):
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="protected"):
                os.rmdir(guarded / "src")

    def test_shutil_rmtree_protected(self, guarded):
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="protected"):
                shutil.rmtree(guarded / "data")

    def test_os_remove_root_file(self, guarded):
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="protected"):
                os.remove(guarded / "main.py")

    def test_os_remove_env(self, guarded):
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="protected"):
                os.remove(guarded / ".env")

    def test_os_remove_nested_protected(self, guarded):
        (guarded / "data" / "sub").mkdir()
        (guarded / "data" / "sub" / "deep.txt").write_text("x")
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="protected"):
                os.remove(guarded / "data" / "sub" / "deep.txt")

    def test_file_survives_block(self, guarded):
        target = guarded / "main.py"
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError):
                os.remove(target)
        assert target.exists()


class TestFSGuardAllowed:
    """Operations on non-protected paths should pass through in agent_mode."""

    def test_remove_non_protected(self, guarded):
        target = guarded / "scratch.log"
        with agent_guard.agent_mode():
            os.remove(target)
        assert not target.exists()

    def test_remove_in_non_protected_dir(self, guarded):
        target = guarded / "build" / "artifact.o"
        with agent_guard.agent_mode():
            os.remove(target)
        assert not target.exists()

    def test_rmtree_non_protected(self, guarded):
        target = guarded / "build"
        with agent_guard.agent_mode():
            shutil.rmtree(target)
        assert not target.exists()

    def test_remove_outside_repo(self, guarded, tmp_path):
        outside = tmp_path / "outside_file.txt"
        outside.write_text("x")
        with agent_guard.agent_mode():
            os.remove(outside)
        assert not outside.exists()


class TestFSGuardAlwaysBlocked:
    """Always-blocked targets — never unlockable."""

    def test_slash_blocked(self, guarded):
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="always blocked"):
                os.remove("/")

    def test_dot_blocked(self, guarded):
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="always blocked"):
                os.remove(".")

    def test_dotdot_blocked(self, guarded):
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="always blocked"):
                os.remove("..")

    def test_tilde_blocked(self, guarded):
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="always blocked"):
                os.remove("~")

    def test_star_blocked(self, guarded):
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="always blocked"):
                os.remove("*")

    def test_repo_root_blocked(self, guarded):
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="always blocked"):
                shutil.rmtree(str(guarded))

    def test_always_blocked_ignores_unlock(self, guarded):
        with patch.dict(os.environ, {"ALLOW_DESTRUCTIVE_OPS": "1"}):
            with agent_guard.agent_mode():
                with pytest.raises(PermissionError, match="always blocked"):
                    os.remove(".")


class TestFSGuardDestinations:
    """Destination protection for rename/replace/move/copy."""

    def test_rename_to_protected_blocked(self, guarded):
        src = guarded / "scratch.log"
        dst = guarded / "data" / "renamed.log"
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="protected"):
                os.rename(src, dst)

    def test_replace_to_protected_blocked(self, guarded):
        src = guarded / "scratch.log"
        dst = guarded / "config" / "replaced.yaml"
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="protected"):
                os.replace(src, dst)

    def test_move_to_protected_blocked(self, guarded):
        src = guarded / "scratch.log"
        dst = guarded / "data" / "moved.log"
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="protected"):
                shutil.move(str(src), str(dst))

    def test_move_from_protected_blocked(self, guarded):
        src = guarded / "data" / "important.json"
        dst = guarded / "build" / "moved.json"
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="protected"):
                shutil.move(str(src), str(dst))

    def test_rename_both_unprotected_allowed(self, guarded):
        src = guarded / "scratch.log"
        dst = guarded / "build" / "renamed.log"
        with agent_guard.agent_mode():
            os.rename(src, dst)
        assert dst.exists()

    def test_copyfile_to_protected_blocked(self, guarded):
        src = guarded / "scratch.log"
        dst = guarded / "data" / "copy.log"
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="protected"):
                shutil.copyfile(str(src), str(dst))

    def test_copy_to_protected_blocked(self, guarded):
        src = guarded / "scratch.log"
        dst = guarded / "data" / "copy.log"
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="protected"):
                shutil.copy(str(src), str(dst))

    def test_copy2_to_protected_blocked(self, guarded):
        src = guarded / "scratch.log"
        dst = guarded / "data" / "copy.log"
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError, match="protected"):
                shutil.copy2(str(src), str(dst))

    def test_copyfile_to_non_protected_allowed(self, guarded):
        src = guarded / "scratch.log"
        dst = guarded / "build" / "copy.log"
        with agent_guard.agent_mode():
            shutil.copyfile(str(src), str(dst))
        assert dst.exists()


class TestFSGuardUnlock:
    """Unlock mechanism for protected paths."""

    def test_env_var_unlocks(self, guarded):
        with patch.dict(os.environ, {"ALLOW_DESTRUCTIVE_OPS": "1"}):
            with agent_guard.agent_mode():
                target = guarded / "data" / "important.json"
                os.remove(target)
                assert not target.exists()

    def test_lockfile_unlocks(self, guarded):
        (guarded / ".agent_allow_destructive_once").touch()
        with agent_guard.agent_mode():
            target = guarded / "data" / "important.json"
            os.remove(target)
            assert not target.exists()

    def test_unlock_does_not_affect_always_blocked(self, guarded):
        with patch.dict(os.environ, {"ALLOW_DESTRUCTIVE_OPS": "1"}):
            with agent_guard.agent_mode():
                with pytest.raises(PermissionError):
                    os.remove(".")


class TestFSGuardContextManager:
    """agent_mode() context manager behavior."""

    def test_sets_and_clears(self, guarded):
        assert agent_guard._agent_mode.get(False) is False
        with agent_guard.agent_mode():
            assert agent_guard._agent_mode.get(False) is True
        assert agent_guard._agent_mode.get(False) is False

    def test_resets_after_exception(self, guarded):
        try:
            with agent_guard.agent_mode():
                raise ValueError("test")
        except ValueError:
            pass
        assert agent_guard._agent_mode.get(False) is False

    def test_nested_contexts(self, guarded):
        with agent_guard.agent_mode():
            with agent_guard.agent_mode():
                assert agent_guard._agent_mode.get(False) is True
            assert agent_guard._agent_mode.get(False) is True
        assert agent_guard._agent_mode.get(False) is False


class TestFSGuardAsync:
    """ContextVar propagation through asyncio."""

    def test_propagates_through_gather(self, guarded):
        results = []

        async def check_mode():
            results.append(agent_guard._agent_mode.get(False))

        async def main():
            with agent_guard.agent_mode():
                await asyncio.gather(check_mode(), check_mode())

        asyncio.run(main())
        assert all(results), f"Expected all True, got {results}"

    def test_off_propagates_through_gather(self, guarded):
        results = []

        async def check_mode():
            results.append(agent_guard._agent_mode.get(False))

        async def main():
            await asyncio.gather(check_mode(), check_mode())

        asyncio.run(main())
        assert not any(results), f"Expected all False, got {results}"


class TestFSGuardPathlib:
    """Path.unlink(), Path.rmdir(), Path.rename() delegation."""

    def test_path_unlink_blocked(self, guarded):
        target = guarded / "main.py"
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError):
                target.unlink()

    def test_path_rmdir_blocked(self, guarded):
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError):
                (guarded / "data").rmdir()

    def test_path_rename_blocked(self, guarded):
        with agent_guard.agent_mode():
            with pytest.raises(PermissionError):
                (guarded / "main.py").rename(guarded / "old.py")

    def test_path_unlink_non_protected_allowed(self, guarded):
        target = guarded / "scratch.log"
        with agent_guard.agent_mode():
            target.unlink()
        assert not target.exists()


class TestFSGuardInactive:
    """When guard is not activated, nothing is blocked."""

    def test_no_block_when_inactive(self, repo):
        assert not agent_guard.is_active()
        target = repo / "data" / "important.json"
        agent_guard.set_agent_mode(True)
        try:
            os.remove(target)
            assert not target.exists()
        finally:
            agent_guard.set_agent_mode(False)
