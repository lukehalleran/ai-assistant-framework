"""Tests for utils.python_fs_guard — Python-level filesystem guard.

All tests use tmp_path fixtures with synthetic directory structures.
No real data directories or files are ever touched.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.python_fs_guard import (
    activate,
    agent_mode,
    deactivate,
    is_active,
    set_agent_mode,
    _agent_mode as _agent_mode_var,
    _check_and_maybe_block,
    _is_always_blocked,
    _is_protected_path,
    _originals,
    _resolve_to_repo_relative,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def repo(tmp_path):
    """Create a synthetic repo structure for testing."""
    for d in ("data", "config", "core", "memory", "scripts", "utils",
              "models", "docs", "tests", "knowledge", "gui", "eval",
              "build", "tmp_work"):
        (tmp_path / d).mkdir()
    for f in ("main.py", "CLAUDE.md", "requirements.txt", ".env",
              "pytest.ini", "daemon.spec", "tempfile.txt", "scratch.log"):
        (tmp_path / f).touch()
    (tmp_path / "data" / "knowledge_graph.json").touch()
    (tmp_path / "data" / "corpus_v4.json").touch()
    (tmp_path / "config" / "config.yaml").touch()
    (tmp_path / "build" / "artifact.o").touch()
    (tmp_path / "tmp_work" / "scratch.txt").touch()
    return tmp_path


@pytest.fixture
def guarded(repo):
    """Activate guard with repo_root=repo, agent mode ON, yield, then clean up."""
    activate(repo_root=repo)
    token = _agent_mode_var.set(True)
    yield repo
    _agent_mode_var.reset(token)
    deactivate()


@pytest.fixture
def guarded_no_agent(repo):
    """Activate guard with repo_root=repo, agent mode OFF."""
    activate(repo_root=repo)
    yield repo
    deactivate()


# ============================================================================
# TestActivation
# ============================================================================

class TestActivation:
    def test_activate_sets_active(self, repo):
        activate(repo_root=repo)
        assert is_active() is True
        deactivate()

    def test_deactivate_clears_active(self, repo):
        activate(repo_root=repo)
        deactivate()
        assert is_active() is False

    def test_activate_idempotent(self, repo):
        activate(repo_root=repo)
        activate(repo_root=repo)  # second call is no-op
        assert is_active() is True
        deactivate()
        assert is_active() is False

    def test_deactivate_idempotent(self, repo):
        activate(repo_root=repo)
        deactivate()
        deactivate()  # second call is no-op
        assert is_active() is False

    def test_deactivate_restores_originals(self, repo):
        orig_remove = os.remove
        orig_unlink = os.unlink
        orig_rmdir = os.rmdir
        orig_rename = os.rename
        orig_replace = os.replace
        orig_rmtree = shutil.rmtree
        orig_move = shutil.move

        activate(repo_root=repo)
        # Functions should be replaced
        assert os.remove is not orig_remove
        assert os.unlink is not orig_unlink

        deactivate()
        # Functions should be restored to exact originals
        assert os.remove is orig_remove
        assert os.unlink is orig_unlink
        assert os.rmdir is orig_rmdir
        assert os.rename is orig_rename
        assert os.replace is orig_replace
        assert shutil.rmtree is orig_rmtree
        assert shutil.move is orig_move


# ============================================================================
# TestAgentModeOff — Daemon runtime is unguarded
# ============================================================================

class TestAgentModeOff:
    """When agent mode is OFF, all operations pass through regardless of target."""

    def test_remove_protected_file_allowed(self, guarded_no_agent):
        """os.remove on protected file passes through when not in agent mode."""
        target = guarded_no_agent / "data" / "knowledge_graph.json"
        assert target.exists()
        os.remove(target)
        assert not target.exists()

    def test_unlink_protected_file_allowed(self, guarded_no_agent):
        target = guarded_no_agent / "config" / "config.yaml"
        assert target.exists()
        os.unlink(target)
        assert not target.exists()

    def test_rmtree_protected_dir_allowed(self, guarded_no_agent):
        target = guarded_no_agent / "build"
        assert target.exists()
        shutil.rmtree(target)
        assert not target.exists()

    def test_rename_protected_allowed(self, guarded_no_agent):
        src = guarded_no_agent / "tempfile.txt"
        dst = guarded_no_agent / "main.py"  # protected destination
        os.rename(src, dst)
        assert dst.exists()

    def test_move_protected_source_allowed(self, guarded_no_agent):
        src = guarded_no_agent / "data" / "corpus_v4.json"
        dst = guarded_no_agent / "tmp_work" / "corpus_backup.json"
        shutil.move(str(src), str(dst))
        assert dst.exists()


# ============================================================================
# TestBlockedInAgentMode — single-path operations
# ============================================================================

class TestBlockedInAgentMode:
    """In agent mode, operations on protected paths are blocked."""

    def test_os_remove_protected_file(self, guarded):
        with pytest.raises(PermissionError, match="protected"):
            os.remove(guarded / "data" / "knowledge_graph.json")

    def test_os_unlink_protected_file(self, guarded):
        with pytest.raises(PermissionError, match="protected"):
            os.unlink(guarded / "config" / "config.yaml")

    def test_os_rmdir_protected_dir(self, guarded):
        with pytest.raises(PermissionError, match="protected"):
            os.rmdir(guarded / "memory")

    def test_shutil_rmtree_protected_dir(self, guarded):
        with pytest.raises(PermissionError, match="protected"):
            shutil.rmtree(guarded / "scripts")

    def test_os_remove_protected_root_file(self, guarded):
        with pytest.raises(PermissionError, match="protected"):
            os.remove(guarded / "main.py")

    def test_os_remove_claude_md(self, guarded):
        with pytest.raises(PermissionError, match="protected"):
            os.remove(guarded / "CLAUDE.md")

    def test_os_remove_env(self, guarded):
        with pytest.raises(PermissionError, match="protected"):
            os.remove(guarded / ".env")

    def test_os_remove_nested_protected(self, guarded):
        """File deep inside a protected dir should be blocked."""
        nested = guarded / "data" / "sub" / "deep" / "file.txt"
        nested.parent.mkdir(parents=True)
        nested.touch()
        with pytest.raises(PermissionError, match="protected"):
            os.remove(nested)

    def test_file_survives_block(self, guarded):
        """Blocked operations should not actually delete the file."""
        target = guarded / "main.py"
        assert target.exists()
        with pytest.raises(PermissionError):
            os.remove(target)
        assert target.exists()  # still there


# ============================================================================
# TestDestinationProtection — dual-path operations
# ============================================================================

class TestDestinationProtection:
    """rename/replace/move check BOTH source AND destination."""

    def test_os_rename_unprotected_to_protected_blocks(self, guarded):
        """Renaming an unprotected file onto a protected file is blocked."""
        src = guarded / "tempfile.txt"
        dst = guarded / "CLAUDE.md"
        with pytest.raises(PermissionError, match="destination"):
            os.rename(src, dst)

    def test_os_replace_unprotected_to_protected_blocks(self, guarded):
        src = guarded / "tempfile.txt"
        dst = guarded / "main.py"
        with pytest.raises(PermissionError, match="destination"):
            os.replace(src, dst)

    def test_shutil_move_unprotected_to_protected_blocks(self, guarded):
        src = guarded / "tempfile.txt"
        dst = guarded / "config" / "config.yaml"
        with pytest.raises(PermissionError, match="destination"):
            shutil.move(str(src), str(dst))

    def test_shutil_move_protected_source_blocks(self, guarded):
        src = guarded / "data" / "corpus_v4.json"
        dst = guarded / "tmp_work" / "backup.json"
        with pytest.raises(PermissionError, match="source"):
            shutil.move(str(src), str(dst))

    def test_os_rename_protected_source_blocks(self, guarded):
        src = guarded / "main.py"
        dst = guarded / "old_main.py"
        with pytest.raises(PermissionError, match="source"):
            os.rename(src, dst)

    def test_os_replace_protected_source_blocks(self, guarded):
        src = guarded / "data" / "knowledge_graph.json"
        dst = guarded / "tmp_work" / "kg_backup.json"
        with pytest.raises(PermissionError, match="source"):
            os.replace(src, dst)

    def test_rename_both_unprotected_allowed(self, guarded):
        """Renaming between unprotected paths should pass through."""
        src = guarded / "tempfile.txt"
        dst = guarded / "renamed.txt"
        os.rename(src, dst)
        assert dst.exists()
        assert not src.exists()

    def test_move_both_unprotected_allowed(self, guarded):
        src = guarded / "scratch.log"
        dst = guarded / "tmp_work" / "scratch.log"
        shutil.move(str(src), str(dst))
        assert dst.exists()


# ============================================================================
# TestAllowedInAgentMode — non-protected paths
# ============================================================================

class TestAllowedInAgentMode:
    """Operations on non-protected paths pass through even in agent mode."""

    def test_remove_non_protected_file(self, guarded):
        target = guarded / "tempfile.txt"
        assert target.exists()
        os.remove(target)
        assert not target.exists()

    def test_remove_file_in_non_protected_dir(self, guarded):
        target = guarded / "build" / "artifact.o"
        assert target.exists()
        os.remove(target)
        assert not target.exists()

    def test_rmtree_non_protected_dir(self, guarded):
        target = guarded / "build"
        assert target.exists()
        shutil.rmtree(target)
        assert not target.exists()

    def test_remove_outside_repo(self, guarded, tmp_path):
        """Files outside repo root should always pass through."""
        outside = tmp_path / "outside_repo" / "file.txt"
        outside.parent.mkdir(parents=True)
        outside.touch()
        os.remove(outside)
        assert not outside.exists()

    def test_rmdir_non_protected(self, guarded):
        target = guarded / "tmp_work" / "scratch.txt"
        target.unlink()  # empty the dir first
        os.rmdir(guarded / "tmp_work")
        assert not (guarded / "tmp_work").exists()


# ============================================================================
# TestAlwaysBlockedOrdering — checked BEFORE outside-repo passthrough
# ============================================================================

class TestAlwaysBlockedOrdering:

    def test_slash_blocked(self, guarded):
        """/ is blocked even though it resolves outside the repo."""
        with pytest.raises(PermissionError, match="always blocked"):
            _check_and_maybe_block("test", "/")

    def test_dot_blocked(self, guarded):
        with pytest.raises(PermissionError, match="always blocked"):
            _check_and_maybe_block("test", ".")

    def test_dotdot_blocked(self, guarded):
        with pytest.raises(PermissionError, match="always blocked"):
            _check_and_maybe_block("test", "..")

    def test_tilde_blocked(self, guarded):
        with pytest.raises(PermissionError, match="always blocked"):
            _check_and_maybe_block("test", "~")

    def test_star_blocked(self, guarded):
        with pytest.raises(PermissionError, match="always blocked"):
            _check_and_maybe_block("test", "*")

    def test_repo_root_blocked(self, guarded):
        """Operating on the repo root itself should be blocked."""
        with pytest.raises(PermissionError, match="always blocked"):
            _check_and_maybe_block("test", str(guarded))

    @patch.dict(os.environ, {"ALLOW_DESTRUCTIVE_OPS": "1"})
    def test_always_blocked_ignores_unlock(self, guarded):
        """Always-blocked targets remain blocked even with unlock."""
        with pytest.raises(PermissionError, match="always blocked"):
            _check_and_maybe_block("test", "/")
        with pytest.raises(PermissionError, match="always blocked"):
            _check_and_maybe_block("test", ".")
        with pytest.raises(PermissionError, match="always blocked"):
            _check_and_maybe_block("test", "..")


# ============================================================================
# TestUnlockMechanism
# ============================================================================

class TestUnlockMechanism:

    @patch.dict(os.environ, {"ALLOW_DESTRUCTIVE_OPS": "1"})
    def test_env_var_unlocks_protected(self, guarded):
        """ALLOW_DESTRUCTIVE_OPS=1 allows protected (non-always) operations."""
        target = guarded / "data" / "knowledge_graph.json"
        # Should not raise — unlock is in effect
        os.remove(target)
        assert not target.exists()

    def test_lockfile_unlocks_protected(self, guarded):
        lockfile = guarded / ".agent_allow_destructive_once"
        lockfile.touch()
        try:
            target = guarded / "data" / "corpus_v4.json"
            os.remove(target)
            assert not target.exists()
        finally:
            if lockfile.exists():
                lockfile.unlink()

    @patch.dict(os.environ, {"ALLOW_DESTRUCTIVE_OPS": "1"})
    def test_unlock_does_not_affect_always_blocked(self, guarded):
        """Even with unlock, always-blocked targets stay blocked."""
        with pytest.raises(PermissionError):
            _check_and_maybe_block("test", "/")


# ============================================================================
# TestContextManager
# ============================================================================

class TestContextManager:

    def test_sets_and_clears_flag(self, repo):
        activate(repo_root=repo)
        try:
            assert _agent_mode_var.get(False) is False
            with agent_mode():
                assert _agent_mode_var.get(False) is True
            assert _agent_mode_var.get(False) is False
        finally:
            deactivate()

    def test_resets_after_exception(self, repo):
        activate(repo_root=repo)
        try:
            assert _agent_mode_var.get(False) is False
            with pytest.raises(ValueError):
                with agent_mode():
                    assert _agent_mode_var.get(False) is True
                    raise ValueError("test error")
            # Flag should be reset even after exception
            assert _agent_mode_var.get(False) is False
        finally:
            deactivate()

    def test_nested_context_preserves_outer(self, repo):
        activate(repo_root=repo)
        try:
            with agent_mode():
                assert _agent_mode_var.get(False) is True
                with agent_mode():
                    assert _agent_mode_var.get(False) is True
                # Inner exited, outer should still be True
                assert _agent_mode_var.get(False) is True
            # Both exited, should be False
            assert _agent_mode_var.get(False) is False
        finally:
            deactivate()


# ============================================================================
# TestAsyncPropagation
# ============================================================================

class TestAsyncPropagation:

    @pytest.mark.asyncio
    async def test_agent_mode_propagates_through_gather(self, repo):
        """ContextVar should propagate to tasks created by asyncio.gather."""
        activate(repo_root=repo)
        try:
            results = []

            async def check_mode(label):
                results.append((label, _agent_mode_var.get(False)))

            with agent_mode():
                await asyncio.gather(
                    check_mode("task1"),
                    check_mode("task2"),
                    check_mode("task3"),
                )

            assert all(v is True for _, v in results), f"Results: {results}"
        finally:
            deactivate()

    @pytest.mark.asyncio
    async def test_agent_mode_off_propagates(self, repo):
        """Without agent_mode(), tasks should see False."""
        activate(repo_root=repo)
        try:
            results = []

            async def check_mode(label):
                results.append((label, _agent_mode_var.get(False)))

            await asyncio.gather(
                check_mode("task1"),
                check_mode("task2"),
            )

            assert all(v is False for _, v in results)
        finally:
            deactivate()


# ============================================================================
# TestPathResolution
# ============================================================================

class TestPathResolution:

    def test_relative_path(self, guarded):
        # _resolve_to_repo_relative uses Path.resolve() which works from CWD.
        # Use absolute paths for deterministic behavior.
        result = _resolve_to_repo_relative(guarded / "data")
        assert result == "data"

    def test_nested_path(self, guarded):
        result = _resolve_to_repo_relative(guarded / "data" / "chroma_db")
        assert result == "data/chroma_db"

    def test_absolute_inside_repo(self, guarded):
        result = _resolve_to_repo_relative(str(guarded / "config"))
        assert result == "config"

    def test_absolute_outside_repo(self, guarded):
        result = _resolve_to_repo_relative("/usr/bin/python")
        assert result is None

    def test_traversal_stays_in_repo(self, guarded):
        result = _resolve_to_repo_relative(guarded / "data" / ".." / "config")
        assert result == "config"

    def test_traversal_escapes_repo(self, guarded, tmp_path):
        result = _resolve_to_repo_relative(guarded / ".." / ".." / "etc" / "passwd")
        assert result is None


# ============================================================================
# TestPathlibDelegation
# ============================================================================

class TestPathlibDelegation:
    """Path.unlink(), Path.rmdir(), Path.rename(), Path.replace() delegate
    to os.unlink, os.rmdir, os.rename, os.replace respectively."""

    def test_path_unlink_blocked(self, guarded):
        target = guarded / "data" / "knowledge_graph.json"
        with pytest.raises(PermissionError, match="protected"):
            target.unlink()

    def test_path_rmdir_blocked(self, guarded):
        # Create empty protected subdir
        target = guarded / "data" / "empty_subdir"
        target.mkdir()
        with pytest.raises(PermissionError, match="protected"):
            target.rmdir()

    def test_path_rename_source_blocked(self, guarded):
        src = guarded / "main.py"
        dst = guarded / "old_main.py"
        with pytest.raises(PermissionError, match="source"):
            src.rename(dst)

    def test_path_rename_destination_blocked(self, guarded):
        src = guarded / "tempfile.txt"
        dst = guarded / "CLAUDE.md"
        with pytest.raises(PermissionError, match="destination"):
            src.rename(dst)

    def test_path_replace_destination_blocked(self, guarded):
        src = guarded / "tempfile.txt"
        dst = guarded / "main.py"
        with pytest.raises(PermissionError, match="destination"):
            src.replace(dst)

    def test_path_unlink_non_protected_allowed(self, guarded):
        target = guarded / "scratch.log"
        assert target.exists()
        target.unlink()
        assert not target.exists()

    def test_path_rename_non_protected_allowed(self, guarded):
        src = guarded / "tempfile.txt"
        dst = guarded / "renamed.txt"
        src.rename(dst)
        assert dst.exists()


# ============================================================================
# TestWrapperCompatibility
# ============================================================================

class TestWrapperCompatibility:
    """Wrappers must preserve *args/**kwargs for platform compatibility."""

    def test_unlink_with_missing_ok(self, guarded):
        """Path.unlink(missing_ok=True) should work for non-protected files."""
        target = guarded / "nonexistent.txt"
        # Should not raise even though file doesn't exist
        target.unlink(missing_ok=True)

    def test_rmtree_with_ignore_errors(self, guarded):
        """shutil.rmtree(path, ignore_errors=True) should preserve the kwarg."""
        target = guarded / "build"
        shutil.rmtree(target, ignore_errors=True)
        assert not target.exists()

    def test_move_with_copy_function(self, guarded):
        """shutil.move should preserve copy_function kwarg."""
        src = guarded / "tempfile.txt"
        dst = guarded / "tmp_work" / "moved.txt"
        shutil.move(str(src), str(dst), copy_function=shutil.copy2)
        assert dst.exists()


# ============================================================================
# TestProtectedPathChecks (unit tests for internal helpers)
# ============================================================================

class TestProtectedPathChecks:

    def test_protected_dir(self):
        assert _is_protected_path("data") is True

    def test_protected_subpath(self):
        assert _is_protected_path("data/chroma_db") is True

    def test_protected_file(self):
        assert _is_protected_path("main.py") is True

    def test_non_protected_dir(self):
        assert _is_protected_path("build") is False

    def test_non_protected_file(self):
        assert _is_protected_path("tempfile.txt") is False

    def test_deeply_nested(self):
        assert _is_protected_path("data/a/b/c/d.txt") is True

    def test_always_blocked_dot(self):
        assert _is_always_blocked(".", None) is True

    def test_always_blocked_slash(self):
        assert _is_always_blocked("/", None) is True

    def test_always_blocked_tilde(self):
        assert _is_always_blocked("~", None) is True

    def test_normal_path_not_always_blocked(self):
        assert _is_always_blocked("data/file.txt", "data/file.txt") is False


# ============================================================================
# TestInactiveGuard — operations pass through when guard is not active
# ============================================================================

class TestInactiveGuard:

    def test_no_block_when_inactive(self, repo):
        """When guard is not active, nothing should be blocked."""
        assert not is_active()
        set_agent_mode(True)
        try:
            # Should not raise even for protected paths
            _check_and_maybe_block("test", repo / "data")
        finally:
            set_agent_mode(False)
