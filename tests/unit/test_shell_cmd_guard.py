"""Tests for utils.shell_cmd_guard — shell command classification.

All tests use tmp_path fixtures with synthetic directory structures.
No real data directories or files are ever touched.
"""

from pathlib import Path

import pytest

from utils.shell_cmd_guard import (
    ALWAYS_BLOCKED_TARGETS,
    PROTECTED_DIRS,
    PROTECTED_FILES,
    classify_shell_cmd,
    is_destructive_shell_cmd,
    _is_protected,
    _parse_rm_flags,
    _resolve_target,
)


# ============================================================================
# Helpers
# ============================================================================

@pytest.fixture
def repo(tmp_path):
    """Create a synthetic repo structure for testing."""
    for d in ("data", "config", "core", "memory", "scripts", "utils",
              "models", "docs", "tests", "build", "tmp_work"):
        (tmp_path / d).mkdir()
    for f in ("main.py", "CLAUDE.md", "requirements.txt", ".env",
              "tempfile.txt", "scratch.log"):
        (tmp_path / f).touch()
    (tmp_path / "data" / "knowledge_graph.json").touch()
    (tmp_path / "data" / "corpus_v4.json").touch()
    (tmp_path / "config" / "config.yaml").touch()
    return tmp_path


# ============================================================================
# Safe commands — should all pass through
# ============================================================================

class TestSafeCommands:
    @pytest.mark.parametrize("args", [
        ["ls"],
        ["ls", "-la"],
        ["cat", "file.txt"],
        ["echo", "hello"],
        ["cp", "a.txt", "b.txt"],
        ["mkdir", "-p", "new_dir"],
        ["touch", "new_file.txt"],
        ["head", "-n", "10", "file.txt"],
        ["tail", "-f", "file.log"],
        ["grep", "pattern", "file.txt"],
        ["wc", "-l", "file.txt"],
        ["diff", "a.txt", "b.txt"],
        ["python", "script.py"],
        ["pip", "install", "package"],
        ["pytest", "tests/"],
        ["git", "status"],  # git passes through (handled by safe_git.sh)
        ["curl", "https://example.com"],
        ["wget", "https://example.com"],
        ["sed", "s/a/b/", "file.txt"],
    ])
    def test_safe_commands_allowed(self, args, repo):
        result = classify_shell_cmd(args, repo)
        assert result["destructive"] is False, f"Expected safe: {' '.join(args)}"
        assert result["reason"] is None

    def test_empty_args_safe(self, repo):
        result = classify_shell_cmd([], repo)
        assert result["destructive"] is False
        assert result["command"] is None

    def test_rm_non_protected_file(self, repo):
        result = classify_shell_cmd(["rm", "tempfile.txt"], repo)
        assert result["destructive"] is False

    def test_rm_non_protected_dir_file(self, repo):
        result = classify_shell_cmd(["rm", "build/artifact.o"], repo)
        assert result["destructive"] is False

    def test_rm_rf_non_protected_dir(self, repo):
        result = classify_shell_cmd(["rm", "-rf", "build"], repo)
        assert result["destructive"] is False

    def test_rm_rf_tmp_work(self, repo):
        result = classify_shell_cmd(["rm", "-rf", "tmp_work"], repo)
        assert result["destructive"] is False

    def test_mv_non_protected(self, repo):
        result = classify_shell_cmd(["mv", "tempfile.txt", "renamed.txt"], repo)
        assert result["destructive"] is False

    def test_mv_build_dir(self, repo):
        result = classify_shell_cmd(["mv", "build", "/tmp/build_backup"], repo)
        assert result["destructive"] is False

    def test_chmod_safe_mode(self, repo):
        result = classify_shell_cmd(["chmod", "755", "tempfile.txt"], repo)
        assert result["destructive"] is False

    def test_chmod_protected_but_safe_mode(self, repo):
        """chmod +x on a protected file with a permissive mode is OK."""
        result = classify_shell_cmd(["chmod", "755", "main.py"], repo)
        assert result["destructive"] is False

    def test_find_without_delete(self, repo):
        result = classify_shell_cmd(["find", "data", "-name", "*.json"], repo)
        assert result["destructive"] is False

    def test_truncate_non_protected(self, repo):
        result = classify_shell_cmd(["truncate", "-s", "0", "scratch.log"], repo)
        assert result["destructive"] is False


# ============================================================================
# Always blocked — catastrophic operations, blocked even with unlock
# ============================================================================

class TestAlwaysBlocked:
    @pytest.mark.parametrize("target", [".", "..", "/", "~", "*"])
    def test_rm_rf_always_blocked_targets(self, target, repo):
        result = classify_shell_cmd(["rm", "-rf", target], repo)
        assert result["destructive"] is True
        assert result["severity"] == "always"

    @pytest.mark.parametrize("flags", ["-rf", "-Rf", "-fr", "--recursive", "-r", "-rfi"])
    def test_rm_various_recursive_flags_on_dot(self, flags, repo):
        result = classify_shell_cmd(["rm", flags, "."], repo)
        assert result["destructive"] is True
        assert result["severity"] == "always"

    def test_rm_rf_slash(self, repo):
        result = classify_shell_cmd(["rm", "-rf", "/"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "always"
        assert "catastrophic" in result["reason"]

    def test_rm_rf_tilde(self, repo):
        result = classify_shell_cmd(["rm", "-rf", "~"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "always"

    def test_rmdir_dot(self, repo):
        result = classify_shell_cmd(["rmdir", "."], repo)
        assert result["destructive"] is True
        assert result["severity"] == "always"

    def test_mv_dot(self, repo):
        result = classify_shell_cmd(["mv", ".", "/tmp/"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "always"

    def test_find_delete_on_dot(self, repo):
        result = classify_shell_cmd(["find", ".", "-name", "*.pyc", "-delete"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "always"

    def test_find_delete_on_slash(self, repo):
        result = classify_shell_cmd(["find", "/", "-type", "f", "-delete"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "always"


# ============================================================================
# rm on protected paths
# ============================================================================

class TestRmProtected:
    @pytest.mark.parametrize("target", [
        "data", "config", "core", "memory", "scripts", "utils",
        "docs", "tests", ".git",
    ])
    def test_rm_rf_protected_dirs(self, target, repo):
        result = classify_shell_cmd(["rm", "-rf", target], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_rm_rf_data_subdir(self, repo):
        result = classify_shell_cmd(["rm", "-rf", "data/chroma_db_v4"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_rm_protected_file(self, repo):
        """rm (non-recursive) on a protected root file should block."""
        result = classify_shell_cmd(["rm", "main.py"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_rm_env_file(self, repo):
        result = classify_shell_cmd(["rm", ".env"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_rm_claude_md(self, repo):
        result = classify_shell_cmd(["rm", "CLAUDE.md"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_rm_r_config(self, repo):
        result = classify_shell_cmd(["rm", "-r", "config"], repo)
        assert result["destructive"] is True

    def test_rm_rf_with_force_separate_flags(self, repo):
        result = classify_shell_cmd(["rm", "-r", "-f", "data"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_rm_rf_long_form_flags(self, repo):
        result = classify_shell_cmd(["rm", "--recursive", "--force", "data"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_rm_data_file_inside_protected(self, repo):
        """rm (non-recursive) on a file inside a protected dir should block."""
        result = classify_shell_cmd(["rm", "data/knowledge_graph.json"], repo)
        assert result["destructive"] is True

    def test_rm_absolute_path_inside_repo(self, repo):
        target = str(repo / "data")
        result = classify_shell_cmd(["rm", "-rf", target], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_rm_path_traversal(self, repo):
        """rm -rf data/../config should resolve to config and block."""
        result = classify_shell_cmd(["rm", "-rf", "data/../config"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"


# ============================================================================
# mv on protected paths
# ============================================================================

class TestMvProtected:
    def test_mv_protected_dir(self, repo):
        result = classify_shell_cmd(["mv", "config", "/tmp/config_backup"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_mv_data_dir(self, repo):
        result = classify_shell_cmd(["mv", "data", "/tmp/data"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_mv_protected_file(self, repo):
        result = classify_shell_cmd(["mv", "main.py", "old_main.py"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_mv_multiple_sources_one_protected(self, repo):
        result = classify_shell_cmd(["mv", "tempfile.txt", "config", "/tmp/"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_mv_data_subpath(self, repo):
        result = classify_shell_cmd(["mv", "data/corpus_v4.json", "/tmp/"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"


# ============================================================================
# chmod on protected paths
# ============================================================================

class TestChmodProtected:
    def test_chmod_000_on_protected(self, repo):
        result = classify_shell_cmd(["chmod", "000", "main.py"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_chmod_recursive_on_protected_dir(self, repo):
        result = classify_shell_cmd(["chmod", "-R", "755", "data"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_chmod_000_on_data(self, repo):
        result = classify_shell_cmd(["chmod", "000", "data"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_chmod_restrictive_0000(self, repo):
        result = classify_shell_cmd(["chmod", "0000", "config/config.yaml"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_chmod_000_non_protected(self, repo):
        """chmod 000 on a non-protected file is safe (user's business)."""
        result = classify_shell_cmd(["chmod", "000", "scratch.log"], repo)
        assert result["destructive"] is False


# ============================================================================
# rmdir on protected paths
# ============================================================================

class TestRmdirProtected:
    def test_rmdir_protected_dir(self, repo):
        result = classify_shell_cmd(["rmdir", "config"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_rmdir_data(self, repo):
        result = classify_shell_cmd(["rmdir", "data"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_rmdir_non_protected(self, repo):
        result = classify_shell_cmd(["rmdir", "build"], repo)
        assert result["destructive"] is False

    def test_rmdir_multiple_with_protected(self, repo):
        result = classify_shell_cmd(["rmdir", "build", "config"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"


# ============================================================================
# find with -delete
# ============================================================================

class TestFindDelete:
    def test_find_delete_on_data(self, repo):
        result = classify_shell_cmd(["find", "data", "-name", "*.tmp", "-delete"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_find_exec_rm_on_data(self, repo):
        result = classify_shell_cmd(
            ["find", "data", "-name", "*.tmp", "-exec", "rm", "{}", ";"],
            repo,
        )
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_find_delete_on_non_protected(self, repo):
        result = classify_shell_cmd(["find", "build", "-name", "*.o", "-delete"], repo)
        assert result["destructive"] is False

    def test_find_delete_on_config(self, repo):
        result = classify_shell_cmd(["find", "config", "-type", "f", "-delete"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"


# ============================================================================
# truncate on protected files
# ============================================================================

class TestTruncateProtected:
    def test_truncate_protected_file(self, repo):
        result = classify_shell_cmd(["truncate", "-s", "0", "main.py"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_truncate_data_file(self, repo):
        result = classify_shell_cmd(
            ["truncate", "-s", "0", "data/knowledge_graph.json"], repo,
        )
        assert result["destructive"] is True
        assert result["severity"] == "protected"

    def test_truncate_non_protected(self, repo):
        result = classify_shell_cmd(["truncate", "-s", "0", "scratch.log"], repo)
        assert result["destructive"] is False

    def test_truncate_env(self, repo):
        result = classify_shell_cmd(["truncate", "-s", "0", ".env"], repo)
        assert result["destructive"] is True
        assert result["severity"] == "protected"


# ============================================================================
# Path resolution
# ============================================================================

class TestResolveTarget:
    def test_relative_path(self, repo):
        assert _resolve_target("data", repo) == "data"

    def test_nested_path(self, repo):
        assert _resolve_target("data/chroma_db_v4", repo) == "data/chroma_db_v4"

    def test_absolute_inside_repo(self, repo):
        target = str(repo / "config")
        assert _resolve_target(target, repo) == "config"

    def test_absolute_outside_repo(self, repo):
        assert _resolve_target("/usr/bin/python", repo) is None

    def test_tilde_outside(self, repo):
        assert _resolve_target("~/Documents", repo) is None

    def test_traversal_stays_in_repo(self, repo):
        result = _resolve_target("data/../config", repo)
        assert result == "config"

    def test_traversal_escapes_repo(self, repo):
        result = _resolve_target("../../etc/passwd", repo)
        assert result is None

    def test_dot_resolves_to_dot(self, repo):
        result = _resolve_target(".", repo)
        assert result == "."

    def test_empty_string(self, repo):
        assert _resolve_target("", repo) is None

    def test_dash_dash(self, repo):
        assert _resolve_target("--", repo) is None


# ============================================================================
# Protected path matching
# ============================================================================

class TestProtectedPaths:
    def test_direct_protected_dir(self):
        assert _is_protected("data", PROTECTED_DIRS, PROTECTED_FILES) is True

    def test_subpath_of_protected_dir(self):
        assert _is_protected("data/chroma_db_v4", PROTECTED_DIRS, PROTECTED_FILES) is True

    def test_protected_file(self):
        assert _is_protected("main.py", PROTECTED_DIRS, PROTECTED_FILES) is True

    def test_non_protected(self):
        assert _is_protected("build", PROTECTED_DIRS, PROTECTED_FILES) is False

    def test_non_protected_file(self):
        assert _is_protected("tempfile.txt", PROTECTED_DIRS, PROTECTED_FILES) is False

    def test_deeply_nested_protected(self):
        assert _is_protected("data/a/b/c/file.txt", PROTECTED_DIRS, PROTECTED_FILES) is True

    def test_dot_is_always_blocked(self):
        assert _is_protected(".", PROTECTED_DIRS, PROTECTED_FILES) is True

    def test_all_protected_dirs_present(self):
        """Verify the default protected dirs set contains key directories."""
        expected = {"data", "config", ".git", "scripts", "memory", "core",
                    "knowledge", "utils", "models", "docs", "tests"}
        assert expected.issubset(PROTECTED_DIRS)

    def test_all_protected_files_present(self):
        expected = {"main.py", "CLAUDE.md", "requirements.txt", ".env"}
        assert expected.issubset(PROTECTED_FILES)


# ============================================================================
# Flag parsing
# ============================================================================

class TestFlagParsing:
    def test_combined_rf(self):
        assert _parse_rm_flags(["-rf"]) == (True, True)

    def test_combined_fr(self):
        assert _parse_rm_flags(["-fr"]) == (True, True)

    def test_separate_r_f(self):
        assert _parse_rm_flags(["-r", "-f"]) == (True, True)

    def test_long_form(self):
        assert _parse_rm_flags(["--recursive", "--force"]) == (True, True)

    def test_recursive_only(self):
        assert _parse_rm_flags(["-r"]) == (True, False)

    def test_force_only(self):
        assert _parse_rm_flags(["-f"]) == (False, True)

    def test_uppercase_R(self):
        assert _parse_rm_flags(["-R"]) == (True, False)

    def test_combined_Rf(self):
        assert _parse_rm_flags(["-Rf"]) == (True, True)

    def test_no_flags(self):
        assert _parse_rm_flags(["file.txt"]) == (False, False)

    def test_interactive_flag_ignored(self):
        assert _parse_rm_flags(["-i"]) == (False, False)

    def test_mixed_flags_and_targets(self):
        assert _parse_rm_flags(["-rf", "data", "config"]) == (True, True)


# ============================================================================
# Convenience wrapper
# ============================================================================

class TestConvenienceWrapper:
    def test_matches_classify(self, repo):
        assert is_destructive_shell_cmd(["rm", "-rf", "data"], repo) is True
        assert is_destructive_shell_cmd(["ls", "-la"], repo) is False
        assert is_destructive_shell_cmd(["rm", "-rf", "."], repo) is True
        assert is_destructive_shell_cmd(["rm", "tempfile.txt"], repo) is False


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    def test_absolute_path_to_binary(self, repo):
        """/usr/bin/rm should be recognized as rm."""
        result = classify_shell_cmd(["/usr/bin/rm", "-rf", "data"], repo)
        assert result["destructive"] is True
        assert result["command"] == "rm"

    def test_rm_no_targets(self, repo):
        result = classify_shell_cmd(["rm"], repo)
        assert result["destructive"] is False

    def test_mv_single_arg(self, repo):
        result = classify_shell_cmd(["mv", "data"], repo)
        assert result["destructive"] is False  # mv with one arg is an error, not destructive

    def test_rm_rf_outside_repo(self, repo):
        """rm -rf on a path outside repo should be safe (not our business)."""
        result = classify_shell_cmd(["rm", "-rf", "/tmp/random_dir"], repo)
        assert result["destructive"] is False

    def test_chmod_no_targets(self, repo):
        result = classify_shell_cmd(["chmod", "755"], repo)
        assert result["destructive"] is False

    def test_find_no_delete(self, repo):
        result = classify_shell_cmd(["find", "data", "-name", "*.json"], repo)
        assert result["destructive"] is False

    def test_unknown_command_safe(self, repo):
        result = classify_shell_cmd(["some_unknown_cmd", "--flag"], repo)
        assert result["destructive"] is False

    def test_git_passes_through(self, repo):
        """git commands should pass through as safe (handled by safe_git.sh)."""
        result = classify_shell_cmd(["git", "restore", "."], repo)
        assert result["destructive"] is False
