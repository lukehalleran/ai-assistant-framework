"""
Tests for core.file_access_manager — FileAccessManager

Covers:
- Path validation (approved vs denied, traversal attacks, extension filtering)
- read_file (content, line ranges, size truncation, errors)
- grep_files (matches, capping, timeout, folder scoping)
- list_directory (flat, recursive, hidden filtering, capping)
- Prompt formatting helpers
- is_available()
"""

import os
import tempfile
import textwrap

import pytest
import pytest_asyncio

from core.file_access_manager import FileAccessManager


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sandbox(tmp_path):
    """Create a small sandbox directory tree for testing."""
    # Files
    (tmp_path / "hello.py").write_text("line1\nline2\nline3\nline4\nline5\n")
    (tmp_path / "notes.md").write_text("# Notes\nSome notes here.\n")
    (tmp_path / "data.json").write_text('{"key": "value"}\n')
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "binary.exe").write_bytes(b"\x00\x01\x02")

    # Sub-directory
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "module.py").write_text("def foo():\n    return 42\n")
    (sub / "config.yaml").write_text("key: value\n")

    # Hidden dir
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "secret.py").write_text("password = 'hunter2'\n")

    # __pycache__
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "junk.pyc").write_bytes(b"\x00")

    return tmp_path


@pytest.fixture
def manager(sandbox):
    """FileAccessManager pointed at the sandbox."""
    return FileAccessManager(
        approved_folders=[str(sandbox)],
        max_read_bytes=500,
        max_grep_results=5,
        max_list_entries=10,
        allowed_extensions=[".py", ".md", ".json", ".yaml", ".yml", ".txt"],
    )


@pytest.fixture
def manager_no_ext(sandbox):
    """FileAccessManager with no extension whitelist (allow all)."""
    return FileAccessManager(
        approved_folders=[str(sandbox)],
        max_read_bytes=500,
        max_grep_results=5,
        max_list_entries=10,
        allowed_extensions=None,
    )


# ── is_available ──────────────────────────────────────────────────────


def test_is_available_with_folders(manager):
    assert manager.is_available() is True


def test_is_available_empty():
    mgr = FileAccessManager(approved_folders=[])
    assert mgr.is_available() is False


# ── _validate_path ────────────────────────────────────────────────────


class TestValidatePath:
    def test_approved_path(self, manager, sandbox):
        resolved = manager._validate_path(str(sandbox / "hello.py"))
        assert resolved == (sandbox / "hello.py").resolve()

    def test_subdirectory_approved(self, manager, sandbox):
        resolved = manager._validate_path(str(sandbox / "subdir" / "module.py"))
        assert resolved.name == "module.py"

    def test_outside_folder_denied(self, manager):
        with pytest.raises(PermissionError, match="not in approved"):
            manager._validate_path("/etc/passwd")

    def test_traversal_attack_denied(self, manager, sandbox):
        """../../etc/passwd should resolve outside sandbox and be rejected."""
        evil_path = str(sandbox / ".." / ".." / "etc" / "passwd")
        with pytest.raises(PermissionError, match="not in approved"):
            manager._validate_path(evil_path)

    def test_disallowed_extension(self, manager, sandbox):
        with pytest.raises(PermissionError, match="not in allowed extensions"):
            manager._validate_path(str(sandbox / "image.png"))

    def test_allowed_extension(self, manager, sandbox):
        resolved = manager._validate_path(str(sandbox / "hello.py"))
        assert resolved.suffix == ".py"

    def test_no_extension_file_passes(self, manager, sandbox):
        """Files with no extension should pass (suffix check only fires when suffix exists)."""
        no_ext = sandbox / "Makefile"
        no_ext.write_text("all:\n\techo hi\n")
        resolved = manager._validate_path(str(no_ext))
        assert resolved.name == "Makefile"

    def test_no_extension_whitelist_allows_all(self, manager_no_ext, sandbox):
        """When allowed_extensions is empty, even .png should pass."""
        resolved = manager_no_ext._validate_path(str(sandbox / "image.png"))
        assert resolved.suffix == ".png"

    def test_symlink_resolved(self, manager, sandbox):
        """Symlink pointing outside approved dir should be rejected."""
        link = sandbox / "evil_link.py"
        try:
            link.symlink_to("/etc/hostname")
        except OSError:
            pytest.skip("Cannot create symlinks in this environment")
        with pytest.raises(PermissionError, match="not in approved"):
            manager._validate_path(str(link))


# ── read_file ─────────────────────────────────────────────────────────


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_success(self, manager, sandbox):
        result = await manager.read_file(str(sandbox / "hello.py"))
        assert result["success"] is True
        assert "line1" in result["content"]
        assert result["lines"] == 6  # 5 lines + trailing newline

    @pytest.mark.asyncio
    async def test_read_with_line_range(self, manager, sandbox):
        result = await manager.read_file(str(sandbox / "hello.py"), start_line=2, end_line=4)
        assert result["success"] is True
        assert "line2" in result["content"]
        assert "line4" in result["content"]
        assert "line5" not in result["content"]

    @pytest.mark.asyncio
    async def test_read_truncation(self, sandbox):
        """File larger than max_read_bytes should be truncated."""
        big = sandbox / "big.py"
        big.write_text("x" * 2000 + "\n")
        mgr = FileAccessManager(
            approved_folders=[str(sandbox)],
            max_read_bytes=100,
            allowed_extensions=[".py"],
        )
        result = await mgr.read_file(str(big))
        assert result["success"] is True
        assert result["truncated"] is True
        assert len(result["content"]) <= 100

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, manager, sandbox):
        result = await manager.read_file(str(sandbox / "nope.py"))
        assert result["success"] is False
        assert "Not a file" in result["error"]

    @pytest.mark.asyncio
    async def test_read_denied_path(self, manager):
        result = await manager.read_file("/etc/passwd")
        assert result["success"] is False
        assert "Access denied" in result["error"]

    @pytest.mark.asyncio
    async def test_read_disallowed_extension(self, manager, sandbox):
        result = await manager.read_file(str(sandbox / "image.png"))
        assert result["success"] is False
        assert "not in allowed extensions" in result["error"]

    @pytest.mark.asyncio
    async def test_read_directory_fails(self, manager, sandbox):
        result = await manager.read_file(str(sandbox / "subdir"))
        assert result["success"] is False
        assert "Not a file" in result["error"]


# ── grep_files ────────────────────────────────────────────────────────


class TestGrepFiles:
    @pytest.mark.asyncio
    async def test_grep_finds_matches(self, manager, sandbox):
        result = await manager.grep_files("line", file_glob="*.py")
        assert result["success"] is True
        assert result["total_matches"] > 0

    @pytest.mark.asyncio
    async def test_grep_no_matches(self, manager, sandbox):
        result = await manager.grep_files("zzzznotfound", file_glob="*.py")
        assert result["success"] is True
        assert result["total_matches"] == 0

    @pytest.mark.asyncio
    async def test_grep_scoped_to_folder(self, manager, sandbox):
        result = await manager.grep_files("def foo", folder=str(sandbox / "subdir"))
        assert result["success"] is True
        assert result["total_matches"] > 0

    @pytest.mark.asyncio
    async def test_grep_denied_folder(self, manager):
        result = await manager.grep_files("root", folder="/etc")
        assert result["success"] is False
        assert "Access denied" in result["error"]

    @pytest.mark.asyncio
    async def test_grep_result_capping(self, sandbox):
        """Create many matches and verify capping works."""
        many = sandbox / "many.py"
        many.write_text("\n".join(f"match_{i}" for i in range(100)))
        mgr = FileAccessManager(
            approved_folders=[str(sandbox)],
            max_grep_results=3,
            allowed_extensions=[".py"],
        )
        result = await mgr.grep_files("match_", file_glob="*.py", context_lines=0)
        assert result["success"] is True
        # Capped at max_grep_results * 3 = 9
        assert len(result["matches"]) <= 9

    @pytest.mark.asyncio
    async def test_grep_case_insensitive_default(self, manager, sandbox):
        (sandbox / "case.py").write_text("Hello World\n")
        result = await manager.grep_files("hello world", file_glob="*.py", context_lines=0)
        assert result["success"] is True
        assert result["total_matches"] > 0

    @pytest.mark.asyncio
    async def test_grep_nonexistent_folder(self, manager, sandbox):
        result = await manager.grep_files("x", folder=str(sandbox / "nope"))
        assert result["success"] is False


# ── list_directory ────────────────────────────────────────────────────


class TestListDirectory:
    @pytest.mark.asyncio
    async def test_list_flat(self, manager, sandbox):
        result = await manager.list_directory(str(sandbox))
        assert result["success"] is True
        paths = [e["path"] for e in result["entries"]]
        assert "hello.py" in paths
        assert "subdir" in paths

    @pytest.mark.asyncio
    async def test_list_recursive(self, manager, sandbox):
        result = await manager.list_directory(str(sandbox), recursive=True)
        assert result["success"] is True
        paths = [e["path"] for e in result["entries"]]
        assert any("module.py" in p for p in paths)

    @pytest.mark.asyncio
    async def test_list_hides_hidden_dirs(self, manager, sandbox):
        result = await manager.list_directory(str(sandbox), recursive=True)
        paths = [e["path"] for e in result["entries"]]
        assert not any(".hidden" in p for p in paths)

    @pytest.mark.asyncio
    async def test_list_hides_pycache(self, manager, sandbox):
        result = await manager.list_directory(str(sandbox), recursive=True)
        paths = [e["path"] for e in result["entries"]]
        assert not any("__pycache__" in p for p in paths)

    @pytest.mark.asyncio
    async def test_list_capping(self, sandbox):
        """Create many entries and verify cap."""
        for i in range(20):
            (sandbox / f"file_{i}.txt").write_text(f"f{i}")
        mgr = FileAccessManager(
            approved_folders=[str(sandbox)],
            max_list_entries=5,
            allowed_extensions=[".txt"],
        )
        result = await mgr.list_directory(str(sandbox))
        assert result["success"] is True
        assert result["total"] <= 5
        assert result["truncated"] is True

    @pytest.mark.asyncio
    async def test_list_denied_path(self, manager):
        result = await manager.list_directory("/etc")
        assert result["success"] is False
        assert "Access denied" in result["error"]

    @pytest.mark.asyncio
    async def test_list_file_instead_of_dir(self, manager, sandbox):
        result = await manager.list_directory(str(sandbox / "hello.py"))
        assert result["success"] is False
        assert "Not a directory" in result["error"]

    @pytest.mark.asyncio
    async def test_list_has_size_info(self, manager, sandbox):
        result = await manager.list_directory(str(sandbox))
        file_entries = [e for e in result["entries"] if e["type"] == "file"]
        assert all("size" in e for e in file_entries)
        dir_entries = [e for e in result["entries"] if e["type"] == "dir"]
        assert all(e["size"] == 0 for e in dir_entries)


# ── Prompt formatters ─────────────────────────────────────────────────


class TestFormatters:
    def test_format_read_success(self, manager):
        result = {
            "success": True,
            "content": "hello world",
            "path": "/foo/bar.py",
            "lines": 1,
            "truncated": False,
        }
        out = manager.format_read_for_prompt(result)
        assert "File: /foo/bar.py" in out
        assert "hello world" in out

    def test_format_read_truncated(self, manager):
        result = {
            "success": True,
            "content": "stuff",
            "path": "/foo/big.py",
            "lines": 9999,
            "truncated": True,
        }
        out = manager.format_read_for_prompt(result)
        assert "[TRUNCATED]" in out

    def test_format_read_error(self, manager):
        result = {"success": False, "error": "Access denied"}
        out = manager.format_read_for_prompt(result)
        assert "error" in out.lower()
        assert "Access denied" in out

    def test_format_grep_success(self, manager):
        result = {
            "success": True,
            "matches": ["file.py:1:hello", "file.py:5:world"],
            "total_matches": 2,
            "truncated": False,
        }
        out = manager.format_grep_for_prompt(result)
        assert "file.py:1:hello" in out

    def test_format_grep_no_matches(self, manager):
        result = {"success": True, "matches": [], "total_matches": 0, "truncated": False}
        out = manager.format_grep_for_prompt(result)
        assert "No matches" in out

    def test_format_grep_error(self, manager):
        result = {"success": False, "error": "timeout"}
        out = manager.format_grep_for_prompt(result)
        assert "error" in out.lower()

    def test_format_list_success(self, manager):
        result = {
            "success": True,
            "entries": [
                {"path": "foo.py", "type": "file", "size": 1024},
                {"path": "subdir", "type": "dir", "size": 0},
            ],
            "total": 2,
            "truncated": False,
        }
        out = manager.format_list_for_prompt(result)
        assert "foo.py" in out
        assert "1.0KB" in out
        assert "subdir/" in out

    def test_format_list_empty(self, manager):
        result = {"success": True, "entries": [], "total": 0, "truncated": False}
        out = manager.format_list_for_prompt(result)
        assert "Empty directory" in out

    def test_format_list_truncated(self, manager):
        result = {
            "success": True,
            "entries": [{"path": "a.py", "type": "file", "size": 10}],
            "total": 1,
            "truncated": True,
        }
        out = manager.format_list_for_prompt(result)
        assert "capped" in out.lower()


# ── Multiple approved folders ─────────────────────────────────────────


class TestMultipleFolders:
    @pytest.mark.asyncio
    async def test_multiple_folders(self, tmp_path):
        """Manager with two approved folders can access files in both."""
        folder_a = tmp_path / "a"
        folder_b = tmp_path / "b"
        folder_a.mkdir()
        folder_b.mkdir()
        (folder_a / "a.py").write_text("A\n")
        (folder_b / "b.py").write_text("B\n")

        mgr = FileAccessManager(
            approved_folders=[str(folder_a), str(folder_b)],
            allowed_extensions=[".py"],
        )
        result_a = await mgr.read_file(str(folder_a / "a.py"))
        result_b = await mgr.read_file(str(folder_b / "b.py"))
        assert result_a["success"] is True
        assert result_b["success"] is True
        assert "A" in result_a["content"]
        assert "B" in result_b["content"]

    @pytest.mark.asyncio
    async def test_cross_folder_denied(self, tmp_path):
        """Manager with folder A cannot access folder C."""
        folder_a = tmp_path / "a"
        folder_c = tmp_path / "c"
        folder_a.mkdir()
        folder_c.mkdir()
        (folder_c / "c.py").write_text("C\n")

        mgr = FileAccessManager(
            approved_folders=[str(folder_a)],
            allowed_extensions=[".py"],
        )
        result = await mgr.read_file(str(folder_c / "c.py"))
        assert result["success"] is False
        assert "Access denied" in result["error"]
