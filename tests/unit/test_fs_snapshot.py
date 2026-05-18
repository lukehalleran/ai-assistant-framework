"""Tests for utils.fs_snapshot — filesystem manifest creation and diffing."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from utils.fs_snapshot import (
    create_manifest,
    diff_manifests,
    iter_snapshot_files,
    load_manifest,
    save_manifest,
    sha256_file,
    should_exclude,
)


# ============================================================================
# Helpers
# ============================================================================

def _write(root: Path, rel: str, content: str = "hello") -> Path:
    """Write a file under *root* at relative path *rel*."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


# ============================================================================
# should_exclude
# ============================================================================

class TestShouldExclude:
    def test_git_dir(self):
        assert should_exclude(".git/config") is True

    def test_pycache(self):
        assert should_exclude("core/__pycache__/foo.cpython-312.pyc") is True

    def test_agent_snapshots(self):
        assert should_exclude(".agent_snapshots/20260518/manifest.json") is True

    def test_venv(self):
        assert should_exclude("venv/lib/python3.12/site.py") is True
        assert should_exclude(".venv/bin/python") is True

    def test_recovery_prefix(self):
        assert should_exclude("RECOVERY_inventory.txt") is True
        assert should_exclude("RECOVERY_surviving_tracked_changes.patch") is True

    def test_rhistory(self):
        assert should_exclude("docs/.Rhistory") is True

    def test_pyc_extension(self):
        assert should_exclude("utils/bootstrap.pyc") is True

    def test_rpm_extension(self):
        assert should_exclude("rstudio-2026.01.0-392-x86_64.rpm") is True

    def test_normal_python_file_allowed(self):
        assert should_exclude("utils/fs_snapshot.py") is False

    def test_normal_test_file_allowed(self):
        assert should_exclude("tests/unit/test_fs_snapshot.py") is False

    def test_config_allowed(self):
        assert should_exclude("config/config.yaml") is False

    def test_nested_normal_allowed(self):
        assert should_exclude("core/prompt/builder.py") is False

    def test_node_modules(self):
        assert should_exclude("node_modules/some-package/index.js") is True

    def test_mypy_cache(self):
        assert should_exclude(".mypy_cache/3.12/utils/fs_snapshot.meta.json") is True

    def test_ruff_cache(self):
        assert should_exclude(".ruff_cache/content/abc123") is True

    def test_pytest_cache(self):
        assert should_exclude(".pytest_cache/v/cache/lastfailed") is True


# ============================================================================
# sha256_file
# ============================================================================

class TestSha256File:
    def test_deterministic(self, tmp_path):
        f = _write(tmp_path, "a.txt", "deterministic content")
        assert sha256_file(f) == sha256_file(f)

    def test_different_content_different_hash(self, tmp_path):
        f1 = _write(tmp_path, "a.txt", "content A")
        f2 = _write(tmp_path, "b.txt", "content B")
        assert sha256_file(f1) != sha256_file(f2)

    def test_empty_file(self, tmp_path):
        f = _write(tmp_path, "empty.txt", "")
        h = sha256_file(f)
        assert len(h) == 64  # sha256 hex is 64 chars


# ============================================================================
# iter_snapshot_files
# ============================================================================

class TestIterSnapshotFiles:
    def test_lists_normal_files(self, tmp_path):
        _write(tmp_path, "a.py", "pass")
        _write(tmp_path, "sub/b.py", "pass")
        files = iter_snapshot_files(tmp_path)
        rel_strs = [str(f) for f in files]
        assert "a.py" in rel_strs
        assert os.path.join("sub", "b.py") in rel_strs

    def test_excludes_pycache(self, tmp_path):
        _write(tmp_path, "a.py", "pass")
        _write(tmp_path, "__pycache__/a.cpython-312.pyc", "bytecode")
        files = [str(f) for f in iter_snapshot_files(tmp_path)]
        assert "a.py" in files
        assert not any("__pycache__" in f for f in files)

    def test_excludes_git(self, tmp_path):
        _write(tmp_path, ".git/config", "[core]")
        _write(tmp_path, "real.py", "pass")
        files = [str(f) for f in iter_snapshot_files(tmp_path)]
        assert "real.py" in files
        assert not any(".git" in f for f in files)

    def test_sorted_output(self, tmp_path):
        _write(tmp_path, "z.py")
        _write(tmp_path, "a.py")
        _write(tmp_path, "m.py")
        files = iter_snapshot_files(tmp_path)
        assert files == sorted(files)

    def test_empty_dir(self, tmp_path):
        assert iter_snapshot_files(tmp_path) == []


# ============================================================================
# create_manifest / save / load
# ============================================================================

class TestManifest:
    def test_create_has_expected_keys(self, tmp_path):
        _write(tmp_path, "hello.txt", "world")
        m = create_manifest(tmp_path)
        assert "hello.txt" in m
        entry = m["hello.txt"]
        assert "size" in entry
        assert "mtime" in entry
        assert "sha256" in entry

    def test_size_correct(self, tmp_path):
        content = "12345"
        _write(tmp_path, "f.txt", content)
        m = create_manifest(tmp_path)
        assert m["f.txt"]["size"] == len(content.encode())

    def test_roundtrip_save_load(self, tmp_path):
        _write(tmp_path, "a.py", "pass")
        _write(tmp_path, "sub/b.py", "import os")
        m = create_manifest(tmp_path)
        out = tmp_path / "manifest.json"
        save_manifest(m, out)
        loaded = load_manifest(out)
        assert m == loaded

    def test_save_creates_parent_dirs(self, tmp_path):
        m = {"a.py": {"size": 4, "mtime": 0.0, "sha256": "abc"}}
        out = tmp_path / "deep" / "nested" / "manifest.json"
        save_manifest(m, out)
        assert out.exists()

    def test_excludes_pycache_from_manifest(self, tmp_path):
        _write(tmp_path, "a.py", "pass")
        _write(tmp_path, "__pycache__/a.cpython-312.pyc", "bytecode")
        m = create_manifest(tmp_path)
        assert "a.py" in m
        assert not any("__pycache__" in k for k in m)

    def test_posix_paths(self, tmp_path):
        _write(tmp_path, "sub/deep/file.py", "pass")
        m = create_manifest(tmp_path)
        assert "sub/deep/file.py" in m  # posix, not OS-specific


# ============================================================================
# diff_manifests
# ============================================================================

class TestDiffManifests:
    def test_no_changes(self):
        m = {"a.py": {"sha256": "abc"}}
        result = diff_manifests(m, m)
        assert result == {"added": [], "deleted": [], "modified": []}

    def test_added_file(self):
        before = {"a.py": {"sha256": "abc"}}
        after = {"a.py": {"sha256": "abc"}, "b.py": {"sha256": "def"}}
        result = diff_manifests(before, after)
        assert result["added"] == ["b.py"]
        assert result["deleted"] == []
        assert result["modified"] == []

    def test_deleted_file(self):
        before = {"a.py": {"sha256": "abc"}, "b.py": {"sha256": "def"}}
        after = {"a.py": {"sha256": "abc"}}
        result = diff_manifests(before, after)
        assert result["deleted"] == ["b.py"]
        assert result["added"] == []

    def test_modified_file(self):
        before = {"a.py": {"sha256": "abc"}}
        after = {"a.py": {"sha256": "xyz"}}
        result = diff_manifests(before, after)
        assert result["modified"] == ["a.py"]

    def test_combined_changes(self):
        before = {"keep.py": {"sha256": "aaa"}, "gone.py": {"sha256": "bbb"}, "changed.py": {"sha256": "ccc"}}
        after = {"keep.py": {"sha256": "aaa"}, "new.py": {"sha256": "ddd"}, "changed.py": {"sha256": "eee"}}
        result = diff_manifests(before, after)
        assert result["added"] == ["new.py"]
        assert result["deleted"] == ["gone.py"]
        assert result["modified"] == ["changed.py"]

    def test_results_sorted(self):
        before = {}
        after = {"z.py": {"sha256": "a"}, "a.py": {"sha256": "b"}, "m.py": {"sha256": "c"}}
        result = diff_manifests(before, after)
        assert result["added"] == ["a.py", "m.py", "z.py"]

    def test_empty_manifests(self):
        result = diff_manifests({}, {})
        assert result == {"added": [], "deleted": [], "modified": []}


# ============================================================================
# CLI (via main())
# ============================================================================

class TestCLI:
    def test_create_command(self, tmp_path):
        from utils.fs_snapshot import main
        _write(tmp_path, "a.py", "pass")
        out = str(tmp_path / "manifest.json")
        main(["create", str(tmp_path), out])
        loaded = load_manifest(out)
        assert "a.py" in loaded

    def test_diff_command(self, tmp_path, capsys):
        from utils.fs_snapshot import main
        m1 = {"a.py": {"sha256": "abc"}}
        m2 = {"a.py": {"sha256": "abc"}, "b.py": {"sha256": "def"}}
        p1 = tmp_path / "before.json"
        p2 = tmp_path / "after.json"
        save_manifest(m1, p1)
        save_manifest(m2, p2)
        main(["diff", str(p1), str(p2)])
        captured = capsys.readouterr()
        assert "ADDED" in captured.out
        assert "b.py" in captured.out

    def test_no_command_exits(self):
        from utils.fs_snapshot import main
        with pytest.raises(SystemExit):
            main([])
