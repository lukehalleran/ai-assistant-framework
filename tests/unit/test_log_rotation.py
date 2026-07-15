"""Tests for utils/log_rotation.py — startup log-growth bounding."""

import gzip
import os
import time

from utils.log_rotation import (
    archive_if_large,
    maintain_debug_archives,
    rotate_if_large,
)


def _age(path, days):
    """Backdate a file's mtime by `days`."""
    old = time.time() - days * 86400
    os.utime(path, (old, old))


class TestRotateIfLarge:
    def test_under_cap_untouched(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text("small")
        assert rotate_if_large(str(p), max_bytes=1000) is False
        assert p.read_text() == "small"

    def test_missing_file_noop(self, tmp_path):
        assert rotate_if_large(str(tmp_path / "nope.log"), 10) is False

    def test_rotates_over_cap(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text("y" * 100)
        assert rotate_if_large(str(p), max_bytes=10) is True
        assert not p.exists()
        assert (tmp_path / "x.jsonl.1").read_text() == "y" * 100

    def test_shifts_numbered_and_drops_oldest(self, tmp_path):
        p = tmp_path / "x.jsonl"
        (tmp_path / "x.jsonl.1").write_text("gen1")
        (tmp_path / "x.jsonl.2").write_text("gen2")
        p.write_text("z" * 100)
        rotate_if_large(str(p), max_bytes=10, keep=2)
        assert (tmp_path / "x.jsonl.1").read_text() == "z" * 100
        assert (tmp_path / "x.jsonl.2").read_text() == "gen1"
        assert not (tmp_path / "x.jsonl.3").exists()  # gen2 dropped (keep=2)


class TestArchiveIfLarge:
    def test_under_cap_untouched(self, tmp_path):
        p = tmp_path / "audit.jsonl"
        p.write_text("a")
        assert archive_if_large(str(p), 1000) is False

    def test_archives_with_timestamp_never_deletes(self, tmp_path):
        p = tmp_path / "audit.jsonl"
        p.write_text("a" * 100)
        assert archive_if_large(str(p), 10) is True
        archives = [f for f in os.listdir(tmp_path) if f.startswith("audit-")]
        assert len(archives) == 1
        assert (tmp_path / archives[0]).read_text() == "a" * 100


class TestDebugArchiveMaintenance:
    def test_live_log_never_touched(self, tmp_path):
        live = tmp_path / "daemon_debug.log"
        live.write_text("live")
        _age(live, 400)
        result = maintain_debug_archives(str(tmp_path), 7, 90)
        assert live.exists()
        assert result == {"compressed": 0, "pruned": 0}

    def test_recent_archive_untouched(self, tmp_path):
        p = tmp_path / "daemon_debug_20260714_120000.log"
        p.write_text("recent")
        result = maintain_debug_archives(str(tmp_path), 7, 90)
        assert p.exists() and result["compressed"] == 0

    def test_old_archive_compressed_mtime_preserved(self, tmp_path):
        p = tmp_path / "daemon_debug_20260601_120000.log"
        p.write_text("old content")
        _age(p, 30)
        original_mtime = os.path.getmtime(p)
        result = maintain_debug_archives(str(tmp_path), 7, 90)
        assert result["compressed"] == 1
        assert not p.exists()
        gz = tmp_path / "daemon_debug_20260601_120000.log.gz"
        assert gzip.open(gz, "rt").read() == "old content"
        assert abs(os.path.getmtime(gz) - original_mtime) < 2

    def test_expired_archive_pruned(self, tmp_path):
        p = tmp_path / "daemon_debug_20260101_120000.log"
        p.write_text("ancient")
        _age(p, 120)
        result = maintain_debug_archives(str(tmp_path), 7, 90)
        assert result["pruned"] == 1
        assert not p.exists()

    def test_expired_gz_pruned(self, tmp_path):
        p = tmp_path / "daemon_debug_20260101_120000.log.gz"
        p.write_bytes(gzip.compress(b"ancient"))
        _age(p, 120)
        result = maintain_debug_archives(str(tmp_path), 7, 90)
        assert result["pruned"] == 1

    def test_unrelated_files_untouched(self, tmp_path):
        other = tmp_path / "important_notes.log"
        other.write_text("keep")
        _age(other, 400)
        maintain_debug_archives(str(tmp_path), 7, 90)
        assert other.exists()
