"""Tests for persistence state fingerprinting and diffing."""

import tempfile
from pathlib import Path

import pytest

from eval.persistence_guard import PersistenceGuard
from eval.schema import PersistenceSnapshot, StoreFingerprint


class TestStoreFingerprint:
    """Test individual store fingerprints."""

    def test_fingerprint_roundtrip(self):
        fp = StoreFingerprint(
            name="test_collection",
            kind="chromadb_collection",
            count=42,
            size_bytes=None,
            mtime_ns=None,
            sha256=None,
        )
        d = fp.to_dict()
        loaded = StoreFingerprint.from_dict(d)
        assert loaded.name == "test_collection"
        assert loaded.count == 42


class TestPersistenceSnapshot:
    """Test persistence snapshot diffing and assertion."""

    def _make_snapshot(self, **fingerprint_overrides) -> PersistenceSnapshot:
        fps = {
            "chromadb:facts": StoreFingerprint(
                name="facts", kind="chromadb_collection",
                count=100, sha256="aabbccdd11223344",
            ),
            "file:data/graph.json": StoreFingerprint(
                name="data/graph.json", kind="json_file",
                size_bytes=5000, mtime_ns=1000000, sha256="1122334455667788",
            ),
        }
        fps.update(fingerprint_overrides)
        return PersistenceSnapshot(
            snapshot_id="test-snap",
            capture_timestamp="2026-05-04T00:00:00",
            fingerprints=fps,
        )

    def test_identical_snapshots_pass(self):
        before = self._make_snapshot()
        after = self._make_snapshot()
        # Should not raise
        before.assert_same_as(after)

    def test_changed_collection_count_fails(self):
        before = self._make_snapshot()
        after = self._make_snapshot(**{
            "chromadb:facts": StoreFingerprint(
                name="facts", kind="chromadb_collection",
                count=101, sha256="aabbccdd11223344",
            ),
        })
        with pytest.raises(AssertionError, match="facts"):
            before.assert_same_as(after)

    def test_changed_file_hash_fails(self):
        before = self._make_snapshot()
        after = self._make_snapshot(**{
            "file:data/graph.json": StoreFingerprint(
                name="data/graph.json", kind="json_file",
                size_bytes=5000, mtime_ns=1000000, sha256="different_hash_!",
            ),
        })
        with pytest.raises(AssertionError, match="graph.json"):
            before.assert_same_as(after)

    def test_changed_file_size_fails(self):
        before = self._make_snapshot()
        after = self._make_snapshot(**{
            "file:data/graph.json": StoreFingerprint(
                name="data/graph.json", kind="json_file",
                size_bytes=9999, mtime_ns=1000000, sha256="1122334455667788",
            ),
        })
        with pytest.raises(AssertionError, match="graph.json"):
            before.assert_same_as(after)

    def test_added_store_fails(self):
        before = self._make_snapshot()
        after = self._make_snapshot(**{
            "chromadb:new_collection": StoreFingerprint(
                name="new_collection", kind="chromadb_collection", count=5,
            ),
        })
        with pytest.raises(AssertionError, match="new_collection"):
            before.assert_same_as(after)

    def test_removed_store_fails(self):
        before = self._make_snapshot()
        after_fps = dict(before.fingerprints)
        del after_fps["chromadb:facts"]
        after = PersistenceSnapshot(
            snapshot_id="test-snap",
            capture_timestamp="2026-05-04T00:00:00",
            fingerprints=after_fps,
        )
        with pytest.raises(AssertionError, match="facts"):
            before.assert_same_as(after)

    def test_diff_reports_all_changes(self):
        before = self._make_snapshot()
        after = self._make_snapshot(**{
            "chromadb:facts": StoreFingerprint(
                name="facts", kind="chromadb_collection",
                count=200, sha256="changed_hash_here",
            ),
            "file:data/graph.json": StoreFingerprint(
                name="data/graph.json", kind="json_file",
                size_bytes=9999, mtime_ns=2000000, sha256="changed_too____",
            ),
        })
        diff = before.diff(after)
        assert "chromadb:facts" in diff
        assert "file:data/graph.json" in diff

    def test_diff_empty_for_identical(self):
        before = self._make_snapshot()
        after = self._make_snapshot()
        diff = before.diff(after)
        assert diff == {}

    def test_assert_direction_before_to_after(self):
        """before.assert_same_as(after) — convention is self=before, other=after."""
        before = self._make_snapshot()
        after = self._make_snapshot(**{
            "chromadb:facts": StoreFingerprint(
                name="facts", kind="chromadb_collection",
                count=200, sha256="aabbccdd11223344",
            ),
        })
        # Verify the error message mentions the change direction
        with pytest.raises(AssertionError, match="100.*200"):
            before.assert_same_as(after)


class TestPersistenceGuard:
    """Test PersistenceGuard with real filesystem files."""

    def test_capture_fingerprints_real_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test.json"
            test_file.write_text('{"key": "value"}')

            guard = PersistenceGuard(
                chromadb_client=None,
                data_paths=[test_file],
            )
            snap = guard.capture()
            key = f"file:{test_file}"
            assert key in snap.fingerprints
            assert snap.fingerprints[key].size_bytes > 0
            assert snap.fingerprints[key].sha256 is not None

    def test_capture_missing_files_skipped(self):
        guard = PersistenceGuard(
            chromadb_client=None,
            data_paths=[Path("/nonexistent/file.json")],
        )
        snap = guard.capture()
        assert len(snap.fingerprints) == 0

    def test_unchanged_files_pass_assert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.json"
            test_file.write_text('{"key": "value"}')

            guard = PersistenceGuard(
                chromadb_client=None,
                data_paths=[test_file],
            )
            before = guard.capture()
            after = guard.capture()
            # Should not raise
            before.assert_same_as(after)

    def test_changed_file_fails_assert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.json"
            test_file.write_text('{"key": "value"}')

            guard = PersistenceGuard(
                chromadb_client=None,
                data_paths=[test_file],
            )
            before = guard.capture()

            # Modify the file
            test_file.write_text('{"key": "changed"}')

            after = guard.capture()
            with pytest.raises(AssertionError):
                before.assert_same_as(after)

    def test_capture_with_fake_chromadb(self):
        """Test ChromaDB fingerprinting with a mock client."""
        class FakeCollection:
            def __init__(self, name, count):
                self.name = name
                self._count = count
            def count(self):
                return self._count

        class FakeClient:
            def list_collections(self):
                return [
                    FakeCollection("facts", 100),
                    FakeCollection("conversations", 50),
                ]

        guard = PersistenceGuard(
            chromadb_client=FakeClient(),
            data_paths=[],
        )
        snap = guard.capture()
        assert "chromadb:facts" in snap.fingerprints
        assert snap.fingerprints["chromadb:facts"].count == 100
        assert "chromadb:conversations" in snap.fingerprints
        assert snap.fingerprints["chromadb:conversations"].count == 50

    def test_chromadb_count_change_detected(self):
        class MutableCollection:
            def __init__(self, name):
                self.name = name
                self._count = 100
            def count(self):
                return self._count

        col = MutableCollection("facts")

        class FakeClient:
            def list_collections(self):
                return [col]

        guard = PersistenceGuard(
            chromadb_client=FakeClient(),
            data_paths=[],
        )
        before = guard.capture()
        col._count = 101  # Simulate a write
        after = guard.capture()

        with pytest.raises(AssertionError, match="facts"):
            before.assert_same_as(after)
