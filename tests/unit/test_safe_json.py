"""Tests for utils/safe_json.py — atomic writes + strict loading of critical stores.

Production-grade audit 2026-07-14: crash mid-write must never truncate a store;
an existing-but-corrupt store must never be silently replaced with empty state.
"""

import json
import os

import pytest

from utils.safe_json import (
    CorruptStoreError,
    StoreVersionError,
    atomic_write_json,
    check_schema_version,
    corrupt_store,
    load_critical_json,
    quarantine_corrupt_file,
)


class TestAtomicWriteJson:
    def test_writes_valid_json(self, tmp_path):
        path = str(tmp_path / "store.json")
        atomic_write_json(path, {"a": 1, "b": [2, 3]})
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == {"a": 1, "b": [2, 3]}

    def test_no_tmp_file_left_behind(self, tmp_path):
        path = str(tmp_path / "store.json")
        atomic_write_json(path, {"a": 1})
        assert not os.path.exists(path + ".tmp")

    def test_creates_parent_directory(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "store.json")
        atomic_write_json(path, [1, 2])
        assert os.path.exists(path)

    def test_failed_serialization_preserves_existing_file(self, tmp_path):
        path = str(tmp_path / "store.json")
        atomic_write_json(path, {"good": True})
        with pytest.raises(TypeError):
            atomic_write_json(path, {"bad": object()})
        # Original untouched, temp cleaned up
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == {"good": True}
        assert not os.path.exists(path + ".tmp")

    def test_overwrites_existing_atomically(self, tmp_path):
        path = str(tmp_path / "store.json")
        atomic_write_json(path, {"v": 1})
        atomic_write_json(path, {"v": 2})
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == {"v": 2}


class TestLoadCriticalJson:
    def test_missing_file_returns_none(self, tmp_path):
        assert load_critical_json(str(tmp_path / "nope.json"), "Test store") is None

    def test_empty_path_returns_none(self):
        assert load_critical_json("", "Test store") is None

    def test_valid_file_loads(self, tmp_path):
        path = str(tmp_path / "store.json")
        atomic_write_json(path, {"k": "v"})
        assert load_critical_json(path, "Test store") == {"k": "v"}

    def test_corrupt_file_raises_and_quarantines(self, tmp_path):
        path = str(tmp_path / "store.json")
        with open(path, "w") as f:
            f.write("{truncated")
        with pytest.raises(CorruptStoreError) as exc_info:
            load_critical_json(path, "Test store")
        err = exc_info.value
        # Original never deleted; quarantine copy preserves the bytes
        assert os.path.exists(path)
        assert err.quarantine_path and os.path.exists(err.quarantine_path)
        with open(err.quarantine_path) as f:
            assert f.read() == "{truncated"

    def test_zero_byte_file_is_fresh_start(self, tmp_path):
        """A 0-byte file has no recoverable data — fresh start, not corruption.
        (Pre-created empty files, e.g. tempfile.mkstemp, are legitimate.)"""
        path = str(tmp_path / "store.json")
        open(path, "w").close()
        assert load_critical_json(path, "Test store") is None

    def test_error_message_is_actionable(self, tmp_path):
        path = str(tmp_path / "store.json")
        with open(path, "w") as f:
            f.write("not json")
        with pytest.raises(CorruptStoreError) as exc_info:
            load_critical_json(path, "Test store")
        msg = str(exc_info.value)
        assert "Test store" in msg
        assert "backup" in msg


class TestCorruptStoreHelper:
    def test_builds_error_with_quarantine(self, tmp_path):
        path = str(tmp_path / "graph.json")
        with open(path, "w") as f:
            f.write("bad")
        err = corrupt_store(path, "Knowledge graph", ValueError("boom"))
        assert isinstance(err, CorruptStoreError)
        assert err.quarantine_path and os.path.exists(err.quarantine_path)

    def test_quarantine_missing_file_returns_none(self, tmp_path):
        assert quarantine_corrupt_file(str(tmp_path / "gone.json")) is None


class TestCheckSchemaVersion:
    def test_missing_version_is_v1(self):
        assert check_schema_version({"nodes": {}}, current=1,
                                    path="x", label="Test") == 1

    def test_matching_version_passes(self):
        assert check_schema_version({"schema_version": 2}, current=2,
                                    path="x", label="Test") == 2

    def test_older_version_returned_for_migration(self):
        assert check_schema_version({"schema_version": 1}, current=3,
                                    path="x", label="Test") == 1

    def test_newer_version_refused(self):
        with pytest.raises(StoreVersionError) as exc_info:
            check_schema_version({"schema_version": 99}, current=1,
                                 path="x.json", label="Test store")
        assert "newer" in str(exc_info.value)

    def test_non_dict_payload_is_v1(self):
        assert check_schema_version([1, 2], current=1, path="x", label="T") == 1

    def test_garbage_version_is_v1(self):
        assert check_schema_version({"schema_version": "abc"}, current=1,
                                    path="x", label="T") == 1


class TestGraphSchemaVersion:
    def test_save_stamps_version_and_load_roundtrips(self, tmp_path):
        from memory.graph_memory import GRAPH_SCHEMA_VERSION, GraphMemory
        from memory.graph_models import GraphNode

        path = str(tmp_path / "graph.json")
        g = GraphMemory(persist_path=path)
        g.add_entity(GraphNode(entity_id="a", display_name="A", entity_type="other"))
        g.save()
        with open(path) as f:
            assert json.load(f)["schema_version"] == GRAPH_SCHEMA_VERSION
        g2 = GraphMemory(persist_path=path)
        assert g2.node_count() == 1

    def test_newer_graph_version_refused(self, tmp_path):
        from memory.graph_memory import GraphMemory

        path = str(tmp_path / "graph.json")
        with open(path, "w") as f:
            json.dump({"schema_version": 999, "nodes": {}, "edges": []}, f)
        with pytest.raises(StoreVersionError):
            GraphMemory(persist_path=path)

    def test_unversioned_legacy_graph_loads(self, tmp_path):
        from memory.graph_memory import GraphMemory

        path = str(tmp_path / "graph.json")
        with open(path, "w") as f:
            json.dump({"nodes": {"a": {"display_name": "A", "entity_type": "other",
                                       "aliases": [], "mention_count": 1,
                                       "metadata": {}}}, "edges": []}, f)
        g = GraphMemory(persist_path=path)
        assert g.node_count() == 1


class TestClaimIndexSchemaVersion:
    def test_newer_claim_index_refused(self, tmp_path):
        from memory.claim_tracker import ClaimIndex

        path = str(tmp_path / "claims.json")
        with open(path, "w") as f:
            json.dump({"schema_version": 999, "index": {}, "doc_claims": {}}, f)
        with pytest.raises(StoreVersionError):
            ClaimIndex(persist_path=path)


class TestUserProfileCorruptLoad:
    def test_corrupt_profile_raises(self, tmp_path):
        from memory.user_profile import UserProfile

        path = str(tmp_path / "user_profile.json")
        with open(path, "w") as f:
            f.write("{invalid")
        with pytest.raises(CorruptStoreError):
            UserProfile(profile_path=path)
        assert os.path.exists(path)


class TestEntityAliasesCorruptLoad:
    def test_corrupt_aliases_raise(self, tmp_path):
        from memory.entity_resolver import EntityResolver
        from memory.graph_memory import GraphMemory

        graph = GraphMemory(persist_path=str(tmp_path / "graph.json"))
        aliases_path = str(tmp_path / "aliases.json")
        with open(aliases_path, "w") as f:
            f.write("[broken")
        with pytest.raises(CorruptStoreError):
            EntityResolver(graph_memory=graph, aliases_path=aliases_path)
        assert os.path.exists(aliases_path)


class TestEntityAliasesAtomicSave:
    def test_save_leaves_no_tmp_and_is_loadable(self, tmp_path):
        from memory.entity_resolver import EntityResolver
        from memory.graph_memory import GraphMemory
        from memory.graph_models import GraphNode

        graph = GraphMemory(persist_path=str(tmp_path / "graph.json"))
        aliases_path = str(tmp_path / "aliases.json")
        resolver = EntityResolver(graph_memory=graph, aliases_path=aliases_path)
        graph.add_entity(GraphNode(entity_id="biscuit", display_name="Biscuit",
                                   entity_type="pet"))
        resolver.learn_alias("the cat", "biscuit")
        resolver.save_external_aliases()
        assert not os.path.exists(aliases_path + ".tmp")
        with open(aliases_path, encoding="utf-8") as f:
            assert "the cat" in json.dumps(json.load(f))
