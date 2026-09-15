"""
CGR-20260913-001 (BC-37, dm17_apply_without_guard): scripts/cleanup_stale_illness.py,
scripts/graph_relation_normalize.py and scripts/reclassify_proposals.py wrote a
Daemon-held store on ``--apply`` without ever consulting
``utils.daemon_guard.daemon_running()`` — the same class as the 2026-08-05
profile-clobber and 2026-09-05 ``graph_junk_cleanup.py --apply`` incidents (a
live Daemon re-saves its in-memory state and silently clobbers or resurrects
the script's change). These drive each script's deployed ``main(argv)`` and
assert: guard-refused (return 1, writer never reached), guard-False control
(writer reached), and dry-run control (guard never consulted, nothing written).

scripts/restore_backup.py (anchor #35) is NOT an instance of the class — its
only write path already sits behind ``utils.single_instance``'s exclusive
lock (stronger than ``daemon_running()``: it *prevents* Daemon from starting
mid-restore rather than just detecting one already running). See
TestRestoreBackupGuardEvidence below, which adds the one requested main()-level
check on top of the existing
test_sep09_backup_recovery.py::test_restore_refuses_when_daemon_lock_is_held
evidence.
"""

import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

import scripts.cleanup_stale_illness as csi
import scripts.graph_relation_normalize as grn
from memory.graph_memory import GraphMemory
from memory.graph_models import GraphEdge, GraphNode
from scripts import restore_backup
from tests.unit.test_sep09_backup_recovery import backup as make_backup
from tests.unit.test_sep09_backup_recovery import snapshot, stores  # noqa: F401  (fixture)

_RECLASSIFY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "scripts", "reclassify_proposals.py")
_spec = importlib.util.spec_from_file_location("reclassify_proposals_guard_test", _RECLASSIFY_PATH)
reclass = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reclass)


# --------------------------------------------------------------------------
# #32 scripts/cleanup_stale_illness.py
# --------------------------------------------------------------------------

class TestCleanupStaleIllnessGuard:
    RELATION = "current_illness"  # contains "illness" -> _is_health_transient True
    VALUE = "flu"

    def _patch_scans(self, monkeypatch, tmp_path):
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        profile = {"categories": {"health": [
            {"relation": self.RELATION, "value": self.VALUE,
             "is_current": True, "timestamp": old_ts},
        ]}}
        p_targets = [("health", 0, self.RELATION, self.VALUE, old_ts)]
        c_targets = [("chroma-1", self.RELATION, self.VALUE, old_ts, {"is_current": True})]
        monkeypatch.setattr(csi, "scan_profile", lambda min_age: (profile, p_targets))
        fake_col = MagicMock()
        monkeypatch.setattr(csi, "scan_chroma", lambda path, min_age: (fake_col, c_targets))
        (tmp_path / "data").mkdir()
        profile_path = tmp_path / "data" / "user_profile.json"
        monkeypatch.setattr(csi, "PROFILE_PATH", str(profile_path))
        monkeypatch.setattr(csi, "ROOT", str(tmp_path))
        return fake_col, profile_path

    def test_apply_refused_when_daemon_running(self, tmp_path, monkeypatch):
        fake_col, profile_path = self._patch_scans(monkeypatch, tmp_path)
        monkeypatch.setattr(csi, "_daemon_running", lambda: True)

        rc = csi.main(["--apply"])

        assert rc == 1
        fake_col.update.assert_not_called()
        assert not profile_path.exists()
        assert list((tmp_path / "data").iterdir()) == []  # no backup/undo written

    def test_apply_writes_when_daemon_not_running(self, tmp_path, monkeypatch):
        fake_col, profile_path = self._patch_scans(monkeypatch, tmp_path)
        monkeypatch.setattr(csi, "_daemon_running", lambda: False)

        rc = csi.main(["--apply"])

        assert rc in (0, None)
        fake_col.update.assert_called_once()
        written = json.loads(profile_path.read_text())
        assert written["categories"]["health"][0]["is_current"] is False

    def test_dry_run_never_refuses_and_writes_nothing(self, tmp_path, monkeypatch):
        fake_col, profile_path = self._patch_scans(monkeypatch, tmp_path)
        spy = MagicMock(return_value=True)
        monkeypatch.setattr(csi, "_daemon_running", spy)

        rc = csi.main([])

        assert rc in (0, None)
        spy.assert_not_called()  # guard's result never consulted without --apply
        fake_col.update.assert_not_called()
        assert not profile_path.exists()


# --------------------------------------------------------------------------
# #33 scripts/graph_relation_normalize.py
# --------------------------------------------------------------------------

class TestGraphRelationNormalizeGuard:
    def _make_graph(self, tmp_path):
        graph_path = tmp_path / "graph.json"
        gm = GraphMemory(persist_path=str(graph_path))
        gm.add_entity(GraphNode(entity_id="user", display_name="User"))
        gm.add_entity(GraphNode(entity_id="chicago", display_name="Chicago"))
        # "lives in" is a variant relation normalize_relation() canonicalizes
        # to "lives_in" (memory/entity_resolver.py docstring example) -- this
        # is the historical-variant edge the migration exists to rewrite.
        gm.add_relation(GraphEdge(source_id="user", relation="lives in", target_id="chicago"))
        gm.save()
        return graph_path

    def test_apply_refused_when_daemon_running(self, tmp_path, monkeypatch):
        graph_path = self._make_graph(tmp_path)
        before = graph_path.read_text()
        monkeypatch.setattr(grn, "_daemon_running", lambda: True)

        rc = grn.main(["--graph", str(graph_path), "--apply"])

        assert rc == 1
        assert graph_path.read_text() == before
        assert list(tmp_path.glob("*.bak-*")) == []  # backup() never ran

    def test_apply_writes_when_daemon_not_running(self, tmp_path, monkeypatch):
        graph_path = self._make_graph(tmp_path)
        monkeypatch.setattr(grn, "_daemon_running", lambda: False)

        rc = grn.main(["--graph", str(graph_path), "--apply"])

        assert rc in (0, None)
        assert list(tmp_path.glob("*.bak-*")) != []
        reloaded = GraphMemory(persist_path=str(graph_path))
        edge = reloaded._edge_index.get("user|lives_in|chicago")
        assert edge is not None and edge.relation == "lives_in"

    def test_dry_run_never_refuses_and_writes_nothing(self, tmp_path, monkeypatch):
        graph_path = self._make_graph(tmp_path)
        before = graph_path.read_text()
        spy = MagicMock(return_value=True)
        monkeypatch.setattr(grn, "_daemon_running", spy)

        rc = grn.main(["--graph", str(graph_path)])

        assert rc in (0, None)
        spy.assert_not_called()
        assert graph_path.read_text() == before
        assert list(tmp_path.glob("*.bak-*")) == []


# --------------------------------------------------------------------------
# #34 scripts/reclassify_proposals.py
# --------------------------------------------------------------------------

def _fake_chroma_store(core_path):
    """A fake MultiCollectionChromaStore recording construction + writes, with
    one stored proposal whose metadata the live classifier upgrades (a core
    path in affected_files_json -> touches_core_system True / risk HIGH,
    differing from the stored medium/False -- same fixture shape as
    test_reclassify_proposals.py::test_core_path_in_affected_files_trips_core)."""
    calls = {"constructed": [], "update_metadata": []}

    class FakeCollection:
        def count(self):
            return 1

    class FakeStore:
        def __init__(self, persist_directory=None):
            calls["constructed"].append(persist_directory)
            self.collections = {"proposals": FakeCollection()}

        def create_collection(self, name):
            pass

        def list_all(self, name):
            md = {
                "proposal_id": "p1", "title": "Touches core",
                "proposal_type": "feature",
                "affected_files_json": json.dumps([core_path]),
                "steps_json": json.dumps([]),
                "description": "", "reasoning": "",
                "risk_level": "medium", "touches_core_system": False,
                "depends_on_json": json.dumps([]),
            }
            return [{"id": "p1", "metadata": md}]

        def update_metadata(self, collection_name, doc_id, updates):
            calls["update_metadata"].append((collection_name, doc_id, updates))

    return FakeStore, calls


class TestReclassifyProposalsGuard:
    def test_apply_refused_when_daemon_running(self, monkeypatch):
        FakeStore, calls = _fake_chroma_store("core/orchestrator.py")
        monkeypatch.setattr(reclass, "MultiCollectionChromaStore", FakeStore)
        monkeypatch.setattr(reclass, "_daemon_running", lambda: True)

        rc = reclass.main(["--apply"])

        assert rc == 1
        assert calls["constructed"] == []  # store never opened
        assert calls["update_metadata"] == []

    def test_apply_writes_when_daemon_not_running(self, monkeypatch):
        FakeStore, calls = _fake_chroma_store("core/orchestrator.py")
        monkeypatch.setattr(reclass, "MultiCollectionChromaStore", FakeStore)
        monkeypatch.setattr(reclass, "_daemon_running", lambda: False)

        rc = reclass.main(["--apply"])

        assert rc in (0, None)
        assert calls["constructed"] == [reclass.CHROMA_PATH]
        assert len(calls["update_metadata"]) == 1
        _, doc_id, updates = calls["update_metadata"][0]
        assert doc_id == "p1" and updates["touches_core_system"] is True

    def test_dry_run_never_refuses_and_writes_nothing(self, monkeypatch):
        FakeStore, calls = _fake_chroma_store("core/orchestrator.py")
        monkeypatch.setattr(reclass, "MultiCollectionChromaStore", FakeStore)
        spy = MagicMock(return_value=True)
        monkeypatch.setattr(reclass, "_daemon_running", spy)

        rc = reclass.main([])

        assert rc in (0, None)
        spy.assert_not_called()
        assert calls["update_metadata"] == []


# --------------------------------------------------------------------------
# #35 scripts/restore_backup.py -- NOT an instance of the class (evidence only)
# --------------------------------------------------------------------------

class TestRestoreBackupGuardEvidence:
    def test_main_cli_refuses_when_daemon_lock_is_held(self, stores, monkeypatch):  # noqa: F811
        from utils.single_instance import SingleInstanceError

        name = make_backup(stores)
        stores.lock.side_effect = SingleInstanceError("synthetic daemon holds the lock")
        before = snapshot(stores.data)
        monkeypatch.setattr(sys, "argv", ["restore_backup.py", "--restore", name, "--apply"])

        with pytest.raises(SystemExit) as exc:
            restore_backup.main()

        assert exc.value.code == 1
        assert snapshot(stores.data) == before
