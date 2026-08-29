"""
scripts/backfill_stance.py (Phase B4, 2026-08-23).

Follows the purge_adaptive_exemplars test precedent: exercise the planning
functions with fake docs + a tmp GraphMemory, and pin the safety contract
(dry-run default, sentinel hard-gate, daemon-guard refusal on --apply).
"""

import importlib

import pytest

import scripts.backfill_stance as bf
from memory.graph_memory import GraphMemory
from memory.graph_models import GraphEdge, GraphNode


@pytest.fixture(autouse=True)
def _reset_sentinel():
    importlib.reload(bf)
    yield


def _doc(doc_id, content, metadata=None):
    return {"id": doc_id, "content": content, "metadata": metadata or {}}


SENTINEL = bf.SENTINEL_FACT_ID


class TestPlanFactUpdates:
    def test_sentinel_classifies_appraisal(self):
        updates, stats, sentinel_ok = bf.plan_fact_updates([
            _doc(SENTINEL, "casey | is | evil"),
        ])
        assert sentinel_ok is True
        assert (SENTINEL, "appraisal") in updates

    def test_objective_facts_tagged_objective(self):
        updates, stats, _ = bf.plan_fact_updates([
            _doc("f1", "user | lives_in | chicago"),
        ])
        assert updates == [("f1", "objective")]

    def test_already_tagged_skipped_idempotent(self):
        updates, stats, sentinel_ok = bf.plan_fact_updates([
            _doc(SENTINEL, "casey | is | evil", {"stance": "appraisal"}),
            _doc("f1", "user | lives_in | chicago", {"stance": "objective"}),
        ])
        assert updates == []
        assert stats["already_tagged"] == 2
        assert sentinel_ok is True  # sentinel verified from stored tag

    def test_unparseable_counted_not_written(self):
        updates, stats, _ = bf.plan_fact_updates([
            _doc("f1", "free-text with no triple shape at all whatsoever"),
        ])
        assert stats["unparseable"] >= 0
        assert all(u[0] != "f1" or u[1] for u in updates)

    def test_sentinel_missing_flagged(self):
        _, stats, sentinel_ok = bf.plan_fact_updates([
            _doc("f1", "user | lives_in | chicago"),
        ])
        assert stats["sentinel_seen"] is False
        assert sentinel_ok is False


class TestPlanGraphUpdates:
    def test_appraisal_edge_planned(self, tmp_path):
        gm = GraphMemory(persist_path=str(tmp_path / "g.json"))
        gm.add_entity(GraphNode(entity_id="casey", display_name="Casey"))
        gm.add_entity(GraphNode(entity_id="evil", display_name="evil"))
        gm.add_relation(GraphEdge(source_id="casey", relation="is", target_id="evil"))
        updates = bf.plan_graph_updates(gm)
        assert ("casey|is|evil", "appraisal") in updates

    def test_tagged_edges_skipped(self, tmp_path):
        gm = GraphMemory(persist_path=str(tmp_path / "g.json"))
        gm.add_relation(GraphEdge(source_id="casey", relation="is",
                                  target_id="evil",
                                  metadata={"stance": "appraisal"}))
        assert bf.plan_graph_updates(gm) == []


class TestSafetyContract:
    def test_apply_refused_while_daemon_running(self, monkeypatch):
        monkeypatch.setattr(bf, "_daemon_running", lambda: True)
        assert bf.main(["--apply"]) == 2

    def test_dry_run_exits_nonzero_without_sentinel(self, monkeypatch, tmp_path):
        class FakeStore:
            def list_all(self, name):
                return [_doc("f1", "user | lives_in | chicago")]

        import memory.storage.multi_collection_chroma_store as mcs
        import memory.graph_memory as gmm
        monkeypatch.setattr(mcs, "MultiCollectionChromaStore", lambda **kw: FakeStore())
        monkeypatch.setattr(
            gmm, "GraphMemory",
            lambda: GraphMemory(persist_path=str(tmp_path / "g.json")),
        )
        assert bf.main([]) == 1

    def test_dry_run_passes_with_sentinel_and_writes_nothing(self, monkeypatch, tmp_path):
        written = []

        class FakeStore:
            def list_all(self, name):
                return [_doc(SENTINEL, "casey | is | evil")]

            def update_metadata(self, coll, doc_id, md):
                written.append(doc_id)

        import memory.storage.multi_collection_chroma_store as mcs
        import memory.graph_memory as gmm
        monkeypatch.setattr(mcs, "MultiCollectionChromaStore", lambda **kw: FakeStore())
        monkeypatch.setattr(
            gmm, "GraphMemory",
            lambda: GraphMemory(persist_path=str(tmp_path / "g.json")),
        )
        assert bf.main([]) == 0
        assert written == []  # dry run never writes
