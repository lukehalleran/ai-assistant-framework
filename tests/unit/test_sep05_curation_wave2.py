"""2026-09-05 Wave-1 curator additions (deterministic ports of the two
cleanups the owner had to run from a terminal that evening —
purge_profile_facts + graph_junk_cleanup's temporal T1 class — as one-click,
reversible Curation Center cards). "wave2" in this file's name means the
second BATCH of curators shipped, not the design doc's Wave 2 (temporal
staleness) — see docs/AUTONOMOUS_CURATION_DESIGN.md §3 for the numbering.

- ProfileJunkFactCurator: quick-profile facts the deployed
  fact_extractor._is_junk_object rejects → supersede (is_current=False).
- GraphTemporalNodeCurator + graph adapter: when-word graph nodes per the
  deployed graph_utils.is_temporal_deictic → node-level quarantine flag,
  honoured by GraphMemory.edge_is_suppressed at read time.
"""
from __future__ import annotations

import json

from memory.curation.curators import (
    ALL_CURATORS,
    GraphTemporalNodeCurator,
    ProfileJunkFactCurator,
)
from memory.curation.engine import CurationEngine, StoreBundle
from memory.curation.journal import CurationJournal
from memory.curation.types import Confidence, Instrument, ItemChange, ProposalStatus
from memory.graph_memory import GraphMemory
from memory.graph_models import GraphEdge, GraphNode


class FakeProfile:
    def __init__(self, facts):
        self.profile = {"categories": {"health": list(facts)}}
        self.saves = 0

    def save(self):
        self.saves += 1


def make_engine(tmp_path, stores):
    return CurationEngine(
        stores,
        queue_path=str(tmp_path / "queue.json"),
        journal=CurationJournal(str(tmp_path / "audit.jsonl")),
    )


_FACTS = [
    {"fact_id": "f_today", "relation": "time_off_work", "value": "today", "is_current": True},
    {"fact_id": "f_weekday", "relation": "texted", "value": "on thursday", "is_current": True},
    {"fact_id": "f_clause", "relation": "has_doctor",
     "value": "Rowan is cautious about drinking due to past accident", "is_current": True},
    {"fact_id": "f_demo", "relation": "works_on", "value": "this assistant", "is_current": True},
    {"fact_id": "f_real", "relation": "likes", "value": "pizza", "is_current": True},
    {"fact_id": "f_portal", "relation": "doctor_communication", "value": "no patient portal", "is_current": True},
    {"fact_id": "f_sched", "relation": "day_off", "value": "thursday", "is_current": True},
    {"fact_id": "f_old", "relation": "goal", "value": "yesterday", "is_current": False},
    {"relation": "goal", "value": "yesterday", "is_current": True},  # no id → untouchable
]


class TestProfileJunkFactCurator:
    def test_registered(self):
        assert ProfileJunkFactCurator in ALL_CURATORS
        assert GraphTemporalNodeCurator in ALL_CURATORS

    def test_sentinels_pass(self):
        assert all(s.passed for s in ProfileJunkFactCurator().sentinels(StoreBundle()))

    def test_scan_flags_only_junk_current_addressable_facts(self):
        props = ProfileJunkFactCurator().scan(StoreBundle(user_profile=FakeProfile(_FACTS)))
        assert len(props) == 1 and props[0].batch
        assert props[0].instrument == Instrument.METADATA
        assert props[0].confidence == Confidence.DETERMINISTIC
        assert {i.doc_id for i in props[0].items} == {"f_today", "f_weekday", "f_clause", "f_demo"}
        assert all(i.change_type == "supersede_profile_fact" for i in props[0].items)

    def test_no_profile_or_clean_profile_no_proposal(self):
        assert ProfileJunkFactCurator().scan(StoreBundle()) == []
        clean = FakeProfile([{"fact_id": "x", "relation": "likes", "value": "pizza", "is_current": True}])
        assert ProfileJunkFactCurator().scan(StoreBundle(user_profile=clean)) == []

    def test_engine_queue_apply_undo_roundtrip(self, tmp_path):
        profile = FakeProfile([dict(f) for f in _FACTS])
        engine = make_engine(tmp_path, StoreBundle(user_profile=profile))
        engine.register(ProfileJunkFactCurator())
        report = engine.run_scan()
        assert report.proposals_queued == 1 and not report.sentinel_failures
        p = engine.pending()[0]
        assert p.status == ProposalStatus.PENDING  # queue ceiling: nothing auto-applies
        engine.apply(p.proposal_id, actor="human")
        by_id = {f.get("fact_id"): f for cat in profile.profile["categories"].values() for f in cat}
        assert by_id["f_today"]["is_current"] is False
        assert by_id["f_today"]["curation_stale_reason"] == "junk_object"
        assert by_id["f_real"]["is_current"] is True
        assert profile.saves >= 1
        # A second scan must not stack a duplicate card.
        assert engine.run_scan().proposals_queued == 0
        engine.undo(p.proposal_id)
        assert by_id["f_today"]["is_current"] is True
        assert "curation_stale_reason" not in by_id["f_today"]


def _graph(tmp_path) -> GraphMemory:
    gm = GraphMemory(persist_path=str(tmp_path / "graph.json"))
    for eid, name, etype in (("user", "User", "person"), ("today", "today", "other"),
                             ("on_thursday", "on Thursday", "other"), ("biscuit", "Biscuit", "pet"),
                             ("rowan", "Rowan", "person")):
        gm.add_entity(GraphNode(entity_id=eid, display_name=name, entity_type=etype))
    gm.add_relation(GraphEdge(source_id="user", target_id="today", relation="dad", weight=1.0))
    gm.add_relation(GraphEdge(source_id="rowan", target_id="on_thursday", relation="texted", weight=1.0))
    gm.add_relation(GraphEdge(source_id="user", target_id="biscuit", relation="has_cat", weight=1.0))
    gm.add_relation(GraphEdge(source_id="user", target_id="rowan", relation="friend_of", weight=1.0))
    gm.save()
    return gm


class TestGraphTemporalNodeCurator:
    def test_sentinels_pass(self):
        assert all(s.passed for s in GraphTemporalNodeCurator().sentinels(StoreBundle()))

    def test_scan_flags_when_word_nodes_only(self, tmp_path):
        gm = _graph(tmp_path)
        props = GraphTemporalNodeCurator().scan(StoreBundle(graph_memory=gm))
        assert len(props) == 1 and props[0].batch
        assert props[0].instrument == Instrument.METADATA
        assert {i.doc_id for i in props[0].items} == {"today", "on_thursday"}
        assert GraphTemporalNodeCurator().scan(StoreBundle()) == []

    def test_engine_apply_flags_node_persists_and_undo_restores(self, tmp_path):
        gm = _graph(tmp_path)
        engine = make_engine(tmp_path, StoreBundle(graph_memory=gm))
        engine.register(GraphTemporalNodeCurator())
        engine.run_scan()
        p = engine.pending()[0]
        engine.apply(p.proposal_id, actor="human")
        assert gm.get_entity("today").metadata.get("curation_quarantined") is True
        assert gm.get_entity("today").metadata.get("curation_quarantine_reason") == "temporal_deictic"
        on_disk = json.load(open(tmp_path / "graph.json"))
        assert on_disk["nodes"]["today"]["metadata"]["curation_quarantined"] is True
        assert gm.node_count() == 5  # nothing deleted
        # Already-quarantined nodes are not re-proposed.
        assert GraphTemporalNodeCurator().scan(StoreBundle(graph_memory=gm)) == []
        engine.undo(p.proposal_id)
        assert "curation_quarantined" not in gm.get_entity("today").metadata
        assert "curation_quarantine_reason" not in gm.get_entity("today").metadata

    def test_node_quarantine_hides_edges_at_read_time(self, tmp_path):
        gm = _graph(tmp_path)
        edge = GraphEdge(source_id="user", target_id="biscuit", relation="has_cat", weight=1.0)
        assert not gm.edge_is_suppressed(edge)
        from memory.curation.adapters import apply_change, revert_change
        change = ItemChange(store="graph", doc_id="biscuit", change_type="quarantine_node",
                            after={"curation_quarantine_reason": "test"})
        apply_change(change, graph_memory=gm)
        assert gm.edge_is_suppressed(edge)
        assert "Biscuit" not in " ".join(gm.get_context_sentences("user"))
        assert change.before == {"curation_quarantined": None, "curation_quarantine_reason": None}
        revert_change(change, graph_memory=gm)
        assert not gm.edge_is_suppressed(edge)
        assert "Biscuit" in " ".join(gm.get_context_sentences("user"))

    def test_adapter_refuses_unknown_node_and_missing_graph(self, tmp_path):
        import pytest
        from memory.curation.adapters import AdapterError, apply_change
        gm = _graph(tmp_path)
        with pytest.raises(AdapterError):
            apply_change(ItemChange(store="graph", doc_id="nope", change_type="quarantine_node"), graph_memory=gm)
        with pytest.raises(AdapterError):
            apply_change(ItemChange(store="graph", doc_id="today", change_type="quarantine_node"))
        with pytest.raises(AdapterError):
            apply_change(ItemChange(store="graph", doc_id="today", change_type="remove_entity"), graph_memory=gm)
