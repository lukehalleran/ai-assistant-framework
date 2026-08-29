"""
Read-side stance consumers (Phase B3, 2026-08-23).

  B3.1 graph_utils.rank_expansion_candidates — explicit appraisal/inferred
       edges excluded from traversal/scoring/terms (fixes the "evil" leak);
       legacy untagged edges unchanged.
  B3.2 GraphEdge.to_natural_language + memory_retriever._present_fact_content
       — appraisals attributed + dated, never asserted in system voice;
       objective/legacy output byte-identical.
  B3.3 cross_deduplicator._find_fact_contradictions — explicit-appraisal
       facts never enter contradiction clusters (perspectives coexist).
  B3.4 user_profile.add_fact — explicit appraisals never promote into the
       always-rendered quick profile.
"""

from datetime import datetime

import pytest

from memory.graph_memory import GraphMemory
from memory.graph_models import GraphEdge, GraphNode
from memory.graph_utils import rank_expansion_candidates
from memory.memory_retriever import _present_fact_content


@pytest.fixture
def graph(tmp_path):
    gm = GraphMemory(persist_path=str(tmp_path / "graph.json"))
    for eid, name in (("casey", "Casey"), ("evil", "evil"), ("chicago", "Chicago"),
                      ("seed", "Seed")):
        gm.add_entity(GraphNode(entity_id=eid, display_name=name))
    return gm


class TestExpansionFilter:
    def test_appraisal_edge_never_routes_expansion(self, graph):
        graph.add_relation(GraphEdge(source_id="casey", relation="is",
                                     target_id="evil",
                                     metadata={"stance": "appraisal"}))
        graph.add_relation(GraphEdge(source_id="casey", relation="lives_in",
                                     target_id="chicago"))
        names = rank_expansion_candidates({"casey"}, graph, depth=1, min_mentions=0)
        assert "evil" not in [n.lower() for n in names]
        assert "chicago" in [n.lower() for n in names]

    def test_inferred_edge_excluded(self, graph):
        graph.add_relation(GraphEdge(source_id="casey", relation="is",
                                     target_id="evil",
                                     metadata={"stance": "inferred"}))
        names = rank_expansion_candidates({"casey"}, graph, depth=1, min_mentions=0)
        assert "evil" not in [n.lower() for n in names]

    def test_legacy_untagged_edge_unchanged(self, graph):
        # conservative missing-field semantics: suppression only on EXPLICIT tags
        graph.add_relation(GraphEdge(source_id="casey", relation="is",
                                     target_id="evil"))
        names = rank_expansion_candidates({"casey"}, graph, depth=1, min_mentions=0)
        assert "evil" in [n.lower() for n in names]


class TestEdgeRendering:
    def _edge(self, **md):
        return GraphEdge(source_id="casey", relation="is", target_id="evil",
                         last_seen=datetime(2026, 8, 18), metadata=md)

    def test_objective_byte_identical(self):
        edge = GraphEdge(source_id="user", relation="lives_in", target_id="chicago")
        assert edge.to_natural_language("User", "Chicago") == "User lives in Chicago"
        assert edge.to_natural_language("User", "Chicago", with_attribution=True) \
            == "User lives in Chicago (from relationship data)"

    def test_appraisal_attributed_and_dated(self):
        out = self._edge(stance="appraisal").to_natural_language("Casey", "evil")
        assert out.startswith("you described Casey as ")
        assert "2026-08-18" in out
        assert "Casey is evil" not in out  # never asserted in system voice

    def test_settled_appraisal_wording(self):
        out = self._edge(stance="appraisal", settled=True).to_natural_language("Casey", "evil")
        assert out.startswith("you've consistently described Casey")

    def test_inferred_marked(self):
        out = self._edge(stance="inferred").to_natural_language("Casey", "evil")
        assert "assistant inference" in out


class TestFactPresentation:
    def test_appraisal_rewritten(self):
        out = _present_fact_content(
            "casey | is | evil",
            {"stance": "appraisal", "timestamp": "2026-08-18T13:52:10"},
        )
        assert out == "you described casey as 'evil' (your words at the time, 2026-08-18)"

    def test_self_appraisal_uses_yourself(self):
        out = _present_fact_content(
            "user | is | a failure", {"stance": "appraisal"},
        )
        assert "you described yourself as 'a failure'" in out

    def test_objective_byte_identical(self):
        content = "user | lives_in | chicago"
        assert _present_fact_content(content, {"stance": "objective"}) == content

    def test_legacy_untagged_byte_identical(self):
        content = "casey | is | evil"
        assert _present_fact_content(content, {}) == content
        assert _present_fact_content(content, None) == content

    def test_inferred_marked(self):
        out = _present_fact_content(
            "user | avoids | conflict", {"stance": "inferred"},
        )
        assert "assistant inference" in out


class TestDedupAppraisalSkip:
    def _dedup(self):
        from memory.cross_deduplicator import CrossCollectionDeduplicator
        return CrossCollectionDeduplicator.__new__(CrossCollectionDeduplicator)

    def _doc(self, i, obj, stance=None):
        md = {"subject": "casey", "relation": "described_as", "object": obj,
              "timestamp": f"2026-08-{10 + i}T12:00:00"}
        if stance:
            md["stance"] = stance
        return {"id": f"f{i}", "content": f"casey | described_as | {obj}",
                "metadata": md, "collection": "facts"}

    def test_appraisal_pair_never_clusters(self):
        d = self._dedup()
        docs = [self._doc(1, "evil", stance="appraisal"),
                self._doc(2, "kind at first", stance="appraisal")]
        assert d._find_fact_contradictions(docs) == []

    def test_untagged_pair_still_clusters(self):
        d = self._dedup()
        docs = [self._doc(1, "evil"), self._doc(2, "kind at first")]
        clusters = d._find_fact_contradictions(docs)
        assert len(clusters) == 1


class TestProfileQuickPromotion:
    def _profile(self, tmp_path):
        from memory.user_profile import UserProfile
        return UserProfile(profile_path=str(tmp_path / "profile.json"))

    def test_objective_promotes(self, tmp_path):
        p = self._profile(tmp_path)
        assert p.add_fact("name", "Luke", confidence=0.9)
        assert p.profile["quick_profile"].get("name") == "Luke"

    def test_explicit_appraisal_never_promotes(self, tmp_path):
        p = self._profile(tmp_path)
        assert p.add_fact("name", "a worthless failure", confidence=0.9,
                          stance="appraisal")
        assert "name" not in p.profile["quick_profile"]

    def test_deterministic_backstop_catches_untagged_appraisal(self, tmp_path):
        # caller passes NO stance; the lexicon still catches the thick term
        p = self._profile(tmp_path)
        assert p.add_fact("name", "a worthless failure", confidence=0.9)
        assert "name" not in p.profile["quick_profile"]

    def test_stance_stored_on_fact(self, tmp_path):
        p = self._profile(tmp_path)
        p.add_fact("self_view", "a failure", confidence=0.8, stance="appraisal")
        cats = p.profile["categories"]
        stored = [f for facts in cats.values() for f in facts
                  if isinstance(f, dict) and f.get("value") == "a failure"]
        assert stored and stored[0].get("stance") == "appraisal"
