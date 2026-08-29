"""
Appraisal settledness (Phase B3.6, 2026-08-23).

An appraisal edge repeated on >= 3 DISTINCT days, each at non-elevated
capture tone, gains metadata["settled"]=True (the learned_relations
distinct-days pattern). Elevated/unknown-tone days never count — settledness
deliberately under-fires; a crisis-week spiral must not mint a "settled" view.
"""

from datetime import datetime

import pytest

from memory.graph_memory import GraphMemory
from memory.graph_models import GraphEdge, GraphNode


@pytest.fixture
def graph(tmp_path):
    gm = GraphMemory(persist_path=str(tmp_path / "graph.json"))
    gm.add_entity(GraphNode(entity_id="casey", display_name="Casey"))
    gm.add_entity(GraphNode(entity_id="evil", display_name="evil"))
    return gm


def _edge(gm):
    return gm.get_relations("casey")[0]


def _mention(gm, tone, day):
    """Simulate a mention on a given day by driving the tracker directly for
    days other than today (add_relation stamps datetime.now())."""
    incoming = {"stance": "appraisal", "capture_tone": tone}
    edges = gm.get_relations("casey")
    if not edges:
        gm.add_relation(GraphEdge(source_id="casey", relation="is",
                                  target_id="evil", metadata=dict(incoming)))
        edges = gm.get_relations("casey")
        # rewrite today's auto-recorded day to the simulated one
        edge = edges[0]
        edge.metadata["appraisal_days"] = [day]
        edge.metadata["appraisal_tones"] = [tone]
        return
    gm._track_appraisal_settledness(edges[0], incoming, datetime.fromisoformat(day))


class TestSettledness:
    def test_three_non_elevated_days_settles(self, graph):
        for day in ("2026-08-01", "2026-08-05", "2026-08-10"):
            _mention(graph, "non_elevated", day)
        assert _edge(graph).metadata.get("settled") is True

    def test_elevated_days_never_count(self, graph):
        for day in ("2026-08-01", "2026-08-05", "2026-08-10", "2026-08-12"):
            _mention(graph, "elevated", day)
        assert _edge(graph).metadata.get("settled") is not True

    def test_unknown_days_never_count(self, graph):
        for day in ("2026-08-01", "2026-08-05", "2026-08-10"):
            _mention(graph, "unknown", day)
        assert _edge(graph).metadata.get("settled") is not True

    def test_mixed_needs_three_non_elevated(self, graph):
        _mention(graph, "non_elevated", "2026-08-01")
        _mention(graph, "elevated", "2026-08-02")
        _mention(graph, "non_elevated", "2026-08-03")
        assert _edge(graph).metadata.get("settled") is not True
        _mention(graph, "non_elevated", "2026-08-04")
        assert _edge(graph).metadata.get("settled") is True

    def test_same_day_counts_once(self, graph):
        _mention(graph, "non_elevated", "2026-08-01")
        _mention(graph, "non_elevated", "2026-08-01")
        _mention(graph, "non_elevated", "2026-08-01")
        assert _edge(graph).metadata.get("settled") is not True
        assert _edge(graph).metadata["appraisal_days"] == ["2026-08-01"]

    def test_same_day_tone_upgrade(self, graph):
        _mention(graph, "elevated", "2026-08-01")
        _mention(graph, "non_elevated", "2026-08-01")
        assert _edge(graph).metadata["appraisal_tones"] == ["non_elevated"]

    def test_objective_edges_untracked(self, graph):
        graph.add_relation(GraphEdge(source_id="casey", relation="lives_in",
                                     target_id="evil",
                                     metadata={"stance": "objective"}))
        edge = next(e for e in graph.get_relations("casey")
                    if e.relation == "lives_in")
        assert "appraisal_days" not in edge.metadata

    def test_add_relation_records_today(self, graph):
        graph.add_relation(GraphEdge(
            source_id="casey", relation="is", target_id="evil",
            metadata={"stance": "appraisal", "capture_tone": "non_elevated"},
        ))
        edge = _edge(graph)
        today = datetime.now().strftime("%Y-%m-%d")
        assert edge.metadata["appraisal_days"] == [today]
        assert edge.metadata.get("settled") is not True
