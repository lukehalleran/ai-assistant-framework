"""
Graph-expansion evidence bar (2026-08-23).

A node the graph has seen ONCE is a single extraction event, not an
established concept — its display name must not steer query expansion into
unrelated retrievals. rank_expansion_candidates skips candidates whose
mention_count is below GRAPH_EXPANSION_MIN_MENTIONS (default 2, env-
overridable, 0 disables). Nodes without the attribute (mocks, legacy
stores) are exempt so nothing regresses on stores that never tracked it.

Drives THE DEPLOYED rank_expansion_candidates with real GraphMemory nodes.
"""

import pytest

from memory.graph_memory import GraphMemory
from memory.graph_models import GraphEdge, GraphNode
from memory.graph_utils import rank_expansion_candidates


def _graph(tmp_path, mentions_by_id):
    g = GraphMemory(persist_path=str(tmp_path / "min_mentions_graph.json"))
    for eid, count in mentions_by_id.items():
        g.add_entity(GraphNode(entity_id=eid, display_name=eid.title(),
                               entity_type="other", mention_count=count))
    for eid in mentions_by_id:
        if eid != "seed":
            g.add_relation(GraphEdge(source_id="seed", relation="related_to",
                                     target_id=eid))
    return g


def _expand(g, **kw):
    return rank_expansion_candidates({"seed"}, g, depth=1, skip_ids={"user"},
                                     max_terms=8, **kw)


class TestMinMentionsBar:
    def test_single_mention_candidate_filtered(self, tmp_path):
        g = _graph(tmp_path, {"seed": 5, "established": 3, "incidental": 1})
        result = _expand(g)
        assert "Established" in result
        assert "Incidental" not in result

    def test_two_mentions_clear_default_bar(self, tmp_path):
        g = _graph(tmp_path, {"seed": 5, "borderline": 2})
        assert "Borderline" in _expand(g)

    def test_param_overrides_default(self, tmp_path):
        g = _graph(tmp_path, {"seed": 5, "incidental": 1})
        assert "Incidental" in _expand(g, min_mentions=0)
        assert "Incidental" not in _expand(g, min_mentions=2)

    def test_env_override(self, tmp_path, monkeypatch):
        g = _graph(tmp_path, {"seed": 5, "borderline": 2})
        monkeypatch.setenv("GRAPH_EXPANSION_MIN_MENTIONS", "3")
        assert "Borderline" not in _expand(g)
        monkeypatch.setenv("GRAPH_EXPANSION_MIN_MENTIONS", "0")
        assert "Borderline" in _expand(g)

    def test_bad_env_value_falls_back_to_default(self, tmp_path, monkeypatch):
        g = _graph(tmp_path, {"seed": 5, "established": 3, "incidental": 1})
        monkeypatch.setenv("GRAPH_EXPANSION_MIN_MENTIONS", "not-a-number")
        result = _expand(g)
        assert "Established" in result
        assert "Incidental" not in result


class TestLegacyNodesExempt:
    def test_node_without_mention_count_kept(self, tmp_path):
        """Mocks / legacy stores whose nodes lack the attribute must pass."""
        g = _graph(tmp_path, {"seed": 5, "established": 3})

        class BareNode:
            def __init__(self, display_name):
                self.display_name = display_name

        real_get = g.get_entity

        def patched(eid):
            node = real_get(eid)
            if node is not None and eid == "established":
                return BareNode(node.display_name)
            return node

        g.get_entity = patched
        assert "Established" in _expand(g)
