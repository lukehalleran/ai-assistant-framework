"""Tests for core/insight/sweep.py — the ungated cross-store evidence sweep."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from core.insight.sweep import SWEEP_COLLECTIONS, _finalize, run_sweep
from core.insight.types import EvidenceItem, FacetPlan, FacetQuery


def _caps(**over):
    caps = {
        "per_facet_cap": 10,
        "total_evidence_cap": 80,
        "evidence_snippet_chars": 280,
        "keyword_scan_max": 50,
        "expand_top_k": 3,
        "expand_window": 2,
        "sweep_timeout_s": 45.0,
    }
    caps.update(over)
    return caps


def _chroma_row(coll, i, content="some content", ts="2026-08-01T12:00:00"):
    return {
        "id": f"{coll}_{i}",
        "content": content,
        "metadata": {"timestamp": ts},
        "relevance_score": 0.5,
        "collection": coll,
        "rank": i,
    }


@pytest.fixture
def store():
    s = MagicMock()
    s.query_collection.side_effect = lambda coll, q, n: [
        _chroma_row(coll, i) for i in range(2)
    ]
    return s


@pytest.fixture
def corpus():
    c = MagicMock()
    c.search_keyword.return_value = [
        {"timestamp": datetime(2026, 8, 18, 13, 52), "speaker": "user",
         "matched_term": "abusive", "excerpt": "wasn't abusive", "query_preview": "x"},
    ]
    return c


class TestSweepCoverage:
    def test_all_six_collections_queried_no_gate(self, store, corpus):
        plan = FacetPlan(facets=[FacetQuery(name="f1", query_text="q", keywords=["abusive"])])
        items = asyncio.run(run_sweep(
            plan, chroma_store=store, corpus_manager=corpus, caps=_caps(expand_top_k=0),
        ))
        queried = {call.args[0] for call in store.query_collection.call_args_list}
        assert queried == set(SWEEP_COLLECTIONS)
        # no gate anywhere: every returned row became evidence + corpus hit
        assert len(items) == 2 * len(SWEEP_COLLECTIONS) + 1
        assert any(i.collection == "corpus" and i.speaker == "user" for i in items)

    def test_collection_failure_is_partial_not_fatal(self, corpus):
        s = MagicMock()

        def _query(coll, q, n):
            if coll == "threads":
                raise RuntimeError("collection unopened")
            return [_chroma_row(coll, 0)]

        s.query_collection.side_effect = _query
        plan = FacetPlan(facets=[FacetQuery(name="f1", query_text="q")])
        items = asyncio.run(run_sweep(
            plan, chroma_store=s, corpus_manager=corpus, caps=_caps(expand_top_k=0),
        ))
        assert len(items) == len(SWEEP_COLLECTIONS) - 1

    def test_timeout_returns_partial(self, corpus):
        s = MagicMock()

        def _slow(coll, q, n):
            import time
            if coll != "conversations":
                time.sleep(3)
            return [_chroma_row(coll, 0)]

        s.query_collection.side_effect = _slow
        plan = FacetPlan(facets=[FacetQuery(name="f1", query_text="q")])
        items = asyncio.run(run_sweep(
            plan, chroma_store=s, corpus_manager=corpus,
            caps=_caps(sweep_timeout_s=0.5, expand_top_k=0),
        ))
        # partial: at least the fast collection, never an exception
        assert isinstance(items, list)


class TestGraphSweep:
    def _graph(self, degree=3, stale=False):
        from memory.graph_models import GraphEdge, GraphNode
        g = MagicMock()
        node = GraphNode(entity_id="casey", display_name="Casey")
        g.get_entity.side_effect = lambda eid: (
            node if eid == "casey" else GraphNode(entity_id=eid, display_name=eid)
        )
        g.graph.degree.return_value = degree
        edge = GraphEdge(
            source_id="casey", relation="is", target_id="evil",
            first_seen=datetime(2026, 8, 18), last_seen=datetime(2026, 8, 18),
            metadata={"stance": "appraisal"},
        )
        g.get_relations.return_value = [edge]
        g._edge_is_stale_transient.return_value = stale
        return g

    def _run(self, graph, entity="casey"):
        store = MagicMock()
        store.query_collection.return_value = []
        resolver = MagicMock()
        resolver.resolve.side_effect = lambda m: m.lower()
        plan = FacetPlan(facets=[FacetQuery(name="f", query_text="q", entities=[entity])])
        return asyncio.run(run_sweep(
            plan, chroma_store=store, corpus_manager=None,
            graph_memory=graph, entity_resolver=resolver, caps=_caps(expand_top_k=0),
        ))

    def test_edge_collected_with_stance(self):
        items = self._run(self._graph())
        assert len(items) == 1
        assert items[0].collection == "graph"
        assert items[0].is_appraisal is True  # explicit stance metadata honored
        # B3.2 stance-aware rendering: the appraisal edge arrives already
        # attributed ("you described Casey as ..."), never bare "Casey is evil"
        assert "Casey" in items[0].text
        assert "you described" in items[0].text

    def test_hub_not_expanded(self):
        from config.app_config import GRAPH_EXPANSION_HUB_DEGREE
        items = self._run(self._graph(degree=GRAPH_EXPANSION_HUB_DEGREE))
        assert items == []

    def test_user_entity_skipped(self):
        graph = self._graph()
        items = self._run(graph, entity="user")
        assert items == []
        graph.get_relations.assert_not_called()

    def test_stale_transient_edges_dropped(self):
        items = self._run(self._graph(stale=True))
        assert items == []


class TestExpansion:
    def test_expansion_only_conversations(self, corpus):
        s = MagicMock()
        s.query_collection.side_effect = lambda coll, q, n: (
            [_chroma_row(coll, 0)] if coll in ("conversations", "threads") else []
        )
        expander = MagicMock()
        expander.expand.return_value = {
            "turns": [{"id": "conversations_x", "content": "expanded turn",
                       "metadata": {"timestamp": "2026-08-02T10:00:00"}}],
        }
        plan = FacetPlan(facets=[FacetQuery(name="f", query_text="q")])
        items = asyncio.run(run_sweep(
            plan, chroma_store=s, corpus_manager=None,
            memory_expander=expander, caps=_caps(),
        ))
        # expand called ONLY for the conversation hit, never the thread hit
        assert expander.expand.call_count == 1
        assert expander.expand.call_args.args[0] == "conversations_0"
        assert any(i.text == "expanded turn" for i in items)


class TestFinalize:
    def test_dedupe_and_sort(self):
        items = [
            EvidenceItem(doc_id="a", text="one", date="2026-08-01", collection="conversations"),
            EvidenceItem(doc_id="a", text="one", date="2026-08-01", collection="conversations"),
            EvidenceItem(doc_id="b", text="two", date="2026-08-20", collection="facts"),
        ]
        out = _finalize(items, _caps())
        assert len(out) == 2
        assert out[0].date == "2026-08-20"  # newest first

    def test_snippet_clip(self):
        items = [EvidenceItem(doc_id="a", text="x" * 1000, collection="conversations")]
        out = _finalize(items, _caps(evidence_snippet_chars=100))
        assert len(out[0].text) <= 101  # clip + ellipsis

    def test_total_cap_proportional(self):
        items = (
            [EvidenceItem(doc_id=f"c{i}", text=f"conv {i}", date="2026-08-01",
                          collection="conversations") for i in range(90)]
            + [EvidenceItem(doc_id=f"g{i}", text=f"graph {i}", date="2026-08-01",
                            collection="graph") for i in range(10)]
        )
        out = _finalize(items, _caps(total_evidence_cap=20))
        assert len(out) <= 20
        # minority collection keeps at least one slot
        assert any(i.collection == "graph" for i in out)
