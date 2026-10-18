"""Graph, proposal and upload reads run off the event loop.

2026-10-17, class BC-41 (blocking call on the event loop).  Live loop-stall
records: get_context_sentences hub walks (759-4688 ms) via get_graph_context and
ContextSurfacer.generate_insights, ProposalFilter._semantic_dedup encode
(497-981 ms), reference_docs query_collection via get_documents (607 ms).  Each
test drives THE deployed async function; only the sync seam is replaced by a
recorder of ``threading.get_ident()``.
"""
import threading
from types import SimpleNamespace

import pytest

from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin


class _G(KnowledgeRetrievalMixin):
    def __init__(self):
        self.memory_coordinator = None
        self.memory_id_map = {}
        self._chroma_store = None


@pytest.mark.asyncio
async def test_graph_context_walk_runs_off_loop(monkeypatch):
    monkeypatch.setattr("config.app_config.KNOWLEDGE_GRAPH_ENABLED", True)
    seen = {}

    def fake_ctx(eid, depth=1, max_sentences=5, with_attribution=False):
        seen["ctx"] = threading.get_ident()
        return ["Tamsin works on the kiln."]

    import core.prompt.gatherer_knowledge as mod

    def fake_extract(query, resolver, graph_memory=None):
        seen["extract"] = threading.get_ident()
        return {"tamsin"}

    monkeypatch.setattr(mod.graph_utils, "extract_graph_entities", fake_extract)
    graph = SimpleNamespace(node_count=lambda: 3, get_context_sentences=fake_ctx)
    g = _G()
    g.memory_coordinator = SimpleNamespace(graph_memory=graph, entity_resolver=object())
    loop_thread = threading.get_ident()
    out = await g.get_graph_context("tell me about tamsin")
    assert list(out) == ["Tamsin works on the kiln."]
    assert seen["ctx"] != loop_thread
    assert seen["extract"] != loop_thread


@pytest.mark.asyncio
async def test_generate_insights_graph_walks_run_off_loop(monkeypatch):
    from memory.context_surfacer import ContextSurfacer
    seen = {}
    monkeypatch.setattr("config.app_config.PROACTIVE_SURFACING_ENABLED", True)
    monkeypatch.setattr("config.app_config.PROACTIVE_SURFACING_MIN_GRAPH_NODES", 1)
    monkeypatch.setattr("config.app_config.PROACTIVE_SURFACING_MIN_GRAPH_EDGES", 1)
    graph = SimpleNamespace(node_count=lambda: 5, edge_count=lambda: 5)
    cs = ContextSurfacer(graph, None, None)

    def classify():
        seen["classify"] = threading.get_ident()
        return {"a": object(), "b": object()}

    def identify(query, clusters):
        seen["identify"] = threading.get_ident()
        return {"a"}

    def select(active, clusters, max_candidates=3):
        seen["select"] = threading.get_ident()
        return []

    monkeypatch.setattr(cs, "_classify_user_edges", classify)
    monkeypatch.setattr(cs, "_identify_active_domains", identify)
    monkeypatch.setattr(cs, "_select_bridge_candidates", select)
    loop_thread = threading.get_ident()
    assert await cs.generate_insights("kiln") == []
    assert set(seen) == {"classify", "identify", "select"}
    assert all(t != loop_thread for t in seen.values())


@pytest.mark.asyncio
async def test_proposal_semantic_dedup_and_query_run_off_loop(monkeypatch):
    from core.prompt.proposal_filter import ProposalFilter
    seen = {}
    monkeypatch.setattr("config.app_config.CODE_PROPOSALS_PROMPT_ENABLED", True)

    class _P:
        def to_embedding_text(self):
            return "p"

        def to_metadata(self):
            return {}

    props = [_P()]

    def query_proposals(q, n_results=20, status_filter=None):
        seen["query"] = threading.get_ident()
        return props

    pf = ProposalFilter.__new__(ProposalFilter)
    pf._proposal_store = SimpleNamespace(query_proposals=query_proposals)
    pf._gate_system = object()
    monkeypatch.setattr(pf, "_is_project_related", lambda q: True, raising=False)
    monkeypatch.setattr(pf, "_build_utility_query", lambda: "u", raising=False)
    monkeypatch.setattr(pf, "_keyword_dedup", lambda p, t: p, raising=False)

    def dedup(p, t):
        seen["dedup"] = threading.get_ident()
        return []

    monkeypatch.setattr(pf, "_semantic_dedup", dedup, raising=False)
    loop_thread = threading.get_ident()
    assert await pf.get_proposals("kiln project") == []
    assert seen["query"] != loop_thread
    assert seen["dedup"] != loop_thread


@pytest.mark.asyncio
async def test_get_documents_store_reads_run_off_loop():
    from knowledge.reference_docs_manager import ReferenceDocsManager
    seen = {}

    def fake_query(name, query, n_results=3, where=None):
        seen["query"] = threading.get_ident()
        return []

    m = ReferenceDocsManager.__new__(ReferenceDocsManager)
    m._chroma_store = SimpleNamespace(query_collection=fake_query)

    def kw(query, limit, doc_type=None):
        seen["kw"] = threading.get_ident()
        return []

    m._keyword_search = kw
    loop_thread = threading.get_ident()
    await m.get_documents("kiln notes", limit=6)
    assert seen["query"] != loop_thread
    assert seen["kw"] != loop_thread
