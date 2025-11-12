"""
Additional integration tests for MemoryCoordinator covering semantic/top paths
and acceptance thresholds without invoking heavy embedding queries.

Covers:
- get_semantic_top_memories meta-query path (recency only)
- get_semantic_top_memories with stubbed semantic results and gate_system
- get_memories acceptance threshold behavior (deictic vs normal)
- search_by_type normalization using a stubbed store query
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from memory.corpus_manager import CorpusManager
from memory.memory_coordinator import MemoryCoordinator
from typing import Any, Dict


class FakeChromaStore:
    """Lightweight stand-in for MultiCollectionChromaStore to avoid model loads."""
    def __init__(self):
        self.collections: Dict[str, Any] = {"facts": object(), "reflections": object()}

    # Optional API used by get_memories() to persist metadata updates
    def bulk_update_metadata(self, *_, **__):
        return None

    # search_by_type relies on this; in tests we monkeypatch it
    def query_collection(self, *_args, **_kwargs):
        return []


class StubGate:
    async def filter_memories(self, _query, chunks):
        # Behaves like gate system: returns same chunk objects (not originals)
        # with a synthetic relevance score to propagate back
        out = []
        for ch in chunks[:3]:
            ch = dict(ch)
            ch["relevance_score"] = 0.88
            out.append(ch)
        return out


@pytest.fixture()
def temp_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield {
            "corpus_file": str(Path(tmpdir) / "corpus.json"),
            "chroma_path": str(Path(tmpdir) / "chroma_db"),
        }


@pytest.fixture()
def coordinator(temp_env):
    cm = CorpusManager(corpus_file=temp_env["corpus_file"])
    store = FakeChromaStore()
    # keep gate_system None by default for tests that avoid it
    return MemoryCoordinator(corpus_manager=cm, chroma_store=store)


@pytest.mark.asyncio
async def test_get_semantic_top_memories_meta_path(coordinator: MemoryCoordinator):
    # Seed a couple of conversations
    await coordinator.store_interaction("We talked math", "Vectors", tags=["topic:math"]) 
    await coordinator.store_interaction("We talked art", "Impressionism", tags=["topic:art"]) 

    # Meta query should route to recency path without hitting semantic search
    out = await coordinator.get_semantic_top_memories("do you recall our talk?", limit=5)
    assert isinstance(out, list) and len(out) >= 1


@pytest.mark.asyncio
async def test_get_semantic_top_memories_with_gate_and_stubbed_semantics(coordinator: MemoryCoordinator, monkeypatch):
    # Stub the semantic gather to avoid embeddings/Chroma queries
    async def _fake_sem(_q: str, n_results: int = 30):
        return [
            {"id": "a1", "content": "alpha about cats", "relevance_score": 0.2},
            {"id": "b2", "content": "beta about cats", "relevance_score": 0.5},
            {"id": "c3", "content": "gamma about cats", "relevance_score": 0.7},
        ]

    monkeypatch.setattr(coordinator, "_get_semantic_memories", _fake_sem, raising=True)
    # Attach a stub gate system; coordinator will use it to filter chunks
    coordinator.gate_system = StubGate()

    out = await coordinator.get_semantic_top_memories("cats topic", limit=2)
    assert isinstance(out, list) and len(out) == 2
    # Marked as pre_gated by the method after gate pass
    assert all(m.get("pre_gated") for m in out)


@pytest.mark.asyncio
async def test_get_memories_acceptance_thresholds(coordinator: MemoryCoordinator, monkeypatch):
    # Bypass heavy combine: feed combined list directly via monkeypatch
    async def _fake_combine(*, very_recent, semantic, hierarchical, query, config):
        return [{"id": "x"}, {"id": "y"}]

    def _fake_rank(_combined, _query):
        # Assign different final scores to exercise thresholds
        return [
            {"id": "x", "final_score": 0.55},
            {"id": "y", "final_score": 0.65},
        ]

    monkeypatch.setattr(coordinator, "_combine_memories", _fake_combine, raising=True)
    monkeypatch.setattr(coordinator, "_rank_memories", _fake_rank, raising=True)

    # Deictic query should keep only the higher-scored memory (threshold ~0.60)
    d = await coordinator.get_memories("explain that again", limit=5)
    assert isinstance(d, list)
    # Expect that at least the higher-scored item is present
    assert any(m.get("final_score", 0) >= 0.65 for m in d)

    # Normal query should accept both (threshold lower ~0.35)
    n = await coordinator.get_memories("tell me about cats", limit=5)
    assert isinstance(n, list) and len(n) == 2


@pytest.mark.asyncio
async def test_search_by_type_uses_store_query(monkeypatch, temp_env):
    # Fresh coordinator instance to avoid cross-test mutations
    cm = CorpusManager(corpus_file=temp_env["corpus_file"])
    c2 = MemoryCoordinator(corpus_manager=cm, chroma_store=FakeChromaStore())
    # Ensure the collection reports non-zero count
    class _DummyColl:
        def count(self):
            return 2
    c2.chroma_store.collections["facts"] = _DummyColl()

    # Monkeypatch the store query to avoid embeddings
    def _fake_query(_name, query_text=None, n_results=5, **_):
        return [
            {"id": "id1", "content": "f1", "metadata": {"timestamp": "2024-01-01T00:00:00"}},
            {"id": "id2", "content": "f2", "metadata": {"timestamp": "2024-01-01T00:00:10"}},
        ][:n_results]

    monkeypatch.setattr(c2.chroma_store, "query_collection", _fake_query, raising=True)

    out = await c2.search_by_type("facts", query="q", limit=2)
    assert isinstance(out, list) and len(out) == 2
    # Timestamp should be hoisted to top-level by search_by_type
    assert all("timestamp" in i for i in out)


@pytest.mark.asyncio
async def test_get_recent_facts_and_fallback(monkeypatch, temp_env):
    cm = CorpusManager(corpus_file=temp_env["corpus_file"])
    c3 = MemoryCoordinator(corpus_manager=cm, chroma_store=FakeChromaStore())

    # Provide get_recent implementation on fake store
    def _recent(name, limit=5):
        assert name == "facts"
        return [
            {"content": "fA", "metadata": {"timestamp": "2024-01-01T00:00:00", "confidence": 0.7}},
            {"content": "fB", "metadata": {"timestamp": "2024-01-01T00:00:10", "confidence": 0.8}},
        ][:limit]

    setattr(c3.chroma_store, "get_recent", _recent)

    # Ensure facts collection reports zero to force fallback path inside get_facts
    class _ZeroColl:
        def count(self):
            return 0

    c3.chroma_store.collections["facts"] = _ZeroColl()

    rf = await c3.get_recent_facts(limit=2)
    assert isinstance(rf, list) and len(rf) == 2

    # get_facts should fallback to get_recent due to count=0
    gf = await c3.get_facts(query="anything", limit=2)
    assert isinstance(gf, list) and len(gf) == 2
