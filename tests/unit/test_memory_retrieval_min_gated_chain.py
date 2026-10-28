"""Behavioural sibling for the source-text ChainWiring tests in
tests/unit/test_gate_min_results_cap.py (BC-63: a test that greps source text
passes while the behaviour is broken).

Incident date: 2026-08-18 (casual_social turn force-fed 8 below-threshold
memories). Bug classes: BC-63, BC-62. The intent budget (``limit``) must flow
gatherer -> coordinator -> retriever -> gate as ``min_gated`` / ``min_results``.
Each hop is DRIVEN here with a recording fake downstream, so a dropped keyword
fails the test whatever the source text says.
"""
import pytest

from core.prompt.gatherer_memory import MemoryRetrievalMixin
from memory.memory_coordinator import MemoryCoordinator
from memory.memory_retriever import MemoryRetriever


class _RecordingCoordinator:
    def __init__(self):
        self.calls = []

    async def get_memories(self, query, limit=20, topic_filter=None, min_gated=None):
        self.calls.append({"limit": limit, "min_gated": min_gated})
        return []


class _Gatherer(MemoryRetrievalMixin):
    def _expand_query_with_graph(self, query):
        return query


@pytest.mark.asyncio
async def test_gatherer_passes_intent_budget_as_min_gated():
    g = _Gatherer()
    g.memory_coordinator = _RecordingCoordinator()
    await g._get_semantic_memories("what did we say about the garden", limit=3)
    assert g.memory_coordinator.calls, "coordinator was never queried"
    assert g.memory_coordinator.calls[0]["min_gated"] == 3


@pytest.mark.asyncio
async def test_coordinator_delegates_min_gated_to_retriever():
    seen = {}

    class _Retriever:
        async def get_memories(self, query, limit, topic_filter, min_gated=None):
            seen["min_gated"] = min_gated
            return []

    coord = object.__new__(MemoryCoordinator)
    coord._retriever = _Retriever()
    coord.current_topic = ""
    coord.conversation_context = []
    await coord.get_memories("q", limit=20, min_gated=4)
    assert seen["min_gated"] == 4
    await coord.get_memories("q", limit=20)
    assert seen["min_gated"] is None


@pytest.mark.asyncio
async def test_retriever_gate_receives_min_results():
    seen = {}

    class _Gate:
        async def filter_memories(self, query, chunks, min_results=None):
            seen["min_results"] = min_results
            return chunks

    r = object.__new__(MemoryRetriever)
    r.gate_system = _Gate()
    out = await r._gate_memories("q", [{"content": "a memory"}], min_results=5)
    assert seen["min_results"] == 5
    assert [m["content"] for m in out] == ["a memory"]


@pytest.mark.asyncio
async def test_retriever_combine_threads_min_gated_to_gate():
    seen = {}

    class _Gate:
        async def filter_memories(self, query, chunks, min_results=None):
            seen["min_results"] = min_results
            return chunks

    r = object.__new__(MemoryRetriever)
    r.gate_system = _Gate()
    await r._combine_memories(
        very_recent=[], semantic=[{"content": "sem one", "timestamp": "t"}],
        hierarchical=[], query="q", config={"max_memories": 10}, min_gated=2,
    )
    assert seen["min_results"] == 2
