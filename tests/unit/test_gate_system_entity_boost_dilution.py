"""Diluted entity sets give no boost.

Incident 2026-10-17, class BC-09 (analysis run on the merged attachment blob):
a 1,629-char entity set extracted from an attachment blob gave nearly every
candidate memory +0.25, and the meta path ran 100 semantic results through it.
"""
import pytest

from processing import gate_system
from processing.gate_system import (
    ENTITY_BOOST_MAX_QUERY_ENTITIES, _entity_match_boost, entity_boost_applicable,
)

VOCAB = {f"vocab{i}" for i in range(ENTITY_BOOST_MAX_QUERY_ENTITIES + 1)}


def test_applicable_bounds():
    assert not entity_boost_applicable(set())
    assert entity_boost_applicable({"rowan"})
    assert not entity_boost_applicable(VOCAB)


def test_diluted_set_zero_boost():
    assert _entity_match_boost(VOCAB, "notes on vocab1 and vocab2 today") == 0.0


def test_names_unchanged():
    assert _entity_match_boost({"rowan", "tamsin"}, "Rowan met Tamsin") == 0.25
    assert _entity_match_boost({"rowan", "tamsin"}, "Rowan alone") == 0.18
    assert _entity_match_boost({"rowan"}, "nobody") == 0.0


def _gate_for(query, cos_map, threshold=0.60):
    # Reuses the controllable-cosine harness of test_gate_forced_quality_floor.
    from tests.unit.test_gate_forced_quality_floor import AngleEmbedder, NoRerankMM
    embedder = AngleEmbedder(query, cos_map)
    ms = gate_system.MultiStageGateSystem(
        NoRerankMM(), retrieval_embedder=embedder, retrieval_threshold=threshold)
    ms.gate_system.embedder = embedder
    ms.gate_system.embed_model = embedder
    ms.embed_model = embedder
    return ms


# Nine capitalized, non-initial words: a vocabulary set, not names.
_DILUTED_QUERY = "notes on Alder Birch Cedar Dogwood Elmwood Firth Gorse Hazelton Ivybridge today"
_NAMED_QUERY = "notes on Alder and Birch today"
# blended = 0.85*0.50 + 0.15*0.5 = 0.50 < 0.60: only an entity boost can pass it.
_BELOW_BAR = {"Alder and Birch met at the shed": 0.50}


@pytest.mark.asyncio
async def test_batch_gate_diluted_query_gives_no_boost(monkeypatch):
    monkeypatch.setenv("GATE_MIN_MEMORIES", "0")
    assert len(gate_system._extract_query_entities(_DILUTED_QUERY)) > ENTITY_BOOST_MAX_QUERY_ENTITIES
    ms = _gate_for(_DILUTED_QUERY, _BELOW_BAR)
    mems = [{"content": t, "metadata": {"truth_score": 0.5}} for t in _BELOW_BAR]
    out = await ms.batch_gate_memories(_DILUTED_QUERY, mems)
    assert out == []


@pytest.mark.asyncio
async def test_batch_gate_named_query_still_boosts(monkeypatch):
    monkeypatch.setenv("GATE_MIN_MEMORIES", "0")
    ms = _gate_for(_NAMED_QUERY, _BELOW_BAR)
    mems = [{"content": t, "metadata": {"truth_score": 0.5}} for t in _BELOW_BAR]
    out = await ms.batch_gate_memories(_NAMED_QUERY, mems)
    assert [m["content"] for m in out] == list(_BELOW_BAR)


@pytest.mark.asyncio
async def test_meta_path_diluted_set_skips_semantic_search_and_boost(monkeypatch):
    from memory.memory_retriever import MemoryRetriever

    r = object.__new__(MemoryRetriever)
    calls = []

    async def fake_semantic(q, n_results=30):
        calls.append(n_results)
        return [{"id": "a", "content": "vocab1 vocab2 stuff", "relevance_score": 0.5}]

    r._get_semantic_memories = fake_semantic
    r._get_recent_conversations = lambda k=10: [{"id": "r", "content": "recent"}]
    r.scorer = None
    r.time_manager = None
    monkeypatch.setattr(gate_system, "_extract_query_entities", lambda q: set(VOCAB))

    out = await MemoryRetriever._get_meta_conversational_memories(r, "what did we talk about?", limit=5)
    assert calls == []
    assert all("entity_boost" not in m for m in out)


@pytest.mark.asyncio
async def test_meta_path_small_set_still_searches(monkeypatch):
    from memory.memory_retriever import MemoryRetriever

    r = object.__new__(MemoryRetriever)
    calls = []

    async def fake_semantic(q, n_results=30):
        calls.append(n_results)
        return [{"id": "a", "content": "Rowan said hi", "relevance_score": 0.5}]

    r._get_semantic_memories = fake_semantic
    r._get_recent_conversations = lambda k=10: []
    r.scorer = None
    r.time_manager = None
    monkeypatch.setattr(gate_system, "_extract_query_entities", lambda q: {"rowan"})

    out = await MemoryRetriever._get_meta_conversational_memories(r, "what did we talk about?", limit=5)
    assert calls == [100]
    assert out and out[0]["entity_boost"] == 0.18
