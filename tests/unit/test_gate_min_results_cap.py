"""
Forced-minimum backfill must respect the caller's budget (2026-08-21).

2026-08-18 audit: "[Batch Gate] Only 0 memories passed, forcing minimum of 8"
fired on a bare "Hey" (casual_social, intent max_mems=3) and 21 memories
rendered. The gate's fail-soft floor (GATE_MIN_MEMORIES=8) existed so
over-aggressive gating can't starve an ordinary turn, but it ignored what the
caller would actually use — below-threshold junk was forced in beyond the
intent budget. filter_memories/batch_gate_memories now accept min_results,
threaded from the gatherer's intent budget via
memory_coordinator.get_memories(min_gated=...) → retriever → gate.

These tests drive THE DEPLOYED batch_gate_memories with deterministic fake
embedders (no model downloads) — never a re-derivation.
"""

import inspect

import numpy as np
import pytest

from processing.gate_system import MultiStageGateSystem

DIM = 8


class FakeEmbedder:
    """Maps texts to fixed unit vectors via a lookup; unknown → default axis."""

    def __init__(self, mapping, default_axis=0):
        self.mapping = mapping
        self.default_axis = default_axis

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False, **kw):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            axis = self.mapping.get(t, self.default_axis)
            out[i, axis] = 1.0
        return out


class NoRerankMM:
    def get_cross_encoder(self, model_name=""):
        raise RuntimeError("no cross encoder in tests")


def _make_gate(query, threshold=0.60):
    # Query on axis 1; every memory on axis 2 → cosine 0 → nothing passes,
    # so the forced-minimum backfill is the ONLY source of output.
    embedder = FakeEmbedder({query: 1}, default_axis=2)
    ms = MultiStageGateSystem(NoRerankMM(), retrieval_embedder=embedder,
                              retrieval_threshold=threshold)
    ms.gate_system.embedder = embedder
    ms.gate_system.embed_model = embedder
    ms.embed_model = embedder
    return ms


def _mems(n):
    return [{"content": f"below-threshold memory {i}", "metadata": {"truth_score": 0.5}}
            for i in range(n)]


@pytest.fixture(autouse=True)
def _default_floor(monkeypatch):
    monkeypatch.setenv("GATE_MIN_MEMORIES", "8")
    # These tests pin the COUNT-cap contract with cosine-0 memories; the
    # 2026-08-23 quality floor would (correctly) refuse them all. Disable it
    # here — floor semantics live in test_gate_forced_quality_floor.py.
    monkeypatch.setenv("GATE_FORCED_FLOOR_MARGIN", "1.0")


@pytest.mark.asyncio
async def test_forced_minimum_default_is_eight():
    q = "hey"
    ms = _make_gate(q)
    out = await ms.batch_gate_memories(q, _mems(12))
    assert len(out) == 8
    assert all(m.get("forced_minimum") for m in out)


@pytest.mark.asyncio
async def test_min_results_caps_forced_minimum():
    q = "hey"
    ms = _make_gate(q)
    out = await ms.batch_gate_memories(q, _mems(12), min_results=3)
    assert len(out) == 3  # casual_social budget, not the global floor of 8


@pytest.mark.asyncio
async def test_min_results_zero_disables_backfill():
    q = "hey"
    ms = _make_gate(q)
    out = await ms.batch_gate_memories(q, _mems(12), min_results=0)
    assert out == []


@pytest.mark.asyncio
async def test_min_results_never_raises_floor():
    q = "hey"
    ms = _make_gate(q)
    out = await ms.batch_gate_memories(q, _mems(30), min_results=20)
    assert len(out) == 8  # min(env floor 8, caller 20) — cap only lowers


@pytest.mark.asyncio
async def test_filter_memories_passes_min_results_through():
    q = "hey"
    ms = _make_gate(q)
    out = await ms.filter_memories(q, _mems(12), min_results=2)
    assert len(out) == 2


@pytest.mark.asyncio
async def test_genuinely_passing_memories_unaffected_by_cap():
    q = "what did we discuss about the gym"
    embedder = FakeEmbedder({q: 1, "gym plan discussion": 1}, default_axis=2)
    ms = MultiStageGateSystem(NoRerankMM(), retrieval_embedder=embedder,
                              retrieval_threshold=0.60)
    ms.gate_system.embedder = embedder
    ms.gate_system.embed_model = embedder
    ms.embed_model = embedder
    mems = [{"content": "gym plan discussion", "metadata": {"truth_score": 0.9}}] + _mems(5)
    out = await ms.batch_gate_memories(q, mems, min_results=1)
    # The passing memory survives on merit; min_results caps only the BACKFILL.
    assert any(m["content"] == "gym plan discussion" for m in out)
    assert not any(m.get("forced_minimum") for m in out
                   if m["content"] == "gym plan discussion")


class TestChainWiring:
    """The budget must actually flow gatherer → coordinator → retriever → gate."""

    def test_gatherer_passes_intent_budget(self):
        import core.prompt.gatherer_memory as gm
        src = inspect.getsource(gm.MemoryRetrievalMixin._get_semantic_memories)
        assert "min_gated=limit" in src

    def test_coordinator_delegates_min_gated(self):
        from memory.memory_coordinator import MemoryCoordinator
        src = inspect.getsource(MemoryCoordinator.get_memories)
        assert "min_gated=min_gated" in src

    def test_retriever_threads_to_gate(self):
        from memory.memory_retriever import MemoryRetriever
        assert "min_gated" in inspect.signature(MemoryRetriever.get_memories).parameters
        src = inspect.getsource(MemoryRetriever._gate_memories)
        assert "min_results=min_results" in src
