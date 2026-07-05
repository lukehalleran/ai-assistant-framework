"""
Tests for the gate's retrieval-space scoring (embedding-model mismatch fix).

ChromaDB retrieves candidates with bge-small; the cosine gate historically
re-scored them with MiniLM against MiniLM-tuned thresholds. The fix injects the
store's embedder (`retrieval_embedder`) so memory/summary gating scores in the
retrieval space against `gate_rel_threshold_retrieval`, while wiki/semantic
paths keep the gate's own embedder.

These tests use deterministic fake embedders (no model downloads).
"""

import numpy as np
import pytest

from processing.gate_system import (
    CosineSimilarityGateSystem,
    MultiStageGateSystem,
    _EMBED_CACHE,
)


DIM = 8


class FakeEmbedder:
    """Maps texts to fixed unit vectors via a lookup; unknown → default axis."""

    def __init__(self, mapping, default_axis=0):
        self.mapping = mapping
        self.default_axis = default_axis
        self.encoded = []  # record of texts encoded (to assert which model ran)

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False, **kw):
        self.encoded.extend(texts)
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            axis = self.mapping.get(t, self.default_axis)
            out[i, axis] = 1.0
        return out


class NoRerankMM:
    """Mock model_manager whose cross-encoder is unavailable → cosine only."""

    def get_cross_encoder(self, model_name=""):
        raise RuntimeError("no cross encoder in tests")


def _make_multistage(gate_embedder, retrieval_embedder=None, **kw):
    ms = MultiStageGateSystem(NoRerankMM(), retrieval_embedder=retrieval_embedder, **kw)
    # Replace the real (or stub) gate embedder with the deterministic fake
    ms.gate_system.embedder = gate_embedder
    ms.gate_system.embed_model = gate_embedder
    ms.embed_model = gate_embedder
    return ms


def _mem(content, truth=0.5):
    return {"content": content, "metadata": {"truth_score": truth}}


@pytest.fixture(autouse=True)
def _no_forced_minimum(monkeypatch):
    monkeypatch.setenv("GATE_MIN_MEMORIES", "0")


@pytest.mark.asyncio
async def test_memory_gate_uses_retrieval_embedder_and_threshold():
    query = "what is my cat's name"
    # Retrieval space: query aligns with "relevant doc" (axis 1), not "junk" (axis 2)
    retrieval = FakeEmbedder({query: 1, "relevant doc": 1, "junk": 2})
    # Gate space would score EVERYTHING as aligned (axis 0 default) — if the
    # gate used it, junk would pass. It must not be consulted.
    gate = FakeEmbedder({})

    ms = _make_multistage(gate, retrieval_embedder=retrieval, retrieval_threshold=0.60)
    out = await ms.batch_gate_memories(query, [_mem("relevant doc"), _mem("junk")])

    contents = [m["content"] for m in out]
    assert contents == ["relevant doc"]
    # blended = 0.85*1.0 + 0.15*0.5 = 0.925 for the relevant doc
    assert out[0]["relevance_score"] == pytest.approx(0.925)
    assert gate.encoded == []  # gate-space embedder never ran
    assert query in retrieval.encoded


@pytest.mark.asyncio
async def test_memory_gate_without_retrieval_embedder_uses_gate_space():
    query = "q"
    gate = FakeEmbedder({query: 1, "match": 1, "miss": 2})
    ms = _make_multistage(gate, retrieval_embedder=None, cosine_threshold=0.30)
    out = await ms.batch_gate_memories(query, [_mem("match"), _mem("miss")])
    assert [m["content"] for m in out] == ["match"]
    assert query in gate.encoded


@pytest.mark.asyncio
async def test_retrieval_threshold_actually_gates():
    # cos=1.0 but low truth: blended = 0.85 + 0.15*0.0 = 0.85 >= 0.60 → passes;
    # with retrieval_threshold=0.90 it must fail.
    query = "q"
    retrieval = FakeEmbedder({query: 1, "doc": 1})
    ms = _make_multistage(FakeEmbedder({}), retrieval_embedder=retrieval,
                          retrieval_threshold=0.90)
    out = await ms.batch_gate_memories(query, [_mem("doc", truth=0.0)])
    assert out == []


@pytest.mark.asyncio
async def test_embedding_cache_keyed_per_model():
    # Same text encoded in both spaces must not share cached vectors.
    query = "q"
    text = "shared text"
    emb_a = FakeEmbedder({query: 1, text: 1})   # sim(q, text) = 1.0
    emb_b = FakeEmbedder({query: 1, text: 2})   # sim(q, text) = 0.0

    gs_a = CosineSimilarityGateSystem(embedder=emb_a, threshold=0.5)
    gs_b = CosineSimilarityGateSystem(embedder=emb_b, threshold=0.5)

    passed_a = await gs_a.batch_cosine_gate_memories(query, [{"content": text, "metadata": {}}])
    passed_b = await gs_b.batch_cosine_gate_memories(query, [{"content": text, "metadata": {}}])

    assert len(passed_a) == 1 and passed_a[0]["__score__"] == pytest.approx(1.0)
    assert passed_b == []  # would wrongly pass if emb_a's cached vector leaked


@pytest.mark.asyncio
async def test_cosine_filter_summaries_space_defaults():
    query = "q"
    retrieval = FakeEmbedder({query: 1, "on-topic": 1, "off-topic": 2})
    ms = _make_multistage(FakeEmbedder({}), retrieval_embedder=retrieval)
    out = await ms.cosine_filter_summaries(query, [
        {"content": "on-topic"}, {"content": "off-topic"},
    ])
    # retrieval-space default threshold (0.65) — cos 1.0 passes, cos 0.0 fails
    assert [i["content"] for i in out] == ["on-topic"]


def test_intent_override_translates_into_retrieval_space():
    # builder.py applies per-intent MiniLM-space overrides by mutating
    # cosine_threshold; the retrieval path must translate the delta rather
    # than ignore it (or worse, use the MiniLM number directly).
    gs = CosineSimilarityGateSystem(
        embedder=FakeEmbedder({}), threshold=0.18,
        retrieval_embedder=FakeEmbedder({}), retrieval_threshold=0.60,
    )
    assert gs.effective_retrieval_threshold() == pytest.approx(0.60)

    gs.cosine_threshold = 0.45  # stricter intent (e.g. FACTUAL_RECALL)
    expected = 0.60 + (0.45 - 0.18) * CosineSimilarityGateSystem._RETRIEVAL_SPACE_STD_RATIO
    assert gs.effective_retrieval_threshold() == pytest.approx(expected)
    assert gs.effective_retrieval_threshold() > 0.60

    gs.cosine_threshold = 0.10  # looser intent
    assert gs.effective_retrieval_threshold() < 0.60

    gs.cosine_threshold = 0.18  # restored after gather
    assert gs.effective_retrieval_threshold() == pytest.approx(0.60)


def test_store_exposes_st_model():
    from types import SimpleNamespace
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    sentinel = object()
    fake_store = SimpleNamespace(embedding_fn=SimpleNamespace(_model=sentinel))
    assert MultiCollectionChromaStore.get_st_model(fake_store) is sentinel

    no_model = SimpleNamespace(embedding_fn=object())
    assert MultiCollectionChromaStore.get_st_model(no_model) is None


def test_config_constants_exist():
    from config.app_config import GATE_REL_THRESHOLD_RETRIEVAL, GATE_DEICTIC_MIN_RETRIEVAL
    assert 0.0 < GATE_REL_THRESHOLD_RETRIEVAL < 1.0
    assert 0.0 < GATE_DEICTIC_MIN_RETRIEVAL < 1.0
    # bge space sits far above the MiniLM-space threshold
    from config.app_config import GATE_REL_THRESHOLD
    assert GATE_REL_THRESHOLD_RETRIEVAL > GATE_REL_THRESHOLD
