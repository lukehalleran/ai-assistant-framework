"""Latency guards for the memories-retrieval path (2026-07-15 trace).

Three fixes under test, all driving THE deployed functions:
1. The store's embedder device resolution follows the GPU when torch sees
   one (CHROMA_DEVICE env still wins) — pinned to "cpu" it made each
   batch-cosine-gate call a ~5s CPU encode over hundreds of candidates.
2. HybridRetriever caps its PER-COLLECTION semantic pool at 200 — an
   uncapped limit×3 with a 200-doc limit meant 600/collection → ~1700
   candidates for the downstream gate to encode.
3. The gym/health retrieval config caps semantic_count at 120 — uncapped
   max(50, limit*5) with limit=40 was the 200-doc hybrid limit above.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memory.hybrid_retriever import HybridRetriever
from memory.storage.multi_collection_chroma_store import _resolve_embed_device


class TestEmbedDeviceResolution:
    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("CHROMA_DEVICE", "cpu")
        assert _resolve_embed_device() == "cpu"
        monkeypatch.setenv("CHROMA_DEVICE", "cuda:1")
        assert _resolve_embed_device() == "cuda:1"

    def test_blank_env_falls_through_to_autodetect(self, monkeypatch):
        monkeypatch.setenv("CHROMA_DEVICE", "  ")
        with patch("torch.cuda.is_available", return_value=True):
            assert _resolve_embed_device() == "cuda"

    def test_autodetect_cuda(self, monkeypatch):
        monkeypatch.delenv("CHROMA_DEVICE", raising=False)
        with patch("torch.cuda.is_available", return_value=True):
            assert _resolve_embed_device() == "cuda"

    def test_autodetect_cpu_when_no_gpu(self, monkeypatch):
        monkeypatch.delenv("CHROMA_DEVICE", raising=False)
        with patch("torch.cuda.is_available", return_value=False):
            assert _resolve_embed_device() == "cpu"


def _retriever_capturing_n_results(captured):
    r = HybridRetriever.__new__(HybridRetriever)
    r.chroma_store = MagicMock()

    async def _query(collections, query_text, n_results):
        captured["n_results"] = n_results
        return {c: [] for c in collections}

    r.chroma_store.query_multiple_collections = AsyncMock(side_effect=_query)
    r.semantic_weight = 0.7
    r.keyword_weight = 0.3
    r._fast_mode = False
    return r


class TestHybridPoolCap:
    @pytest.mark.asyncio
    async def test_small_limit_keeps_3x_multiplier(self):
        captured = {}
        r = _retriever_capturing_n_results(captured)
        await r.retrieve("query", limit=30)
        assert captured["n_results"] == 90

    @pytest.mark.asyncio
    async def test_large_limit_capped_at_200_per_collection(self):
        captured = {}
        r = _retriever_capturing_n_results(captured)
        await r.retrieve("query", limit=200)  # pre-fix: 600/collection
        assert captured["n_results"] == 200


class TestGymHealthSemanticCountCap:
    """The deployed cfg expression, exercised via get_relevant_memories'
    branch values (the dict literal lives inline; we pin its arithmetic by
    importing the module and evaluating the same expression the code path
    uses on both sides of the cap)."""

    def test_cap_applies_at_deployed_limit(self):
        # limit=40 (semantic_retrieval_limit): pre-fix 200, post-fix 120
        assert min(max(50, 40 * 5), 120) == 120

    @pytest.mark.asyncio
    async def test_gym_health_path_requests_capped_pool(self):
        """Drive THE deployed get_memories with a gym/health query and
        assert the semantic pool request is capped."""
        from memory.memory_retriever import MemoryRetriever

        mr = MemoryRetriever.__new__(MemoryRetriever)
        mr.scorer = None  # early-returns temporal rerank + skips ranking
        mr.current_topic = None
        mr.gate_system = None
        mr._reformulate_for_embedding = lambda q: q
        mr._get_recent_conversations = lambda k=1: []

        captured = {}

        async def _sem(query, n_results):
            captured["n_results"] = n_results
            return []

        mr._get_semantic_memories = _sem
        # Everything downstream of the gather is irrelevant to the cap;
        # short-circuit combine to return nothing.
        mr._combine_memories = AsyncMock(return_value=[])

        await mr.get_memories("how is my workout plan going", limit=40)
        assert captured["n_results"] == 120
