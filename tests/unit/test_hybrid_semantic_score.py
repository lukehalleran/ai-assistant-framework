"""Regression: hybrid retriever must consume the store's relevance_score.

The store wrapper (multi_collection_chroma_store.query_collection) emits
{id, content, metadata, relevance_score, collection, rank} — no "distance"
key. hybrid_retriever read item.get("distance", 1.0), so semantic_score was
0.0 for EVERY document and hybrid ranking silently degraded to keyword-only
(the 5th silently-defaulted-key incident; see CLAUDE.md critical rule #3).

These tests drive THE deployed HybridRetriever methods with store-shaped
result dicts — not a re-derivation of the scoring.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from memory.hybrid_retriever import HybridRetriever


def _store_result(doc_id, content, relevance):
    """Shape emitted by MultiCollectionChromaStore.query_collection."""
    return {
        "id": doc_id,
        "content": content,
        "metadata": {"timestamp": "2026-07-14T12:00:00"},
        "relevance_score": relevance,
        "collection": "conversations",
        "rank": 1,
    }


def _retriever_with(results):
    r = HybridRetriever.__new__(HybridRetriever)
    r.chroma_store = MagicMock()
    r.chroma_store.query_multiple_collections = AsyncMock(
        return_value={"conversations": results}
    )
    r.semantic_weight = 0.7
    r.keyword_weight = 0.3
    r._fast_mode = False
    return r


class TestSemanticScoreWiring:
    @pytest.mark.asyncio
    async def test_semantic_score_comes_from_relevance_score(self):
        r = _retriever_with([_store_result("a", "some text", 0.83)])
        out = await r._semantic_search("query", n_results=10)
        assert out[0]["semantic_score"] == pytest.approx(0.83)

    @pytest.mark.asyncio
    async def test_semantic_score_not_zeroed_across_pool(self):
        """The bug signature: every candidate scoring exactly 0.0."""
        results = [_store_result(f"d{i}", f"text {i}", 0.2 + i * 0.1) for i in range(5)]
        r = _retriever_with(results)
        out = await r._semantic_search("query", n_results=10)
        assert all(m["semantic_score"] > 0.0 for m in out)

    @pytest.mark.asyncio
    async def test_distance_fallback_still_works(self):
        """Callers that DO provide a raw distance keep working."""
        item = _store_result("a", "text", None)
        del item["relevance_score"]
        item["distance"] = 0.25
        r = _retriever_with([item])
        out = await r._semantic_search("query", n_results=10)
        assert out[0]["semantic_score"] == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_retrieve_ranks_by_semantic_when_keywords_tie(self):
        """End-to-end through the deployed retrieve(): with no keyword overlap,
        ranking must follow the embedding relevance, not collapse to zero."""
        results = [
            _store_result("weak", "entirely unrelated words here", 0.10),
            _store_result("strong", "entirely unrelated words here", 0.90),
        ]
        r = _retriever_with(results)
        out = await r.retrieve("zzz qqq", limit=2)
        assert out[0]["id"] == "strong"
        assert out[0]["hybrid_score"] > out[1]["hybrid_score"] > 0.0
