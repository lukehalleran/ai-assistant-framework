"""HybridRetriever.retrieve runs keyword match + hybrid scoring off the loop.

2026-10-17, class BC-41.  Live: 19 stalls (max 1189 ms, sum 12026 ms) in
utils/keyword_matcher.calculate_keyword_score via `_keyword_match`.  Drives
THE deployed `retrieve`; the semantic search is stubbed.
"""
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

import memory.hybrid_retriever as hr


@pytest.mark.asyncio
async def test_keyword_match_and_hybrid_score_share_one_worker_thread(monkeypatch):
    seen = {}
    retriever = hr.HybridRetriever(chroma_store=MagicMock())
    monkeypatch.setattr(hr, "rewrite_query", lambda q: q)
    retriever._semantic_search = AsyncMock(return_value=[{"content": "Rowan trip", "metadata": {}}])

    def fake_kw(query, sem):
        seen["kw"] = threading.get_ident()
        return []

    def fake_hybrid(sem, kw, query):
        seen["hy"] = threading.get_ident()
        return sem

    retriever._keyword_match = fake_kw
    retriever._hybrid_score = fake_hybrid
    loop_thread = threading.get_ident()
    out = await retriever.retrieve("Rowan trip plans", limit=5)
    assert out == [{"content": "Rowan trip", "metadata": {}}]
    assert seen["kw"] != loop_thread
    assert seen["hy"] != loop_thread
    assert seen["kw"] == seen["hy"], "expected ONE worker-thread call"
