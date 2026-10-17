"""HybridRetriever.retrieve runs the query rewrite off the event loop.

2026-10-17, class BC-41 (blocking call on the event loop).

Live loop-stall records: `[LoopStall] event loop blocked 1262 ms at
models/model_manager.py:1106 generate_with_openai (entry:
memory/memory_retriever.py:1272 _get_semantic_memories)`. Chain:
HybridRetriever.retrieve -> rewrite_query -> TopicManager.get_primary_topic
-> _llm_fallback -> model_manager.generate -> generate_with_openai.
Drives THE deployed `retrieve`; only the store seams are stubbed.
"""
import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

import memory.hybrid_retriever as hr


@pytest.mark.asyncio
async def test_rewrite_query_runs_off_the_loop_thread(monkeypatch):
    seen = {}

    def fake_rewrite(q):
        seen["thread"] = threading.get_ident()
        return q

    monkeypatch.setattr(hr, "rewrite_query", fake_rewrite)
    retriever = hr.HybridRetriever(chroma_store=MagicMock())
    retriever._semantic_search = AsyncMock(return_value=[])

    loop_thread = threading.get_ident()
    assert asyncio.get_running_loop() is not None
    result = await retriever.retrieve("what did Rowan say about the trip", limit=5)

    assert result == []
    assert seen["thread"] != loop_thread, "rewrite_query ran on the event loop thread"


@pytest.mark.asyncio
async def test_slow_rewrite_does_not_stall_the_loop(monkeypatch):
    import time

    def slow_rewrite(q):
        time.sleep(0.3)
        return q

    monkeypatch.setattr(hr, "rewrite_query", slow_rewrite)
    retriever = hr.HybridRetriever(chroma_store=MagicMock())
    retriever._semantic_search = AsyncMock(return_value=[])

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    t = asyncio.create_task(ticker())
    await retriever.retrieve("Tamsin schedule", limit=5)
    t.cancel()
    assert ticks >= 5, f"loop only ticked {ticks} times during a 0.3 s rewrite"
