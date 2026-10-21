"""Summary reads run off the event loop (incident 2026-10-19/20, BC-41).

Live: ``[LoopStall] 584 ms at multi_collection_chroma_store.py:426 list_all <- get_recent
<- memory_retriever.get_summaries <- gatherer_memory._get_summaries_separate``. The
coordinator's sync ``get_summaries`` was called directly in the coroutine.
"""
import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from core.prompt.gatherer_memory import MemoryRetrievalMixin


class _Coordinator:
    def __init__(self):
        self.threads = []

    def get_summaries(self, n):
        self.threads.append(threading.get_ident())
        return [{"content": "s1", "timestamp": "2026-10-19T10:00:00"}]


def _gatherer(coord):
    g = MemoryRetrievalMixin.__new__(MemoryRetrievalMixin)
    g.memory_coordinator = coord
    g.memory_id_map = {}
    return g


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["_get_summaries_separate", "_get_summaries_separated"])
async def test_coordinator_get_summaries_not_on_the_loop_thread(method):
    coord = _Coordinator()
    g = _gatherer(coord)
    g._get_summaries_hybrid_filtered = MagicMock()
    g._get_recent_conversations = MagicMock()
    loop_thread = threading.get_ident()
    kwargs = {"limit": 2} if method.endswith("separated") else {}
    await getattr(g, method)("", **kwargs)
    assert coord.threads, "get_summaries never called"
    assert all(t != loop_thread for t in coord.threads)


@pytest.mark.asyncio
async def test_corpus_fallback_not_on_the_loop_thread():
    seen = []

    class _Corpus:
        def get_summaries(self, n):
            seen.append(threading.get_ident())
            return []

    class _NoSummaries:
        corpus_manager = _Corpus()

    g = _gatherer(_NoSummaries())
    await g._get_summaries_separate("")
    assert seen and seen[0] != threading.get_ident()
