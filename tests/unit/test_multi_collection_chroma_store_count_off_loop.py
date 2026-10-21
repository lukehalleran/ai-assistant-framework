"""Batch query opens/counts collections off the event loop (incident 2026-10-19/20, BC-41).

Live: ``[LoopStall] 304 ms at multi_collection_chroma_store.py:910 query_single_collection
| chromadb ... Collection.py:39 count`` -- the lazy open and ``count()`` ran on the loop.
"""
import threading
from unittest.mock import MagicMock

import pytest

from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore


@pytest.mark.asyncio
async def test_open_and_count_not_on_the_loop_thread():
    seen = []

    class _Coll:
        def count(self):
            seen.append(("count", threading.get_ident()))
            return 0

    store = MultiCollectionChromaStore.__new__(MultiCollectionChromaStore)
    store.collections = {"facts": None}

    def _get(name):
        seen.append(("open", threading.get_ident()))
        return _Coll()

    store._get_collection = _get
    store._cached_embed = MagicMock(return_value=[0.0])
    out = await store.query_multiple_collections(["facts"], "q", 3, query_embedding=[0.0])
    assert out == {"facts": []}
    loop_thread = threading.get_ident()
    assert {k for k, _ in seen} == {"open", "count"}
    assert all(t != loop_thread for _, t in seen)
