"""WebSearchCache reads run off the event loop (BC-41, incident 2026-10-20).

LoopStall 898 ms at ``WebSearchCache.get`` (a sync Chroma ``collection.get``)
called from the async ``WebSearchManager.search``; the budget-blocked path in the
gatherer read the cache up to three more times inline.  Drives the deployed async
callers; the cache is a recorder of the calling thread.
"""
import threading
from types import SimpleNamespace

import pytest

import knowledge.web_search_manager as wsm
from core.prompt.gatherer_web import WebSearchMixin


class _Cache:
    def __init__(self):
        self.threads = []

    def get(self, query, depth):
        self.threads.append(threading.get_ident())
        return None

    def put(self, result):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["latest rust release", "  \n   latest rust release\n  "])
async def test_search_cache_get_off_loop(query):
    manager = wsm.WebSearchManager(api_key="synthetic")
    cache = _Cache()
    manager.cache = cache
    manager.is_enabled = lambda: True

    class _Limiter:
        def estimate_credits(self, depth):
            return 1.0

        def reserve(self, credits):
            return None  # budget spent: search stops right after the cache read

        def get_remaining_credits(self):
            return 0.0

    manager.rate_limiter = _Limiter()
    result = await manager.search(query, depth=wsm.WebSearchDepth.QUICK)
    assert result.blocked == "budget"
    assert cache.threads and all(t != threading.get_ident() for t in cache.threads)


@pytest.mark.asyncio
async def test_gatherer_budget_blocked_cache_reads_off_loop():
    cache = _Cache()

    class _Manager:
        rate_limiter = None

        def is_available(self):
            return True

        def is_enabled(self):
            return True

    mgr = _Manager()
    mgr.cache = cache

    async def _trigger(**kwargs):
        return SimpleNamespace(
            should_search=False, blocked_reason="budget", depth=None,
            blocked_search_terms=["latest rust release"], source="test",
            reason="budget", confidence=0.9, budget_remaining=0.0)

    host = WebSearchMixin.__new__(WebSearchMixin)
    host.web_search_manager = mgr
    host.web_search_trigger_llm = _trigger
    host.web_search_trigger = None
    host.model_manager = object()
    host.memory_id_map = {}
    await host._get_web_search_results("latest rust release")
    assert cache.threads, "the budget-blocked cache read never happened"
    assert all(t != threading.get_ident() for t in cache.threads)
