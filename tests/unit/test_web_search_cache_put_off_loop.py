"""WebSearchManager.search writes the cache off the event loop.

2026-10-17, class BC-41.  Live: 25 loop stalls (max 2771 ms, sum 7431 ms) at
WebSearchCache.put, a Chroma upsert.  Drives THE deployed `search`; only the
Tavily execution and the cache are faked.  The write must still happen.
"""
import threading

import pytest

import knowledge.web_search_manager as wsm
from knowledge.web_search_manager import WebPage, WebSearchDepth, WebSearchResult


class _FakeCache:
    def __init__(self):
        self.put_thread = None
        self.put_calls = []

    def get(self, query, depth):
        return None

    def put(self, result):
        self.put_thread = threading.get_ident()
        self.put_calls.append(result)


@pytest.mark.asyncio
async def test_search_puts_cache_on_worker_thread(monkeypatch):
    cache = _FakeCache()
    mgr = wsm.WebSearchManager(api_key="fake", cache=cache)
    monkeypatch.setattr(mgr, "_ensure_tavily", lambda: True)
    result = WebSearchResult(
        query="quasar telescope schedule",
        pages=[WebPage(url="https://example.com/q", title="Quasar", content="c", snippet="s")],
        total_credits_used=1.0,
        search_depth=WebSearchDepth.QUICK,
    )

    async def fake_exec(*a, **k):
        return result

    monkeypatch.setattr(mgr, "_execute_search", fake_exec)
    loop_thread = threading.get_ident()
    out = await mgr.search("quasar telescope schedule", depth=WebSearchDepth.QUICK)
    assert out.has_results
    assert cache.put_calls == [result], "cache write was dropped"
    assert cache.put_thread != loop_thread
