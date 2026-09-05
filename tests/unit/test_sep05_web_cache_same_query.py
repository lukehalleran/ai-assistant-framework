# tests/unit/test_sep05_web_cache_same_query.py
"""
Regression tests for the 2026-09-05 same-query-different-depth cache reuse fix.

Live defect (2026-09-05 13:07, daemon_debug.log): the enhanced-path gatherer
searched "Were any UK or US politicians charged with crimes this week? What
did the police or courts announce?" at QUICK depth (1 credit, cached at
13:07:27), and the agentic loop's Round-1 seeded search re-ran the IDENTICAL
string at STANDARD depth 11 seconds later (2 more credits, "Credits used:
2.0, total today: 3.0/100"). Cause: WebSearchCache._generate_cache_key is
f"{query.lower().strip()}:{depth.value}", so the second lookup missed the
exact-key cache. Within one turn, a same-string re-search at a different
depth is waste, not a quality upgrade.

WebSearchCache now keeps a small in-process `_recent_by_query` map (see
WEB_SEARCH_SAME_QUERY_REUSE_S in knowledge/web_search_manager.py) that lets a
same-normalized-query lookup at a DIFFERENT depth reuse the recent result
within a short window, without touching the exact-key Chroma cache's
TTL/persistence behavior.
"""

import knowledge.web_search_manager as wsm
from knowledge.web_search_manager import (
    WebPage,
    WebSearchCache,
    WebSearchDepth,
    WebSearchResult,
)


def _make_cache() -> WebSearchCache:
    """
    A WebSearchCache with no live Chroma collection.

    Mirrors tests/test_web_search_manager.py::TestWebSearchCache.test_cache_miss:
    forcing `_initialized = True` / `_collection = None` makes `_ensure_initialized()`
    return False without touching a real store, exercising the exact same
    store-unavailable path a Chroma-less deployment hits in production.
    """
    cache = WebSearchCache()
    cache._initialized = True
    cache._collection = None
    return cache


def _make_result(query: str, depth: WebSearchDepth, pages=None) -> WebSearchResult:
    if pages is None:
        pages = [
            WebPage(
                url="https://example.com/politics-charges",
                title="Politicians charged this week",
                content="Some content about charges.",
                snippet="Some content about charges.",
            )
        ]
    return WebSearchResult(
        query=query,
        pages=pages,
        total_credits_used=1.0,
        search_depth=depth,
    )


class TestSameQueryReuse:
    def test_same_query_different_depth_reuses_result(self):
        cache = _make_cache()
        query = (
            "Were any UK or US politicians charged with crimes this week? "
            "What did the police or courts announce?"
        )
        original = _make_result(query, WebSearchDepth.QUICK)
        cache.put(original)

        hit = cache.get(query, WebSearchDepth.STANDARD)

        assert hit is not None
        assert hit.from_cache is True
        assert [p.url for p in hit.pages] == [p.url for p in original.pages]

        # The originally stored object must not be mutated by the reuse copy.
        assert original.from_cache is False

    def test_different_query_is_not_reused(self):
        cache = _make_cache()
        cache.put(_make_result("first query about politics", WebSearchDepth.QUICK))

        assert cache.get("a completely different query", WebSearchDepth.STANDARD) is None

    def test_expired_entry_is_dropped_from_map(self, monkeypatch):
        monkeypatch.setattr(wsm, "WEB_SEARCH_SAME_QUERY_REUSE_S", 1)

        cache = _make_cache()
        query = "expiry test query"
        cache.put(_make_result(query, WebSearchDepth.QUICK))

        norm = cache._norm(query)
        ts, stored = cache._recent_by_query[norm]
        # Backdate the stored entry so it is 5s old against a 1s reuse window.
        cache._recent_by_query[norm] = (ts - 5, stored)

        hit = cache.get(query, WebSearchDepth.STANDARD)

        assert hit is None
        assert norm not in cache._recent_by_query

    def test_normalization_ignores_case_and_whitespace(self):
        cache = _make_cache()
        cache.put(
            _make_result("  Were Any UK Politicians Charged?  ", WebSearchDepth.QUICK)
        )

        hit = cache.get("were any uk politicians charged?", WebSearchDepth.STANDARD)

        assert hit is not None
        assert hit.from_cache is True

    def test_disabled_via_zero_constant(self, monkeypatch):
        monkeypatch.setattr(wsm, "WEB_SEARCH_SAME_QUERY_REUSE_S", 0)

        cache = _make_cache()
        query = "disabled reuse query"
        cache.put(_make_result(query, WebSearchDepth.QUICK))

        assert cache.get(query, WebSearchDepth.STANDARD) is None

    def test_put_without_results_records_nothing(self):
        cache = _make_cache()
        empty_result = WebSearchResult(
            query="no results query",
            pages=[],
            search_depth=WebSearchDepth.QUICK,
        )

        cache.put(empty_result)

        assert cache._recent_by_query == {}
        assert cache.get("no results query", WebSearchDepth.STANDARD) is None

    def test_recent_map_is_bounded_and_keeps_newest(self):
        cache = _make_cache()
        for i in range(70):
            cache.put(_make_result(f"distinct query number {i}", WebSearchDepth.QUICK))

        assert len(cache._recent_by_query) <= 64
        newest_norm = cache._norm("distinct query number 69")
        assert newest_norm in cache._recent_by_query

    def test_storeless_cache_still_returns_in_memory_hit(self):
        """
        This is the live shape when Chroma is unavailable: `_ensure_initialized()`
        returns False, so `get()` must fall back to the in-process map rather than
        unconditionally returning None.
        """
        cache = _make_cache()
        query = "storeless cache query"
        cache.put(_make_result(query, WebSearchDepth.QUICK))

        hit = cache.get(query, WebSearchDepth.QUICK)

        assert hit is not None
        assert hit.from_cache is True
