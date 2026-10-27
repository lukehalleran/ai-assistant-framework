"""2026-10-12: "news today" pre-gather returned Jan-2026 Reuters items, evergreen
explainers and homepage navigation junk: the five facet queries ran as
topic="general" with no recency window (a facet like "world affairs" has no news
keyword, so `_is_news_query` was False) and nothing filtered the results.
2026-10-26 (lane R2): broad-news facets are searched with Tavily's own
topic="news" + days window, and news-intent results older than NEWS_MAX_AGE_DAYS or
shaped like a homepage/section index are dropped (URL shape, no site list).

Incident date: 2026-10-12. class: BC-25.
Drives the DEPLOYED WebSearchManager.multi_search / search with a fake Tavily
client (no network, no LLM; the facet generator is replaced by a fixed list).
Result dicts are invented.
"""
from __future__ import annotations

import asyncio
import weakref
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

import knowledge.web_search_manager as wsm
from knowledge.web_search_manager import (
    NEWS_FACET_FRESHNESS_DAYS, NEWS_MAX_AGE_DAYS, WebPage, WebSearchDepth,
    filter_news_pages,
)

from tests.unit.test_web_search_manager_budget_reservation import _make_manager

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
FRESH = format_datetime(NOW - timedelta(hours=5))
OLD = format_datetime(datetime(2026, 1, 14, 9, 0, tzinfo=timezone.utc))


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(wsm, "_LIVE_RATE_LIMITERS", weakref.WeakSet())


class _Client:
    """Fake Tavily client: records every search kwargs, returns canned rows."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def search(self, **kw):
        self.calls.append(kw)
        return {"results": list(self.rows)}

    def extract(self, **kw):
        return {"results": []}


def _row(url, published=None, title="T"):
    r = {"url": url, "title": title, "content": "body " * 20, "score": 0.5}
    if published is not None:
        r["published_date"] = published
    return r


ROWS = [
    _row("https://wire.example.test/world/2026/10/26/talks-resume-after-ceasefire", FRESH),
    _row("https://wire.example.test/world/2026/01/14/old-border-story", OLD),
    _row("https://wire.example.test/", FRESH),                    # homepage
    _row("https://wire.example.test/world", FRESH),               # section index (depth 1)
    _row("https://wire.example.test/world/index.html", FRESH),    # section index file
    _row("https://explain.example.test/guides/how-sanctions-work", OLD),  # evergreen, old date
    _row("https://wire.example.test/talks-resume-after-ceasefire-in-border-region", FRESH),  # lone article slug
]
KEPT = {
    "https://wire.example.test/world/2026/10/26/talks-resume-after-ceasefire",
    "https://wire.example.test/talks-resume-after-ceasefire-in-border-region",
}


def _manager(tmp_path, monkeypatch, client):
    manager, _ = _make_manager(tmp_path, 50, monkeypatch)
    manager._tavily_client = client
    monkeypatch.setattr(manager, "_localize_query", lambda q: q, raising=False)
    monkeypatch.setattr(manager, "location_unknown_for", lambda qs: False, raising=False)
    return manager


async def _fake_facets(self, query):
    return ["Reuters AP geopolitics world affairs", "markets economy outlook Reuters CNBC"]


class TestFacetQueriesGetProviderRecency:
    def test_facets_are_searched_as_news_with_a_days_window(self, tmp_path, monkeypatch):
        client = _Client([])
        manager = _manager(tmp_path, monkeypatch, client)
        monkeypatch.setattr(wsm.WebSearchManager, "_decompose_news_query", _fake_facets)
        asyncio.run(manager.multi_search("what's in the news today?", depth=WebSearchDepth.QUICK))
        assert len(client.calls) == 2
        for kw in client.calls:
            assert kw["topic"] == "news" and kw["days"] == NEWS_FACET_FRESHNESS_DAYS
        # the facets alone carry no news keyword: this is what the base tree sent as "general"
        assert not any(wsm._is_news_query(q) for q in ("Reuters AP geopolitics world affairs",))

    def test_non_news_search_unchanged(self, tmp_path, monkeypatch):
        client = _Client([])
        manager = _manager(tmp_path, monkeypatch, client)
        asyncio.run(manager.search("python asyncio wait_for documentation", depth=WebSearchDepth.QUICK, localize=False))
        assert client.calls[0]["topic"] == "general" and "days" not in client.calls[0]

    def test_keyword_news_query_keeps_its_one_day_window(self, tmp_path, monkeypatch):
        client = _Client([])
        manager = _manager(tmp_path, monkeypatch, client)
        asyncio.run(manager.search("latest news today", depth=WebSearchDepth.QUICK, localize=False))
        assert client.calls[0]["topic"] == "news" and client.calls[0]["days"] == 1


class TestNewsResultFiltering:
    def test_facet_pre_gather_drops_stale_evergreen_and_index_pages(self, tmp_path, monkeypatch):
        client = _Client(ROWS)
        manager = _manager(tmp_path, monkeypatch, client)
        monkeypatch.setattr(wsm.WebSearchManager, "_decompose_news_query", _fake_facets)
        # the provider rows' "now" must be real for the age rule: rebuild dates relative to today
        real_now = datetime.now(timezone.utc)
        rows = [dict(r) for r in ROWS]
        for r in rows:
            if r.get("published_date") == FRESH:
                r["published_date"] = format_datetime(real_now - timedelta(hours=5))
        rows[0]["url"] = f"https://wire.example.test/world/{real_now:%Y/%m/%d}/talks-resume-after-ceasefire"
        client.rows = rows
        result = asyncio.run(manager.multi_search("what's in the news today?", depth=WebSearchDepth.QUICK))
        urls = {p.url for p in result.pages}
        assert urls == {rows[0]["url"], "https://wire.example.test/talks-resume-after-ceasefire-in-border-region"}

    def test_non_news_query_results_are_not_filtered(self, tmp_path, monkeypatch):
        client = _Client([_row("https://wire.example.test/", OLD), _row("https://docs.example.test/guide/page", OLD)])
        manager = _manager(tmp_path, monkeypatch, client)
        result = asyncio.run(manager.search("python asyncio wait_for documentation", depth=WebSearchDepth.QUICK, localize=False))
        assert len(result.pages) == 2


class TestFilterRules:
    def _pages(self, rows):
        return [WebPage(url=r["url"], title="T", content="c", published_date=r.get("published_date")) for r in rows]

    def test_pinned_clock_verdicts(self):
        kept = {p.url for p in filter_news_pages(self._pages(ROWS), now=NOW)}
        assert kept == KEPT

    def test_undated_article_is_kept(self):
        page = WebPage(url="https://wire.example.test/world/politics/summit-opens", title="T", content="c")
        assert filter_news_pages([page], now=NOW) == [page]

    def test_age_boundary_uses_the_documented_window(self):
        edge_in = format_datetime(NOW - timedelta(days=NEWS_MAX_AGE_DAYS - 1))
        edge_out = format_datetime(NOW - timedelta(days=NEWS_MAX_AGE_DAYS + 1))
        pages = [WebPage(url="https://a.example.test/n/one/story", title="T", content="c", published_date=edge_in),
                 WebPage(url="https://a.example.test/n/two/story", title="T", content="c", published_date=edge_out)]
        assert [p.url for p in filter_news_pages(pages, now=NOW)] == ["https://a.example.test/n/one/story"]

    def test_iso_and_garbage_dates(self):
        pages = [WebPage(url="https://a.example.test/n/one/story", title="T", content="c",
                         published_date="2026-01-02T03:04:05Z"),
                 WebPage(url="https://a.example.test/n/two/story", title="T", content="c", published_date="not a date")]
        assert [p.url for p in filter_news_pages(pages, now=NOW)] == ["https://a.example.test/n/two/story"]

    def test_story_id_single_segment_is_an_article(self):
        page = WebPage(url="https://a.example.test/20481733", title="T", content="c")
        assert filter_news_pages([page], now=NOW) == [page]

    def test_bare_numeric_segment_is_a_story_id(self):
        page = WebPage(url="https://a.example.test/1", title="T", content="c")
        assert filter_news_pages([page], now=NOW) == [page]
