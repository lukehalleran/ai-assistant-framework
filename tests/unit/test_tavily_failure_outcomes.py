"""CGR-20260913-008 (#110, #111): the Tavily search/extract wrappers stop
collapsing a provider failure into a plain `[]`.

F2 adopts the F1 leaf `utils/retrieval_outcome.py` at
`knowledge/web_search_manager.py`'s two Tavily wrappers:

- `_tavily_search`/`_tavily_extract` raise `RetrievalError` (instead of
  returning `[]`) on a no-client or provider-exception path. `_tavily_extract`
  still returns `[]` for a genuine empty `urls` argument — that is not a
  failure.
- `search()` (via `_execute_search`) converts a Step-1 `RetrievalError` into
  today's exact `WebSearchResult.error` text/credit shape — including the
  unchanged invalid-key branch.
- A Step-2/3 extract failure (STANDARD/DEEP) KEEPS the already-fetched search
  pages and records the new `WebSearchResult.extract_error` field instead of
  `error`, so `has_results`/caching still treat it as a partial success.
- `fetch_url_content` converts an extract failure into
  `OutcomeList.failed(...)` (still falsy, still no `blocked`) instead of a
  bare `[]`; the budget-refusal `FetchedPages(blocked="budget")` path is
  unchanged.

Every case drives the DEPLOYED wrapper/method. Only the fake Tavily provider
client (`.search`/`.extract`) is faked; no network, no LLM, no `pip install`.
Every fake exception message carries a MARKER substring, and every assertion
on `reason`/`str()`/`error`/`extract_error` checks that marker is absent —
the privacy contract (no query/URL/provider text in those fields).

class: BC-20, BC-47, CM-05
"""

from __future__ import annotations

import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import knowledge.web_search_manager as wsm
from knowledge.web_search_manager import WebSearchDepth, WebSearchManager
from utils.retrieval_outcome import OutcomeList, RetrievalError, outcome_status

from tests.unit.test_sep12_search_budget_reservation import _make_manager, _page
from tests.unit.test_sep12_web_evidence_budget import _GathererHost


# Fresh live-limiter registry per test (identical fixture to
# tests/unit/test_sep12_search_budget_reservation.py; never imported across
# modules, only re-declared, per that file's own precedent).
@pytest.fixture(autouse=True)
def _isolated_rate_limiter_registry(monkeypatch):
    monkeypatch.setattr(wsm, "_LIVE_RATE_LIMITERS", weakref.WeakSet())


# Fake Tavily provider clients.
class _RaisingClient:
    """A fake Tavily client whose .search()/.extract() always raise `exc`."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.search_calls = 0
        self.extract_calls = 0

    def search(self, **kwargs):
        self.search_calls += 1
        raise self._exc

    def extract(self, **kwargs):
        self.extract_calls += 1
        raise self._exc


class _OkClient:
    """A fake Tavily client returning a genuine zero-result response."""

    def search(self, **kwargs):
        return {"results": []}

    def extract(self, **kwargs):
        return {"results": []}


class _MixedClient:
    """search() succeeds with real pages; extract() either raises or
    succeeds, independently, to drive the STANDARD-depth partial-failure
    path without touching the search half."""

    def __init__(self, search_results, *, extract_exc=None, extract_results=None):
        self._search_results = search_results
        self._extract_exc = extract_exc
        self._extract_results = extract_results if extract_results is not None else []

    def search(self, **kwargs):
        return {"results": self._search_results}

    def extract(self, **kwargs):
        if self._extract_exc is not None:
            raise self._extract_exc
        return {"results": self._extract_results}


def _bare_manager(client):
    """A WebSearchManager with only the attributes `_tavily_search` /
    `_tavily_extract` read directly — no rate limiter, no cache. Same
    construction as `_bare_manager` in tests/unit/test_url_fetch_layers.py."""
    mgr = object.__new__(WebSearchManager)
    mgr.max_content_chars = 10000
    mgr._tavily_client = client
    mgr._api_key_invalid = False
    return mgr


QUERY_TEXT = "PRIVACY_QUERY_MARKER weather forecast for tomorrow"
URL_TEXT = "https://example.test/PRIVACY_URL_MARKER"


# Wrapper level: _tavily_search and _tavily_extract share the same no-client
# / provider-exception / invalid-key / empty-result contract — parametrized
# over both wrappers so each assertion is written once.
WRAPPER_CASES = [
    ("tavily_search", lambda mgr: mgr._tavily_search(QUERY_TEXT, 5)),
    ("tavily_extract", lambda mgr: mgr._tavily_extract([URL_TEXT])),
]


class TestTavilyWrappers:

    @pytest.mark.parametrize("source,call", WRAPPER_CASES)
    async def test_provider_exception_raises_retrieval_error(self, source, call):
        mgr = _bare_manager(_RaisingClient(RuntimeError("PROVIDER_MESSAGE_MARKER")))
        with pytest.raises(RetrievalError) as exc_info:
            await call(mgr)
        err = exc_info.value
        assert err.source == source
        assert err.reason == "RuntimeError"
        assert "PROVIDER_MESSAGE_MARKER" not in str(err)
        assert QUERY_TEXT not in str(err) and URL_TEXT not in str(err)
        assert mgr._api_key_invalid is False

    @pytest.mark.parametrize("source,call", WRAPPER_CASES)
    async def test_invalid_key_shaped_exception(self, source, call):
        mgr = _bare_manager(_RaisingClient(RuntimeError("401 Unauthorized")))
        with pytest.raises(RetrievalError) as exc_info:
            await call(mgr)
        assert exc_info.value.source == source
        assert exc_info.value.reason == "invalid_api_key"
        assert mgr._api_key_invalid is True

    @pytest.mark.parametrize("source,call", WRAPPER_CASES)
    async def test_no_client_raises_client_unavailable(self, source, call):
        mgr = _bare_manager(None)
        with pytest.raises(RetrievalError) as exc_info:
            await call(mgr)
        assert exc_info.value.source == source
        assert exc_info.value.reason == "client_unavailable"

    @pytest.mark.parametrize("source,call", WRAPPER_CASES)
    async def test_control_genuine_empty_result_no_raise(self, source, call):
        mgr = _bare_manager(_OkClient())
        assert await call(mgr) == []

    async def test_control_empty_urls_no_raise_even_without_client(self):
        """Extract-only: empty `urls` is not a failure, even with no client."""
        mgr = _bare_manager(None)
        assert await mgr._tavily_extract([]) == []


# search() end to end.
class TestSearchEndToEnd:

    async def test_provider_exception_becomes_error_not_cached_reservation_settled(
            self, tmp_path, monkeypatch):
        client = _RaisingClient(RuntimeError("PROVIDER_MESSAGE_MARKER"))
        manager, limiter = _make_manager(tmp_path, 5, monkeypatch)
        manager._tavily_client = client
        puts = []
        manager.cache = SimpleNamespace(get=lambda q, d: None, put=lambda r: puts.append(r))

        result1 = await manager.search(
            "unique provider exception query", depth=WebSearchDepth.QUICK, localize=False)

        assert result1.error == "Web search provider failed (RuntimeError)"
        assert result1.has_results is False
        assert "PROVIDER_MESSAGE_MARKER" not in result1.error
        assert puts == [], "a failed result must never be cached"
        assert limiter._reserved_today == 0.0, "the reservation is always settled"
        assert client.search_calls == 1

        result2 = await manager.search(
            "unique provider exception query", depth=WebSearchDepth.QUICK, localize=False)
        assert client.search_calls == 2, "not cached -> the provider is reached again"

    async def test_invalid_key_error_text_unchanged(self, tmp_path, monkeypatch):
        client = _RaisingClient(RuntimeError("401 Unauthorized"))
        manager, limiter = _make_manager(tmp_path, 5, monkeypatch)
        manager._tavily_client = client

        result = await manager.search(
            "unique invalid key query", depth=WebSearchDepth.QUICK, localize=False)

        assert result.error == "Tavily API key is invalid"
        assert manager._api_key_invalid is True
        assert limiter._reserved_today == 0.0

    async def test_control_genuine_zero_result_provider_response(self, tmp_path, monkeypatch):
        manager, limiter = _make_manager(tmp_path, 5, monkeypatch)
        manager._tavily_client = _OkClient()

        result = await manager.search(
            "unique zero result control query", depth=WebSearchDepth.QUICK, localize=False)

        assert result.error is None
        assert result.pages == []
        assert result.has_results is False


# STANDARD depth: search succeeds, extract raises -> partial success.
class TestStandardDepthExtractFailure:

    _SEARCH_PAGES = [
        {"url": "https://a.test/1", "title": "A", "content": "content a", "score": 0.9},
        {"url": "https://a.test/2", "title": "B", "content": "content b", "score": 0.5},
    ]

    async def test_extract_raises_keeps_search_pages_and_sets_extract_error(
            self, tmp_path, monkeypatch):
        client = _MixedClient(
            self._SEARCH_PAGES, extract_exc=RuntimeError("EXTRACT_MESSAGE_MARKER"))
        manager, limiter = _make_manager(tmp_path, 5, monkeypatch)
        manager._tavily_client = client

        result = await manager.search(
            "unique standard extract fail query", depth=WebSearchDepth.STANDARD, localize=False)

        assert result.error is None
        assert len(result.pages) == 2
        assert result.extract_error is not None
        assert result.extract_error.startswith("tavily_extract:")
        assert "EXTRACT_MESSAGE_MARKER" not in result.extract_error
        assert result.has_results is True

    async def test_control_extract_succeeds_no_extract_error(self, tmp_path, monkeypatch):
        extracted = [{"url": "https://a.test/1", "title": "A", "raw_content": "full text"}]
        client = _MixedClient(self._SEARCH_PAGES[:1], extract_results=extracted)
        manager, limiter = _make_manager(tmp_path, 5, monkeypatch)
        manager._tavily_client = client

        result = await manager.search(
            "unique standard extract ok query", depth=WebSearchDepth.STANDARD, localize=False)

        assert result.error is None
        assert result.extract_error is None


# fetch_url_content.
class TestFetchUrlContent:

    async def test_direct_empty_extract_raises_returns_failed_outcome_list(self):
        mgr = _bare_manager(None)
        mgr._direct_fetch = AsyncMock(return_value=[])
        mgr._tavily_extract = AsyncMock(
            side_effect=RetrievalError(source="tavily_extract", reason="RuntimeError"))

        pages = await mgr.fetch_url_content("https://example.test/story")

        assert pages == []
        assert isinstance(pages, OutcomeList)
        assert pages.status == "failed"
        assert pages.reason.startswith("tavily_extract:")
        assert outcome_status(pages) == ("failed", pages.reason)

    async def test_direct_thin_extract_raises_returns_the_direct_page(self):
        mgr = _bare_manager(None)
        mgr._direct_fetch = AsyncMock(return_value=[_page(content="thin but real")])
        mgr._tavily_extract = AsyncMock(
            side_effect=RetrievalError(source="tavily_extract", reason="RuntimeError"))

        pages = await mgr.fetch_url_content("https://example.test/thin")

        assert pages and pages[0].content == "thin but real"

    async def test_control_budget_refusal_still_typed_blocked(self, tmp_path, monkeypatch):
        manager, limiter = _make_manager(tmp_path, 0, monkeypatch)
        manager._direct_fetch = AsyncMock(return_value=[])
        manager._tavily_extract = AsyncMock(
            return_value=[_page("t" * 1000, source="tavily_extract")])

        pages = await manager.fetch_url_content("https://example.test/budget")

        assert pages == []
        assert getattr(pages, "blocked", None) == "budget"
        manager._tavily_extract.assert_not_awaited()


# Deployed consumers (read-only drive; no edits to either consumer file).
class TestDeployedConsumers:

    async def test_agentic_fetch_url_tool_reports_could_not_fetch_on_failed_outcome(self):
        from core.agentic.tools import ToolExecutor

        executor = object.__new__(ToolExecutor)
        executor.web_search_manager = MagicMock()
        executor.web_search_manager.fetch_url_content = AsyncMock(
            return_value=OutcomeList.failed("tavily_extract:RuntimeError"))
        executor._current_web_source_map = {}
        executor._merge_web_ids = MagicMock(return_value=[])

        result = await executor._execute_fetch_url("https://example.test/failed")

        assert result == "[Could not fetch content from https://example.test/failed]"

    async def test_gatherer_receipt_records_provider_failure(self, tmp_path, monkeypatch):
        """gatherer_web.py's `_get_web_search_results` end-to-end receipt
        proof. A fixed sync `trigger` decision supplies one non-news search
        term, and `_is_broad_news_query` is stubbed False, so `multi_search`
        never reaches its LLM/semantic decompose paths (not gated by
        `auto_decompose=False`) — the only model-shaped code this could hit."""
        client = _RaisingClient(RuntimeError("GATHERER_PROVIDER_MARKER"))
        manager, limiter = _make_manager(tmp_path, 5, monkeypatch)
        manager._tavily_client = client
        manager._is_broad_news_query = lambda q: False

        safe_term = "documentation for python asyncio wait_for timeout behavior"
        decision = SimpleNamespace(
            should_search=True,
            source="test",
            reason="test",
            confidence=0.9,
            depth=SimpleNamespace(value="quick"),
            search_terms=[safe_term],
            evidence_needed=False,
        )
        host = _GathererHost(manager, trigger=lambda q: decision)

        result = await host._get_web_search_results(safe_term)

        # CGR-20260913-007 #92 sibling (F8b): a provider-error result
        # (`result.error` set, not budget-blocked) is now a typed failure
        # with a constant label -- never the raw provider text -- instead
        # of the bare `None` a genuine empty search returns.
        assert outcome_status(result) == ("failed", "provider_error")
        assert result == []
        assert host.last_web_decision["error"]
        assert "GATHERER_PROVIDER_MARKER" not in host.last_web_decision["error"]
