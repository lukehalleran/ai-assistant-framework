"""2026-09-12: web-search credit reservation closes a check-then-act race.

Live/reproduced defect: `can_search()` only reads the shared counter and
`record_usage()` only writes it at the very end of `_execute_search()` — two
concurrent `search()` calls (reachable via `multi_search()`'s concurrent
sub-searches) can both pass the budget check against the same last credit
before either records usage, so a 1-credit daily limit lets 2 credits' worth
of provider calls through.

`WebSearchRateLimiter.reserve()` closes it: the estimated cost is debited
into `_reserved_today` UP FRONT (before any await), so a concurrent caller
sees the hold and is refused. `_execute_search()` then spends against that
reservation "on dispatch" — immediately before each billable provider call,
never refunded even if that call later times out, raises, or is cancelled —
and `search()` always `settle()`s the reservation exactly once, in a
`finally`, folding the real spend into `_credits_today` and releasing
whatever of the hold went unused.

Every test drives the REAL `WebSearchManager` + REAL `WebSearchRateLimiter`;
only the Tavily provider calls (`_tavily_search`/`_tavily_extract`/
`_select_links_for_following`) are faked, and only with real interleaving
(`asyncio.sleep(0)` / `asyncio.Event`) where a race needs to be forced.
Assertions check provider call counts and the limiter's actual credit/
reservation state, never just a result's error string.
"""
from __future__ import annotations

import asyncio
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import knowledge.web_search_manager as wsm
from knowledge.web_search_manager import (
    WebPage,
    WebSearchDepth,
    WebSearchManager,
    WebSearchRateLimiter,
)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _isolated_rate_limiter_registry(monkeypatch):
    """A fresh, empty live-limiter registry per test. The process-wide
    WeakSet otherwise accumulates every limiter any earlier test (or module
    singleton) left alive — reading/registering against it unpatched is an
    uncontrolled oracle."""
    monkeypatch.setattr(wsm, "_LIVE_RATE_LIMITERS", weakref.WeakSet())


def _page(url: str = "https://example.test/a", content: str = "c",
          source: str = "tavily_search") -> WebPage:
    return WebPage(url=url, title="T", content=content, snippet=content[:200], source=source)


def _make_manager(tmp_path, daily_limit, monkeypatch, name="credits.json"):
    """A real WebSearchManager wired to a real WebSearchRateLimiter whose
    state file lives under tmp_path. Only cache + tavily-readiness are
    stubbed; provider calls are faked per-test."""
    limiter = WebSearchRateLimiter(daily_limit=daily_limit, state_file=str(tmp_path / name))
    manager = WebSearchManager(api_key="synthetic-key", rate_limiter=limiter)
    manager.cache = SimpleNamespace(get=lambda q, d: None, put=lambda r: None)
    monkeypatch.setattr(manager, "_ensure_tavily", lambda: True, raising=False)
    monkeypatch.setattr(manager, "is_enabled", lambda: True, raising=False)
    return manager, limiter


# ---------------------------------------------------------------------------
# a/b — the race: reserve() closes the check-then-act gap
# ---------------------------------------------------------------------------

class TestReservationClosesTheRace:

    def test_two_concurrent_quick_searches_only_one_is_admitted(self, tmp_path, monkeypatch):
        manager, limiter = _make_manager(tmp_path, daily_limit=1, monkeypatch=monkeypatch)
        search_calls = []

        async def fake_tavily_search(query, max_results, **kw):
            search_calls.append(query)
            await asyncio.sleep(0)  # force real interleaving with the 2nd caller
            return []

        monkeypatch.setattr(manager, "_tavily_search", fake_tavily_search, raising=False)

        async def _both():
            return await asyncio.gather(
                manager.search("same query", depth=WebSearchDepth.QUICK, localize=False),
                manager.search("same query", depth=WebSearchDepth.QUICK, localize=False),
            )

        r1, r2 = _run(_both())

        assert len(search_calls) == 1, "only one provider call should ever dispatch"
        results = [r1, r2]
        oks = [r for r in results if not r.error]
        errors = [r.error for r in results if r.error]
        assert len(oks) == 1
        assert len(errors) == 1
        assert errors[0].startswith("Daily credit limit reached")
        assert limiter._credits_today == 1.0
        assert limiter._reserved_today == 0.0

    def test_two_concurrent_standard_searches_over_a_tight_budget(self, tmp_path, monkeypatch):
        manager, limiter = _make_manager(tmp_path, daily_limit=3, monkeypatch=monkeypatch)
        search_calls = []
        extract_calls = []

        async def fake_tavily_search(query, max_results, **kw):
            search_calls.append(query)
            await asyncio.sleep(0)  # force interleaving before the 2nd reserve() attempt
            return [_page("https://example.test/1"), _page("https://example.test/2")]

        async def fake_tavily_extract(urls):
            extract_calls.append(list(urls))
            return [_page(u, source="tavily_extract") for u in urls]

        monkeypatch.setattr(manager, "_tavily_search", fake_tavily_search, raising=False)
        monkeypatch.setattr(manager, "_tavily_extract", fake_tavily_extract, raising=False)

        async def _both():
            return await asyncio.gather(
                manager.search("q", depth=WebSearchDepth.STANDARD, localize=False),
                manager.search("q", depth=WebSearchDepth.STANDARD, localize=False),
            )

        r1, r2 = _run(_both())

        assert len(search_calls) == 1
        results = [r1, r2]
        oks = [r for r in results if not r.error]
        errors = [r.error for r in results if r.error]
        assert len(oks) == 1
        assert len(errors) == 1
        assert errors[0].startswith("Daily credit limit reached")
        assert limiter._credits_today == 2.0
        assert limiter._reserved_today == 0.0


# ---------------------------------------------------------------------------
# c/d/e/f — charge on dispatch: never refunded, never phantom-held
# ---------------------------------------------------------------------------

class TestChargeOnDispatch:

    def test_timeout_during_extract_still_charges_base_and_extract(self, tmp_path, monkeypatch):
        manager, limiter = _make_manager(tmp_path, daily_limit=5, monkeypatch=monkeypatch)
        stuck = asyncio.Event()

        async def fake_tavily_search(query, max_results, **kw):
            return [_page("https://example.test/1"), _page("https://example.test/2")]

        async def fake_tavily_extract(urls):
            await stuck.wait()  # never set -> only the timeout ends this
            return []

        monkeypatch.setattr(manager, "_tavily_search", fake_tavily_search, raising=False)
        monkeypatch.setattr(manager, "_tavily_extract", fake_tavily_extract, raising=False)

        result = _run(manager.search("q", depth=WebSearchDepth.STANDARD,
                                     localize=False, timeout=0.05))

        assert result.error and "timed out" in result.error.lower()
        assert limiter._credits_today == 2.0, "base (1.0) + extract (2*0.5) were dispatched"
        assert limiter._reserved_today == 0.0

    def test_timeout_during_base_search_charges_only_base(self, tmp_path, monkeypatch):
        manager, limiter = _make_manager(tmp_path, daily_limit=5, monkeypatch=monkeypatch)
        stuck = asyncio.Event()

        async def fake_tavily_search(query, max_results, **kw):
            await stuck.wait()
            return []

        monkeypatch.setattr(manager, "_tavily_search", fake_tavily_search, raising=False)

        result = _run(manager.search("q", depth=WebSearchDepth.QUICK,
                                     localize=False, timeout=0.05))

        assert result.error and "timed out" in result.error.lower()
        assert limiter._credits_today == 1.0
        assert limiter._reserved_today == 0.0

    def test_provider_exception_still_charges_the_dispatched_base_cost(self, tmp_path, monkeypatch):
        manager, limiter = _make_manager(tmp_path, daily_limit=5, monkeypatch=monkeypatch)

        async def fake_tavily_search(query, max_results, **kw):
            raise RuntimeError("provider exploded")

        monkeypatch.setattr(manager, "_tavily_search", fake_tavily_search, raising=False)

        result = _run(manager.search("q", depth=WebSearchDepth.QUICK, localize=False))

        assert result.error
        assert limiter._credits_today == 1.0
        assert limiter._reserved_today == 0.0

    def test_ensure_tavily_false_charges_nothing_and_releases_the_hold(self, tmp_path, monkeypatch):
        manager, limiter = _make_manager(tmp_path, daily_limit=5, monkeypatch=monkeypatch)
        monkeypatch.setattr(manager, "_ensure_tavily", lambda: False, raising=False)

        result = _run(manager.search("q", depth=WebSearchDepth.QUICK, localize=False))

        assert result.error == "Tavily client not available"
        assert limiter._credits_today == 0.0
        assert limiter._reserved_today == 0.0
        assert limiter.get_remaining_credits() == 5.0


# ---------------------------------------------------------------------------
# g — unused portion of a reservation is released, not silently kept held
# ---------------------------------------------------------------------------

class TestUnusedReservationIsReleased:

    def test_standard_search_with_no_pages_releases_the_unspent_extract_hold(
            self, tmp_path, monkeypatch):
        manager, limiter = _make_manager(tmp_path, daily_limit=5, monkeypatch=monkeypatch)
        extract = AsyncMock()

        async def fake_tavily_search(query, max_results, **kw):
            return []  # no pages -> extraction stage is never entered

        monkeypatch.setattr(manager, "_tavily_search", fake_tavily_search, raising=False)
        monkeypatch.setattr(manager, "_tavily_extract", extract, raising=False)

        result = _run(manager.search("q", depth=WebSearchDepth.STANDARD, localize=False))

        assert not result.error
        extract.assert_not_awaited()
        assert limiter._credits_today == 1.0, "only the base search was actually spent"
        assert limiter._reserved_today == 0.0
        assert limiter.get_remaining_credits() == 4.0


# ---------------------------------------------------------------------------
# h — DEEP link-following spends per-URL and trims to what's affordable
# ---------------------------------------------------------------------------

class TestDeepLinkFollowingBudget:

    def _wire(self, tmp_path, monkeypatch, daily_limit):
        manager, limiter = _make_manager(tmp_path, daily_limit=daily_limit, monkeypatch=monkeypatch)
        extract_calls = []
        four_links = [f"https://example.test/link{i}" for i in range(4)]

        async def fake_tavily_search(query, max_results, **kw):
            return [_page("https://example.test/1"), _page("https://example.test/2")]

        async def fake_tavily_extract(urls):
            extract_calls.append(list(urls))
            return [_page(u, source="tavily_extract") for u in urls]

        async def fake_select_links(query, current_pages, max_links=2):
            # Simulates the LLM selector proposing more links than the
            # normal max_links=2 cap would — the budget, not the selector,
            # must be what trims the list.
            return list(four_links)

        monkeypatch.setattr(manager, "_tavily_search", fake_tavily_search, raising=False)
        monkeypatch.setattr(manager, "_tavily_extract", fake_tavily_extract, raising=False)
        monkeypatch.setattr(manager, "_select_links_for_following", fake_select_links, raising=False)
        return manager, limiter, extract_calls, four_links

    def test_tight_budget_trims_the_followed_links(self, tmp_path, monkeypatch):
        manager, limiter, extract_calls, four_links = self._wire(tmp_path, monkeypatch, daily_limit=3)

        result = _run(manager.search("q", depth=WebSearchDepth.DEEP, localize=False))

        assert not result.error
        assert limiter._credits_today == 3.0, "never exceeds the 3-credit daily limit"
        assert limiter._reserved_today == 0.0
        assert extract_calls[-1] == four_links[:2], "trimmed to what the budget could afford"

    def test_generous_budget_follows_every_selected_link(self, tmp_path, monkeypatch):
        manager, limiter, extract_calls, four_links = self._wire(tmp_path, monkeypatch, daily_limit=50)

        result = _run(manager.search("q", depth=WebSearchDepth.DEEP, localize=False))

        assert not result.error
        assert extract_calls[-1] == four_links, "budget is generous -> nothing trimmed"
        assert limiter._credits_today == 4.0  # 1 base + 1 extract(top2) + 2 (4 links * 0.5)
        assert limiter._reserved_today == 0.0


# ---------------------------------------------------------------------------
# i — midnight rollover between reserve() and settle() touches neither
#     today's nor the stale reservation's counters wrongly
# ---------------------------------------------------------------------------

class TestMidnightRollover:

    def test_settle_after_the_date_rolls_over_debits_and_releases_nothing(
            self, tmp_path, monkeypatch):
        from datetime import datetime as real_datetime

        class _FakeClock:
            current = real_datetime(2026, 9, 12, 23, 59)

            @classmethod
            def now(cls):
                return cls.current

        monkeypatch.setattr(wsm, "datetime", _FakeClock)

        limiter = WebSearchRateLimiter(daily_limit=5, state_file=str(tmp_path / "rollover.json"))
        reservation = limiter.reserve(2.0)
        assert reservation is not None
        assert reservation.spend(1.5)
        assert limiter._reserved_today == 2.0

        _FakeClock.current = real_datetime(2026, 9, 13, 0, 1)  # midnight passes
        reservation.settle()

        assert limiter._credits_today == 0.0, "never debited against the wrong (new) day"
        assert limiter._reserved_today == 0.0, "never left as a phantom hold either"
        assert limiter._current_date == "2026-09-13"


# ---------------------------------------------------------------------------
# j — fetch_url_content's Tavily fallback is now budgeted
# ---------------------------------------------------------------------------

class TestFetchUrlContentBudget:

    def test_zero_budget_skips_the_billed_tavily_fallback(self, tmp_path, monkeypatch):
        manager, limiter = _make_manager(tmp_path, daily_limit=0, monkeypatch=monkeypatch)
        manager._direct_fetch = AsyncMock(return_value=[])
        manager._tavily_extract = AsyncMock(
            return_value=[_page("t" * 1000, source="tavily_extract")])

        pages = _run(manager.fetch_url_content("https://example.test/page"))

        manager._tavily_extract.assert_not_awaited()
        assert pages == []
        assert limiter._credits_today == 0.0
        assert limiter._reserved_today == 0.0

    def test_funded_budget_spends_exactly_once(self, tmp_path, monkeypatch):
        manager, limiter = _make_manager(tmp_path, daily_limit=5, monkeypatch=monkeypatch)
        manager._direct_fetch = AsyncMock(return_value=[])
        manager._tavily_extract = AsyncMock(
            return_value=[_page("t" * 1000, source="tavily_extract")])

        pages = _run(manager.fetch_url_content("https://example.test/page"))

        manager._tavily_extract.assert_awaited_once()
        assert pages and pages[0].source == "tavily_extract"
        assert limiter._credits_today == 0.5
        assert limiter._reserved_today == 0.0


# ---------------------------------------------------------------------------
# k — get_remaining_credits() reflects an in-flight reservation
# ---------------------------------------------------------------------------

class TestRemainingCreditsReflectsInFlightReservations:

    def test_reserve_then_settle_round_trips_the_remaining_credits(self, tmp_path):
        limiter = WebSearchRateLimiter(daily_limit=1, state_file=str(tmp_path / "k.json"))
        assert limiter.get_remaining_credits() == 1.0

        reservation = limiter.reserve(1.0)
        assert reservation is not None
        assert limiter.get_remaining_credits() == 0.0, "the hold is visible while in flight"

        reservation.settle()  # nothing was ever spent
        assert limiter.get_remaining_credits() == 1.0
