"""2026-09-12 adversarial review, focused follow-up (findings 2 and 3).

Finding 2 — cross-midnight dispatch escapes new-day accounting.
``WebSearchRateLimiter._reservation_spend`` never validated a reservation's
``date`` against the limiter's current date: a reservation taken at 23:59 kept
charging its ORIGINAL day even after ``_check_date_reset_locked`` had already
zeroed that day's hold and rolled the limiter onto tomorrow — a search whose
base call went out before midnight but whose extraction call dispatched after
it landed the extraction's cost nowhere the limiter could see, so the new day
opened with the full budget available despite two credits' worth of actual
provider calls. Fixed by giving ``SearchReservation`` a ``day_used`` field —
the part of ``used`` charged against ``date`` — and having ``spend()`` re-book
a stale reservation onto today's budget (empty hold) before charging it, so a
dispatch after midnight always competes for TODAY's credits like anything
else. The same gap let ``fetch_url_content`` dispatch its billed Tavily
fallback even when the actual ``spend()`` call was refused (only ``reserve()``
was checked).

Finding 3 — a budget refusal disappears whenever something else in the same
turn succeeded. ``multi_search`` discarded its joined ``errors`` string
whenever ANY sub-query returned pages, so a 1-credit budget with two
sub-queries (one funded, one refused) reported ``error=None`` and the receipt
built from it said nothing was blocked. The same shape existed for
``fetch_url_content`` (a free direct fetch that returns nothing, whose billed
Tavily fallback the budget then refuses, returned a bare empty list — no way
to tell "nothing there" from "budget stopped us from checking") and for
``utils/web_evidence_receipt.py``'s round-scanning loop (it only ever read
``results.error`` for the marker string, never a structured block reason).
Fixed with a typed ``blocked: Optional[str]`` field threaded through
``WebSearchResult``/``MultiSearchResult`` (set at the two refusal sites and
survived through ``multi_search``'s aggregation and ``to_web_search_result``),
a ``SearchRound.blocked`` field, a ``FetchedPages``/``BlockedFetchText``
pair (list/str subclasses that carry the reason without breaking any
"not pages"/"in text" caller), and receipt-building code that reads all of
the above ahead of the old marker-string fallback.

Every case drives the DEPLOYED functions (``WebSearchRateLimiter``,
``WebSearchManager``, the gatherer mixin, the loop dispatch, ``handle_submit``).
Only the clock and provider/LLM I/O are faked.

class: BC-38, BC-47
"""

from __future__ import annotations

import asyncio
import weakref
from datetime import datetime as real_datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import knowledge.web_search_manager as wsm
import utils.web_search_trigger as wst
from core.agentic.formatters import AgenticFormatter
from core.agentic.types import AgenticSearchSession, SearchDecision
from utils.web_evidence_receipt import (
    BUDGET_NOTICE,
    BUDGET_NOTICE_PARTIAL,
    build_web_evidence_receipt,
    web_evidence_notice,
)

from tests.unit.test_sep12_search_budget_reservation import _make_manager, _page
from tests.unit.test_sep12_web_evidence_budget import (
    NEWS_QUERY,
    PUBLIC_CLAIM_QUERY,
    _GathererHost,
    _limiter,
    _manager,
    _round,
)


# ---------------------------------------------------------------------------
# Isolation (copied from tests/unit/test_sep12_web_evidence_budget.py — see
# the plan: never import a fixture across test modules).
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolation(monkeypatch):
    """Fresh live-limiter registry, trigger cache and deferred-request slot;
    no location resolution (search keys stay the literal terms)."""
    import utils.location_resolver as lr
    from core.agentic import gate as gate_mod

    monkeypatch.setattr(wsm, "_LIVE_RATE_LIMITERS", weakref.WeakSet())
    monkeypatch.setattr(lr, "get_user_location", lambda: None)
    wst._llm_trigger_cache.clear()
    wst._llm_trigger_inflight.clear()
    gate_mod._DEFERRED_REQUEST_SLOT.clear()
    yield
    wst._llm_trigger_cache.clear()
    wst._llm_trigger_inflight.clear()
    gate_mod._DEFERRED_REQUEST_SLOT.clear()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fake_clock(start: real_datetime):
    """A class monkeypatched over ``wsm.datetime`` whose ``.now()`` returns a
    mutable ``current`` — the same pattern as
    TestMidnightRollover.test_settle_after_the_date_rolls_over_debits_and_releases_nothing
    in tests/unit/test_sep12_search_budget_reservation.py."""

    class _Clock:
        current = start

        @classmethod
        def now(cls):
            return cls.current

    return _Clock


async def _partial_multi_search_result_async(tmp_path):
    """A real multi_search over a 1-credit budget with two sub-queries: the
    shared "yielding provider" fake (from _manager()) forces real
    interleaving (``asyncio.sleep(0)``) so the reservation race genuinely
    admits only the first dispatch, exactly like
    TestReservationClosesTheRace in test_sep12_search_budget_reservation.py —
    one sub-query is funded and returns a page, the other is refused."""
    limiter = _limiter(tmp_path, used=0.0, limit=1)
    manager, provider_calls = _manager(limiter)
    result = await manager.multi_search(
        "alpha solar output",
        depth=wsm.WebSearchDepth.QUICK,
        sub_queries=["alpha solar output", "beta wind output"],
        auto_decompose=False,
    )
    return result, provider_calls, limiter


def _fake_llm_search_response(terms):
    async def _classify(query, model_manager, remaining_credits, timeout,
                        conversation_context=None):
        return wst.LLMSearchTriggerResponse(
            should_search=True, confidence=0.8, reason="stub",
            search_terms=list(terms), search_depth="quick",
            num_searches=len(terms))
    return _classify


async def _partial_web_search_decision(tmp_path, monkeypatch):
    """The gatherer's own receipt for the same partial-refusal shape, reached
    through analyze_for_web_search_llm + WebSearchMixin instead of calling
    multi_search directly (PUBLIC_CLAIM_QUERY is probed to reach the Tier-4/
    LLM-trigger classifier — see test_sep12_web_evidence_budget.py)."""
    limiter = _limiter(tmp_path, used=0.0, limit=1)
    manager, _ = _manager(limiter)
    monkeypatch.setattr(
        wst, "_classify_with_llm_unified",
        _fake_llm_search_response(["alpha solar output", "beta wind output"]),
    )
    host = _GathererHost(
        manager, trigger_llm=wst.analyze_for_web_search_llm, model_manager=object())
    result = await host._get_web_search_results(PUBLIC_CLAIM_QUERY)
    return result, host.last_web_decision


async def _blocked_fetch_round(tmp_path, monkeypatch):
    """A [Fetch URL] round whose free direct layer found nothing and whose
    billed Tavily fallback the budget refused."""
    from core.agentic.tools import ToolExecutor

    manager, _ = _make_manager(tmp_path, daily_limit=0, monkeypatch=monkeypatch)
    manager._direct_fetch = AsyncMock(return_value=[])
    executor = ToolExecutor(
        model_manager=MagicMock(), web_search_manager=manager, formatter=AgenticFormatter())
    decision = SearchDecision(
        wants_fetch_url=True, fetch_url="https://example.test/story", fetch_url_reason="r")
    result = await executor._dispatch_fetch_url(decision, 1)
    return result.round_data


def _capture_storage(monkeypatch):
    """Copied from TestDeliverySites._capture_storage in
    test_sep12_web_evidence_budget.py (a plain staticmethod, not a fixture)."""
    import gui.handlers as handlers

    captured = {}

    def _capture(*args, **kwargs):
        captured["final_output"] = kwargs.get(
            "final_output", args[4] if len(args) > 4 else None)

    monkeypatch.setattr(handlers, "_dispatch_storage", _capture)
    return captured


# ---------------------------------------------------------------------------
# B1 — cross-midnight dispatch accounting (finding 2)
# ---------------------------------------------------------------------------

class TestMidnightDispatchAccounting:

    def test_extraction_after_midnight_is_charged_to_the_new_day(self, tmp_path, monkeypatch):
        clock = _fake_clock(real_datetime(2026, 9, 12, 23, 59))
        monkeypatch.setattr(wsm, "datetime", clock)

        limiter = _limiter(tmp_path, used=0.0, limit=5)
        manager, _ = _manager(limiter)
        extract_calls = []

        async def _search(query, max_results, **kw):
            clock.current = real_datetime(2026, 9, 13, 0, 1)
            return [_page("https://example.test/1"), _page("https://example.test/2")]

        async def _extract(urls):
            extract_calls.append(list(urls))
            return [_page(u, source="tavily_extract") for u in urls]

        manager._tavily_search = _search
        manager._tavily_extract = _extract

        result = asyncio.run(manager.search(
            "cross midnight extraction test", depth=wsm.WebSearchDepth.STANDARD,
            localize=False))

        assert len(extract_calls) == 1
        assert result.total_credits_used == 2.0
        assert limiter._current_date == "2026-09-13"
        assert limiter._credits_today == 1.0
        assert limiter._reserved_today == 0.0
        assert limiter.get_remaining_credits() == 4.0

        import json
        with open(limiter.state_file) as f:
            state = json.load(f)
        assert state["date"] == "2026-09-13"
        assert state["credits_today"] == 1.0

    def test_new_day_competitor_holds_the_budget_so_the_late_stage_is_refused(
            self, tmp_path, monkeypatch):
        clock = _fake_clock(real_datetime(2026, 9, 12, 23, 59))
        monkeypatch.setattr(wsm, "datetime", clock)

        limiter = _limiter(tmp_path, used=0.0, limit=2)
        manager, _ = _manager(limiter)
        extract_calls = []
        holder = {}

        async def _search(query, max_results, **kw):
            clock.current = real_datetime(2026, 9, 13, 0, 1)
            holder["competitor"] = limiter.reserve(2.0)
            return [_page("https://example.test/1"), _page("https://example.test/2")]

        async def _extract(urls):
            extract_calls.append(list(urls))
            return [_page(u, source="tavily_extract") for u in urls]

        manager._tavily_search = _search
        manager._tavily_extract = _extract

        result = asyncio.run(manager.search(
            "cross midnight competitor test", depth=wsm.WebSearchDepth.STANDARD,
            localize=False))

        assert holder["competitor"] is not None
        assert len(extract_calls) == 0
        assert result.total_credits_used == 1.0
        assert len(result.pages) == 2
        assert limiter._reserved_today == 2.0

        competitor = holder["competitor"]
        assert competitor.spend(2.0) is True
        competitor.settle()
        assert limiter._credits_today == 2.0
        assert limiter._reserved_today == 0.0
        assert limiter.get_remaining_credits() == 0.0

    def test_spend_rebooks_a_prior_day_reservation(self, tmp_path, monkeypatch):
        clock = _fake_clock(real_datetime(2026, 9, 12, 23, 59))
        monkeypatch.setattr(wsm, "datetime", clock)

        limiter = wsm.WebSearchRateLimiter(
            daily_limit=5, state_file=str(tmp_path / "reboot.json"))
        r = limiter.reserve(2.0)
        assert r is not None
        assert r.spend(1.0) is True

        clock.current = real_datetime(2026, 9, 13, 0, 1)
        assert r.spend(1.0) is True
        assert r.date == "2026-09-13"
        assert r.used == 2.0

        r.settle()
        assert limiter._credits_today == 1.0
        assert limiter._reserved_today == 0.0


# ---------------------------------------------------------------------------
# B1 — fetch_url_content's spend must actually be checked (finding 3, related)
# ---------------------------------------------------------------------------

class TestFetchUrlSpendGuard:

    def test_refused_spend_never_dispatches_the_billed_fetch(self, tmp_path, monkeypatch):
        manager, limiter = _make_manager(tmp_path, daily_limit=5, monkeypatch=monkeypatch)
        manager._direct_fetch = AsyncMock(return_value=[])
        manager._tavily_extract = AsyncMock(
            return_value=[_page("t" * 1000, source="tavily_extract")])
        monkeypatch.setattr(limiter, "_reservation_spend", lambda r, c: False)

        pages = asyncio.run(manager.fetch_url_content("https://example.test/refused"))

        assert pages == []
        assert getattr(pages, "blocked", None) == "budget"
        manager._tavily_extract.assert_not_awaited()
        assert limiter._credits_today == 0.0
        assert limiter._reserved_today == 0.0

    def test_zero_budget_with_nothing_free_is_a_typed_block(self, tmp_path, monkeypatch):
        manager, limiter = _make_manager(tmp_path, daily_limit=0, monkeypatch=monkeypatch)
        manager._direct_fetch = AsyncMock(return_value=[])
        manager._tavily_extract = AsyncMock(
            return_value=[_page("t" * 1000, source="tavily_extract")])

        pages = asyncio.run(manager.fetch_url_content("https://example.test/zero"))

        assert pages == []
        assert getattr(pages, "blocked", None) == "budget"
        manager._tavily_extract.assert_not_awaited()

    def test_free_direct_fetch_is_unaffected_at_zero_budget(self, tmp_path, monkeypatch):
        manager, limiter = _make_manager(tmp_path, daily_limit=0, monkeypatch=monkeypatch)
        manager._direct_fetch = AsyncMock(return_value=[_page(content="d" * 1000)])
        manager._tavily_extract = AsyncMock()

        pages = asyncio.run(manager.fetch_url_content("https://example.test/free"))

        assert pages and pages[0].content == "d" * 1000
        assert getattr(pages, "blocked", None) is None
        manager._tavily_extract.assert_not_awaited()


# ---------------------------------------------------------------------------
# B2 — a partial budget refusal must survive alongside real pages (finding 3)
# ---------------------------------------------------------------------------

class TestPartialBudgetOutcome:

    def test_multi_search_keeps_pages_and_records_the_refusal(self, tmp_path):
        result, provider_calls, limiter = asyncio.run(
            _partial_multi_search_result_async(tmp_path))

        assert len(provider_calls) == 1
        assert len(result.pages) == 1
        assert result.error is None
        assert result.blocked == "budget"
        assert limiter._credits_today == 1.0

    def test_loop_round_receipt_is_partial(self, tmp_path):
        result, _, _ = asyncio.run(_partial_multi_search_result_async(tmp_path))
        session = SimpleNamespace(rounds=[_round("alpha solar output", results=result)])

        receipt = build_web_evidence_receipt(None, None, session=session)

        assert receipt["requested"] is True
        assert receipt["blocked"] == "budget"
        assert receipt["acquired"] == 1
        assert web_evidence_notice(receipt) == BUDGET_NOTICE_PARTIAL

    async def test_gatherer_records_the_partial_refusal(self, tmp_path, monkeypatch):
        result, decision = await _partial_web_search_decision(tmp_path, monkeypatch)

        assert result is not None
        assert len(result.pages) == 1
        assert decision["blocked"] == "budget"
        assert decision["requested"] is True
        assert decision["results"] == 1
        notice = web_evidence_notice(build_web_evidence_receipt(None, decision))
        assert notice == BUDGET_NOTICE_PARTIAL


# ---------------------------------------------------------------------------
# B2 — the loop's fetch_url dispatch carries the same typed reason
# ---------------------------------------------------------------------------

class TestFetchDispatchBudgetOutcome:

    async def test_dispatch_fetch_url_records_a_budget_block(self, tmp_path, monkeypatch):
        round_data = await _blocked_fetch_round(tmp_path, monkeypatch)

        assert round_data.blocked == "budget"
        assert "Could not fetch" in round_data.summary

        session = SimpleNamespace(rounds=[round_data])
        receipt = build_web_evidence_receipt(None, None, session=session)
        assert receipt["requested"] is True
        assert receipt["blocked"] == "budget"
        assert receipt["fetched"] == 0
        assert web_evidence_notice(receipt) == BUDGET_NOTICE

    async def test_free_fetch_round_is_not_blocked(self, tmp_path, monkeypatch):
        from core.agentic.tools import ToolExecutor

        manager, _ = _make_manager(tmp_path, daily_limit=0, monkeypatch=monkeypatch)
        manager._direct_fetch = AsyncMock(return_value=[_page(content="d" * 1000)])
        executor = ToolExecutor(
            model_manager=MagicMock(), web_search_manager=manager, formatter=AgenticFormatter())
        decision = SearchDecision(
            wants_fetch_url=True, fetch_url="https://example.test/free2", fetch_url_reason="r")

        result = await executor._dispatch_fetch_url(decision, 1)

        assert result.round_data.blocked is None

        session = SimpleNamespace(rounds=[result.round_data])
        receipt = build_web_evidence_receipt(None, None, session=session)
        assert receipt["fetched"] == 1
        assert receipt["blocked"] is None
        assert web_evidence_notice(receipt) == ""


# ---------------------------------------------------------------------------
# Delivery — the partial notice reaches display and storage exactly once
# ---------------------------------------------------------------------------

class TestDeliveryDisclosesPartialOutcomes:

    async def test_enhanced_turn_discloses_a_partial_budget_refusal_once(
            self, tmp_path, monkeypatch):
        from tests.unit.test_handle_submit import _final_content, _make_orchestrator, _run_submit

        _, decision = await _partial_web_search_decision(tmp_path, monkeypatch)
        captured = _capture_storage(monkeypatch)
        orch = _make_orchestrator(
            streaming_chunks=["The count", " is still open."],
            prepare_result=("P", "S", {"web_search_decision": decision}))
        results = await _run_submit(NEWS_QUERY, orch)

        content = _final_content(results)
        assert content.count(BUDGET_NOTICE_PARTIAL.strip()) == 1
        assert content.count(BUDGET_NOTICE.strip()) == 0
        stored = captured.get("final_output") or ""
        assert stored.count(BUDGET_NOTICE_PARTIAL.strip()) == 1
        assert stored.count(BUDGET_NOTICE.strip()) == 0

    async def test_agentic_turn_discloses_a_loop_partial_budget_refusal_once(
            self, tmp_path, monkeypatch):
        from tests.unit.test_handle_submit import (
            _debug_record, _final_content, _make_orchestrator, _run_submit)

        result, _, _ = await _partial_multi_search_result_async(tmp_path)
        captured = _capture_storage(monkeypatch)
        orch = _make_orchestrator(
            agentic_enabled=True, agentic_items=["Here is what I can tell you."])
        session = AgenticSearchSession(query="q")
        session.rounds.append(_round("alpha solar output", results=result))
        orch._agentic_controller._last_session = session

        results = await _run_submit("do you remember my brother's name?", orch)

        assert _debug_record(results)["mode"] == "agentic-search"
        content = _final_content(results)
        assert content.count(BUDGET_NOTICE_PARTIAL.strip()) == 1
        assert content.count(BUDGET_NOTICE.strip()) == 0
        stored = captured.get("final_output") or ""
        assert stored.count(BUDGET_NOTICE_PARTIAL.strip()) == 1

    async def test_agentic_turn_discloses_a_loop_fetch_budget_refusal_once(
            self, tmp_path, monkeypatch):
        from tests.unit.test_handle_submit import (
            _debug_record, _final_content, _make_orchestrator, _run_submit)

        round_data = await _blocked_fetch_round(tmp_path, monkeypatch)
        captured = _capture_storage(monkeypatch)
        orch = _make_orchestrator(
            agentic_enabled=True, agentic_items=["Here is what I found."])
        session = AgenticSearchSession(query="q")
        session.rounds.append(round_data)
        orch._agentic_controller._last_session = session

        results = await _run_submit("do you remember my brother's name?", orch)

        assert _debug_record(results)["mode"] == "agentic-search"
        content = _final_content(results)
        assert content.count(BUDGET_NOTICE.strip()) == 1
        assert content.count(BUDGET_NOTICE_PARTIAL.strip()) == 0
        stored = captured.get("final_output") or ""
        assert stored.count(BUDGET_NOTICE.strip()) == 1
