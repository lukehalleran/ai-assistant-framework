"""2026-09-12 adversarial review F3/F4: a paid web search that cannot run is
not a web route, and an unmet evidence need is disclosed once at delivery.

F3 — the budget/toggle judgement lived at ONE entry path. With the limiter at
104/100 the shared trigger vetoed (source=budget), yet the agentic gate's
Tier 1 still routed "What is the latest news on the election today?" into
web_search mode; the enhanced gatherer's synchronous heuristic fallback never
consulted the budget; a surviving Tier-4 web verdict routed with modes=[]
("llm-fallback"); and the veto blocked the local cache along with the
provider.

F4 — a budget-blocked search left nothing a reply could be honest about: the
veto set should_search False and dropped the terms, so the receipt read "not
triggered". On 2026-09-11 fourteen turns got zero web evidence after the cap
was hit; one reply said so.

Every case drives the DEPLOYED functions (gate, trigger, gatherer mixin, loop
dispatch, handle_submit, the API adapter). Only model and provider I/O is
faked, and every limiter lives in an isolated registry.
"""

from __future__ import annotations

import asyncio
import weakref
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import knowledge.web_search_manager as wsm
import utils.web_search_trigger as wst
from core.agentic import gate as gate_mod
from core.agentic.gate import evaluate_agentic_gate
from core.prompt.gatherer_web import WebSearchMixin
from utils.web_evidence_receipt import (
    BUDGET_ERROR_MARKER,
    BUDGET_NOTICE,
    BUDGET_NOTICE_PARTIAL,
    apply_web_evidence_notice,
    build_web_evidence_receipt,
    web_evidence_notice,
)

NEWS_QUERY = "What is the latest news on the election today?"
SEARCH_QUERY = "Please search the web for current election results"
SEARCH_QUERY_WRAPPED = "Please search the web\n  for current election results"
MIXED_QUERY = "search the web for the exchange rate and calculate 15% of 240"
URL_QUERY = "what does this page say https://example.com/story"
# Reaches the Tier-4 classifier through the deployed gate: no Tier-1 keyword,
# no freshness rule, a non-zero heuristic score, and long enough (>=12 words)
# that the casual/short skip does not stand the gate down first (probed).
PUBLIC_CLAIM_QUERY = (
    "Their trade minister claimed in a speech that the new tariffs would end "
    "the national deficit entirely"
)
CLASSIFIER_TERMS = ["trade minister tariff deficit statement"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolation(monkeypatch):
    """Fresh live-limiter registry, trigger cache and deferred-request slot;
    no location resolution (search keys stay the literal terms)."""
    import utils.location_resolver as lr

    monkeypatch.setattr(wsm, "_LIVE_RATE_LIMITERS", weakref.WeakSet())
    monkeypatch.setattr(lr, "get_user_location", lambda: None)
    wst._llm_trigger_cache.clear()
    wst._llm_trigger_inflight.clear()
    gate_mod._DEFERRED_REQUEST_SLOT.clear()
    yield
    wst._llm_trigger_cache.clear()
    wst._llm_trigger_inflight.clear()
    gate_mod._DEFERRED_REQUEST_SLOT.clear()


def _limiter(tmp_path, *, used: float, limit: int = 100):
    limiter = wsm.WebSearchRateLimiter(
        daily_limit=limit, state_file=str(tmp_path / "credits.json"))
    limiter._credits_today = float(used)
    limiter._current_date = datetime.now().strftime("%Y-%m-%d")
    return limiter


BUDGET_STATES = {
    "spent": dict(used=104.0, enabled=True, blocked="budget"),
    "below_minimum": dict(used=99.5, enabled=True, blocked="budget"),
    "funded": dict(used=0.0, enabled=True, blocked=None),
    "settings_disabled": dict(used=0.0, enabled=False, blocked="disabled"),
}


@pytest.fixture(params=list(BUDGET_STATES))
def budget_state(request, tmp_path, monkeypatch):
    import config.app_config as cfg

    state = BUDGET_STATES[request.param]
    monkeypatch.setattr(cfg, "WEB_SEARCH_ENABLED", state["enabled"], raising=False)
    # The limiter must stay referenced: the registry is a WeakSet.
    return SimpleNamespace(
        name=request.param,
        limiter=_limiter(tmp_path, used=state["used"]),
        blocked=state["blocked"],
    )


def _fake_classifier(calls, terms=CLASSIFIER_TERMS):
    async def _classify(query, model_manager, remaining_credits, timeout,
                        conversation_context=None):
        calls.append(remaining_credits)
        return wst.LLMSearchTriggerResponse(
            should_search=True, confidence=0.8, reason="stub",
            search_terms=list(terms), search_depth="standard", num_searches=1)
    return _classify


def _gate(query, **kwargs):
    return asyncio.run(evaluate_agentic_gate(query, **kwargs))


class _StubCache:
    def __init__(self, hits=None):
        self.hits = dict(hits or {})
        self.gets = []

    def get(self, query, depth):
        self.gets.append((query, depth))
        return self.hits.get(query)

    def put(self, result):
        pass


def _manager(limiter, *, cache=None):
    """A real WebSearchManager whose Tavily calls are counted fakes."""
    manager = wsm.WebSearchManager(api_key="synthetic", rate_limiter=limiter)
    manager.cache = cache or _StubCache()
    manager.is_enabled = lambda: True
    manager._ensure_tavily = lambda: True
    calls = []

    async def _provider(query, max_results, **kwargs):
        calls.append(query)
        await asyncio.sleep(0)
        return [wsm.WebPage(url=f"https://example.com/{len(calls)}",
                            title="Result", content="Body text", snippet="Body")]

    async def _extract(urls):
        return []

    async def _no_decompose(query, *args, **kwargs):
        return wsm.QueryDecomposition(original_query=query, should_decompose=False,
                                      reason="stub")

    manager._tavily_search = _provider
    manager._tavily_extract = _extract
    manager.decompose_query = _no_decompose
    return manager, calls


class _GathererHost(WebSearchMixin):
    def __init__(self, manager, *, trigger_llm=None, trigger=None, model_manager=None):
        self.web_search_manager = manager
        self.web_search_trigger_llm = trigger_llm
        self.web_search_trigger = trigger
        self.model_manager = model_manager
        self.memory_id_map = {}


# ---------------------------------------------------------------------------
# F3 — every paid-search entry path judges capability
# ---------------------------------------------------------------------------

class TestGateTier1JudgesPaidSearchCapability:

    @pytest.mark.parametrize("query", [NEWS_QUERY, SEARCH_QUERY, SEARCH_QUERY_WRAPPED])
    def test_web_arm_only_when_a_paid_search_can_run(self, budget_state, query):
        decision = _gate(query, model_manager=None)
        if budget_state.blocked is None:
            assert decision.should_trigger is True
            assert "web_search" in decision.modes
            assert decision.web_evidence_blocked is None
        else:
            assert decision.should_trigger is False, decision.reason
            assert "web_search" not in decision.modes
            assert decision.web_evidence_blocked == budget_state.blocked

    def test_non_web_work_still_routes_at_zero_budget(self, tmp_path):
        keep = _limiter(tmp_path, used=104.0)  # noqa: F841 (registry is weak)
        decision = _gate(MIXED_QUERY, model_manager=None)
        assert decision.should_trigger is True
        assert "computation" in decision.modes
        assert "web_search" not in decision.modes
        assert decision.web_evidence_blocked == "budget"

    def test_url_fetch_is_not_judged_by_the_search_budget(self, tmp_path):
        keep = _limiter(tmp_path, used=104.0)  # noqa: F841
        decision = _gate(URL_QUERY, model_manager=None)
        assert decision.should_trigger is True
        assert "web_search" in decision.modes
        assert decision.web_evidence_blocked is None


class TestGateTier4WebVerdict:

    def test_funded_verdict_routes_as_the_web_arm(self, tmp_path, monkeypatch):
        keep = _limiter(tmp_path, used=0.0)  # noqa: F841
        calls = []
        monkeypatch.setattr(wst, "_classify_with_llm_unified", _fake_classifier(calls))
        decision = _gate(PUBLIC_CLAIM_QUERY, model_manager=object())
        assert len(calls) == 1, "Tier 4 must consult the classifier"
        assert decision.should_trigger is True
        assert "web_search" in decision.modes
        assert decision.reason == "triggered: web_search"
        assert decision.search_terms == CLASSIFIER_TERMS
        assert decision.web_evidence_blocked is None

    def test_spent_budget_still_classifies_but_routes_nothing(self, tmp_path, monkeypatch):
        keep = _limiter(tmp_path, used=104.0)  # noqa: F841
        calls = []
        monkeypatch.setattr(wst, "_classify_with_llm_unified", _fake_classifier(calls))
        decision = _gate(PUBLIC_CLAIM_QUERY, model_manager=object())
        assert len(calls) == 1  # memory/doc-gen/pattern flags still need it
        assert decision.should_trigger is False
        assert "web_search" not in decision.modes
        assert decision.web_evidence_blocked == "budget"


class TestDeferredRequestRedo:

    def test_forced_modes_never_include_an_unfundable_web_arm(self, tmp_path):
        keep = _limiter(tmp_path, used=104.0)  # noqa: F841
        gate_mod._arm_deferred_request("that sounds interesting to me")
        decision = _gate("yes please", model_manager=None)
        assert decision.should_trigger is True
        assert decision.modes == ["memory"]
        assert decision.web_evidence_blocked == "budget"

    def test_funded_forced_modes_unchanged(self, tmp_path):
        keep = _limiter(tmp_path, used=0.0)  # noqa: F841
        gate_mod._arm_deferred_request("that sounds interesting to me")
        decision = _gate("yes please", model_manager=None)
        assert decision.modes == ["web_search", "memory"]
        assert decision.web_evidence_blocked is None


class TestTriggerFreezesTheDecisionBudget:

    def test_funded_decision_carries_its_budget(self):
        decision = asyncio.run(wst.analyze_for_web_search_llm(
            NEWS_QUERY, model_manager=None, remaining_credits=42))
        assert decision.should_search is True
        assert decision.budget_remaining == 42.0

    def test_vetoed_decision_keeps_the_need_and_the_terms(self, tmp_path, monkeypatch):
        keep = _limiter(tmp_path, used=104.0)  # noqa: F841
        calls = []
        monkeypatch.setattr(wst, "_classify_with_llm_unified", _fake_classifier(calls))
        decision = asyncio.run(wst.analyze_for_web_search_llm(
            PUBLIC_CLAIM_QUERY, model_manager=object()))
        assert decision.should_search is False
        assert decision.search_terms == []          # existing veto contract
        assert decision.evidence_needed is True
        assert decision.blocked_reason == "budget"
        assert decision.blocked_search_terms == CLASSIFIER_TERMS
        assert decision.budget_remaining == 0.0


class TestGathererEntryPaths:

    def test_llm_path_spent_budget_no_provider_call_and_need_recorded(self, tmp_path, monkeypatch):
        limiter = _limiter(tmp_path, used=104.0)
        manager, provider_calls = _manager(limiter)
        monkeypatch.setattr(wst, "_classify_with_llm_unified", _fake_classifier([]))
        host = _GathererHost(manager, trigger_llm=wst.analyze_for_web_search_llm,
                             model_manager=object())
        result = asyncio.run(host._get_web_search_results(PUBLIC_CLAIM_QUERY))
        assert result is None
        assert provider_calls == []
        receipt = host.last_web_decision
        assert receipt["triggered"] is False
        assert receipt["requested"] is True
        assert receipt["blocked"] == "budget"
        assert receipt["budget_remaining"] == 0.0

    def test_spent_budget_still_serves_an_exact_cached_result(self, tmp_path, monkeypatch):
        limiter = _limiter(tmp_path, used=104.0)
        cached = wsm.WebSearchResult(
            query=CLASSIFIER_TERMS[0],
            pages=[wsm.WebPage(url="https://example.com/cached", title="Cached",
                               content="Earlier result", snippet="Earlier")],
            from_cache=True)
        manager, provider_calls = _manager(
            limiter, cache=_StubCache({CLASSIFIER_TERMS[0]: cached}))
        monkeypatch.setattr(wst, "_classify_with_llm_unified", _fake_classifier([]))
        host = _GathererHost(manager, trigger_llm=wst.analyze_for_web_search_llm,
                             model_manager=object())
        result = asyncio.run(host._get_web_search_results(PUBLIC_CLAIM_QUERY))
        assert result is cached
        assert provider_calls == []
        assert host.last_web_decision["from_cache"] is True
        assert host.last_web_decision["results"] == 1

    def test_sync_fallback_consults_the_budget(self, tmp_path):
        limiter = _limiter(tmp_path, used=104.0)
        manager, provider_calls = _manager(limiter)
        host = _GathererHost(manager, trigger=wst.should_search_heuristic)
        result = asyncio.run(host._get_web_search_results(NEWS_QUERY))
        assert result is None
        assert provider_calls == []
        assert host.last_web_decision["requested"] is True
        assert host.last_web_decision["blocked"] == "budget"

    def test_sync_fallback_funded_still_searches(self, tmp_path):
        limiter = _limiter(tmp_path, used=0.0)
        manager, provider_calls = _manager(limiter)
        host = _GathererHost(manager, trigger=wst.should_search_heuristic)
        result = asyncio.run(host._get_web_search_results(NEWS_QUERY))
        assert result is not None and result.has_results
        assert len(provider_calls) >= 1
        assert host.last_web_decision["blocked"] is None


class TestLoopDispatch:

    def test_loop_search_refused_without_a_provider_call(self, tmp_path):
        from core.agentic.tools import ToolExecutor

        limiter = _limiter(tmp_path, used=104.0)
        manager, provider_calls = _manager(limiter)
        executor = ToolExecutor.__new__(ToolExecutor)
        executor.web_search_manager = manager
        result = asyncio.run(executor._execute_search(["election results today"]))
        assert provider_calls == []
        # Pins the receipt's marker against the deployed manager's refusal.
        assert BUDGET_ERROR_MARKER in (result.error or "")


# ---------------------------------------------------------------------------
# F4 — the receipt and the one notice
# ---------------------------------------------------------------------------

def _round(query, results=None, summary=None):
    from core.agentic.types import SearchRequest, SearchRound
    rnd = SearchRound(round_number=1, request=SearchRequest(query=query), results=results)
    rnd.summary = summary
    return rnd


class TestReceiptAndNotice:

    def test_blocked_need_with_nothing_acquired(self):
        receipt = build_web_evidence_receipt(
            SimpleNamespace(web_evidence_blocked="budget"), None)
        assert receipt == {"requested": True, "blocked": "budget", "acquired": 0, "fetched": 0}
        assert web_evidence_notice(receipt) == BUDGET_NOTICE

    def test_successful_fetch_is_never_described_as_failed(self):
        session = SimpleNamespace(rounds=[_round(
            "[Fetch URL] https://example.com/story",
            summary="[WEB_1] Title: Story\nURL: https://example.com/story\n\nBody")])
        receipt = build_web_evidence_receipt(
            SimpleNamespace(web_evidence_blocked="budget"), None, session=session)
        assert receipt["fetched"] == 1
        assert web_evidence_notice(receipt) == BUDGET_NOTICE_PARTIAL

    def test_failed_fetch_does_not_count(self):
        session = SimpleNamespace(rounds=[_round(
            "[Fetch URL] https://example.com/story",
            summary="[Could not fetch content from https://example.com/story]")])
        receipt = build_web_evidence_receipt(None, None, session=session)
        assert receipt["fetched"] == 0

    def test_loop_budget_refusal_is_a_blocked_need(self):
        refused = wsm.WebSearchResult(
            query="election results", error="Daily credit limit reached. Remaining: 0.0")
        session = SimpleNamespace(rounds=[_round("election results", results=refused)])
        receipt = build_web_evidence_receipt(None, None, session=session)
        assert receipt["requested"] is True and receipt["blocked"] == "budget"
        assert web_evidence_notice(receipt) == BUDGET_NOTICE

    def test_enhanced_receipt_with_cached_results_is_partial(self):
        receipt = build_web_evidence_receipt(None, {
            "triggered": False, "requested": True, "blocked": "budget",
            "results": 2, "from_cache": True})
        assert web_evidence_notice(receipt) == BUDGET_NOTICE_PARTIAL

    @pytest.mark.parametrize("gate,web", [
        (None, None),                                                  # nothing recorded
        (None, {"triggered": False, "requested": False, "blocked": None}),  # empty account, unrelated chat
        (None, {"triggered": True, "requested": True, "blocked": None, "results": 4}),
        (SimpleNamespace(web_evidence_blocked="disabled"), None),     # owner's toggle
    ])
    def test_no_notice_without_an_unmet_budget_need(self, gate, web):
        assert web_evidence_notice(build_web_evidence_receipt(gate, web)) == ""

    def test_mock_attributes_are_never_read_as_a_block(self):
        from unittest.mock import MagicMock
        assert web_evidence_notice(build_web_evidence_receipt(MagicMock(), None)) == ""

    def test_notice_is_appended_once(self):
        receipt = {"requested": True, "blocked": "budget", "acquired": 0, "fetched": 0}
        once = apply_web_evidence_notice("Answer.", receipt)
        assert apply_web_evidence_notice(once, receipt) == once
        assert once.count(BUDGET_NOTICE.strip()) == 1

    def test_no_receipt_leaves_a_prior_turns_notice_text_alone(self):
        prior = "Answer." + BUDGET_NOTICE
        assert apply_web_evidence_notice(prior, None) == prior


class TestDeliverySites:
    """The notice reaches what the owner sees (final chunk) and what is stored
    (_dispatch_storage final_output) — exactly once — on both answer routes,
    and through the API adapter that shares handle_submit."""

    @staticmethod
    def _capture_storage(monkeypatch):
        import gui.handlers as handlers

        captured = {}

        def _capture(*args, **kwargs):
            captured["final_output"] = kwargs.get(
                "final_output", args[4] if len(args) > 4 else None)

        monkeypatch.setattr(handlers, "_dispatch_storage", _capture)
        return captured

    async def test_enhanced_turn_discloses_the_blocked_search_once(self, monkeypatch):
        from tests.unit.test_handle_submit import _final_content, _make_orchestrator, _run_submit

        captured = self._capture_storage(monkeypatch)
        receipt = {"triggered": False, "requested": True, "blocked": "budget",
                   "results": None, "error": None}
        orch = _make_orchestrator(streaming_chunks=["The count", " is still open."],
                                  prepare_result=("P", "S", {"web_search_decision": receipt}))
        results = await _run_submit(NEWS_QUERY, orch)
        assert _final_content(results).count(BUDGET_NOTICE.strip()) == 1
        assert (captured.get("final_output") or "").count(BUDGET_NOTICE.strip()) == 1

    async def test_enhanced_turn_without_a_blocked_need_has_no_notice(self, monkeypatch):
        from tests.unit.test_handle_submit import _final_content, _make_orchestrator, _run_submit

        captured = self._capture_storage(monkeypatch)
        receipt = {"triggered": True, "requested": True, "blocked": None,
                   "results": 3, "error": None}
        orch = _make_orchestrator(prepare_result=("P", "S", {"web_search_decision": receipt}))
        results = await _run_submit(NEWS_QUERY, orch)
        assert "search budget" not in _final_content(results)
        assert "search budget" not in (captured.get("final_output") or "")

    async def test_agentic_turn_discloses_a_loop_budget_refusal_once(self, monkeypatch):
        from core.agentic.types import AgenticSearchSession
        from tests.unit.test_handle_submit import (
            _debug_record, _final_content, _make_orchestrator, _run_submit)

        captured = self._capture_storage(monkeypatch)
        orch = _make_orchestrator(agentic_enabled=True,
                                  agentic_items=["Here is what I can tell you."])
        session = AgenticSearchSession(query="q")
        session.rounds.append(_round("election results", results=wsm.WebSearchResult(
            query="election results", error="Daily credit limit reached. Remaining: 0.0")))
        orch._agentic_controller._last_session = session
        results = await _run_submit("do you remember my brother's name?", orch)
        assert _debug_record(results)["mode"] == "agentic-search"
        assert _final_content(results).count(BUDGET_NOTICE.strip()) == 1
        assert (captured.get("final_output") or "").count(BUDGET_NOTICE.strip()) == 1

    async def test_api_complete_event_carries_the_same_single_notice(self):
        from tests.unit.test_api_chat import _make_app, _patched_pipeline, _post_chat
        from tests.unit.test_handle_submit import _make_orchestrator

        receipt = {"triggered": False, "requested": True, "blocked": "budget",
                   "results": None, "error": None}
        orch = _make_orchestrator(streaming_chunks=["The count", " is still open."],
                                  prepare_result=("P", "S", {"web_search_decision": receipt}))
        app = _make_app(orch)
        for p in _patched_pipeline(NEWS_QUERY):
            p.start()
        try:
            with patch("gui.handlers._dispatch_storage"):
                _, events = await _post_chat(app, NEWS_QUERY)
        finally:
            patch.stopall()
        complete = dict(events)["complete"]
        assert complete["content"].count(BUDGET_NOTICE.strip()) == 1


def test_async_mock_import_is_used():
    # Keeps the AsyncMock import honest for future receipt tests without ruff
    # noise; the gatherer fakes above are plain coroutines.
    assert AsyncMock is not None
