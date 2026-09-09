"""2026-09-09 audit repairs, batch B1 — live controls (F04, F10, F06).

Every test initialises the REAL consumer once, observes it, calls the deployed
Settings setter (or executes the deployed calendar action), then uses the SAME
instance again. Asserting on config dicts alone is exactly what let these
defects ship (docs/HANDOFF_20260909_independent_bug_audit.md).
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import config.app_config as cfg
from core.prompt import gatherer_web
from core.prompt.gatherer_web import WebSearchMixin
from gui.settings_core import apply_streaming, apply_web_search
from knowledge.web_search_manager import WebSearchManager, WebSearchRateLimiter


def _run(coro):
    return asyncio.run(coro)


_OK_SAVE = lambda updater: (True, None)  # noqa: E731 — persistence stub


# ---------------------------------------------------------------------------
# F04 — web search toggle + credit limit reach the running consumer
# ---------------------------------------------------------------------------

def _recording_gatherer(limiter, calls):
    async def search(**kwargs):
        calls.append(kwargs["query"])
        return NS(has_results=False, error="synthetic provider; no network")

    decision = NS(should_search=True, depth=NS(value="quick"), confidence=1.0,
                  reason="synthetic", search_terms=[])
    manager = NS(is_available=lambda: True, multi_search=search, rate_limiter=limiter)
    # No lazy property on the double: the setter must find the manager anyway.
    return NS(web_search_manager=manager, web_search_trigger_llm=None,
              web_search_trigger=lambda q: decision, model_manager=None,
              memory_id_map={})


class TestWebSearchSetterReachesLiveConsumer:
    @pytest.fixture(autouse=True)
    def _restore_flags(self, monkeypatch):
        monkeypatch.setattr(cfg, "WEB_SEARCH_ENABLED", True)
        monkeypatch.setattr(cfg, "WEB_SEARCH_DAILY_CREDIT_LIMIT", 100)

    def test_disable_then_enable_on_the_same_gatherer(self, tmp_path):
        calls: list = []
        limiter = WebSearchRateLimiter(daily_limit=100,
                                       state_file=str(tmp_path / "credits.json"))
        gatherer = _recording_gatherer(limiter, calls)
        orch = NS(config={"features": {}, "web_search": {}},
                  prompt_builder=NS(context_gatherer=gatherer))

        _run(WebSearchMixin._get_web_search_results(gatherer, "Synthetic current query"))
        assert calls == ["Synthetic current query"], "baseline: enabled search dispatches"

        resp = apply_web_search(orch, enabled=False, daily_credit_limit=3, save=_OK_SAVE)
        assert resp["ok"]
        _run(WebSearchMixin._get_web_search_results(gatherer, "Synthetic second query"))
        assert calls == ["Synthetic current query"], "disabled: no provider call on the SAME instance"

        resp = apply_web_search(orch, enabled=True, daily_credit_limit=50, save=_OK_SAVE)
        assert resp["ok"]
        _run(WebSearchMixin._get_web_search_results(gatherer, "Synthetic third query"))
        assert calls == ["Synthetic current query", "Synthetic third query"]

    def test_credit_limit_updates_the_existing_limiter(self, tmp_path):
        limiter = WebSearchRateLimiter(daily_limit=100,
                                       state_file=str(tmp_path / "credits.json"))
        gatherer = _recording_gatherer(limiter, [])
        orch = NS(config={"features": {}, "web_search": {}},
                  prompt_builder=NS(context_gatherer=gatherer))
        assert limiter.can_search(4)

        apply_web_search(orch, enabled=True, daily_credit_limit=3, save=_OK_SAVE)
        assert limiter.daily_limit == 3
        assert not limiter.can_search(4), "the live limiter enforces the new cap"
        assert limiter.can_search(3)

    def test_agentic_controller_manager_is_updated_without_building_one(self, tmp_path):
        limiter_a = WebSearchRateLimiter(daily_limit=100, state_file=str(tmp_path / "a.json"))
        limiter_b = WebSearchRateLimiter(daily_limit=100, state_file=str(tmp_path / "b.json"))
        gatherer = _recording_gatherer(limiter_a, [])
        ctrl = NS(web_search_manager=NS(rate_limiter=limiter_b))
        built = []

        class Orch:
            config = {"features": {}, "web_search": {}}
            prompt_builder = NS(context_gatherer=gatherer)
            _agentic_controller = ctrl

            @property
            def agentic_controller(self):  # the public property BUILDS one
                built.append(True)
                return ctrl

        apply_web_search(Orch(), enabled=True, daily_credit_limit=7, save=_OK_SAVE)
        assert limiter_a.daily_limit == 7 and limiter_b.daily_limit == 7
        assert built == [], "the setter must not trigger lazy controller construction"

    def test_lazy_manager_created_after_toggle_uses_live_limit(self, monkeypatch):
        from core.prompt.context_gatherer import ContextGatherer

        gatherer = ContextGatherer.__new__(ContextGatherer)
        gatherer._web_search_manager = None
        orch = NS(config={"features": {}, "web_search": {}},
                  prompt_builder=NS(context_gatherer=gatherer))
        apply_web_search(orch, enabled=True, daily_credit_limit=9, save=_OK_SAVE)
        assert gatherer._web_search_manager is None, "setter never builds the manager"

        manager = gatherer.web_search_manager  # deployed lazy property
        assert manager is not None
        assert manager.rate_limiter.daily_limit == 9

    def test_manager_availability_honours_the_live_toggle(self, monkeypatch):
        manager = WebSearchManager.__new__(WebSearchManager)
        manager.api_key = "synthetic-key"
        manager._api_key_invalid = False
        monkeypatch.setattr(manager, "_ensure_tavily", lambda: True, raising=False)
        assert manager.is_available()
        monkeypatch.setattr(cfg, "WEB_SEARCH_ENABLED", False)
        assert not manager.is_available(), "the shared chokepoint reports disabled"
        assert not WebSearchManager.is_enabled()

    def test_tool_health_reports_disabled_not_broken(self, monkeypatch):
        from core.agentic.tools import ToolExecutor

        manager = WebSearchManager.__new__(WebSearchManager)
        manager.api_key = "synthetic-key"
        manager._api_key_invalid = False
        monkeypatch.setattr(manager, "_ensure_tavily", lambda: True, raising=False)
        executor = ToolExecutor.__new__(ToolExecutor)
        executor.web_search_manager = manager
        for attr in ("wolfram_manager", "sandbox_manager", "file_access_manager",
                     "git_stats_manager", "github_manager", "memory_expander",
                     "chroma_store"):
            setattr(executor, attr, None)
        assert "web_search: AVAILABLE" in executor.get_tool_health()
        monkeypatch.setattr(cfg, "WEB_SEARCH_ENABLED", False)
        health = executor.get_tool_health()
        assert "web_search: DISABLED" in health
        assert "no API key" not in health.split("\n")[0]

    def test_search_entry_points_refuse_when_disabled(self, monkeypatch, tmp_path):
        """search() and multi_search() — the paths the agentic loop and
        instrument callers use directly — never reach the provider adapter
        when Settings has search off, and never serve a cached result."""
        from knowledge.web_search_manager import DISABLED_ERROR, WebSearchDepth

        manager = WebSearchManager.__new__(WebSearchManager)
        manager.api_key = "synthetic-key"
        manager._api_key_invalid = False
        manager.rate_limiter = WebSearchRateLimiter(daily_limit=100,
                                                    state_file=str(tmp_path / "c.json"))
        manager.default_timeout = 5.0
        adapter_calls: list = []

        from knowledge.web_search_manager import WebSearchResult

        async def fake_execute(query, depth, max_results, **kw):
            adapter_calls.append(query)
            return WebSearchResult(query=query, search_depth=depth)

        cache_hits: list = []
        manager.cache = NS(get=lambda q, d: cache_hits.append(q) or None,
                           set=lambda *a, **k: None)
        monkeypatch.setattr(manager, "_ensure_tavily", lambda: True, raising=False)
        monkeypatch.setattr(manager, "_execute_search", fake_execute, raising=False)
        monkeypatch.setattr(manager, "_localize_query", lambda q: q, raising=False)
        monkeypatch.setattr(cfg, "WEB_SEARCH_ENABLED", False)

        res = _run(manager.search("Synthetic direct query", depth=WebSearchDepth.QUICK))
        assert res.error == DISABLED_ERROR and not res.has_results
        multi = _run(manager.multi_search("Synthetic multi query", depth=WebSearchDepth.QUICK,
                                          auto_decompose=False,
                                          sub_queries=["Synthetic sub a", "Synthetic sub b"]))
        assert multi.error == DISABLED_ERROR
        assert adapter_calls == [], "no provider-adapter call through either entry point"
        assert cache_hits == [], "disabled search is decided before the cache"

        monkeypatch.setattr(cfg, "WEB_SEARCH_ENABLED", True)
        res = _run(manager.search("Synthetic direct query", depth=WebSearchDepth.QUICK))
        assert res.error is None and adapter_calls == ["Synthetic direct query"]

    def test_should_trigger_reads_live_value(self, monkeypatch):
        gatherer = NS(web_search_trigger=lambda q: NS(should_search=True))
        assert WebSearchMixin.should_trigger_web_search(gatherer, "any query")
        monkeypatch.setattr(cfg, "WEB_SEARCH_ENABLED", False)
        assert not WebSearchMixin.should_trigger_web_search(gatherer, "any query")
        assert gatherer_web.WEB_SEARCH_ENABLED is True, "module binding untouched; live read wins"


# ---------------------------------------------------------------------------
# F10 — query-rewrite toggle reaches the initialised ContextPipeline
# ---------------------------------------------------------------------------

def _pipeline(enable: bool):
    from core.context_pipeline import ContextPipeline

    with patch.object(ContextPipeline, "_load_persisted_tone", return_value=None):
        return ContextPipeline(model_manager=None, topic_manager=None,
                               config={"enable_query_rewrite": enable, "REWRITE_TIMEOUT_S": 1})


class TestQueryRewriteToggleReachesPipeline:
    def _apply(self, orch, disable):
        return apply_streaming(orch, disable_best_of=True, disable_query_rewrite=disable,
                               disable_llm_summaries=True, best_of_latency_budget_s=0,
                               save=_OK_SAVE)

    def test_disable_on_enabled_pipeline(self):
        cp = _pipeline(True)
        orch = NS(config={"features": {}}, context_pipeline=cp, prompt_builder=None)
        assert cp._enable_query_rewrite is True
        resp = self._apply(orch, disable=True)
        assert resp["ok"] and orch.config["features"]["enable_query_rewrite"] is False
        assert cp._enable_query_rewrite is False, "the RUNNING pipeline stops rewriting"
        assert cp.config["enable_query_rewrite"] is False

    def test_enable_on_disabled_pipeline(self):
        cp = _pipeline(False)
        orch = NS(config={"features": {}}, context_pipeline=cp, prompt_builder=None)
        assert cp._enable_query_rewrite is False
        self._apply(orch, disable=False)
        assert cp._enable_query_rewrite is True

    def test_build_stops_and_resumes_rewriting_on_the_same_pipeline(self, monkeypatch):
        """Drive the DEPLOYED ContextPipeline.build() enabled → disabled →
        enabled on one instance. Unrelated stages (topics/tone/heavy-topic/
        STM/identity/thread) are stubbed on the instance; the rewrite
        decision itself is the deployed code path."""
        from core.context_pipeline import ToneLevel

        cp = _pipeline(True)
        orch = NS(config={"features": {}}, context_pipeline=cp, prompt_builder=None)
        invoked: list = []

        async def fake_rewrite(user_input, query_analysis):
            invoked.append(user_input)
            return "REWRITTEN: " + user_input

        async def topics(user_input, last_exchange=None):
            return "general", ["general"]

        async def tone(user_input, history):
            return ToneLevel.CONVERSATIONAL, None

        async def heavy(user_input, topics_):
            return False, [], None

        async def thread():
            return None

        monkeypatch.setattr(cp, "_rewrite_query", fake_rewrite)
        monkeypatch.setattr(cp, "_extract_topics", topics)
        monkeypatch.setattr(cp, "_detect_tone", tone)
        monkeypatch.setattr(cp, "_check_heavy_topics", heavy)
        monkeypatch.setattr(cp, "_should_run_stm", lambda history: False)
        monkeypatch.setattr(cp, "_get_identity_context", lambda: ("", None))
        monkeypatch.setattr(cp, "_get_thread_context", thread)
        monkeypatch.setattr(cp, "_get_tone_instructions", lambda level: "")
        monkeypatch.setattr(cp, "_intent_classifier", None)
        # ≥12 words: below that a CONVERSATIONAL turn skips the heavy stage and
        # the rewrite gate requires ≥10 — keep the query well inside both.
        query = ("please explain in detail how the memory gate scores its candidate "
                 "documents before the reranker runs today")
        assert len(query.split()) >= 12

        r1 = _run(cp.build(query))
        assert invoked == [query]
        assert r1.processed_query.startswith("REWRITTEN: ")

        assert self._apply(orch, disable=True)["ok"]
        r2 = _run(cp.build(query))
        assert invoked == [query], "disabled: build() must not invoke the rewriter"
        assert r2.processed_query == query

        assert self._apply(orch, disable=False)["ok"]
        r3 = _run(cp.build(query))
        assert invoked == [query, query], "re-enabled: build() rewrites again"
        assert r3.processed_query.startswith("REWRITTEN: ")

    def test_orchestrator_without_pipeline_still_ok(self):
        orch = NS(config={"features": {}}, prompt_builder=None)
        assert self._apply(orch, disable=True)["ok"]


# ---------------------------------------------------------------------------
# F06 — calendar mutations invalidate the [UPCOMING SCHEDULE] read cache
# ---------------------------------------------------------------------------

from core.actions import google_calendar as gcal  # noqa: E402
from core.actions.google_calendar_create import create_calendar_event  # noqa: E402
from core.actions.google_calendar_modify import (  # noqa: E402
    delete_calendar_event, update_calendar_event,
)
from core.actions.types import ActionProposal, ActionType  # noqa: E402

_EVENT = {"id": "ev1", "summary": "Synthetic meeting",
          "start": {"dateTime": "2026-09-09T12:00:00-05:00"},
          "end": {"dateTime": "2026-09-09T13:00:00-05:00"}}


def _resp(status=200, payload=None):
    r = MagicMock(); r.status_code = status
    r.json.return_value = payload if payload is not None else {}
    r.text = ""
    return r


def _client(list_payload=None, patch_resp=None, delete_resp=None, post_side=None):
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=_resp(payload=list_payload or {"items": [_EVENT]}))
    client.patch = AsyncMock(return_value=patch_resp or _resp(payload=_EVENT))
    client.delete = AsyncMock(return_value=delete_resp or _resp(status=204))
    if post_side is not None:
        client.post = AsyncMock(side_effect=post_side)
    return client


def _auth_ok():
    auth = MagicMock()
    auth.is_authenticated = True
    auth.token_expired_no_refresh = False
    auth.has_scope.return_value = True
    auth.get_credentials.return_value = NS(token="tok")
    return auth


def _warm_cache():
    gcal._cache = [{"summary": "Synthetic meeting", "start": "2026-09-09T12:00:00-05:00",
                    "end": "2026-09-09T13:00:00-05:00", "all_day": False, "location": ""}]
    gcal._cache_ts = time.time()


class TestCalendarMutationsInvalidateCache:
    @pytest.fixture(autouse=True)
    def _isolate_cache(self):
        gcal.clear_cache()
        yield
        gcal.clear_cache()

    def _exec(self, fn, proposal, client):
        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=_auth_ok()), \
             patch("httpx.AsyncClient", return_value=client):
            return _run(fn(proposal))

    def test_delete_success_drops_cache(self):
        _warm_cache()
        p = ActionProposal(action_type=ActionType.CALENDAR_DELETE_EVENT, summary="t",
                           params={"summary": "Synthetic meeting", "date": "2026-09-09"})
        result = self._exec(delete_calendar_event, p, _client())
        assert result.success, result.message
        assert gcal._cache is None, "deleted event must not survive in the read cache"

    def test_delete_already_gone_drops_cache(self):
        _warm_cache()
        p = ActionProposal(action_type=ActionType.CALENDAR_DELETE_EVENT, summary="t",
                           params={"summary": "Synthetic meeting", "date": "2026-09-09"})
        result = self._exec(delete_calendar_event, p, _client(delete_resp=_resp(status=410)))
        assert result.success
        assert gcal._cache is None

    def test_delete_failure_keeps_cache(self):
        _warm_cache()
        p = ActionProposal(action_type=ActionType.CALENDAR_DELETE_EVENT, summary="t",
                           params={"summary": "Synthetic meeting", "date": "2026-09-09"})
        result = self._exec(delete_calendar_event, p, _client(delete_resp=_resp(status=500)))
        assert not result.success
        assert gcal._cache is not None, "a failed mutation changes nothing; cache stays"

    def test_update_success_drops_cache(self):
        _warm_cache()
        p = ActionProposal(action_type=ActionType.CALENDAR_UPDATE_EVENT, summary="t",
                           params={"summary": "Synthetic meeting", "date": "2026-09-09",
                                   "new_start_time": "14:00", "new_end_time": "15:00"})
        result = self._exec(update_calendar_event, p, _client())
        assert result.success, result.message
        assert gcal._cache is None

    def test_next_read_refetches_after_delete(self):
        """read → delete → read on the SAME cache: the second read hits the API."""
        _warm_cache()
        p = ActionProposal(action_type=ActionType.CALENDAR_DELETE_EVENT, summary="t",
                           params={"summary": "Synthetic meeting", "date": "2026-09-09"})
        assert self._exec(delete_calendar_event, p, _client()).success
        read_client = _client(list_payload={"items": []})
        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("config.app_config.GOOGLE_CLIENT_ID", "synthetic-id", create=True), \
             patch("core.actions.google_auth.get_google_auth", return_value=_auth_ok()), \
             patch("httpx.AsyncClient", return_value=read_client):
            events = _run(gcal.fetch_upcoming_events())
        assert read_client.get.call_count == 1, "cache miss → real GET"
        assert events == []

    def _create_proposal(self, n):
        events = [{"summary": f"Synthetic event {i}",
                   "start_time": f"2026-09-10T{10 + i:02d}:00:00-05:00",
                   "end_time": f"2026-09-10T{11 + i:02d}:00:00-05:00",
                   "time_zone": "America/Chicago"} for i in range(n)]
        return ActionProposal(action_type=ActionType.CALENDAR_CREATE_EVENT, summary="t",
                              params={"events": events})

    def test_partial_create_drops_cache(self):
        _warm_cache()
        ok = _resp(200, {"id": "e1", "htmlLink": "https://calendar.google.com/x"})
        bad = _resp(500, {}); bad.text = "boom"
        result = self._exec(create_calendar_event, self._create_proposal(2),
                            _client(post_side=[ok, bad]))
        assert not result.success and "Created 1 of 2" in result.message
        assert gcal._cache is None, "one created event stales the schedule"

    def test_failure_only_create_keeps_cache(self):
        _warm_cache()
        bad = _resp(500, {}); bad.text = "boom"
        result = self._exec(create_calendar_event, self._create_proposal(2),
                            _client(post_side=[bad, bad]))
        assert not result.success
        assert gcal._cache is not None

    def test_full_create_still_drops_cache(self):
        _warm_cache()
        ok = _resp(200, {"id": "e1", "htmlLink": "https://calendar.google.com/x"})
        result = self._exec(create_calendar_event, self._create_proposal(1),
                            _client(post_side=[ok]))
        assert result.success, result.message
        assert gcal._cache is None
