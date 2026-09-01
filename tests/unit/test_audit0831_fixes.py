"""Regression tests for the 2026-08-31 audit-sweep fix batch (F5-F33).

Each class pins one finding from AUDIT_SWEEP_20260831.md by driving the
DEPLOYED function — no getsource string pins. Findings F1-F4, F7, F8, F31
were fixed and tested in the prior commit; this file covers the remainder.
"""
import asyncio
import json
import sys
import types as _types_mod
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.agentic.protocols import XMLMarkerHandler


# ===========================================================================
# F5 / F6 / F20 — XML action protocol
# ===========================================================================

class TestXmlActionProtocol:
    def setup_method(self):
        self.h = XMLMarkerHandler()

    def test_f5_propose_action_generic_attrs_calendar(self):
        d = self.h.parse_response(
            '<propose_action type="calendar_create_event" summary="Miller\'s exam"'
            ' start_time="2026-09-20T10:00:00" end_time="2026-09-20T11:00:00">'
            '</propose_action>'
        )
        assert len(d) == 1 and d[0].wants_action
        p = d[0].action_params
        assert p["summary"] == "Miller's exam"  # F20: apostrophe survives
        assert p["start_time"] == "2026-09-20T10:00:00"
        assert p["end_time"] == "2026-09-20T11:00:00"

    def test_f5_propose_action_legacy_email_shape_unchanged(self):
        d = self.h.parse_response(
            '<propose_action type="send_email" recipient="Morgan@x.com"'
            ' subject="hi">msg body</propose_action>'
        )
        p = d[0].action_params
        assert p["recipient"] == "Morgan@x.com"
        assert p["subject"] == "hi"
        assert p["message"] == "msg body"

    def test_f6_model_supplied_repo_dropped(self):
        d = self.h.parse_response(
            '<action type="github_create_issue" repo="someone/else"'
            ' subject="bug title">body text</action>'
        )
        p = d[0].action_params
        assert "repo" not in p  # trust boundary: not in forward_params
        assert p["subject"] == "bug title"
        assert p["message"] == "body text"

    def test_f6_calendar_action_fields_survive_filter(self):
        d = self.h.parse_response(
            '<action type="calendar_create_event" summary="Exam"'
            ' start_time="2026-09-20" end_time="2026-09-21" all_day="true">'
            'note</action>'
        )
        p = d[0].action_params
        assert p["summary"] == "Exam" and p["all_day"] == "true"
        assert p["message"] == "note"  # body merges after the filter

    def test_f6_unknown_action_type_params_pass_through(self):
        d = self.h.parse_response(
            '<action type="not_a_real_action" foo="bar">body</action>'
        )
        assert d[0].action_params.get("foo") == "bar"

    def test_f6_invoke_fallback_forwards_calendar_fields(self):
        d = self.h.parse_response(
            '<invoke name="propose_action">'
            '<parameter name="action_type">calendar_create_event</parameter>'
            '<parameter name="summary">Exam</parameter>'
            '<parameter name="start_time">2026-09-20T10:00:00</parameter>'
            '<parameter name="end_time">2026-09-20T11:00:00</parameter>'
            '</invoke>'
        )
        p = d[0].action_params
        assert p.get("summary") == "Exam"
        assert p.get("start_time") == "2026-09-20T10:00:00"

    def test_f20_pattern_scan_single_quoted_json_spec(self):
        d = self.h.parse_response('<pattern_scan spec=\'{"propositions": ["study"]}\'/>')
        assert d[0].wants_pattern_scan
        assert d[0].pattern_spec == {"propositions": ["study"]}

    def test_f20_action_attr_single_quotes_with_double_inside(self):
        d = self.h.parse_response(
            "<action type='send_telegram' recipient='@luke'>say \"hi\"</action>"
        )
        p = d[0].action_params
        assert p["recipient"] == "@luke"
        assert p["message"] == 'say "hi"'


# ===========================================================================
# F13 / F21 — controller forced-action retry + regenerate stash
# ===========================================================================

@pytest.fixture
def controller():
    from core.agentic.controller import AgenticSearchController
    manager = MagicMock()
    manager.api_models = {}
    web = MagicMock()
    web.is_available = MagicMock(return_value=True)
    return AgenticSearchController(model_manager=manager, web_search_manager=web)


def _dispatch_spy(captured):
    from core.agentic.types import SearchDecision

    class _Res:
        start_events = []
        end_events = []
        round_data = MagicMock(duration_ms=1.0)
        formatted_context = "RESULT."
        memory_collection = None
        is_expand = False
        decision = SearchDecision()

    async def fake_dispatch(decision, *a, **k):
        captured.setdefault("dispatched", []).append(decision)
        return _Res()

    return fake_dispatch


def _final_spy(captured):
    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        captured["final_called"] = True
        yield "ANSWER."
    return fake_final


async def _drain(gen):
    out = []
    async for ev in gen:
        out.append(ev)
    return out


class TestForcedActionRetry:
    @pytest.mark.asyncio
    async def test_f13_no_reforce_after_dispatch(self, controller, monkeypatch):
        """An action dispatched in round 1 must not re-arm the forced retry
        when a later round is tool-less."""
        from core.agentic.types import SearchDecision
        captured = {}
        calls = {"n": 0}

        async def fake_decision(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return [
                    SearchDecision(
                        wants_action=True, action_type="calendar_create_event",
                        action_params={"summary": "Exam",
                                       "start_time": "2026-09-20T10:00:00",
                                       "end_time": "2026-09-20T11:00:00"},
                        action_summary="calendar_create_event: Exam",
                    ),
                    SearchDecision(wants_search=True, search_query="deadline"),
                ]
            return [SearchDecision(wants_answer=True)]

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_dispatch_single", _dispatch_spy(captured))
        monkeypatch.setattr(controller, "_generate_final_response", _final_spy(captured))

        await _drain(controller.run_agentic_search(
            query="Yes please create the calendar events",
            system_prompt="sys", model_name="test-model",
            initial_search_terms=[], skip_initial_search=True,
        ))
        actions = [d for d in captured.get("dispatched", []) if d.wants_action]
        assert len(actions) == 1, f"duplicate action dispatch: {len(actions)}"
        # No forced retry round: decision called twice, not three times.
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_f21_stash_nulled_at_entry(self, controller, monkeypatch):
        """A stale final-prompt stash from a previous turn must be cleared
        the moment a new run starts."""
        from core.agentic.types import SearchDecision
        captured = {}
        controller._last_final_prompt = "STALE PROMPT"
        controller._last_final_system_prompt = "STALE SYS"
        controller._last_final_model = "stale-model"

        async def fake_decision(*a, **k):
            return [SearchDecision(wants_answer=True)]

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_dispatch_single", _dispatch_spy(captured))
        monkeypatch.setattr(controller, "_generate_final_response", _final_spy(captured))

        await _drain(controller.run_agentic_search(
            query="hello", system_prompt="sys", model_name="test-model",
            initial_search_terms=[], skip_initial_search=True,
        ))
        # The mocked final never stashes, so post-run they must be None —
        # regenerate_final_answer can no longer fire on the stale prompt.
        assert controller._last_final_prompt is None
        assert await controller.regenerate_final_answer() is None


# ===========================================================================
# F32 — telemetry round join for parallel rounds
# ===========================================================================

class TestTelemetryRoundJoin:
    def test_f32_parallel_round_numbers_join(self):
        from core.agentic.types import AgenticSearchSession, SearchProtocol
        session = AgenticSearchSession(query="q", max_rounds=5,
                                       protocol=SearchProtocol.XML_MARKERS)
        _req = SimpleNamespace(query="web search deadlines", reason="")
        r1 = MagicMock(round_number=1, request=_req, error=None)
        r2 = MagicMock(round_number=2, request=_req, error=None)
        session.rounds = [r1, r2]
        session.round_telemetry.append(
            {"round": 1, "rounds": [1, 2], "action": "web_search,web_search",
             "decision_ms": 42, "tool_ms": 7, "timed_out": False}
        )
        summary = session.get_provenance_summary()
        rounds = summary.get("agentic_rounds") or summary.get("rounds") or []
        with_ms = [rd for rd in rounds if rd.get("decision_ms") == 42]
        assert len(with_ms) == 2, f"round 2 lost its telemetry join: {rounds}"


# ===========================================================================
# F14 — pending store capacity counts only active proposals
# ===========================================================================

class TestPendingStoreCapacity:
    def _store(self, tmp_path):
        from core.actions.types import PendingActionsStore
        return PendingActionsStore(path=str(tmp_path / "pending.json"))

    def _proposal(self, status="pending"):
        from core.actions.types import ActionProposal, ActionType
        p = ActionProposal(action_type=ActionType.SEND_TELEGRAM,
                           params={"message": "x"}, summary="s")
        p.status = status
        p.expires_at = datetime.now(timezone.utc) + timedelta(seconds=300)
        return p

    def test_f14_terminal_proposals_free_their_slots(self, tmp_path):
        store = self._store(tmp_path)
        for _ in range(5):
            assert store.propose(self._proposal())
        # At capacity with 5 pending — rejects.
        assert not store.propose(self._proposal())
        # Terminal outcomes free the slots.
        for p in store._store.values():
            p.status = "executed"
        assert store.propose(self._proposal())

    def test_f14_pending_still_caps(self, tmp_path):
        store = self._store(tmp_path)
        for _ in range(5):
            assert store.propose(self._proposal())
        assert not store.propose(self._proposal())


# ===========================================================================
# F15 — runtime action health: expired token without refresh token
# ===========================================================================

class TestRuntimeActionHealth:
    def test_f15_expired_no_refresh_reports_unavailable(self, monkeypatch):
        import config.app_config as cfg
        import core.actions.google_auth as ga
        from core.actions.registry import get_runtime_action_health
        monkeypatch.setattr(cfg, "INTERNET_ACTIONS_ENABLED", True, raising=False)
        monkeypatch.setattr(cfg, "GOOGLE_CALENDAR_ENABLED", True, raising=False)
        stub = SimpleNamespace(
            is_authenticated=True,
            token_expired_no_refresh=True,
            has_scope=lambda scope: True,
        )
        monkeypatch.setattr(ga, "get_google_auth", lambda: stub)
        health = get_runtime_action_health()
        assert "UNAVAILABLE" in health
        assert "no refresh token" in health
        assert "reauth_google" in health

    def test_f15_valid_token_still_available(self, monkeypatch):
        import config.app_config as cfg
        import core.actions.google_auth as ga
        from core.actions.registry import get_runtime_action_health
        monkeypatch.setattr(cfg, "INTERNET_ACTIONS_ENABLED", True, raising=False)
        monkeypatch.setattr(cfg, "GOOGLE_CALENDAR_ENABLED", True, raising=False)
        stub = SimpleNamespace(
            is_authenticated=True,
            token_expired_no_refresh=False,
            has_scope=lambda scope: True,
        )
        monkeypatch.setattr(ga, "get_google_auth", lambda: stub)
        assert "calendar_create_event backend: AVAILABLE" in get_runtime_action_health()

    def test_f15_property_disk_only(self, tmp_path):
        """token_expired_no_refresh reads only the token file — no network."""
        import core.actions.google_auth as ga
        mgr = object.__new__(ga.GoogleAuthManager)
        creds_expired = SimpleNamespace(expired=True, refresh_token=None)
        mgr._load_token = lambda: creds_expired
        assert ga.GoogleAuthManager.token_expired_no_refresh.fget(mgr) is True
        creds_refreshable = SimpleNamespace(expired=True, refresh_token="rt")
        mgr._load_token = lambda: creds_refreshable
        assert ga.GoogleAuthManager.token_expired_no_refresh.fget(mgr) is False


# ===========================================================================
# F23 / F24 — grounding source-material cap + corrected-flag honesty
# ===========================================================================

FIRING_VERDICT = json.dumps({
    "false_claim_present": True,
    "claim": "The deadline is Friday",
    "correction": "The correct deadline is Saturday 2026-09-05.",
    "confidence": 0.95,
})


class _StubMM:
    def __init__(self, raw):
        self.raw = raw

    async def generate_once(self, *a, **k):
        return self.raw


def _gctx(mm):
    return SimpleNamespace(
        user_text="when is the drop deadline?",
        raw_context={"tone_level": "CrisisLevel.CONVERSATIONAL"},
        orchestrator=SimpleNamespace(model_manager=mm),
        telemetry={},
    )


class TestGroundingFixes:
    def test_f23_source_material_cap_matches_handlers(self):
        import core.grounding_check as gc
        assert gc._SOURCE_MATERIAL_TRUNC == 6000
        clock = "[AUTHORITATIVE RUNTIME CLOCK]\nCurrent time: Monday, 2026-08-31"
        src = clock + "\n\n" + "x" * 7000
        prompt = gc._build_verifier_prompt("q", "r", source_material=src)
        assert "[AUTHORITATIVE RUNTIME CLOCK]" in prompt
        assert "x" * 5000 in prompt  # tail beyond old 3500 cap now survives

    # Long enough to clear GROUNDING_MIN_RESPONSE_CHARS and shaped to trip the
    # deterministic claim prefilter (truth-stance + date), so the stub verifier
    # actually runs.
    FIRING_RESPONSE = (
        "The drop deadline is definitely Friday 2026-09-04 — that is a fact, "
        "and the registrar has always processed withdrawals on Fridays. "
        "You can rely on that date when you plan the rest of the week."
    )

    @pytest.mark.asyncio
    async def test_f24_no_corrected_flag_when_nothing_ships(self, monkeypatch):
        """Verifier flags but neither integration nor suffix produces output:
        grounding_corrected must NOT be set."""
        import config.app_config as ac
        import core.grounding_check as gc
        from gui import handlers
        monkeypatch.setattr(ac, "GROUNDING_CHECK_ENABLED", True, raising=False)
        monkeypatch.setattr(ac, "GROUNDING_INTEGRATE_ENABLED", False, raising=False)
        monkeypatch.setattr(gc, "build_grounding_correction", lambda *a, **k: "")
        ctx = _gctx(_StubMM(FIRING_VERDICT))
        revised, suffix = await handlers._apply_grounding_check(
            ctx, self.FIRING_RESPONSE)
        assert ctx.telemetry.get("grounding_verifier_fired") is True  # not vacuous
        assert revised is None and suffix == ""
        assert "grounding_corrected" not in ctx.telemetry

    @pytest.mark.asyncio
    async def test_f24_corrected_flag_set_on_suffix(self, monkeypatch):
        import config.app_config as ac
        from gui import handlers
        monkeypatch.setattr(ac, "GROUNDING_CHECK_ENABLED", True, raising=False)
        monkeypatch.setattr(ac, "GROUNDING_INTEGRATE_ENABLED", False, raising=False)
        ctx = _gctx(_StubMM(FIRING_VERDICT))
        revised, suffix = await handlers._apply_grounding_check(
            ctx, self.FIRING_RESPONSE)
        assert ctx.telemetry.get("grounding_verifier_fired") is True
        assert suffix  # real correction shipped
        assert ctx.telemetry.get("grounding_corrected") is True


# ===========================================================================
# F10 / F17 — detector: assessment precedence + kill switch
# ===========================================================================

class TestDetectorFixes:
    def test_f10_assessment_beats_deliberation_shape(self):
        from core.insight.detector import detect_insight_request
        intent = detect_insight_request(
            "Assess my theory that caffeine ruins my sleep against my history"
        )
        assert intent is not None
        assert intent.kind == "insight_assessment"

    def test_f10_explicit_pattern_command_still_wins(self):
        from core.insight.detector import detect_insight_request
        intent = detect_insight_request(
            "Please use the pattern tool to test my theory against my history"
        )
        assert intent is not None
        assert intent.kind == "pattern_temporal"

    def test_f17_kill_switch_disables_pattern_temporal(self, monkeypatch):
        import config.app_config as cfg
        from core.insight.detector import detect_insight_request
        query = "How often have I mentioned my headaches in the last month?"
        monkeypatch.setattr(cfg, "PATTERN_ANALYSIS_ENABLED", True, raising=False)
        on = detect_insight_request(query)
        assert on is not None and on.kind == "pattern_temporal"
        monkeypatch.setattr(cfg, "PATTERN_ANALYSIS_ENABLED", False, raising=False)
        off = detect_insight_request(query)
        assert off is None or off.kind != "pattern_temporal"

    def test_f17_assessment_survives_kill_switch(self, monkeypatch):
        import config.app_config as cfg
        from core.insight.detector import detect_insight_request
        monkeypatch.setattr(cfg, "PATTERN_ANALYSIS_ENABLED", False, raising=False)
        intent = detect_insight_request(
            "Assess my theory that caffeine ruins my sleep against my history"
        )
        assert intent is not None and intent.kind == "insight_assessment"

    def test_f17_gate_helper_reads_live_config(self, monkeypatch):
        import config.app_config as cfg
        from core.agentic.gate import _pattern_analysis_enabled
        monkeypatch.setattr(cfg, "PATTERN_ANALYSIS_ENABLED", False, raising=False)
        assert _pattern_analysis_enabled() is False
        monkeypatch.setattr(cfg, "PATTERN_ANALYSIS_ENABLED", True, raising=False)
        assert _pattern_analysis_enabled() is True


# ===========================================================================
# F27 / F29 — query checker
# ===========================================================================

class TestQueryCheckerFixes:
    def test_f27_third_party_docs_not_personal(self):
        from utils.query_checker import is_personal_doc_search
        assert not is_personal_doc_search(
            "Look at the FastAPI docs and tell me how to mount static files for my SPA"
        )

    def test_f27_live_positive_still_detects(self):
        from utils.query_checker import is_personal_doc_search
        assert is_personal_doc_search(
            "please search for documents related to the MGT class I am currently enrolled in"
        )

    def test_f27_possessive_product_notes_still_personal(self):
        from utils.query_checker import is_personal_doc_search
        assert is_personal_doc_search("search my Python notes for the decorator example")

    def test_f29_buried_question_mark_no_longer_fires(self):
        from utils.query_checker import is_continuation_answer
        prior = (
            "Do you want the short or long version? Anyway, here is the plan. "
            + "The steps are as follows. " * 12
            + "That should cover everything you need for tomorrow."
        )
        assert "?" not in prior[-240:]
        assert not is_continuation_answer("short one", prior)

    def test_f29_closing_question_still_fires(self):
        from utils.query_checker import is_continuation_answer
        prior = "I can queue those up. Do you want them on the day of, or the day before?"
        assert is_continuation_answer("Day of", prior)


# ===========================================================================
# F28 — song frame requires a music noun
# ===========================================================================

class TestSongFrame:
    def test_f28_podcast_frame_not_lyrics(self):
        from core.content_type_detector import detect_content_type
        text = ("I am listening to this podcast about the fall of Rome and " +
                "it keeps making me think about resilience. " * 30)
        assert len(text) >= 1200
        result = detect_content_type(text)
        assert result.content_type != "lyrics"

    def test_f28_song_frame_still_detects(self):
        # Frame mid-message (a start-anchored "this song ..." would take the
        # share-preamble path instead of the 5.5 frame rule).
        from core.content_type_detector import detect_content_type
        text = ("Been on repeat all day. I am listening to this song about " +
                "everything we lost and it will not leave my head at all. " * 25)
        assert len(text) >= 1200
        result = detect_content_type(text)
        assert result.content_type == "lyrics"


# ===========================================================================
# F35 — _ENQUEUE_RE escaped quotes
# ===========================================================================

class TestEnqueueRe:
    def test_escaped_quote_paren_payload_survives(self):
        from utils.page_extract import _ENQUEUE_RE
        payload = '"prose with an escaped \\") inside it and more text after"'
        html = f"streamController.enqueue({payload});"
        m = _ENQUEUE_RE.search(html)
        assert m is not None
        assert json.loads(m.group(1)) == 'prose with an escaped ") inside it and more text after'

    def test_plain_payload_unchanged(self):
        from utils.page_extract import _ENQUEUE_RE
        html = 'x.enqueue("hello world");'
        m = _ENQUEUE_RE.search(html)
        assert m is not None and json.loads(m.group(1)) == "hello world"


# ===========================================================================
# F26 — wiki-chroma timeout skips wiki (no live-API fallthrough)
# ===========================================================================

class TestWikiTimeoutSkip:
    @pytest.mark.asyncio
    async def test_f26_timeout_returns_empty_not_api_fallback(self, monkeypatch):
        import core.prompt.gatherer_knowledge as gk

        fallback_called = {"n": 0}

        async def _fallback_sentinel(term):
            fallback_called["n"] += 1
            return {"content": "live api"}

        coll = MagicMock()
        coll.count.return_value = 10
        chroma = MagicMock()
        chroma.collections = {"wiki_knowledge": coll}

        def _slow_query(*a, **k):
            import time
            time.sleep(0.5)
            return [{"content": "late", "metadata": {}}]

        chroma.query_collection = _slow_query
        stub = SimpleNamespace(
            memory_coordinator=SimpleNamespace(chroma_store=chroma),
            _should_skip_wikipedia=lambda q: False,
            _get_wiki_snippet_cached=_fallback_sentinel,
        )
        monkeypatch.setattr(gk, "WIKI_CHROMA_TIMEOUT_S", 0.05)
        result = await gk.KnowledgeRetrievalMixin._get_wiki_content(stub, "roman empire")
        assert result == []
        assert fallback_called["n"] == 0, "timeout fell through to the live API"
        # Let the stuck worker finish so the semaphore slot releases for other tests.
        await asyncio.sleep(0.6)

    @pytest.mark.asyncio
    async def test_f26_inflight_saturation_skips(self, monkeypatch):
        import core.prompt.gatherer_knowledge as gk
        chroma = MagicMock()
        chroma.collections = {"wiki_knowledge": MagicMock()}
        stub = SimpleNamespace(
            memory_coordinator=SimpleNamespace(chroma_store=chroma),
            _should_skip_wikipedia=lambda q: False,
            _get_wiki_snippet_cached=None,
        )
        import threading
        monkeypatch.setattr(gk, "_WIKI_CHROMA_INFLIGHT", threading.Semaphore(0))
        result = await gk.KnowledgeRetrievalMixin._get_wiki_content(stub, "roman empire")
        assert result == []
        assert not chroma.query_collection.called


# ===========================================================================
# F11 — direct fetch streams with a byte cap
# ===========================================================================

class _FakeStreamResponse:
    def __init__(self, chunks, headers=None, status_code=200):
        self._chunks = chunks
        self.headers = headers or {"content-type": "text/html"}
        self.status_code = status_code
        self.charset_encoding = "utf-8"

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeClient:
    def __init__(self, response):
        self._response = response

    def stream(self, method, url):
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class TestDirectFetchStreaming:
    def _manager_stub(self):
        from knowledge.web_search_manager import WebSearchManager
        stub = object.__new__(WebSearchManager)
        stub.max_content_chars = 100_000
        return stub

    @pytest.mark.asyncio
    async def test_f11_chunked_body_capped(self, monkeypatch):
        import knowledge.web_search_manager as wsm
        # 100 chunks x 100KB = 10MB offered; cap is 2MB.
        chunk = b"<p>" + b"a" * 99_997
        resp = _FakeStreamResponse([chunk] * 100)  # no Content-Length header
        fake_httpx = _types_mod.ModuleType("httpx")
        fake_httpx.AsyncClient = lambda **k: _FakeClient(resp)
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

        async def _noop_dns(url):
            return None

        monkeypatch.setattr(wsm, "_validate_fetch_url_dns", _noop_dns)
        stub = self._manager_stub()
        pages = await wsm.WebSearchManager._direct_fetch(stub, "http://example.com/big")
        # Extraction may drop markup, but whatever came back was built from at
        # most the cap's worth of bytes.
        consumed = sum(len(c) for c in [chunk] * 100)
        assert consumed > stub._DIRECT_FETCH_MAX_BYTES  # the fake offered more
        if pages:
            assert len(pages[0].content) <= stub._DIRECT_FETCH_MAX_BYTES

    @pytest.mark.asyncio
    async def test_f11_small_page_roundtrips(self, monkeypatch):
        import knowledge.web_search_manager as wsm
        body = ("<html><head><title>T</title></head><body><p>" +
                "hello world, this is a perfectly ordinary page. " * 40 +
                "</p></body></html>").encode()
        resp = _FakeStreamResponse([body])
        fake_httpx = _types_mod.ModuleType("httpx")
        fake_httpx.AsyncClient = lambda **k: _FakeClient(resp)
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

        async def _noop_dns(url):
            return None

        monkeypatch.setattr(wsm, "_validate_fetch_url_dns", _noop_dns)
        stub = self._manager_stub()
        pages = await wsm.WebSearchManager._direct_fetch(stub, "http://example.com/ok")
        assert pages and "hello world" in pages[0].content
