"""2026-09-12 audit of the 2026-09-11 sessions (163 turn records, four debug
logs). Two root causes, both driven here through the DEPLOYED functions.

R1 — keyword matcher bounded on the LEFT only (class BC-01, 7th incident).
`utils.trigger_match.KeywordMatcher` compiled every bare word as `\\bkw`, so a
keyword fired inside any longer word it prefixes. Live consequences on
2026-09-11:
  * `'numb'` matched "number"/"numbers" (567 hits in the corpus) and `'down'`
    matched "downloaded"/"download" (154) — two CONCERN keyword hits set a
    13,343-char statistics-homework paste to `harm_score: 4.0 (0H, 0M, 2C)` →
    CONCERN, which armed the distress-sticky floor for the rest of the
    session (39 of the day's 163 turns carried `tone_trigger=
    distress_sticky_floor`, including nine R-package-install debugging turns
    that got LIGHT SUPPORT — "don't offer unsolicited advice" — while asking
    technical yes/no questions).
  * `'dead'` matched "deadline"/"deadlift" (177) and `'war'` matched
    "warning"/"warm"/"warehouses" (105) in HEAVY_KEYWORDS, leaving 17 corpus
    rows stored `is_heavy_topic=True` with no valid heavy hit at all (10 of
    them ordinary prose) — the read-time neutralizer added on 2026-09-08 only
    covered rows containing code-shaped lines, so a
    prose row ("I dropped <course> today before the deadline") stayed
    distress evidence forever.

R2 — the web-search trigger was credit-blind (classes BC-32, BC-46, BC-72).
The agentic gate called `analyze_for_web_search_llm` without a credit count
(default 100) while the prompt builder passed the live count. On 2026-09-11
the real budget was 0 from 19:11, so the two callers hashed into different
cache buckets: gpt-4o-mini ran TWICE per turn (87 calls / 12 cache hits in
one session) and returned contradictory verdicts for the same message — at
18:58:38 the gate's copy returned should_search=True with terms
['ethical justification of actions 2026', ...] and at 18:58:41 the builder's
copy said should_search=False ("philosophical discussion ... does not require
real-time information"). The gate's True won, routing a philosophy turn into
a 24 s agentic loop. Nothing stood the search arm down at zero budget either:
83 "Daily limit reached" searches ran between 19:11 and 22:14, six turns
spending 25-33 s each on loops whose web results were all empty.
"""

import asyncio

import pytest

from utils.trigger_match import compile_keyword_matcher, prefix_only_hits


# --------------------------------------------------------------------------
# R1a — the matcher contract
# --------------------------------------------------------------------------

def _hits(keywords, text):
    return sorted({h.keyword for h in compile_keyword_matcher(keywords).iter_hits(text)})


class TestMorphologyBoundedMatcher:
    """A bare keyword matches itself or a sense-preserving inflection, and
    nothing else — on BOTH sides."""

    @pytest.mark.parametrize("kw,text", [
        # the live 2026-09-11 false positives
        ("numb", "#nrow just gives number rows, in r subsetting with numbers"),
        ("numb", "| doors |  |  | number of doors |  |"),
        ("down", "content type 'application/x-gzip' length 243856 bytes\ndownloaded 238 kb"),
        ("down", "download the data file, downloading now, downloads/downstream"),
        ("dead", "i dropped the course today before the deadline"),
        ("dead", "after deadlifts my back hurt"),
        ("war", "warning message: nas introduced by coercion"),
        ("war", "warm weather and the warehouses"),
        # the two earlier incidents this class already cost
        ("solve", "crisis resolution, unresolved thread"),
        ("ice", "model <- lm(price ~ ., data = used_car_data)"),
        ("compute", "my computer is slow"),
        # an ALTERED stem never matches on its own: the first cut of this fix
        # made 'hate' match "hat" and a gcc banner "(Red Hat 14.3.1-4)" in an
        # R install log scored a HEAVY hit
        ("hate", "using c compiler: gcc (gcc) 14.3.1 (red hat 14.3.1-4)"),
        ("solve", "solv"),
        ("panic", "panick"),
    ])
    def test_prefix_of_a_longer_word_does_not_fire(self, kw, text):
        assert _hits([kw], text) == []

    @pytest.mark.parametrize("kw,text", [
        ("numb", "i just feel numb"),
        ("numb", "total numbness in my hands"),
        ("down", "i feel so down today"),
        ("dead", "the battery is dead"),
        ("war", "the war and the wars before it"),
        ("solve", "solve it / solves it / solving it / solved it"),
        ("panic", "i panicked and kept panicking"),
        ("stop", "it stopped and then kept stopping"),
        ("hopeless", "a wave of hopelessness"),
        ("fear", "fearful of the outcome"),
        ("suicidal", "measures of suicidality"),
        ("sad", "the sadness is heavy / said it sadly"),
        ("ice", "ice raids downtown"),
    ])
    def test_the_word_and_its_inflections_still_fire(self, kw, text):
        assert _hits([kw], text) == [kw]

    @pytest.mark.parametrize("kw,text", [
        # inflections the LEFT-only rule used to MISS (e-drop): measured
        # against the corpus, these 8 tokens are the only new matches the
        # bounded rule adds, and every one is a true inflection.
        ("calculate", "calculating the residuals"),
        ("solve", "solving for x"),
        ("compute", "computing the mean"),
        ("hate", "i keep hating myself for it"),
        ("rage", "raging at everything"),
        ("abuse", "he was abusing her"),
        ("pressure", "pressuring me to decide"),
        ("police", "aggressive policing"),
    ])
    def test_e_drop_inflections_the_old_rule_missed_now_fire(self, kw, text):
        assert _hits([kw], text) == [kw]

    def test_phrases_keep_substring_semantics(self):
        assert _hits(["go to http"], "go to https://example.com") == ["go to http"]
        assert _hits(["tear gas"], "they used tear gas") == ["tear gas"]

    def test_identifier_forms_still_match_the_bare_tool_keyword(self):
        assert _hits(["git_stats"], "look at git_stats_manager") == ["git_stats"]
        assert _hits(["wolfram"], "call wolfram_alpha") == ["wolfram"]
        # but a foreign continuation is still not a match
        assert _hits(["down"], "downloaded_packages") == []

    def test_trailing_star_declares_prefix_semantics(self):
        m = ["discriminat*"]
        assert _hits(m, "workplace discrimination") == ["discriminat"]
        assert _hits(m, "discriminatory policy") == ["discriminat"]
        assert _hits(m, "they discriminate") == ["discriminat"]

    def test_prefix_only_hits_reports_what_the_old_rule_would_have_matched(self):
        m = compile_keyword_matcher(["dead", "war", "depressed"])
        assert prefix_only_hits(m, "before the deadline") == ["dead"]
        assert prefix_only_hits(m, "warning message") == ["war"]
        # a real hit is not a prefix-only hit
        assert prefix_only_hits(m, "i am depressed") == []
        assert prefix_only_hits(m, "nothing here") == []


# --------------------------------------------------------------------------
# R1b — the deployed consumers on the live 2026-09-11 text
# --------------------------------------------------------------------------

# Verbatim fragments of the 13,343-char message stored at 2026-09-11T18:02:53,
# whose tone was set to CONCERN by `harm_score: 4.0 (0H, 0M, 2C)`.
LIVE_HOMEWORK_PASTE = (
    'R version 4.4.3 (2025-02-28) -- "Trophy Case"\n'
    "trying URL 'https://cran.rstudio.com/src/contrib/utf8_1.2.6.tar.gz'\n"
    "Content type 'application/x-gzip' length 243856 bytes (238 KB)\n"
    "downloaded 238 KB\n"
    "#nrow just gives number rows, in r subsetting with the second index\n"
    "| doors |  |  | number of doors |  |  |\n"
    "download the data file \"usedcars2.csv\"\n"
    "i havent written down the intrepations and still need to\n"
)

LIVE_HEAVY_PROSE_ROWS = [
    "Hang on. I dropped cse 6040 today before the deadline which I believe",
    "Water. I don't understand why my doctor didn't warn me about this",
    "Yeah that's like a warning sign I watch for. If I get itchy",
    "I need to finish this before the deadline tonight and I am tired",
]


class TestDeployedHarmScoreOnLiveText:
    def test_homework_paste_no_longer_scores_concern_keywords(self):
        from utils.tone_detector import _calculate_harm_score
        score, matched, _counts = _calculate_harm_score(LIVE_HOMEWORK_PASTE.lower())
        # "written down" is a real bare-word hit and still counts (+2);
        # "number"/"numbers"/"downloaded"/"download" must not.
        assert not any("numb" in m for m in matched), matched
        assert score <= 2.0, (score, matched)

    def test_genuine_distress_still_scores(self):
        from utils.tone_detector import _calculate_harm_score
        score, matched, _counts = _calculate_harm_score(
            "i feel numb and hopeless and i can't stop crying")
        assert score >= 4.0, (score, matched)


class TestDeployedHeavyRowNeutralizer:
    @pytest.mark.parametrize("text", LIVE_HEAVY_PROSE_ROWS)
    def test_prose_row_stored_heavy_by_the_boundary_bug_is_not_distress(self, text):
        from utils.tone_detector import _heavy_row_is_distress_evidence
        assert _heavy_row_is_distress_evidence({"query": text}) is False

    def test_genuine_heavy_first_person_row_still_is(self):
        from utils.tone_detector import _heavy_row_is_distress_evidence
        assert _heavy_row_is_distress_evidence(
            {"query": "i have been depressed for weeks and my therapist agrees"}) is True

    def test_llm_flagged_prose_with_no_keyword_keeps_the_first_person_check(self):
        """A heavy flag set by another path (the LLM classifier, an unlisted
        phrase) must not be neutralized just because no keyword matches."""
        from utils.tone_detector import _heavy_row_is_distress_evidence
        assert _heavy_row_is_distress_evidence(
            {"query": "i can't carry any of this anymore"}) is True

    def test_row_without_text_still_fails_closed(self):
        from utils.tone_detector import _heavy_row_is_distress_evidence
        assert _heavy_row_is_distress_evidence({}) is True

    def test_heavy_keyword_hits_agrees_with_the_neutralizer(self):
        from utils.query_checker import heavy_keyword_hits, heavy_prefix_only_hits
        for text in LIVE_HEAVY_PROSE_ROWS:
            assert heavy_keyword_hits(text) == []
            assert heavy_prefix_only_hits(text) != []


class TestLengthIsNotHeaviness:
    """R1c: `_is_heavy_topic_heuristic` returned True for ANY message over
    HEAVY_TOPIC_CHAR_THRESHOLD chars, as a proxy for "pasted news article".
    The 2026-09-11 17:08 turn was an 18,549-char R package-install log with
    ZERO heavy-keyword hits, stored `is_heavy_topic=True` on length alone;
    `_recent_distress_from_history` read that row and armed the
    distress-sticky floor over the nine debugging turns that followed."""

    R_INSTALL_LOG = (
        'sucess here? install.packages("utf8")\n'
        "Installing package into '/home/u/R/x86_64-redhat-linux-gnu-library/4.4'\n"
        "trying URL 'https://cran.rstudio.com/src/contrib/utf8_1.2.6.tar.gz'\n"
        "downloaded 238 KB\n"
        "** using staged installation\n"
        "using C compiler: 'gcc (GCC) 14.3.1 20251022 (Red Hat 14.3.1-4)'\n"
        "/usr/bin/gcc -I\"/usr/include/R\" -DNDEBUG -c utf8.c -o utf8.o\n"
    ) * 40

    def test_long_paste_with_no_heavy_content_is_not_heavy(self):
        from utils.query_checker import _is_heavy_topic_heuristic, heavy_keyword_hits
        assert len(self.R_INSTALL_LOG) > 2500
        assert heavy_keyword_hits(self.R_INSTALL_LOG) == []
        assert _is_heavy_topic_heuristic(self.R_INSTALL_LOG) is False

    def test_a_long_heavy_article_is_still_heavy(self):
        from utils.query_checker import _is_heavy_topic_heuristic
        article = ("Police raided the building and made several arrests; "
                   "deportation proceedings began. " * 60)
        assert len(article) > 2500
        assert _is_heavy_topic_heuristic(article) is True

    def test_a_short_genuine_distress_message_is_still_heavy(self):
        from utils.query_checker import _is_heavy_topic_heuristic
        assert _is_heavy_topic_heuristic(
            "i'm depressed and my therapist is away") is True

    def test_the_log_row_no_longer_arms_the_floor_through_history(self):
        from utils.tone_detector import _recent_distress_from_history
        # the row as the daemon stored it on 2026-09-11
        poisoned = [{"query": self.R_INSTALL_LOG, "is_heavy_topic": True}]
        assert _recent_distress_from_history(poisoned) is False


# --------------------------------------------------------------------------
# R2 — credit-aware web-search routing
# --------------------------------------------------------------------------

@pytest.fixture
def clean_trigger_cache():
    import utils.web_search_trigger as wst
    wst._llm_trigger_cache.clear()
    wst._llm_trigger_inflight.clear()
    yield wst
    wst._llm_trigger_cache.clear()
    wst._llm_trigger_inflight.clear()


@pytest.fixture
def isolated_limiter_registry(monkeypatch):
    """A fresh, empty live-limiter registry for one test. The process-wide
    WeakSet holds every limiter any earlier test (or module singleton) left
    alive, so reading it unpatched is an uncontrolled oracle."""
    import weakref

    import knowledge.web_search_manager as wsm
    registry = weakref.WeakSet()
    monkeypatch.setattr(wsm, "_LIVE_RATE_LIMITERS", registry)
    return registry


@pytest.fixture
def spent_limiter(isolated_limiter_registry):
    """A live rate limiter whose daily budget is gone (the 2026-09-11 19:11
    state: credits_today 104 against a 100 limit) — the ONLY limiter the
    registry holds for the duration of the test."""
    from datetime import datetime

    from knowledge.web_search_manager import WebSearchRateLimiter
    limiter = WebSearchRateLimiter(daily_limit=100, state_file="/dev/null")
    limiter._credits_today = 104.0
    limiter._current_date = datetime.now().strftime("%Y-%m-%d")
    assert list(isolated_limiter_registry) == [limiter]
    yield limiter


class TestCreditResolutionIsOneChokepoint:
    def test_none_resolves_to_the_live_budget(self, spent_limiter):
        from utils.web_search_trigger import _resolve_remaining_credits
        assert _resolve_remaining_credits(None) == 0.0

    def test_explicit_value_is_honoured(self, spent_limiter):
        from utils.web_search_trigger import _resolve_remaining_credits
        assert _resolve_remaining_credits(42) == 42.0

    def test_no_limiter_in_process_is_not_exhausted(self, isolated_limiter_registry):
        from utils.web_search_trigger import (
            _ASSUMED_CREDITS_NO_LIMITER, _resolve_remaining_credits)
        import knowledge.web_search_manager as wsm
        assert wsm.live_remaining_credits() is None
        assert _resolve_remaining_credits(None) == _ASSUMED_CREDITS_NO_LIMITER

    def test_gate_and_gatherer_shaped_calls_classify_once(
            self, clean_trigger_cache, spent_limiter, monkeypatch):
        """The live divergence: one turn, two callers, two LLM calls with
        opposite verdicts. They must now share one classification."""
        wst = clean_trigger_cache
        calls = []

        async def _fake_classify(query, model_manager, remaining_credits,
                                 timeout, conversation_context=None):
            calls.append(remaining_credits)
            return wst.LLMSearchTriggerResponse(
                should_search=True, confidence=0.8, reason="stub",
                search_terms=["t"], search_depth="standard", num_searches=1)

        # monkeypatch, never a bare assignment: the fake must not outlive
        # this test (2026-09-12 adversarial review, finding 8).
        monkeypatch.setattr(wst, "_classify_with_llm_unified", _fake_classify)
        query = "Oh but that doesn't matter. A wrong action can be right if the models point to right."
        ctx = "User: earlier turn\nAssistant: earlier reply"

        async def _run():
            gate = await wst.analyze_for_web_search_llm(          # gate shape
                query=query, model_manager=object(), conversation_context=ctx)
            gatherer = await wst.analyze_for_web_search_llm(      # gatherer shape
                query=query, model_manager=object(), crisis_level="CONVERSATIONAL",
                web_search_enabled=True, remaining_credits=0.0,
                conversation_context=ctx)
            return gate, gatherer

        gate, gatherer = asyncio.run(_run())
        assert len(calls) == 1, f"classified {len(calls)} times: {calls}"
        assert gate.should_search == gatherer.should_search
        assert gate.reason == gatherer.reason


class TestBudgetVeto:
    def test_exhausted_budget_vetoes_the_search_arm(self, spent_limiter):
        from utils.web_search_trigger import analyze_for_web_search_llm
        d = asyncio.run(analyze_for_web_search_llm(
            "what is the latest news on the election today", model_manager=None))
        assert d.should_search is False
        assert d.source == "budget"
        assert d.search_terms == [] and d.num_searches == 0
        assert "budget exhausted" in d.reason

    def test_funded_budget_leaves_the_decision_alone(self):
        from utils.web_search_trigger import analyze_for_web_search_llm
        d = asyncio.run(analyze_for_web_search_llm(
            "what is the latest news on the election today",
            model_manager=None, remaining_credits=100))
        assert d.should_search is True
        assert d.source != "budget"

    def test_other_routing_flags_survive_the_veto(self):
        """The gate's Tier 4 reads memory / knowledge / document-generation /
        pattern flags off this SAME decision — a budget veto must not take
        them down with the search arm."""
        from dataclasses import replace
        from utils.web_search_trigger import WebSearchDecision, _apply_budget_veto
        from utils.web_search_trigger import WebSearchDepth
        d = WebSearchDecision(
            should_search=True, depth=WebSearchDepth.QUICK, confidence=0.9,
            reason="LLM: x", matched_keywords=[], matched_patterns=[],
            search_terms=["a"], num_searches=2, source="llm",
            needs_memory_search=True, needs_document_generation=True,
            document_topic="t", document_type="report")
        out = _apply_budget_veto(d, 0.0)
        assert out.should_search is False
        assert out.needs_memory_search is True
        assert out.needs_document_generation is True
        assert out.document_topic == "t"
        # a decision that was already no-search is returned untouched
        no = replace(d, should_search=False)
        assert _apply_budget_veto(no, 0.0) is no

    def test_veto_does_not_teach_the_no_search_exemplar_store(self, spent_limiter):
        """Suppressions never teach (adaptive-exemplar doctrine)."""
        import utils.web_search_trigger as wst
        from utils.web_search_trigger import analyze_for_web_search_llm
        taught = []

        class _Store:
            def record(self, *a, **k):
                taught.append(a)

        import utils.adaptive_exemplars as ae
        original = ae.get_store
        ae.get_store = lambda: _Store()
        try:
            asyncio.run(analyze_for_web_search_llm(
                "what is the latest news on the election today", model_manager=None))
        finally:
            ae.get_store = original
        assert not any("no_search" in str(t) for t in taught), taught


class TestLiveToggleIsResolvedNotAsserted:
    """The sibling of the credit default, found by the new dm31 scanner: the
    gate passes no `web_search_enabled`, so a hardcoded True had the trigger
    classify as if search were on whatever the Settings toggle said."""

    def test_none_resolves_to_the_live_settings_toggle(self, monkeypatch):
        import config.app_config as cfg
        from utils.web_search_trigger import _resolve_web_search_enabled
        monkeypatch.setattr(cfg, "WEB_SEARCH_ENABLED", False, raising=False)
        assert _resolve_web_search_enabled(None) is False
        monkeypatch.setattr(cfg, "WEB_SEARCH_ENABLED", True, raising=False)
        assert _resolve_web_search_enabled(None) is True

    def test_explicit_value_is_honoured(self):
        from utils.web_search_trigger import _resolve_web_search_enabled
        assert _resolve_web_search_enabled(False) is False
        assert _resolve_web_search_enabled(True) is True

    def test_gate_shaped_call_sees_a_disabled_toggle(self, monkeypatch,
                                                    clean_trigger_cache):
        """A gate-shaped call (no toggle argument) must not search while web
        search is switched off."""
        import asyncio

        import config.app_config as cfg
        monkeypatch.setattr(cfg, "WEB_SEARCH_ENABLED", False, raising=False)
        wst = clean_trigger_cache
        d = asyncio.run(wst.analyze_for_web_search_llm(
            "what is the latest news on the election today", model_manager=None))
        assert d.should_search is False

    def test_no_public_entry_point_asserts_a_live_toggle(self):
        """The dm31 scanner gates this repo-wide; this pins the two entry
        points the 2026-09-11 turns actually went through."""
        import inspect
        from utils.web_search_trigger import (
            analyze_for_web_search_llm, get_search_decision_for_prompt)
        for fn in (analyze_for_web_search_llm, get_search_decision_for_prompt):
            default = inspect.signature(fn).parameters["web_search_enabled"].default
            assert default is None, (fn.__name__, default)


class TestToolHealthTellsTheTruthAboutBudget:
    def test_exhausted_budget_is_reported_unavailable(self):
        from knowledge.web_search_manager import WebSearchManager, WebSearchRateLimiter
        from datetime import datetime
        limiter = WebSearchRateLimiter(daily_limit=100, state_file="/dev/null")
        limiter._credits_today = 104.0
        limiter._current_date = datetime.now().strftime("%Y-%m-%d")
        mgr = WebSearchManager(api_key="k", rate_limiter=limiter)
        assert mgr.budget_exhausted() is True
        limiter._credits_today = 0.0
        assert mgr.budget_exhausted() is False

    def test_health_block_names_the_budget(self):
        from core.agentic.tools import ToolExecutor

        class _Mgr:
            def is_available(self):
                return True

            def is_enabled(self):
                return True

            def budget_exhausted(self):
                return True

        ex = ToolExecutor.__new__(ToolExecutor)
        ex.web_search_manager = _Mgr()
        for attr in ("chroma_store", "file_access_manager", "github_manager",
                     "git_stats_manager", "memory_expander", "wolfram_manager"):
            setattr(ex, attr, None)
        line = [ln for ln in ex.get_tool_health().splitlines() if ln.startswith("web_search")]
        assert line and "UNAVAILABLE" in line[0] and "budget" in line[0].lower(), line


class TestNoCallerAssumesAFullBudget:
    def test_orchestrator_does_not_hardcode_the_web_toggle(self):
        src = open("core/orchestrator.py").read()
        assert "web_search_enabled=True" not in src

    def test_trigger_default_is_resolve_not_a_constant(self):
        import inspect
        from utils.web_search_trigger import analyze_for_web_search_llm
        sig = inspect.signature(analyze_for_web_search_llm)
        assert sig.parameters["remaining_credits"].default is None


# --------------------------------------------------------------------------
# Test-lane hygiene (carried over from the 2026-09-12 handoff's open item 4)
# --------------------------------------------------------------------------

class TestLocationLookupIsOffInTestMode:
    """A pre-push run logged "[Location] IP geolocation resolved: <city>":
    outbound HTTPS to ipinfo.io plus a location leak from the test lane."""

    def test_background_refresh_is_a_no_op_under_daemon_test_mode(self, monkeypatch):
        from utils.location_resolver import LocationResolver
        started = []
        monkeypatch.setenv("DAEMON_TEST_MODE", "1")
        r = LocationResolver()
        monkeypatch.setattr(r, "_refresh_ip_location",
                            lambda: started.append(True))
        r._start_background_refresh()
        assert started == []
        assert r._refresh_in_flight is False

    def test_without_test_mode_the_thread_still_starts(self, monkeypatch):
        from utils.location_resolver import LocationResolver
        monkeypatch.delenv("DAEMON_TEST_MODE", raising=False)
        calls = []

        class _Thread:
            def __init__(self, target=None, daemon=None):
                self.target = target

            def start(self):
                calls.append("started")

        monkeypatch.setattr("utils.location_resolver.threading.Thread", _Thread)
        LocationResolver()._start_background_refresh()
        assert calls == ["started"]
