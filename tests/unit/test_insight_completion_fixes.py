"""Regression tests for the 2026-08-31 insight-completion fix batch.

Live failures driving these:
- All three evening insight turns hit "deliberation planner timed out after
  35s" on the active model (kimi-3) — planner/recovery JSON calls now route
  to a fast registered model (insight_mode.planner_model → review model).
- A co_occurrence planner response without phases died with "requires at
  least 2 phase(s)"; an explicit ISO window in the request now expands into
  weekly buckets on the MAIN planner path, not only compact recovery.
- Week-bucket headline numbers (both/only/neither) are now computed
  deterministically and survive manifest compaction.
- A 2025-H1 question retrieved 2026-clustered notes with no disclosure —
  windowed specs now carry per-channel in-window accounting.
- A degenerate kimi-3 stream looped garbage for ~3.5 minutes (19:59 resend
  turn); only the owner killing the process kept it out of storage.
- The resend itself: a mobile client that lost the SSE at completion
  re-submitted the identical query 3 minutes later and re-ran the whole
  pipeline; the ingress guard now serves the stored reply.
- Audit F1: _retry_fetch_urls_from_context read the current turn / typing
  placeholder instead of the failed turn on BOTH production history shapes.
"""
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.insight.deliberation import (
    _resolve_planner_model,
    _weekly_buckets_from_window,
    plan_deliberation,
    validate_and_freeze,
)
from core.response_parser import ResponseParser


class _FakeModelManager:
    def __init__(self, response: str, api_models=None):
        self.api_models = api_models if api_models is not None else {
            "gpt-4o-mini": "openai/gpt-4o-mini", "kimi-3": "moonshotai/kimi-k3",
        }
        self.calls = []
        self._response = response

    async def generate_once(self, prompt, model_name=None, **kwargs):
        self.calls.append({"prompt": prompt, "model_name": model_name, **kwargs})
        return self._response


_CO_OCCURRENCE_PLAN = """{
  "analysis_kind": "co_occurrence",
  "claims": [{"claim_id": "A",
    "proposition": "study and exercise co-occur in the same weeks",
    "dependencies": [], "authority": "assessment",
    "required_channels": ["pattern", "corpus"],
    "evidence_standard": "dated user observations"}],
  "outcome_terms": ["study", "exercise"],
  "series_terms": {"study": ["study", "homework"],
                   "exercise": ["gym", "workout"]},
  "behavioral_indicators": [],
  "directional_indicators": {"increase": [], "decrease": []},
  "phases": [],
  "requested_channels": ["pattern", "corpus"],
  "research_queries": {},
  "assumptions": []
}"""


class TestPlannerModelRouting:
    def test_configured_planner_model_wins(self, monkeypatch):
        import config.app_config as app_config
        monkeypatch.setattr(app_config, "INSIGHT_PLANNER_MODEL", "kimi-3", raising=False)
        monkeypatch.setattr(app_config, "RESPONSE_REVIEW_MODEL", "gpt-4o-mini", raising=False)
        mm = _FakeModelManager("{}")
        assert _resolve_planner_model(mm) == "kimi-3"

    def test_falls_back_to_review_model(self, monkeypatch):
        import config.app_config as app_config
        monkeypatch.setattr(app_config, "INSIGHT_PLANNER_MODEL", None, raising=False)
        monkeypatch.setattr(app_config, "RESPONSE_REVIEW_MODEL", "gpt-4o-mini", raising=False)
        mm = _FakeModelManager("{}")
        assert _resolve_planner_model(mm) == "gpt-4o-mini"

    def test_unregistered_models_yield_none(self, monkeypatch):
        import config.app_config as app_config
        monkeypatch.setattr(app_config, "INSIGHT_PLANNER_MODEL", "nope", raising=False)
        monkeypatch.setattr(app_config, "RESPONSE_REVIEW_MODEL", "also-nope", raising=False)
        mm = _FakeModelManager("{}")
        assert _resolve_planner_model(mm) is None

    def test_non_dict_registry_yields_none(self):
        assert _resolve_planner_model(SimpleNamespace(api_models=None)) is None
        assert _resolve_planner_model(object()) is None

    @pytest.mark.asyncio
    async def test_plan_deliberation_passes_planner_model(self, monkeypatch):
        import config.app_config as app_config
        monkeypatch.setattr(app_config, "INSIGHT_PLANNER_MODEL", None, raising=False)
        monkeypatch.setattr(app_config, "RESPONSE_REVIEW_MODEL", "gpt-4o-mini", raising=False)
        mm = _FakeModelManager(_CO_OCCURRENCE_PLAN)
        result = await plan_deliberation(
            "Examine whether my study and exercise co-occur from "
            "2026-06-01 through 2026-08-31.",
            model_manager=mm,
        )
        assert mm.calls and mm.calls[0]["model_name"] == "gpt-4o-mini"
        assert result.status == "ready"


class TestWeeklyBucketsFromWindow:
    def test_live_during_through_phrasing(self):
        buckets = _weekly_buckets_from_window(
            "Analyze whether my recorded sleep quality and productive work "
            "tend to co-occur during 2025-01-01 through 2025-06-30."
        )
        assert len(buckets) >= 20
        assert buckets[0] == {"label": "week-2025-01-01",
                              "start": "2025-01-01", "end": "2025-01-07"}
        assert buckets[-1]["end"] == "2025-06-30"

    def test_from_through_phrasing(self):
        buckets = _weekly_buckets_from_window(
            "from 2026-06-01 through 2026-08-31")
        assert buckets and buckets[0]["start"] == "2026-06-01"

    def test_no_window_no_buckets(self):
        assert _weekly_buckets_from_window("compare my study and exercise") == []

    def test_reversed_window_rejected(self):
        assert _weekly_buckets_from_window(
            "from 2026-08-31 through 2026-06-01") == []

    @pytest.mark.asyncio
    async def test_main_planner_co_occurrence_without_phases_gets_buckets(self, monkeypatch):
        import config.app_config as app_config
        monkeypatch.setattr(app_config, "INSIGHT_PLANNER_MODEL", None, raising=False)
        monkeypatch.setattr(app_config, "RESPONSE_REVIEW_MODEL", None, raising=False)
        mm = _FakeModelManager(_CO_OCCURRENCE_PLAN)
        result = await plan_deliberation(
            "Examine whether my social contact and household-maintenance "
            "activity co-occur across my recorded history from 2024-10-01 "
            "through 2025-03-31.",
            model_manager=mm,
        )
        assert result.status == "ready", result.limitations
        assert result.spec is not None
        assert len(result.spec.phases) >= 20
        assert result.spec.phases[0].start == "2024-10-01"


class TestCoOccurrenceBucketSummary:
    def _frozen_spec(self):
        proposal = {
            "analysis_kind": "co_occurrence",
            "claims": [{"claim_id": "A",
                        "proposition": "study and exercise co-occur",
                        "dependencies": [], "authority": "assessment",
                        "required_channels": [], "evidence_standard": ""}],
            "outcome_terms": ["study", "exercise"],
            "series_terms": {"study": ["study"], "exercise": ["gym"]},
            "behavioral_indicators": [],
            "directional_indicators": {"increase": [], "decrease": []},
            "phases": [
                {"label": "week-1", "start": "2026-06-01", "end": "2026-06-07"},
                {"label": "week-2", "start": "2026-06-08", "end": "2026-06-14"},
                {"label": "week-3", "start": "2026-06-15", "end": "2026-06-21"},
            ],
            "requested_channels": ["pattern", "corpus"],
            "research_queries": {},
            "assumptions": [],
        }
        result = validate_and_freeze(proposal)
        assert result.status == "ready", result.limitations
        return result.spec

    def _event(self, day, text):
        from memory.pattern_engine import LongitudinalEvent
        return LongitudinalEvent(
            timestamp=f"2026-06-{day:02d}", text=text, source_class="user",
            source_id=f"e-{day}-{abs(hash(text)) % 997}", speaker="user",
            evidence_class="direct_observation",
        )

    def test_bucket_classification(self):
        from memory.pattern_engine import run_longitudinal_scan
        spec = self._frozen_spec()
        events = [
            # week 1: both series present
            self._event(2, "long study session then gym"),
            # week 2: study only
            self._event(9, "study grind all day"),
            # week 3: nothing matching either series
            self._event(16, "watched a movie"),
        ]
        scan = run_longitudinal_scan(spec, events)
        summary = scan.co_occurrence
        assert summary["bucket_count"] == 3
        assert summary["buckets_with_all_series"] == 1
        assert summary["buckets_with_no_series"] == 1
        assert summary["buckets_with_partial_series"] == 1
        assert summary["buckets_with_series"] == {"study": 2, "exercise": 1}
        assert summary["same_event_joint_mentions"] == 1

    def test_summary_survives_both_compaction_tiers(self):
        from core.insight.synthesizer import _render_deliberation_manifest as _encode_manifest
        summary = {"bucket_count": 26, "buckets_with_all_series": 4}
        manifest = {
            "status": "ready", "spec": {}, "co_occurrence": summary,
            "phase_summary": [{"label": f"week-{i}", "filler": "x" * 400}
                              for i in range(26)],
            "channels": [], "claim_chain": [], "limitations": [],
        }
        compacted = _encode_manifest(manifest, max_chars=4000)
        assert '"co_occurrence"' in compacted
        assert '"buckets_with_all_series":4' in compacted
        minimal = _encode_manifest(manifest, max_chars=900)
        assert '"co_occurrence"' in minimal


class TestInWindowAccounting:
    @pytest.mark.asyncio
    async def test_out_of_window_notes_are_disclosed(self):
        from core.insight.coordinator import LongitudinalDeliberationCoordinator

        async def notes_adapter(_query):
            return [
                {"text": "studied hard", "timestamp": "2026-03-01"},
                {"text": "gym then study", "timestamp": "2026-04-01"},
                {"text": "in-window study note", "timestamp": "2025-02-01"},
            ]

        proposal = {
            "analysis_kind": "period_comparison",
            "claims": [{"claim_id": "A", "proposition": "study differed",
                        "dependencies": [], "authority": "assessment",
                        "required_channels": [], "evidence_standard": ""}],
            "outcome_terms": ["study"],
            "behavioral_indicators": [],
            "directional_indicators": {"increase": [], "decrease": []},
            "phases": [
                {"label": "jan", "start": "2025-01-01", "end": "2025-01-31"},
                {"label": "feb", "start": "2025-02-01", "end": "2025-02-28"},
            ],
            "requested_channels": ["pattern", "notes"],
            "research_queries": {},
            "assumptions": [],
        }
        coordinator = LongitudinalDeliberationCoordinator(
            adapters={"notes": notes_adapter},
        )
        result = await coordinator.run("compare my study in jan vs feb 2025",
                                       proposed_spec=proposal)
        counts = result.manifest["internal_in_window_counts"]
        assert counts["notes"] == {"dated_rows": 3, "in_window": 1}
        assert any("date-blind" in item for item in result.manifest["limitations"])


class TestDegenerateStreamGuard:
    def test_verbatim_line_loop_detected(self):
        garbage = "| a | b |\n" * 200
        assert ResponseParser.looks_degenerate_stream(garbage)

    def test_shingle_spam_detected(self):
        garbage = ("the the report report status status pending " * 120)
        assert ResponseParser.looks_degenerate_stream(garbage)

    def test_real_report_with_tables_not_flagged(self):
        rows = "\n".join(
            f"| week-{i} | {i % 3} | {i % 2} | {'yes' if i % 2 else 'no'} |"
            for i in range(30)
        )
        report = (
            "# Co-occurrence Report\n\n## Channel Status\n\n"
            "| Channel | Status | Count |\n|---|---|---|\n"
            "| pattern | succeeded | 14 |\n| notes | succeeded | 40 |\n\n"
            "## Weekly counts\n\n| Week | Study | Exercise | Both |\n"
            "|---|---|---|---|\n" + rows + "\n\n"
            "The record shows study activity in 9 of 13 weeks and exercise "
            "in 6 of 13, with 4 weeks containing both. Absence in the record "
            "is not absence in behavior; the denominators above bound every "
            "claim. Two weeks had no observations at all, so their empty "
            "cells reflect missing data rather than inactivity."
        )
        assert not ResponseParser.looks_degenerate_stream(report)

    def test_short_text_never_flagged(self):
        assert not ResponseParser.looks_degenerate_stream("ok\nok\nok")

    def test_sanitize_for_storage_drops_degenerate_output(self):
        garbage = "|| s @@ || s @@\n" * 300
        assert ResponseParser.sanitize_for_storage(garbage) == ""

    def test_sanitize_for_storage_keeps_real_answer(self):
        text = "Here is the analysis you asked for. " + " ".join(
            f"Point {i}: the record shows a distinct observation here."
            for i in range(40)
        )
        assert ResponseParser.sanitize_for_storage(text) != ""


class TestCompletedResendWindow:
    def _orchestrator(self, entries):
        corpus_manager = SimpleNamespace(corpus=entries)
        return SimpleNamespace(
            memory_system=SimpleNamespace(corpus_manager=corpus_manager))

    def test_serves_recent_completed_duplicate(self):
        import gui.handlers as handlers
        ts = (datetime.now() - timedelta(seconds=90)).isoformat()
        orch = self._orchestrator([
            {"query": "Examine whether my social contact and household "
                      "activity co-occur", "response": "# Frozen Contract Report",
             "timestamp": ts},
        ])
        norm = ("examine whether my social contact and household activity "
                "co-occur")
        assert handlers._recent_completed_duplicate(orch, norm) == \
            "# Frozen Contract Report"

    def test_old_or_empty_entries_do_not_match(self):
        import gui.handlers as handlers
        old_ts = (datetime.now() - timedelta(seconds=900)).isoformat()
        fresh_ts = (datetime.now() - timedelta(seconds=30)).isoformat()
        orch = self._orchestrator([
            {"query": "same question here", "response": "old reply",
             "timestamp": old_ts},
            {"query": "same question here", "response": "",
             "timestamp": fresh_ts},
        ])
        assert handlers._recent_completed_duplicate(
            orch, "same question here") is None

    def test_different_query_does_not_match(self):
        import gui.handlers as handlers
        ts = datetime.now().isoformat()
        orch = self._orchestrator([
            {"query": "question A", "response": "reply", "timestamp": ts},
        ])
        assert handlers._recent_completed_duplicate(orch, "question b") is None

    @pytest.mark.asyncio
    async def test_handle_submit_serves_stored_reply_without_reprocessing(self, monkeypatch):
        import gui.handlers as handlers
        ts = (datetime.now() - timedelta(seconds=60)).isoformat()
        query = "Examine whether my social contact and household activity co-occur over time"
        orch = self._orchestrator([
            {"query": query, "response": "# Frozen Contract Report",
             "timestamp": ts},
        ])
        inner_called = []

        async def fake_inner(*args, **kwargs):
            inner_called.append(True)
            yield {"role": "assistant", "content": "should not run"}

        monkeypatch.setattr(handlers, "_handle_submit_inner", fake_inner)
        chunks = []
        async for chunk in handlers.handle_submit(
            query, None, [], False, orch,
        ):
            chunks.append(chunk)
        assert not inner_called
        assert len(chunks) == 1
        assert "# Frozen Contract Report" in chunks[0]["content"]
        assert "resend" in chunks[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_fresh_query_still_processes(self, monkeypatch):
        import gui.handlers as handlers
        orch = self._orchestrator([])
        inner_called = []

        async def fake_inner(*args, **kwargs):
            inner_called.append(True)
            yield {"role": "assistant", "content": "processed"}

        monkeypatch.setattr(handlers, "_handle_submit_inner", fake_inner)
        chunks = []
        async for chunk in handlers.handle_submit(
            "a brand new question about something else entirely",
            None, [], False, orch,
        ):
            chunks.append(chunk)
        assert inner_called
        assert chunks[-1]["content"] == "processed"


class TestRetryFetchHistoryShapes:
    _FAILED_REPLY = "I tried but the page came back as a blank page."
    _URL_TURN = "Check out https://chatgpt.com/share/abc123 please"
    _RETRY = "Okay try again now, I fixed it"

    def test_spa_shape_current_turn_appended(self):
        import gui.handlers as handlers
        history = [
            {"role": "user", "content": self._URL_TURN},
            {"role": "assistant", "content": self._FAILED_REPLY},
            {"role": "user", "content": self._RETRY},   # SPA appends current turn
        ]
        urls = handlers._retry_fetch_urls_from_context(self._RETRY, history)
        assert urls == ["https://chatgpt.com/share/abc123"]

    def test_gradio_shape_with_typing_placeholder(self):
        import gui.handlers as handlers
        history = [
            {"role": "user", "content": self._URL_TURN},
            {"role": "assistant", "content": self._FAILED_REPLY},
            {"role": "user", "content": self._RETRY},
            {"role": "assistant", "content": "…"},      # Gradio typing placeholder
        ]
        urls = handlers._retry_fetch_urls_from_context(self._RETRY, history)
        assert urls == ["https://chatgpt.com/share/abc123"]

    def test_no_failure_marker_no_retry(self):
        import gui.handlers as handlers
        history = [
            {"role": "user", "content": self._URL_TURN},
            {"role": "assistant", "content": "Here is the summary you asked for."},
            {"role": "user", "content": self._RETRY},
        ]
        assert handlers._retry_fetch_urls_from_context(self._RETRY, history) == []


# ---------------------------------------------------------------------------
# Round 2 — fixes from the 2026-08-31 post-restart live test turns
# ---------------------------------------------------------------------------

_CO_OCCURRENCE_PLAN_NO_SERIES = _CO_OCCURRENCE_PLAN.replace(
    '"series_terms": {"study": ["study", "homework"],\n'
    '                   "exercise": ["gym", "workout"]},',
    '"series_terms": {},',
)


class TestSeriesFromQueryBackstop:
    def test_live_turn1_phrasing(self):
        from core.insight.deliberation import _series_from_query
        series = _series_from_query(
            "Examine whether my social contact and household-maintenance "
            "activity co-occur across my recorded history from 2024-10-01 "
            "through 2025-03-31."
        )
        assert set(series) == {"social contact", "household maintenance"}
        assert "social contact" in series["social contact"]
        assert "household" in series["household maintenance"]

    def test_live_turn3_phrasing(self):
        from core.insight.deliberation import _series_from_query
        series = _series_from_query(
            "Analyze whether my recorded sleep quality and productive work "
            "tend to co-occur during 2025-01-01 through 2025-06-30."
        )
        assert set(series) == {"sleep quality", "productive work"}

    def test_no_conjunction_yields_nothing(self):
        from core.insight.deliberation import _series_from_query
        assert _series_from_query("how often have I mentioned headaches") == {}

    @pytest.mark.asyncio
    async def test_planner_omitting_series_recovers_from_query(self, monkeypatch):
        import config.app_config as app_config
        monkeypatch.setattr(app_config, "INSIGHT_PLANNER_MODEL", None, raising=False)
        monkeypatch.setattr(app_config, "RESPONSE_REVIEW_MODEL", None, raising=False)
        mm = _FakeModelManager(_CO_OCCURRENCE_PLAN_NO_SERIES)
        result = await plan_deliberation(
            "Examine whether my social contact and household-maintenance "
            "activity co-occur across my recorded history from 2024-10-01 "
            "through 2025-03-31.",
            model_manager=mm,
        )
        # The live turn died here with "co_occurrence requires at least two
        # named series" — the request's own conjunction now recovers them.
        assert result.status == "ready", result.limitations
        assert set(result.spec.series_terms) == {
            "social contact", "household maintenance",
        }
        assert any("conjunction" in a for a in result.spec.assumptions)


class TestPersonalClaimNeedsInternalEvidence:
    @pytest.mark.asyncio
    async def test_external_only_support_demoted_when_pattern_empty(self):
        import json as _json
        from core.insight.coordinator import ChannelStatus, assess_claim_chain

        proposal = {
            "analysis_kind": "period_comparison",
            "claims": [{"claim_id": "A",
                        "proposition": "my sleep and work co-occur",
                        "dependencies": [], "authority": "assessment",
                        "required_channels": ["pattern"],
                        "evidence_standard": "dated personal observations"}],
            "outcome_terms": ["sleep", "work"],
            "behavioral_indicators": [],
            "directional_indicators": {"increase": [], "decrease": []},
            "phases": [
                {"label": "jan", "start": "2025-01-01", "end": "2025-01-31"},
                {"label": "feb", "start": "2025-02-01", "end": "2025-02-28"},
            ],
            "requested_channels": ["pattern", "corpus"],
            "research_queries": {},
            "assumptions": [],
        }
        frozen = validate_and_freeze(proposal)
        assert frozen.status == "ready", frozen.limitations
        external = [{"pmid": "1", "title": "sleep and work in nurses",
                     "abstract": "sleep quality predicted productivity",
                     "source_class": "pubmed"}]
        channels = [
            ChannelStatus("pattern", "succeeded", attempted=True, count=0),
            ChannelStatus("pubmed", "succeeded", attempted=True, count=1),
        ]
        verdict = _json.dumps({"claims": [{
            "claim_id": "A", "status": "partially_supported",
            "confidence": 0.45, "coverage": "one external study",
            "directness": "indirect", "support_source_ids": ["pmid:1"],
            "refute_source_ids": [], "rationale": "population research",
        }]})
        chain = await assess_claim_chain(
            frozen.spec, None, external, channels, [],
            model_manager=_FakeModelManager(verdict),
        )
        # 2026-08-31 live turn 3: this exact shape shipped partially_supported
        # at 0.45 on one nurse study with ZERO personal observations.
        assert chain[0].status == "insufficient"
        assert chain[0].confidence <= 0.25
        assert "pattern" in chain[0].rationale

    @pytest.mark.asyncio
    async def test_internal_evidence_present_keeps_verdict(self):
        import json as _json
        from core.insight.coordinator import ChannelStatus, assess_claim_chain

        proposal = {
            "analysis_kind": "period_comparison",
            "claims": [{"claim_id": "A",
                        "proposition": "my sleep and work co-occur",
                        "dependencies": [], "authority": "assessment",
                        "required_channels": ["pattern"],
                        "evidence_standard": "dated personal observations"}],
            "outcome_terms": ["sleep", "work"],
            "behavioral_indicators": [],
            "directional_indicators": {"increase": [], "decrease": []},
            "phases": [
                {"label": "jan", "start": "2025-01-01", "end": "2025-01-31"},
                {"label": "feb", "start": "2025-02-01", "end": "2025-02-28"},
            ],
            "requested_channels": ["pattern", "corpus"],
            "research_queries": {},
            "assumptions": [],
        }
        frozen = validate_and_freeze(proposal)
        external = [{"pmid": "1", "title": "sleep and work",
                     "abstract": "x", "source_class": "pubmed"}]
        channels = [
            ChannelStatus("pattern", "succeeded", attempted=True, count=14),
            ChannelStatus("pubmed", "succeeded", attempted=True, count=1),
        ]
        verdict = _json.dumps({"claims": [{
            "claim_id": "A", "status": "partially_supported",
            "confidence": 0.45, "coverage": "personal + external",
            "directness": "mixed", "support_source_ids": ["pmid:1"],
            "refute_source_ids": [], "rationale": "ok",
        }]})
        chain = await assess_claim_chain(
            frozen.spec, None, external, channels, [],
            model_manager=_FakeModelManager(verdict),
        )
        assert chain[0].status == "partially_supported"


class TestPubMedQueryHygiene:
    def test_tool_names_never_become_query_axes(self):
        from knowledge.pubmed_search import _query_terms, build_pubmed_query_ladder
        terms = _query_terms(
            "Is there PubMed evidence on caffeine and sleep fragmentation? "
            "Check it against my own recorded history for the last two weeks."
        )
        assert "pubmed" not in terms
        assert terms[0] == "caffeine"
        ladder = build_pubmed_query_ladder(
            "Is there PubMed evidence on caffeine and sleep fragmentation?"
        )
        assert all("pubmed" not in q.lower() for q in ladder)

    def test_anchor_concept_required_for_relevance(self):
        from knowledge.pubmed_search import rank_pubmed_rows
        rows = [
            # matches only the non-anchor concepts — the live turn's junk
            # shape (sleep-quality concept paper for a caffeine question)
            {"pmid": "junk", "title": "Sleep quality: concept analysis",
             "abstract": "sleep quality fragmentation defined"},
            # matches anchor + another concept — a real topical paper
            {"pmid": "real", "title": "Caffeine effects on sleep",
             "abstract": "caffeine increased sleep fragmentation"},
        ]
        ranked = rank_pubmed_rows(rows, "caffeine sleep fragmentation")
        pmids = [row["pmid"] for row in ranked]
        assert pmids == ["real"]


class TestNoteDateInference:
    """2026-08-31 round 3: after the full reindex, ZERO chunks carried a
    2024/2025 note_date — legacy dailies use a dual-frontmatter template
    (prose block first, `date:` in a LATER block) or encode the date only in
    the filename/heading, so every pre-2026 daily was invisible to windowed
    longitudinal scans."""

    def _om(self):
        from knowledge.obsidian_manager import ObsidianManager
        return ObsidianManager.__new__(ObsidianManager)

    _DUAL = (
        '---\n\n## 1. Daily Summary\n\nGot great sleep\n\n---\n'
        'type: "Daily Note"\ndate: 2025-03-01\ntags: mood: ""\n---\n\n'
        '# 📅 2025-03-01\n'
    )

    def test_dual_frontmatter_date_recovered(self):
        assert self._om()._infer_note_date(self._DUAL, "03 01 25 Daily Note", {}) == "2025-03-01"

    def test_first_block_frontmatter_still_authoritative(self):
        assert self._om()._infer_note_date(self._DUAL, "x", {"date": "2025-01-01"}) == "2025-01-01"

    def test_iso_filename(self):
        assert self._om()._infer_note_date("no dates", "2024-11-19", {}) == "2024-11-19"

    def test_short_us_filename_validated(self):
        om = self._om()
        assert om._infer_note_date("x", "02 19 25 Daily Note", {}) == "2025-02-19"
        # 45 is not a month/day — invalid short dates never bind
        assert om._infer_note_date("x", "19 45 25 note", {}) == ""

    def test_heading_date(self):
        assert self._om()._infer_note_date("# 📅 2025-03-01\ntext", "Untitled", {}) == "2025-03-01"

    def test_undated_note_stays_undated(self):
        assert self._om()._infer_note_date("just prose", "Chi-Squared Distribution", {}) == ""


# ---------------------------------------------------------------------------
# Round 3 — relative windows, closed-set channels, date-range retrieval arm
# ---------------------------------------------------------------------------


class TestRelativeWindowBuckets:
    def test_last_two_weeks(self):
        now = datetime(2026, 8, 31)
        buckets = _weekly_buckets_from_window(
            "Check it against my own recorded history for the last two weeks.",
            now=now,
        )
        assert len(buckets) == 2
        assert buckets[-1]["end"] == "2026-08-31"
        assert buckets[0]["start"] == "2026-08-18"

    def test_past_numeric_months(self):
        now = datetime(2026, 8, 31)
        buckets = _weekly_buckets_from_window(
            "over the past 3 months", now=now)
        assert buckets and buckets[-1]["end"] == "2026-08-31"
        assert len(buckets) == 13  # 90 days of weekly buckets

    def test_explicit_dates_still_win(self):
        buckets = _weekly_buckets_from_window(
            "during 2025-01-01 through 2025-06-30 over the last two weeks")
        assert buckets[0]["start"] == "2025-01-01"

    def test_nonsense_counts_rejected(self):
        assert _weekly_buckets_from_window("the last 900 weeks") == []


class TestClosedSetChannelRestriction:
    @pytest.mark.asyncio
    async def test_use_my_corpus_and_notes_drops_pubmed(self, monkeypatch):
        import config.app_config as app_config
        monkeypatch.setattr(app_config, "INSIGHT_PLANNER_MODEL", None, raising=False)
        monkeypatch.setattr(app_config, "RESPONSE_REVIEW_MODEL", None, raising=False)
        plan = _CO_OCCURRENCE_PLAN.replace(
            '"requested_channels": ["pattern", "corpus"],',
            '"requested_channels": ["pattern", "corpus", "notes", "pubmed"],'
            ' "research_queries": {"pubmed": "sleep productivity"},',
        ).replace('"research_queries": {},', '')
        mm = _FakeModelManager(plan)
        result = await plan_deliberation(
            "Analyze whether my sleep and work co-occur during 2025-01-01 "
            "through 2025-06-30. Use my corpus and notes.",
            model_manager=mm,
        )
        assert result.status == "ready", result.limitations
        assert "pubmed" not in result.spec.requested_channels
        assert "notes" in result.spec.requested_channels
        assert "pubmed" not in result.spec.research_queries

    def test_no_restriction_cue_means_no_restriction(self):
        from core.insight.deliberation import _restricted_channel_set, DEFAULT_CHANNELS
        assert _restricted_channel_set(
            "check my notes and also PubMed please", DEFAULT_CHANNELS) is None
        assert _restricted_channel_set(
            "compare my sleep and work", DEFAULT_CHANNELS) is None
        restricted = _restricted_channel_set(
            "Use my corpus and notes.",
            DEFAULT_CHANNELS,
        )
        assert restricted == {"corpus", "notes", "pattern"} | {"corpus", "pattern"}


class TestWindowedRetrievalArm:
    @pytest.mark.asyncio
    async def test_window_offered_to_window_aware_adapters(self):
        from core.insight.coordinator import LongitudinalDeliberationCoordinator

        seen = {}

        async def notes_adapter(query, window=None):
            seen["notes_window"] = window
            return [{"text": "cleaned the kitchen", "timestamp": "2025-01-10"}]

        async def web_adapter(query):
            seen["web_called"] = True
            return []

        proposal = {
            "analysis_kind": "period_comparison",
            "claims": [{"claim_id": "A", "proposition": "chores differed",
                        "dependencies": [], "authority": "assessment",
                        "required_channels": [], "evidence_standard": ""}],
            "outcome_terms": ["cleaning"],
            "behavioral_indicators": [],
            "directional_indicators": {"increase": [], "decrease": []},
            "phases": [
                {"label": "jan", "start": "2025-01-01", "end": "2025-01-31"},
                {"label": "feb", "start": "2025-02-01", "end": "2025-02-28"},
            ],
            "requested_channels": ["pattern", "notes", "web"],
            "research_queries": {"web": "cleaning habits"},
            "assumptions": [],
        }
        coordinator = LongitudinalDeliberationCoordinator(
            adapters={"notes": notes_adapter, "web": web_adapter},
        )
        await coordinator.run("compare my cleaning in jan vs feb 2025",
                              proposed_spec=proposal)
        # notes (window-aware, internal) got the frozen window; web did not
        # blow up despite lacking the kwarg.
        assert seen["notes_window"] == ("2025-01-01", "2025-02-28")
        assert seen.get("web_called")

    def test_window_scan_filters_sorts_and_samples(self):
        import gui.handlers as handlers

        docs, metas, ids = [], [], []
        for day in range(1, 21):  # 20 dated chunks in Jan 2025
            docs.append(f"note {day}")
            metas.append({"note_date": f"2025-01-{day:02d}"})
            ids.append(f"c{day}")
        docs.append("out of window")
        metas.append({"note_date": "2026-05-01"})
        ids.append("out")
        docs.append("undated")
        metas.append({})
        ids.append("undated")

        class FakeColl:
            def get(self, include=None):
                return {"documents": docs, "metadatas": metas, "ids": ids}

        store = SimpleNamespace(_get_collection=lambda name: FakeColl())
        rows = handlers._window_scan_collection(
            store, "obsidian_notes", ("2025-01-01", "2025-01-31"), 10)
        assert len(rows) == 10
        dates = [r["metadata"]["note_date"] for r in rows]
        assert dates == sorted(dates)
        # even sampling keeps both window edges represented
        assert dates[0] <= "2025-01-03" and dates[-1] >= "2025-01-18"
        assert all("2025-01" in d for d in dates)

    def test_window_scan_failure_degrades_to_empty(self):
        import gui.handlers as handlers
        store = SimpleNamespace(
            _get_collection=lambda name: (_ for _ in ()).throw(RuntimeError("boom")))
        assert handlers._window_scan_collection(
            store, "obsidian_notes", ("2025-01-01", "2025-01-31"), 10) == []
