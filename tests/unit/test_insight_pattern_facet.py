"""Insight-mode pattern_temporal facet (2026-08-29).

Covers: detector shapes (temporal cues route to pattern_temporal; vents and
ambiguous requests stay on their existing routes), the deterministic window
parser, the temporal stage's engine orchestration + REAL-exemplar evidence,
and the synthesizer's pattern tail + computed-aggregate block.
"""

import json
from datetime import datetime, timedelta

import pytest

from core.insight.detector import detect_insight_request, parse_window_days
from core.insight.synthesizer import build_synthesis_prompts
from core.insight.temporal import run_pattern_stage, theme_keywords
from core.insight.types import InsightIntent

NOW = datetime(2026, 8, 29, 15, 0, 0)


class TestDetectorPatternShapes:
    @pytest.mark.parametrize("q,theme_word", [
        ("How often have I mentioned itching over the last month", "itching"),
        ("how many times have I talked about my mom this year", "mom"),
        ("Is my sleep getting worse", "sleep"),
        ("show me the trend in my anxiety over the past 3 months", "anxiety"),
        ("what's my pattern with heavy songs over time", "songs"),
        ("track my kavarin use over the last 2 weeks", "kavarin"),
        ("Has my sleep changed since I moved?", "sleep"),
        ("Compare how I was before and after starting the night shift", "compare"),
        ("What tends to happen when I skip breakfast?", "breakfast"),
        ("Does my mood track with exercise?", "mood"),
        ("Check whether my theory fits the record using PubMed", "theory"),
        ("Use my history and PubMed to decide whether my concentration changed", "history"),
    ])
    def test_temporal_cues_route_pattern(self, q, theme_word):
        intent = detect_insight_request(q)
        assert intent is not None, q
        assert intent.kind == "pattern_temporal", q
        assert theme_word in intent.theme.lower()

    def test_bare_pattern_theme_stays_theme_sweep_or_none(self):
        # no temporal cue → NOT pattern kind (may be None or theme_sweep)
        intent = detect_insight_request(
            "gather everything I've said about my pattern with my mom")
        assert intent is not None
        assert intent.kind == "theme_sweep"

    @pytest.mark.parametrize("q", [
        "I feel like I am way more reactive than normal",
        "I keep thinking about that apartment",
        "ugh my sleep was terrible again",
    ])
    def test_vents_and_reports_do_not_route(self, q):
        intent = detect_insight_request(q)
        assert intent is None or intent.kind != "pattern_temporal"

    def test_temporal_assessment_uses_longitudinal_owner(self):
        intent = detect_insight_request(
            "Am I right that my anxiety is getting worse over the last month?")
        assert intent is not None
        assert intent.kind == "pattern_temporal"

    def test_personal_doc_shape_sets_wants_document(self):
        intent = detect_insight_request(
            "write me a summary of the trend in my sleep over the last 3 "
            "months for my therapist")
        assert intent is not None
        assert intent.kind == "pattern_temporal"
        assert intent.wants_document is True


class TestWindowParser:
    @pytest.mark.parametrize("text,days", [
        ("over the last month", 30),
        ("in the past 2 weeks", 14),
        ("over the last 6 months", 180),
        ("past 10 days", 10),
        ("over the years", -1),
        ("all time", -1),
        ("this year", 365),
        ("no window named here", 0),
    ])
    def test_windows(self, text, days):
        assert parse_window_days(text) == days


class TestThemeKeywords:
    def test_stopwords_removed_order_kept(self):
        assert theme_keywords("how often have I mentioned the itching and kavarin") \
            == ["mentioned", "itching", "kavarin"]


def _mk_corpus(tmp_path):
    from memory.corpus_manager import CorpusManager
    entries = []
    for d in (10, 6, 3, 1):
        entries.append({"query": "the itching came back again",
                        "response": "noted",
                        "timestamp": (NOW - timedelta(days=d)).isoformat()})
    for d in range(12, 0, -1):
        entries.append({"query": f"filler {d}", "response": "ok",
                        "timestamp": (NOW - timedelta(days=d, hours=1)).isoformat()})
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(entries))
    return CorpusManager(corpus_file=str(path))


class TestTemporalStage:
    def test_stage_returns_patterns_and_real_evidence(self, tmp_path):
        intent = InsightIntent(kind="pattern_temporal",
                               theme="itching", window_days=14,
                               raw_query="how often have I mentioned itching")
        patterns, evidence = run_pattern_stage(
            intent, corpus_manager=_mk_corpus(tmp_path), now=NOW)
        assert patterns and patterns[0].dimension == "topic_keyword"
        assert patterns[0].total == 4
        assert evidence  # exemplar EvidenceItems
        for item in evidence:
            assert item.collection == "pattern"
            assert item.date          # real timestamps
            assert "itching" in item.text.lower()
            assert item.stance_label == "user-stated"

    def test_mood_theme_adds_tone_overlay(self, tmp_path):
        telem = tmp_path / "turns.jsonl"
        telem.write_text(json.dumps(
            {"ts": (NOW - timedelta(days=2)).isoformat(),
             "tone_level": "CONCERN"}) + "\n")
        intent = InsightIntent(kind="pattern_temporal",
                               theme="my anxiety", window_days=14)
        patterns, _ = run_pattern_stage(
            intent, corpus_manager=_mk_corpus(tmp_path),
            telemetry_path=str(telem), now=NOW)
        assert {p.dimension for p in patterns} == {"topic_keyword", "tone", "daily_notes"}

    def test_music_theme_adds_content_type_overlay(self, tmp_path):
        intent = InsightIntent(kind="pattern_temporal",
                               theme="heavy songs", window_days=14)
        patterns, _ = run_pattern_stage(
            intent, corpus_manager=_mk_corpus(tmp_path), now=NOW)
        assert {p.dimension for p in patterns} == {"topic_keyword", "content_type"}

    def test_never_raises_without_components(self):
        intent = InsightIntent(kind="pattern_temporal", theme="anything")
        patterns, evidence = run_pattern_stage(intent, now=NOW)
        assert isinstance(patterns, list) and isinstance(evidence, list)


class TestSynthesizerPatternPrompts:
    def _pattern(self, tmp_path):
        intent = InsightIntent(kind="pattern_temporal",
                               theme="itching", window_days=14)
        patterns, _ = run_pattern_stage(
            intent, corpus_manager=_mk_corpus(tmp_path), now=NOW)
        return intent, patterns

    def test_pattern_tail_and_aggregate_block(self, tmp_path):
        intent, patterns = self._pattern(tmp_path)
        system, user = build_synthesis_prompts(
            intent, [], None, patterns=patterns)
        assert "COMPUTED AGGREGATE" in system
        assert "never recount" in system
        assert "TREND CONFIDENCE" in system
        assert "Deterministic aggregate" in user
        assert "total=4" in user           # the engine's real number
        assert "user turns in window" in user

    def test_non_pattern_kinds_unchanged(self):
        intent = InsightIntent(kind="theme_sweep", theme="x")
        system, user = build_synthesis_prompts(intent, [], None)
        assert "COMPUTED AGGREGATE" not in system
        assert "ASSEMBLING evidence" in system

    def test_elevated_tail_still_applies(self, tmp_path):
        intent, patterns = self._pattern(tmp_path)
        system, _ = build_synthesis_prompts(
            intent, [], None, patterns=patterns, tone_elevated=True)
        assert "TONE GUARD" in system

    def test_deliberation_manifest_keeps_claims_separate_and_is_structurally_bounded(self):
        intent = InsightIntent(kind="pattern_temporal", theme="work and sleep")
        manifest = {
            "status": "ready",
            "spec": {
                "analysis_kind": "period_comparison",
                "claims": [
                    {"claim_id": "A", "proposition": "historical claim"},
                    {"claim_id": "B", "proposition": "research claim"},
                ],
                "outcome_terms": ["sleep"],
                "phases": [],
            },
            "external_sources": [
                {"source_id": f"s{i}", "source_class": "web", "title": "T" * 100,
                 "snippet": "x" * 2000, "url": f"https://example.test/{i}"}
                for i in range(30)
            ],
            "channels": [{"channel": "web", "status": "succeeded", "source_ids": [f"s{i}" for i in range(30)]}],
            "claim_chain": [
                {"claim_id": "A", "status": "supported", "rationale": "r" * 1000},
                {"claim_id": "B", "status": "mixed", "rationale": "r" * 1000},
            ],
            "limitations": [],
        }
        system, user = build_synthesis_prompts(
            intent, [], None, deliberation_manifest=manifest,
        )
        assert "CLAIM CHAIN" in system
        assert '"claim_id":"A"' in user and '"claim_id":"B"' in user
        assert '"manifest_compacted":true' in user
        assert len(user) < 24000


class TestHandlersWiring:
    def test_insight_runner_threads_patterns(self):
        # source-level pin: the runner runs the pattern stage and passes
        # patterns= into synthesize_stream (runtime path is covered by
        # test_insight_mode_handler's fake-run harness for the base mode).
        import inspect
        import gui.handlers as h
        src = inspect.getsource(h._run_insight_mode)
        assert "run_pattern_stage" in src
        assert "patterns=_patterns" in src
        assert "pattern_temporal" in src
