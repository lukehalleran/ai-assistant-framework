"""Tests for section utilization analysis."""

import pytest

from core.intent_classifier import IntentType
from eval.corpus import CorpusQuery, ExpectedBehavior
from eval.schema import (
    PromptProvenance,
    PromptSnapshot,
    SectionSnapshot,
    SnapshotLayer,
    compute_prompt_hash,
)
from eval.section_registry import SECTION_REGISTRY
from eval.utilization import SectionUtilization, UtilizationAnalyzer, UtilizationReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_section(key: str, text: str = "", token_count: int = 0) -> SectionSnapshot:
    reg = SECTION_REGISTRY.get(key)
    if reg is None:
        raise ValueError(f"Unknown section key: {key}")
    return SectionSnapshot(
        key=key,
        header=reg.header,
        structured_content=text if text else None,
        formatted_text=text,
        token_count=token_count if token_count else max(1, len(text) // 4),
        source_field=reg.source_field,
        category=reg.category.value,
        eligible_for_ablation=reg.eligible_for_ablation,
        structurally_required=reg.structurally_required,
        assembly_order=reg.assembly_order,
    )


def _make_snapshot(
    section_map: dict[str, str],
    snapshot_id: str = "snap0001",
) -> PromptSnapshot:
    """Build a snapshot where section_map maps key -> formatted_text."""
    sections = {}
    for key, text in section_map.items():
        sections[key] = _make_section(key, text=text, token_count=max(1, len(text) // 4))

    prompt_text = "\n\n".join(
        s.formatted_text
        for s in sorted(sections.values(), key=lambda s: s.assembly_order)
        if s.formatted_text
    )

    layer = SnapshotLayer(
        layer_name="post_hygiene",
        sections=sections,
        layer_content_hash="fakehash",
        prompt_text=prompt_text,
        prompt_hash_exact=compute_prompt_hash(prompt_text) if prompt_text else None,
        prompt_hash_normalized=compute_prompt_hash(prompt_text, normalize=True) if prompt_text else None,
        capture_timestamp="2026-05-05T00:00:00+00:00",
    )

    provenance = PromptProvenance(
        model_name="test-model",
        git_commit_hash="abc1234",
        system_prompt_hash="sys_hash",
    )

    return PromptSnapshot(
        snapshot_id=snapshot_id,
        query_text="test query",
        query_timestamp="2026-05-05T00:00:00+00:00",
        processed_query="test query",
        detected_intent="general",
        detected_tone="CONVERSATIONAL",
        provenance=provenance,
        layers={"post_hygiene": layer},
        retrieval_metadata={},
        assembly_metadata={},
    )


def _make_query(query_id: str, intent: IntentType = IntentType.GENERAL) -> CorpusQuery:
    return CorpusQuery(
        query_id=query_id,
        query_text="test",
        intent=intent,
        tone="CONVERSATIONAL",
    )


# ---------------------------------------------------------------------------
# SectionUtilization tests
# ---------------------------------------------------------------------------

class TestSectionUtilization:
    def test_presence_rate(self):
        u = SectionUtilization(section_key="memories", category="retrieved")
        u.total_queries = 10
        u.times_present = 7
        assert u.presence_rate == pytest.approx(0.7)

    def test_presence_rate_zero_queries(self):
        u = SectionUtilization(section_key="memories", category="retrieved")
        assert u.presence_rate == 0.0

    def test_nonempty_rate(self):
        u = SectionUtilization(section_key="memories", category="retrieved")
        u.total_queries = 10
        u.times_nonempty = 3
        assert u.nonempty_rate == pytest.approx(0.3)

    def test_avg_tokens_when_present(self):
        u = SectionUtilization(section_key="memories", category="retrieved")
        u.times_present = 4
        u.total_tokens = 400
        assert u.avg_tokens_when_present == pytest.approx(100.0)

    def test_avg_tokens_zero_present(self):
        u = SectionUtilization(section_key="memories", category="retrieved")
        assert u.avg_tokens_when_present == 0.0


# ---------------------------------------------------------------------------
# UtilizationAnalyzer tests
# ---------------------------------------------------------------------------

class TestUtilizationAnalyzer:
    def test_empty_sections_detected(self):
        """Sections not in any snapshot have 0% presence."""
        snap1 = _make_snapshot({
            "current_query": "[CURRENT USER QUERY]\ntest",
            "time_context": "[TIME CONTEXT]\nnow",
        })
        corpus = {"q1": _make_query("q1")}
        snapshots = {"q1": snap1}

        analyzer = UtilizationAnalyzer()
        report = analyzer.analyze(snapshots, corpus)

        # memories is not in the snapshot, so 0% presence
        assert report.sections["memories"].presence_rate == 0.0
        assert "memories" in report.always_empty

    def test_always_present_detected(self):
        """Sections present in every snapshot have 100% presence."""
        snap1 = _make_snapshot({
            "current_query": "[CURRENT USER QUERY]\ntest1",
            "memories": "[RELEVANT MEMORIES]\nsome memory content that is long enough",
        }, snapshot_id="s1")
        snap2 = _make_snapshot({
            "current_query": "[CURRENT USER QUERY]\ntest2",
            "memories": "[RELEVANT MEMORIES]\nanother memory content that is long enough",
        }, snapshot_id="s2")

        corpus = {"q1": _make_query("q1"), "q2": _make_query("q2")}
        snapshots = {"q1": snap1, "q2": snap2}

        analyzer = UtilizationAnalyzer()
        report = analyzer.analyze(snapshots, corpus)

        assert report.sections["memories"].presence_rate == 1.0
        assert report.sections["current_query"].presence_rate == 1.0
        assert "memories" in report.always_present
        assert "current_query" in report.always_present

    def test_high_variance_classification(self):
        """Sections with 20-80% presence classified as high_variance."""
        # 3 snapshots: wiki in 2 of 3 = 66%
        snap1 = _make_snapshot({"current_query": "q1", "wiki": "wiki content here"}, "s1")
        snap2 = _make_snapshot({"current_query": "q2", "wiki": "wiki content here"}, "s2")
        snap3 = _make_snapshot({"current_query": "q3"}, "s3")

        corpus = {
            "q1": _make_query("q1"),
            "q2": _make_query("q2"),
            "q3": _make_query("q3"),
        }
        snapshots = {"q1": snap1, "q2": snap2, "q3": snap3}

        analyzer = UtilizationAnalyzer()
        report = analyzer.analyze(snapshots, corpus)

        wiki_rate = report.sections["wiki"].presence_rate
        assert 0.2 <= wiki_rate <= 0.8
        assert "wiki" in report.high_variance

    def test_intent_specific_detection(self):
        """Section only appearing for one intent is flagged."""
        snap1 = _make_snapshot(
            {"current_query": "q1", "web_search_results": "web results content here"},
            "s1",
        )
        snap2 = _make_snapshot({"current_query": "q2"}, "s2")

        corpus = {
            "q1": _make_query("q1", IntentType.GENERAL),
            "q2": _make_query("q2", IntentType.CASUAL_SOCIAL),
        }
        snapshots = {"q1": snap1, "q2": snap2}

        analyzer = UtilizationAnalyzer()
        report = analyzer.analyze(snapshots, corpus)

        # web_search_results only appears for GENERAL intent
        assert report.sections["web_search_results"].times_present == 1
        assert "general" in report.intent_specific
        assert "web_search_results" in report.intent_specific["general"]

    def test_token_averaging(self):
        """avg_tokens_when_present calculated correctly."""
        snap1 = _make_snapshot({
            "current_query": "q1",
            "memories": "x" * 400,  # ~100 tokens
        }, "s1")
        snap2 = _make_snapshot({
            "current_query": "q2",
            "memories": "y" * 800,  # ~200 tokens
        }, "s2")

        corpus = {"q1": _make_query("q1"), "q2": _make_query("q2")}
        snapshots = {"q1": snap1, "q2": snap2}

        analyzer = UtilizationAnalyzer()
        report = analyzer.analyze(snapshots, corpus)

        mem_util = report.sections["memories"]
        assert mem_util.times_present == 2
        assert mem_util.avg_tokens_when_present == pytest.approx(
            mem_util.total_tokens / 2
        )

    def test_format_report_runs(self):
        """format_report produces non-empty string."""
        snap1 = _make_snapshot({"current_query": "q1"}, "s1")
        corpus = {"q1": _make_query("q1")}
        snapshots = {"q1": snap1}

        analyzer = UtilizationAnalyzer()
        report = analyzer.analyze(snapshots, corpus)
        text = analyzer.format_report(report)

        assert len(text) > 0
        assert "Section Utilization Report" in text
        assert "Corpus: 1 queries" in text

    def test_corpus_size_matches(self):
        snap1 = _make_snapshot({"current_query": "q1"}, "s1")
        snap2 = _make_snapshot({"current_query": "q2"}, "s2")
        corpus = {"q1": _make_query("q1"), "q2": _make_query("q2")}
        snapshots = {"q1": snap1, "q2": snap2}

        analyzer = UtilizationAnalyzer()
        report = analyzer.analyze(snapshots, corpus)
        assert report.corpus_size == 2

    def test_missing_layer_still_counts(self):
        """Snapshot with no post_hygiene layer still counts toward total_queries."""
        snap = _make_snapshot({"current_query": "q1"}, "s1")
        # Remove the post_hygiene layer
        snap.layers.clear()

        corpus = {"q1": _make_query("q1")}
        snapshots = {"q1": snap}

        analyzer = UtilizationAnalyzer()
        report = analyzer.analyze(snapshots, corpus)

        # total_queries should be 1 for all sections
        for util in report.sections.values():
            assert util.total_queries == 1
            assert util.times_present == 0

    def test_nonempty_vs_empty_distinction(self):
        """Short text (< 50 chars) counts as present but empty."""
        snap = _make_snapshot({
            "current_query": "[CURRENT USER QUERY]\ntest",
            "memories": "tiny",  # < 50 chars
        }, "s1")

        corpus = {"q1": _make_query("q1")}
        snapshots = {"q1": snap}

        analyzer = UtilizationAnalyzer()
        report = analyzer.analyze(snapshots, corpus)

        mem = report.sections["memories"]
        assert mem.times_present == 1
        assert mem.times_nonempty == 0  # text too short
        assert mem.times_empty == 1

    def test_report_to_dict(self):
        snap = _make_snapshot({"current_query": "q1", "memories": "content " * 20}, "s1")
        corpus = {"q1": _make_query("q1")}
        snapshots = {"q1": snap}

        analyzer = UtilizationAnalyzer()
        report = analyzer.analyze(snapshots, corpus)
        d = report.to_dict()

        assert "corpus_size" in d
        assert "sections" in d
        assert "always_empty" in d
        assert "memories" in d["sections"]
        assert "presence_rate" in d["sections"]["memories"]
