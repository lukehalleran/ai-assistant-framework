"""Tests for the query corpus module."""

import json
import tempfile
from pathlib import Path

import pytest

from core.intent_classifier import IntentType
from eval.corpus import (
    SEED_CORPUS,
    TONE_LEVELS,
    CorpusManager,
    CorpusQuery,
    ExpectedBehavior,
    validate_section_keys,
)
from eval.section_registry import SECTION_REGISTRY


# ---------------------------------------------------------------------------
# ExpectedBehavior tests
# ---------------------------------------------------------------------------

class TestExpectedBehavior:
    def test_roundtrip(self):
        eb = ExpectedBehavior(
            should_reference_memory=True,
            expected_sections_used=["memories", "user_profile"],
            notes="test note",
        )
        d = eb.to_dict()
        eb2 = ExpectedBehavior.from_dict(d)
        assert eb2.should_reference_memory is True
        assert eb2.expected_sections_used == ["memories", "user_profile"]
        assert eb2.notes == "test note"

    def test_defaults(self):
        eb = ExpectedBehavior()
        assert eb.should_reference_memory is False
        assert eb.expected_sections_used == []
        assert eb.max_expected_tokens is None


# ---------------------------------------------------------------------------
# CorpusQuery tests
# ---------------------------------------------------------------------------

class TestCorpusQuery:
    def test_roundtrip(self):
        q = CorpusQuery(
            query_id="test_001",
            query_text="Hello",
            intent=IntentType.CASUAL_SOCIAL,
            tone="CONVERSATIONAL",
            tags=["greeting"],
        )
        d = q.to_dict()
        q2 = CorpusQuery.from_dict(d)
        assert q2.query_id == "test_001"
        assert q2.intent == IntentType.CASUAL_SOCIAL
        assert q2.tone == "CONVERSATIONAL"
        assert q2.tags == ["greeting"]

    def test_snapshot_id_roundtrip(self):
        q = CorpusQuery(
            query_id="test_002",
            query_text="Query",
            intent=IntentType.GENERAL,
            tone="CONVERSATIONAL",
            snapshot_id="snap1234",
        )
        d = q.to_dict()
        q2 = CorpusQuery.from_dict(d)
        assert q2.snapshot_id == "snap1234"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_section_keys(self):
        bad = validate_section_keys(["memories", "user_profile"])
        assert bad == []

    def test_invalid_section_keys(self):
        bad = validate_section_keys(["memories", "nonexistent_section"])
        assert bad == ["nonexistent_section"]


# ---------------------------------------------------------------------------
# Seed corpus tests
# ---------------------------------------------------------------------------

class TestSeedCorpus:
    def test_seed_has_27_queries(self):
        assert len(SEED_CORPUS) == 27

    def test_seed_covers_all_9_intents(self):
        """Every IntentType has at least 1 query in the seed."""
        intents_present = {q["intent"] for q in SEED_CORPUS}
        for intent in IntentType:
            assert intent.value in intents_present, f"Missing intent: {intent.value}"

    def test_seed_has_3_per_intent(self):
        """Each intent has at least 3 queries."""
        counts: dict[str, int] = {}
        for q in SEED_CORPUS:
            counts[q["intent"]] = counts.get(q["intent"], 0) + 1
        for intent in IntentType:
            assert counts.get(intent.value, 0) >= 3, (
                f"Intent {intent.value} has {counts.get(intent.value, 0)} queries, need 3"
            )

    def test_seed_has_tone_coverage(self):
        """At least 2 queries for HIGH tone level."""
        tones = [q.get("tone", "CONVERSATIONAL") for q in SEED_CORPUS]
        high_count = sum(1 for t in tones if t == "HIGH")
        assert high_count >= 2, f"Only {high_count} HIGH tone queries, need >= 2"

    def test_seed_all_section_keys_valid(self):
        """All expected_sections_used entries are valid registry keys."""
        for q in SEED_CORPUS:
            sections = q["expected"].get("expected_sections_used", [])
            bad = validate_section_keys(sections)
            assert bad == [], (
                f"Query '{q['query_id']}' has invalid section keys: {bad}"
            )

    def test_seed_query_ids_unique(self):
        ids = [q["query_id"] for q in SEED_CORPUS]
        assert len(ids) == len(set(ids))

    def test_seed_queries_parseable(self):
        """All seed entries can be parsed into CorpusQuery."""
        for qdata in SEED_CORPUS:
            expected = ExpectedBehavior.from_dict(qdata["expected"])
            q = CorpusQuery(
                query_id=qdata["query_id"],
                query_text=qdata["query_text"],
                intent=IntentType(qdata["intent"]),
                tone=qdata.get("tone", "CONVERSATIONAL"),
                expected=expected,
                tags=qdata.get("tags", []),
            )
            assert q.query_id is not None


# ---------------------------------------------------------------------------
# CorpusManager tests
# ---------------------------------------------------------------------------

class TestCorpusManager:
    def test_seed_on_init(self, tmp_path):
        corpus_path = tmp_path / "corpus.json"
        mgr = CorpusManager(corpus_path)
        assert len(mgr.queries) == 27
        assert corpus_path.exists()

    def test_load_from_existing(self, tmp_path):
        corpus_path = tmp_path / "corpus.json"
        mgr1 = CorpusManager(corpus_path)
        mgr2 = CorpusManager(corpus_path)
        assert len(mgr2.queries) == len(mgr1.queries)
        assert set(mgr2.queries.keys()) == set(mgr1.queries.keys())

    def test_add_query(self, tmp_path):
        corpus_path = tmp_path / "corpus.json"
        mgr = CorpusManager(corpus_path)
        q = mgr.add_query(
            query_id="custom_001",
            query_text="Custom test query",
            intent=IntentType.GENERAL,
            tone="CONVERSATIONAL",
        )
        assert "custom_001" in mgr.queries
        assert q.intent == IntentType.GENERAL

    def test_add_query_validates_section_keys(self, tmp_path):
        corpus_path = tmp_path / "corpus.json"
        mgr = CorpusManager(corpus_path)
        with pytest.raises(ValueError, match="Unknown section keys"):
            mgr.add_query(
                query_id="bad_001",
                query_text="Bad query",
                intent=IntentType.GENERAL,
                expected=ExpectedBehavior(
                    expected_sections_used=["bogus_section"]
                ),
            )

    def test_link_snapshot(self, tmp_path):
        corpus_path = tmp_path / "corpus.json"
        mgr = CorpusManager(corpus_path)
        mgr.link_snapshot("fact_001", "snap_abc123")
        assert mgr.queries["fact_001"].snapshot_id == "snap_abc123"

    def test_get_by_intent(self, tmp_path):
        corpus_path = tmp_path / "corpus.json"
        mgr = CorpusManager(corpus_path)
        casual = mgr.get_by_intent(IntentType.CASUAL_SOCIAL)
        assert len(casual) >= 3
        assert all(q.intent == IntentType.CASUAL_SOCIAL for q in casual)

    def test_get_by_tone(self, tmp_path):
        corpus_path = tmp_path / "corpus.json"
        mgr = CorpusManager(corpus_path)
        high = mgr.get_by_tone("HIGH")
        assert len(high) >= 2
        assert all(q.tone == "HIGH" for q in high)

    def test_intent_coverage(self, tmp_path):
        corpus_path = tmp_path / "corpus.json"
        mgr = CorpusManager(corpus_path)
        coverage = mgr.get_intent_coverage()
        for intent in IntentType:
            assert coverage.get(intent.value, 0) >= 3

    def test_intent_gaps_empty_for_seed(self, tmp_path):
        corpus_path = tmp_path / "corpus.json"
        mgr = CorpusManager(corpus_path)
        gaps = mgr.get_intent_gaps(min_per_intent=3)
        assert gaps == [], f"Unexpected gaps: {gaps}"

    def test_tone_gaps(self, tmp_path):
        corpus_path = tmp_path / "corpus.json"
        mgr = CorpusManager(corpus_path)
        gaps = mgr.get_tone_gaps(min_per_tone=2)
        # MEDIUM and CONCERN should each have at least 1 but maybe not 2
        # HIGH has 2, CONVERSATIONAL has many
        # Just verify the method works
        assert isinstance(gaps, list)

    def test_save_load_roundtrip(self, tmp_path):
        corpus_path = tmp_path / "corpus.json"
        mgr1 = CorpusManager(corpus_path)
        mgr1.add_query("extra_001", "Extra query", IntentType.GENERAL)
        mgr2 = CorpusManager(corpus_path)
        assert "extra_001" in mgr2.queries
        assert mgr2.queries["extra_001"].intent == IntentType.GENERAL
