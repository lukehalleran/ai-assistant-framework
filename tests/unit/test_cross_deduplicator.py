# tests/unit/test_cross_deduplicator.py
"""
Unit tests for the CrossCollectionDeduplicator service.

Tests cover:
- Duplicate detection across and within collections
- Fact contradiction detection and resolution
- Keep/delete priority logic
- Dry-run vs live execution
- Edge cases (empty collections, no duplicates, missing metadata)
- DedupPlan markdown rendering
"""

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from memory.cross_deduplicator import CrossCollectionDeduplicator
from memory.dedup_models import (
    ContradictionCluster,
    DedupAction,
    DedupPlan,
    DedupReason,
    DuplicatePair,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

def _make_doc(doc_id, content, collection, metadata=None):
    """Helper to create a document dict."""
    meta = metadata or {}
    if "timestamp" not in meta:
        meta["timestamp"] = datetime.now().isoformat()
    return {
        "id": doc_id,
        "content": content,
        "metadata": meta,
        "collection": collection,
    }


def _make_chroma_store(collection_docs=None):
    """Create a mock ChromaDB store with test data.

    Args:
        collection_docs: dict of {collection_name: [doc_dicts]}
    """
    store = MagicMock()
    collection_docs = collection_docs or {}

    def mock_list_all(name):
        return collection_docs.get(name, [])

    store.list_all = MagicMock(side_effect=mock_list_all)

    # Mock collections dict for deletion
    mock_collections = {}
    for coll_name in collection_docs:
        mock_coll = MagicMock()
        mock_coll.delete = MagicMock()
        mock_collections[coll_name] = mock_coll
    store.collections = mock_collections

    return store


def _make_embedding_fn(dim=384):
    """Create a mock embedding function that returns deterministic vectors.

    Documents with identical content get identical embeddings.
    Different content gets random but consistent embeddings.
    """
    cache = {}

    def embed(texts):
        results = []
        for t in texts:
            if t not in cache:
                # Use hash-seeded random for reproducibility
                rng = np.random.RandomState(hash(t) % (2**31))
                vec = rng.randn(dim).astype(np.float32)
                vec /= np.linalg.norm(vec)
                cache[t] = vec
            results.append(cache[t].tolist())
        return results

    return embed


def _make_identical_embedding_fn(dim=384):
    """Embedding function where specific texts map to the same vector."""
    base_vec = np.random.RandomState(42).randn(dim).astype(np.float32)
    base_vec /= np.linalg.norm(base_vec)

    other_vec = np.random.RandomState(99).randn(dim).astype(np.float32)
    other_vec /= np.linalg.norm(other_vec)

    def embed(texts):
        results = []
        for t in texts:
            if "IDENTICAL" in t:
                results.append(base_vec.tolist())
            else:
                # Each unique text gets a different random vec
                rng = np.random.RandomState(hash(t) % (2**31))
                vec = rng.randn(dim).astype(np.float32)
                vec /= np.linalg.norm(vec)
                results.append(vec.tolist())
        return results

    return embed


# ---------------------------------------------------------------
# DedupModels tests
# ---------------------------------------------------------------

class TestDedupModels:
    """Tests for Pydantic data models."""

    def test_dedup_plan_defaults(self):
        plan = DedupPlan()
        assert plan.dry_run is True
        assert plan.duplicates_found == 0
        assert plan.contradictions_found == 0
        assert plan.deletions_executed == 0
        assert plan.duplicate_pairs == []
        assert plan.contradiction_clusters == []

    def test_dedup_plan_markdown_empty(self):
        plan = DedupPlan(collections_scanned=["facts", "summaries"])
        md = plan.to_markdown()
        assert "facts" in md
        assert "summaries" in md
        assert "DRY RUN" in md

    def test_dedup_plan_markdown_with_pairs(self):
        pair = DuplicatePair(
            doc_id_a="a1", doc_id_b="b1",
            collection_a="facts", collection_b="summaries",
            similarity=0.95,
            content_preview_a="User likes Python",
            content_preview_b="User likes Python programming",
            reason=DedupReason.CROSS_DUPLICATE,
            action=DedupAction.DELETE,
            keep_id="a1", delete_id="b1",
        )
        plan = DedupPlan(
            duplicate_pairs=[pair],
            duplicates_found=1,
            dry_run=False,
        )
        md = plan.to_markdown()
        assert "LIVE" in md
        assert "0.950" in md
        assert "Duplicate Pairs" in md

    def test_dedup_plan_markdown_with_contradictions(self):
        cluster = ContradictionCluster(
            subject="user",
            predicate="lives_in",
            entries=[
                {"id": "f1", "content": "user lives_in NYC"},
                {"id": "f2", "content": "user lives_in LA"},
            ],
            keep_id="f1",
            delete_ids=["f2"],
        )
        plan = DedupPlan(
            contradiction_clusters=[cluster],
            contradictions_found=1,
        )
        md = plan.to_markdown()
        assert "Fact Contradictions" in md
        assert "lives_in" in md
        assert "KEEP" in md
        assert "DELETE" in md

    def test_duplicate_pair_validation(self):
        pair = DuplicatePair(
            doc_id_a="a", doc_id_b="b",
            collection_a="facts", collection_b="facts",
            similarity=0.93,
            keep_id="a", delete_id="b",
        )
        assert pair.confidence == 1.0
        assert pair.reason == DedupReason.CROSS_DUPLICATE

    def test_duplicate_pair_similarity_bounds(self):
        with pytest.raises(Exception):
            DuplicatePair(
                doc_id_a="a", doc_id_b="b",
                collection_a="facts", collection_b="facts",
                similarity=1.5,  # out of bounds
                keep_id="a", delete_id="b",
            )


# ---------------------------------------------------------------
# CrossCollectionDeduplicator tests
# ---------------------------------------------------------------

class TestCrossCollectionDeduplicator:
    """Tests for the core deduplicator service."""

    def test_empty_collections(self):
        """No documents = no work."""
        store = _make_chroma_store({})
        store.embedding_fn = _make_embedding_fn()
        dedup = CrossCollectionDeduplicator(store)
        plan = dedup.run(dry_run=True)

        assert plan.total_docs_scanned == 0
        assert plan.duplicates_found == 0
        assert plan.contradictions_found == 0

    def test_single_document(self):
        """One document = nothing to compare."""
        store = _make_chroma_store({
            "facts": [_make_doc("f1", "User likes Python", "facts")],
        })
        store.embedding_fn = _make_embedding_fn()
        dedup = CrossCollectionDeduplicator(store)
        plan = dedup.run(dry_run=True)

        assert plan.total_docs_scanned == 1
        assert plan.duplicates_found == 0

    def test_no_duplicates(self):
        """Distinct documents should not be flagged."""
        store = _make_chroma_store({
            "facts": [
                _make_doc("f1", "User likes Python", "facts"),
                _make_doc("f2", "Weather is sunny today", "facts"),
            ],
            "summaries": [
                _make_doc("s1", "Had a discussion about machine learning", "summaries"),
            ],
        })
        store.embedding_fn = _make_embedding_fn()
        dedup = CrossCollectionDeduplicator(store)
        plan = dedup.run(dry_run=True)

        assert plan.total_docs_scanned == 3
        assert plan.duplicates_found == 0

    def test_cross_duplicate_detected(self):
        """Identical content across collections should be flagged."""
        store = _make_chroma_store({
            "facts": [
                _make_doc("f1", "IDENTICAL content across collections", "facts"),
            ],
            "summaries": [
                _make_doc("s1", "IDENTICAL content across collections", "summaries"),
            ],
        })
        store.embedding_fn = _make_identical_embedding_fn()
        dedup = CrossCollectionDeduplicator(store)
        plan = dedup.run(dry_run=True)

        assert plan.duplicates_found == 1
        pair = plan.duplicate_pairs[0]
        assert pair.reason == DedupReason.CROSS_DUPLICATE
        # Summaries have higher priority than facts, so fact should be deleted
        assert pair.keep_id == "s1"
        assert pair.delete_id == "f1"

    def test_within_collection_duplicate(self):
        """Identical content within same collection should be flagged."""
        store = _make_chroma_store({
            "facts": [
                _make_doc("f1", "IDENTICAL within same collection", "facts",
                          {"timestamp": "2026-01-01T00:00:00"}),
                _make_doc("f2", "IDENTICAL within same collection", "facts",
                          {"timestamp": "2026-02-01T00:00:00"}),
            ],
        })
        store.embedding_fn = _make_identical_embedding_fn()
        dedup = CrossCollectionDeduplicator(store)
        plan = dedup.run(dry_run=True)

        assert plan.duplicates_found == 1
        pair = plan.duplicate_pairs[0]
        assert pair.reason == DedupReason.WITHIN_DUPLICATE
        # More recent should be kept
        assert pair.keep_id == "f2"
        assert pair.delete_id == "f1"

    def test_dry_run_no_deletions(self):
        """Dry run should not call delete."""
        store = _make_chroma_store({
            "facts": [
                _make_doc("f1", "IDENTICAL dry run test", "facts"),
                _make_doc("f2", "IDENTICAL dry run test", "facts"),
            ],
        })
        store.embedding_fn = _make_identical_embedding_fn()
        dedup = CrossCollectionDeduplicator(store)
        plan = dedup.run(dry_run=True)

        assert plan.duplicates_found == 1
        # Verify no delete was called
        store.collections["facts"].delete.assert_not_called()

    def test_live_run_executes_deletions(self):
        """Live run should call delete on the store."""
        store = _make_chroma_store({
            "facts": [
                _make_doc("f1", "IDENTICAL live run test", "facts",
                          {"timestamp": "2026-01-01T00:00:00"}),
                _make_doc("f2", "IDENTICAL live run test", "facts",
                          {"timestamp": "2026-02-01T00:00:00"}),
            ],
        })
        store.embedding_fn = _make_identical_embedding_fn()
        dedup = CrossCollectionDeduplicator(store)
        plan = dedup.run(dry_run=False)

        assert plan.deletions_executed == 1
        store.collections["facts"].delete.assert_called_once_with(ids=["f1"])

    def test_protected_collections_excluded(self):
        """Protected collections (conversations, obsidian_notes, etc.) should never be scanned."""
        store = _make_chroma_store({
            "conversations": [
                _make_doc("c1", "IDENTICAL protected test", "conversations"),
            ],
            "facts": [
                _make_doc("f1", "IDENTICAL protected test", "facts"),
            ],
        })
        store.embedding_fn = _make_identical_embedding_fn()
        dedup = CrossCollectionDeduplicator(store)

        # conversations is protected, should not be in collections to scan
        assert "conversations" not in dedup.collections
        plan = dedup.run(dry_run=True)

        # Only facts scanned, and with a single doc no pair is possible
        assert plan.total_docs_scanned == 1
        assert plan.duplicates_found == 0


class TestKeepDeletePriority:
    """Tests for the collection priority and recency logic."""

    def _make_dedup(self):
        store = _make_chroma_store({})
        store.embedding_fn = _make_embedding_fn()
        return CrossCollectionDeduplicator(store)

    def test_summary_beats_fact(self):
        dedup = self._make_dedup()
        doc_fact = _make_doc("f1", "content", "facts")
        doc_summary = _make_doc("s1", "content", "summaries")
        keep, delete = dedup._pick_keep_delete(doc_fact, doc_summary)
        assert keep["id"] == "s1"
        assert delete["id"] == "f1"

    def test_reflection_beats_proposal(self):
        dedup = self._make_dedup()
        doc_refl = _make_doc("r1", "content", "reflections")
        doc_prop = _make_doc("p1", "content", "proposals")
        keep, delete = dedup._pick_keep_delete(doc_refl, doc_prop)
        assert keep["id"] == "r1"
        assert delete["id"] == "p1"

    def test_skill_beats_fact(self):
        dedup = self._make_dedup()
        doc_skill = _make_doc("sk1", "content", "procedural_skills")
        doc_fact = _make_doc("f1", "content", "facts")
        keep, delete = dedup._pick_keep_delete(doc_skill, doc_fact)
        assert keep["id"] == "sk1"
        assert delete["id"] == "f1"

    def test_same_collection_recency_wins(self):
        dedup = self._make_dedup()
        doc_old = _make_doc("f1", "content", "facts", {"timestamp": "2025-01-01T00:00:00"})
        doc_new = _make_doc("f2", "content", "facts", {"timestamp": "2026-02-01T00:00:00"})
        keep, delete = dedup._pick_keep_delete(doc_old, doc_new)
        assert keep["id"] == "f2"
        assert delete["id"] == "f1"

    def test_same_priority_different_collections(self):
        """Two unknown collections with same priority: recency wins."""
        dedup = self._make_dedup()
        doc_a = _make_doc("a1", "content", "custom_a", {"timestamp": "2026-01-01T00:00:00"})
        doc_b = _make_doc("b1", "content", "custom_b", {"timestamp": "2026-02-01T00:00:00"})
        keep, delete = dedup._pick_keep_delete(doc_a, doc_b)
        assert keep["id"] == "b1"
        assert delete["id"] == "a1"


class TestTimestampExtraction:
    """Tests for extracting timestamps from various metadata formats."""

    def _make_dedup(self):
        store = _make_chroma_store({})
        store.embedding_fn = _make_embedding_fn()
        return CrossCollectionDeduplicator(store)

    def test_iso_timestamp(self):
        dedup = self._make_dedup()
        doc = _make_doc("d1", "x", "facts", {"timestamp": "2026-01-15T10:30:00"})
        ts = dedup._extract_timestamp(doc)
        assert ts > 0

    def test_float_timestamp(self):
        dedup = self._make_dedup()
        doc = _make_doc("d1", "x", "facts", {"timestamp": 1700000000.0})
        ts = dedup._extract_timestamp(doc)
        assert ts == 1700000000.0

    def test_created_at_fallback(self):
        dedup = self._make_dedup()
        doc = _make_doc("d1", "x", "facts", {"created_at": "2026-01-15T10:30:00"})
        # Remove the auto-added timestamp
        doc["metadata"].pop("timestamp", None)
        ts = dedup._extract_timestamp(doc)
        assert ts > 0

    def test_missing_timestamp(self):
        dedup = self._make_dedup()
        doc = {"id": "d1", "content": "x", "metadata": {}, "collection": "facts"}
        ts = dedup._extract_timestamp(doc)
        assert ts == 0.0

    def test_invalid_timestamp_string(self):
        dedup = self._make_dedup()
        doc = _make_doc("d1", "x", "facts", {"timestamp": "not-a-date"})
        ts = dedup._extract_timestamp(doc)
        assert ts == 0.0


class TestFactContradictions:
    """Tests for fact contradiction detection."""

    def _make_dedup(self):
        store = _make_chroma_store({})
        store.embedding_fn = _make_embedding_fn()
        return CrossCollectionDeduplicator(store)

    def test_no_contradictions_with_same_object(self):
        """Same subject+predicate+object = not a contradiction."""
        dedup = self._make_dedup()
        docs = [
            _make_doc("f1", "x", "facts", {"subject": "user", "predicate": "lives_in", "object": "NYC"}),
            _make_doc("f2", "x", "facts", {"subject": "user", "predicate": "lives_in", "object": "NYC"}),
        ]
        clusters = dedup._find_fact_contradictions(docs)
        assert len(clusters) == 0

    def test_contradiction_detected(self):
        """Same subject+predicate, different object = contradiction."""
        dedup = self._make_dedup()
        docs = [
            _make_doc("f1", "user lives_in NYC", "facts", {
                "subject": "user", "predicate": "lives_in", "object": "NYC",
                "timestamp": "2025-01-01T00:00:00",
            }),
            _make_doc("f2", "user lives_in LA", "facts", {
                "subject": "user", "predicate": "lives_in", "object": "LA",
                "timestamp": "2026-02-01T00:00:00",
            }),
        ]
        clusters = dedup._find_fact_contradictions(docs)
        assert len(clusters) == 1
        assert clusters[0].subject == "user"
        assert clusters[0].predicate == "lives_in"
        # More recent (LA) should be kept
        assert clusters[0].keep_id == "f2"
        assert clusters[0].delete_ids == ["f1"]

    def test_contradiction_three_way(self):
        """Three contradicting facts: keep most recent, delete two."""
        dedup = self._make_dedup()
        docs = [
            _make_doc("f1", "x", "facts", {
                "subject": "user", "predicate": "job", "object": "engineer",
                "timestamp": "2024-01-01T00:00:00",
            }),
            _make_doc("f2", "x", "facts", {
                "subject": "user", "predicate": "job", "object": "manager",
                "timestamp": "2025-06-01T00:00:00",
            }),
            _make_doc("f3", "x", "facts", {
                "subject": "user", "predicate": "job", "object": "director",
                "timestamp": "2026-01-01T00:00:00",
            }),
        ]
        clusters = dedup._find_fact_contradictions(docs)
        assert len(clusters) == 1
        assert clusters[0].keep_id == "f3"
        assert set(clusters[0].delete_ids) == {"f1", "f2"}

    def test_no_contradictions_when_empty(self):
        dedup = self._make_dedup()
        clusters = dedup._find_fact_contradictions([])
        assert len(clusters) == 0

    def test_single_fact_no_contradiction(self):
        dedup = self._make_dedup()
        docs = [
            _make_doc("f1", "x", "facts", {"subject": "user", "predicate": "name", "object": "Luke"}),
        ]
        clusters = dedup._find_fact_contradictions(docs)
        assert len(clusters) == 0

    def test_case_insensitive_grouping(self):
        """Subject and predicate matching should be case-insensitive."""
        dedup = self._make_dedup()
        docs = [
            _make_doc("f1", "x", "facts", {
                "subject": "User", "predicate": "Lives_In", "object": "NYC",
                "timestamp": "2025-01-01T00:00:00",
            }),
            _make_doc("f2", "x", "facts", {
                "subject": "user", "predicate": "lives_in", "object": "LA",
                "timestamp": "2026-01-01T00:00:00",
            }),
        ]
        clusters = dedup._find_fact_contradictions(docs)
        assert len(clusters) == 1


class TestTripleExtraction:
    """Tests for extracting (subject, predicate, object) from fact content."""

    def _make_dedup(self):
        store = _make_chroma_store({})
        store.embedding_fn = _make_embedding_fn()
        return CrossCollectionDeduplicator(store)

    def test_extract_from_metadata(self):
        dedup = self._make_dedup()
        doc = _make_doc("f1", "content", "facts", {
            "subject": "Luke", "predicate": "likes", "object": "Python",
        })
        s, p, o = dedup._extract_triple(doc)
        assert s == "Luke"
        assert p == "likes"
        assert o == "Python"

    def test_extract_from_entity_relation_metadata(self):
        dedup = self._make_dedup()
        doc = _make_doc("f1", "content", "facts", {
            "entity": "Luke", "relation": "works_at", "value": "Acme",
        })
        s, p, o = dedup._extract_triple(doc)
        assert s == "Luke"
        assert p == "works_at"
        assert o == "Acme"

    def test_extract_from_pipe_content(self):
        dedup = self._make_dedup()
        doc = _make_doc("f1", "Luke | likes | Python programming", "facts")
        doc["metadata"] = {}  # no metadata fields
        s, p, o = dedup._extract_triple(doc)
        assert s == "Luke"
        assert p == "likes"
        assert o == "Python programming"

    def test_extract_from_dash_content(self):
        dedup = self._make_dedup()
        doc = _make_doc("f1", "Luke - lives_in - NYC", "facts")
        doc["metadata"] = {}
        s, p, o = dedup._extract_triple(doc)
        assert s == "Luke"
        assert p == "lives_in"
        assert o == "NYC"

    def test_extract_from_natural_language(self):
        dedup = self._make_dedup()
        doc = _make_doc("f1", "Luke is a software engineer", "facts")
        doc["metadata"] = {}
        s, p, o = dedup._extract_triple(doc)
        assert s == "Luke"
        assert p == "is"
        assert o == "a software engineer"

    def test_extract_fails_gracefully(self):
        dedup = self._make_dedup()
        doc = _make_doc("f1", "random text", "facts")
        doc["metadata"] = {}
        s, p, o = dedup._extract_triple(doc)
        # Should return empty strings, not crash
        assert isinstance(s, str)
        assert isinstance(p, str)
        assert isinstance(o, str)


class TestCascadePrevention:
    """Test that cascading deletions are prevented."""

    def test_no_cascade_three_way_duplicate(self):
        """If A=B and B=C, don't delete both B and C — only one per pair."""
        store = _make_chroma_store({
            "facts": [
                _make_doc("f1", "IDENTICAL cascade test", "facts",
                          {"timestamp": "2026-03-01T00:00:00"}),
                _make_doc("f2", "IDENTICAL cascade test", "facts",
                          {"timestamp": "2026-02-01T00:00:00"}),
                _make_doc("f3", "IDENTICAL cascade test", "facts",
                          {"timestamp": "2026-01-01T00:00:00"}),
            ],
        })
        store.embedding_fn = _make_identical_embedding_fn()
        dedup = CrossCollectionDeduplicator(store)
        plan = dedup.run(dry_run=True)

        # f1 is newest and should be kept
        # f2 and f3 should be deleted, but cascade prevention means
        # we process pairs: (f1,f2) -> delete f2, (f1,f3) -> delete f3
        # but (f2,f3) is skipped because f2 is already marked
        delete_ids = {p.delete_id for p in plan.duplicate_pairs}
        keep_ids = {p.keep_id for p in plan.duplicate_pairs}

        # f1 should never be deleted
        assert "f1" not in delete_ids
        # f1 should be the keep target
        assert "f1" in keep_ids


class TestMaxDocsCap:
    """Test per-collection document cap."""

    def test_cap_limits_documents(self):
        """Collections exceeding max_docs should be trimmed to most recent."""
        docs = [
            _make_doc(f"f{i}", f"Fact number {i}", "facts",
                      {"timestamp": f"2026-01-{i+1:02d}T00:00:00"})
            for i in range(20)
        ]
        store = _make_chroma_store({"facts": docs})
        store.embedding_fn = _make_embedding_fn()
        dedup = CrossCollectionDeduplicator(store)
        dedup.max_docs = 5  # Cap at 5

        all_docs, coll_map = dedup._load_all_documents()
        assert len(all_docs) == 5
        # Should have the most recent 5
        ids = {d["id"] for d in all_docs}
        for i in range(15, 20):
            assert f"f{i}" in ids


class TestEmbeddingFailure:
    """Test graceful handling of embedding computation failures."""

    def test_embedding_failure_returns_error_plan(self):
        store = _make_chroma_store({
            "facts": [
                _make_doc("f1", "content A", "facts"),
                _make_doc("f2", "content B", "facts"),
            ],
        })
        store.embedding_fn = MagicMock(side_effect=RuntimeError("model not loaded"))
        dedup = CrossCollectionDeduplicator(store)
        plan = dedup.run(dry_run=True)

        assert plan.duplicates_found == 0
        assert len(plan.errors) == 1
        assert "Failed to compute embeddings" in plan.errors[0]


class TestDeleteExecution:
    """Tests for live deletion execution."""

    def test_delete_contradiction_facts(self):
        """Live run should delete contradicting facts."""
        store = _make_chroma_store({
            "facts": [
                _make_doc("f1", "user lives_in NYC", "facts", {
                    "subject": "user", "predicate": "lives_in", "object": "NYC",
                    "timestamp": "2025-01-01T00:00:00",
                }),
                _make_doc("f2", "user lives_in LA", "facts", {
                    "subject": "user", "predicate": "lives_in", "object": "LA",
                    "timestamp": "2026-02-01T00:00:00",
                }),
            ],
        })
        store.embedding_fn = _make_embedding_fn()
        dedup = CrossCollectionDeduplicator(store)
        # Set high threshold so these aren't caught as semantic duplicates
        dedup.duplicate_threshold = 0.99
        plan = dedup.run(dry_run=False)

        assert plan.contradictions_found == 1
        # f1 (older) should be deleted
        store.collections["facts"].delete.assert_called_once_with(ids=["f1"])

    def test_delete_failure_logged(self):
        """Deletion errors should be captured in plan.errors, not raise."""
        store = _make_chroma_store({
            "facts": [
                _make_doc("f1", "IDENTICAL delete fail test", "facts",
                          {"timestamp": "2026-01-01T00:00:00"}),
                _make_doc("f2", "IDENTICAL delete fail test", "facts",
                          {"timestamp": "2026-02-01T00:00:00"}),
            ],
        })
        store.embedding_fn = _make_identical_embedding_fn()
        store.collections["facts"].delete.side_effect = RuntimeError("DB locked")
        dedup = CrossCollectionDeduplicator(store)
        plan = dedup.run(dry_run=False)

        assert plan.duplicates_found == 1
        assert len(plan.errors) >= 1
        assert "DB locked" in plan.errors[0]


class TestEphemeralSkip:
    """Tests that ephemeral relations are skipped in both dedup modes."""

    def test_ephemeral_facts_skip_contradiction(self):
        """Ephemeral predicates like current_feeling should not be flagged as contradictions."""
        store = _make_chroma_store({
            "facts": [
                _make_doc("f1", "user | current_feeling | tired", "facts", {
                    "subject": "user", "predicate": "current_feeling", "object": "tired",
                    "timestamp": "2026-02-12T00:00:00",
                }),
                _make_doc("f2", "user | current_feeling | fine", "facts", {
                    "subject": "user", "predicate": "current_feeling", "object": "fine",
                    "timestamp": "2026-02-13T00:00:00",
                }),
            ],
        })
        store.embedding_fn = _make_embedding_fn()
        dedup = CrossCollectionDeduplicator(store)
        plan = dedup.run(dry_run=True)

        # Should NOT detect as contradiction — these are temporal snapshots
        assert plan.contradictions_found == 0

    def test_ephemeral_facts_skip_duplicate(self):
        """Semantically similar ephemeral facts should not be flagged as duplicates."""
        store = _make_chroma_store({
            "facts": [
                _make_doc("f1", "IDENTICAL ephemeral dup test", "facts", {
                    "predicate": "current_activity",
                    "timestamp": "2026-02-12T00:00:00",
                }),
                _make_doc("f2", "IDENTICAL ephemeral dup test", "facts", {
                    "predicate": "current_activity",
                    "timestamp": "2026-02-13T00:00:00",
                }),
            ],
        })
        store.embedding_fn = _make_identical_embedding_fn()
        dedup = CrossCollectionDeduplicator(store)
        plan = dedup.run(dry_run=True)

        # Should NOT detect as duplicate — ephemeral predicate
        assert plan.duplicates_found == 0

    def test_non_ephemeral_still_detected(self):
        """Non-ephemeral predicates should still be caught normally."""
        store = _make_chroma_store({
            "facts": [
                _make_doc("f1", "user | favorite_language | Python", "facts", {
                    "subject": "user", "predicate": "favorite_language", "object": "Python",
                    "timestamp": "2026-01-01T00:00:00",
                }),
                _make_doc("f2", "user | favorite_language | Rust", "facts", {
                    "subject": "user", "predicate": "favorite_language", "object": "Rust",
                    "timestamp": "2026-02-01T00:00:00",
                }),
            ],
        })
        store.embedding_fn = _make_embedding_fn()
        dedup = CrossCollectionDeduplicator(store)
        plan = dedup.run(dry_run=True)

        # favorite_language is NOT ephemeral — should be detected
        assert plan.contradictions_found == 1

    def test_ephemeral_only_applies_to_facts(self):
        """Ephemeral skip should not affect non-facts collections."""
        store = _make_chroma_store({
            "facts": [
                _make_doc("f1", "IDENTICAL cross-coll ephemeral test", "facts", {
                    "predicate": "current_feeling",
                }),
            ],
            "summaries": [
                _make_doc("s1", "IDENTICAL cross-coll ephemeral test", "summaries"),
            ],
        })
        store.embedding_fn = _make_identical_embedding_fn()
        dedup = CrossCollectionDeduplicator(store)
        plan = dedup.run(dry_run=True)

        # Cross-collection pair (facts vs summaries) should still be detected
        # even if the fact has an ephemeral predicate
        assert plan.duplicates_found == 1

    def test_multiple_ephemeral_predicates(self):
        """Multiple different ephemeral predicates should all be skipped."""
        store = _make_chroma_store({
            "facts": [
                _make_doc("f1", "user emotional_state anxious yesterday", "facts", {
                    "subject": "user", "predicate": "emotional_state",
                    "object": "anxious", "timestamp": "2026-02-12T00:00:00",
                }),
                _make_doc("f2", "user emotional_state calm today", "facts", {
                    "subject": "user", "predicate": "emotional_state",
                    "object": "calm", "timestamp": "2026-02-13T00:00:00",
                }),
                _make_doc("f3", "user current_feeling poor last night", "facts", {
                    "subject": "user", "predicate": "current_feeling",
                    "object": "poor", "timestamp": "2026-02-12T00:00:00",
                }),
                _make_doc("f4", "user current_feeling good tonight", "facts", {
                    "subject": "user", "predicate": "current_feeling",
                    "object": "good", "timestamp": "2026-02-13T00:00:00",
                }),
            ],
        })
        store.embedding_fn = _make_embedding_fn()
        dedup = CrossCollectionDeduplicator(store)
        plan = dedup.run(dry_run=True)

        assert plan.contradictions_found == 0


class TestConfigIntegration:
    """Tests that config values are properly wired."""

    @patch("memory.cross_deduplicator.cfg")
    def test_thresholds_from_config(self, mock_cfg):
        mock_cfg.CROSS_DEDUP_DUPLICATE_THRESHOLD = 0.95
        mock_cfg.CROSS_DEDUP_CONTRADICTION_THRESHOLD = 0.88
        mock_cfg.CROSS_DEDUP_MAX_DOCS_PER_COLLECTION = 500
        mock_cfg.CROSS_DEDUP_COLLECTIONS = ["facts", "summaries"]

        store = _make_chroma_store({})
        store.embedding_fn = _make_embedding_fn()
        dedup = CrossCollectionDeduplicator(store)

        assert dedup.duplicate_threshold == 0.95
        assert dedup.contradiction_threshold == 0.88
        assert dedup.max_docs == 500
        assert "facts" in dedup.collections
        assert "summaries" in dedup.collections
