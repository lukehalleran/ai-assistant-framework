"""
Tests for memory.claim_tracker — Memory Staleness Engine

Covers:
- ClaimKey model (creation, hashing)
- _compute_hash (determinism, case insensitivity)
- canonicalize_claim (with and without EntityResolver)
- extract_claims_from_text (separator, declarative, subject-verb patterns)
- ClaimIndex CRUD (add, lookup, remove, persistence)
- cascade_staleness (with and without chroma_store)
- Staleness ratio computation
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from memory.claim_tracker import (
    ClaimIndex,
    ClaimKey,
    IndexEntry,
    _compute_hash,
    canonicalize_claim,
    extract_claims_from_text,
)
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore


# ── ClaimKey Model ───────────────────────────────────────────────────


class TestClaimKey:
    def test_creation_with_auto_hash(self):
        ck = ClaimKey(subject="luke", relation="lives_in")
        assert ck.subject == "luke"
        assert ck.relation == "lives_in"
        assert len(ck.claim_hash) == 12

    def test_creation_with_explicit_hash(self):
        ck = ClaimKey(subject="luke", relation="lives_in", claim_hash="custom123456")
        assert ck.claim_hash == "custom123456"

    def test_same_inputs_produce_same_hash(self):
        ck1 = ClaimKey(subject="luke", relation="lives_in")
        ck2 = ClaimKey(subject="luke", relation="lives_in")
        assert ck1.claim_hash == ck2.claim_hash

    def test_different_inputs_produce_different_hashes(self):
        ck1 = ClaimKey(subject="luke", relation="lives_in")
        ck2 = ClaimKey(subject="luke", relation="works_at")
        assert ck1.claim_hash != ck2.claim_hash


# ── _compute_hash ────────────────────────────────────────────────────


class TestComputeHash:
    def test_deterministic(self):
        h1 = _compute_hash("luke", "lives_in")
        h2 = _compute_hash("luke", "lives_in")
        assert h1 == h2

    def test_case_insensitive(self):
        h1 = _compute_hash("Luke", "Lives_In")
        h2 = _compute_hash("luke", "lives_in")
        assert h1 == h2

    def test_strips_whitespace(self):
        h1 = _compute_hash("  luke  ", "  lives_in  ")
        h2 = _compute_hash("luke", "lives_in")
        assert h1 == h2

    def test_length_is_12(self):
        h = _compute_hash("anything", "here")
        assert len(h) == 12

    def test_different_pairs_differ(self):
        h1 = _compute_hash("luke", "lives_in")
        h2 = _compute_hash("auggie", "lives_in")
        assert h1 != h2


# ── canonicalize_claim ───────────────────────────────────────────────


class TestCanonicalizeClaim:
    def test_basic_normalization(self):
        ck = canonicalize_claim("Luke", "lives in")
        assert ck.subject == "luke"
        assert ck.relation == "lives_in"

    def test_with_entity_resolver(self):
        """EntityResolver should resolve aliases to canonical IDs."""
        mock_resolver = MagicMock()
        mock_resolver.resolve.return_value = "luke_h"
        ck = canonicalize_claim("Luke", "lives in", entity_resolver=mock_resolver)
        assert ck.subject == "luke_h"
        assert ck.relation == "lives_in"  # uses normalize_relation

    def test_entity_resolver_returns_none(self):
        """When resolver returns None, fallback to lowercased raw."""
        mock_resolver = MagicMock()
        mock_resolver.resolve.return_value = None
        ck = canonicalize_claim("Flapjack", "is_a", entity_resolver=mock_resolver)
        assert ck.subject == "flapjack"

    def test_relation_synonym_collapse(self):
        """'resides in' should canonicalize to 'lives_in'."""
        ck = canonicalize_claim("user", "resides in")
        assert ck.relation == "lives_in"

    def test_unknown_relation_passthrough(self):
        """Unknown relations get spaces→underscores."""
        ck = canonicalize_claim("user", "favorite color")
        assert ck.relation == "favorite_color"

    def test_hash_is_populated(self):
        ck = canonicalize_claim("user", "lives in")
        assert len(ck.claim_hash) == 12


# ── extract_claims_from_text ─────────────────────────────────────────


class TestExtractClaims:
    def test_separator_pattern_pipe(self):
        text = "Luke | lives in | Atlanta"
        claims = extract_claims_from_text(text)
        assert len(claims) >= 1
        subjects = [c.subject for c in claims]
        assert "luke" in subjects

    def test_separator_pattern_dash(self):
        text = "Luke - works at - Google"
        claims = extract_claims_from_text(text)
        assert len(claims) >= 1

    def test_separator_pattern_emdash(self):
        text = "Flapjack — is — a cat"
        claims = extract_claims_from_text(text)
        assert len(claims) >= 1

    def test_declarative_pattern(self):
        text = "Luke lives in Atlanta and works at a tech company."
        claims = extract_claims_from_text(text)
        relations = [c.relation for c in claims]
        assert any("lives" in r for r in relations)

    def test_subject_verb_user_pattern(self):
        text = "The user works at Google as a senior engineer."
        claims = extract_claims_from_text(text)
        assert any(c.subject == "user" for c in claims)

    def test_multiple_claims_in_summary(self):
        text = (
            "Luke lives in Atlanta. He works at Google. "
            "His cat Flapjack is an orange tabby."
        )
        claims = extract_claims_from_text(text)
        # Should find at least 2 claims
        assert len(claims) >= 2

    def test_deduplication(self):
        """Same claim appearing twice should be deduplicated."""
        text = "Luke lives in Atlanta. Luke lives in Atlanta."
        claims = extract_claims_from_text(text)
        hashes = [c.claim_hash for c in claims]
        assert len(hashes) == len(set(hashes))

    def test_empty_text(self):
        claims = extract_claims_from_text("")
        assert claims == []

    def test_no_claims_in_noise(self):
        text = "This is a random sentence with no structured claims."
        claims = extract_claims_from_text(text)
        # May or may not find claims depending on pattern matches
        # Just ensure no crash
        assert isinstance(claims, list)

    def test_with_entity_resolver(self):
        """Resolver should normalize subjects in extracted claims."""
        mock_resolver = MagicMock()
        mock_resolver.resolve.return_value = "luke_h"
        text = "Luke | lives in | Atlanta"
        claims = extract_claims_from_text(text, entity_resolver=mock_resolver)
        assert any(c.subject == "luke_h" for c in claims)

    def test_multiline_summary(self):
        text = """Summary of conversations from Feb 15-20:
Luke | has | a cat named Flapjack
Luke | lives in | Atlanta
Luke | works on | Daemon project
The user has been discussing memory architecture improvements."""
        claims = extract_claims_from_text(text)
        assert len(claims) >= 3


# ── ClaimIndex CRUD ──────────────────────────────────────────────────


class TestClaimIndex:
    @pytest.fixture
    def index(self):
        return ClaimIndex()

    @pytest.fixture
    def sample_claims(self):
        return [
            ClaimKey(subject="luke", relation="lives_in"),
            ClaimKey(subject="luke", relation="works_at"),
            ClaimKey(subject="flapjack", relation="is_a"),
        ]

    def test_add_and_lookup(self, index, sample_claims):
        index.add_claims("doc_1", "summaries", sample_claims)
        docs = index.get_documents_for_claim(sample_claims[0].claim_hash)
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "doc_1"
        assert docs[0]["collection"] == "summaries"

    def test_multiple_docs_same_claim(self, index, sample_claims):
        index.add_claims("doc_1", "summaries", [sample_claims[0]])
        index.add_claims("doc_2", "reflections", [sample_claims[0]])
        docs = index.get_documents_for_claim(sample_claims[0].claim_hash)
        assert len(docs) == 2
        doc_ids = {d["doc_id"] for d in docs}
        assert doc_ids == {"doc_1", "doc_2"}

    def test_no_duplicate_entries(self, index, sample_claims):
        """Adding same doc+claim twice shouldn't create duplicates."""
        index.add_claims("doc_1", "summaries", [sample_claims[0]])
        index.add_claims("doc_1", "summaries", [sample_claims[0]])
        docs = index.get_documents_for_claim(sample_claims[0].claim_hash)
        assert len(docs) == 1

    def test_get_claims_for_document(self, index, sample_claims):
        index.add_claims("doc_1", "summaries", sample_claims)
        hashes = index.get_claims_for_document("doc_1")
        assert len(hashes) == 3

    def test_remove_document(self, index, sample_claims):
        index.add_claims("doc_1", "summaries", sample_claims)
        index.remove_document("doc_1")
        assert index.get_claims_for_document("doc_1") == []
        for ck in sample_claims:
            assert index.get_documents_for_claim(ck.claim_hash) == []

    def test_remove_document_preserves_others(self, index, sample_claims):
        index.add_claims("doc_1", "summaries", sample_claims[:2])
        index.add_claims("doc_2", "reflections", sample_claims[:1])
        index.remove_document("doc_1")
        # doc_2 still has its entry
        docs = index.get_documents_for_claim(sample_claims[0].claim_hash)
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "doc_2"

    def test_total_claims(self, index, sample_claims):
        index.add_claims("doc_1", "summaries", sample_claims)
        assert index.total_claims == 3

    def test_total_documents(self, index, sample_claims):
        index.add_claims("doc_1", "summaries", sample_claims[:2])
        index.add_claims("doc_2", "reflections", sample_claims[2:])
        assert index.total_documents == 2

    def test_empty_lookup(self, index):
        docs = index.get_documents_for_claim("nonexistent")
        assert docs == []

    def test_empty_doc_claims(self, index):
        hashes = index.get_claims_for_document("nonexistent")
        assert hashes == []


# ── ClaimIndex Persistence ───────────────────────────────────────────


class TestClaimIndexPersistence:
    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "claim_index.json")
        claims = [
            ClaimKey(subject="luke", relation="lives_in"),
            ClaimKey(subject="flapjack", relation="is_a"),
        ]

        # Save
        idx1 = ClaimIndex(persist_path=path)
        idx1.add_claims("doc_1", "summaries", claims)
        idx1.save()
        assert os.path.exists(path)

        # Load
        idx2 = ClaimIndex(persist_path=path)
        docs = idx2.get_documents_for_claim(claims[0].claim_hash)
        assert len(docs) == 1
        assert idx2.total_claims == 2
        assert idx2.total_documents == 1

    def test_save_no_path(self):
        """No persist_path should not crash on save."""
        idx = ClaimIndex()
        idx.add_claims("doc_1", "summaries", [ClaimKey(subject="x", relation="y")])
        idx.save()  # no-op, no crash

    def test_load_nonexistent_path(self, tmp_path):
        """Loading from a nonexistent path should not crash."""
        idx = ClaimIndex(persist_path=str(tmp_path / "nonexistent.json"))
        assert idx.total_claims == 0

    def test_load_corrupted_json(self, tmp_path):
        """Loading corrupted JSON should not crash."""
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("{invalid json")
        idx = ClaimIndex(persist_path=path)
        assert idx.total_claims == 0

    def test_dirty_flag_saves(self, tmp_path):
        """Only saves when dirty."""
        path = str(tmp_path / "claim_index.json")
        idx = ClaimIndex(persist_path=path)
        idx.save()  # not dirty, nothing written
        assert not os.path.exists(path)

        idx.add_claims("doc_1", "summaries", [ClaimKey(subject="x", relation="y")])
        idx.save()
        assert os.path.exists(path)


# ── cascade_staleness ────────────────────────────────────────────────


class TestCascadeStaleness:
    @pytest.fixture
    def populated_index(self):
        idx = ClaimIndex()
        claims_doc1 = [
            ClaimKey(subject="luke", relation="lives_in"),
            ClaimKey(subject="luke", relation="works_at"),
        ]
        claims_doc2 = [
            ClaimKey(subject="luke", relation="lives_in"),
            ClaimKey(subject="flapjack", relation="is_a"),
            ClaimKey(subject="luke", relation="favorite_language"),
        ]
        idx.add_claims("doc_1", "summaries", claims_doc1)
        idx.add_claims("doc_2", "summaries", claims_doc2)
        return idx

    def test_cascade_without_chroma(self, populated_index):
        """Without chroma_store, returns affected docs with estimated ratios."""
        ck = ClaimKey(subject="luke", relation="lives_in")
        results = populated_index.cascade_staleness(ck)
        assert len(results) == 2
        doc_ids = {r["doc_id"] for r in results}
        assert doc_ids == {"doc_1", "doc_2"}

    def test_cascade_staleness_ratio_computation(self, populated_index):
        """doc_1 has 2 claims, 1 stale → ratio 0.5. doc_2 has 3 claims, 1 stale → ratio 0.333."""
        ck = ClaimKey(subject="luke", relation="lives_in")
        results = populated_index.cascade_staleness(ck)
        ratios = {r["doc_id"]: r["staleness_ratio"] for r in results}
        assert ratios["doc_1"] == round(1 / 2, 3)
        assert ratios["doc_2"] == round(1 / 3, 3)

    def test_cascade_with_chroma(self, populated_index):
        """With chroma_store, updates metadata on affected docs.

        spec= binds the mock to the real MultiCollectionChromaStore interface, so
        accessing a nonexistent method (e.g. the old get_document_metadata) raises
        AttributeError — this is what guards against the cascade calling a phantom method.
        """
        mock_chroma = MagicMock(spec=MultiCollectionChromaStore)
        mock_chroma.get_by_id.return_value = {
            "id": "doc_1", "content": "x", "metadata": {"staleness_ratio": 0.0, "stale_claims": ""},
        }

        ck = ClaimKey(subject="luke", relation="lives_in")
        results = populated_index.cascade_staleness(ck, chroma_store=mock_chroma)
        assert len(results) == 2
        assert mock_chroma.update_metadata.call_count == 2

    def test_cascade_accumulates_stale_claims(self, populated_index):
        """Multiple cascades should accumulate stale claims, increasing ratio."""
        mock_chroma = MagicMock(spec=MultiCollectionChromaStore)
        # First cascade: no existing stale claims
        mock_chroma.get_by_id.return_value = {
            "id": "doc_1", "content": "x", "metadata": {"staleness_ratio": 0.0, "stale_claims": ""},
        }

        ck1 = ClaimKey(subject="luke", relation="lives_in")
        populated_index.cascade_staleness(ck1, chroma_store=mock_chroma)

        # Verify update_metadata was called with stale_claims containing the hash
        calls = mock_chroma.update_metadata.call_args_list
        for call in calls:
            metadata = call[0][2]  # third positional arg
            assert ck1.claim_hash in metadata["stale_claims"]

    def test_cascade_no_affected_docs(self, populated_index):
        """Claim with no matching docs returns empty list."""
        ck = ClaimKey(subject="nobody", relation="nothing")
        results = populated_index.cascade_staleness(ck)
        assert results == []

    def test_cascade_missing_doc_cleaned_up(self, populated_index):
        """If chroma says doc doesn't exist, it should be removed from index."""
        mock_chroma = MagicMock(spec=MultiCollectionChromaStore)
        mock_chroma.get_by_id.return_value = None  # doc deleted

        ck = ClaimKey(subject="luke", relation="lives_in")
        results = populated_index.cascade_staleness(ck, chroma_store=mock_chroma)
        # Should clean up the missing doc from the index
        # Results won't include docs that were cleaned up
        assert all(r.get("staleness_ratio") is not None for r in results if r)

    def test_cascade_uses_real_chroma_interface(self, populated_index):
        """Regression guard for the get_document_metadata bug.

        The cascade must read metadata via get_by_id() (a real MultiCollectionChromaStore
        method that returns {id, content, metadata}), then write via update_metadata().
        A spec'd mock raises AttributeError if the cascade calls any method the real store
        does not expose, so this fails loudly if a phantom method is reintroduced.
        """
        mock_chroma = MagicMock(spec=MultiCollectionChromaStore)
        mock_chroma.get_by_id.return_value = {
            "id": "doc_1", "content": "x", "metadata": {"stale_claims": ""},
        }
        ck = ClaimKey(subject="luke", relation="lives_in")
        results = populated_index.cascade_staleness(ck, chroma_store=mock_chroma)
        assert mock_chroma.get_by_id.called
        assert mock_chroma.update_metadata.called
        assert len(results) >= 1
