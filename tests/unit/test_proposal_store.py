# tests/unit/test_proposal_store.py
"""Unit tests for memory.proposal_store."""

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from memory.code_proposal import CodeProposal, ProposalStatus, ProposalType
from memory.proposal_store import ProposalStore


# ------------------------------------------------------------------
# Mock ChromaDB store
# ------------------------------------------------------------------


class MockChromaCollection:
    """Minimal mock of a ChromaDB collection for testing."""

    def __init__(self):
        self._docs = {}  # id -> {document, metadata}

    def count(self):
        return len(self._docs)

    def delete(self, ids=None):
        for doc_id in (ids or []):
            self._docs.pop(doc_id, None)


class MockChromaStore:
    """
    Mock of MultiCollectionChromaStore with the methods used by ProposalStore.
    """

    def __init__(self):
        self.collections = {"proposals": MockChromaCollection()}
        self._stored = []  # track add_to_collection calls

    def create_collection(self, name):
        self.collections[name] = MockChromaCollection()

    def add_to_collection(self, name, text, metadata):
        doc_id = str(uuid.uuid4())
        coll = self.collections.get(name)
        if coll:
            coll._docs[doc_id] = {"document": text, "metadata": dict(metadata)}
        self._stored.append({"id": doc_id, "text": text, "metadata": metadata})
        return doc_id

    def list_all(self, collection_name):
        coll = self.collections.get(collection_name)
        if not coll:
            return []
        return [
            {"id": doc_id, "content": data["document"], "metadata": data["metadata"]}
            for doc_id, data in coll._docs.items()
        ]

    def query_collection(self, collection_name, query_text, n_results=5):
        """Return all stored items with fake relevance scores."""
        items = self.list_all(collection_name)
        results = []
        for i, item in enumerate(items[:n_results]):
            results.append({
                "id": item["id"],
                "content": item["content"],
                "metadata": item["metadata"],
                "relevance_score": 0.9 - (i * 0.1),  # decreasing relevance
                "collection": collection_name,
                "rank": i + 1,
            })
        return results


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_store():
    return MockChromaStore()


@pytest.fixture
def proposal_store(mock_store):
    return ProposalStore(chroma_store=mock_store)


@pytest.fixture
def sample_proposal():
    return CodeProposal(
        title="Add caching layer",
        proposal_type=ProposalType.FEATURE,
        priority=8,
        reasoning="API responses are slow",
        tags=["performance", "caching"],
    )


# ------------------------------------------------------------------
# Basic storage tests
# ------------------------------------------------------------------


class TestStoreProposal:
    def test_store_and_retrieve(self, proposal_store, sample_proposal):
        doc_id = proposal_store.store_proposal(sample_proposal)
        assert doc_id is not None

        retrieved = proposal_store.get_proposal(sample_proposal.id)
        assert retrieved is not None
        assert retrieved.title == "Add caching layer"
        assert retrieved.proposal_type == ProposalType.FEATURE

    def test_store_without_chroma_returns_none(self):
        store = ProposalStore(chroma_store=None)
        p = CodeProposal(title="Test")
        result = store.store_proposal(p)
        assert result is None

    def test_store_multiple(self, proposal_store):
        p1 = CodeProposal(title="First proposal")
        p2 = CodeProposal(title="Second proposal")
        id1 = proposal_store.store_proposal(p1)
        id2 = proposal_store.store_proposal(p2)
        assert id1 is not None
        assert id2 is not None
        assert id1 != id2


# ------------------------------------------------------------------
# Query tests
# ------------------------------------------------------------------


class TestQueryProposals:
    def test_query_returns_results(self, proposal_store, sample_proposal):
        proposal_store.store_proposal(sample_proposal)
        results = proposal_store.query_proposals("caching performance")
        assert len(results) >= 1
        assert results[0].title == "Add caching layer"

    def test_query_with_status_filter(self, proposal_store):
        p1 = CodeProposal(title="Pending one", status=ProposalStatus.PENDING)
        p2 = CodeProposal(title="Approved one", status=ProposalStatus.APPROVED)
        proposal_store.store_proposal(p1)
        proposal_store.store_proposal(p2)

        pending = proposal_store.query_proposals("one", status_filter=["pending"])
        assert all(p.status == ProposalStatus.PENDING for p in pending)

    def test_query_empty_collection(self, proposal_store):
        results = proposal_store.query_proposals("anything")
        assert results == []

    def test_query_no_chroma(self):
        store = ProposalStore(chroma_store=None)
        assert store.query_proposals("test") == []


# ------------------------------------------------------------------
# get_pending tests
# ------------------------------------------------------------------


class TestGetPending:
    def test_returns_only_pending(self, proposal_store):
        p1 = CodeProposal(title="Pending A")
        p2 = CodeProposal(title="Approved B")
        p2.mark_approved()
        proposal_store.store_proposal(p1)
        proposal_store.store_proposal(p2)

        pending = proposal_store.get_pending()
        assert len(pending) == 1
        assert pending[0].title == "Pending A"

    def test_empty_collection(self, proposal_store):
        assert proposal_store.get_pending() == []

    def test_no_chroma(self):
        store = ProposalStore(chroma_store=None)
        assert store.get_pending() == []


# ------------------------------------------------------------------
# update_status tests
# ------------------------------------------------------------------


class TestUpdateStatus:
    def test_approve(self, proposal_store, sample_proposal):
        proposal_store.store_proposal(sample_proposal)
        result = proposal_store.update_status(sample_proposal.id, ProposalStatus.APPROVED)
        assert result is True

        updated = proposal_store.get_proposal(sample_proposal.id)
        assert updated is not None
        assert updated.status == ProposalStatus.APPROVED

    def test_reject_with_reason(self, proposal_store, sample_proposal):
        proposal_store.store_proposal(sample_proposal)
        result = proposal_store.update_status(
            sample_proposal.id, ProposalStatus.REJECTED, reason="Not needed"
        )
        assert result is True

        updated = proposal_store.get_proposal(sample_proposal.id)
        assert updated.status == ProposalStatus.REJECTED
        assert updated.rejection_reason == "Not needed"

    def test_complete_with_hash(self, proposal_store, sample_proposal):
        proposal_store.store_proposal(sample_proposal)
        result = proposal_store.update_status(
            sample_proposal.id, ProposalStatus.COMPLETED, commit_hash="abc123"
        )
        assert result is True

        updated = proposal_store.get_proposal(sample_proposal.id)
        assert updated.status == ProposalStatus.COMPLETED
        assert updated.commit_hash == "abc123"

    def test_update_nonexistent_returns_false(self, proposal_store):
        result = proposal_store.update_status("nonexistent-id", ProposalStatus.APPROVED)
        assert result is False

    def test_update_no_chroma(self):
        store = ProposalStore(chroma_store=None)
        assert store.update_status("id", ProposalStatus.APPROVED) is False


# ------------------------------------------------------------------
# check_similarity tests
# ------------------------------------------------------------------


class TestCheckSimilarity:
    def test_no_duplicates_returns_none(self, proposal_store, sample_proposal):
        # Empty collection
        result = proposal_store.check_similarity(sample_proposal)
        assert result is None

    def test_duplicate_detected(self, proposal_store, sample_proposal):
        proposal_store.store_proposal(sample_proposal)

        # Same proposal should be detected as duplicate
        similar = CodeProposal(
            title="Add caching layer",
            reasoning="API responses are slow",
        )
        result = proposal_store.check_similarity(similar, threshold=0.5)
        assert result is not None  # should return existing ID

    def test_no_chroma(self, sample_proposal):
        store = ProposalStore(chroma_store=None)
        assert store.check_similarity(sample_proposal) is None


# ------------------------------------------------------------------
# get_for_dedup tests
# ------------------------------------------------------------------


class TestGetForDedup:
    def test_returns_formatted_string(self, proposal_store, sample_proposal):
        proposal_store.store_proposal(sample_proposal)
        text = proposal_store.get_for_dedup()
        assert "Add caching layer" in text
        assert "pending" in text
        assert "feature" in text

    def test_empty_collection(self, proposal_store):
        assert proposal_store.get_for_dedup() == ""

    def test_no_chroma(self):
        store = ProposalStore(chroma_store=None)
        assert store.get_for_dedup() == ""

    def test_limit_respected(self, proposal_store):
        for i in range(15):
            p = CodeProposal(title=f"Proposal {i}")
            proposal_store.store_proposal(p)

        text = proposal_store.get_for_dedup(limit=5)
        lines = [l for l in text.strip().split("\n") if l.strip()]
        assert len(lines) == 5
