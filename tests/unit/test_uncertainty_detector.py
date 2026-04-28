"""Tests for core/uncertainty_detector.py — response uncertainty detection."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from core.uncertainty_detector import (
    UncertaintyDetector,
    UncertaintyResult,
    _strip_hedge_prefix,
    _anchor_embeddings_cache,
)
import core.uncertainty_detector as _mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_anchor_cache():
    """Reset cached anchor embeddings between tests."""
    _mod._anchor_embeddings_cache = None
    yield
    _mod._anchor_embeddings_cache = None


@pytest.fixture
def mock_embedder():
    """Mock embedder returning predictable 384-dim vectors."""
    embedder = MagicMock()
    rng = np.random.RandomState(42)

    def encode_fn(texts, convert_to_numpy=True, normalize_embeddings=True):
        n = len(texts) if isinstance(texts, list) else 1
        vecs = rng.randn(n, 384).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms

    embedder.encode = MagicMock(side_effect=encode_fn)
    return embedder


@pytest.fixture
def high_sim_embedder():
    """Mock embedder where response embedding is very similar to anchors."""
    embedder = MagicMock()

    def encode_fn(texts, convert_to_numpy=True, normalize_embeddings=True):
        n = len(texts) if isinstance(texts, list) else 1
        # Return identical unit vectors so cosine sim ≈ 1.0
        vec = np.ones((n, 384), dtype=np.float32)
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / norms

    embedder.encode = MagicMock(side_effect=encode_fn)
    return embedder


# ---------------------------------------------------------------------------
# Keyword detection tests
# ---------------------------------------------------------------------------

class TestKeywordDetection:
    """Test keyword-layer uncertainty patterns."""

    def test_dont_recall(self):
        result = UncertaintyDetector.detect(
            "I don't recall what we discussed last Monday."
        )
        assert result.is_uncertain
        assert result.trigger_type == "keyword"
        assert result.confidence >= 0.70

    def test_no_information(self):
        result = UncertaintyDetector.detect(
            "I don't have any information about that topic."
        )
        assert result.is_uncertain
        assert result.trigger_type == "keyword"

    def test_couldnt_find(self):
        result = UncertaintyDetector.detect(
            "I couldn't find any records of that conversation."
        )
        assert result.is_uncertain

    def test_unable_to_recall(self):
        result = UncertaintyDetector.detect(
            "I'm unable to recall the specifics of our discussion."
        )
        assert result.is_uncertain

    def test_not_sure(self):
        result = UncertaintyDetector.detect(
            "I'm not sure what we talked about last week."
        )
        assert result.is_uncertain

    def test_unfortunately_cant(self):
        result = UncertaintyDetector.detect(
            "Unfortunately, I can't find anything about that in my records."
        )
        assert result.is_uncertain

    def test_no_previous_conversation(self):
        result = UncertaintyDetector.detect(
            "There's no previous conversation about that topic in my context."
        )
        assert result.is_uncertain

    def test_havent_discussed(self):
        result = UncertaintyDetector.detect(
            "We haven't discussed that topic before."
        )
        assert result.is_uncertain

    def test_my_memory_doesnt_contain(self):
        result = UncertaintyDetector.detect(
            "My memory doesn't contain any information about last Thursday."
        )
        assert result.is_uncertain

    def test_cant_remember_what_we_talked(self):
        result = UncertaintyDetector.detect(
            "I can't recall what we talked about on Monday."
        )
        assert result.is_uncertain
        assert result.confidence >= 0.85


# ---------------------------------------------------------------------------
# Confident response tests (should NOT trigger)
# ---------------------------------------------------------------------------

class TestConfidentResponses:
    """Test that confident responses are NOT flagged."""

    def test_normal_answer(self):
        result = UncertaintyDetector.detect(
            "Last Monday we discussed your Python project and the FastAPI migration."
        )
        assert not result.is_uncertain

    def test_detailed_answer(self):
        result = UncertaintyDetector.detect(
            "You mentioned last week that you were working on the Daemon project."
        )
        assert not result.is_uncertain

    def test_casual_response(self):
        result = UncertaintyDetector.detect(
            "Sure! I'd be happy to help with that."
        )
        assert not result.is_uncertain

    def test_greeting(self):
        result = UncertaintyDetector.detect(
            "Hey! How are you doing today?"
        )
        assert not result.is_uncertain

    def test_factual_answer(self):
        result = UncertaintyDetector.detect(
            "The capital of France is Paris. It has been the capital since the 10th century."
        )
        assert not result.is_uncertain


# ---------------------------------------------------------------------------
# Length guard tests
# ---------------------------------------------------------------------------

class TestLengthGuard:
    """Test that long responses skip fallback even with hedge words."""

    def test_long_hedged_response(self):
        """Long response with hedge but real content should NOT trigger."""
        response = (
            "I'm not sure about the exact details, but from what I recall, "
            "we had a conversation about your project architecture. You were "
            "working on the memory system and discussing how to implement "
            "better temporal retrieval. We also covered the agentic search "
            "loop and its integration with ChromaDB collections. You mentioned "
            "wanting to add support for obsidian notes and improving the "
            "semantic search pipeline for better recall accuracy."
        )
        result = UncertaintyDetector.detect(response, max_length=400)
        assert not result.is_uncertain

    def test_short_hedge_triggers(self):
        """Short uncertain response SHOULD trigger."""
        result = UncertaintyDetector.detect(
            "I don't recall what we discussed last Monday.",
            max_length=400,
        )
        assert result.is_uncertain

    def test_hedge_prefix_stripped_for_length(self):
        """Hedge prefix is stripped before measuring length."""
        # "Unfortunately, " gets stripped, leaving substantive content
        short = "Unfortunately, I don't have records about that."
        result = UncertaintyDetector.detect(short, max_length=400)
        assert result.is_uncertain  # Short after stripping


# ---------------------------------------------------------------------------
# Hedge stripping helper tests
# ---------------------------------------------------------------------------

class TestHedgeStripping:
    """Test the hedge prefix stripping utility."""

    def test_strip_unfortunately(self):
        assert _strip_hedge_prefix("Unfortunately, the data is missing.") == "the data is missing."

    def test_strip_im_sorry(self):
        result = _strip_hedge_prefix("I'm sorry, I can't help with that.")
        assert result.startswith("I can't") or result.startswith("i can't") or "sorry" not in result.lower()[:5]

    def test_no_hedge_unchanged(self):
        text = "The answer is 42."
        assert _strip_hedge_prefix(text) == text

    def test_strip_not_sure_but(self):
        result = _strip_hedge_prefix("I'm not sure, but I think it was last week.")
        assert result.startswith("I think") or result.startswith("i think")


# ---------------------------------------------------------------------------
# Semantic layer tests
# ---------------------------------------------------------------------------

class TestSemanticDetection:
    """Test semantic-layer uncertainty detection."""

    def test_semantic_runs_with_embedder(self, mock_embedder):
        """Verify semantic layer runs when embedder provided."""
        result = UncertaintyDetector.detect(
            "Some normal response that doesn't match keywords.",
            embedder=mock_embedder,
            semantic_threshold=0.70,
        )
        assert isinstance(result, UncertaintyResult)
        # With random embeddings, unlikely to exceed 0.70 threshold
        # but the function should complete without error

    def test_high_similarity_triggers(self, high_sim_embedder):
        """When embeddings are very similar to anchors, semantic triggers."""
        result = UncertaintyDetector.detect(
            "Some text that doesn't match keywords at all.",
            embedder=high_sim_embedder,
            semantic_threshold=0.70,
        )
        assert result.is_uncertain
        assert result.trigger_type == "semantic"
        assert result.confidence >= 0.70

    def test_no_embedder_skips_semantic(self):
        """Without embedder, only keyword layer runs."""
        result = UncertaintyDetector.detect(
            "A normal response that doesn't match keywords.",
            embedder=None,
        )
        assert not result.is_uncertain
        assert result.trigger_type == ""

    def test_anchor_cache_reused(self, mock_embedder):
        """Anchor embeddings computed once, then cached."""
        UncertaintyDetector.detect("First call.", embedder=mock_embedder)
        UncertaintyDetector.detect("Second call.", embedder=mock_embedder)
        # encode should be called: once for anchors + once per response = 3 total
        # (anchors cached after first call, so second call = 1 encode for response)
        assert mock_embedder.encode.call_count == 3  # anchors(1) + response(1) + response(1)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_response(self):
        result = UncertaintyDetector.detect("")
        assert not result.is_uncertain

    def test_whitespace_response(self):
        result = UncertaintyDetector.detect("   \n  \t  ")
        assert not result.is_uncertain

    def test_very_short_response(self):
        """Responses under 10 chars return not uncertain."""
        result = UncertaintyDetector.detect("Hi")
        assert not result.is_uncertain

    def test_none_response(self):
        result = UncertaintyDetector.detect(None)
        assert not result.is_uncertain


# ---------------------------------------------------------------------------
# Result model tests
# ---------------------------------------------------------------------------

class TestUncertaintyResult:
    """Test the Pydantic result model."""

    def test_valid_result(self):
        r = UncertaintyResult(
            is_uncertain=True,
            confidence=0.85,
            trigger_type="keyword",
            matched_pattern="don't recall",
        )
        assert r.is_uncertain
        assert r.confidence == 0.85

    def test_default_values(self):
        r = UncertaintyResult(is_uncertain=False)
        assert r.confidence == 0.0
        assert r.trigger_type == ""
        assert r.matched_pattern == ""

    def test_confidence_bounds(self):
        """Confidence must be between 0.0 and 1.0."""
        with pytest.raises(Exception):
            UncertaintyResult(is_uncertain=True, confidence=1.5)
        with pytest.raises(Exception):
            UncertaintyResult(is_uncertain=True, confidence=-0.1)
