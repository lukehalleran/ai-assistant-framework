"""Tests for core/competitive_scorer.py"""
import pytest
from core.competitive_scorer import apply_competitive_selection, _extract_text


class MockTokenizerManager:
    """Mock tokenizer for testing."""

    def count_tokens(self, text, model_name):
        """Simple token count: ~1.3 tokens per word."""
        return int(len(text.split()) * 1.3)


def test_extract_text_from_string():
    """Test extracting text from string."""
    assert _extract_text("hello world") == "hello world"


def test_extract_text_from_dict_content():
    """Test extracting text from dict with content field."""
    item = {"content": "test content"}
    assert _extract_text(item) == "test content"


def test_extract_text_from_dict_text():
    """Test extracting text from dict with text field."""
    item = {"text": "test text"}
    assert _extract_text(item) == "test text"


def test_extract_text_from_dict_response():
    """Test extracting text from dict with response field."""
    item = {"response": "test response"}
    assert _extract_text(item) == "test response"


def test_extract_text_from_conversation():
    """Test extracting text from conversation dict."""
    # Note: response field is checked first in the source code (line 139)
    # so it returns just the response, not query+response
    item = {"query": "What is this?", "response": "This is a test"}
    result = _extract_text(item)
    assert result == "This is a test"  # Gets response field first


def test_apply_competitive_selection_basic():
    """Test basic competitive selection."""
    tokenizer = MockTokenizerManager()

    context = {
        "memories": [
            {"content": "Memory 1", "metadata": {"cosine_similarity": 0.9}},
            {"content": "Memory 2", "metadata": {"cosine_similarity": 0.7}},
        ],
        "facts": [
            {"content": "Fact 1", "metadata": {"relevance_score": 0.8}},
        ],
    }

    result = apply_competitive_selection(
        context=context,
        query="test query",
        tokenizer_manager=tokenizer,
        budget=100
    )

    assert "memories" in result
    assert "facts" in result
    # Should have selected some items
    assert len(result["memories"]) + len(result["facts"]) > 0


def test_apply_competitive_selection_respects_budget():
    """Test that selection respects token budget."""
    tokenizer = MockTokenizerManager()

    # Create items that together exceed budget
    context = {
        "memories": [
            {"content": "This is a long memory " * 50, "metadata": {"cosine_similarity": 0.9}},
            {"content": "Another long memory " * 50, "metadata": {"cosine_similarity": 0.8}},
            {"content": "Third long memory " * 50, "metadata": {"cosine_similarity": 0.7}},
        ],
    }

    small_budget = 100
    result = apply_competitive_selection(
        context=context,
        query="test",
        tokenizer_manager=tokenizer,
        budget=small_budget
    )

    # Count tokens in result
    total_tokens = 0
    for mem in result["memories"]:
        text = _extract_text(mem)
        total_tokens += tokenizer.count_tokens(text, "gpt-4")

    # Should be under budget
    assert total_tokens <= small_budget


def test_apply_competitive_selection_prioritizes_by_score():
    """Test that higher scored items are selected first."""
    tokenizer = MockTokenizerManager()

    context = {
        "memories": [
            {"content": "Low score", "metadata": {"cosine_similarity": 0.1}},
            {"content": "High score", "metadata": {"cosine_similarity": 0.9}},
            {"content": "Medium score", "metadata": {"cosine_similarity": 0.5}},
        ],
    }

    result = apply_competitive_selection(
        context=context,
        query="test",
        tokenizer_manager=tokenizer,
        budget=50  # Only enough for 1-2 items
    )

    # Should select the highest scored item
    assert len(result["memories"]) >= 1
    # High score item should be included
    contents = [_extract_text(m) for m in result["memories"]]
    assert "High score" in contents


def test_apply_competitive_selection_with_multiple_fields():
    """Test selection across multiple field types."""
    tokenizer = MockTokenizerManager()

    context = {
        "memories": [{"content": "Memory", "metadata": {"cosine_similarity": 0.5}}],
        "facts": [{"content": "Fact", "metadata": {"relevance_score": 0.8}}],
        "semantic_chunks": [{"text": "Chunk", "metadata": {"final_score": 0.7}}],
        "summaries": [{"content": "Summary", "metadata": {"cosine_similarity": 0.6}}],
    }

    result = apply_competitive_selection(
        context=context,
        query="test",
        tokenizer_manager=tokenizer,
        budget=200
    )

    # Should have items from multiple fields
    field_count = sum(1 for k, v in result.items() if k in context and len(v) > 0)
    assert field_count >= 2


def test_apply_competitive_selection_empty_context():
    """Test handling empty context."""
    tokenizer = MockTokenizerManager()

    context = {
        "memories": [],
        "facts": [],
    }

    result = apply_competitive_selection(
        context=context,
        query="test",
        tokenizer_manager=tokenizer,
        budget=100
    )

    assert result["memories"] == []
    assert result["facts"] == []


def test_apply_competitive_selection_preserves_wiki():
    """Test that wiki field is preserved as-is."""
    tokenizer = MockTokenizerManager()

    wiki_content = "Important wiki content"
    context = {
        "memories": [{"content": "Memory", "metadata": {"cosine_similarity": 0.5}}],
        "wiki": wiki_content,
    }

    result = apply_competitive_selection(
        context=context,
        query="test",
        tokenizer_manager=tokenizer,
        budget=100
    )

    # Wiki should be preserved
    assert result["wiki"] == wiki_content


def test_apply_competitive_selection_handles_missing_metadata():
    """Test handling items without metadata."""
    tokenizer = MockTokenizerManager()

    context = {
        "memories": [
            {"content": "Memory without metadata"},
        ],
    }

    result = apply_competitive_selection(
        context=context,
        query="test",
        tokenizer_manager=tokenizer,
        budget=100
    )

    # Should still work with default scores
    assert "memories" in result


def test_apply_competitive_selection_tokenizer_fallback():
    """Test fallback when tokenizer fails."""

    class FailingTokenizer:
        def count_tokens(self, text, model_name):
            raise Exception("Tokenizer failed")

    context = {
        "memories": [{"content": "Test memory", "metadata": {"cosine_similarity": 0.8}}],
    }

    # Should not crash, use fallback estimation
    result = apply_competitive_selection(
        context=context,
        query="test",
        tokenizer_manager=FailingTokenizer(),
        budget=100
    )

    assert "memories" in result


def test_apply_competitive_selection_filters_short_text():
    """Test that very short text is filtered out."""
    tokenizer = MockTokenizerManager()

    context = {
        "memories": [
            {"content": "a"},  # Too short
            {"content": "This is long enough", "metadata": {"cosine_similarity": 0.8}},
        ],
    }

    result = apply_competitive_selection(
        context=context,
        query="test",
        tokenizer_manager=tokenizer,
        budget=100
    )

    # Should only have the longer item
    assert len(result["memories"]) == 1
    assert _extract_text(result["memories"][0]) == "This is long enough"


def test_competitive_selection_value_per_token():
    """Test that value-per-token scoring works correctly."""
    tokenizer = MockTokenizerManager()

    context = {
        "memories": [
            # High score but very long (low value-per-token)
            {"content": "Long text " * 100, "metadata": {"cosine_similarity": 0.9}},
            # Lower score but short (high value-per-token)
            {"content": "Short", "metadata": {"cosine_similarity": 0.7}},
        ],
    }

    result = apply_competitive_selection(
        context=context,
        query="test",
        tokenizer_manager=tokenizer,
        budget=30  # Small budget
    )

    # Should prefer the short, efficient item
    assert len(result["memories"]) == 1
    assert _extract_text(result["memories"][0]) == "Short"
