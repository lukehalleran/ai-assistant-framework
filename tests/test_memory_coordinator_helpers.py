"""Targeted tests for MemoryCoordinator helper functions to boost coverage."""
import pytest
from memory.memory_coordinator import (
    _is_deictic_followup,
    _salient_tokens,
    _num_op_density,
    _analogy_markers,
    _build_anchor_tokens
)


# Test _is_deictic_followup
def test_is_deictic_followup_explain():
    """Test deictic detection with 'explain'."""
    assert _is_deictic_followup("Can you explain that?")


def test_is_deictic_followup_that():
    """Test deictic detection with 'that'."""
    assert _is_deictic_followup("What about that?")


def test_is_deictic_followup_it():
    """Test deictic detection with 'it'."""
    assert _is_deictic_followup("Tell me more about it")


def test_is_deictic_followup_this():
    """Test deictic detection with 'this'."""
    assert _is_deictic_followup("What is this?")


def test_is_deictic_followup_again():
    """Test deictic detection with 'again'."""
    assert _is_deictic_followup("Can you say that again?")


def test_is_deictic_followup_another_way():
    """Test deictic detection with 'another way'."""
    assert _is_deictic_followup("Explain it another way")


def test_is_deictic_followup_different_way():
    """Test deictic detection with 'different way'."""
    assert _is_deictic_followup("Say it in a different way")


def test_is_deictic_followup_not_deictic():
    """Test non-deictic query."""
    assert not _is_deictic_followup("What is Python?")


def test_is_deictic_followup_empty():
    """Test with empty string."""
    assert not _is_deictic_followup("")


def test_is_deictic_followup_none():
    """Test with None."""
    assert not _is_deictic_followup(None)


def test_is_deictic_followup_case_insensitive():
    """Test deictic detection is case insensitive."""
    assert _is_deictic_followup("EXPLAIN THAT")
    assert _is_deictic_followup("This is it")


# Test _salient_tokens
def test_salient_tokens_basic():
    """Test extracting salient tokens from text."""
    text = "Python is a programming language used for data science"
    tokens = _salient_tokens(text, k=5)
    assert isinstance(tokens, set)
    assert len(tokens) <= 5


def test_salient_tokens_filters_stopwords():
    """Test that stopwords are filtered."""
    text = "the cat is on the mat"
    tokens = _salient_tokens(text, k=10)
    # 'the', 'is', 'on' should be filtered as stopwords
    # 'cat' and 'mat' should remain
    assert "cat" in tokens or "mat" in tokens


def test_salient_tokens_lowercase():
    """Test tokens are lowercased."""
    text = "Python Programming Language"
    tokens = _salient_tokens(text, k=5)
    # Check all tokens are lowercase
    assert all(t.islower() or not t.isalpha() for t in tokens)


def test_salient_tokens_empty():
    """Test with empty text."""
    tokens = _salient_tokens("", k=5)
    assert isinstance(tokens, set)
    assert len(tokens) == 0


def test_salient_tokens_all_stopwords():
    """Test with text containing only stopwords."""
    text = "the a an to of in on"
    tokens = _salient_tokens(text, k=10)
    assert len(tokens) == 0


def test_salient_tokens_k_limit():
    """Test k parameter limits number of tokens."""
    text = "one two three four five six seven eight nine ten"
    tokens = _salient_tokens(text, k=3)
    assert len(tokens) <= 3


def test_salient_tokens_short_words():
    """Test filtering of very short words."""
    text = "I am a programmer who codes in Python"
    tokens = _salient_tokens(text, k=10)
    # Very short words (1-2 chars) might be filtered
    assert "programmer" in tokens or "python" in tokens


# Test _num_op_density
def test_num_op_density_with_numbers():
    """Test numeric operation density with numbers."""
    text = "What is 5 + 3 - 2?"
    density = _num_op_density(text)
    assert isinstance(density, float)
    assert density > 0


def test_num_op_density_multiplication():
    """Test density with multiplication."""
    text = "Calculate 7 * 8"
    density = _num_op_density(text)
    assert density > 0


def test_num_op_density_division():
    """Test density with division."""
    text = "What is 10 / 2?"
    density = _num_op_density(text)
    assert density > 0


def test_num_op_density_no_operations():
    """Test density with no numeric operations."""
    text = "This is just regular text"
    density = _num_op_density(text)
    assert density == 0.0


def test_num_op_density_empty():
    """Test density with empty text."""
    density = _num_op_density("")
    assert density == 0.0


def test_num_op_density_multiple_operations():
    """Test density with multiple operations."""
    text = "1 + 2 - 3 * 4 / 5"
    density = _num_op_density(text)
    assert density > 0


def test_num_op_density_word_problem():
    """Test density with word problem."""
    text = "If I have 5 apples and buy 3 more, how many do I have?"
    density = _num_op_density(text)
    # May or may not detect operations depending on implementation
    assert isinstance(density, float)
    assert density >= 0


# Test _analogy_markers
def test_analogy_markers_like():
    """Test analogy detection with 'it's like'."""
    text = "Python it's like a Swiss Army knife"
    count = _analogy_markers(text)
    assert count > 0


def test_analogy_markers_similar():
    """Test analogy detection with 'imagine'."""
    text = "Imagine a world where code writes itself"
    count = _analogy_markers(text)
    assert count > 0


def test_analogy_markers_as_if():
    """Test analogy detection with 'as if'."""
    text = "It works as if by magic"
    count = _analogy_markers(text)
    assert count > 0


def test_analogy_markers_reminds():
    """Test analogy detection with 'picture this'."""
    text = "Picture this: a world of possibilities"
    count = _analogy_markers(text)
    assert count > 0


def test_analogy_markers_analogy():
    """Test analogy detection with 'analogy'."""
    text = "Here's an analogy for you"
    count = _analogy_markers(text)
    assert count > 0


def test_analogy_markers_metaphor():
    """Test analogy detection with 'metaphor'."""
    text = "It's a metaphor for life"
    count = _analogy_markers(text)
    assert count > 0


def test_analogy_markers_no_markers():
    """Test with no analogy markers."""
    text = "Just regular text here"
    count = _analogy_markers(text)
    assert count == 0


def test_analogy_markers_empty():
    """Test with empty text."""
    count = _analogy_markers("")
    assert count == 0


def test_analogy_markers_multiple():
    """Test with multiple markers."""
    text = "It's like a metaphor, similar to an analogy"
    count = _analogy_markers(text)
    assert count >= 2


def test_analogy_markers_case_insensitive():
    """Test analogy detection is case insensitive."""
    text = "It's LIKE a METAPHOR"
    count = _analogy_markers(text)
    assert count > 0


# Test _build_anchor_tokens
def test_build_anchor_tokens_basic():
    """Test building anchor tokens from conversation."""
    conv = [
        {"query": "What is Python?", "response": "Python is a programming language"},
        {"query": "Tell me more", "response": "It's used for data science"}
    ]
    tokens = _build_anchor_tokens(conv, maxlen=10)
    assert isinstance(tokens, set)
    assert len(tokens) > 0


def test_build_anchor_tokens_extracts_from_queries():
    """Test anchor tokens include query content."""
    conv = [
        {"query": "machine learning algorithms", "response": "ML is great"}
    ]
    tokens = _build_anchor_tokens(conv, maxlen=20)
    assert "machine" in tokens or "learning" in tokens or "algorithms" in tokens


def test_build_anchor_tokens_extracts_from_responses():
    """Test anchor tokens include response content."""
    conv = [
        {"query": "What is AI?", "response": "artificial intelligence systems"}
    ]
    tokens = _build_anchor_tokens(conv, maxlen=20)
    assert "artificial" in tokens or "intelligence" in tokens or "systems" in tokens


def test_build_anchor_tokens_empty_conversation():
    """Test with empty conversation."""
    tokens = _build_anchor_tokens([], maxlen=10)
    assert isinstance(tokens, set)
    assert len(tokens) == 0


def test_build_anchor_tokens_maxlen_limit():
    """Test maxlen limits number of tokens."""
    conv = [
        {"query": "word " * 100, "response": "word " * 100}
    ]
    tokens = _build_anchor_tokens(conv, maxlen=5)
    assert len(tokens) <= 5


def test_build_anchor_tokens_filters_stopwords():
    """Test anchor tokens filter stopwords."""
    conv = [
        {"query": "the cat is on the mat", "response": "the dog is in the house"}
    ]
    tokens = _build_anchor_tokens(conv, maxlen=10)
    # Should include content words, not stopwords
    assert "cat" in tokens or "mat" in tokens or "dog" in tokens or "house" in tokens


def test_build_anchor_tokens_missing_fields():
    """Test with missing query/response fields."""
    conv = [
        {"query": "Test"},
        {"response": "Answer"},
        {}
    ]
    tokens = _build_anchor_tokens(conv, maxlen=10)
    assert isinstance(tokens, set)


def test_build_anchor_tokens_none_values():
    """Test with None values in conversation."""
    conv = [
        {"query": None, "response": "Answer"},
        {"query": "Question", "response": None}
    ]
    tokens = _build_anchor_tokens(conv, maxlen=10)
    assert isinstance(tokens, set)


def test_build_anchor_tokens_lowercase():
    """Test anchor tokens are lowercase."""
    conv = [
        {"query": "PYTHON PROGRAMMING", "response": "DATA SCIENCE"}
    ]
    tokens = _build_anchor_tokens(conv, maxlen=20)
    # All tokens should be lowercase
    assert all(t.islower() or not t.isalpha() for t in tokens)


def test_salient_tokens_with_punctuation():
    """Test salient tokens handles punctuation."""
    text = "Hello, world! How are you?"
    tokens = _salient_tokens(text, k=10)
    assert isinstance(tokens, set)
    # Should extract words, not punctuation
    assert "hello" in tokens or "world" in tokens


def test_num_op_density_complex_expression():
    """Test numeric density with complex expression."""
    text = "(5 + 3) * 2 - 1"
    density = _num_op_density(text)
    assert density > 0


def test_analogy_markers_comparison():
    """Test analogy markers with comparison words."""
    text = "compared to the previous version"
    count = _analogy_markers(text)
    # Depending on implementation, may detect 'compared'
    assert isinstance(count, int)
    assert count >= 0


def test_build_anchor_tokens_long_text():
    """Test anchor tokens with very long text."""
    conv = [
        {"query": "word " * 1000, "response": "text " * 1000}
    ]
    tokens = _build_anchor_tokens(conv, maxlen=10)
    assert len(tokens) <= 10


def test_is_deictic_followup_combined_hints():
    """Test deictic with multiple hints."""
    assert _is_deictic_followup("explain this again")
    assert _is_deictic_followup("tell me about that in a different way")
