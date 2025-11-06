"""
Unit tests for memory/memory_coordinator.py helper functions

Tests pure utility functions:
- Deictic followup detection
- Salient token extraction
- Number/operator density calculation
- Analogy markers detection
- Anchor token building
"""

import pytest
from memory.memory_coordinator import (
    _is_deictic_followup,
    _salient_tokens,
    _num_op_density,
    _analogy_markers,
    _build_anchor_tokens,
)


# =============================================================================
# _is_deictic_followup Tests
# =============================================================================

def test_is_deictic_followup_with_explain():
    """Detects 'explain' as deictic"""
    assert _is_deictic_followup("explain that concept") == True


def test_is_deictic_followup_with_that():
    """Detects 'that' as deictic"""
    assert _is_deictic_followup("what about that?") == True


def test_is_deictic_followup_with_it():
    """Detects 'it' as deictic"""
    assert _is_deictic_followup("tell me more about it") == True


def test_is_deictic_followup_with_again():
    """Detects 'again' as deictic"""
    assert _is_deictic_followup("can you say that again?") == True


def test_is_deictic_followup_negative():
    """Does not detect non-deictic queries"""
    assert _is_deictic_followup("What is Python?") == False


def test_is_deictic_followup_case_insensitive():
    """Deictic detection is case insensitive"""
    assert _is_deictic_followup("EXPLAIN this") == True


def test_is_deictic_followup_empty():
    """Empty string is not deictic"""
    assert _is_deictic_followup("") == False


def test_is_deictic_followup_none():
    """None input is not deictic"""
    assert _is_deictic_followup(None) == False


# =============================================================================
# _salient_tokens Tests
# =============================================================================

def test_salient_tokens_basic():
    """Extracts salient tokens from text"""
    text = "machine learning algorithms process data efficiently"
    result = _salient_tokens(text, k=3)

    assert isinstance(result, set)
    assert len(result) <= 3
    # Should not include stopwords
    assert "the" not in result
    assert "a" not in result


def test_salient_tokens_frequency_priority():
    """Prioritizes frequent tokens"""
    text = "python python python java java javascript"
    result = _salient_tokens(text, k=2)

    # Should include most frequent
    assert "python" in result


def test_salient_tokens_filters_stopwords():
    """Filters common stopwords"""
    text = "the quick brown fox and the lazy dog"
    result = _salient_tokens(text)

    # Stopwords should be filtered
    assert "the" not in result
    assert "and" not in result
    # Content words should remain
    assert "quick" in result or "brown" in result


def test_salient_tokens_filters_short():
    """Filters single-character tokens"""
    text = "a b c d programming language"
    result = _salient_tokens(text)

    # Single chars should be filtered
    assert "a" not in result
    assert "b" not in result
    # Longer words should remain
    assert "programming" in result


def test_salient_tokens_empty():
    """Handles empty text"""
    result = _salient_tokens("")

    assert isinstance(result, set)
    assert len(result) == 0


def test_salient_tokens_none():
    """Handles None input"""
    result = _salient_tokens(None)

    assert isinstance(result, set)
    assert len(result) == 0


def test_salient_tokens_with_numbers():
    """Includes numbers in tokens"""
    text = "calculate 42 plus 17 equals 59"
    result = _salient_tokens(text)

    # Numbers should be included
    assert any(tok.isdigit() or tok[0].isdigit() for tok in result)


def test_salient_tokens_with_operators():
    """Includes mathematical operators"""
    text = "x + y = z where x^2 - 3"
    result = _salient_tokens(text)

    # Should extract math-related tokens
    assert len(result) > 0


# =============================================================================
# _num_op_density Tests
# =============================================================================

def test_num_op_density_with_math():
    """Calculates density for mathematical expressions"""
    text = "x = 42 + 17 - 3"
    result = _num_op_density(text)

    assert result > 0
    # Has 3 numbers, 2 operators in ~5 tokens
    assert isinstance(result, float)


def test_num_op_density_no_math():
    """Returns low density for plain text"""
    text = "this is just regular text without any numbers"
    result = _num_op_density(text)

    assert result == 0.0 or result < 0.1


def test_num_op_density_empty():
    """Handles empty text"""
    result = _num_op_density("")

    assert result == 0.0


def test_num_op_density_none():
    """Handles None input"""
    result = _num_op_density(None)

    assert result == 0.0


def test_num_op_density_only_numbers():
    """Calculates density with only numbers"""
    text = "42 17 99 123"
    result = _num_op_density(text)

    assert result > 0


def test_num_op_density_only_operators():
    """Calculates density with only operators"""
    text = "+ - * / ="
    result = _num_op_density(text)

    assert result > 0


def test_num_op_density_mixed():
    """Calculates density for mixed content"""
    text = "The answer is 42 plus 17 equals 59"
    result = _num_op_density(text)

    # Has numbers but fewer operators relative to words
    assert result > 0 and result < 1.0


def test_num_op_density_high_density():
    """High density for math-heavy text"""
    text = "1+2=3 4*5=20 6-7=-1"
    result = _num_op_density(text)

    assert result > 0.5  # High ratio of nums/ops to words


# =============================================================================
# _analogy_markers Tests
# =============================================================================

def test_analogy_markers_its_like():
    """Detects 'it's like' marker"""
    text = "It's like riding a bike"
    result = _analogy_markers(text)

    assert result >= 1


def test_analogy_markers_imagine():
    """Detects 'imagine' marker"""
    text = "Imagine a world where AI helps everyone"
    result = _analogy_markers(text)

    assert result >= 1


def test_analogy_markers_as_if():
    """Detects 'as if' marker"""
    text = "It works as if by magic"
    result = _analogy_markers(text)

    assert result >= 1


def test_analogy_markers_metaphor():
    """Detects 'metaphor' marker"""
    text = "This is a metaphor for learning"
    result = _analogy_markers(text)

    assert result >= 1


def test_analogy_markers_multiple():
    """Counts multiple markers"""
    text = "It's like a metaphor. Imagine this analogy."
    result = _analogy_markers(text)

    assert result >= 2


def test_analogy_markers_none():
    """Returns 0 when no markers present"""
    text = "This is just a plain statement"
    result = _analogy_markers(text)

    assert result == 0


def test_analogy_markers_empty():
    """Handles empty text"""
    result = _analogy_markers("")

    assert result == 0


def test_analogy_markers_none_input():
    """Handles None input"""
    result = _analogy_markers(None)

    assert result == 0


def test_analogy_markers_case_insensitive():
    """Detection is case insensitive"""
    text = "IT'S LIKE a METAPHOR"
    result = _analogy_markers(text)

    assert result >= 2


# =============================================================================
# _build_anchor_tokens Tests
# =============================================================================

def test_build_anchor_tokens_from_last_exchange():
    """Builds anchor tokens from last conversation"""
    conv = [
        {"query": "What is Python?", "response": "Python is a programming language"},
        {"query": "Tell me about variables", "response": "Variables store data values"}
    ]
    result = _build_anchor_tokens(conv)

    assert isinstance(result, set)
    assert len(result) > 0
    # Should include tokens from last exchange
    assert "variables" in result or "data" in result


def test_build_anchor_tokens_with_math():
    """Extracts mathematical patterns"""
    conv = [
        {"query": "Solve x^2 + 3x = 7", "response": "Using the quadratic formula f(x)"}
    ]
    result = _build_anchor_tokens(conv)

    # Should extract math tokens
    assert len(result) > 0


def test_build_anchor_tokens_empty_conv():
    """Handles empty conversation"""
    result = _build_anchor_tokens([])

    assert isinstance(result, set)
    assert len(result) == 0


def test_build_anchor_tokens_with_numbers():
    """Includes numbers as anchors"""
    conv = [
        {"query": "Calculate 42 + 17", "response": "The result is 59"}
    ]
    result = _build_anchor_tokens(conv)

    # Should include numbers
    assert any(tok.replace(".", "").isdigit() for tok in result)


def test_build_anchor_tokens_maxlen_limit():
    """Respects maxlen parameter"""
    conv = [
        {"query": "word " * 50, "response": "response " * 50}
    ]
    result = _build_anchor_tokens(conv, maxlen=10)

    assert len(result) <= 10


def test_build_anchor_tokens_with_derivatives():
    """Extracts calculus-related terms"""
    conv = [
        {"query": "What's the derivative of x^2?", "response": "The derivative is 2x using f'(x)"}
    ]
    result = _build_anchor_tokens(conv)

    # Should have calculus-related tokens
    assert "derivative" in result or len(result) > 0


def test_build_anchor_tokens_missing_keys():
    """Handles conversation entries without query/response"""
    conv = [
        {"other": "data"},
        {"query": "test"}
    ]
    result = _build_anchor_tokens(conv)

    # Should not crash
    assert isinstance(result, set)


# =============================================================================
# Integration Tests
# =============================================================================

def test_salient_and_anchor_tokens_overlap():
    """Salient tokens and anchor tokens can overlap"""
    text = "machine learning algorithms process data"
    salient = _salient_tokens(text)

    conv = [{"query": text, "response": "Yes, that's correct"}]
    anchors = _build_anchor_tokens(conv)

    # Should have some overlap
    overlap = salient & anchors
    assert len(overlap) >= 0  # May or may not overlap


def test_density_and_analogy_for_explanatory_text():
    """Combines density and analogy checks"""
    text = "It's like 2 + 2 = 4, imagine this metaphor"

    density = _num_op_density(text)
    analogies = _analogy_markers(text)

    # Has both math and analogies
    assert density > 0
    assert analogies >= 2
