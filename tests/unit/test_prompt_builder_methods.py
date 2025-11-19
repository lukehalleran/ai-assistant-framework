"""
Unit tests for core/prompt module formatter and token manager methods

Tests helper methods from PromptFormatter and TokenManager classes:
- _extract_text: Extract text from various formats (TokenManager)
- _format_memory: Format memory entries for prompt (PromptFormatter)
- _get_time_context: Format current time (PromptFormatter)
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock
from core.prompt import UnifiedPromptBuilder
from core.prompt.formatter import PromptFormatter
from core.prompt.token_manager import TokenManager


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_model_manager():
    """Mock model manager"""
    manager = Mock()
    manager.get_active_model_name = Mock(return_value="gpt-4")
    return manager


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer manager"""
    tokenizer = Mock()
    tokenizer.count_tokens = Mock(return_value=10)
    return tokenizer


@pytest.fixture
def mock_memory_coordinator():
    """Mock memory coordinator with corpus manager"""
    coordinator = Mock()
    coordinator.corpus_manager = Mock()
    coordinator.corpus_manager.get_recent_memories = Mock(return_value=[])
    coordinator.get_summaries = Mock(return_value=[])
    return coordinator


@pytest.fixture
def token_manager(mock_model_manager, mock_tokenizer):
    """Create TokenManager instance for testing"""
    return TokenManager(
        model_manager=mock_model_manager,
        tokenizer_manager=mock_tokenizer,
        token_budget=2048
    )


@pytest.fixture
def formatter(token_manager):
    """Create PromptFormatter instance for testing"""
    return PromptFormatter(token_manager=token_manager)


# =============================================================================
# _extract_text Tests (TokenManager)
# =============================================================================

def test_extract_text_from_string(token_manager):
    """Extracts plain string as-is"""
    result = token_manager._extract_text("Hello world")

    assert result == "Hello world"


def test_extract_text_from_dict_content(token_manager):
    """Extracts 'content' key from dict"""
    item = {"content": "This is content", "other": "ignored"}
    result = token_manager._extract_text(item)

    assert result == "This is content"


def test_extract_text_from_dict_text(token_manager):
    """Extracts 'text' key from dict"""
    item = {"text": "This is text"}
    result = token_manager._extract_text(item)

    assert result == "This is text"


def test_extract_text_from_dict_response(token_manager):
    """Extracts 'response' key from dict"""
    item = {"response": "This is a response"}
    result = token_manager._extract_text(item)

    assert result == "This is a response"


def test_extract_text_from_dict_filtered_content(token_manager):
    """Extracts 'filtered_content' key from dict"""
    item = {"filtered_content": "Filtered text"}
    result = token_manager._extract_text(item)

    assert result == "Filtered text"


def test_extract_text_key_priority(token_manager):
    """Checks key priority: content > text > response > filtered_content"""
    item = {
        "content": "content_value",
        "text": "text_value",
        "response": "response_value",
        "filtered_content": "filtered_value"
    }
    result = token_manager._extract_text(item)

    # Should prefer 'content' first
    assert result == "content_value"


def test_extract_text_from_dict_no_keys(token_manager):
    """Converts dict to string if no known keys"""
    item = {"unknown": "value", "other": 123}
    result = token_manager._extract_text(item)

    # Should return string representation
    assert "unknown" in result or "value" in result


def test_extract_text_from_empty_dict(token_manager):
    """Handles empty dict"""
    result = token_manager._extract_text({})

    assert isinstance(result, str)


def test_extract_text_from_none(token_manager):
    """Handles None input"""
    result = token_manager._extract_text(None)

    assert result == "None"


def test_extract_text_from_number(token_manager):
    """Converts numbers to string"""
    assert token_manager._extract_text(42) == "42"
    assert token_manager._extract_text(3.14) == "3.14"


def test_extract_text_from_list(token_manager):
    """Converts list to string"""
    result = token_manager._extract_text(["a", "b", "c"])

    assert isinstance(result, str)
    assert "a" in result


def test_extract_text_empty_string_values(token_manager):
    """Handles dict with empty string values"""
    item = {"content": "", "text": "fallback"}
    result = token_manager._extract_text(item)

    # Empty string is falsy, should skip to 'text'
    assert result == "fallback"


# =============================================================================
# _format_memory Tests (PromptFormatter)
# =============================================================================

def test_format_memory_with_query_response(formatter):
    """Formats memory with Q&A format"""
    memory = {
        "query": "What is 2+2?",
        "response": "4"
    }
    result = formatter._format_memory(memory)

    # New format uses "User:" and "Daemon:" prefixes
    assert "User: What is 2+2?" in result
    assert "Daemon: 4" in result


def test_format_memory_with_content(formatter):
    """Formats memory with content field"""
    memory = {"content": "Some interesting fact"}
    result = formatter._format_memory(memory)

    assert "Some interesting fact" in result


def test_format_memory_with_timestamp(formatter):
    """Includes timestamp in output"""
    memory = {
        "query": "Question",
        "response": "Answer",
        "timestamp": "2024-01-01T12:00:00"
    }

    result = formatter._format_memory(memory)

    # Timestamp should be included at the start
    assert "2024-01-01T12:00:00" in result
    assert "User: Question" in result


def test_format_memory_without_timestamp(formatter):
    """Shows 'Unknown time' when no timestamp provided"""
    memory = {
        "query": "Question",
        "response": "Answer"
    }
    result = formatter._format_memory(memory)

    # Should show "Unknown time" as placeholder
    assert "Unknown time" in result
    assert "User: Question" in result


def test_format_memory_empty_query_response(formatter):
    """Handles empty query/response strings"""
    memory = {
        "query": "",
        "response": ""
    }
    result = formatter._format_memory(memory)

    # Should return some formatted output
    assert isinstance(result, str)


def test_format_memory_whitespace_stripping(formatter):
    """Strips whitespace from query/response"""
    memory = {
        "query": "  Question  ",
        "response": "  Answer  "
    }
    result = formatter._format_memory(memory)

    assert "User: Question" in result
    assert "Daemon: Answer" in result


def test_format_memory_fallback_to_str(formatter):
    """Falls back to memory id for unknown format"""
    memory = {"unknown_field": "value", "other": 123}
    result = formatter._format_memory(memory)

    assert isinstance(result, str)


def test_format_memory_content_with_timestamp(formatter):
    """Content format includes timestamps"""
    memory = {
        "content": "Fact about X",
        "timestamp": "2024-01-01T12:00:00"
    }

    result = formatter._format_memory(memory)

    assert "2024-01-01T12:00:00" in result
    assert "Fact about X" in result


# =============================================================================
# _get_time_context Tests (PromptFormatter)
# =============================================================================

def test_get_time_context_format(formatter):
    """Returns formatted time context"""
    result = formatter._get_time_context()

    assert result.startswith("Current time:")
    assert len(result) > 20  # Should have datetime string


def test_get_time_context_year(formatter):
    """Includes current year"""
    result = formatter._get_time_context()
    current_year = datetime.now().year

    assert str(current_year) in result


def test_get_time_context_format_structure(formatter):
    """Follows YYYY-MM-DD HH:MM:SS format"""
    result = formatter._get_time_context()

    # Should match pattern: "Current time: 2024-01-01 12:00:00"
    assert "Current time: " in result
    # Check for date separators
    parts = result.split("Current time: ")[1]
    assert "-" in parts  # Date separators
    assert ":" in parts  # Time separators


# =============================================================================
# Integration Tests
# =============================================================================

def test_extract_and_format_workflow(token_manager, formatter):
    """Extract text then format as memory"""
    # Simulate extracting response from memory dict
    memory_item = {"response": "The capital is Paris"}
    extracted = token_manager._extract_text(memory_item)

    # Then format it as a memory
    formatted_memory = {
        "query": "What is the capital of France?",
        "response": extracted
    }
    result = formatter._format_memory(formatted_memory)

    assert "Paris" in result
    assert "User:" in result


def test_time_context_in_prompt(formatter):
    """Time context can be included in prompt"""
    time_ctx = formatter._get_time_context()

    # Should be non-empty and formatted
    assert len(time_ctx) > 0
    assert "Current time:" in time_ctx


def test_memory_formatting_with_various_types(formatter):
    """Format different memory types"""
    memories = [
        {"query": "Q1", "response": "A1"},
        {"content": "Fact 1"},
        {"query": "Q2", "response": "A2", "timestamp": "2024-01-01"}
    ]

    results = [formatter._format_memory(m) for m in memories]

    assert all(isinstance(r, str) for r in results)
    assert "Q1" in results[0]
    assert "Fact 1" in results[1]
    assert "Q2" in results[2]
