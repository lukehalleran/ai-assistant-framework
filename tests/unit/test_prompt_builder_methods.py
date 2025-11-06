"""
Unit tests for core/prompt.py UnifiedPromptBuilder methods

Tests helper methods from UnifiedPromptBuilder class:
- _extract_text: Extract text from various formats
- _format_memory: Format memory entries for prompt
- _get_time_context: Format current time
- _wiki_cache_key: Generate cache keys
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock
from core.prompt import UnifiedPromptBuilder


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
def prompt_builder():
    """Create minimal UnifiedPromptBuilder instance for testing helper methods"""
    # Create instance without calling __init__ to avoid complex dependencies
    builder = object.__new__(UnifiedPromptBuilder)
    # Set only the attributes needed by the methods we're testing
    builder.model_manager = Mock()
    builder.tokenizer_manager = Mock()
    builder.memory_coordinator = Mock()
    builder.wiki_manager = None
    builder.topic_manager = None
    return builder


# =============================================================================
# _extract_text Tests
# =============================================================================

def test_extract_text_from_string(prompt_builder):
    """Extracts plain string as-is"""
    result = prompt_builder._extract_text("Hello world")

    assert result == "Hello world"


def test_extract_text_from_dict_content(prompt_builder):
    """Extracts 'content' key from dict"""
    item = {"content": "This is content", "other": "ignored"}
    result = prompt_builder._extract_text(item)

    assert result == "This is content"


def test_extract_text_from_dict_text(prompt_builder):
    """Extracts 'text' key from dict"""
    item = {"text": "This is text"}
    result = prompt_builder._extract_text(item)

    assert result == "This is text"


def test_extract_text_from_dict_response(prompt_builder):
    """Extracts 'response' key from dict"""
    item = {"response": "This is a response"}
    result = prompt_builder._extract_text(item)

    assert result == "This is a response"


def test_extract_text_from_dict_filtered_content(prompt_builder):
    """Extracts 'filtered_content' key from dict"""
    item = {"filtered_content": "Filtered text"}
    result = prompt_builder._extract_text(item)

    assert result == "Filtered text"


def test_extract_text_key_priority(prompt_builder):
    """Checks key priority: content > text > response > filtered_content"""
    item = {
        "content": "content_value",
        "text": "text_value",
        "response": "response_value",
        "filtered_content": "filtered_value"
    }
    result = prompt_builder._extract_text(item)

    # Should prefer 'content' first
    assert result == "content_value"


def test_extract_text_from_dict_no_keys(prompt_builder):
    """Converts dict to string if no known keys"""
    item = {"unknown": "value", "other": 123}
    result = prompt_builder._extract_text(item)

    # Should return string representation
    assert "unknown" in result or "value" in result


def test_extract_text_from_empty_dict(prompt_builder):
    """Handles empty dict"""
    result = prompt_builder._extract_text({})

    assert isinstance(result, str)


def test_extract_text_from_none(prompt_builder):
    """Handles None input"""
    result = prompt_builder._extract_text(None)

    assert result == "None"


def test_extract_text_from_number(prompt_builder):
    """Converts numbers to string"""
    assert prompt_builder._extract_text(42) == "42"
    assert prompt_builder._extract_text(3.14) == "3.14"


def test_extract_text_from_list(prompt_builder):
    """Converts list to string"""
    result = prompt_builder._extract_text(["a", "b", "c"])

    assert isinstance(result, str)
    assert "a" in result


def test_extract_text_empty_string_values(prompt_builder):
    """Handles dict with empty string values"""
    item = {"content": "", "text": "fallback"}
    result = prompt_builder._extract_text(item)

    # Empty string is falsy, should skip to 'text'
    assert result == "fallback"


# =============================================================================
# _format_memory Tests
# =============================================================================

def test_format_memory_with_query_response(prompt_builder):
    """Formats memory with Q&A format"""
    memory = {
        "query": "What is 2+2?",
        "response": "4"
    }
    result = prompt_builder._format_memory(memory)

    assert "Q: What is 2+2?" in result
    assert "A: 4" in result


def test_format_memory_with_content(prompt_builder):
    """Formats memory with content field"""
    memory = {"content": "Some interesting fact"}
    result = prompt_builder._format_memory(memory)

    assert "Some interesting fact" in result
    assert result.endswith("\n")


def test_format_memory_with_timestamp(prompt_builder):
    """Includes formatted timestamp when formatter provided"""
    memory = {
        "query": "Question",
        "response": "Answer",
        "timestamp": "2024-01-01T12:00:00"
    }

    def format_ts(ts):
        return "Jan 1, 12:00"

    result = prompt_builder._format_memory(memory, fmt_ts_func=format_ts)

    assert "[Jan 1, 12:00]" in result
    assert "Q: Question" in result


def test_format_memory_without_timestamp_formatter(prompt_builder):
    """Omits timestamp when no formatter provided"""
    memory = {
        "query": "Question",
        "response": "Answer",
        "timestamp": "2024-01-01T12:00:00"
    }
    result = prompt_builder._format_memory(memory)

    assert "[" not in result
    assert "Q: Question" in result


def test_format_memory_timestamp_returns_none(prompt_builder):
    """Handles formatter returning None"""
    memory = {
        "query": "Question",
        "response": "Answer",
        "timestamp": "invalid"
    }

    def format_ts(ts):
        return None

    result = prompt_builder._format_memory(memory, fmt_ts_func=format_ts)

    assert "[" not in result


def test_format_memory_no_timestamp_field(prompt_builder):
    """Handles memory without timestamp field"""
    memory = {
        "query": "Question",
        "response": "Answer"
    }

    def format_ts(ts):
        return "formatted"

    result = prompt_builder._format_memory(memory, fmt_ts_func=format_ts)

    # No timestamp field, so no bracket
    assert "[" not in result


def test_format_memory_empty_query_response(prompt_builder):
    """Handles empty query/response strings"""
    memory = {
        "query": "",
        "response": ""
    }
    result = prompt_builder._format_memory(memory)

    assert "Q:" in result
    assert "A:" in result


def test_format_memory_whitespace_stripping(prompt_builder):
    """Strips whitespace from query/response"""
    memory = {
        "query": "  Question  ",
        "response": "  Answer  "
    }
    result = prompt_builder._format_memory(memory)

    assert "Q: Question\n" in result
    assert "A: Answer\n" in result


def test_format_memory_fallback_to_str(prompt_builder):
    """Falls back to str() for unknown format"""
    memory = {"unknown_field": "value", "other": 123}
    result = prompt_builder._format_memory(memory)

    assert isinstance(result, str)
    assert result.endswith("\n")


def test_format_memory_content_with_timestamp(prompt_builder):
    """Content format supports timestamps"""
    memory = {
        "content": "Fact about X",
        "timestamp": "2024-01-01T12:00:00"
    }

    def format_ts(ts):
        return "Jan 1"

    result = prompt_builder._format_memory(memory, fmt_ts_func=format_ts)

    assert "[Jan 1]" in result
    assert "Fact about X" in result


# =============================================================================
# _get_time_context Tests
# =============================================================================

def test_get_time_context_format(prompt_builder):
    """Returns formatted time context"""
    result = prompt_builder._get_time_context()

    assert result.startswith("Current time:")
    assert len(result) > 20  # Should have datetime string


def test_get_time_context_year(prompt_builder):
    """Includes current year"""
    result = prompt_builder._get_time_context()
    current_year = datetime.now().year

    assert str(current_year) in result


def test_get_time_context_format_structure(prompt_builder):
    """Follows YYYY-MM-DD HH:MM:SS format"""
    result = prompt_builder._get_time_context()

    # Should match pattern: "Current time: 2024-01-01 12:00:00"
    assert "Current time: " in result
    # Check for date separators
    parts = result.split("Current time: ")[1]
    assert "-" in parts  # Date separators
    assert ":" in parts  # Time separators


# =============================================================================
# _wiki_cache_key Tests
# =============================================================================

def test_wiki_cache_key_basic(prompt_builder):
    """Generates cache key from query"""
    result = prompt_builder._wiki_cache_key("Python programming")

    assert "Python programming" in result
    assert "|" in result  # Separator between raw and cleaned


def test_wiki_cache_key_strips_whitespace(prompt_builder):
    """Strips leading/trailing whitespace"""
    result = prompt_builder._wiki_cache_key("  query  ")

    # Should strip whitespace from raw part
    assert result.startswith("query|")


def test_wiki_cache_key_empty_string(prompt_builder):
    """Handles empty string"""
    result = prompt_builder._wiki_cache_key("")

    assert result.startswith("|")  # Empty raw part


def test_wiki_cache_key_none_input(prompt_builder):
    """Handles None input"""
    result = prompt_builder._wiki_cache_key(None)

    # Should handle None gracefully (converts to empty)
    assert isinstance(result, str)


def test_wiki_cache_key_includes_cleaned_query(prompt_builder):
    """Includes cleaned query in cache key"""
    # The function calls clean_query() which we can't test without
    # the actual implementation, but we can verify structure
    result = prompt_builder._wiki_cache_key("Test Query")

    parts = result.split("|")
    assert len(parts) == 2
    assert parts[0] == "Test Query"
    # parts[1] would be clean_query("Test Query")


def test_wiki_cache_key_consistency(prompt_builder):
    """Same query produces same key"""
    key1 = prompt_builder._wiki_cache_key("test")
    key2 = prompt_builder._wiki_cache_key("test")

    assert key1 == key2


def test_wiki_cache_key_different_queries(prompt_builder):
    """Different queries produce different keys"""
    key1 = prompt_builder._wiki_cache_key("query1")
    key2 = prompt_builder._wiki_cache_key("query2")

    assert key1 != key2


# =============================================================================
# Integration Tests
# =============================================================================

def test_extract_and_format_workflow(prompt_builder):
    """Extract text then format as memory"""
    # Simulate extracting response from memory dict
    memory_item = {"response": "The capital is Paris"}
    extracted = prompt_builder._extract_text(memory_item)

    # Then format it as a memory
    formatted_memory = {
        "query": "What is the capital of France?",
        "response": extracted
    }
    result = prompt_builder._format_memory(formatted_memory)

    assert "Paris" in result
    assert "Q:" in result


def test_time_context_in_prompt(prompt_builder):
    """Time context can be included in prompt"""
    time_ctx = prompt_builder._get_time_context()

    # Should be non-empty and formatted
    assert len(time_ctx) > 0
    assert "Current time:" in time_ctx


def test_memory_formatting_with_various_types(prompt_builder):
    """Format different memory types"""
    memories = [
        {"query": "Q1", "response": "A1"},
        {"content": "Fact 1"},
        {"query": "Q2", "response": "A2", "timestamp": "2024-01-01"}
    ]

    results = [prompt_builder._format_memory(m) for m in memories]

    assert all(isinstance(r, str) for r in results)
    assert all(r.endswith("\n") for r in results)
    assert "Q1" in results[0]
    assert "Fact 1" in results[1]
    assert "Q2" in results[2]
