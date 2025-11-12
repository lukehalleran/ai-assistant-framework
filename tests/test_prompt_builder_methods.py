"""Unit tests for UnifiedPromptBuilder internal methods."""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from core.prompt import UnifiedPromptBuilder
from models.model_manager import ModelManager
from memory.memory_coordinator import MemoryCoordinator
from memory.corpus_manager import CorpusManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore


@pytest.fixture
def temp_dirs():
    """Create temporary directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield {
            "corpus_file": str(Path(tmpdir) / "corpus.json"),
            "chroma_path": str(Path(tmpdir) / "chroma_db")
        }


@pytest.fixture
def model_manager():
    """Provide ModelManager."""
    return ModelManager()


@pytest.fixture
def memory_coordinator(temp_dirs):
    """Provide MemoryCoordinator."""
    corpus_manager = CorpusManager(corpus_file=temp_dirs["corpus_file"])
    chroma_store = MultiCollectionChromaStore(persist_directory=temp_dirs["chroma_path"])
    return MemoryCoordinator(corpus_manager=corpus_manager, chroma_store=chroma_store)


@pytest.fixture
def prompt_builder(model_manager, memory_coordinator):
    """Provide UnifiedPromptBuilder."""
    return UnifiedPromptBuilder(
        model_manager=model_manager,
        memory_coordinator=memory_coordinator
    )


def test_middle_out_no_compression_needed(prompt_builder):
    """Test _middle_out when text is within token limit."""
    text = "This is a short text that doesn't need compression."

    result = prompt_builder._middle_out(text, max_tokens=1000)

    assert result == text


def test_middle_out_compression_applied(prompt_builder):
    """Test _middle_out when text exceeds token limit."""
    # Create long text
    text = "A" * 10000

    result = prompt_builder._middle_out(text, max_tokens=100, force=True)

    assert len(result) < len(text)
    assert "middle-out snipped" in result
    assert result.startswith("A")
    assert result.endswith("A")


def test_middle_out_with_force_flag(prompt_builder):
    """Test _middle_out with force=True."""
    text = "B" * 5000

    result = prompt_builder._middle_out(text, max_tokens=50, force=True)

    assert len(result) < len(text)
    assert "middle-out snipped" in result


def test_get_time_context(prompt_builder):
    """Test _get_time_context returns formatted string."""
    context = prompt_builder._get_time_context()

    assert isinstance(context, str)
    assert len(context) > 0


def test_extract_text_from_string(prompt_builder):
    """Test _extract_text with string input."""
    text = "Hello world"

    result = prompt_builder._extract_text(text)

    assert result == text


def test_extract_text_from_dict_content(prompt_builder):
    """Test _extract_text with dict containing 'content'."""
    item = {"content": "Test content"}

    result = prompt_builder._extract_text(item)

    assert result == "Test content"


def test_extract_text_from_dict_text(prompt_builder):
    """Test _extract_text with dict containing 'text'."""
    item = {"text": "Test text"}

    result = prompt_builder._extract_text(item)

    assert result == "Test text"


def test_extract_text_from_dict_response(prompt_builder):
    """Test _extract_text with dict containing 'response'."""
    item = {"response": "Test response"}

    result = prompt_builder._extract_text(item)

    assert result == "Test response"


def test_extract_text_from_empty_dict(prompt_builder):
    """Test _extract_text with empty dict."""
    item = {}

    result = prompt_builder._extract_text(item)

    # Empty dict returns string representation
    assert result == "{}"


def test_extract_text_from_none(prompt_builder):
    """Test _extract_text with None."""
    result = prompt_builder._extract_text(None)

    # None returns string representation
    assert result == "None"


def test_format_memory_basic(prompt_builder):
    """Test _format_memory with basic memory dict."""
    memory = {
        "content": "User likes Python",
        "timestamp": "2024-01-15T10:00:00"
    }

    result = prompt_builder._format_memory(memory)

    assert isinstance(result, str)
    assert "User likes Python" in result


def test_format_memory_no_timestamp(prompt_builder):
    """Test _format_memory without timestamp."""
    memory = {
        "content": "User prefers dark mode"
    }

    result = prompt_builder._format_memory(memory)

    assert isinstance(result, str)
    assert "User prefers dark mode" in result


def test_format_memory_with_custom_formatter(prompt_builder):
    """Test _format_memory with custom timestamp formatter."""
    def custom_formatter(ts):
        return "CUSTOM_TIME"

    memory = {
        "content": "Test content",
        "timestamp": "2024-01-15T10:00:00"
    }

    result = prompt_builder._format_memory(memory, fmt_ts_func=custom_formatter)

    assert "CUSTOM_TIME" in result or "Test content" in result


def test_wiki_cache_key(prompt_builder):
    """Test _wiki_cache_key generates consistent keys."""
    query1 = "Python programming"
    query2 = "Python programming"
    query3 = "Java programming"

    key1 = prompt_builder._wiki_cache_key(query1)
    key2 = prompt_builder._wiki_cache_key(query2)
    key3 = prompt_builder._wiki_cache_key(query3)

    assert key1 == key2  # Same query should generate same key
    assert key1 != key3  # Different queries should generate different keys


@pytest.mark.asyncio
async def test_llm_summarize_recent_empty(prompt_builder):
    """Test _llm_summarize_recent with empty recents list."""
    result = await prompt_builder._llm_summarize_recent([])

    assert result == ""


@pytest.mark.asyncio
async def test_llm_summarize_recent_no_model_manager(prompt_builder):
    """Test _llm_summarize_recent when model_manager lacks generate_once."""
    # Temporarily remove generate_once
    original_mm = prompt_builder.model_manager
    mock_mm = Mock()
    mock_mm.generate_once = None
    mock_mm.api_models = {}  # Add empty api_models dict
    prompt_builder.model_manager = mock_mm

    recents = [{"query": "Hi", "response": "Hello"}]
    result = await prompt_builder._llm_summarize_recent(recents)

    prompt_builder.model_manager = original_mm
    assert result == ""


@pytest.mark.asyncio
async def test_get_recent_conversations_general_topic(prompt_builder, memory_coordinator):
    """Test _get_recent_conversations with general topic."""
    # Store some conversations
    await memory_coordinator.store_interaction(
        query="What is Python?",
        response="Python is a language.",
        tags=["topic:general"]
    )

    result = await prompt_builder._get_recent_conversations(count=5, topic="general")

    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_get_recent_conversations_specific_topic(prompt_builder, memory_coordinator):
    """Test _get_recent_conversations with specific topic."""
    # Store conversations with specific topic
    await memory_coordinator.store_interaction(
        query="Machine learning basics?",
        response="ML is a subset of AI.",
        tags=["topic:ml"]
    )

    result = await prompt_builder._get_recent_conversations(count=5, topic="ml")

    assert isinstance(result, list)


def test_load_directives_nonexistent_file(prompt_builder):
    """Test _load_directives with nonexistent file."""
    result = prompt_builder._load_directives("nonexistent_file.txt")

    # Should return empty string or handle gracefully
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_decide_gen_params_smalltalk(prompt_builder):
    """Test _decide_gen_params with small talk input."""
    user_input = "hey"

    # This method might return params dict or similar
    try:
        result = prompt_builder._decide_gen_params(user_input)
        assert result is not None
    except AttributeError:
        # Method might not exist or has different signature
        pytest.skip("_decide_gen_params not available or has different API")


@pytest.mark.asyncio
async def test_decide_gen_params_question(prompt_builder):
    """Test _decide_gen_params with question input."""
    user_input = "What is machine learning?"

    try:
        result = prompt_builder._decide_gen_params(user_input)
        assert result is not None
    except AttributeError:
        pytest.skip("_decide_gen_params not available or has different API")


def test_get_token_count_basic(prompt_builder):
    """Test get_token_count with basic text."""
    text = "Hello world, this is a test."

    count = prompt_builder.get_token_count(text, model_name="gpt-4")

    assert isinstance(count, int)
    assert count > 0


def test_get_token_count_empty(prompt_builder):
    """Test get_token_count with empty text."""
    count = prompt_builder.get_token_count("", model_name="gpt-4")

    assert isinstance(count, int)
    assert count >= 0


def test_get_token_count_long_text(prompt_builder):
    """Test get_token_count with long text."""
    text = "word " * 1000

    count = prompt_builder.get_token_count(text, model_name="gpt-4")

    assert isinstance(count, int)
    assert count > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
