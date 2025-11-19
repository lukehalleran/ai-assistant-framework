"""Tests for internal UnifiedPromptBuilder methods to boost coverage."""
import pytest
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime
from core.prompt import UnifiedPromptBuilder
from memory.memory_coordinator import MemoryCoordinator
from memory.corpus_manager import CorpusManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from models.model_manager import ModelManager


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
def prompt_builder(memory_coordinator, model_manager):
    """Provide UnifiedPromptBuilder."""
    return UnifiedPromptBuilder(
        memory_coordinator=memory_coordinator,
        model_manager=model_manager
    )


def test_middle_out(prompt_builder):
    """Test _middle_out method for text truncation."""
    long_text = "A" * 1000
    # _middle_out is now on token_manager
    truncated = prompt_builder.token_manager._middle_out(
        text=long_text,
        max_tokens=100,
        head_ratio=0.6
    )
    assert isinstance(truncated, str)
    assert len(truncated) <= len(long_text)


def test_middle_out_short_text(prompt_builder):
    """Test _middle_out with text shorter than max_tokens."""
    short_text = "Short text"
    result = prompt_builder.token_manager._middle_out(
        text=short_text,
        max_tokens=1000,
        head_ratio=0.6
    )
    assert result == short_text


def test_middle_out_force_mode(prompt_builder):
    """Test _middle_out with force=True."""
    text = "Some text"
    truncated = prompt_builder.token_manager._middle_out(
        text=text,
        max_tokens=5,
        force=True
    )
    assert isinstance(truncated, str)


@pytest.mark.asyncio
async def test_get_summaries(prompt_builder, memory_coordinator):
    """Test getting summaries via context_gatherer."""
    # Use the separated summaries method
    result = await prompt_builder.context_gatherer._get_summaries_separate("test", 5, 5)
    assert isinstance(result, dict)
    assert "recent" in result or "semantic" in result


@pytest.mark.asyncio
async def test_get_reflections(prompt_builder, memory_coordinator):
    """Test getting reflections via context_gatherer."""
    result = await prompt_builder.context_gatherer._get_reflections_separate("test", 3, 3)
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_get_summaries_hybrid_filtered(prompt_builder, memory_coordinator):
    """Test getting summaries with query filtering."""
    result = await prompt_builder.context_gatherer._get_summaries_separate(
        query="Python",
        recent_limit=5,
        semantic_limit=5
    )
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_get_reflections_hybrid_filtered(prompt_builder, memory_coordinator):
    """Test getting reflections with query filtering."""
    result = await prompt_builder.context_gatherer._get_reflections_separate(
        query="programming",
        recent_limit=3,
        semantic_limit=3
    )
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_hygiene_and_caps(prompt_builder):
    """Test _hygiene_and_caps method."""
    context = {
        "recent_conversations": [{"query": "Q1", "response": "A1"}] * 100,
        "memories": [{"query": "Q1", "response": "A1"}] * 100,
        "semantic_facts": [{"content": "fact"}] * 50,
        "fresh_facts": [],
        "summaries": [{"content": "summary"}] * 30,
        "reflections": [],
        "dreams": [],
        "semantic_chunks": [],
        "wiki": []
    }

    cleaned = await prompt_builder._hygiene_and_caps(context)
    assert isinstance(cleaned, dict)
    assert "memories" in cleaned


def test_manage_token_budget(prompt_builder):
    """Test _manage_token_budget method on token_manager."""
    if prompt_builder.tokenizer_manager is None:
        pytest.skip("tokenizer_manager not available")

    context = {
        "recent_conversations": [{"query": "Q", "response": "A"}] * 20,
        "memories": [{"query": "Q", "response": "A"}] * 20,
        "facts": [{"content": "fact"}] * 10,
        "summaries": [{"content": "summary"}] * 5
    }

    managed = prompt_builder.token_manager._manage_token_budget(context)
    assert isinstance(managed, dict)


def test_assemble_prompt(prompt_builder):
    """Test _assemble_prompt method."""
    context = {
        "recent_conversations": [{"query": "Q1", "response": "A1"}],
        "memories": [],
        "semantic_facts": [],
        "fresh_facts": [],
        "summaries": [],
        "reflections": [],
        "semantic_chunks": [],
        "dreams": [],
        "wiki": []
    }

    prompt = prompt_builder._assemble_prompt(
        context=context,
        user_input="Test query",
        system_prompt="You are a helpful assistant."
    )

    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_get_token_count(prompt_builder):
    """Test get_token_count method."""
    if prompt_builder.tokenizer_manager is None:
        pytest.skip("tokenizer_manager not available")

    text = "This is a test sentence with some words."
    count = prompt_builder.get_token_count(text, model_name="gpt-4")
    assert isinstance(count, int)
    assert count > 0


def test_extract_text(prompt_builder):
    """Test _extract_text method."""
    # String input
    text1 = prompt_builder._extract_text("plain string")
    assert text1 == "plain string"

    # Dict with content
    text2 = prompt_builder._extract_text({"content": "dict content"})
    assert "dict content" in text2

    # Dict with response
    text3 = prompt_builder._extract_text({"response": "A"})
    assert "A" in text3


def test_format_memory(prompt_builder):
    """Test _format_memory method on formatter."""
    memory = {
        "query": "What is Python?",
        "response": "Python is a programming language.",
        "timestamp": datetime.now().isoformat()
    }

    formatted = prompt_builder.formatter._format_memory(memory)
    assert isinstance(formatted, str)
    assert "Python" in formatted


def test_format_memory_without_timestamp(prompt_builder):
    """Test _format_memory without timestamp."""
    memory = {
        "query": "Test query",
        "response": "Test response"
    }

    formatted = prompt_builder.formatter._format_memory(memory)
    assert isinstance(formatted, str)


@pytest.mark.asyncio
async def test_get_recent_facts(prompt_builder, memory_coordinator):
    """Test get_recent_facts via context_gatherer."""
    facts = await prompt_builder.context_gatherer.get_recent_facts(limit=5)
    assert isinstance(facts, list)


@pytest.mark.asyncio
async def test_get_facts_with_query(prompt_builder, memory_coordinator):
    """Test get_facts via context_gatherer."""
    facts = await prompt_builder.context_gatherer.get_facts(query="Python", limit=8)
    assert isinstance(facts, list)


@pytest.mark.asyncio
async def test_gather_context(prompt_builder, memory_coordinator):
    """Test build_prompt method (replaces _gather_context)."""
    await memory_coordinator.store_interaction("Q", "A")

    context = await prompt_builder.build_prompt(
        user_input="Test query"
    )

    assert isinstance(context, dict)
    assert "memories" in context


def test_build_default_memory_coordinator(prompt_builder, model_manager):
    """Test _build_default_memory_coordinator method."""
    mc = prompt_builder._build_default_memory_coordinator()
    assert mc is not None
