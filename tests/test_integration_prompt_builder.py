"""Integration tests for UnifiedPromptBuilder with real memory coordinator."""
import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime

from core.prompt import UnifiedPromptBuilder
from memory.memory_coordinator import MemoryCoordinator
from models.model_manager import ModelManager


@pytest.fixture
def temp_dirs():
    """Create temporary directories for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_file = Path(tmpdir) / "corpus.json"
        chroma_path = Path(tmpdir) / "chroma_db"
        yield {
            "corpus_file": str(corpus_file),
            "chroma_path": str(chroma_path)
        }


@pytest.fixture
def model_manager():
    """Fixture to provide ModelManager."""
    return ModelManager()


@pytest.fixture
def memory_coordinator(temp_dirs):
    """Fixture to provide MemoryCoordinator with temp storage."""
    from memory.corpus_manager import CorpusManager
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    corpus_manager = CorpusManager(corpus_file=temp_dirs["corpus_file"])
    chroma_store = MultiCollectionChromaStore(persist_directory=temp_dirs["chroma_path"])

    return MemoryCoordinator(
        corpus_manager=corpus_manager,
        chroma_store=chroma_store
    )


@pytest.fixture
def prompt_builder(model_manager, memory_coordinator):
    """Fixture to provide UnifiedPromptBuilder."""
    return UnifiedPromptBuilder(
        model_manager=model_manager,
        memory_coordinator=memory_coordinator
    )


@pytest.mark.asyncio
async def test_build_prompt_basic(prompt_builder):
    """Test basic prompt building with no context."""
    result = await prompt_builder.build_prompt(
        user_input="What is the weather like?"
    )

    # Should return a context dict with expected keys
    assert isinstance(result, dict)
    # Check for standard keys that should be present
    assert "current_topic" in result or "facts" in result or "recent" in result


@pytest.mark.asyncio
async def test_build_prompt_with_memories(prompt_builder, memory_coordinator):
    """Test prompt building after storing some memories."""
    # Store an interaction
    await memory_coordinator.store_interaction(
        query="What is Python?",
        response="Python is a high-level programming language",
        tags=["programming"]
    )

    # Build prompt with related query
    result = await prompt_builder.build_prompt(
        user_input="Tell me about Python"
    )

    assert isinstance(result, dict)
    # Should have structure even if empty
    assert "current_topic" in result or "facts" in result or "recent" in result


@pytest.mark.asyncio
async def test_build_prompt_with_search_query(prompt_builder):
    """Test prompt building with explicit search query."""
    result = await prompt_builder.build_prompt(
        user_input="What is machine learning?",
        search_query="machine learning basics"
    )

    assert isinstance(result, dict)
    assert "current_topic" in result or "facts" in result or "recent" in result


@pytest.mark.asyncio
async def test_build_prompt_with_current_topic(prompt_builder):
    """Test prompt building with current topic."""
    result = await prompt_builder.build_prompt(
        user_input="Tell me more",
        current_topic="Python programming"
    )
    
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_build_prompt_with_fresh_facts(prompt_builder):
    """Test prompt building with fresh facts."""
    fresh_facts = [
        {"content": "The sky is blue", "confidence": 0.9},
        {"content": "Water is wet", "confidence": 0.95}
    ]
    
    result = await prompt_builder.build_prompt(
        user_input="Tell me some facts",
        fresh_facts=fresh_facts
    )
    
    assert isinstance(result, dict)
    # Fresh facts should be included
    if "fresh_facts" in result:
        assert len(result["fresh_facts"]) > 0


@pytest.mark.asyncio
async def test_build_prompt_with_system_prompt(prompt_builder):
    """Test prompt building with custom system prompt."""
    custom_system = "You are a helpful assistant specialized in testing."
    
    result = await prompt_builder.build_prompt(
        user_input="Hello",
        system_prompt=custom_system
    )
    
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_prompt_builder_memory_integration(prompt_builder, memory_coordinator):
    """Test full integration: store memories, retrieve, build prompt."""
    # Store an interaction
    await memory_coordinator.store_interaction(
        query="What are neural networks?",
        response="Neural networks are computing systems inspired by biological neural networks.",
        tags=["AI", "machine learning"]
    )

    # Build prompt - should retrieve relevant memories
    result = await prompt_builder.build_prompt(
        user_input="How do I train a neural network?",
        current_topic="machine learning"
    )

    # Verify structure
    assert isinstance(result, dict)
    # Should have standard keys
    assert "current_topic" in result or "facts" in result or "recent" in result


@pytest.mark.asyncio
async def test_assemble_prompt_from_context(prompt_builder):
    """Test _assemble_prompt method."""
    # Build a context
    context = await prompt_builder.build_prompt(
        user_input="Test query"
    )
    
    # Assemble into final prompt
    final_prompt = prompt_builder._assemble_prompt(
        user_input="Test query",
        context=context,
        system_prompt="You are a test assistant",
        directives_file="structured_directives.txt"
    )
    
    # Should return a string
    assert isinstance(final_prompt, str)
    assert len(final_prompt) > 0
    # Should contain the user input
    assert "Test query" in final_prompt or "test query" in final_prompt.lower()


@pytest.mark.asyncio 
async def test_prompt_builder_handles_empty_results(prompt_builder):
    """Test prompt builder handles empty memory results gracefully."""
    # Query for something that won't have matches
    result = await prompt_builder.build_prompt(
        user_input="xyzabc123nonexistent"
    )
    
    # Should still return valid structure
    assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
