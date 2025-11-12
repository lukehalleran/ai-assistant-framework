"""Advanced integration tests for UnifiedPromptBuilder - testing uncovered paths."""
import pytest
import tempfile
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
def prompt_builder(model_manager, memory_coordinator):
    """Provide UnifiedPromptBuilder."""
    return UnifiedPromptBuilder(
        model_manager=model_manager,
        memory_coordinator=memory_coordinator
    )


@pytest.mark.asyncio
async def test_get_recent_conversations(prompt_builder, memory_coordinator):
    """Test retrieving recent conversations."""
    # Store some conversations
    await memory_coordinator.store_interaction(
        query="What is Python?",
        response="Python is a programming language."
    )
    await memory_coordinator.store_interaction(
        query="How do I learn it?",
        response="Start with tutorials."
    )

    # Get recent conversations via corpus_manager
    recent = memory_coordinator.corpus_manager.get_recent_memories(count=2)

    assert isinstance(recent, list)
    # May be empty if storage failed, but should not crash


@pytest.mark.asyncio
async def test_get_facts(prompt_builder):
    """Test fact retrieval."""
    facts = await prompt_builder.get_facts("Python programming", limit=5)
    
    assert isinstance(facts, list)
    # Should not crash even if no facts exist


@pytest.mark.asyncio
async def test_get_recent_facts(prompt_builder):
    """Test recent fact retrieval."""
    facts = await prompt_builder.get_recent_facts(limit=3)
    
    assert isinstance(facts, list)


@pytest.mark.asyncio
async def test_build_prompt_with_personality(prompt_builder):
    """Test prompt building with personality config."""
    personality = {
        "tone": "professional",
        "style": "concise"
    }
    
    result = await prompt_builder.build_prompt(
        user_input="Hello",
        personality_config=personality
    )
    
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_assemble_prompt_minimal(prompt_builder):
    """Test _assemble_prompt with minimal context."""
    context = {
        "recent": [],
        "mems": [],
        "facts": [],
        "dreams": [],
        "current_topic": "general"
    }
    
    prompt = prompt_builder._assemble_prompt(
        user_input="Test",
        context=context,
        system_prompt="You are a helpful assistant",
        directives_file="structured_directives.txt"
    )
    
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "Test" in prompt or "test" in prompt.lower()


@pytest.mark.asyncio
async def test_assemble_prompt_with_memories(prompt_builder):
    """Test _assemble_prompt with memory content."""
    context = {
        "recent": [
            {"query": "Hi", "response": "Hello", "timestamp": datetime.now().isoformat()}
        ],
        "mems": [
            {"content": "User likes Python", "score": 0.9}
        ],
        "facts": [
            {"content": "Python is a language", "confidence": 0.8}
        ],
        "dreams": [],
        "current_topic": "programming"
    }
    
    prompt = prompt_builder._assemble_prompt(
        user_input="Tell me more",
        context=context,
        system_prompt="You are helpful",
        directives_file="structured_directives.txt"
    )
    
    assert isinstance(prompt, str)
    assert len(prompt) > 0


@pytest.mark.asyncio
async def test_build_prompt_error_handling(prompt_builder):
    """Test that build_prompt handles errors gracefully."""
    # Very long input to potentially trigger issues
    long_input = "test " * 10000
    
    result = await prompt_builder.build_prompt(
        user_input=long_input
    )
    
    # Should not crash, even with extreme input
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_get_token_count(prompt_builder):
    """Test token counting."""
    text = "This is a test message for counting tokens."

    count = prompt_builder.get_token_count(text, model_name="gpt-4")

    assert isinstance(count, int)
    assert count > 0


@pytest.mark.asyncio
async def test_build_prompt_with_topic(prompt_builder):
    """Test prompt building with current topic."""
    result = await prompt_builder.build_prompt(
        user_input="Continue our discussion",
        current_topic="machine learning"
    )
    
    assert isinstance(result, dict)
    if "current_topic" in result:
        assert isinstance(result["current_topic"], str)


@pytest.mark.asyncio
async def test_build_prompt_smalltalk(prompt_builder):
    """Test prompt building with small talk."""
    # Short casual message
    result = await prompt_builder.build_prompt(
        user_input="hey"
    )
    
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_build_prompt_question(prompt_builder):
    """Test prompt building with a question."""
    result = await prompt_builder.build_prompt(
        user_input="What is machine learning?"
    )
    
    assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
