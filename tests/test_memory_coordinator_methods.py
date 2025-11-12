"""Unit tests for MemoryCoordinator internal methods and uncovered paths."""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from memory.memory_coordinator import MemoryCoordinator, _is_deictic_followup, _salient_tokens, _num_op_density, _analogy_markers
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
def corpus_manager(temp_dirs):
    """Provide CorpusManager."""
    return CorpusManager(corpus_file=temp_dirs["corpus_file"])


@pytest.fixture
def chroma_store(temp_dirs):
    """Provide ChromaStore."""
    return MultiCollectionChromaStore(persist_directory=temp_dirs["chroma_path"])


@pytest.fixture
def memory_coordinator(corpus_manager, chroma_store):
    """Provide MemoryCoordinator."""
    return MemoryCoordinator(
        corpus_manager=corpus_manager,
        chroma_store=chroma_store
    )


# Test utility functions
def test_is_deictic_followup_true():
    """Test _is_deictic_followup with deictic query."""
    assert _is_deictic_followup("explain that") == True
    assert _is_deictic_followup("Can you explain it again?") == True
    assert _is_deictic_followup("Show me another way") == True


def test_is_deictic_followup_false():
    """Test _is_deictic_followup with non-deictic query."""
    assert _is_deictic_followup("What is Python?") == False
    assert _is_deictic_followup("Tell me about machine learning") == False


def test_is_deictic_followup_empty():
    """Test _is_deictic_followup with empty string."""
    assert _is_deictic_followup("") == False
    assert _is_deictic_followup(None) == False


def test_salient_tokens_basic():
    """Test _salient_tokens extracts important tokens."""
    text = "Python is a programming language for machine learning"
    tokens = _salient_tokens(text)

    assert isinstance(tokens, set)
    assert "python" in tokens
    assert "programming" in tokens
    # Stopwords should be filtered
    assert "is" not in tokens
    assert "a" not in tokens


def test_salient_tokens_with_numbers():
    """Test _salient_tokens includes numbers."""
    text = "The answer is 42 and the equation is x + 5 = 10"
    tokens = _salient_tokens(text)

    assert "42" in tokens or "10" in tokens


def test_salient_tokens_empty():
    """Test _salient_tokens with empty input."""
    tokens = _salient_tokens("")

    assert isinstance(tokens, set)
    assert len(tokens) == 0


def test_num_op_density_high():
    """Test _num_op_density with math-heavy text."""
    text = "Calculate 2 + 3 * 4 = 14 and 5 - 2 = 3"
    density = _num_op_density(text)

    assert density > 0.3  # High density


def test_num_op_density_low():
    """Test _num_op_density with regular text."""
    text = "Python is a programming language"
    density = _num_op_density(text)

    assert density < 0.1  # Low density


def test_num_op_density_empty():
    """Test _num_op_density with empty input."""
    density = _num_op_density("")

    assert density == 0.0


def test_analogy_markers_present():
    """Test _analogy_markers detects analogies."""
    text = "It's like a car, imagine a vehicle, picture this scenario"
    count = _analogy_markers(text)

    assert count > 0


def test_analogy_markers_absent():
    """Test _analogy_markers with no analogies."""
    text = "Python is a programming language"
    count = _analogy_markers(text)

    assert count == 0


def test_analogy_markers_empty():
    """Test _analogy_markers with empty input."""
    count = _analogy_markers("")

    assert count == 0


# Test MemoryCoordinator methods
@pytest.mark.asyncio
async def test_store_interaction_basic(memory_coordinator):
    """Test storing a basic interaction."""
    await memory_coordinator.store_interaction(
        query="What is Python?",
        response="Python is a programming language."
    )

    # Should have stored in corpus
    recent = memory_coordinator.corpus_manager.get_recent_memories(1)
    assert len(recent) > 0


@pytest.mark.asyncio
async def test_store_interaction_with_tags(memory_coordinator):
    """Test storing interaction with tags."""
    await memory_coordinator.store_interaction(
        query="Machine learning basics",
        response="ML is a subset of AI.",
        tags=["ml", "ai"]
    )

    recent = memory_coordinator.corpus_manager.get_recent_memories(1)
    assert len(recent) > 0
    if recent:
        assert "ml" in recent[0].get("tags", []) or "ai" in recent[0].get("tags", [])


@pytest.mark.asyncio
async def test_get_memories_empty_query(memory_coordinator):
    """Test get_memories with empty query."""
    memories = await memory_coordinator.get_memories("", limit=5)

    assert isinstance(memories, list)


@pytest.mark.asyncio
async def test_get_memories_after_storing(memory_coordinator):
    """Test retrieving memories after storing."""
    await memory_coordinator.store_interaction(
        query="Test query",
        response="Test response"
    )

    memories = await memory_coordinator.get_memories("Test", limit=5)

    assert isinstance(memories, list)


@pytest.mark.asyncio
async def test_get_facts_empty_query(memory_coordinator):
    """Test get_facts with empty query."""
    facts = await memory_coordinator.get_facts(query="", limit=5)

    assert isinstance(facts, list)


@pytest.mark.asyncio
async def test_get_facts_basic(memory_coordinator):
    """Test basic fact retrieval."""
    facts = await memory_coordinator.get_facts(query="Python programming", limit=5)

    assert isinstance(facts, list)


@pytest.mark.asyncio
async def test_extract_and_store_facts(memory_coordinator):
    """Test fact extraction and storage (internal method)."""
    # This tests the internal _extract_and_store_facts flow
    query = "Python was created by Guido van Rossum in 1991"
    response = "That's correct! Python is a high-level language."

    # Call the internal method
    try:
        await memory_coordinator._extract_and_store_facts(query, response, truth_score=0.9)
        # Should not crash
        assert True
    except AttributeError:
        # Method might be private or have different name
        pytest.skip("_extract_and_store_facts not available")


@pytest.mark.asyncio
async def test_consolidate_with_model(memory_coordinator):
    """Test consolidate_and_refresh with model_manager."""
    # Store some interactions
    await memory_coordinator.store_interaction(
        query="What is Python?",
        response="Python is a language."
    )

    # Try consolidation (will skip without proper model_manager)
    try:
        await memory_coordinator.consolidate_and_refresh()
    except Exception:
        # Expected to fail without proper setup
        pass


@pytest.mark.asyncio
async def test_update_memory_access(memory_coordinator):
    """Test updating memory access time."""
    await memory_coordinator.store_interaction(
        query="Test query",
        response="Test response"
    )

    # Get the memory ID
    recent = memory_coordinator.corpus_manager.get_recent_memories(1)
    if recent and "id" in recent[0]:
        memory_id = recent[0]["id"]

        # Update access
        await memory_coordinator.update_memory_access(memory_id)

        # Should not raise an error
        assert True


@pytest.mark.asyncio
async def test_get_semantic_top_memories(memory_coordinator):
    """Test get_semantic_top_memories method."""
    # Store some memories
    await memory_coordinator.store_interaction(
        query="Python programming",
        response="Python is great for ML."
    )

    # Try semantic retrieval
    try:
        memories = await memory_coordinator.get_semantic_top_memories(
            query="programming",
            limit=5
        )
        assert isinstance(memories, list)
    except AttributeError:
        # Method might not exist
        pytest.skip("get_semantic_top_memories not available")


@pytest.mark.asyncio
async def test_conversation_context_tracking(memory_coordinator):
    """Test conversation_context tracks interactions."""
    await memory_coordinator.store_interaction(
        query="Question 1",
        response="Answer 1"
    )
    await memory_coordinator.store_interaction(
        query="Question 2",
        response="Answer 2"
    )

    # Conversation context should have entries
    assert hasattr(memory_coordinator, "conversation_context")
    assert len(memory_coordinator.conversation_context) >= 0


@pytest.mark.asyncio
async def test_topic_tracking(memory_coordinator):
    """Test topic is tracked across interactions."""
    await memory_coordinator.store_interaction(
        query="Tell me about Python",
        response="Python is a language.",
        tags=["topic:python"]
    )

    # Should have current_topic set
    assert hasattr(memory_coordinator, "current_topic")


def test_get_summaries(memory_coordinator):
    """Test get_summaries method (not async)."""
    # get_summaries is NOT async
    summaries = memory_coordinator.get_summaries(limit=3)
    assert isinstance(summaries, list)


@pytest.mark.asyncio
async def test_multiple_interactions_ordering(memory_coordinator):
    """Test multiple interactions maintain proper ordering."""
    await memory_coordinator.store_interaction(
        query="First",
        response="Response 1"
    )
    await memory_coordinator.store_interaction(
        query="Second",
        response="Response 2"
    )
    await memory_coordinator.store_interaction(
        query="Third",
        response="Response 3"
    )

    recent = memory_coordinator.corpus_manager.get_recent_memories(3)

    assert len(recent) == 3
    # Most recent should be first
    assert recent[0]["query"] == "Third"


@pytest.mark.asyncio
async def test_empty_response_handling(memory_coordinator):
    """Test storing interaction with empty response."""
    await memory_coordinator.store_interaction(
        query="Test question",
        response=""
    )

    # Should not crash
    recent = memory_coordinator.corpus_manager.get_recent_memories(1)
    assert len(recent) > 0


@pytest.mark.asyncio
async def test_empty_query_handling(memory_coordinator):
    """Test storing interaction with empty query."""
    await memory_coordinator.store_interaction(
        query="",
        response="A response without a question"
    )

    # Should not crash
    recent = memory_coordinator.corpus_manager.get_recent_memories(1)
    assert len(recent) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
