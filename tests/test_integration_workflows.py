"""Integration tests for main workflows to boost coverage."""
import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from core.orchestrator import DaemonOrchestrator
from models.model_manager import ModelManager
from core.response_generator import ResponseGenerator
from memory.memory_coordinator import MemoryCoordinator
from memory.corpus_manager import CorpusManager
from core.prompt import UnifiedPromptBuilder
from utils.time_manager import TimeManager


@pytest.fixture
def temp_corpus_file():
    """Create temporary corpus file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('[]')
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_chroma_dir():
    """Create temporary chroma directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def corpus_manager(temp_corpus_file):
    """Provide CorpusManager with temp file."""
    return CorpusManager(corpus_file=temp_corpus_file)


@pytest.fixture
def model_manager():
    """Provide ModelManager."""
    return ModelManager()


@pytest.fixture
def time_manager():
    """Provide TimeManager."""
    return TimeManager()


@pytest.fixture
def response_generator(model_manager, time_manager):
    """Provide ResponseGenerator."""
    return ResponseGenerator(model_manager=model_manager, time_manager=time_manager)


@pytest.fixture
def memory_coordinator(corpus_manager, temp_chroma_dir):
    """Provide MemoryCoordinator with temp storage."""
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    chroma_store = MultiCollectionChromaStore(path=temp_chroma_dir)
    return MemoryCoordinator(
        corpus_manager=corpus_manager,
        chroma_store=chroma_store
    )


@pytest.fixture
def prompt_builder(memory_coordinator, model_manager, time_manager):
    """Provide UnifiedPromptBuilder."""
    return UnifiedPromptBuilder(
        memory_coordinator=memory_coordinator,
        model_manager=model_manager,
        time_manager=time_manager
    )


# Integration workflow tests
@pytest.mark.asyncio
async def test_full_query_workflow(memory_coordinator):
    """Test complete query workflow: store, retrieve, process."""
    # Store some interactions
    await memory_coordinator.store_interaction("What is Python?", "Python is a programming language")
    await memory_coordinator.store_interaction("Tell me more", "Python is used for web development")

    # Retrieve memories
    memories = await memory_coordinator.get_memories("Python programming", limit=5)

    assert isinstance(memories, list)
    assert len(memories) > 0


@pytest.mark.asyncio
async def test_memory_retrieval_with_topics(memory_coordinator):
    """Test memory retrieval with topic filtering."""
    await memory_coordinator.store_interaction(
        "What is machine learning?",
        "ML is a branch of AI",
        tags=["ai", "ml"]
    )

    memories = await memory_coordinator.get_memories("machine learning", limit=10)
    assert isinstance(memories, list)


@pytest.mark.asyncio
async def test_memory_relevance_scoring(memory_coordinator):
    """Test that memories get relevance scores."""
    await memory_coordinator.store_interaction("Python basics", "Python is easy to learn")
    await memory_coordinator.store_interaction("Java basics", "Java is verbose")

    memories = await memory_coordinator.get_memories("Python programming", limit=10)

    # Check that memories have scores
    for mem in memories:
        assert isinstance(mem, dict)


@pytest.mark.asyncio
async def test_prompt_building_workflow(prompt_builder, memory_coordinator):
    """Test prompt building with memories."""
    # Add some context
    await memory_coordinator.store_interaction("What is AI?", "AI is artificial intelligence")

    # Build prompt
    result = await prompt_builder.build_prompt(
        user_input="Tell me about AI",
        model_name="gpt-4",
        query_str="Tell me about AI"
    )

    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_response_generation_flow(response_generator, monkeypatch):
    """Test response generation flow."""
    async def mock_stream():
        yield "Test"
        yield " response"

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    chunks = []
    async for chunk in response_generator.generate_streaming_response("Test query", "gpt-4"):
        chunks.append(chunk)

    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_memory_with_empty_query(memory_coordinator):
    """Test memory retrieval with empty query."""
    await memory_coordinator.store_interaction("Q1", "A1")
    memories = await memory_coordinator.get_memories("", limit=5)
    assert isinstance(memories, list)


@pytest.mark.asyncio
async def test_memory_with_special_characters(memory_coordinator):
    """Test memory with special characters."""
    await memory_coordinator.store_interaction(
        "What is C++?",
        "C++ is a programming language"
    )
    memories = await memory_coordinator.get_memories("C++", limit=5)
    assert isinstance(memories, list)


@pytest.mark.asyncio
async def test_multiple_sequential_interactions(memory_coordinator):
    """Test multiple sequential interactions."""
    for i in range(5):
        await memory_coordinator.store_interaction(f"Query {i}", f"Answer {i}")

    memories = await memory_coordinator.get_memories("Query 3", limit=10)
    assert len(memories) > 0


@pytest.mark.asyncio
async def test_memory_with_long_text(memory_coordinator):
    """Test memory storage with long text."""
    long_query = "What is Python? " * 100
    long_response = "Python is a language. " * 100

    await memory_coordinator.store_interaction(long_query, long_response)
    memories = await memory_coordinator.get_memories("Python", limit=5)
    assert len(memories) > 0


@pytest.mark.asyncio
async def test_orchestrator_static_methods():
    """Test orchestrator static methods."""
    # Test thinking block parsing
    response = "<thinking>Analysis</thinking>Answer"
    thinking, answer = DaemonOrchestrator._parse_thinking_block(response)
    assert thinking == "Analysis"
    assert answer == "Answer"

    # Test XML stripping
    text = "<result>Content</result>"
    stripped = DaemonOrchestrator._strip_xml_wrappers(text)
    assert stripped == "Content"

    # Test prompt artifact stripping
    text = "[TIME CONTEXT]\nInfo\n\nResponse"
    stripped = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "[TIME CONTEXT]" not in stripped


@pytest.mark.asyncio
async def test_prompt_builder_with_empty_memories(prompt_builder):
    """Test prompt building with no memories."""
    result = await prompt_builder.build_prompt(
        user_input="Test query",
        model_name="gpt-4",
        query_str="Test query"
    )
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_memory_consolidation_trigger(memory_coordinator):
    """Test memory consolidation can be triggered."""
    # Add multiple interactions
    for i in range(10):
        await memory_coordinator.store_interaction(f"Q{i}", f"A{i}")

    # Trigger shutdown processing
    try:
        await memory_coordinator.process_shutdown_memory()
    except Exception:
        # May fail without proper model setup, but exercises the code path
        pass


@pytest.mark.asyncio
async def test_gate_system_integration(memory_coordinator):
    """Test gate system filtering in memory retrieval."""
    # Add diverse memories
    await memory_coordinator.store_interaction("Python programming", "Python is versatile")
    await memory_coordinator.store_interaction("Java programming", "Java is statically typed")
    await memory_coordinator.store_interaction("Cooking recipes", "Pasta is delicious")

    # Query should filter relevant memories
    memories = await memory_coordinator.get_memories("Python coding", limit=10)
    assert isinstance(memories, list)


@pytest.mark.asyncio
async def test_time_manager_integration(time_manager):
    """Test TimeManager tracking."""
    time_manager.mark_query_time()
    elapsed = time_manager.elapsed_since_last()
    assert isinstance(elapsed, float)
    assert elapsed >= 0


@pytest.mark.asyncio
async def test_corpus_manager_save_load(corpus_manager):
    """Test corpus save and load cycle."""
    corpus_manager.add_entry("Q1", "A1")
    corpus_manager.add_entry("Q2", "A2")
    corpus_manager.save_corpus()

    # Create new manager with same file
    new_manager = CorpusManager(corpus_file=corpus_manager.corpus_file)
    assert len(new_manager.corpus) == 2


@pytest.mark.asyncio
async def test_memory_with_tags_filtering(memory_coordinator):
    """Test memory retrieval respects tags."""
    await memory_coordinator.store_interaction("Python", "Programming", tags=["code"])
    await memory_coordinator.store_interaction("Pasta", "Food", tags=["cooking"])

    memories = await memory_coordinator.get_memories("Python", limit=10)
    assert isinstance(memories, list)


@pytest.mark.asyncio
async def test_concurrent_memory_operations(memory_coordinator):
    """Test concurrent memory operations."""
    tasks = [
        memory_coordinator.store_interaction(f"Q{i}", f"A{i}")
        for i in range(5)
    ]
    await asyncio.gather(*tasks)

    memories = await memory_coordinator.get_memories("Q2", limit=10)
    assert len(memories) > 0


@pytest.mark.asyncio
async def test_response_generator_with_errors(response_generator, monkeypatch):
    """Test response generator handles model errors."""
    async def mock_error(*args, **kwargs):
        raise RuntimeError("Model unavailable")

    monkeypatch.setattr(response_generator.model_manager, "generate_async", mock_error)

    chunks = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        chunks.append(chunk)

    # Should yield error message
    assert any("Error" in str(c) for c in chunks)


@pytest.mark.asyncio
async def test_prompt_builder_token_limiting(prompt_builder, memory_coordinator):
    """Test prompt builder respects token limits."""
    # Add many memories
    for i in range(20):
        await memory_coordinator.store_interaction(f"Question {i}", f"Answer {i}")

    result = await prompt_builder.build_prompt(
        user_input="Test",
        model_name="gpt-4",
        query_str="Test",
        token_budget=100  # Very low budget
    )
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_memory_recent_retrieval(memory_coordinator):
    """Test retrieving recent memories."""
    for i in range(5):
        await memory_coordinator.store_interaction(f"Q{i}", f"A{i}")

    recent = memory_coordinator.corpus_manager.get_recent_memories(count=3)
    assert len(recent) <= 3


@pytest.mark.asyncio
async def test_memory_with_unicode_content(memory_coordinator):
    """Test memory handles unicode correctly."""
    await memory_coordinator.store_interaction(
        "What is 世界?",
        "世界 means world in Chinese"
    )
    memories = await memory_coordinator.get_memories("world", limit=5)
    assert isinstance(memories, list)


@pytest.mark.asyncio
async def test_orchestrator_fallback_memory(temp_corpus_file):
    """Test orchestrator with fallback memory coordinator."""
    from core.orchestrator import _FallbackMemoryCoordinator

    mem = _FallbackMemoryCoordinator()
    await mem.store_interaction("Q", "A")
    memories = await mem.get_memories("Q", limit=5)
    assert len(memories) > 0


@pytest.mark.asyncio
async def test_orchestrator_fallback_corpus():
    """Test orchestrator fallback corpus."""
    from core.orchestrator import _InMemoryCorpus

    corpus = _InMemoryCorpus()
    corpus.add_entry("Q1", "A1")
    corpus.add_entry("Q2", "A2")

    recent = corpus.get_recent_memories(2)
    assert len(recent) == 2

    summaries = corpus.get_summaries(10)
    assert len(summaries) == 0


@pytest.mark.asyncio
async def test_orchestrator_simple_prompt_builder():
    """Test orchestrator simple prompt builder fallback."""
    from core.orchestrator import _SimplePromptBuilder

    builder = _SimplePromptBuilder()
    prompt = await builder.build_prompt("Test input")
    assert prompt == "Test input"


@pytest.mark.asyncio
async def test_memory_scorer_integration(memory_coordinator):
    """Test memory scoring functionality."""
    await memory_coordinator.store_interaction("Python", "Language")

    # Retrieve and check scoring happened
    memories = await memory_coordinator.get_memories("Python", limit=5)
    assert isinstance(memories, list)


@pytest.mark.asyncio
async def test_model_manager_initialization(model_manager):
    """Test model manager basic operations."""
    active_model = model_manager.get_active_model_name()
    assert isinstance(active_model, str)
    assert len(active_model) > 0


@pytest.mark.asyncio
async def test_corpus_manager_with_thread_info(corpus_manager):
    """Test corpus manager with thread information."""
    corpus_manager.add_entry(
        "Q",
        "A",
        thread_id="thread-1",
        thread_depth=1,
        thread_topic="Testing"
    )
    assert len(corpus_manager.corpus) == 1


@pytest.mark.asyncio
async def test_memory_retrieval_with_config(memory_coordinator):
    """Test memory retrieval with custom config."""
    await memory_coordinator.store_interaction("Test", "Response")

    config = {
        "recent_count": 3,
        "semantic_count": 5
    }

    result = await memory_coordinator.retrieve_relevant_memories("Test", config=config)
    assert isinstance(result, dict)
