"""Comprehensive edge case tests for core components."""
import pytest
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from memory.memory_coordinator import MemoryCoordinator
from memory.corpus_manager import CorpusManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from core.prompt import UnifiedPromptBuilder
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
def memory_coordinator(temp_dirs):
    """Provide MemoryCoordinator."""
    corpus_manager = CorpusManager(corpus_file=temp_dirs["corpus_file"])
    chroma_store = MultiCollectionChromaStore(persist_directory=temp_dirs["chroma_path"])
    return MemoryCoordinator(corpus_manager=corpus_manager, chroma_store=chroma_store)


@pytest.fixture
def model_manager():
    """Provide ModelManager."""
    return ModelManager()


@pytest.fixture
def prompt_builder(memory_coordinator, model_manager):
    """Provide UnifiedPromptBuilder."""
    return UnifiedPromptBuilder(
        memory_coordinator=memory_coordinator,
        model_manager=model_manager
    )


# Memory Coordinator Edge Cases
@pytest.mark.asyncio
async def test_store_interaction_empty_strings(memory_coordinator):
    """An empty assistant response is skipped, not stored (documented
    behavior in MemoryStorage.store_interaction: returns None and never
    touches the corpus for an empty/whitespace-only response)."""
    memory_id = await memory_coordinator.store_interaction(query="", response="")
    assert memory_id is None
    assert memory_coordinator.corpus_manager.corpus == []


@pytest.mark.asyncio
async def test_store_interaction_very_long_text(memory_coordinator):
    """A long but non-degenerate turn is persisted verbatim to both stores."""
    long_query = "Q" * 10000
    long_response = "A" * 10000
    memory_id = await memory_coordinator.store_interaction(
        query=long_query,
        response=long_response
    )
    assert isinstance(memory_id, str) and memory_id
    assert memory_coordinator.corpus_manager.corpus[-1]["query"] == long_query
    assert memory_coordinator.corpus_manager.corpus[-1]["response"] == long_response


@pytest.mark.asyncio
async def test_store_interaction_unicode_edge_cases(memory_coordinator):
    """Unicode text round-trips through the corpus without mangling."""
    query = "Hello 世界 مرحبا שלום"
    response = "Testing émojis 🎉🔥💻"
    memory_id = await memory_coordinator.store_interaction(query=query, response=response)
    assert isinstance(memory_id, str) and memory_id
    stored = memory_coordinator.corpus_manager.corpus[-1]
    assert stored["query"] == query
    assert stored["response"] == response


@pytest.mark.asyncio
async def test_store_interaction_special_chars(memory_coordinator):
    """XML-shaped/escaped and whitespace-control characters round-trip intact."""
    query = "What about <xml> & 'quotes' and \"double\"?"
    response = "Handling \n newlines \t tabs \\ backslashes"
    memory_id = await memory_coordinator.store_interaction(query=query, response=response)
    assert isinstance(memory_id, str) and memory_id
    stored = memory_coordinator.corpus_manager.corpus[-1]
    assert stored["query"] == query
    assert stored["response"] == response


@pytest.mark.asyncio
async def test_get_memories_zero_limit(memory_coordinator):
    """Test get_memories with limit=0."""
    await memory_coordinator.store_interaction("Q", "A")
    memories = await memory_coordinator.get_memories("Q", limit=0)
    assert isinstance(memories, list)


@pytest.mark.asyncio
async def test_get_memories_negative_limit(memory_coordinator):
    """A negative limit is never validated — get_memories applies it as a
    plain Python list slice (`accepted[:limit]`), which drops exactly one
    trailing item rather than raising."""
    for i in range(3):
        await memory_coordinator.store_interaction(f"Topic {i}", f"Content {i}")

    full = await memory_coordinator.get_memories("Topic", limit=10)
    negative = await memory_coordinator.get_memories("Topic", limit=-1)

    assert isinstance(negative, list)
    assert len(negative) == len(full) - 1


@pytest.mark.asyncio
async def test_get_memories_huge_limit(memory_coordinator):
    """Test get_memories with unreasonably large limit."""
    await memory_coordinator.store_interaction("Q", "A")
    memories = await memory_coordinator.get_memories("Q", limit=999999)
    assert isinstance(memories, list)


@pytest.mark.asyncio
async def test_get_memories_empty_query(memory_coordinator):
    """Test get_memories with empty query string."""
    await memory_coordinator.store_interaction("Q", "A")
    memories = await memory_coordinator.get_memories("", limit=10)
    assert isinstance(memories, list)


@pytest.mark.asyncio
async def test_get_memories_unicode_query(memory_coordinator):
    """Test get_memories with unicode query."""
    await memory_coordinator.store_interaction("Python 编程", "Answer")
    memories = await memory_coordinator.get_memories("编程", limit=5)
    assert isinstance(memories, list)


@pytest.mark.asyncio
async def test_add_reflection_empty(memory_coordinator):
    """Test add_reflection with empty text."""
    result = await memory_coordinator.add_reflection("")
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_add_reflection_very_long(memory_coordinator):
    """Test add_reflection with very long text."""
    long_text = "Reflection " * 1000
    result = await memory_coordinator.add_reflection(long_text)
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_add_reflection_with_all_params(memory_coordinator):
    """Test add_reflection with all optional parameters."""
    result = await memory_coordinator.add_reflection(
        text="Test reflection",
        tags=["test", "reflection"],
        source="test_source",
        timestamp=datetime.now().isoformat()
    )
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_get_reflections_zero(memory_coordinator):
    """Test get_reflections with limit=0."""
    reflections = await memory_coordinator.get_reflections(limit=0)
    assert isinstance(reflections, list)


@pytest.mark.asyncio
async def test_process_shutdown_memory_empty(memory_coordinator):
    """With zero non-summary corpus entries (T=0), no due-summary block
    exists, so no consolidator summary is appended to the corpus."""
    await memory_coordinator.process_shutdown_memory(session_conversations=[])
    assert memory_coordinator.corpus_manager.corpus == []


@pytest.mark.asyncio
async def test_process_shutdown_memory_none(memory_coordinator):
    """None falls back to reading the real (here empty) corpus — same no-op
    result as passing an explicit empty list."""
    await memory_coordinator.process_shutdown_memory(session_conversations=None)
    assert memory_coordinator.corpus_manager.corpus == []


@pytest.mark.asyncio
async def test_run_shutdown_reflection_empty(memory_coordinator):
    """No model_manager wired (this fixture builds MemoryCoordinator without
    one) means run_shutdown_reflection's own guard returns False rather than
    attempting a reflection generation call."""
    result = await memory_coordinator.run_shutdown_reflection(session_conversations=[])
    assert result is False


# Prompt Builder Edge Cases
@pytest.mark.asyncio
async def test_build_prompt_all_none(prompt_builder):
    """Test build_prompt with all optional params as None."""
    result = await prompt_builder.build_prompt(
        user_input="test",
        model_name=None,
        personality_config=None,
        system_prompt=None
    )
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_build_prompt_very_long_input(prompt_builder):
    """Test build_prompt with very long user input."""
    long_input = "Tell me about " * 500
    result = await prompt_builder.build_prompt(
        user_input=long_input,
        model_name="gpt-4"
    )
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_build_prompt_unicode_input(prompt_builder):
    """Test build_prompt with unicode characters."""
    result = await prompt_builder.build_prompt(
        user_input="Расскажи о программировании 🚀",
        model_name="gpt-4"
    )
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_build_prompt_special_model_name(prompt_builder):
    """Test build_prompt with various model names."""
    for model in ["gpt-4", "claude-3", "local-model", "unknown-model"]:
        result = await prompt_builder.build_prompt(
            user_input="test",
            model_name=model
        )
        assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_build_prompt_custom_personality(prompt_builder):
    """Test build_prompt with custom personality."""
    personality = {
        "name": "test",
        "traits": ["friendly", "helpful"],
        "num_memories": 50,
        "temperature": 0.9
    }
    result = await prompt_builder.build_prompt(
        user_input="test",
        model_name="gpt-4",
        personality_config=personality
    )
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_build_prompt_with_search_query(prompt_builder):
    """Test build_prompt with separate search query."""
    result = await prompt_builder.build_prompt(
        user_input="What is Python?",
        search_query="Python programming language",
        model_name="gpt-4"
    )
    assert isinstance(result, dict)


# Corpus Manager Edge Cases
def test_corpus_manager_nonexistent_file(temp_dirs):
    """Test CorpusManager with nonexistent file."""
    cm = CorpusManager(corpus_file=temp_dirs["corpus_file"])
    assert cm.corpus == []


def test_corpus_manager_add_entry_minimal(temp_dirs):
    """Test adding entry with minimal info."""
    cm = CorpusManager(corpus_file=temp_dirs["corpus_file"])
    cm.add_entry(query="", response="")
    assert len(cm.corpus) > 0


def test_corpus_manager_get_recent_zero(temp_dirs):
    """Test get_recent_memories with count=0."""
    cm = CorpusManager(corpus_file=temp_dirs["corpus_file"])
    cm.add_entry("Q", "A")
    recent = cm.get_recent_memories(count=0)
    assert isinstance(recent, list)


def test_corpus_manager_get_recent_negative(temp_dirs):
    """Test get_recent_memories with negative count."""
    cm = CorpusManager(corpus_file=temp_dirs["corpus_file"])
    cm.add_entry("Q", "A")
    recent = cm.get_recent_memories(count=-1)
    assert isinstance(recent, list)


def test_corpus_manager_multiple_entries(temp_dirs):
    """Test adding many entries."""
    cm = CorpusManager(corpus_file=temp_dirs["corpus_file"])
    for i in range(100):
        cm.add_entry(f"Q{i}", f"A{i}")
    assert len(cm.corpus) == 100


# ChromaDB Store Edge Cases
def test_chroma_store_add_empty_text(temp_dirs):
    """Empty query/response text is accepted (the document is still the
    non-empty "User: \\nAssistant: " template) and gets a real UUID id back."""
    store = MultiCollectionChromaStore(persist_directory=temp_dirs["chroma_path"])
    doc_id = store.add_conversation_memory(query="", response="", metadata={})
    assert isinstance(doc_id, str) and doc_id
    fetched = store._get_collection("conversations").get(ids=[doc_id])
    assert fetched["ids"] == [doc_id]


def test_chroma_store_add_very_long_text(temp_dirs):
    """A 50K-char turn is stored without truncation or rejection."""
    store = MultiCollectionChromaStore(persist_directory=temp_dirs["chroma_path"])
    long_text = "A" * 50000
    doc_id = store.add_conversation_memory(query=long_text, response=long_text, metadata={})
    assert isinstance(doc_id, str) and doc_id
    fetched = store._get_collection("conversations").get(ids=[doc_id])
    assert fetched["documents"][0] == f"User: {long_text}\nAssistant: {long_text}"


def test_chroma_store_query_nonexistent_collection(temp_dirs):
    """query_collection raises ValueError for a name that was never registered
    in self.collections — it never returns an empty list silently."""
    store = MultiCollectionChromaStore(persist_directory=temp_dirs["chroma_path"])
    with pytest.raises(ValueError, match="Unknown collection"):
        store.query_collection(
            collection_name="nonexistent_collection",
            query_text="test",
            n_results=5
        )


def test_chroma_store_add_with_unicode_metadata(temp_dirs):
    """Test adding document with unicode in metadata."""
    store = MultiCollectionChromaStore(persist_directory=temp_dirs["chroma_path"])
    doc_id = store.add_conversation_memory(
        query="Test",
        response="Response",
        metadata={"topic": "编程", "language": "русский"}
    )
    assert doc_id is not None


# Model Manager Edge Cases
def test_model_manager_stub_response(model_manager):
    """Test _stub_response method."""
    result = model_manager._stub_response("Test prompt")
    assert isinstance(result, str)


def test_model_manager_stub_response_long(model_manager):
    """Test _stub_response with long prompt."""
    long_prompt = "A" * 1000
    result = model_manager._stub_response(long_prompt)
    assert isinstance(result, str)





# Concurrent Operations
@pytest.mark.asyncio
async def test_concurrent_store_interactions(memory_coordinator):
    """10 concurrent store_interaction calls each get a distinct memory id
    and each lands its own corpus entry — no lost updates, no id collisions."""
    tasks = [
        memory_coordinator.store_interaction(f"Q{i}", f"A{i}")
        for i in range(10)
    ]
    memory_ids = await asyncio.gather(*tasks)
    assert all(isinstance(mid, str) and mid for mid in memory_ids)
    assert len(set(memory_ids)) == 10
    assert len(memory_coordinator.corpus_manager.corpus) == 10


@pytest.mark.asyncio
async def test_concurrent_get_memories(memory_coordinator):
    """Test getting memories concurrently."""
    await memory_coordinator.store_interaction("Q", "A")

    tasks = [
        memory_coordinator.get_memories("Q", limit=5)
        for _ in range(10)
    ]
    results = await asyncio.gather(*tasks)
    assert all(isinstance(r, list) for r in results)


@pytest.mark.asyncio
async def test_concurrent_build_prompts(prompt_builder):
    """Test building prompts concurrently."""
    tasks = [
        prompt_builder.build_prompt(user_input=f"Query {i}", model_name="gpt-4")
        for i in range(5)
    ]
    results = await asyncio.gather(*tasks)
    assert all(isinstance(r, dict) for r in results)
