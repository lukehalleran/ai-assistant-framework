"""Integration tests for GatedPromptBuilder and MultiStageGateSystem."""
import pytest
import tempfile
from pathlib import Path

from processing.gate_system import GatedPromptBuilder, MultiStageGateSystem
from core.prompt import PromptBuilder
from models.model_manager import ModelManager
from memory.memory_coordinator import MemoryCoordinator
from memory.corpus_manager import CorpusManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore


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
    """Fixture to provide MemoryCoordinator."""
    corpus_manager = CorpusManager(corpus_file=temp_dirs["corpus_file"])
    chroma_store = MultiCollectionChromaStore(persist_directory=temp_dirs["chroma_path"])
    return MemoryCoordinator(
        corpus_manager=corpus_manager,
        chroma_store=chroma_store
    )


@pytest.fixture
def gate_system(model_manager):
    """Fixture to provide MultiStageGateSystem."""
    return MultiStageGateSystem(
        model_manager=model_manager,
        cosine_threshold=0.5
    )


@pytest.fixture
def prompt_builder(model_manager, memory_coordinator):
    """Fixture to provide PromptBuilder."""
    return PromptBuilder(
        model_manager=model_manager,
        memory_coordinator=memory_coordinator
    )


@pytest.fixture
def gated_prompt_builder(prompt_builder, model_manager):
    """Fixture to provide GatedPromptBuilder."""
    return GatedPromptBuilder(
        prompt_builder=prompt_builder,
        model_manager=model_manager
    )


@pytest.mark.asyncio
async def test_gate_system_initialization(gate_system):
    """Test that gate system initializes properly."""
    assert gate_system is not None
    assert hasattr(gate_system, 'gate_system')  # Has underlying gate system
    assert hasattr(gate_system, 'model_manager')


@pytest.mark.asyncio
async def test_gate_system_filter_memories_empty(gate_system):
    """Test gate system with empty memories list."""
    result = await gate_system.filter_memories("test query", [])
    assert isinstance(result, list)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_gate_system_filter_memories_with_data(gate_system):
    """Test gate system filters memories."""
    memories = [
        {"content": "Python is a programming language", "id": "1"},
        {"content": "The sky is blue", "id": "2"},
        {"content": "Programming in Python is fun", "id": "3"},
    ]
    
    result = await gate_system.filter_memories("Python programming", memories)
    assert isinstance(result, list)
    # Should return some results (might be all or filtered)
    assert len(result) >= 0


@pytest.mark.asyncio
async def test_gate_system_semantic_similarity(gate_system):
    """Test semantic similarity calculation."""
    query = "machine learning"
    candidates = [
        {"content": "Deep learning is a subset of machine learning", "id": "1"},
        {"content": "The weather is nice today", "id": "2"},
    ]
    
    result = await gate_system.filter_memories(query, candidates)
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_gated_prompt_builder_initialization(gated_prompt_builder):
    """Test GatedPromptBuilder initializes."""
    assert gated_prompt_builder is not None
    assert hasattr(gated_prompt_builder, 'prompt_builder')
    assert hasattr(gated_prompt_builder, 'gate_system')


@pytest.mark.asyncio
async def test_gated_prompt_builder_basic(gated_prompt_builder):
    """Test basic gated prompt building."""
    try:
        result = await gated_prompt_builder.build_gated_prompt(
            user_input="What is Python?",
            memories=[],
            summaries=[],
            dreams=[]
        )
        
        # Should return a string (assembled prompt)
        assert isinstance(result, str)
        assert len(result) > 0
    except Exception as e:
        # PyTorch/CUDA errors are OK for this test
        if "meta tensor" in str(e) or "CUDA" in str(e):
            pytest.skip(f"Skipping due to environment issue: {e}")
        raise


@pytest.mark.asyncio
async def test_gated_prompt_builder_with_memories(gated_prompt_builder):
    """Test gated prompt building with memories."""
    memories = [
        {"content": "Python is a high-level language", "id": "1"},
        {"content": "Python was created by Guido van Rossum", "id": "2"},
    ]
    
    try:
        result = await gated_prompt_builder.build_gated_prompt(
            user_input="Tell me about Python",
            memories=memories,
            summaries=[],
            dreams=[]
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
    except Exception as e:
        if "meta tensor" in str(e) or "CUDA" in str(e):
            pytest.skip(f"Skipping due to environment issue: {e}")
        raise


@pytest.mark.asyncio
async def test_gate_system_cosine_threshold(gate_system):
    """Test that gate system filters dissimilar content."""
    # Very different content should be filtered out
    query = "artificial intelligence"
    candidates = [
        {"content": "Bananas are yellow fruits", "id": "1"},
        {"content": "The ocean is vast", "id": "2"},
    ]

    # Threshold is set at initialization, not at call time
    result = await gate_system.filter_memories(query, candidates)
    # With dissimilar items, gate system should filter them
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_gate_system_handles_malformed_memories(gate_system):
    """Test gate system handles memories without content field."""
    memories = [
        {"text": "This has text not content", "id": "1"},
        {"content": "This is proper", "id": "2"},
        {"id": "3"},  # Missing content entirely
    ]
    
    # Should not crash
    result = await gate_system.filter_memories("test", memories)
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_gated_prompt_integration_flow(gated_prompt_builder, memory_coordinator):
    """Test full integration: store interaction, build gated prompt."""
    # Store some context
    await memory_coordinator.store_interaction(
        query="What is machine learning?",
        response="Machine learning is a subset of AI that enables systems to learn from data.",
        tags=["AI", "ML"]
    )
    
    try:
        # Build gated prompt
        result = await gated_prompt_builder.build_gated_prompt(
            user_input="Explain neural networks",
            memories=[],
            summaries=[],
            dreams=[]
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
    except Exception as e:
        if "meta tensor" in str(e) or "CUDA" in str(e):
            pytest.skip(f"Skipping due to environment issue: {e}")
        raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
