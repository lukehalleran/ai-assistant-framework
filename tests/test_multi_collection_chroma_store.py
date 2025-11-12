"""Tests for MultiCollectionChromaStore."""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from memory.storage.multi_collection_chroma_store import (
    MultiCollectionChromaStore,
    _flatten_for_chroma
)


# Test utility function
def test_flatten_for_chroma_primitives():
    """Test _flatten_for_chroma with primitive types."""
    md = {
        "str": "value",
        "int": 42,
        "float": 3.14,
        "bool": True
    }

    result = _flatten_for_chroma(md)

    assert result == md


def test_flatten_for_chroma_none():
    """Test _flatten_for_chroma filters None values."""
    md = {
        "valid": "value",
        "invalid": None
    }

    result = _flatten_for_chroma(md)

    assert "valid" in result
    assert "invalid" not in result


def test_flatten_for_chroma_list():
    """Test _flatten_for_chroma converts lists to comma-separated strings."""
    md = {
        "tags": ["python", "ml", "ai"]
    }

    result = _flatten_for_chroma(md)

    assert result["tags"] == "python,ml,ai"


def test_flatten_for_chroma_dict():
    """Test _flatten_for_chroma converts dicts to JSON."""
    md = {
        "data": {"key": "value", "nested": {"a": 1}}
    }

    result = _flatten_for_chroma(md)

    assert isinstance(result["data"], str)
    assert "key" in result["data"]


def test_flatten_for_chroma_empty():
    """Test _flatten_for_chroma with empty dict."""
    result = _flatten_for_chroma({})

    assert result == {}


def test_flatten_for_chroma_none_input():
    """Test _flatten_for_chroma with None input."""
    result = _flatten_for_chroma(None)

    assert result == {}


# Test MultiCollectionChromaStore
@pytest.fixture
def temp_chroma_dir():
    """Provide temporary ChromaDB directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "chroma_test")


@pytest.fixture
def chroma_store(temp_chroma_dir):
    """Provide MultiCollectionChromaStore."""
    return MultiCollectionChromaStore(persist_directory=temp_chroma_dir)


def test_chroma_store_init(temp_chroma_dir):
    """Test MultiCollectionChromaStore initialization."""
    store = MultiCollectionChromaStore(persist_directory=temp_chroma_dir)

    assert store.persist_directory == temp_chroma_dir
    assert store.client is not None
    assert store.embedding_fn is not None
    assert len(store.collections) == 5


def test_chroma_store_collections_initialized(chroma_store):
    """Test all collections are initialized."""
    expected_collections = ['conversations', 'summaries', 'wiki_knowledge', 'facts', 'reflections']

    for name in expected_collections:
        assert name in chroma_store.collections
        assert chroma_store.collections[name] is not None


def test_add_conversation_memory(chroma_store):
    """Test adding conversation memory."""
    query = "What is Python?"
    response = "Python is a programming language."
    metadata = {"topic": "programming"}

    doc_id = chroma_store.add_conversation_memory(query, response, metadata)

    assert doc_id is not None
    assert isinstance(doc_id, str)


def test_add_conversation_memory_with_none_metadata(chroma_store):
    """Test adding conversation memory with None metadata."""
    doc_id = chroma_store.add_conversation_memory(
        query="Test",
        response="Response",
        metadata=None
    )

    assert doc_id is not None


def test_add_summary(chroma_store):
    """Test adding summary."""
    summary = "This is a summary of recent conversations."
    period = "last_hour"

    doc_id = chroma_store.add_summary(summary, period)

    assert doc_id is not None
    assert doc_id.startswith("summ_")


def test_add_summary_with_metadata(chroma_store):
    """Test adding summary with custom metadata."""
    summary = "Summary text"
    period = "daily"
    metadata = {"user_id": "user123"}

    doc_id = chroma_store.add_summary(summary, period, metadata)

    assert doc_id is not None


def test_add_wiki_chunk(chroma_store):
    """Test adding wiki chunk."""
    chunk = {
        "title": "Python (programming language)",
        "text": "Python is an interpreted high-level language.",
        "id": "12345",
        "chunk_index": 0
    }

    doc_id = chroma_store.add_wiki_chunk(chunk)

    assert doc_id is not None
    assert doc_id.startswith("wiki_")


def test_add_fact_basic(chroma_store):
    """Test adding basic fact."""
    fact = "Python was created by Guido van Rossum"
    source = "wikipedia"
    confidence = 0.95

    doc_id = chroma_store.add_fact(fact, source, confidence)

    assert doc_id is not None
    assert doc_id.startswith("fact_")


def test_add_fact_with_legacy_dict_source(chroma_store):
    """Test adding fact with legacy dict source parameter."""
    fact = "entity | relation | value"
    source_dict = {
        "source": "manual",
        "confidence": 0.9,
        "extra_field": "extra_value"
    }

    doc_id = chroma_store.add_fact(fact, source_dict)

    assert doc_id is not None


def test_add_fact_with_parent_child(chroma_store):
    """Test adding fact with parent and child IDs."""
    fact = "Test fact"
    source = "test"
    parent_id = "parent_123"
    child_ids = ["child_1", "child_2"]

    doc_id = chroma_store.add_fact(
        fact, source,
        parent_id=parent_id,
        child_ids=child_ids
    )

    assert doc_id is not None


def test_add_reflection(chroma_store):
    """Test adding reflection."""
    reflection = "User prefers concise explanations"
    source_ids = ["conv_123", "conv_456"]
    reflection_type = "preference"

    doc_id = chroma_store.add_reflection(reflection, source_ids, reflection_type)

    assert doc_id is not None
    assert doc_id.startswith("refl_")


def test_query_collection(chroma_store):
    """Test querying a collection."""
    # Add some data first
    chroma_store.add_conversation_memory(
        query="What is Python?",
        response="Python is a language.",
        metadata={"topic": "python"}
    )

    # Query the collection
    results = chroma_store.query_collection(
        collection_name="conversations",
        query_text="Python programming",
        n_results=5
    )

    assert isinstance(results, list)
    if results:
        assert "content" in results[0]
        assert "metadata" in results[0]
        assert "relevance_score" in results[0]


def test_query_collection_with_alias_kwargs(chroma_store):
    """Test query_collection with alias kwargs (n, k)."""
    chroma_store.add_conversation_memory("Q", "A", {})

    # Test with 'n' alias
    results = chroma_store.query_collection(
        collection_name="conversations",
        query_text="test",
        n=3
    )
    assert isinstance(results, list)

    # Test with 'k' alias
    results = chroma_store.query_collection(
        collection_name="conversations",
        query_text="test",
        k=2
    )
    assert isinstance(results, list)


def test_query_collection_invalid_name(chroma_store):
    """Test querying with invalid collection name."""
    with pytest.raises(ValueError, match="Unknown collection"):
        chroma_store.query_collection(
            collection_name="invalid_collection",
            query_text="test",
            n_results=5
        )


def test_search_all(chroma_store):
    """Test searching across all collections."""
    # Add data to multiple collections
    chroma_store.add_conversation_memory("Q1", "A1", {})
    chroma_store.add_summary("Summary", "period", {})

    results = chroma_store.search_all("test query", n_results_per_type=3)

    assert isinstance(results, dict)
    assert "conversations" in results
    assert "summaries" in results


def test_get_collection_stats(chroma_store):
    """Test getting collection statistics."""
    stats = chroma_store.get_collection_stats()

    assert isinstance(stats, dict)
    assert len(stats) == 5

    for name in ['conversations', 'summaries', 'wiki_knowledge', 'facts', 'reflections']:
        assert name in stats
        assert 'count' in stats[name]


def test_add_to_collection(chroma_store):
    """Test generic add_to_collection method."""
    doc_id = chroma_store.add_to_collection(
        name="conversations",
        text="Test content",
        metadata={"key": "value"}
    )

    assert doc_id is not None
    assert isinstance(doc_id, str)


def test_add_to_collection_creates_if_missing(chroma_store):
    """Test add_to_collection creates collection if missing."""
    # Temporarily remove collection
    chroma_store.collections["conversations"] = None

    doc_id = chroma_store.add_to_collection(
        name="conversations",
        text="Test",
        metadata={}
    )

    assert doc_id is not None
    assert chroma_store.collections["conversations"] is not None


def test_list_all_empty_collection(chroma_store):
    """Test list_all with empty collection."""
    items = chroma_store.list_all("conversations")

    assert isinstance(items, list)
    assert len(items) == 0


def test_list_all_with_data(chroma_store):
    """Test list_all returns all documents."""
    chroma_store.add_conversation_memory("Q1", "A1", {})
    chroma_store.add_conversation_memory("Q2", "A2", {})

    items = chroma_store.list_all("conversations")

    assert isinstance(items, list)
    assert len(items) == 2
    for item in items:
        assert "content" in item
        assert "metadata" in item


def test_list_all_invalid_collection(chroma_store):
    """Test list_all with invalid collection."""
    items = chroma_store.list_all("nonexistent")

    assert items == []


def test_get_recent(chroma_store):
    """Test get_recent retrieves most recent items."""
    import time

    # Add items with slight delays
    chroma_store.add_conversation_memory("Q1", "A1", {"timestamp": datetime.now().isoformat()})
    time.sleep(0.01)
    chroma_store.add_conversation_memory("Q2", "A2", {"timestamp": datetime.now().isoformat()})
    time.sleep(0.01)
    chroma_store.add_conversation_memory("Q3", "A3", {"timestamp": datetime.now().isoformat()})

    recent = chroma_store.get_recent("conversations", limit=2)

    assert isinstance(recent, list)
    assert len(recent) <= 2


def test_get_recent_without_timestamps(chroma_store):
    """Test get_recent handles items without timestamps."""
    chroma_store.add_conversation_memory("Q1", "A1", {})
    chroma_store.add_conversation_memory("Q2", "A2", {})

    recent = chroma_store.get_recent("conversations", limit=5)

    # Should not crash
    assert isinstance(recent, list)


def test_create_collection(chroma_store):
    """Test create_collection method."""
    new_coll = chroma_store.create_collection("test_collection")

    assert new_coll is not None
    assert "test_collection" in chroma_store.collections


def test_generate_id(chroma_store):
    """Test _generate_id creates IDs with expected format."""
    import time

    id1 = chroma_store._generate_id("content1", "test")
    time.sleep(0.001)  # Ensure different timestamp
    id2 = chroma_store._generate_id("content2", "test")

    # IDs should have the prefix
    assert "test_" in id1
    assert "test_" in id2
    # Different content should produce different IDs
    assert id1 != id2


def test_collection_embedder_name(chroma_store):
    """Test _collection_embedder_name method."""
    coll = chroma_store.collections["conversations"]

    name = chroma_store._collection_embedder_name(coll)

    assert isinstance(name, str)


def test_multiple_facts_added(chroma_store):
    """Test adding multiple facts."""
    fact1 = chroma_store.add_fact("Fact 1", "source1", 0.9)
    fact2 = chroma_store.add_fact("Fact 2", "source2", 0.8)

    assert fact1 != fact2

    stats = chroma_store.get_collection_stats()
    assert stats["facts"]["count"] == 2


def test_query_returns_formatted_results(chroma_store):
    """Test query results have expected format."""
    chroma_store.add_conversation_memory("Test query", "Test response", {})

    results = chroma_store.query_collection(
        collection_name="conversations",
        query_text="test",
        n_results=5
    )

    if results:
        result = results[0]
        assert "id" in result
        assert "content" in result
        assert "metadata" in result
        assert "relevance_score" in result
        assert "collection" in result
        assert "rank" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
