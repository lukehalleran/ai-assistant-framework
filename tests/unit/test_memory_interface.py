"""
Unit tests for memory/memory_interface.py

Tests:
- MemoryType enum
- MemoryNode dataclass
- Serialization (to_dict/from_dict)
"""

import pytest
import json
from datetime import datetime
from memory.memory_interface import MemoryType, MemoryNode


# =============================================================================
# MemoryType Enum Tests
# =============================================================================

def test_memory_type_values():
    """All memory types have correct string values"""
    assert MemoryType.EPISODIC.value == "episodic"
    assert MemoryType.SEMANTIC.value == "semantic"
    assert MemoryType.PROCEDURAL.value == "procedural"
    assert MemoryType.SUMMARY.value == "summary"
    assert MemoryType.META.value == "meta"
    assert MemoryType.FACT.value == "fact"


def test_memory_type_count():
    """Verify expected number of memory types"""
    assert len(MemoryType) == 6


def test_memory_type_from_string():
    """Can construct MemoryType from string value"""
    assert MemoryType("episodic") == MemoryType.EPISODIC
    assert MemoryType("semantic") == MemoryType.SEMANTIC
    assert MemoryType("fact") == MemoryType.FACT


def test_memory_type_iteration():
    """Can iterate over all memory types"""
    types = list(MemoryType)
    assert len(types) == 6
    assert MemoryType.EPISODIC in types
    assert MemoryType.FACT in types


# =============================================================================
# MemoryNode Creation Tests
# =============================================================================

def test_memory_node_basic_creation():
    """Create memory node with required fields"""
    node = MemoryNode(
        id="test_id",
        content="Test content",
        type=MemoryType.SEMANTIC,
        timestamp=datetime(2024, 1, 1, 12, 0, 0)
    )

    assert node.id == "test_id"
    assert node.content == "Test content"
    assert node.type == MemoryType.SEMANTIC
    assert node.timestamp == datetime(2024, 1, 1, 12, 0, 0)


def test_memory_node_default_values():
    """Default values are set correctly"""
    node = MemoryNode(
        id="id",
        content="content",
        type=MemoryType.EPISODIC,
        timestamp=datetime.now()
    )

    assert node.access_count == 0
    assert node.importance_score == 0.5
    assert node.decay_rate == 0.1
    assert node.truth_score == 0.5
    assert node.parent_id is None
    assert node.child_ids == []
    assert node.tags == []
    assert node.embeddings is None
    assert node.metadata == {}


def test_memory_node_with_optional_fields():
    """Create node with all optional fields"""
    node = MemoryNode(
        id="id",
        content="content",
        type=MemoryType.META,
        timestamp=datetime.now(),
        access_count=5,
        importance_score=0.9,
        decay_rate=0.05,
        truth_score=0.8,
        parent_id="parent_123",
        child_ids=["child1", "child2"],
        tags=["tag1", "tag2"],
        embeddings=[0.1, 0.2, 0.3],
        metadata={"key": "value"}
    )

    assert node.access_count == 5
    assert node.importance_score == 0.9
    assert node.decay_rate == 0.05
    assert node.truth_score == 0.8
    assert node.parent_id == "parent_123"
    assert node.child_ids == ["child1", "child2"]
    assert node.tags == ["tag1", "tag2"]
    assert node.embeddings == [0.1, 0.2, 0.3]
    assert node.metadata == {"key": "value"}


def test_memory_node_last_accessed_default():
    """last_accessed defaults to current time"""
    node = MemoryNode(
        id="id",
        content="content",
        type=MemoryType.EPISODIC,
        timestamp=datetime.now()
    )

    # Should be set to a recent time
    assert isinstance(node.last_accessed, datetime)
    assert (datetime.now() - node.last_accessed).total_seconds() < 1


# =============================================================================
# MemoryNode to_dict Tests
# =============================================================================

def test_to_dict_basic():
    """Serialize node to dict with basic fields"""
    node = MemoryNode(
        id="test_id",
        content="Test content",
        type=MemoryType.SEMANTIC,
        timestamp=datetime(2024, 1, 1, 12, 0, 0)
    )

    result = node.to_dict()

    assert result["id"] == "test_id"
    assert result["content"] == "Test content"
    assert result["type"] == "semantic"
    assert result["timestamp"] == "2024-01-01T12:00:00"
    assert result["importance_score"] == 0.5
    assert result["decay_rate"] == 0.1
    assert result["truth_score"] == 0.5


def test_to_dict_with_metadata():
    """Serialize node with metadata"""
    node = MemoryNode(
        id="id",
        content="content",
        type=MemoryType.FACT,
        timestamp=datetime(2024, 1, 1),
        metadata={"key1": "value1", "key2": 42}
    )

    result = node.to_dict()

    assert "metadata" in result
    assert result["metadata"]["key1"] == "value1"
    assert result["metadata"]["key2"] == 42


def test_to_dict_metadata_serializes_lists():
    """Lists in metadata are JSON-serialized"""
    node = MemoryNode(
        id="id",
        content="content",
        type=MemoryType.EPISODIC,
        timestamp=datetime.now(),
        metadata={"list_field": ["a", "b", "c"]}
    )

    result = node.to_dict()

    # Should be JSON string
    assert isinstance(result["metadata"]["list_field"], str)
    parsed = json.loads(result["metadata"]["list_field"])
    assert parsed == ["a", "b", "c"]


def test_to_dict_metadata_serializes_dicts():
    """Dicts in metadata are JSON-serialized"""
    node = MemoryNode(
        id="id",
        content="content",
        type=MemoryType.PROCEDURAL,
        timestamp=datetime.now(),
        metadata={"nested": {"inner": "value"}}
    )

    result = node.to_dict()

    # Should be JSON string
    assert isinstance(result["metadata"]["nested"], str)
    parsed = json.loads(result["metadata"]["nested"])
    assert parsed == {"inner": "value"}


def test_to_dict_preserves_simple_values():
    """Simple metadata values (str, int, float) aren't serialized"""
    node = MemoryNode(
        id="id",
        content="content",
        type=MemoryType.SEMANTIC,
        timestamp=datetime.now(),
        metadata={"string": "value", "number": 42, "float": 3.14}
    )

    result = node.to_dict()

    assert result["metadata"]["string"] == "value"
    assert result["metadata"]["number"] == 42
    assert result["metadata"]["float"] == 3.14


def test_to_dict_empty_metadata():
    """Empty metadata dict is preserved"""
    node = MemoryNode(
        id="id",
        content="content",
        type=MemoryType.META,
        timestamp=datetime.now(),
        metadata={}
    )

    result = node.to_dict()

    assert result["metadata"] == {}


# =============================================================================
# MemoryNode from_dict Tests
# =============================================================================

def test_from_dict_basic():
    """Deserialize node from dict"""
    data = {
        "id": "test_id",
        "content": "Test content",
        "timestamp": "2024-01-01T12:00:00",
        "importance_score": 0.7,
        "decay_rate": 0.05,
        "truth_score": 0.9,
        "metadata": {"type": "semantic"}
    }

    node = MemoryNode.from_dict(data)

    assert node.id == "test_id"
    assert node.content == "Test content"
    assert node.type == MemoryType.SEMANTIC
    assert node.timestamp == datetime(2024, 1, 1, 12, 0, 0)
    assert node.importance_score == 0.7
    assert node.decay_rate == 0.05
    assert node.truth_score == 0.9


def test_from_dict_default_values():
    """Missing fields use defaults"""
    data = {
        "id": "id",
        "content": "content",
        "timestamp": "2024-01-01T00:00:00",
        "metadata": {"type": "episodic"}
    }

    node = MemoryNode.from_dict(data)

    assert node.importance_score == 0.5
    assert node.decay_rate == 0.01
    assert node.truth_score == 0.5


def test_from_dict_with_tags_list():
    """Parse tags from metadata as list"""
    data = {
        "id": "id",
        "content": "content",
        "timestamp": "2024-01-01T00:00:00",
        "metadata": {"type": "semantic", "tags": ["tag1", "tag2"]}
    }

    node = MemoryNode.from_dict(data)

    assert node.tags == ["tag1", "tag2"]


def test_from_dict_with_tags_json_string():
    """Parse tags from JSON string in metadata"""
    data = {
        "id": "id",
        "content": "content",
        "timestamp": "2024-01-01T00:00:00",
        "metadata": {"type": "fact", "tags": '["tag1", "tag2"]'}
    }

    node = MemoryNode.from_dict(data)

    assert node.tags == ["tag1", "tag2"]


def test_from_dict_tags_invalid_json():
    """Invalid tag JSON falls back to empty list"""
    data = {
        "id": "id",
        "content": "content",
        "timestamp": "2024-01-01T00:00:00",
        "metadata": {"type": "semantic", "tags": "not valid json"}
    }

    node = MemoryNode.from_dict(data)

    assert node.tags == []


def test_from_dict_no_tags():
    """Missing tags field defaults to empty list"""
    data = {
        "id": "id",
        "content": "content",
        "timestamp": "2024-01-01T00:00:00",
        "metadata": {"type": "semantic"}
    }

    node = MemoryNode.from_dict(data)

    assert node.tags == []


def test_from_dict_missing_type():
    """Missing type defaults to semantic"""
    data = {
        "id": "id",
        "content": "content",
        "timestamp": "2024-01-01T00:00:00",
        "metadata": {}
    }

    node = MemoryNode.from_dict(data)

    assert node.type == MemoryType.SEMANTIC


def test_from_dict_all_memory_types():
    """Can deserialize all memory types"""
    for mem_type in MemoryType:
        data = {
            "id": "id",
            "content": "content",
            "timestamp": "2024-01-01T00:00:00",
            "metadata": {"type": mem_type.value}
        }

        node = MemoryNode.from_dict(data)
        assert node.type == mem_type


# =============================================================================
# Round-trip Serialization Tests
# =============================================================================

def test_roundtrip_serialization():
    """to_dict -> from_dict preserves data"""
    original = MemoryNode(
        id="roundtrip_id",
        content="Roundtrip content",
        type=MemoryType.PROCEDURAL,
        timestamp=datetime(2024, 6, 15, 14, 30, 0),
        importance_score=0.75,
        decay_rate=0.03,
        truth_score=0.85,
        metadata={"key": "value", "number": 123}
    )

    # Serialize
    data = original.to_dict()

    # Deserialize
    restored = MemoryNode.from_dict(data)

    assert restored.id == original.id
    assert restored.content == original.content
    assert restored.type == original.type
    assert restored.timestamp == original.timestamp
    assert restored.importance_score == original.importance_score
    assert restored.decay_rate == original.decay_rate
    assert restored.truth_score == original.truth_score


def test_roundtrip_with_complex_metadata():
    """Round-trip with nested metadata structures"""
    original = MemoryNode(
        id="id",
        content="content",
        type=MemoryType.META,
        timestamp=datetime(2024, 1, 1),
        metadata={
            "simple": "value",
            "list": ["a", "b", "c"],
            "dict": {"nested": "data"},
            "tags": ["tag1", "tag2"]
        }
    )

    data = original.to_dict()
    restored = MemoryNode.from_dict(data)

    # Simple values preserved
    assert restored.metadata.get("simple") == "value"

    # Lists/dicts are JSON strings in dict form
    list_data = json.loads(data["metadata"]["list"])
    assert list_data == ["a", "b", "c"]

    dict_data = json.loads(data["metadata"]["dict"])
    assert dict_data == {"nested": "data"}

    # Tags are parsed correctly
    assert restored.tags == ["tag1", "tag2"]


def test_roundtrip_all_fields():
    """Round-trip with all optional fields populated"""
    original = MemoryNode(
        id="full_id",
        content="Full content",
        type=MemoryType.FACT,
        timestamp=datetime(2024, 3, 10, 8, 15, 30),
        access_count=10,
        importance_score=0.95,
        decay_rate=0.02,
        truth_score=0.88,
        parent_id="parent",
        child_ids=["c1", "c2"],
        tags=["important", "verified"],
        embeddings=[0.1, 0.2, 0.3, 0.4],
        metadata={"source": "test", "count": 5}
    )

    data = original.to_dict()
    # Note: to_dict doesn't include all fields, only the core ones
    # This is expected behavior based on the implementation

    restored = MemoryNode.from_dict(data)

    # Core fields are preserved
    assert restored.id == original.id
    assert restored.content == original.content
    assert restored.type == original.type
    assert restored.timestamp == original.timestamp
    assert restored.importance_score == original.importance_score
    assert restored.decay_rate == original.decay_rate
    assert restored.truth_score == original.truth_score


# =============================================================================
# Edge Cases
# =============================================================================

def test_memory_node_empty_content():
    """Node can have empty content string"""
    node = MemoryNode(
        id="id",
        content="",
        type=MemoryType.EPISODIC,
        timestamp=datetime.now()
    )

    assert node.content == ""


def test_memory_node_very_long_content():
    """Node handles very long content"""
    long_content = "x" * 10000
    node = MemoryNode(
        id="id",
        content=long_content,
        type=MemoryType.SEMANTIC,
        timestamp=datetime.now()
    )

    assert len(node.content) == 10000
    data = node.to_dict()
    assert len(data["content"]) == 10000


def test_timestamp_microseconds():
    """Timestamp with microseconds is preserved"""
    ts = datetime(2024, 1, 1, 12, 30, 45, 123456)
    node = MemoryNode(
        id="id",
        content="content",
        type=MemoryType.EPISODIC,
        timestamp=ts
    )

    data = node.to_dict()
    restored = MemoryNode.from_dict(data)

    assert restored.timestamp == ts


def test_extreme_scores():
    """Handles extreme score values"""
    node = MemoryNode(
        id="id",
        content="content",
        type=MemoryType.SEMANTIC,
        timestamp=datetime.now(),
        importance_score=0.0,
        decay_rate=1.0,
        truth_score=1.0
    )

    data = node.to_dict()
    restored = MemoryNode.from_dict(data)

    assert restored.importance_score == 0.0
    assert restored.decay_rate == 1.0
    assert restored.truth_score == 1.0
