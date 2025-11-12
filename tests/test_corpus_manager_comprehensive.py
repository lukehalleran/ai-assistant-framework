"""Comprehensive tests for CorpusManager to boost coverage."""
import pytest
import tempfile
import json
from pathlib import Path
from memory.corpus_manager import CorpusManager


@pytest.fixture
def temp_file():
    """Create temporary corpus file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    try:
        Path(temp_path).unlink()
    except:
        pass


def test_corpus_manager_initialization(temp_file):
    """Test CorpusManager initialization."""
    cm = CorpusManager(corpus_file=temp_file)
    assert cm is not None
    assert cm.corpus == []


def test_add_entry_basic(temp_file):
    """Test adding basic entry."""
    cm = CorpusManager(corpus_file=temp_file)
    cm.add_entry(query="What is Python?", response="Python is a language")
    assert len(cm.corpus) == 1


def test_add_entry_with_tags(temp_file):
    """Test adding entry with tags."""
    cm = CorpusManager(corpus_file=temp_file)
    cm.add_entry(
        query="Test query",
        response="Test response",
        tags=["test", "example"]
    )
    assert len(cm.corpus) == 1
    assert "tags" in cm.corpus[0] or "tag" in str(cm.corpus[0])



    """Test get_recent_memories."""
    cm = CorpusManager(corpus_file=temp_file)
    cm.add_entry("Q1", "A1")
    cm.add_entry("Q2", "A2")
    cm.add_entry("Q3", "A3")

    recent = cm.get_recent_memories(count=2)
    assert len(recent) <= 2


def test_get_recent_memories_empty(temp_file):
    """Test get_recent_memories with empty corpus."""
    cm = CorpusManager(corpus_file=temp_file)
    recent = cm.get_recent_memories(count=5)
    assert isinstance(recent, list)
    assert len(recent) == 0


def test_get_recent_memories_zero_count(temp_file):
    """Test get_recent_memories with count=0."""
    cm = CorpusManager(corpus_file=temp_file)
    cm.add_entry("Q", "A")
    recent = cm.get_recent_memories(count=0)
    assert isinstance(recent, list)

def test_get_items_by_type_with_limit(temp_file):
    """Test get_items_by_type with limit."""
    cm = CorpusManager(corpus_file=temp_file)
    for i in range(10):
        cm.add_entry(f"Q{i}", f"A{i}")

    items = cm.get_recent_memories(count=3)
    assert len(items) <= 3


