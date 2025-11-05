"""
Unit tests for memory/corpus_manager.py

Tests corpus management functionality:
- Initialization and file loading
- Adding entries with metadata
- Retrieving recent memories and summaries
- JSON serialization (clean_for_json)
- File persistence
- Corpus maintenance (clear, prune)
"""

import pytest
import json
import os
from datetime import datetime, timedelta
from memory.corpus_manager import CorpusManager


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_corpus_file(tmp_path):
    """Create a temporary corpus file path"""
    return str(tmp_path / "test_corpus.json")


@pytest.fixture
def corpus_manager(temp_corpus_file):
    """Create a CorpusManager instance with temp file"""
    return CorpusManager(corpus_file=temp_corpus_file)


@pytest.fixture
def corpus_with_data(temp_corpus_file):
    """Create a CorpusManager with some test data"""
    cm = CorpusManager(corpus_file=temp_corpus_file)

    # Add some test entries
    cm.add_entry("What is Python?", "Python is a programming language", tags=["question"])
    cm.add_entry("Tell me more", "Python is used for web development", tags=["followup"])
    cm.add_entry("Thanks", "You're welcome!", tags=["social"])

    return cm


# =============================================================================
# Initialization Tests
# =============================================================================

def test_init_no_existing_file(corpus_manager):
    """CorpusManager initializes with empty corpus when no file exists"""
    assert corpus_manager.corpus == []
    assert len(corpus_manager.corpus) == 0


def test_init_with_existing_file(temp_corpus_file):
    """CorpusManager loads existing corpus from file"""
    # Create a corpus file
    test_data = [
        {"query": "Hello", "response": "Hi", "timestamp": datetime.now().isoformat(), "tags": []}
    ]
    with open(temp_corpus_file, "w") as f:
        json.dump(test_data, f)

    cm = CorpusManager(corpus_file=temp_corpus_file)

    assert len(cm.corpus) == 1
    assert cm.corpus[0]["query"] == "Hello"
    assert cm.corpus[0]["response"] == "Hi"


def test_init_converts_timestamp_strings(temp_corpus_file):
    """CorpusManager converts ISO timestamp strings to datetime objects"""
    test_time = datetime(2024, 1, 1, 12, 0, 0)
    test_data = [
        {"query": "Test", "response": "Response", "timestamp": test_time.isoformat(), "tags": []}
    ]
    with open(temp_corpus_file, "w") as f:
        json.dump(test_data, f)

    cm = CorpusManager(corpus_file=temp_corpus_file)

    assert isinstance(cm.corpus[0]["timestamp"], datetime)


def test_init_handles_corrupt_file(temp_corpus_file):
    """CorpusManager handles corrupted JSON gracefully"""
    with open(temp_corpus_file, "w") as f:
        f.write("{invalid json")

    cm = CorpusManager(corpus_file=temp_corpus_file)

    assert cm.corpus == []


def test_init_handles_invalid_timestamp(temp_corpus_file):
    """CorpusManager handles invalid timestamp strings"""
    test_data = [
        {"query": "Test", "response": "Response", "timestamp": "not a timestamp", "tags": []}
    ]
    with open(temp_corpus_file, "w") as f:
        json.dump(test_data, f)

    cm = CorpusManager(corpus_file=temp_corpus_file)

    # Should load but keep string timestamp
    assert len(cm.corpus) == 1
    assert cm.corpus[0]["timestamp"] == "not a timestamp"


# =============================================================================
# Add Entry Tests
# =============================================================================

def test_add_entry_basic(corpus_manager):
    """add_entry adds a basic conversation entry"""
    corpus_manager.add_entry("Hello", "Hi there")

    assert len(corpus_manager.corpus) == 1
    assert corpus_manager.corpus[0]["query"] == "Hello"
    assert corpus_manager.corpus[0]["response"] == "Hi there"
    assert isinstance(corpus_manager.corpus[0]["timestamp"], datetime)
    assert corpus_manager.corpus[0]["tags"] == []


def test_add_entry_with_tags(corpus_manager):
    """add_entry includes tags"""
    corpus_manager.add_entry("Question", "Answer", tags=["test", "example"])

    assert corpus_manager.corpus[0]["tags"] == ["test", "example"]


def test_add_entry_with_custom_timestamp(corpus_manager):
    """add_entry uses provided timestamp"""
    test_time = datetime(2024, 1, 1, 12, 0, 0)
    corpus_manager.add_entry("Query", "Response", timestamp=test_time)

    assert corpus_manager.corpus[0]["timestamp"] == test_time


def test_add_entry_strips_whitespace(corpus_manager):
    """add_entry strips leading/trailing whitespace"""
    corpus_manager.add_entry("  Query  ", "  Response  ")

    assert corpus_manager.corpus[0]["query"] == "Query"
    assert corpus_manager.corpus[0]["response"] == "Response"


def test_add_entry_with_thread_metadata(corpus_manager):
    """add_entry includes thread metadata when provided"""
    corpus_manager.add_entry(
        "Question",
        "Answer",
        thread_id="thread_123",
        thread_depth=2,
        thread_started="2024-01-01T12:00:00",
        thread_topic="python",
        is_heavy_topic=False,
        topic="programming"
    )

    entry = corpus_manager.corpus[0]
    assert entry["thread_id"] == "thread_123"
    assert entry["thread_depth"] == 2
    assert entry["thread_started"] == "2024-01-01T12:00:00"
    assert entry["thread_topic"] == "python"
    assert entry["is_heavy_topic"] == False
    assert entry["topic"] == "programming"


def test_add_entry_saves_to_file(corpus_manager):
    """add_entry persists to file"""
    corpus_manager.add_entry("Test", "Response")

    assert os.path.exists(corpus_manager.corpus_file)

    # Verify file content
    with open(corpus_manager.corpus_file, "r") as f:
        data = json.load(f)

    assert len(data) == 1
    assert data[0]["query"] == "Test"


def test_add_multiple_entries(corpus_manager):
    """add_entry can be called multiple times"""
    corpus_manager.add_entry("First", "Response 1")
    corpus_manager.add_entry("Second", "Response 2")
    corpus_manager.add_entry("Third", "Response 3")

    assert len(corpus_manager.corpus) == 3
    assert corpus_manager.corpus[0]["query"] == "First"
    assert corpus_manager.corpus[2]["query"] == "Third"


# =============================================================================
# Get Recent Memories Tests
# =============================================================================

def test_get_recent_memories_basic(corpus_with_data):
    """get_recent_memories returns most recent entries"""
    recent = corpus_with_data.get_recent_memories(count=2)

    assert len(recent) == 2
    # Should be in reverse chronological order (most recent first)
    assert recent[0]["query"] == "Thanks"
    assert recent[1]["query"] == "Tell me more"


def test_get_recent_memories_all(corpus_with_data):
    """get_recent_memories can return all entries"""
    recent = corpus_with_data.get_recent_memories(count=10)

    assert len(recent) == 3


def test_get_recent_memories_excludes_summaries(corpus_manager):
    """get_recent_memories excludes summary entries"""
    # Add regular entry
    corpus_manager.add_entry("Question", "Answer")

    # Add summary
    corpus_manager.add_summary("This is a summary")

    recent = corpus_manager.get_recent_memories(count=10)

    # Should only return the regular entry, not the summary
    assert len(recent) == 1
    assert recent[0]["query"] == "Question"


def test_get_recent_memories_empty_corpus(corpus_manager):
    """get_recent_memories returns empty list when corpus is empty"""
    recent = corpus_manager.get_recent_memories(count=5)

    assert recent == []


# =============================================================================
# Add Summary Tests
# =============================================================================

def test_add_summary_string(corpus_manager):
    """add_summary accepts string content"""
    corpus_manager.add_summary("This is a summary of recent conversations")

    assert len(corpus_manager.corpus) == 1
    assert corpus_manager.corpus[0]["content"] == "This is a summary of recent conversations"
    assert corpus_manager.corpus[0]["type"] == "summary"
    assert "@summary" in corpus_manager.corpus[0]["tags"]


def test_add_summary_with_tags(corpus_manager):
    """add_summary accepts tags parameter"""
    corpus_manager.add_summary("Summary content", tags=["manual", "test"])

    assert len(corpus_manager.corpus) == 1
    assert corpus_manager.corpus[0]["content"] == "Summary content"
    assert "manual" in corpus_manager.corpus[0]["tags"]
    assert "test" in corpus_manager.corpus[0]["tags"]
    assert "@summary" in corpus_manager.corpus[0]["tags"]


def test_add_summary_with_timestamp(corpus_manager):
    """add_summary uses provided timestamp"""
    test_time = datetime(2024, 1, 1, 12, 0, 0)
    corpus_manager.add_summary("Summary", timestamp=test_time)

    assert corpus_manager.corpus[0]["timestamp"] == test_time


def test_add_summary_saves_to_file(corpus_manager):
    """add_summary persists to file"""
    corpus_manager.add_summary("Test summary")

    assert os.path.exists(corpus_manager.corpus_file)


# =============================================================================
# Get Summaries Tests
# =============================================================================

def test_get_summaries_basic(corpus_manager):
    """get_summaries returns summary entries"""
    # Add regular entry
    corpus_manager.add_entry("Question", "Answer")

    # Add summaries
    corpus_manager.add_summary("Summary 1")
    corpus_manager.add_summary("Summary 2")

    summaries = corpus_manager.get_summaries(count=10)

    assert len(summaries) == 2


def test_get_summaries_most_recent_first(corpus_manager):
    """get_summaries returns most recent first"""
    import time

    corpus_manager.add_summary("Old summary")
    time.sleep(0.01)  # Ensure different timestamps
    corpus_manager.add_summary("New summary")

    summaries = corpus_manager.get_summaries(count=2)

    assert summaries[0]["content"] == "New summary"
    assert summaries[1]["content"] == "Old summary"


def test_get_summaries_limited_count(corpus_manager):
    """get_summaries respects count limit"""
    corpus_manager.add_summary("Summary 1")
    corpus_manager.add_summary("Summary 2")
    corpus_manager.add_summary("Summary 3")

    summaries = corpus_manager.get_summaries(count=2)

    assert len(summaries) == 2


def test_get_summaries_empty_corpus(corpus_manager):
    """get_summaries returns empty list when no summaries"""
    corpus_manager.add_entry("Question", "Answer")

    summaries = corpus_manager.get_summaries(count=5)

    assert summaries == []


# =============================================================================
# Clean for JSON Tests
# =============================================================================

def test_clean_for_json_datetime(corpus_manager):
    """clean_for_json converts datetime to ISO string"""
    test_time = datetime(2024, 1, 1, 12, 0, 0)

    result = corpus_manager.clean_for_json(test_time)

    assert result == "2024-01-01T12:00:00"


def test_clean_for_json_dict_with_datetime(corpus_manager):
    """clean_for_json handles dict with datetime values"""
    data = {
        "name": "Test",
        "timestamp": datetime(2024, 1, 1, 12, 0, 0)
    }

    result = corpus_manager.clean_for_json(data)

    assert result["name"] == "Test"
    assert result["timestamp"] == "2024-01-01T12:00:00"


def test_clean_for_json_nested_dict(corpus_manager):
    """clean_for_json handles nested dicts"""
    data = {
        "outer": {
            "inner": {
                "timestamp": datetime(2024, 1, 1, 12, 0, 0)
            }
        }
    }

    result = corpus_manager.clean_for_json(data)

    assert result["outer"]["inner"]["timestamp"] == "2024-01-01T12:00:00"


def test_clean_for_json_list(corpus_manager):
    """clean_for_json handles lists"""
    data = [
        datetime(2024, 1, 1, 12, 0, 0),
        datetime(2024, 1, 2, 12, 0, 0)
    ]

    result = corpus_manager.clean_for_json(data)

    assert len(result) == 2
    assert result[0] == "2024-01-01T12:00:00"
    assert result[1] == "2024-01-02T12:00:00"


def test_clean_for_json_primitives(corpus_manager):
    """clean_for_json passes through primitives unchanged"""
    assert corpus_manager.clean_for_json("string") == "string"
    assert corpus_manager.clean_for_json(42) == 42
    assert corpus_manager.clean_for_json(3.14) == 3.14
    assert corpus_manager.clean_for_json(True) == True
    assert corpus_manager.clean_for_json(None) == None


# =============================================================================
# Clear Corpus Tests
# =============================================================================

def test_clear_corpus_all(corpus_with_data):
    """clear_corpus removes all entries when keep_summaries=False"""
    corpus_with_data.clear_corpus(keep_summaries=False)

    assert len(corpus_with_data.corpus) == 0


def test_clear_corpus_keep_summaries(corpus_manager):
    """clear_corpus keeps summaries when keep_summaries=True"""
    # Add regular entries
    corpus_manager.add_entry("Q1", "A1")
    corpus_manager.add_entry("Q2", "A2")

    # Add summary
    corpus_manager.add_summary("Summary")

    corpus_manager.clear_corpus(keep_summaries=True)

    # Should only have summary left
    assert len(corpus_manager.corpus) == 1
    assert corpus_manager.corpus[0]["type"] == "summary"


# =============================================================================
# Prune Corpus Tests
# =============================================================================

def test_prune_corpus_basic(corpus_manager):
    """prune_corpus keeps only N most recent entries"""
    # Add 5 entries
    for i in range(5):
        corpus_manager.add_entry(f"Q{i}", f"A{i}")

    corpus_manager.prune_corpus(keep=2, preserve_summaries=False)

    assert len(corpus_manager.corpus) == 2


def test_prune_corpus_preserve_summaries(corpus_manager):
    """prune_corpus preserves summaries when requested"""
    # Add entries
    corpus_manager.add_entry("Q1", "A1")
    corpus_manager.add_entry("Q2", "A2")
    corpus_manager.add_entry("Q3", "A3")

    # Add summary
    corpus_manager.add_summary("Summary")

    # Prune to 1 entry but preserve summaries
    corpus_manager.prune_corpus(keep=1, preserve_summaries=True)

    # Should have 1 entry + 1 summary = 2 total
    assert len(corpus_manager.corpus) == 2


# =============================================================================
# File Persistence Tests
# =============================================================================

def test_save_corpus_creates_file(corpus_manager):
    """save_corpus creates file"""
    corpus_manager.add_entry("Test", "Response")

    assert os.path.exists(corpus_manager.corpus_file)


def test_persistence_roundtrip(temp_corpus_file):
    """CorpusManager can save and reload data"""
    # Create and add data
    cm1 = CorpusManager(corpus_file=temp_corpus_file)
    cm1.add_entry("Question", "Answer", tags=["test"])

    # Create new instance (simulates restart)
    cm2 = CorpusManager(corpus_file=temp_corpus_file)

    assert len(cm2.corpus) == 1
    assert cm2.corpus[0]["query"] == "Question"
    assert cm2.corpus[0]["response"] == "Answer"
    assert "test" in cm2.corpus[0]["tags"]


def test_save_uses_temp_file(corpus_manager):
    """save_corpus uses temp file for atomic write"""
    corpus_manager.add_entry("Test", "Response")

    # Temp file should be cleaned up after save
    tmp_file = corpus_manager.corpus_file + ".tmp"
    assert not os.path.exists(tmp_file)


# =============================================================================
# Max Entries Tests
# =============================================================================

def test_add_entry_respects_max_entries(temp_corpus_file, monkeypatch):
    """add_entry trims corpus to max_entries"""
    # Set max to 3 via env
    monkeypatch.setenv("CORPUS_MAX_ENTRIES", "3")

    cm = CorpusManager(corpus_file=temp_corpus_file)

    # Add 5 entries
    for i in range(5):
        cm.add_entry(f"Q{i}", f"A{i}")

    # Should only keep last 3
    assert len(cm.corpus) == 3
    assert cm.corpus[0]["query"] == "Q2"  # Entries 2, 3, 4
    assert cm.corpus[2]["query"] == "Q4"
