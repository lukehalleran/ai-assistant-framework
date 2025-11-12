"""Tests for CorpusManager."""
import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

from memory.corpus_manager import CorpusManager


@pytest.fixture
def temp_corpus_file():
    """Provide temporary corpus file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "corpus.json")


@pytest.fixture
def corpus_manager(temp_corpus_file):
    """Provide CorpusManager."""
    return CorpusManager(corpus_file=temp_corpus_file)


def test_corpus_manager_init(temp_corpus_file):
    """Test CorpusManager initialization."""
    cm = CorpusManager(corpus_file=temp_corpus_file)

    assert cm.corpus_file == temp_corpus_file
    assert isinstance(cm.corpus, list)


def test_add_entry(corpus_manager):
    """Test adding an entry to corpus."""
    corpus_manager.add_entry(
        query="What is Python?",
        response="Python is a language.",
        tags=["programming"]
    )

    assert len(corpus_manager.corpus) == 1
    assert corpus_manager.corpus[0]["query"] == "What is Python?"


def test_add_entry_with_timestamp(corpus_manager):
    """Test adding entry with custom timestamp."""
    ts = datetime(2024, 1, 15, 10, 0, 0)

    corpus_manager.add_entry(
        query="Test",
        response="Response",
        timestamp=ts
    )

    assert len(corpus_manager.corpus) == 1
    entry = corpus_manager.corpus[0]
    assert "timestamp" in entry


def test_get_recent_memories_empty(corpus_manager):
    """Test get_recent_memories with empty corpus."""
    recent = corpus_manager.get_recent_memories(count=5)

    assert isinstance(recent, list)
    assert len(recent) == 0


def test_get_recent_memories_with_data(corpus_manager):
    """Test get_recent_memories returns recent entries."""
    corpus_manager.add_entry("Q1", "A1")
    corpus_manager.add_entry("Q2", "A2")
    corpus_manager.add_entry("Q3", "A3")

    recent = corpus_manager.get_recent_memories(count=2)

    assert len(recent) == 2
    # Most recent first
    assert recent[0]["query"] == "Q3"
    assert recent[1]["query"] == "Q2"


def test_get_recent_memories_filters_summaries(corpus_manager):
    """Test get_recent_memories excludes summaries."""
    corpus_manager.add_entry("Q1", "A1")
    corpus_manager.add_summary("Summary text")
    corpus_manager.add_entry("Q2", "A2")

    recent = corpus_manager.get_recent_memories(count=10)

    # Should only return 2 non-summary entries
    assert len(recent) == 2
    assert all("@summary" not in e.get("tags", []) for e in recent)


def test_add_summary(corpus_manager):
    """Test adding a summary."""
    corpus_manager.add_summary(
        content="This is a summary",
        tags=["test"]
    )

    assert len(corpus_manager.corpus) == 1
    entry = corpus_manager.corpus[0]
    assert entry["type"] == "summary"
    assert "@summary" in entry["tags"]


def test_add_summary_with_timestamp(corpus_manager):
    """Test adding summary with custom timestamp."""
    ts = datetime(2024, 1, 15, 12, 0, 0)

    corpus_manager.add_summary("Summary", timestamp=ts)

    assert len(corpus_manager.corpus) == 1


def test_get_summaries_empty(corpus_manager):
    """Test get_summaries with no summaries."""
    summaries = corpus_manager.get_summaries(count=5)

    assert isinstance(summaries, list)
    assert len(summaries) == 0


def test_get_summaries_returns_summaries(corpus_manager):
    """Test get_summaries returns summary entries."""
    corpus_manager.add_entry("Q1", "A1")
    corpus_manager.add_summary("Summary 1")
    corpus_manager.add_summary("Summary 2")
    corpus_manager.add_entry("Q2", "A2")

    summaries = corpus_manager.get_summaries(count=10)

    assert len(summaries) == 2
    assert all(e["type"] == "summary" for e in summaries)


def test_save_and_load_corpus(temp_corpus_file):
    """Test saving and loading corpus from disk."""
    cm1 = CorpusManager(corpus_file=temp_corpus_file)
    cm1.add_entry("Question", "Answer")
    cm1.save_corpus()

    # Create new manager and load
    cm2 = CorpusManager(corpus_file=temp_corpus_file)

    assert len(cm2.corpus) == 1
    assert cm2.corpus[0]["query"] == "Question"


def test_clear_corpus_keep_summaries(corpus_manager):
    """Test clear_corpus keeps summaries."""
    corpus_manager.add_entry("Q1", "A1")
    corpus_manager.add_summary("Summary 1")
    corpus_manager.add_entry("Q2", "A2")

    corpus_manager.clear_corpus(keep_summaries=True)

    assert len(corpus_manager.corpus) == 1
    assert corpus_manager.corpus[0]["type"] == "summary"


def test_clear_corpus_delete_all(corpus_manager):
    """Test clear_corpus removes everything."""
    corpus_manager.add_entry("Q1", "A1")
    corpus_manager.add_summary("Summary 1")
    corpus_manager.add_entry("Q2", "A2")

    corpus_manager.clear_corpus(keep_summaries=False)

    assert len(corpus_manager.corpus) == 0


def test_prune_corpus(corpus_manager):
    """Test pruning corpus to size limit."""
    for i in range(10):
        corpus_manager.add_entry(f"Q{i}", f"A{i}")

    corpus_manager.prune_corpus(keep=5, preserve_summaries=False)

    assert len(corpus_manager.corpus) <= 5


def test_prune_corpus_preserve_summaries(corpus_manager):
    """Test pruning preserves summaries."""
    for i in range(5):
        corpus_manager.add_entry(f"Q{i}", f"A{i}")
    corpus_manager.add_summary("Important summary")

    corpus_manager.prune_corpus(keep=3, preserve_summaries=True)

    # Should have 3 entries + 1 summary
    assert len(corpus_manager.corpus) <= 4
    assert any(e.get("type") == "summary" for e in corpus_manager.corpus)


def test_get_summaries_of_type(corpus_manager):
    """Test get_summaries_of_type."""
    corpus_manager.add_summary("Summary 1", tags=["type:summary"])
    corpus_manager.add_entry("Q1", "A1")

    summaries = corpus_manager.get_summaries_of_type(types=("summary",), limit=5)

    assert isinstance(summaries, list)


def test_get_items_by_type(corpus_manager):
    """Test get_items_by_type."""
    corpus_manager.add_summary("Summary 1")
    corpus_manager.add_entry("Q1", "A1")

    items = corpus_manager.get_items_by_type("summary", limit=5)

    assert isinstance(items, list)


def test_clean_for_json(corpus_manager):
    """Test clean_for_json cleans datetime objects."""
    entry = {
        "query": "Test",
        "timestamp": datetime.now()
    }

    cleaned = corpus_manager.clean_for_json(entry)

    assert isinstance(cleaned["timestamp"], str)


def test_multiple_entries_ordering(corpus_manager):
    """Test entries maintain chronological order."""
    corpus_manager.add_entry("First", "R1")
    corpus_manager.add_entry("Second", "R2")
    corpus_manager.add_entry("Third", "R3")

    recent = corpus_manager.get_recent_memories(count=10)

    # Most recent should be first
    assert recent[0]["query"] == "Third"
    assert recent[1]["query"] == "Second"
    assert recent[2]["query"] == "First"


def test_add_entry_with_empty_strings(corpus_manager):
    """Test adding entry with empty strings."""
    corpus_manager.add_entry("", "")

    assert len(corpus_manager.corpus) == 1


def test_add_entry_with_none_response(corpus_manager):
    """Test adding entry with None response."""
    corpus_manager.add_entry("Question", None)

    assert len(corpus_manager.corpus) == 1


def test_add_summary_empty_content(corpus_manager):
    """Test adding summary with empty content."""
    corpus_manager.add_summary("")

    assert len(corpus_manager.corpus) == 1


def test_corpus_persistence(temp_corpus_file):
    """Test corpus persists across multiple saves."""
    cm = CorpusManager(corpus_file=temp_corpus_file)
    cm.add_entry("Q1", "A1")
    cm.save_corpus()

    cm.add_entry("Q2", "A2")
    cm.save_corpus()

    # Reload
    cm2 = CorpusManager(corpus_file=temp_corpus_file)
    assert len(cm2.corpus) == 2


def test_load_nonexistent_file(temp_corpus_file):
    """Test loading from nonexistent file creates empty corpus."""
    # Don't create the file
    cm = CorpusManager(corpus_file=temp_corpus_file)

    assert isinstance(cm.corpus, list)
    assert len(cm.corpus) == 0


def test_save_requires_existing_directory(tmpdir):
    """Test save_corpus requires parent directory to exist."""
    corpus_file = str(Path(tmpdir) / "subdir" / "corpus.json")

    cm = CorpusManager(corpus_file=corpus_file)
    cm.add_entry("Test", "Response")

    # save_corpus will fail if directory doesn't exist
    cm.save_corpus()

    # File won't be created because directory doesn't exist
    assert not Path(corpus_file).exists()

    # But if we create the directory first, it works
    Path(tmpdir / "subdir").mkdir()
    cm.save_corpus()
    assert Path(corpus_file).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
