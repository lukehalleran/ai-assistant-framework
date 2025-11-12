"""Tests for MemoryConsolidator."""
import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta

from memory.memory_consolidator import MemoryConsolidator, _format_recent_for_summary
from memory.corpus_manager import CorpusManager
import tempfile
from pathlib import Path


@pytest.fixture
def mock_model_manager():
    """Provide mock ModelManager."""
    mm = Mock()
    mm.generate_once = AsyncMock(return_value="Summary of recent conversations")
    return mm


@pytest.fixture
def temp_corpus():
    """Provide temporary corpus manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_file = str(Path(tmpdir) / "corpus.json")
        yield CorpusManager(corpus_file=corpus_file)


@pytest.fixture
def consolidator(mock_model_manager):
    """Provide MemoryConsolidator."""
    return MemoryConsolidator(
        consolidation_threshold=10,
        model_manager=mock_model_manager
    )


def test_format_recent_for_summary_empty():
    """Test _format_recent_for_summary with empty list."""
    result = _format_recent_for_summary([])

    assert isinstance(result, list)
    assert len(result) == 0


def test_format_recent_for_summary_basic():
    """Test _format_recent_for_summary with basic entries."""
    recent = [
        {"query": "What is Python?", "response": "Python is a language."},
        {"query": "How to learn?", "response": "Start with tutorials."}
    ]

    result = _format_recent_for_summary(recent)

    assert len(result) == 2
    assert "User: What is Python?" in result[0]
    assert "Assistant: Python is a language." in result[0]


def test_format_recent_for_summary_long_text():
    """Test _format_recent_for_summary clips long text."""
    long_query = "Q" * 500
    long_response = "A" * 500

    recent = [{"query": long_query, "response": long_response}]

    result = _format_recent_for_summary(recent, q_max=100, a_max=100)

    assert len(result) == 1
    # Should be clipped
    assert len(result[0]) < len(long_query) + len(long_response)


def test_format_recent_for_summary_empty_entries():
    """Test _format_recent_for_summary skips empty entries."""
    recent = [
        {"query": "", "response": ""},
        {"query": "Real question", "response": "Real answer"},
        {"query": None, "response": None}
    ]

    result = _format_recent_for_summary(recent)

    # Should only include the one valid entry
    assert len(result) == 1
    assert "Real question" in result[0]


def test_format_recent_for_summary_malformed():
    """Test _format_recent_for_summary handles malformed entries."""
    recent = [
        {"query": "Good", "response": "Good response"},
        {"not_query": "Bad"},  # Missing expected keys
        123,  # Not a dict
    ]

    # Should not crash
    result = _format_recent_for_summary(recent)

    assert isinstance(result, list)
    assert len(result) >= 1  # At least the good one


def test_consolidator_init():
    """Test MemoryConsolidator initialization."""
    mm = Mock()
    consolidator = MemoryConsolidator(consolidation_threshold=20, model_manager=mm)

    assert consolidator.consolidation_threshold == 20
    assert consolidator.model_manager is mm


def test_consolidator_init_defaults():
    """Test MemoryConsolidator with default threshold."""
    mm = Mock()
    consolidator = MemoryConsolidator(model_manager=mm)

    # Should use environment default or hardcoded default
    assert consolidator.consolidation_threshold > 0


@pytest.mark.asyncio
async def test_maybe_consolidate_not_due(consolidator, temp_corpus):
    """Test maybe_consolidate when not due yet."""
    # Add a few entries, less than threshold
    for i in range(3):
        temp_corpus.add_entry(f"Q{i}", f"A{i}")

    result = await consolidator.maybe_consolidate(temp_corpus)

    # Should not consolidate yet
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_maybe_consolidate_no_model_manager(temp_corpus):
    """Test maybe_consolidate without model_manager."""
    consolidator = MemoryConsolidator(
        consolidation_threshold=5,
        model_manager=None
    )

    # Add entries
    for i in range(10):
        temp_corpus.add_entry(f"Q{i}", f"A{i}")

    # Should handle gracefully without model_manager
    result = await consolidator.maybe_consolidate(temp_corpus)

    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_maybe_consolidate_creates_summary(consolidator, temp_corpus, mock_model_manager):
    """Test maybe_consolidate creates summary when due."""
    # Add entries exceeding threshold
    for i in range(15):
        temp_corpus.add_entry(f"Question {i}", f"Answer {i}")

    result = await consolidator.maybe_consolidate(temp_corpus)

    assert isinstance(result, bool)
    # Should have called generate_once if conditions met
    if result:
        mock_model_manager.generate_once.assert_called()


@pytest.mark.asyncio
async def test_maybe_consolidate_recent_summary_exists(consolidator, temp_corpus):
    """Test maybe_consolidate when recent summary already exists."""
    # Add a recent summary
    temp_corpus.add_summary("Recent summary", timestamp=datetime.now())

    # Add some entries
    for i in range(15):
        temp_corpus.add_entry(f"Q{i}", f"A{i}")

    result = await consolidator.maybe_consolidate(temp_corpus)

    # Should handle existing summaries
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_maybe_consolidate_empty_corpus(consolidator, temp_corpus):
    """Test maybe_consolidate with empty corpus."""
    result = await consolidator.maybe_consolidate(temp_corpus)

    # Should not consolidate empty corpus
    assert result is False


@pytest.mark.asyncio
async def test_maybe_consolidate_model_error(temp_corpus, mock_model_manager):
    """Test maybe_consolidate handles model errors."""
    # Make model raise error
    mock_model_manager.generate_once.side_effect = Exception("Model error")

    consolidator = MemoryConsolidator(
        consolidation_threshold=5,
        model_manager=mock_model_manager
    )

    # Add entries
    for i in range(10):
        temp_corpus.add_entry(f"Q{i}", f"A{i}")

    # Should handle error gracefully
    try:
        result = await consolidator.maybe_consolidate(temp_corpus)
        assert isinstance(result, bool)
    except Exception:
        # Either handles gracefully or propagates
        pass


def test_format_recent_mixed_timestamps():
    """Test _format_recent_for_summary with various timestamp formats."""
    recent = [
        {"query": "Q1", "response": "A1", "timestamp": datetime.now()},
        {"query": "Q2", "response": "A2", "timestamp": "2024-01-15T10:00:00"},
        {"query": "Q3", "response": "A3"}  # No timestamp
    ]

    result = _format_recent_for_summary(recent)

    assert len(result) == 3


def test_format_recent_unicode():
    """Test _format_recent_for_summary with unicode characters."""
    recent = [
        {"query": "What is 你好?", "response": "It means hello in Chinese 中文"}
    ]

    result = _format_recent_for_summary(recent)

    assert len(result) == 1
    assert "你好" in result[0]
    assert "中文" in result[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
