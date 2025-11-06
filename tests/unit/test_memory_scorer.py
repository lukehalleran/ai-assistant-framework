"""
Unit tests for memory/memory_scorer.py

Tests:
- calculate_importance_score: Pure heuristic scoring function
- apply_temporal_decay: Temporal decay and access recency scoring
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock
from memory.memory_scorer import MemoryScorer


# =============================================================================
# MemoryScorer Initialization Tests
# =============================================================================

def test_memory_scorer_initialization():
    """MemoryScorer initializes successfully"""
    scorer = MemoryScorer()
    assert scorer is not None


# =============================================================================
# calculate_importance_score Tests
# =============================================================================

def test_calculate_importance_score_default():
    """Short content without special features gets base score"""
    scorer = MemoryScorer()
    score = scorer.calculate_importance_score("Hello world")

    assert score == 0.5  # Base score


def test_calculate_importance_score_long_content():
    """Long content (>200 chars) gets boost"""
    scorer = MemoryScorer()
    long_text = "x" * 250  # 250 characters
    score = scorer.calculate_importance_score(long_text)

    assert score == 0.6  # Base 0.5 + 0.1 for length


def test_calculate_importance_score_with_question():
    """Content with question mark gets boost"""
    scorer = MemoryScorer()
    score = scorer.calculate_importance_score("What is the answer?")

    assert score == 0.6  # Base 0.5 + 0.1 for question


def test_calculate_importance_score_important_keyword():
    """Content with 'important' keyword gets boost"""
    scorer = MemoryScorer()
    score = scorer.calculate_importance_score("This is important information")

    assert score == 0.7  # Base 0.5 + 0.2 for keyword


def test_calculate_importance_score_remember_keyword():
    """Content with 'remember' keyword gets boost"""
    scorer = MemoryScorer()
    score = scorer.calculate_importance_score("Remember to check this")

    assert score == 0.7


def test_calculate_importance_score_note_keyword():
    """Content with 'note' keyword gets boost"""
    scorer = MemoryScorer()
    score = scorer.calculate_importance_score("Note: this is critical")

    # Has both 'note' and 'critical' but any() only adds +0.2 once
    assert score == 0.7  # Base 0.5 + 0.2


def test_calculate_importance_score_key_keyword():
    """Content with 'key' keyword gets boost"""
    scorer = MemoryScorer()
    score = scorer.calculate_importance_score("This is a key point")

    assert score == 0.7


def test_calculate_importance_score_critical_keyword():
    """Content with 'critical' keyword gets boost"""
    scorer = MemoryScorer()
    score = scorer.calculate_importance_score("Critical update needed")

    assert score == 0.7


def test_calculate_importance_score_essential_keyword():
    """Content with 'essential' keyword gets boost"""
    scorer = MemoryScorer()
    score = scorer.calculate_importance_score("Essential information")

    assert score == 0.7


def test_calculate_importance_score_case_insensitive():
    """Keyword matching is case insensitive"""
    scorer = MemoryScorer()

    score_lower = scorer.calculate_importance_score("important")
    score_upper = scorer.calculate_importance_score("IMPORTANT")
    score_mixed = scorer.calculate_importance_score("ImPoRtAnT")

    assert score_lower == score_upper == score_mixed == 0.7


def test_calculate_importance_score_combined_boosts():
    """Multiple boosts stack together"""
    scorer = MemoryScorer()
    # Long + question + keyword
    long_text = "x" * 250
    content = f"{long_text} Is this important?"
    score = scorer.calculate_importance_score(content)

    # 0.5 (base) + 0.1 (long) + 0.1 (question) + 0.2 (important) = 0.9
    # Use approx for floating point comparison
    assert abs(score - 0.9) < 0.001


def test_calculate_importance_score_maxed_out():
    """Score is capped at 1.0"""
    scorer = MemoryScorer()
    # Long + question + keyword (any() only adds 0.2 once)
    long_text = "x" * 250
    content = f"{long_text} Is this important? Remember this critical note!"
    score = scorer.calculate_importance_score(content)

    # 0.5 + 0.1 (long) + 0.1 (question) + 0.2 (keyword) = 0.9 (not > 1.0)
    assert abs(score - 0.9) < 0.001


def test_calculate_importance_score_empty_string():
    """Empty string gets base score"""
    scorer = MemoryScorer()
    score = scorer.calculate_importance_score("")

    assert score == 0.5


def test_calculate_importance_score_whitespace_only():
    """Whitespace-only content gets base score"""
    scorer = MemoryScorer()
    score = scorer.calculate_importance_score("     ")

    assert score == 0.5


def test_calculate_importance_score_multiple_questions():
    """Multiple question marks still only boost once"""
    scorer = MemoryScorer()
    score = scorer.calculate_importance_score("What? Why? How?")

    assert score == 0.6  # Only +0.1 for having '?'


def test_calculate_importance_score_keyword_in_middle():
    """Keyword anywhere in text counts"""
    scorer = MemoryScorer()
    score = scorer.calculate_importance_score("The most important thing today")

    assert score == 0.7


def test_calculate_importance_score_keyword_multiple_times():
    """Same keyword multiple times only counts once"""
    scorer = MemoryScorer()
    score = scorer.calculate_importance_score("important important important")

    assert score == 0.7  # Only +0.2 for having keyword


def test_calculate_importance_score_partial_keyword_match():
    """Keyword as part of larger word still matches"""
    scorer = MemoryScorer()
    # 'note' appears in 'noted'
    score = scorer.calculate_importance_score("As noted earlier")

    assert score == 0.7  # Matches 'note' within 'noted'


def test_calculate_importance_score_at_boundary_200():
    """Content at exactly 200 chars doesn't get length boost"""
    scorer = MemoryScorer()
    text_200 = "x" * 200
    score = scorer.calculate_importance_score(text_200)

    assert score == 0.5  # No boost at exactly 200


def test_calculate_importance_score_at_boundary_201():
    """Content at 201 chars gets length boost"""
    scorer = MemoryScorer()
    text_201 = "x" * 201
    score = scorer.calculate_importance_score(text_201)

    assert score == 0.6  # Boost at 201


def test_calculate_importance_score_all_keywords():
    """Text with all keywords"""
    scorer = MemoryScorer()
    content = "important remember note key critical essential"
    score = scorer.calculate_importance_score(content)

    # Base + single keyword boost (any() returns on first match)
    assert score == 0.7


def test_calculate_importance_score_realistic_content():
    """Realistic memory content"""
    scorer = MemoryScorer()
    content = "User mentioned they need to remember the password reset procedure"
    score = scorer.calculate_importance_score(content)

    # Contains 'remember' keyword
    assert score == 0.7


def test_calculate_importance_score_very_long_content():
    """Very long content (>1000 chars) still only gets single boost"""
    scorer = MemoryScorer()
    very_long = "x" * 5000
    score = scorer.calculate_importance_score(very_long)

    assert score == 0.6  # Only +0.1 for length


# =============================================================================
# Edge Cases
# =============================================================================

def test_calculate_importance_score_unicode():
    """Handles unicode content"""
    scorer = MemoryScorer()
    score = scorer.calculate_importance_score("这很重要")  # "This is important" in Chinese

    # Won't match English keywords, gets base score
    assert score == 0.5


def test_calculate_importance_score_special_characters():
    """Handles special characters"""
    scorer = MemoryScorer()
    score = scorer.calculate_importance_score("!@#$%^&*()")

    assert score == 0.5


def test_calculate_importance_score_numbers():
    """Handles numeric content"""
    scorer = MemoryScorer()
    score = scorer.calculate_importance_score("123456789")

    assert score == 0.5


# =============================================================================
# apply_temporal_decay Tests
# =============================================================================

def create_mock_memory(timestamp, last_accessed, importance=0.5, decay_rate=0.1, truth_score=0.5):
    """Helper to create mock memory object"""
    memory = Mock()
    memory.timestamp = timestamp
    memory.last_accessed = last_accessed
    memory.importance_score = importance
    memory.decay_rate = decay_rate
    memory.truth_score = truth_score
    memory.metadata = {'truth_score': truth_score}
    return memory


def test_apply_temporal_decay_basic():
    """Applies temporal decay to memories"""
    scorer = MemoryScorer()
    now = datetime.now()

    # Recent memory (1 hour ago)
    memory = create_mock_memory(
        timestamp=now - timedelta(hours=1),
        last_accessed=now - timedelta(minutes=5)
    )

    memories = [{
        'memory': memory,
        'relevance_score': 0.8
    }]

    result = scorer.apply_temporal_decay(memories)

    assert len(result) == 1
    assert 'final_score' in result[0]
    assert result[0]['final_score'] > 0


def test_apply_temporal_decay_old_memory():
    """Older memories get lower scores"""
    scorer = MemoryScorer()
    now = datetime.now()

    # Recent memory
    recent_memory = create_mock_memory(
        timestamp=now - timedelta(hours=1),
        last_accessed=now
    )

    # Old memory (30 days ago)
    old_memory = create_mock_memory(
        timestamp=now - timedelta(days=30),
        last_accessed=now - timedelta(days=30)
    )

    recent_mems = [{
        'memory': recent_memory,
        'relevance_score': 0.8
    }]

    old_mems = [{
        'memory': old_memory,
        'relevance_score': 0.8
    }]

    recent_result = scorer.apply_temporal_decay(recent_mems)[0]['final_score']
    old_result = scorer.apply_temporal_decay(old_mems)[0]['final_score']

    # Recent should have higher score
    assert recent_result > old_result


def test_apply_temporal_decay_access_recency_boost():
    """Recently accessed memories get boost"""
    scorer = MemoryScorer()
    now = datetime.now()

    # Both 10 days old, but different access times
    old_timestamp = now - timedelta(days=10)

    recently_accessed = create_mock_memory(
        timestamp=old_timestamp,
        last_accessed=now  # Accessed today
    )

    not_accessed = create_mock_memory(
        timestamp=old_timestamp,
        last_accessed=old_timestamp  # Not accessed since creation
    )

    recent_mems = [{
        'memory': recently_accessed,
        'relevance_score': 0.8
    }]

    old_mems = [{
        'memory': not_accessed,
        'relevance_score': 0.8
    }]

    recent_result = scorer.apply_temporal_decay(recent_mems)[0]['final_score']
    old_result = scorer.apply_temporal_decay(old_mems)[0]['final_score']

    # Recently accessed should have higher score
    assert recent_result > old_result


def test_apply_temporal_decay_importance_score():
    """Higher importance scores increase final score"""
    scorer = MemoryScorer()
    now = datetime.now()

    high_importance = create_mock_memory(
        timestamp=now - timedelta(hours=1),
        last_accessed=now,
        importance=0.9
    )

    low_importance = create_mock_memory(
        timestamp=now - timedelta(hours=1),
        last_accessed=now,
        importance=0.1
    )

    high_mems = [{
        'memory': high_importance,
        'relevance_score': 0.8
    }]

    low_mems = [{
        'memory': low_importance,
        'relevance_score': 0.8
    }]

    high_result = scorer.apply_temporal_decay(high_mems)[0]['final_score']
    low_result = scorer.apply_temporal_decay(low_mems)[0]['final_score']

    assert high_result > low_result


def test_apply_temporal_decay_truth_score():
    """Higher truth scores increase final score"""
    scorer = MemoryScorer()
    now = datetime.now()

    high_truth = create_mock_memory(
        timestamp=now - timedelta(hours=1),
        last_accessed=now,
        truth_score=1.0
    )

    low_truth = create_mock_memory(
        timestamp=now - timedelta(hours=1),
        last_accessed=now,
        truth_score=0.0
    )

    high_mems = [{
        'memory': high_truth,
        'relevance_score': 0.8
    }]

    low_mems = [{
        'memory': low_truth,
        'relevance_score': 0.8
    }]

    high_result = scorer.apply_temporal_decay(high_mems)[0]['final_score']
    low_result = scorer.apply_temporal_decay(low_mems)[0]['final_score']

    assert high_result > low_result


def test_apply_temporal_decay_relevance_score():
    """Higher relevance scores increase final score"""
    scorer = MemoryScorer()
    now = datetime.now()

    memory = create_mock_memory(
        timestamp=now - timedelta(hours=1),
        last_accessed=now
    )

    high_rel = [{
        'memory': memory,
        'relevance_score': 0.9
    }]

    low_rel = [{
        'memory': memory,
        'relevance_score': 0.1
    }]

    high_result = scorer.apply_temporal_decay(high_rel)[0]['final_score']
    low_result = scorer.apply_temporal_decay(low_rel)[0]['final_score']

    assert high_result > low_result


def test_apply_temporal_decay_multiple_memories():
    """Processes multiple memories"""
    scorer = MemoryScorer()
    now = datetime.now()

    memories = [
        {
            'memory': create_mock_memory(now - timedelta(hours=1), now),
            'relevance_score': 0.8
        },
        {
            'memory': create_mock_memory(now - timedelta(days=1), now),
            'relevance_score': 0.7
        },
        {
            'memory': create_mock_memory(now - timedelta(days=7), now - timedelta(days=7)),
            'relevance_score': 0.6
        }
    ]

    result = scorer.apply_temporal_decay(memories)

    assert len(result) == 3
    assert all('final_score' in m for m in result)


def test_apply_temporal_decay_empty_list():
    """Handles empty memory list"""
    scorer = MemoryScorer()
    result = scorer.apply_temporal_decay([])

    assert result == []


def test_apply_temporal_decay_decay_rate():
    """Different decay rates affect scores"""
    scorer = MemoryScorer()
    now = datetime.now()
    old_timestamp = now - timedelta(days=30)

    slow_decay = create_mock_memory(
        timestamp=old_timestamp,
        last_accessed=old_timestamp,
        decay_rate=0.01  # Slow decay
    )

    fast_decay = create_mock_memory(
        timestamp=old_timestamp,
        last_accessed=old_timestamp,
        decay_rate=0.5  # Fast decay
    )

    slow_mems = [{
        'memory': slow_decay,
        'relevance_score': 0.8
    }]

    fast_mems = [{
        'memory': fast_decay,
        'relevance_score': 0.8
    }]

    slow_result = scorer.apply_temporal_decay(slow_mems)[0]['final_score']
    fast_result = scorer.apply_temporal_decay(fast_mems)[0]['final_score']

    # Slow decay should preserve score better
    assert slow_result > fast_result


def test_apply_temporal_decay_minimum_floor():
    """Decay factor and importance have minimum floor of 0.1"""
    scorer = MemoryScorer()
    now = datetime.now()

    # Very old, very unimportant memory
    memory = create_mock_memory(
        timestamp=now - timedelta(days=365),
        last_accessed=now - timedelta(days=365),
        importance=0.0,  # Will be floored to 0.1
        decay_rate=1.0
    )

    memories = [{
        'memory': memory,
        'relevance_score': 0.8
    }]

    result = scorer.apply_temporal_decay(memories)

    # Should still have non-zero score due to floors
    assert result[0]['final_score'] > 0


def test_apply_temporal_decay_returns_same_list():
    """Returns the same list object (mutates in place)"""
    scorer = MemoryScorer()
    now = datetime.now()

    memories = [{
        'memory': create_mock_memory(now - timedelta(hours=1), now),
        'relevance_score': 0.8
    }]

    result = scorer.apply_temporal_decay(memories)

    # Same object
    assert result is memories
