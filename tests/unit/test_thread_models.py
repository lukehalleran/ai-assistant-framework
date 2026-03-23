"""Tests for memory/thread_models.py — OpenThread data models, serialization, and scoring."""

import pytest
import time
from unittest.mock import patch

from memory.thread_models import OpenThread, ThreadType, ThreadStatus, TYPE_PRIORITY


# ---------------------------------------------------------------------------
# ThreadType enum
# ---------------------------------------------------------------------------

class TestThreadType:
    """Test ThreadType enum values."""

    def test_commitment_value(self):
        assert ThreadType.COMMITMENT.value == "commitment"

    def test_deadline_value(self):
        assert ThreadType.DEADLINE.value == "deadline"

    def test_unfinished_value(self):
        assert ThreadType.UNFINISHED.value == "unfinished"

    def test_question_value(self):
        assert ThreadType.QUESTION.value == "question"

    def test_all_four_members(self):
        assert len(ThreadType) == 4

    def test_string_enum_construction(self):
        """ThreadType(str, Enum) allows construction from string value."""
        assert ThreadType("commitment") is ThreadType.COMMITMENT
        assert ThreadType("deadline") is ThreadType.DEADLINE


# ---------------------------------------------------------------------------
# ThreadStatus enum
# ---------------------------------------------------------------------------

class TestThreadStatus:
    """Test ThreadStatus enum values."""

    def test_open_value(self):
        assert ThreadStatus.OPEN.value == "open"

    def test_resolved_value(self):
        assert ThreadStatus.RESOLVED.value == "resolved"

    def test_stale_value(self):
        assert ThreadStatus.STALE.value == "stale"

    def test_all_three_members(self):
        assert len(ThreadStatus) == 3


# ---------------------------------------------------------------------------
# TYPE_PRIORITY mapping
# ---------------------------------------------------------------------------

class TestTypePriority:
    """Test TYPE_PRIORITY constant values."""

    def test_deadline_has_highest_priority(self):
        assert TYPE_PRIORITY[ThreadType.DEADLINE] == 1.0

    def test_commitment_priority(self):
        assert TYPE_PRIORITY[ThreadType.COMMITMENT] == 0.8

    def test_question_priority(self):
        assert TYPE_PRIORITY[ThreadType.QUESTION] == 0.6

    def test_unfinished_has_lowest_priority(self):
        assert TYPE_PRIORITY[ThreadType.UNFINISHED] == 0.4

    def test_priority_ordering(self):
        """DEADLINE > COMMITMENT > QUESTION > UNFINISHED."""
        assert (
            TYPE_PRIORITY[ThreadType.DEADLINE]
            > TYPE_PRIORITY[ThreadType.COMMITMENT]
            > TYPE_PRIORITY[ThreadType.QUESTION]
            > TYPE_PRIORITY[ThreadType.UNFINISHED]
        )

    def test_all_thread_types_covered(self):
        for tt in ThreadType:
            assert tt in TYPE_PRIORITY


# ---------------------------------------------------------------------------
# OpenThread construction
# ---------------------------------------------------------------------------

class TestOpenThreadConstruction:
    """Test OpenThread creation with defaults and explicit values."""

    def test_minimal_construction(self):
        """Only topic is required; everything else has defaults."""
        t = OpenThread(topic="Fix the bug")
        assert t.topic == "Fix the bug"
        assert t.summary == ""
        assert t.status == ThreadStatus.OPEN
        assert t.thread_type == ThreadType.UNFINISHED
        assert t.urgency == 0.5
        assert t.resolution_hint == ""
        assert t.source_summary == ""
        assert t.deadline_date is None
        # thread_id should be a UUID string
        assert len(t.thread_id) == 36  # UUID4 format: 8-4-4-4-12

    def test_full_construction(self):
        """All fields explicitly provided."""
        now = time.time()
        t = OpenThread(
            thread_id="abc-123",
            topic="Ship v2.0",
            summary="Release version 2 by end of month",
            status=ThreadStatus.OPEN,
            thread_type=ThreadType.DEADLINE,
            urgency=0.9,
            mentioned_at=now - 3600,
            last_referenced=now,
            resolution_hint="Merge remaining PRs",
            source_summary="session-42",
            deadline_date="2026-03-31",
        )
        assert t.thread_id == "abc-123"
        assert t.topic == "Ship v2.0"
        assert t.summary == "Release version 2 by end of month"
        assert t.status == ThreadStatus.OPEN
        assert t.thread_type == ThreadType.DEADLINE
        assert t.urgency == 0.9
        assert t.mentioned_at == pytest.approx(now - 3600, abs=1)
        assert t.last_referenced == pytest.approx(now, abs=1)
        assert t.resolution_hint == "Merge remaining PRs"
        assert t.source_summary == "session-42"
        assert t.deadline_date == "2026-03-31"

    def test_default_timestamps_are_recent(self):
        before = time.time()
        t = OpenThread(topic="test timestamps")
        after = time.time()
        assert before <= t.mentioned_at <= after
        assert before <= t.last_referenced <= after

    def test_empty_summary_allowed(self):
        t = OpenThread(topic="Minimal", summary="")
        assert t.summary == ""

    def test_long_topic_capped_at_200(self):
        """Pydantic max_length=200 on topic field."""
        with pytest.raises(Exception):
            OpenThread(topic="x" * 201)

    def test_topic_exactly_200_chars(self):
        t = OpenThread(topic="a" * 200)
        assert len(t.topic) == 200

    def test_urgency_clamped_low(self):
        """Urgency below 0.0 should be rejected by Pydantic ge=0.0."""
        with pytest.raises(Exception):
            OpenThread(topic="test", urgency=-0.1)

    def test_urgency_clamped_high(self):
        """Urgency above 1.0 should be rejected by Pydantic le=1.0."""
        with pytest.raises(Exception):
            OpenThread(topic="test", urgency=1.1)


# ---------------------------------------------------------------------------
# to_embedding_text()
# ---------------------------------------------------------------------------

class TestToEmbeddingText:
    """Test embedding text generation."""

    def test_basic_format(self):
        t = OpenThread(topic="Database migration", thread_type=ThreadType.COMMITMENT)
        text = t.to_embedding_text()
        assert "Thread: Database migration" in text
        assert "Type: commitment" in text

    def test_includes_summary_when_present(self):
        t = OpenThread(
            topic="API refactor",
            summary="Rewrite REST endpoints to GraphQL",
            thread_type=ThreadType.UNFINISHED,
        )
        text = t.to_embedding_text()
        assert "Summary: Rewrite REST endpoints to GraphQL" in text

    def test_excludes_summary_when_empty(self):
        t = OpenThread(topic="Quick fix", summary="", thread_type=ThreadType.QUESTION)
        text = t.to_embedding_text()
        assert "Summary:" not in text

    def test_includes_deadline_when_present(self):
        t = OpenThread(
            topic="Tax filing",
            thread_type=ThreadType.DEADLINE,
            deadline_date="2026-04-15",
        )
        text = t.to_embedding_text()
        assert "Deadline: 2026-04-15" in text

    def test_excludes_deadline_when_none(self):
        t = OpenThread(topic="Someday", thread_type=ThreadType.UNFINISHED)
        text = t.to_embedding_text()
        assert "Deadline:" not in text

    def test_pipe_separated_parts(self):
        t = OpenThread(
            topic="Deploy",
            summary="Push to prod",
            thread_type=ThreadType.DEADLINE,
            deadline_date="2026-12-31",
        )
        text = t.to_embedding_text()
        assert text.count(" | ") == 3  # 4 parts joined by 3 separators


# ---------------------------------------------------------------------------
# to_metadata() / from_metadata() round-trip
# ---------------------------------------------------------------------------

class TestMetadataSerialization:
    """Test ChromaDB metadata serialization round-trip."""

    def test_round_trip_preserves_all_fields(self):
        now = time.time()
        original = OpenThread(
            thread_id="rt-001",
            topic="Round trip test",
            summary="Ensure metadata preserves data",
            status=ThreadStatus.OPEN,
            thread_type=ThreadType.COMMITMENT,
            urgency=0.75,
            mentioned_at=now - 7200,
            last_referenced=now,
            resolution_hint="Check all fields",
            source_summary="src-99",
            deadline_date="2026-06-01",
        )
        meta = original.to_metadata()
        restored = OpenThread.from_metadata(meta)

        assert restored.thread_id == original.thread_id
        assert restored.topic == original.topic
        assert restored.summary == original.summary
        assert restored.status == original.status
        assert restored.thread_type == original.thread_type
        assert restored.urgency == original.urgency
        assert restored.mentioned_at == original.mentioned_at
        assert restored.last_referenced == original.last_referenced
        assert restored.resolution_hint == original.resolution_hint
        assert restored.source_summary == original.source_summary
        assert restored.deadline_date == original.deadline_date

    def test_none_deadline_becomes_empty_string_in_metadata(self):
        t = OpenThread(topic="No deadline", deadline_date=None)
        meta = t.to_metadata()
        assert meta["deadline_date"] == ""

    def test_empty_deadline_string_becomes_none_after_round_trip(self):
        meta = {
            "thread_id": "x",
            "topic": "Test",
            "summary": "",
            "status": "open",
            "thread_type": "unfinished",
            "urgency": 0.5,
            "mentioned_at": time.time(),
            "last_referenced": time.time(),
            "resolution_hint": "",
            "source_summary": "",
            "deadline_date": "",
        }
        restored = OpenThread.from_metadata(meta)
        assert restored.deadline_date is None

    def test_metadata_values_are_flat_primitives(self):
        """ChromaDB requires flat primitives (str, int, float, bool)."""
        t = OpenThread(topic="Flat check", thread_type=ThreadType.QUESTION)
        meta = t.to_metadata()
        for key, value in meta.items():
            assert isinstance(value, (str, int, float, bool)), (
                f"metadata[{key!r}] = {type(value).__name__}, expected flat primitive"
            )

    def test_metadata_truncates_long_fields(self):
        t = OpenThread(
            topic="a" * 200,
            summary="b" * 1500,
            resolution_hint="c" * 800,
            source_summary="d" * 800,
        )
        meta = t.to_metadata()
        assert len(meta["topic"]) <= 200
        assert len(meta["summary"]) <= 1000
        assert len(meta["resolution_hint"]) <= 500
        assert len(meta["source_summary"]) <= 500

    def test_from_metadata_defaults_for_missing_keys(self):
        """from_metadata should handle sparse dicts gracefully."""
        restored = OpenThread.from_metadata({"topic": "Sparse"})
        assert restored.topic == "Sparse"
        assert restored.status == ThreadStatus.OPEN
        assert restored.thread_type == ThreadType.UNFINISHED
        assert restored.urgency == 0.5
        assert restored.summary == ""
        assert restored.resolution_hint == ""
        assert restored.deadline_date is None


# ---------------------------------------------------------------------------
# to_dict() / from_dict() round-trip
# ---------------------------------------------------------------------------

class TestDictSerialization:
    """Test full dict serialization round-trip."""

    def test_round_trip_preserves_all_fields(self):
        now = time.time()
        original = OpenThread(
            thread_id="dict-001",
            topic="Dict round trip",
            summary="Full JSON export",
            status=ThreadStatus.RESOLVED,
            thread_type=ThreadType.QUESTION,
            urgency=0.3,
            mentioned_at=now - 86400,
            last_referenced=now - 3600,
            resolution_hint="Answered in session",
            source_summary="session-7",
            deadline_date=None,
        )
        d = original.to_dict()
        restored = OpenThread.from_dict(d)

        assert restored.thread_id == original.thread_id
        assert restored.topic == original.topic
        assert restored.summary == original.summary
        assert restored.status == original.status
        assert restored.thread_type == original.thread_type
        assert restored.urgency == original.urgency
        assert restored.mentioned_at == original.mentioned_at
        assert restored.last_referenced == original.last_referenced
        assert restored.resolution_hint == original.resolution_hint
        assert restored.source_summary == original.source_summary
        assert restored.deadline_date == original.deadline_date

    def test_to_dict_preserves_none_deadline(self):
        """Unlike to_metadata, to_dict keeps None as None (not empty string)."""
        t = OpenThread(topic="Nil deadline", deadline_date=None)
        d = t.to_dict()
        assert d["deadline_date"] is None

    def test_to_dict_preserves_deadline_string(self):
        t = OpenThread(topic="Has deadline", deadline_date="2026-12-31")
        d = t.to_dict()
        assert d["deadline_date"] == "2026-12-31"

    def test_from_dict_defaults_for_missing_keys(self):
        restored = OpenThread.from_dict({"topic": "Sparse dict"})
        assert restored.topic == "Sparse dict"
        assert restored.status == ThreadStatus.OPEN
        assert restored.thread_type == ThreadType.UNFINISHED


# ---------------------------------------------------------------------------
# priority_score()
# ---------------------------------------------------------------------------

class TestPriorityScore:
    """Test priority scoring formula."""

    def test_recent_deadline_high_urgency_is_highest(self):
        """A brand-new DEADLINE thread with urgency=1.0 should score 1.0."""
        now = time.time()
        with patch("memory.thread_models.time") as mock_time:
            mock_time.time.return_value = now
            t = OpenThread(
                topic="Due today",
                thread_type=ThreadType.DEADLINE,
                urgency=1.0,
                last_referenced=now,
            )
            score = t.priority_score()
        # type=1.0, urgency=1.0, recency_decay=1.0 (0 days) => 1.0
        assert score == pytest.approx(1.0, abs=0.01)

    def test_zero_urgency_gives_zero_score(self):
        t = OpenThread(topic="No urgency", urgency=0.0, thread_type=ThreadType.DEADLINE)
        assert t.priority_score() == 0.0

    def test_score_decays_over_time(self):
        now = time.time()
        t = OpenThread(
            topic="Aging thread",
            thread_type=ThreadType.COMMITMENT,
            urgency=1.0,
            last_referenced=now,
        )
        # Score at time 0
        with patch("memory.thread_models.time") as mock_time:
            mock_time.time.return_value = now
            score_now = t.priority_score()

        # Score 7 days later
        with patch("memory.thread_models.time") as mock_time:
            mock_time.time.return_value = now + 7 * 86400
            score_7d = t.priority_score()

        assert score_7d < score_now

    def test_decay_floors_at_0_1(self):
        """Recency decay has a floor of 0.1, even after many days."""
        now = time.time()
        t = OpenThread(
            topic="Ancient thread",
            thread_type=ThreadType.DEADLINE,
            urgency=1.0,
            last_referenced=now,
        )
        # 100 days later — well past 14-day full decay
        with patch("memory.thread_models.time") as mock_time:
            mock_time.time.return_value = now + 100 * 86400
            score = t.priority_score()
        # type=1.0 * urgency=1.0 * decay=0.1 = 0.1
        assert score == pytest.approx(0.1, abs=0.01)

    def test_priority_ordering_by_type(self):
        """With same urgency and recency, DEADLINE > COMMITMENT > QUESTION > UNFINISHED."""
        now = time.time()
        threads = []
        for tt in [ThreadType.DEADLINE, ThreadType.COMMITMENT, ThreadType.QUESTION, ThreadType.UNFINISHED]:
            t = OpenThread(
                topic=f"Thread {tt.value}",
                thread_type=tt,
                urgency=1.0,
                last_referenced=now,
            )
            threads.append(t)

        with patch("memory.thread_models.time") as mock_time:
            mock_time.time.return_value = now
            scores = [t.priority_score() for t in threads]

        # Strictly decreasing
        assert scores[0] > scores[1] > scores[2] > scores[3]

    def test_priority_score_at_7_days(self):
        """At 7 days: decay = max(0.1, 1.0 - 7/14) = 0.5."""
        now = time.time()
        t = OpenThread(
            topic="One week old",
            thread_type=ThreadType.COMMITMENT,  # weight 0.8
            urgency=1.0,
            last_referenced=now,
        )
        with patch("memory.thread_models.time") as mock_time:
            mock_time.time.return_value = now + 7 * 86400
            score = t.priority_score()
        # 0.8 * 1.0 * 0.5 = 0.4
        assert score == pytest.approx(0.4, abs=0.01)


# ---------------------------------------------------------------------------
# is_stale()
# ---------------------------------------------------------------------------

class TestIsStale:
    """Test staleness detection."""

    def test_fresh_thread_is_not_stale(self):
        t = OpenThread(topic="Just created")
        assert t.is_stale() is False

    def test_old_thread_is_stale(self):
        now = time.time()
        t = OpenThread(topic="Old thread", last_referenced=now - 15 * 86400)
        with patch("memory.thread_models.time") as mock_time:
            mock_time.time.return_value = now
            assert t.is_stale() is True

    def test_exactly_14_days_is_stale(self):
        now = time.time()
        t = OpenThread(topic="Boundary", last_referenced=now - 14 * 86400)
        with patch("memory.thread_models.time") as mock_time:
            mock_time.time.return_value = now
            assert t.is_stale() is True

    def test_just_under_14_days_is_not_stale(self):
        now = time.time()
        t = OpenThread(topic="Almost stale", last_referenced=now - 13.9 * 86400)
        with patch("memory.thread_models.time") as mock_time:
            mock_time.time.return_value = now
            assert t.is_stale() is False

    def test_custom_stale_days(self):
        now = time.time()
        t = OpenThread(topic="Short window", last_referenced=now - 3 * 86400)
        with patch("memory.thread_models.time") as mock_time:
            mock_time.time.return_value = now
            # Not stale with default 14 days
            assert t.is_stale() is False
            # Stale with custom 2 days
            assert t.is_stale(stale_days=2) is True


# ---------------------------------------------------------------------------
# mark_resolved() / mark_stale()
# ---------------------------------------------------------------------------

class TestStatusTransitions:
    """Test status lifecycle transitions."""

    def test_mark_resolved_changes_status(self):
        t = OpenThread(topic="Resolve me")
        assert t.status == ThreadStatus.OPEN
        t.mark_resolved()
        assert t.status == ThreadStatus.RESOLVED

    def test_mark_resolved_updates_last_referenced(self):
        now = time.time()
        t = OpenThread(topic="Resolve with timestamp", last_referenced=now - 86400)
        before = t.last_referenced
        t.mark_resolved()
        assert t.last_referenced > before

    def test_mark_resolved_with_resolution_text(self):
        t = OpenThread(topic="Has resolution")
        t.mark_resolved(resolution="Fixed in PR #42")
        assert t.status == ThreadStatus.RESOLVED
        assert t.resolution_hint == "Fixed in PR #42"

    def test_mark_resolved_without_resolution_preserves_hint(self):
        t = OpenThread(topic="Keep hint", resolution_hint="Original hint")
        t.mark_resolved()
        assert t.resolution_hint == "Original hint"

    def test_mark_stale_changes_status(self):
        t = OpenThread(topic="Stale me")
        assert t.status == ThreadStatus.OPEN
        t.mark_stale()
        assert t.status == ThreadStatus.STALE
