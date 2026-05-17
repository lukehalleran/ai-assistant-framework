"""
Unit tests for session boundary detection and formatting in prompt assembly.
"""

import pytest
from datetime import datetime, timedelta
from core.prompt.formatter import (
    _detect_session_boundary,
    _format_session_header,
    _parse_entry_timestamp,
)


class TestDetectSessionBoundary:
    """Tests for _detect_session_boundary()."""

    def test_first_entry_is_always_boundary(self):
        ts = datetime(2026, 5, 17, 10, 0, 0)
        assert _detect_session_boundary(None, ts) is True

    def test_same_session_no_boundary(self):
        ts1 = datetime(2026, 5, 17, 10, 0, 0)
        ts2 = datetime(2026, 5, 17, 10, 30, 0)
        assert _detect_session_boundary(ts1, ts2) is False

    def test_cross_day_is_boundary(self):
        ts1 = datetime(2026, 5, 16, 23, 0, 0)
        ts2 = datetime(2026, 5, 17, 8, 0, 0)
        assert _detect_session_boundary(ts1, ts2) is True

    def test_same_day_large_gap_is_boundary(self):
        ts1 = datetime(2026, 5, 17, 8, 0, 0)
        ts2 = datetime(2026, 5, 17, 14, 0, 0)  # 6 hours later
        assert _detect_session_boundary(ts1, ts2) is True

    def test_same_day_small_gap_no_boundary(self):
        ts1 = datetime(2026, 5, 17, 10, 0, 0)
        ts2 = datetime(2026, 5, 17, 11, 30, 0)  # 1.5 hours
        assert _detect_session_boundary(ts1, ts2) is False

    def test_exactly_at_threshold(self):
        ts1 = datetime(2026, 5, 17, 10, 0, 0)
        ts2 = datetime(2026, 5, 17, 12, 0, 0)  # exactly 2 hours
        assert _detect_session_boundary(ts1, ts2) is True

    def test_custom_gap_hours(self):
        ts1 = datetime(2026, 5, 17, 10, 0, 0)
        ts2 = datetime(2026, 5, 17, 11, 0, 0)  # 1 hour gap
        # Default 2h threshold: no boundary
        assert _detect_session_boundary(ts1, ts2) is False
        # Custom 0.5h threshold: boundary
        assert _detect_session_boundary(ts1, ts2, gap_hours=0.5) is True

    def test_none_current_no_boundary(self):
        ts = datetime(2026, 5, 17, 10, 0, 0)
        assert _detect_session_boundary(ts, None) is False

    def test_both_none(self):
        assert _detect_session_boundary(None, None) is True

    def test_reverse_chronological_order(self):
        """Entries are typically most-recent-first."""
        ts_newer = datetime(2026, 5, 17, 14, 0, 0)
        ts_older = datetime(2026, 5, 16, 10, 0, 0)
        # First entry (newer) -> boundary because prev=None
        assert _detect_session_boundary(None, ts_newer) is True
        # Second entry (older) -> boundary because different day
        assert _detect_session_boundary(ts_newer, ts_older) is True

    def test_timezone_aware_datetimes(self):
        from datetime import timezone
        ts1 = datetime(2026, 5, 17, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 5, 17, 10, 30, 0, tzinfo=timezone.utc)
        assert _detect_session_boundary(ts1, ts2) is False


class TestFormatSessionHeader:
    """Tests for _format_session_header()."""

    def test_contains_session_marker(self):
        ts = datetime(2026, 5, 17, 10, 0, 0)
        header = _format_session_header(ts)
        assert "--- Session:" in header
        assert "---" in header

    def test_contains_day_name(self):
        ts = datetime(2026, 5, 17, 10, 0, 0)  # Sunday
        header = _format_session_header(ts)
        assert "Sun" in header
        assert "May" in header
        assert "17" in header

    def test_different_dates_produce_different_headers(self):
        ts1 = datetime(2026, 5, 16, 10, 0, 0)
        ts2 = datetime(2026, 5, 17, 10, 0, 0)
        h1 = _format_session_header(ts1)
        h2 = _format_session_header(ts2)
        assert h1 != h2


class TestParseEntryTimestamp:
    """Tests for _parse_entry_timestamp()."""

    def test_iso_string_timestamp(self):
        mem = {"timestamp": "2026-05-17T10:30:00"}
        ts = _parse_entry_timestamp(mem)
        assert ts is not None
        assert ts.year == 2026
        assert ts.month == 5
        assert ts.day == 17

    def test_datetime_object(self):
        dt = datetime(2026, 5, 17, 10, 0, 0)
        mem = {"timestamp": dt}
        ts = _parse_entry_timestamp(mem)
        assert ts == dt

    def test_metadata_timestamp(self):
        mem = {"metadata": {"timestamp": "2026-05-17T10:30:00"}}
        ts = _parse_entry_timestamp(mem)
        assert ts is not None

    def test_missing_timestamp(self):
        mem = {"query": "hello"}
        ts = _parse_entry_timestamp(mem)
        assert ts is None

    def test_invalid_timestamp(self):
        mem = {"timestamp": "not-a-date"}
        ts = _parse_entry_timestamp(mem)
        assert ts is None


class TestSessionBoundaryRendering:
    """Integration test: verify session markers appear in formatted output."""

    def test_two_sessions_produce_two_headers(self):
        """Construct mock conversations spanning 2 days, verify markers."""
        conversations = [
            {
                "query": "How is the benchmark?",
                "response": "Looking good.",
                "timestamp": "2026-05-17T10:00:00",
            },
            {
                "query": "This song makes me happy cry",
                "response": "Those are the best kind.",
                "timestamp": "2026-05-16T12:03:00",
            },
        ]

        # Render using the session boundary logic directly
        from core.prompt.formatter import (
            _detect_session_boundary,
            _format_session_header,
            _parse_entry_timestamp,
        )

        lines = []
        prev_ts = None
        for i, mem in enumerate(conversations, start=1):
            entry_ts = _parse_entry_timestamp(mem)
            if _detect_session_boundary(prev_ts, entry_ts):
                if entry_ts:
                    lines.append(_format_session_header(entry_ts))
            if entry_ts:
                prev_ts = entry_ts
            lines.append(f"{i}) {mem.get('query', '')}")

        output = "\n".join(lines)
        # Should have exactly 2 session headers
        assert output.count("--- Session:") == 2

    def test_same_session_produces_one_header(self):
        """Conversations within the same session get one header."""
        conversations = [
            {"query": "msg 1", "timestamp": "2026-05-17T10:00:00"},
            {"query": "msg 2", "timestamp": "2026-05-17T10:15:00"},
            {"query": "msg 3", "timestamp": "2026-05-17T10:30:00"},
        ]

        from core.prompt.formatter import (
            _detect_session_boundary,
            _format_session_header,
            _parse_entry_timestamp,
        )

        headers = 0
        prev_ts = None
        for mem in conversations:
            entry_ts = _parse_entry_timestamp(mem)
            if _detect_session_boundary(prev_ts, entry_ts):
                headers += 1
            if entry_ts:
                prev_ts = entry_ts

        assert headers == 1  # Only the first entry triggers a boundary
