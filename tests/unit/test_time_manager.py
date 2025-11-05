"""
Unit tests for utils/time_manager.py

Tests all time management functionality:
- Time tracking and formatting
- File persistence (with temp files)
- Elapsed time calculations
- Response time measurement
"""

import pytest
import json
import os
from datetime import datetime, timedelta
from utils.time_manager import TimeManager


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_time_file(tmp_path):
    """Create a temporary file for time storage"""
    return str(tmp_path / "test_query_time.json")


@pytest.fixture
def time_manager(temp_time_file):
    """Create a TimeManager instance with temp file"""
    return TimeManager(time_file=temp_time_file)


# =============================================================================
# Initialization Tests
# =============================================================================

def test_init_no_existing_file(temp_time_file):
    """TimeManager initializes with None when no file exists"""
    tm = TimeManager(time_file=temp_time_file)
    assert tm.last_query_time is None
    assert tm.last_response_time is None
    assert tm.time_file == temp_time_file


def test_init_with_existing_file(temp_time_file):
    """TimeManager loads existing timestamp from file"""
    # Create a file with a timestamp
    test_time = datetime(2024, 1, 1, 12, 0, 0)
    with open(temp_time_file, "w") as f:
        json.dump({"last_query_time": test_time.isoformat()}, f)

    tm = TimeManager(time_file=temp_time_file)
    assert tm.last_query_time == test_time


def test_init_with_corrupted_file(temp_time_file):
    """TimeManager handles corrupted JSON gracefully"""
    # Write invalid JSON
    with open(temp_time_file, "w") as f:
        f.write("not valid json{")

    tm = TimeManager(time_file=temp_time_file)
    assert tm.last_query_time is None


def test_init_with_missing_key(temp_time_file):
    """TimeManager handles missing key in JSON"""
    with open(temp_time_file, "w") as f:
        json.dump({"wrong_key": "2024-01-01T12:00:00"}, f)

    tm = TimeManager(time_file=temp_time_file)
    assert tm.last_query_time is None


def test_init_with_invalid_timestamp(temp_time_file):
    """TimeManager handles invalid timestamp format"""
    with open(temp_time_file, "w") as f:
        json.dump({"last_query_time": "not a timestamp"}, f)

    tm = TimeManager(time_file=temp_time_file)
    assert tm.last_query_time is None


# =============================================================================
# Current Time Tests
# =============================================================================

def test_current_returns_datetime(time_manager):
    """current() returns a datetime object"""
    result = time_manager.current()
    assert isinstance(result, datetime)


def test_current_is_recent(time_manager):
    """current() returns current time (within 1 second)"""
    before = datetime.now()
    result = time_manager.current()
    after = datetime.now()

    assert before <= result <= after


def test_current_iso_format(time_manager):
    """current_iso() returns ISO formatted string"""
    result = time_manager.current_iso()
    assert isinstance(result, str)
    # Should be in format: YYYY-MM-DD HH:MM:SS
    assert len(result.split()) == 2  # Date and time parts
    assert len(result.split()[0].split("-")) == 3  # YYYY-MM-DD
    assert len(result.split()[1].split(":")) == 3  # HH:MM:SS


# =============================================================================
# Elapsed Time Formatting Tests
# =============================================================================

def test_elapsed_since_last_no_previous(time_manager):
    """elapsed_since_last() returns N/A when no previous query"""
    assert time_manager.elapsed_since_last() == "N/A (first query)"


def test_elapsed_since_last_seconds(time_manager):
    """elapsed_since_last() formats seconds correctly"""
    time_manager.last_query_time = datetime.now() - timedelta(seconds=30)
    result = time_manager.elapsed_since_last()
    assert result.endswith("s")
    assert "30" in result or "29" in result  # Account for timing variation


def test_elapsed_since_last_minutes(time_manager):
    """elapsed_since_last() formats minutes correctly"""
    time_manager.last_query_time = datetime.now() - timedelta(minutes=5, seconds=30)
    result = time_manager.elapsed_since_last()
    assert "m" in result
    assert "5" in result


def test_elapsed_since_last_hours(time_manager):
    """elapsed_since_last() formats hours correctly"""
    time_manager.last_query_time = datetime.now() - timedelta(hours=2, minutes=30)
    result = time_manager.elapsed_since_last()
    assert "h" in result
    assert "m" in result
    assert "2" in result


def test_elapsed_since_last_days(time_manager):
    """elapsed_since_last() formats days correctly"""
    time_manager.last_query_time = datetime.now() - timedelta(days=3, hours=5)
    result = time_manager.elapsed_since_last()
    assert "d" in result
    assert "h" in result
    assert "3" in result


def test_elapsed_since_last_just_under_minute(time_manager):
    """elapsed_since_last() handles 59 seconds as seconds"""
    time_manager.last_query_time = datetime.now() - timedelta(seconds=59)
    result = time_manager.elapsed_since_last()
    assert "s" in result
    assert "m" not in result


def test_elapsed_since_last_exactly_one_minute(time_manager):
    """elapsed_since_last() handles exactly 60 seconds as minutes"""
    time_manager.last_query_time = datetime.now() - timedelta(seconds=60)
    result = time_manager.elapsed_since_last()
    assert "m" in result


# =============================================================================
# Mark Query Time Tests
# =============================================================================

def test_mark_query_time_sets_timestamp(time_manager):
    """mark_query_time() sets last_query_time"""
    before = datetime.now()
    result = time_manager.mark_query_time()
    after = datetime.now()

    assert before <= result <= after
    assert time_manager.last_query_time == result


def test_mark_query_time_saves_to_file(time_manager):
    """mark_query_time() persists to file"""
    time_manager.mark_query_time()

    # Check file was created
    assert os.path.exists(time_manager.time_file)

    # Check content
    with open(time_manager.time_file, "r") as f:
        data = json.load(f)

    assert "last_query_time" in data
    saved_time = datetime.fromisoformat(data["last_query_time"])
    assert abs((saved_time - datetime.now()).total_seconds()) < 2


def test_mark_query_time_returns_timestamp(time_manager):
    """mark_query_time() returns the marked time"""
    result = time_manager.mark_query_time()
    assert result == time_manager.last_query_time


# =============================================================================
# Response Measurement Tests
# =============================================================================

def test_measure_response_basic(time_manager):
    """measure_response() calculates elapsed time"""
    start = datetime.now()
    end = start + timedelta(seconds=5.5)

    result = time_manager.measure_response(start, end)

    assert result == "5.50 s"
    assert time_manager.last_response_time == (end - start)


def test_measure_response_subsecond(time_manager):
    """measure_response() handles subsecond durations"""
    start = datetime.now()
    end = start + timedelta(milliseconds=250)

    result = time_manager.measure_response(start, end)

    assert result == "0.25 s"


def test_measure_response_long_duration(time_manager):
    """measure_response() handles long durations"""
    start = datetime.now()
    end = start + timedelta(minutes=2, seconds=30)

    result = time_manager.measure_response(start, end)

    assert result == "150.00 s"


def test_measure_response_zero_duration(time_manager):
    """measure_response() handles zero duration"""
    start = datetime.now()
    end = start

    result = time_manager.measure_response(start, end)

    assert result == "0.00 s"


# =============================================================================
# Last Response Tests
# =============================================================================

def test_last_response_no_previous(time_manager):
    """last_response() returns N/A when no previous response"""
    assert time_manager.last_response() == "N/A"


def test_last_response_with_previous(time_manager):
    """last_response() returns formatted last response time"""
    start = datetime.now()
    end = start + timedelta(seconds=3.5)
    time_manager.measure_response(start, end)

    result = time_manager.last_response()
    assert result == "3.50 s"


def test_last_response_after_multiple_measurements(time_manager):
    """last_response() returns most recent measurement"""
    # First measurement
    start1 = datetime.now()
    end1 = start1 + timedelta(seconds=2)
    time_manager.measure_response(start1, end1)

    # Second measurement
    start2 = datetime.now()
    end2 = start2 + timedelta(seconds=5)
    time_manager.measure_response(start2, end2)

    result = time_manager.last_response()
    assert result == "5.00 s"


# =============================================================================
# File Persistence Integration Tests
# =============================================================================

def test_persistence_roundtrip(temp_time_file):
    """TimeManager can save and reload timestamps"""
    # Create and mark time
    tm1 = TimeManager(time_file=temp_time_file)
    marked_time = tm1.mark_query_time()

    # Create new instance (simulates restart)
    tm2 = TimeManager(time_file=temp_time_file)

    # Should load the same time
    assert tm2.last_query_time is not None
    # Allow 1 second difference due to file I/O timing
    assert abs((tm2.last_query_time - marked_time).total_seconds()) < 1


def test_save_only_when_time_set(time_manager):
    """_save_last_query_time() only saves when last_query_time is set"""
    # Initially None, so save should not create file
    time_manager._save_last_query_time()
    assert not os.path.exists(time_manager.time_file)

    # After marking, file should exist
    time_manager.mark_query_time()
    assert os.path.exists(time_manager.time_file)


def test_multiple_marks_update_file(time_manager):
    """Multiple mark_query_time() calls update the file"""
    time1 = time_manager.mark_query_time()

    # Wait a tiny bit
    import time
    time.sleep(0.01)

    time2 = time_manager.mark_query_time()

    # Reload from file
    with open(time_manager.time_file, "r") as f:
        data = json.load(f)

    saved_time = datetime.fromisoformat(data["last_query_time"])

    # Should be closer to time2 than time1
    assert abs((saved_time - time2).total_seconds()) < abs((saved_time - time1).total_seconds())
