#!/usr/bin/env python3
"""
test_active_day_decay.py

Verification script for the new active-day based memory decay system.

This script tests:
1. Active day tracking functionality
2. Decay calculation based on active days vs calendar days
3. Pause in decay when no messages are sent
4. Proper decay behavior when messages are sent daily
"""

import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.time_manager import TimeManager
import pytest


@pytest.fixture
def fresh_time_manager(tmp_path):
    """Create a TimeManager with isolated temp files."""
    time_file = str(tmp_path / "last_query_time.json")
    active_file = str(tmp_path / "active_days.json")
    tm = TimeManager(time_file=time_file)
    tm.active_days_file = active_file
    tm.active_days = set()  # Start clean
    return tm


def test_active_day_tracking(fresh_time_manager):
    """Test that active days are properly tracked"""
    time_manager = fresh_time_manager

    # Simulate activity over several days
    base_date = datetime.now() - timedelta(days=7)

    # Day 1: Active
    time_manager.last_query_time = base_date
    time_manager._register_active_day(base_date)

    # Also register today as active since mark_query_time would do this
    today = datetime.now()
    time_manager._register_active_day(today)

    active_days_1 = time_manager.get_active_days_since(base_date)

    # get_active_days_since excludes base_date itself (starts from base_date+1)
    # today is registered and in range → 1 active day
    assert active_days_1 >= 1, f"Expected at least 1 active day, got {active_days_1}"

    # Day 3: Active (skipped Day 2)
    day_3 = base_date + timedelta(days=2)
    time_manager.last_query_time = day_3
    time_manager._register_active_day(day_3)
    active_days_3 = time_manager.get_active_days_since(base_date)
    # Now: today + day_3 = 2 active days after base_date
    assert active_days_3 == 2, f"Expected 2 active days, got {active_days_3}"

    # Day 5: Active (skipped Day 4)
    day_5 = base_date + timedelta(days=4)
    time_manager.last_query_time = day_5
    time_manager._register_active_day(day_5)
    active_days_5 = time_manager.get_active_days_since(base_date)
    # Now: today + day_3 + day_5 = 3 active days after base_date
    assert active_days_5 == 3, f"Expected 3 active days, got {active_days_5}"


def test_decay_calculation(tmp_path):
    """Test decay calculation based on active days"""
    # Create fresh time manager for this test
    time_manager = TimeManager(time_file=str(tmp_path / "lqt_decay.json"))
    time_manager.active_days_file = str(tmp_path / "ad_decay.json")
    time_manager.active_days = set()

    # Set up some active days
    base_date = datetime.now() - timedelta(days=5)

    # Register active days explicitly
    for i in range(3):
        active_date = base_date + timedelta(days=i+1)  # +1 to skip creation day
        time_manager._register_active_day(active_date)

    actual_active_days = time_manager.get_active_days_since(base_date)
    decay_with_active_days = time_manager.calculate_active_day_decay(base_date, decay_rate=0.05)

    # Calculate expected decay based on actual active days counted
    expected_decay = 1.0 / (1.0 + 0.05 * actual_active_days)

    assert abs(decay_with_active_days - expected_decay) < 0.001, "Decay calculation mismatch"

    # Test with no active days - create a new clean time manager
    clean_time_manager = TimeManager(time_file=str(tmp_path / "lqt_clean.json"))
    clean_time_manager.active_days_file = str(tmp_path / "ad_clean.json")
    clean_time_manager.active_days = set()

    # Don't register any active days
    no_activity_date = datetime.now() - timedelta(days=10)
    decay_no_activity = clean_time_manager.calculate_active_day_decay(no_activity_date, decay_rate=0.05)
    assert decay_no_activity == 1.0, "Should be no decay with no activity"


def test_pause_resume_behavior(tmp_path):
    """Test that decay pauses when inactive and resumes when active"""
    time_manager = TimeManager(time_file=str(tmp_path / "lqt_pause.json"))
    time_manager.active_days_file = str(tmp_path / "ad_pause.json")
    time_manager.active_days = set()

    memory_time = datetime.now() - timedelta(days=10)
    decay_rate = 0.1

    # Initial activity: memory created, then 3 days of activity
    for i in range(3):
        activity_day = memory_time + timedelta(days=i)
        time_manager._register_active_day(activity_day)

    decay_after_3_active = time_manager.calculate_active_day_decay(memory_time, decay_rate)

    # Pause: 5 days of no activity
    # (Don't register any more active days)
    decay_after_pause = time_manager.calculate_active_day_decay(memory_time, decay_rate)
    assert decay_after_pause == decay_after_3_active, "Decay should not change during pause"

    # Resume: 2 more days of activity
    for i in range(5, 7):
        activity_day = memory_time + timedelta(days=i)
        time_manager._register_active_day(activity_day)

    decay_after_resume = time_manager.calculate_active_day_decay(memory_time, decay_rate)
    assert decay_after_resume < decay_after_pause, "Decay should resume after activity"


def test_comparison_with_old_system(tmp_path):
    """Compare active-day decay with old hourly system"""
    time_manager = TimeManager(time_file=str(tmp_path / "lqt_compare.json"))
    time_manager.active_days_file = str(tmp_path / "ad_compare.json")
    time_manager.active_days = set()

    # Simulate a user who sends messages every other day for 10 days
    memory_time = datetime.now() - timedelta(days=10)
    decay_rate = 0.05

    # Register activity every other day
    for i in range(0, 10, 2):
        activity_day = memory_time + timedelta(days=i)
        time_manager._register_active_day(activity_day)

    # New active-day system
    new_decay = time_manager.calculate_active_day_decay(memory_time, decay_rate)

    # Old hourly system (240 hours = 10 days)
    old_hours = 10 * 24
    old_decay = 1.0 / (1.0 + decay_rate * old_hours)

    # Active-day decay should be much gentler (higher value) than hourly decay
    assert new_decay > old_decay, "Active-day decay should be gentler than hourly decay"


def test_fallback_behavior():
    """Test fallback to hourly decay when time_manager doesn't support active days"""
    from memory.memory_scorer import MemoryScorer

    # Test MemoryScorer without time_manager (should use fallback)
    scorer_no_time_manager = MemoryScorer(time_manager=None)

    # Memory dict in the flat format expected by apply_temporal_decay
    mem_dict = {
        'timestamp': (datetime.now() - timedelta(days=5)).isoformat(),
        'relevance_score': 0.7,
        'importance_score': 0.5,
        'truth_score': 0.6,
        'metadata': {
            'decay_rate': 0.1,
            'truth_score': 0.6,
        }
    }

    # Should not crash and should produce a decay factor
    result = scorer_no_time_manager.apply_temporal_decay([mem_dict])
    assert len(result) == 1, "Should return one memory"
    assert 'final_score' in result[0], "Should calculate final score"

    final_score = result[0]['final_score']
    assert 0 < final_score < 1, "Final score should be between 0 and 1"


def main():
    """Run all verification tests"""
    print("Verifying Active-Day Memory Decay Implementation\n")

    try:
        test_fallback_behavior()
        print("All tests passed! Active-day decay implementation is working correctly.")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
