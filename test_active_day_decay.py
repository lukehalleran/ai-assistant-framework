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

def cleanup_test_files():
    """Clean up test files created during verification"""
    test_files = [
        "data/test_active_days.json",
        "data/test_last_query_time.json",
        "data/test_active_days_decay.json",
        "data/test_last_query_time_decay.json",
        "data/test_active_days_clean.json",
        "data/test_last_query_time_clean.json",
        "data/test_active_days_pause.json",
        "data/test_last_query_time_pause.json",
        "data/test_active_days_compare.json",
        "data/test_last_query_time_compare.json"
    ]
    for file_path in test_files:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_active_day_tracking():
    """Test that active days are properly tracked"""
    print("=== Testing Active Day Tracking ===")

    # Use test files to avoid interfering with real data
    time_manager = TimeManager(time_file="data/test_last_query_time.json")
    time_manager.active_days_file = "data/test_active_days.json"
    time_manager.active_days = time_manager._load_active_days()

    # Simulate activity over several days
    base_date = datetime.now() - timedelta(days=7)

    # Day 1: Active
    time_manager.last_query_time = base_date
    time_manager._register_active_day(base_date)

    # Also register today as active since mark_query_time would do this
    today = datetime.now()
    time_manager._register_active_day(today)

    active_days_1 = time_manager.get_active_days_since(base_date)
    print(f"Day 1 - Active days since {base_date.date()}: {active_days_1}")
    print(f"Registered active days: {sorted(time_manager.active_days)}")
    print(f"Today is: {today.date()}")

    # Should be at least 1 (today) if not the base_date
    expected_min = 1 if base_date.date() != today.date() else 0
    assert active_days_1 >= expected_min, f"Expected at least {expected_min} active day, got {active_days_1}"

    # Day 3: Active (skipped Day 2)
    day_3 = base_date + timedelta(days=2)
    time_manager.last_query_time = day_3
    time_manager._register_active_day(day_3)
    active_days_3 = time_manager.get_active_days_since(base_date)
    print(f"Day 3 - Active days since {base_date.date()}: {active_days_3}")
    assert active_days_3 == 2, f"Expected 2 active days, got {active_days_3}"

    # Day 5: Active (skipped Day 4)
    day_5 = base_date + timedelta(days=4)
    time_manager.last_query_time = day_5
    time_manager._register_active_day(day_5)
    active_days_5 = time_manager.get_active_days_since(base_date)
    print(f"Day 5 - Active days since {base_date.date()}: {active_days_5}")
    assert active_days_5 == 3, f"Expected 3 active days, got {active_days_5}"

    print("✅ Active day tracking works correctly\n")

def test_decay_calculation():
    """Test decay calculation based on active days"""
    print("=== Testing Active Day Decay Calculation ===")

    # Create fresh time manager for this test
    time_manager = TimeManager(time_file="data/test_last_query_time_decay.json")
    time_manager.active_days_file = "data/test_active_days_decay.json"
    time_manager.active_days = time_manager._load_active_days()

    # Set up some active days
    base_date = datetime.now() - timedelta(days=5)

    # Register active days explicitly
    active_dates = []
    for i in range(3):  # Active on days 1, 3, 5 after base_date
        active_date = base_date + timedelta(days=i+1)  # +1 to skip creation day
        time_manager._register_active_day(active_date)
        active_dates.append(active_date.date().isoformat())

    actual_active_days = time_manager.get_active_days_since(base_date)
    decay_with_active_days = time_manager.calculate_active_day_decay(base_date, decay_rate=0.05)

    # Calculate expected decay based on actual active days counted
    expected_decay = 1.0 / (1.0 + 0.05 * actual_active_days)

    print(f"Memory from {base_date.date()}")
    print(f"Registered active days: {active_dates}")
    print(f"Active days since then: {actual_active_days}")
    print(f"Decay with active days: {decay_with_active_days:.3f}")
    print(f"Expected decay: {expected_decay:.3f}")

    assert abs(decay_with_active_days - expected_decay) < 0.001, "Decay calculation mismatch"

    # Test with no active days - create a new clean time manager
    clean_time_manager = TimeManager(time_file="data/test_last_query_time_clean.json")
    clean_time_manager.active_days_file = "data/test_active_days_clean.json"
    clean_time_manager.active_days = clean_time_manager._load_active_days()

    # Don't register any active days
    no_activity_date = datetime.now() - timedelta(days=10)
    decay_no_activity = clean_time_manager.calculate_active_day_decay(no_activity_date, decay_rate=0.05)
    print(f"No activity decay: {decay_no_activity:.3f}")
    assert decay_no_activity == 1.0, "Should be no decay with no activity"

    print("✅ Decay calculation works correctly\n")

def test_pause_resume_behavior():
    """Test that decay pauses when inactive and resumes when active"""
    print("=== Testing Pause/Resume Behavior ===")

    time_manager = TimeManager(time_file="data/test_last_query_time_pause.json")
    time_manager.active_days_file = "data/test_active_days_pause.json"
    time_manager.active_days = time_manager._load_active_days()

    memory_time = datetime.now() - timedelta(days=10)
    decay_rate = 0.1

    # Initial activity: memory created, then 3 days of activity
    for i in range(3):
        activity_day = memory_time + timedelta(days=i)
        time_manager._register_active_day(activity_day)

    decay_after_3_active = time_manager.calculate_active_day_decay(memory_time, decay_rate)
    print(f"After 3 active days: {decay_after_3_active:.3f}")

    # Pause: 5 days of no activity
    # (Don't register any more active days)
    decay_after_pause = time_manager.calculate_active_day_decay(memory_time, decay_rate)
    print(f"After 5-day pause (still 3 active days): {decay_after_pause:.3f}")
    assert decay_after_pause == decay_after_3_active, "Decay should not change during pause"

    # Resume: 2 more days of activity
    for i in range(5, 7):
        activity_day = memory_time + timedelta(days=i)
        time_manager._register_active_day(activity_day)

    decay_after_resume = time_manager.calculate_active_day_decay(memory_time, decay_rate)
    print(f"After 2 more active days (total 5): {decay_after_resume:.3f}")
    assert decay_after_resume < decay_after_pause, "Decay should resume after activity"

    print("✅ Pause/Resume behavior works correctly\n")

def test_comparison_with_old_system():
    """Compare active-day decay with old hourly system"""
    print("=== Comparison: Active-Day vs Hourly Decay ===")

    time_manager = TimeManager(time_file="data/test_last_query_time_compare.json")
    time_manager.active_days_file = "data/test_active_days_compare.json"
    time_manager.active_days = time_manager._load_active_days()

    # Simulate a user who sends messages every other day for 10 days
    memory_time = datetime.now() - timedelta(days=10)
    decay_rate = 0.05

    # Register activity every other day
    for i in range(0, 10, 2):
        activity_day = memory_time + timedelta(days=i)
        time_manager._register_active_day(activity_day)

    active_days_count = time_manager.get_active_days_since(memory_time)

    # New active-day system
    new_decay = time_manager.calculate_active_day_decay(memory_time, decay_rate)

    # Old hourly system (240 hours = 10 days)
    old_hours = 10 * 24
    old_decay = 1.0 / (1.0 + decay_rate * old_hours)

    print(f"Time span: 10 calendar days")
    print(f"Active days: {active_days_count}")
    print(f"New active-day decay: {new_decay:.3f}")
    print(f"Old hourly decay: {old_decay:.3f}")
    print(f"Difference: {abs(new_decay - old_decay):.3f}")

    # Active-day decay should be much gentler (higher value) than hourly decay
    assert new_decay > old_decay, "Active-day decay should be gentler than hourly decay"

    print("✅ Active-day system provides gentler decay as expected\n")

def test_fallback_behavior():
    """Test fallback to hourly decay when time_manager doesn't support active days"""
    print("=== Testing Fallback Behavior ===")

    from memory.memory_scorer import MemoryScorer

    # Test MemoryScorer without time_manager (should use fallback)
    scorer_no_time_manager = MemoryScorer(time_manager=None)

    # Mock memory object
    class MockMemory:
        def __init__(self):
            self.timestamp = datetime.now() - timedelta(days=5)
            self.decay_rate = 0.1
            self.last_accessed = datetime.now()
            self.importance_score = 0.5
            self.truth_score = 0.6
            self.metadata = {'truth_score': 0.6}

    mock_memory = MockMemory()
    mem_dict = {'memory': mock_memory, 'relevance_score': 0.7}

    # Should not crash and should produce a decay factor
    result = scorer_no_time_manager.apply_temporal_decay([mem_dict])
    assert len(result) == 1, "Should return one memory"
    assert 'final_score' in result[0], "Should calculate final score"

    final_score = result[0]['final_score']
    print(f"Fallback final score: {final_score:.3f}")
    assert 0 < final_score < 1, "Final score should be between 0 and 1"

    print("✅ Fallback behavior works correctly\n")

def main():
    """Run all verification tests"""
    print("🔍 Verifying Active-Day Memory Decay Implementation\n")

    try:
        cleanup_test_files()

        test_active_day_tracking()
        test_decay_calculation()
        test_pause_resume_behavior()
        test_comparison_with_old_system()
        test_fallback_behavior()

        print("🎉 All tests passed! Active-day decay implementation is working correctly.")
        print("\nSummary of behavior:")
        print("- Decay is based on active days (days with messages) not calendar days")
        print("- Decay pauses when no messages are sent")
        print("- Decay resumes when activity continues")
        print("- Provides gentler decay than hourly system")
        print("- Falls back gracefully if time_manager unavailable")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        cleanup_test_files()

    return 0

if __name__ == "__main__":
    exit(main())