"""
Test script for temporal window detection in meta-conversational queries.

This tests the hybrid retrieval approach (Option C) that dynamically adjusts
the retrieval window based on temporal markers in the query.
"""
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.query_checker import extract_temporal_window, is_meta_conversational

# Test queries with various temporal markers
test_queries = [
    # Recent (1-2 days)
    ("Do you recall what I said yesterday?", 1),
    ("What did we discuss last night?", 1),
    ("Remember this morning when I mentioned...?", 1),
    ("couple days ago you told me", 2),

    # Medium range (3-7 days)
    ("What did we talk about a few days ago?", 3),
    ("Do you remember last week when...?", 7),
    ("We discussed this last Tuesday", 7),

    # Long range (8+ days)
    ("Do you recall what I said a few weeks ago?", 14),
    ("What did we discuss last month?", 30),
    ("A while back you mentioned something...", 14),
    ("Long time ago we talked about...", 30),

    # Explicit dates
    ("Do you recall what I said on Nov 1st?", 30),
    ("What did we discuss on October 15th?", 30),

    # Numeric days
    ("What did I say 5 days ago?", 5),
    ("10 days ago we discussed...", 10),

    # No temporal marker (should use default)
    ("Do you recall what I said about sleep?", 0),
    ("What did we discuss about work?", 0),
]

print("=" * 80)
print("TEMPORAL WINDOW DETECTION TEST")
print("=" * 80)
print()

for query, expected_days in test_queries:
    # Check if meta-conversational
    is_meta = is_meta_conversational(query)

    # Extract temporal window
    detected_days = extract_temporal_window(query)

    # Determine retrieval parameters based on temporal window
    if detected_days == 0:
        retrieval = "50 memories (default 3-5 days)"
        multiplier = "3x"
    elif detected_days <= 2:
        retrieval = "30 memories (1-2 days)"
        multiplier = "2x"
    elif detected_days <= 7:
        retrieval = "80 memories (3-7 days)"
        multiplier = "4x"
    else:
        retrieval = "150 memories (8+ days)"
        multiplier = "6x"

    status = "✅" if detected_days == expected_days else "❌"

    print(f"{status} Query: {query}")
    print(f"   Meta-conversational: {is_meta}")
    print(f"   Expected: {expected_days} days")
    print(f"   Detected: {detected_days} days")
    print(f"   Retrieval: {retrieval} (return {multiplier})")
    print()

print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
