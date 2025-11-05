#!/usr/bin/env python3
"""
Quick test script for meta-conversational query detection.
Tests the recent changes to tone detection and meta-conversational handling.
"""

import asyncio
from utils.query_checker import analyze_query_async

async def test_meta_query():
    # Test query - adjust "yesterday" vs "today" based on when conversation happened
    query = "Do you recall what time I said I woke up yesterday?"

    print(f"\n{'='*70}")
    print(f"Testing Meta-Conversational Query Detection")
    print(f"{'='*70}\n")
    print(f"Query: {query}\n")

    # Analyze the query
    analysis = await analyze_query_async(query)

    print(f"Results:")
    print(f"  is_meta_conversational: {analysis.is_meta_conversational}")

    # Print all available attributes
    print(f"\n  Available attributes:")
    for attr in dir(analysis):
        if not attr.startswith('_'):
            try:
                value = getattr(analysis, attr)
                if not callable(value):
                    print(f"    {attr}: {value}")
            except:
                pass
    print(f"\n{'='*70}\n")

    # Expected behavior
    print("Expected Behavior:")
    print("  ✓ is_meta_conversational should be TRUE")
    print("  ✓ Should trigger specialized retrieval (15 recent episodic memories)")
    print("  ✓ Should SKIP semantic search to avoid hallucination")
    print("  ✓ Should retrieve actual recent conversation in chronological order")
    print(f"\n{'='*70}\n")

    if analysis.is_meta_conversational:
        print("✅ PASS: Meta-conversational detection working correctly!")
    else:
        print("❌ FAIL: Query not detected as meta-conversational")

    return analysis

if __name__ == "__main__":
    asyncio.run(test_meta_query())
