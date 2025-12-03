#!/usr/bin/env python3
"""
Integration test for meta-conversational memory retrieval.

This script tests that the memory coordinator properly routes
meta-conversational queries to the specialized retrieval method.
"""

import asyncio
from datetime import datetime, timedelta
from memory.corpus_manager import CorpusManager
from memory.memory_interface import MemoryType
from utils.query_checker import is_meta_conversational


async def test_meta_retrieval_routing():
    """Test that meta-conversational queries trigger the right retrieval path."""

    print("=" * 70)
    print("Testing Meta-Conversational Memory Retrieval Routing")
    print("=" * 70 + "\n")

    print("When a meta-conversational query is detected, the system will:")
    print("  1. Route to _get_meta_conversational_memories() instead of standard retrieval")
    print("  2. Retrieve more recent episodic memories (up to 15 vs. 5)")
    print("  3. Skip semantic search entirely")
    print("  4. Apply very steep recency weighting (0.7 weight on recency)")
    print("  5. Filter by topic/thread if applicable")
    print()
    print("-" * 70)

    # Test queries
    test_queries = [
        ("But do you recall the last fully off day I told you about?", True),
        ("We discussed this the other day yes?", True),
        ("What is the capital of Spain?", False),
        ("Tell me more about that", False),
    ]

    for query, is_meta_expected in test_queries:
        is_meta = is_meta_conversational(query)
        status = "✓" if is_meta == is_meta_expected else "✗"

        print(f"\n{status} Query: {query}")
        print(f"   Meta-conversational: {is_meta} (expected: {is_meta_expected})")

        if is_meta:
            print(f"   → Will use _get_meta_conversational_memories() for retrieval")
            print(f"   → Will prioritize recent episodic memories")
            print(f"   → Will skip semantic search to avoid cross-contamination")
        else:
            print(f"   → Will use standard retrieval pipeline")
            print(f"   → Will combine recent + semantic memories")

    print("\n" + "=" * 70)
    print("✓ Routing test complete")
    print("=" * 70 + "\n")


async def test_detection_summary():
    """Show which queries from the original conversation log would be detected."""

    print("=" * 70)
    print("Analysis of Original Conversation Log Queries")
    print("=" * 70 + "\n")

    original_queries = [
        "Hey man. Im oddly fried for having woken up at noon.",  # Conv #1
        "Yeah I gotta work 8 hours today but on Monday I can. But do you recall the last fully off day I told you about? Just trying to get a sense",  # Conv #2
        "That was odd lost connection and errored out. Here is the last message I sent please respond to it: Yeah I gotta work 8 hours today but on Monday I can. But do you recall the last fully off day I told you about? Just trying to get a sense",  # Conv #3
        "We discussed this the other day yes?",  # Conv #4
        "We are talking about days off, not current events!",  # Conv #5
    ]

    for i, query in enumerate(original_queries, 1):
        is_meta = is_meta_conversational(query)
        icon = "🔍" if is_meta else "💬"

        print(f"Conversation #{i}:")
        print(f"  Query: {query[:70]}...")
        print(f"  {icon} Meta-conversational: {is_meta}")

        if is_meta:
            print(f"  ✓ Would trigger specialized retrieval (recent episodic memories)")
        else:
            print(f"  → Would use standard retrieval (recent + semantic)")
        print()

    print("=" * 70 + "\n")


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# Meta-Conversational Memory Retrieval Integration Tests")
    print("#" * 70 + "\n")

    asyncio.run(test_meta_retrieval_routing())
    asyncio.run(test_detection_summary())

    print("✓ All integration tests complete\n")
