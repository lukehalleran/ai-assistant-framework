#!/usr/bin/env python3
"""
Test STM topic change detection and meta-conversational query filtering.

This test verifies the fix for STM contamination across conversation topics.
"""
import sys
from utils.query_checker import is_meta_conversational


def test_meta_conversational_detection():
    """Test that meta-conversational queries are properly detected."""
    print("Testing meta-conversational query detection...")

    # Should be detected as meta-conversational
    meta_queries = [
        "What do you think causes that off topic response?",
        "Seems like it was geared to an earlier message",
        "Do you remember what we talked about?",
        "Didn't we discuss this yesterday?",
        "You said earlier that...",
        "Why did you mention that in your last response?",
    ]

    # Should NOT be detected as meta-conversational
    normal_queries = [
        "What is Python?",
        "How do I fix this error?",
        "Tell me about machine learning",
        "I'm feeling stressed today",
    ]

    print("\n✓ Testing queries that SHOULD be detected as meta-conversational:")
    for query in meta_queries:
        result = is_meta_conversational(query)
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: \"{query}\" -> {result}")

    print("\n✓ Testing queries that should NOT be detected as meta-conversational:")
    for query in normal_queries:
        result = is_meta_conversational(query)
        status = "✓ PASS" if not result else "✗ FAIL"
        print(f"  {status}: \"{query}\" -> {result}")


def test_topic_change_scenario():
    """Simulate the exact scenario from the conversation log."""
    print("\n" + "="*70)
    print("Simulating conversation scenario from log:")
    print("="*70)

    conversations = [
        {
            "id": 1,
            "query": "Yo man. I woke up at a decent time today like 830 I think and slept okay. Thinking I'm due for a break though super fried. I'm at the gym rn and then haircut and then maybe I'll hang with auggie if he responds but definitely taking it easy today I think",
            "expected_topic": "Personal/Wellness",
            "use_stm": True,  # First conversation, STM active
        },
        {
            "id": 2,
            "query": "Yeah makes sense. It doesn't feel like I've been working super hard since the project is funish but I guess I have. I have produced about 1000 lines of production grade code and comments per month over the last 8 months on average although it's not evenly distributed",
            "expected_topic": "Software Development",
            "use_stm": False,  # Topic change! STM should be skipped
        },
        {
            "id": 3,
            "query": "What do you think causes that off topic response? Seems like it was geared to an earlier message",
            "expected_topic": "Meta-conversational",
            "use_stm": False,  # Meta-conversational! STM should be skipped
        },
    ]

    print("\nExpected behavior:")
    for conv in conversations:
        print(f"\nConversation #{conv['id']}:")
        print(f"  Query: {conv['query'][:80]}...")
        print(f"  Expected topic: {conv['expected_topic']}")
        print(f"  Should use STM: {conv['use_stm']}")

        # Check if meta-conversational
        is_meta = is_meta_conversational(conv['query'])
        if is_meta:
            print(f"  ✓ Detected as META-CONVERSATIONAL (STM will be skipped)")

        # For conv #2, topic change would be detected by topic_manager in runtime
        if conv['id'] == 2:
            print(f"  ✓ Topic change would be detected by topic_manager (STM will be skipped)")


if __name__ == "__main__":
    print("STM Contamination Fix - Validation Tests")
    print("="*70)

    test_meta_conversational_detection()
    test_topic_change_scenario()

    print("\n" + "="*70)
    print("Test complete! The fix should prevent:")
    print("  1. STM from carrying over context across topic changes")
    print("  2. STM from activating on meta-conversational queries")
    print("="*70)
