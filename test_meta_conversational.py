#!/usr/bin/env python3
"""
Test meta-conversational query detection and retrieval.

This script tests the new meta-conversational query detection system
to ensure queries like "do you recall" are properly handled.
"""

from utils.query_checker import (
    is_meta_conversational,
    analyze_query,
    META_CONVERSATIONAL_MARKERS
)

def test_meta_conversational_detection():
    """Test that meta-conversational queries are properly detected."""

    print("=" * 70)
    print("Testing Meta-Conversational Query Detection")
    print("=" * 70)

    # Test cases from the conversation log
    test_cases = [
        # Meta-conversational queries (should be detected)
        ("But do you recall the last fully off day I told you about?", True),
        ("We discussed this the other day yes?", True),
        ("Do you remember when I mentioned that?", True),
        ("Didn't we talk about this before?", True),
        ("You said something earlier about that", True),
        ("Last time you told me about this", True),
        ("The other day we discussed days off", True),

        # Normal queries (should NOT be detected)
        ("Hey man. I'm oddly fried for having woken up at noon.", False),
        ("Yeah I gotta work 8 hours today", False),
        ("What is the capital of France?", False),
        ("Tell me about Python programming", False),
        ("I want to learn about machine learning", False),
        ("How do I fix this bug?", False),
    ]

    print(f"\nConfigured markers: {META_CONVERSATIONAL_MARKERS}\n")

    passed = 0
    failed = 0

    for query, expected_meta in test_cases:
        is_meta = is_meta_conversational(query)
        analysis = analyze_query(query)

        status = "✓ PASS" if is_meta == expected_meta else "✗ FAIL"

        if is_meta == expected_meta:
            passed += 1
        else:
            failed += 1

        print(f"{status}: {query[:60]:<60}")
        print(f"         Expected: {expected_meta}, Got: {is_meta}")
        print(f"         Analysis: meta={analysis.is_meta_conversational}, "
              f"intents={analysis.intents}")
        print()

    print("-" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 70)

    return failed == 0


def test_query_analysis():
    """Test the full query analysis for meta-conversational queries."""

    print("\n" + "=" * 70)
    print("Testing Full Query Analysis")
    print("=" * 70 + "\n")

    queries = [
        "But do you recall the last fully off day I told you about? Just trying to get a sense",
        "We discussed this the other day yes?",
        "What is machine learning?",
    ]

    for query in queries:
        print(f"Query: {query}")
        analysis = analyze_query(query)

        print(f"  - is_question: {analysis.is_question}")
        print(f"  - is_command: {analysis.is_command}")
        print(f"  - is_deictic: {analysis.is_deictic}")
        print(f"  - is_followup: {analysis.is_followup}")
        print(f"  - is_meta_conversational: {analysis.is_meta_conversational}")
        print(f"  - is_heavy_topic: {analysis.is_heavy_topic}")
        print(f"  - intents: {analysis.intents}")
        print(f"  - tokens: {analysis.tokens}")
        print()


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# Meta-Conversational Query Detection Test Suite")
    print("#" * 70 + "\n")

    # Run detection tests
    detection_passed = test_meta_conversational_detection()

    # Run analysis tests
    test_query_analysis()

    # Summary
    print("\n" + "=" * 70)
    if detection_passed:
        print("✓ All tests PASSED")
    else:
        print("✗ Some tests FAILED - review output above")
    print("=" * 70 + "\n")
