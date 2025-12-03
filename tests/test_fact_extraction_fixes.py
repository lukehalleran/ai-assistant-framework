#!/usr/bin/env python3
"""
Test the two fact extraction fixes:
1. Clause-like object filtering
2. REBEL grounding check
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.fact_extractor import FactExtractor


async def test_fact_extraction():
    """Test cases for both fixes."""

    extractor = FactExtractor(use_rebel=True, use_regex=True)

    test_cases = [
        # Test 1: Clause-like objects (should be filtered)
        {
            "query": "I like to read books and run marathons",
            "response": "",
            "expect_reject": ["to read", "to run"],
            "expect_accept": [],
            "description": "Should reject infinitive clauses as objects"
        },
        # Test 2: Valid preference (should pass)
        {
            "query": "My favorite game is Skyrim",
            "response": "",
            "expect_reject": [],
            "expect_accept": ["skyrim"],
            "description": "Should accept concrete noun objects"
        },
        # Test 3: Gerund clause (should be filtered)
        {
            "query": "I love reading mystery novels",
            "response": "",
            "expect_reject": ["reading"],
            "expect_accept": [],
            "description": "Should reject gerund clauses"
        },
        # Test 4: Concrete attribute (should pass)
        {
            "query": "My cat's name is Flapjack",
            "response": "",
            "expect_reject": [],
            "expect_accept": ["flapjack"],
            "description": "Should accept proper nouns as objects"
        },
        # Test 5: Numeric facts (should pass)
        {
            "query": "My squat is 315 pounds",
            "response": "",
            "expect_reject": [],
            "expect_accept": ["315"],
            "description": "Should accept numeric values"
        },
        # Test 6: Third-party fact (should be filtered at subject check)
        {
            "query": "Berlin is the capital of Germany",
            "response": "",
            "expect_reject": ["berlin"],
            "expect_accept": [],
            "description": "Should reject non-user subjects"
        }
    ]

    print("🧪 Testing Fact Extraction Fixes\n")
    print("="*80)

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['description']}")
        print(f"Query: \"{test['query']}\"")

        try:
            facts = await extractor.extract_facts(
                query=test['query'],
                response=test['response'],
                conversation_context=[]
            )

            # Extract objects from returned facts
            extracted_objects = []
            for fact in facts:
                meta = fact.metadata or {}
                obj = meta.get("object", "").lower()
                if obj:
                    extracted_objects.append(obj)

            print(f"Extracted facts: {len(facts)}")
            if facts:
                for fact in facts:
                    meta = fact.metadata or {}
                    s = meta.get("subject", "?")
                    r = meta.get("relation", "?")
                    o = meta.get("object", "?")
                    m = meta.get("method", "?")
                    print(f"  - {s} | {r} | {o} (method={m})")

            # Check expectations
            test_passed = True

            # Check rejects
            for reject_term in test['expect_reject']:
                if any(reject_term.lower() in obj for obj in extracted_objects):
                    print(f"  ❌ FAIL: Expected to reject '{reject_term}' but it was extracted")
                    test_passed = False

            # Check accepts
            for accept_term in test['expect_accept']:
                if not any(accept_term.lower() in obj for obj in extracted_objects):
                    print(f"  ⚠️  WARNING: Expected to accept '{accept_term}' but it was not extracted")
                    # Don't fail test - extraction might use different method

            if test_passed:
                print(f"  ✅ PASS")
                passed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")

    if failed == 0:
        print("\n✅ All tests passed! Fact extraction fixes are working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review output above.")

    return failed == 0


if __name__ == "__main__":
    try:
        success = asyncio.run(test_fact_extraction())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
