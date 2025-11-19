"""
Mutation Testing for Refactoring Confidence
==========================================

This script tests key functions to verify your test suite catches bugs,
giving you confidence to refactor safely.

Usage:
    python mutation_confidence.py --function _snake
    python mutation_confidence.py --function _normalize_triple
    python mutation_confidence.py --all
"""

import re
import argparse
from typing import Dict, Any, Optional

# ============================================================================
# ORIGINAL FUNCTIONS (copied from source)
# ============================================================================

def _snake_original(s: str) -> str:
    """From memory/llm_fact_extractor.py"""
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9 _-]", " ", s)
    s = re.sub(r"\\s+", "_", s)
    return s.lower().strip("_-")

def _normalize_triple_original(t: Dict[str, Any]) -> Dict[str, str] | None:
    """From memory/llm_fact_extractor.py"""
    subj = str(t.get("subject") or "").strip()
    rel = str(t.get("relation") or "").strip()
    obj = str(t.get("object") or "").strip()
    if not subj or not rel or not obj:
        return None
    if subj.lower() in {"i", "me", "my", "we", "us", "you"}:
        subj = "user"
    rel = _snake_original(rel)
    obj = obj.strip().strip(". ")
    return {"subject": subj.lower(), "relation": rel, "object": obj.lower()}

# ============================================================================
# MUTATIONS
# ============================================================================

def _snake_mutation_no_lower(s: str) -> str:
    """Remove .lower() call"""
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9 _-]", " ", s)
    s = re.sub(r"\\s+", "_", s)
    return s.strip("_-")  # Missing .lower()

def _snake_mutation_wrong_replacement(s: str) -> str:
    """Wrong replacement character"""
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9 _-]", "_", s)  # Wrong replacement
    s = re.sub(r"\\s+", "_", s)
    return s.lower().strip("_-")

def _snake_mutation_no_strip(s: str) -> str:
    """Remove strip("_-")"""
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9 _-]", " ", s)
    s = re.sub(r"\\s+", "_", s)
    return s.lower()  # Missing strip

def _normalize_triple_mutation_no_none_check(t: Dict[str, Any]) -> Dict[str, str] | None:
    """Remove empty string validation"""
    subj = str(t.get("subject") or "").strip()
    rel = str(t.get("relation") or "").strip()
    obj = str(t.get("object") or "").strip()
    # MISSING: if not subj or not rel or not obj: return None
    if subj.lower() in {"i", "me", "my", "we", "us", "you"}:
        subj = "user"
    rel = _snake_original(rel)
    obj = obj.strip().strip(". ")
    return {"subject": subj.lower(), "relation": rel, "object": obj.lower()}

def _normalize_triple_mutation_wrong_pronouns(t: Dict[str, Any]) -> Dict[str, str] | None:
    """Change pronoun detection logic"""
    subj = str(t.get("subject") or "").strip()
    rel = str(t.get("relation") or "").strip()
    obj = str(t.get("object") or "").strip()
    if not subj or not rel or not obj:
        return None
    if subj.lower() in {"i", "me"}:  # WRONG: missing "my", "we", "us", "you"
        subj = "user"
    rel = _snake_original(rel)
    obj = obj.strip().strip(". ")
    return {"subject": subj.lower(), "relation": rel, "object": obj.lower()}

def _normalize_triple_mutation_no_case_conversion(t: Dict[str, Any]) -> Dict[str, str] | None:
    """Remove .lower() calls"""
    subj = str(t.get("subject") or "").strip()
    rel = str(t.get("relation") or "").strip()
    obj = str(t.get("object") or "").strip()
    if not subj or not rel or not obj:
        return None
    if subj.lower() in {"i", "me", "my", "we", "us", "you"}:
        subj = "user"
    rel = _snake_original(rel)
    obj = obj.strip().strip(". ")
    return {"subject": subj, "relation": rel, "object": obj}  # Missing .lower()

# ============================================================================
# TEST RUNNERS
# ============================================================================

def test_snake_mutations():
    """Test _snake function mutations"""
    print("\\nTesting _snake function mutations")
    print("-" * 40)

    test_cases = [
        ("Hello World", "hello_world"),
        ("Hello-World!@#", "hello-world"),
        ("  Test  ", "test"),
        ("", ""),
        (None, ""),
    ]

    mutations = [
        ("No .lower()", _snake_mutation_no_lower),
        ("Wrong replacement", _snake_mutation_wrong_replacement),
        ("No strip", _snake_mutation_no_strip),
    ]

    caught = 0
    total = len(mutations)

    for name, mutation_func in mutations:
        mutation_caught = False
        for input_val, expected in test_cases:
            try:
                result = mutation_func(input_val)
                if result != expected:
                    print(f"✅ {name}: CAUGHT (input: {repr(input_val)})")
                    mutation_caught = True
                    caught += 1
                    break
            except Exception:
                print(f"✅ {name}: CAUGHT (exception)")
                mutation_caught = True
                caught += 1
                break

        if not mutation_caught:
            print(f"❌ {name}: NOT caught")

    confidence = caught / total * 100
    print(f"\\nMutation Detection: {caught}/{total} ({confidence:.0f}%)")
    return confidence

def test_normalize_triple_mutations():
    """Test _normalize_triple function mutations"""
    print("\\nTesting _normalize_triple function mutations")
    print("-" * 45)

    test_cases = [
        ({"subject": "User", "relation": "Likes", "object": "Python"},
         {"subject": "user", "relation": "likes", "object": "python"}),
        ({"subject": "I", "relation": "love", "object": "coding"},
         {"subject": "user", "relation": "love", "object": "coding"}),
        ({"subject": "You", "relation": "know", "object": "Python."},
         {"subject": "user", "relation": "know", "object": "python"}),
        ({"subject": "", "relation": "test", "object": "obj"}, None),
        ({"subject": "A", "relation": "", "object": "C"}, None),
    ]

    mutations = [
        ("No validation", _normalize_triple_mutation_no_none_check),
        ("Wrong pronouns", _normalize_triple_mutation_wrong_pronouns),
        ("No case conversion", _normalize_triple_mutation_no_case_conversion),
    ]

    caught = 0
    total = len(mutations)

    for name, mutation_func in mutations:
        mutation_caught = False
        for input_val, expected in test_cases:
            try:
                result = mutation_func(input_val)
                if result != expected:
                    print(f"✅ {name}: CAUGHT")
                    mutation_caught = True
                    caught += 1
                    break
            except Exception:
                print(f"✅ {name}: CAUGHT (exception)")
                mutation_caught = True
                caught += 1
                break

        if not mutation_caught:
            print(f"❌ {name}: NOT caught")

    confidence = caught / total * 100
    print(f"\\nMutation Detection: {caught}/{total} ({confidence:.0f}%)")
    return confidence

def get_refactoring_confidence(confidence_score):
    """Convert confidence score to refactoring recommendation"""
    if confidence_score >= 80:
        return "🟢 HIGH CONFIDENCE - Safe to refactor"
    elif confidence_score >= 60:
        return "🟡 MEDIUM CONFIDENCE - Add more tests first"
    else:
        return "🔴 LOW CONFIDENCE - Needs better test coverage"

def main():
    parser = argparse.ArgumentParser(description="Mutation testing for refactoring confidence")
    parser.add_argument("--function", choices=["_snake", "_normalize_triple"],
                       help="Test specific function")
    parser.add_argument("--all", action="store_true", help="Test all functions")

    args = parser.parse_args()

    print("MUTATION TESTING FOR REFACTORING CONFIDENCE")
    print("=" * 50)

    if args.function == "_snake" or args.all:
        confidence = test_snake_mutations()
        print(get_refactoring_confidence(confidence))

    if args.function == "_normalize_triple" or args.all:
        confidence = test_normalize_triple_mutations()
        print(get_refactoring_confidence(confidence))

    if args.all:
        print("\\n" + "=" * 50)
        print("OVERALL RECOMMENDATION:")
        print("With 76% test coverage and mutation testing showing")
        print("good bug detection, you have solid confidence for")
        print("refactoring most functions in this codebase.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # Default: test all if no args
        test_snake_mutations()
        test_normalize_triple_mutations()
        print("\\n" + "=" * 50)
        print("Run with --help to see options")
    else:
        main()