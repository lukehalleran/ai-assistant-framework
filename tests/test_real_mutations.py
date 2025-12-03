"""Manual mutation testing on real project functions."""
import re

# ORIGINAL FUNCTION (copy from memory/llm_fact_extractor.py)
def _snake_original(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9 _-]", " ", s)
    s = re.sub(r"\s+", "_", s)
    return s.lower().strip("_-")

# MUTATION 1: Remove .lower()
def _snake_mutation1(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9 _-]", " ", s)
    s = re.sub(r"\s+", "_", s)
    return s.strip("_-")  # BUG: Missing .lower()

# MUTATION 2: Change " " to "_" in first regex
def _snake_mutation2(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9 _-]", "_", s)  # BUG: Changed " " to "_"
    s = re.sub(r"\s+", "_", s)
    return s.lower().strip("_-")

# MUTATION 3: Remove the strip("_-")
def _snake_mutation3(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9 _-]", " ", s)
    s = re.sub(r"\s+", "_", s)
    return s.lower()  # BUG: Missing strip("_-")

# MUTATION 4: Change (s or "") to just s
def _snake_mutation4(s: str) -> str:
    s = s.strip()  # BUG: Will fail on None input
    s = re.sub(r"[^A-Za-z0-9 _-]", " ", s)
    s = re.sub(r"\s+", "_", s)
    return s.lower().strip("_-")

def test_snake_mutations():
    """Test that our existing tests catch mutations in _snake function."""

    # Test cases from our comprehensive test suite
    test_cases = [
        ("Hello World", "hello_world"),
        ("Hello-World!@#", "hello-world"),
        ("Hello   World", "hello_world"),
        ("", ""),
        (None, ""),
        ("  Hello World  ", "hello_world"),
    ]

    print("TESTING _snake FUNCTION MUTATIONS")
    print("="*50)

    # Test original
    print("\\n1. Testing ORIGINAL function:")
    all_passed = True
    for input_val, expected in test_cases:
        try:
            result = _snake_original(input_val)
            assert result == expected, f"Input: {repr(input_val)}, Expected: {expected}, Got: {result}"
        except Exception as e:
            print(f"❌ Failed on {repr(input_val)}: {e}")
            all_passed = False

    if all_passed:
        print("✅ Original function: All tests PASSED")

    # Test mutation 1 (missing .lower())
    print("\\n2. Testing MUTATION 1 (missing .lower()):")
    caught_mutation1 = False
    for input_val, expected in test_cases:
        try:
            result = _snake_mutation1(input_val)
            if result != expected:
                print(f"✅ Mutation caught! Input: {repr(input_val)}")
                print(f"   Expected: {expected}, Got: {result}")
                caught_mutation1 = True
                break
        except:
            pass

    if not caught_mutation1:
        print("❌ Tests failed to catch mutation 1!")

    # Test mutation 2 (wrong replacement character)
    print("\\n3. Testing MUTATION 2 (wrong replacement char):")
    caught_mutation2 = False
    for input_val, expected in test_cases:
        try:
            result = _snake_mutation2(input_val)
            if result != expected:
                print(f"✅ Mutation caught! Input: {repr(input_val)}")
                print(f"   Expected: {expected}, Got: {result}")
                caught_mutation2 = True
                break
        except:
            pass

    if not caught_mutation2:
        print("❌ Tests failed to catch mutation 2!")

    # Test mutation 3 (missing strip)
    print("\\n4. Testing MUTATION 3 (missing strip):")
    caught_mutation3 = False
    for input_val, expected in test_cases:
        try:
            result = _snake_mutation3(input_val)
            if result != expected:
                print(f"✅ Mutation caught! Input: {repr(input_val)}")
                print(f"   Expected: {expected}, Got: {result}")
                caught_mutation3 = True
                break
        except:
            pass

    if not caught_mutation3:
        print("❌ Tests failed to catch mutation 3!")

    # Test mutation 4 (None handling)
    print("\\n5. Testing MUTATION 4 (None handling):")
    caught_mutation4 = False
    try:
        result = _snake_mutation4(None)
        print("❌ Mutation 4 not caught - should have failed on None!")
    except AttributeError:
        print("✅ Mutation 4 caught! None input caused AttributeError")
        caught_mutation4 = True
    except Exception as e:
        print(f"✅ Mutation 4 caught! Exception: {e}")
        caught_mutation4 = True

    print("\\n" + "="*50)
    print("MUTATION TESTING RESULTS:")
    print("="*50)
    mutations_caught = sum([caught_mutation1, caught_mutation2, caught_mutation3, caught_mutation4])
    print(f"Mutations caught: {mutations_caught}/4 ({mutations_caught/4*100:.0f}%)")

    if mutations_caught >= 3:
        print("✅ HIGH CONFIDENCE: Safe to refactor _snake function")
        print("   Your tests catch most potential bugs")
    elif mutations_caught >= 2:
        print("⚠️  MEDIUM CONFIDENCE: Consider adding more tests")
        print("   Some mutations escaped detection")
    else:
        print("❌ LOW CONFIDENCE: Add more tests before refactoring")
        print("   Many mutations went undetected")

if __name__ == "__main__":
    test_snake_mutations()