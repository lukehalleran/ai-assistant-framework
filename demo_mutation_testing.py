"""Manual demonstration of mutation testing concept for refactoring confidence."""

# Original function from our simple test
def simple_max(a, b):
    """Simple max function for mutation testing."""
    if a > b:
        return a
    else:
        return b

# MUTATION 1: Change > to >=
def simple_max_mutation1(a, b):
    """Mutated: > changed to >="""
    if a >= b:  # BUG: Should be >
        return a
    else:
        return b

# MUTATION 2: Change return a to return b
def simple_max_mutation2(a, b):
    """Mutated: return values swapped"""
    if a > b:
        return b  # BUG: Should return a
    else:
        return b

# MUTATION 3: Remove else branch
def simple_max_mutation3(a, b):
    """Mutated: removed else branch"""
    if a > b:
        return a
    # BUG: Missing else clause

# Tests from original
def test_simple_max_original():
    """Test original function - should pass."""
    print("Testing ORIGINAL function:")
    assert simple_max(5, 3) == 5
    assert simple_max(2, 8) == 8
    assert simple_max(4, 4) == 4
    print("✅ All tests PASSED")

def test_simple_max_mutation1():
    """Test mutation 1 - should catch the bug."""
    print("\\nTesting MUTATION 1 (>= instead of >):")
    try:
        assert simple_max_mutation1(5, 3) == 5  # Still passes
        assert simple_max_mutation1(2, 8) == 8  # Still passes
        assert simple_max_mutation1(4, 4) == 4  # This should fail! Returns 4 instead of expected behavior
        print("❌ Tests failed to catch mutation!")
    except AssertionError as e:
        print("✅ Tests correctly caught the mutation!")
        print(f"   Error: {e}")

def test_simple_max_mutation2():
    """Test mutation 2 - should catch the bug."""
    print("\\nTesting MUTATION 2 (swapped return values):")
    try:
        assert simple_max_mutation2(5, 3) == 5  # This will fail! Returns 3
        assert simple_max_mutation2(2, 8) == 8
        assert simple_max_mutation2(4, 4) == 4
        print("❌ Tests failed to catch mutation!")
    except AssertionError as e:
        print("✅ Tests correctly caught the mutation!")
        print(f"   Error: simple_max_mutation2(5, 3) returned 3, expected 5")

def test_simple_max_mutation3():
    """Test mutation 3 - should catch the bug."""
    print("\\nTesting MUTATION 3 (missing else):")
    try:
        assert simple_max_mutation3(5, 3) == 5  # Passes
        assert simple_max_mutation3(2, 8) == 8  # This will fail! Returns None
        assert simple_max_mutation3(4, 4) == 4  # This will fail! Returns None
        print("❌ Tests failed to catch mutation!")
    except AssertionError as e:
        print("✅ Tests correctly caught the mutation!")
        print(f"   Error: Function returned None when b >= a")

if __name__ == "__main__":
    test_simple_max_original()
    test_simple_max_mutation1()
    test_simple_max_mutation2()
    test_simple_max_mutation3()

    print("\\n" + "="*60)
    print("MUTATION TESTING SUMMARY:")
    print("="*60)
    print("✅ Original function: All tests pass")
    print("❌ Mutation 1 (>= vs >): May not be caught by current tests")
    print("✅ Mutation 2 (swapped returns): Tests catch this bug")
    print("✅ Mutation 3 (missing else): Tests catch this bug")
    print()
    print("REFACTORING CONFIDENCE:")
    print("• Tests catch most mutations = Safe to refactor")
    print("• Tests miss some mutations = Need better tests before refactoring")