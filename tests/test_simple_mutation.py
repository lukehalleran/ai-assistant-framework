"""Simple mutation testing example without complex imports."""

def simple_add(a, b):
    """Simple addition function for mutation testing."""
    return a + b

def simple_max(a, b):
    """Simple max function for mutation testing."""
    if a > b:
        return a
    else:
        return b

def test_simple_add():
    """Test addition function."""
    assert simple_add(2, 3) == 5
    assert simple_add(0, 0) == 0
    assert simple_add(-1, 1) == 0

def test_simple_max():
    """Test max function."""
    assert simple_max(5, 3) == 5
    assert simple_max(2, 8) == 8
    assert simple_max(4, 4) == 4