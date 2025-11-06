"""
Unit tests for core/prompt.py helper functions

Tests pure utility functions:
- Boolean parsing
- Summary dict creation
- Deduplication with order preservation
- List truncation
- Configuration parsing
"""

import pytest
from core.prompt import (
    _parse_bool,
    _as_summary_dict,
    _dedupe_keep_order,
    _truncate_list,
    _cfg_int,
)


# =============================================================================
# _parse_bool Tests
# =============================================================================

def test_parse_bool_true_values():
    """Parses various true values"""
    assert _parse_bool("1") == True
    assert _parse_bool("true") == True
    assert _parse_bool("True") == True
    assert _parse_bool("TRUE") == True
    assert _parse_bool("t") == True
    assert _parse_bool("yes") == True
    assert _parse_bool("y") == True
    assert _parse_bool("on") == True


def test_parse_bool_false_values():
    """Parses various false values"""
    assert _parse_bool("0") == False
    assert _parse_bool("false") == False
    assert _parse_bool("False") == False
    assert _parse_bool("FALSE") == False
    assert _parse_bool("f") == False
    assert _parse_bool("no") == False
    assert _parse_bool("n") == False
    assert _parse_bool("off") == False


def test_parse_bool_none():
    """None returns default"""
    assert _parse_bool(None, default=False) == False
    assert _parse_bool(None, default=True) == True


def test_parse_bool_invalid_returns_default():
    """Invalid values return default"""
    assert _parse_bool("invalid", default=False) == False
    assert _parse_bool("invalid", default=True) == True
    assert _parse_bool("maybe", default=False) == False


def test_parse_bool_whitespace():
    """Handles whitespace"""
    assert _parse_bool("  true  ") == True
    assert _parse_bool("  false  ") == False


def test_parse_bool_empty_string():
    """Empty string returns default"""
    assert _parse_bool("", default=False) == False
    assert _parse_bool("", default=True) == True


def test_parse_bool_case_insensitive():
    """Case insensitive parsing"""
    assert _parse_bool("TrUe") == True
    assert _parse_bool("FaLsE") == False
    assert _parse_bool("YES") == True
    assert _parse_bool("NO") == False


# =============================================================================
# _as_summary_dict Tests
# =============================================================================

def test_as_summary_dict_basic():
    """Creates summary dict with required fields"""
    result = _as_summary_dict("Summary text", ["tag1"], "test_source")

    assert result["content"] == "Summary text"
    assert result["type"] == "summary"
    assert result["tags"] == ["tag1"]
    assert result["source"] == "test_source"


def test_as_summary_dict_with_timestamp():
    """Includes timestamp when provided"""
    result = _as_summary_dict("Text", [], "source", timestamp="2024-01-01T12:00:00")

    assert result["timestamp"] == "2024-01-01T12:00:00"


def test_as_summary_dict_without_timestamp():
    """Excludes timestamp when not provided"""
    result = _as_summary_dict("Text", [], "source")

    assert "timestamp" not in result


def test_as_summary_dict_empty_tags():
    """Handles empty tags list"""
    result = _as_summary_dict("Text", [], "source")

    assert result["tags"] == []


def test_as_summary_dict_multiple_tags():
    """Handles multiple tags"""
    result = _as_summary_dict("Text", ["tag1", "tag2", "tag3"], "source")

    assert result["tags"] == ["tag1", "tag2", "tag3"]


def test_as_summary_dict_empty_content():
    """Handles empty content"""
    result = _as_summary_dict("", ["tag"], "source")

    assert result["content"] == ""


# =============================================================================
# _dedupe_keep_order Tests
# =============================================================================

def test_dedupe_keep_order_basic():
    """Removes duplicates while preserving order"""
    items = ["apple", "banana", "apple", "cherry", "banana"]
    result = _dedupe_keep_order(items)

    assert result == ["apple", "banana", "cherry"]


def test_dedupe_keep_order_case_insensitive():
    """Deduplication is case insensitive by default"""
    items = ["Apple", "APPLE", "apple", "Banana"]
    result = _dedupe_keep_order(items)

    # Should keep first occurrence
    assert len(result) == 2
    assert result[0] == "Apple"
    assert result[1] == "Banana"


def test_dedupe_keep_order_with_whitespace():
    """Handles whitespace in items"""
    items = ["  apple  ", "apple", " apple ", "banana"]
    result = _dedupe_keep_order(items)

    # Should treat all as same due to strip()
    assert len(result) == 2


def test_dedupe_keep_order_empty_list():
    """Handles empty list"""
    result = _dedupe_keep_order([])

    assert result == []


def test_dedupe_keep_order_none():
    """Handles None input"""
    result = _dedupe_keep_order(None)

    assert result == []


def test_dedupe_keep_order_single_item():
    """Handles single item"""
    result = _dedupe_keep_order(["apple"])

    assert result == ["apple"]


def test_dedupe_keep_order_no_duplicates():
    """Handles list with no duplicates"""
    items = ["apple", "banana", "cherry"]
    result = _dedupe_keep_order(items)

    assert result == items


def test_dedupe_keep_order_all_duplicates():
    """Handles all duplicates"""
    items = ["apple", "apple", "apple"]
    result = _dedupe_keep_order(items)

    assert result == ["apple"]


def test_dedupe_keep_order_custom_key_fn():
    """Works with custom key function"""
    items = [{"name": "Alice"}, {"name": "Bob"}, {"name": "alice"}]
    result = _dedupe_keep_order(items, key_fn=lambda x: x["name"].lower())

    assert len(result) == 2
    assert result[0]["name"] == "Alice"
    assert result[1]["name"] == "Bob"


def test_dedupe_keep_order_filters_empty_keys():
    """Filters items with empty keys"""
    items = ["apple", "", "banana", "  ", "cherry"]
    result = _dedupe_keep_order(items)

    # Empty/whitespace-only items should be filtered
    assert "" not in result
    assert "apple" in result


# =============================================================================
# _truncate_list Tests
# =============================================================================

def test_truncate_list_within_limit():
    """Returns original list if within limit"""
    items = [1, 2, 3]
    result = _truncate_list(items, limit=5)

    assert result == items
    assert result is items  # Same object


def test_truncate_list_exceeds_limit():
    """Truncates list exceeding limit"""
    items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = _truncate_list(items, limit=5)

    assert result == [1, 2, 3, 4, 5]


def test_truncate_list_exact_limit():
    """Handles list exactly at limit"""
    items = [1, 2, 3, 4, 5]
    result = _truncate_list(items, limit=5)

    assert result == items
    assert result is items


def test_truncate_list_empty():
    """Handles empty list"""
    result = _truncate_list([], limit=5)

    assert result == []


def test_truncate_list_zero_limit():
    """Handles zero limit"""
    items = [1, 2, 3]
    result = _truncate_list(items, limit=0)

    assert result == []


def test_truncate_list_negative_limit():
    """Handles negative limit (Python slicing behavior)"""
    items = [1, 2, 3]
    result = _truncate_list(items, limit=-1)

    # Python slice [:−1] excludes last item
    assert result == [1, 2]


def test_truncate_list_non_list():
    """Returns input unchanged if not a list"""
    result = _truncate_list("not a list", limit=5)

    assert result == "not a list"


def test_truncate_list_single_item():
    """Handles single item list"""
    result = _truncate_list([42], limit=1)

    assert result == [42]


def test_truncate_list_preserves_types():
    """Preserves item types"""
    items = ["string", 42, 3.14, None, True]
    result = _truncate_list(items, limit=3)

    assert result == ["string", 42, 3.14]
    assert isinstance(result[0], str)
    assert isinstance(result[1], int)
    assert isinstance(result[2], float)


# =============================================================================
# _cfg_int Tests
# =============================================================================

def test_cfg_int_returns_int():
    """Returns integer value"""
    # This tests the function exists and returns int
    # Actual behavior depends on config which we can't easily mock
    result = _cfg_int("nonexistent_key", default_val=42)

    assert isinstance(result, int)
    # Should return default for nonexistent key
    assert result == 42


def test_cfg_int_with_default():
    """Uses default value for missing keys"""
    result = _cfg_int("definitely_not_a_real_config_key_12345", default_val=100)

    assert result == 100


def test_cfg_int_default_zero():
    """Handles zero as default"""
    result = _cfg_int("nonexistent", default_val=0)

    assert result == 0


def test_cfg_int_default_negative():
    """Handles negative default"""
    result = _cfg_int("nonexistent", default_val=-10)

    assert result == -10


# =============================================================================
# Integration Tests
# =============================================================================

def test_dedupe_and_truncate_workflow():
    """Combine deduplication and truncation"""
    items = ["apple", "banana", "apple", "cherry", "banana", "date", "elderberry"]

    # First dedupe
    deduped = _dedupe_keep_order(items)
    # Then truncate
    result = _truncate_list(deduped, limit=3)

    assert result == ["apple", "banana", "cherry"]


def test_parse_bool_for_feature_flags():
    """Use case: parsing feature flags from env"""
    # Simulating common env var patterns
    assert _parse_bool("1", default=False) == True
    assert _parse_bool("0", default=True) == False
    assert _parse_bool("true", default=False) == True
    assert _parse_bool(None, default=True) == True


def test_summary_dict_with_dedupe_tags():
    """Create summary with deduped tags"""
    tags = ["tag1", "tag2", "tag1", "tag3"]
    deduped_tags = _dedupe_keep_order(tags)

    result = _as_summary_dict("Text", deduped_tags, "source")

    assert result["tags"] == ["tag1", "tag2", "tag3"]


def test_truncate_preserves_dedupe_order():
    """Truncation after deduplication preserves order"""
    items = ["z", "a", "z", "b", "a", "c", "d", "e"]

    deduped = _dedupe_keep_order(items)
    truncated = _truncate_list(deduped, limit=3)

    # Should be first 3 unique items in original order
    assert truncated == ["z", "a", "b"]
