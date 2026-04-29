"""
Tests for core.prompt backward-compatibility contract.

Verifies that:
- Class imports still work (UnifiedPromptBuilder, ContextGatherer, etc.)
- Helper function imports still work (_truncate_at_spurious_turns, etc.)
- The modular package is the active entry point (core/prompt/ package, not core/prompt.py)
"""

import pytest


def test_unified_prompt_builder_importable():
    """UnifiedPromptBuilder must be importable from core.prompt."""
    from core.prompt import UnifiedPromptBuilder
    assert UnifiedPromptBuilder is not None
    assert callable(UnifiedPromptBuilder)


def test_context_gatherer_importable():
    from core.prompt import ContextGatherer
    assert ContextGatherer is not None


def test_prompt_formatter_importable():
    from core.prompt import PromptFormatter
    assert PromptFormatter is not None


def test_token_manager_importable():
    from core.prompt import TokenManager
    assert TokenManager is not None


def test_truncate_at_spurious_turns_importable():
    """gui/handlers.py imports this from core.prompt — must remain available."""
    from core.prompt import _truncate_at_spurious_turns
    assert callable(_truncate_at_spurious_turns)
    # Basic sanity: no-op on clean text
    assert _truncate_at_spurious_turns("hello") == "hello"


def test_helper_utilities_importable():
    from core.prompt import _parse_bool, _dedupe_keep_order, _truncate_list
    assert callable(_parse_bool)
    assert callable(_dedupe_keep_order)
    assert callable(_truncate_list)


def test_package_is_active_entry_point():
    """core.prompt must resolve to the package (core/prompt/__init__.py),
    not the legacy shim file (core/prompt.py)."""
    import core.prompt as cp
    # The package __init__ exports UnifiedPromptBuilder directly; the shim
    # defines a deprecated build_prompt() function. If the package is active,
    # build_prompt is NOT in the namespace (it's only in the shadowed shim).
    assert not hasattr(cp, "build_prompt"), (
        "core/prompt.py shim is being loaded instead of core/prompt/ package"
    )
