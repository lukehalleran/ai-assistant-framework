"""Tests for the Active Features Inventory feature.

Tests the _build_feature_inventory() method in UnifiedPromptBuilder and
the [ACTIVE FEATURES] prompt section.
"""

import pytest
from unittest.mock import patch, MagicMock


def _make_builder():
    """Create a minimal UnifiedPromptBuilder for testing."""
    from core.prompt.builder import UnifiedPromptBuilder

    mc = MagicMock()
    mc.corpus_manager = MagicMock()
    mm = MagicMock()

    return UnifiedPromptBuilder(memory_coordinator=mc, model_manager=mm)


def _base_context(**overrides):
    """Create a base context dict with sensible defaults, applying overrides."""
    ctx = {
        "recent_conversations": [],
        "memories": [],
        "user_profile": "",
        "narrative_state": "",
        "summaries": [],
        "recent_summaries": [],
        "semantic_summaries": [],
        "reflections": [],
        "recent_reflections": [],
        "semantic_reflections": [],
        "dreams": [],
        "semantic_chunks": [],
        "wiki": [],
        "personal_notes": [],
        "user_uploads": [],
        "git_commits": [],
        "procedural_skills": [],
        "proposed_features": [],
        "graph_context": [],
        "unresolved_threads": [],
        "proactive_insights": [],
        "web_search_results": None,
        "codebase_changes": {},
    }
    ctx.update(overrides)
    return ctx


class TestBuildFeatureInventory:
    """Tests for _build_feature_inventory()."""

    def test_all_enabled_shows_all_categories(self):
        """All four category lines (Memory, Knowledge, Proactive, Analysis) appear."""
        builder = _make_builder()
        context = _base_context()

        result = builder._build_feature_inventory(context)

        assert "Memory:" in result
        assert "Knowledge:" in result
        assert "Proactive:" in result
        assert "Analysis:" in result

    def test_disabled_features_show_off(self):
        """When a feature flag is OFF, the inventory shows OFF."""
        builder = _make_builder()
        context = _base_context()

        with patch("config.app_config.KNOWLEDGE_GRAPH_ENABLED", False):
            result = builder._build_feature_inventory(context)

        assert "knowledge_graph=OFF" in result

    def test_result_counts_from_context(self):
        """Counts from context dict appear in the inventory output."""
        builder = _make_builder()
        context = _base_context(
            graph_context=["user knows Python", "user likes cats", "user has a dog"],
            git_commits=[{"content": "abc123 fix: bug"}, {"content": "def456 feat: new"}],
            unresolved_threads=[{"topic": "t1"}, {"topic": "t2"}],
            procedural_skills=[{"metadata": {"trigger": "x", "action_pattern": "y"}}],
        )

        result = builder._build_feature_inventory(context)

        assert "(3 edges)" in result
        assert "(2)" in result  # git commits count
        assert "(2 open)" in result  # threads count
        assert "(1)" in result  # skills count

    def test_section_in_assembled_prompt(self):
        """The [ACTIVE FEATURES] section appears in the assembled prompt."""
        builder = _make_builder()
        context = _base_context()

        prompt = builder._assemble_prompt(context=context, user_input="hello")

        assert "[ACTIVE FEATURES]" in prompt
        assert "Memory:" in prompt
        assert "Knowledge:" in prompt

    def test_compact_token_size(self):
        """Feature inventory should be compact — under ~200 tokens (~800 chars)."""
        builder = _make_builder()
        context = _base_context(
            graph_context=["a", "b", "c"],
            git_commits=[{"content": "x"}],
            unresolved_threads=[{"topic": "t"}],
            procedural_skills=[{"metadata": {"trigger": "t", "action_pattern": "a"}}],
            proactive_insights=["insight1"],
            personal_notes=[{"content": "note"}],
            reference_docs=[{"content": "doc"}],
        )

        result = builder._build_feature_inventory(context)

        # 4 lines, each ~80 chars → ~320 chars total. Must be under 800.
        assert len(result) < 800
        assert result.count("\n") == 3  # exactly 4 lines (3 newlines)
