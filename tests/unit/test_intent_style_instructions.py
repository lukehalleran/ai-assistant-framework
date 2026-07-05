"""Tests for get_intent_style_instructions() in core/tone_instructions.py."""

import pytest

from core.tone_instructions import (
    get_intent_style_instructions,
    _INTENT_STYLE_BLOCKS,
)
from core.intent_classifier import IntentType


class TestIntentStyleInstructions:

    def test_technical_help_block(self):
        out = get_intent_style_instructions(IntentType.TECHNICAL_HELP, 0.75, "CONVERSATIONAL")
        assert "## INTENT STYLE: TECHNICAL HELP" in out
        assert "code blocks" in out

    def test_accepts_plain_string_intent(self):
        out = get_intent_style_instructions("factual_recall", 0.9, "CONVERSATIONAL")
        assert "## INTENT STYLE: FACTUAL RECALL" in out

    def test_none_crisis_level_treated_conversational(self):
        assert get_intent_style_instructions("project_work", 0.8, None) != ""

    @pytest.mark.parametrize("level", ["HIGH", "MEDIUM", "CONCERN", "high", "concern"])
    def test_non_conversational_tone_suppresses(self, level):
        assert get_intent_style_instructions(IntentType.TECHNICAL_HELP, 0.95, level) == ""

    def test_low_confidence_suppresses(self):
        assert get_intent_style_instructions(IntentType.TECHNICAL_HELP, 0.59, "CONVERSATIONAL") == ""

    def test_confidence_floor_inclusive(self):
        assert get_intent_style_instructions(IntentType.TECHNICAL_HELP, 0.60, "CONVERSATIONAL") != ""

    def test_none_confidence_suppresses(self):
        assert get_intent_style_instructions(IntentType.TECHNICAL_HELP, None, "CONVERSATIONAL") == ""

    def test_emotional_support_has_no_block_by_design(self):
        assert get_intent_style_instructions(IntentType.EMOTIONAL_SUPPORT, 0.95, "CONVERSATIONAL") == ""

    def test_general_has_no_block(self):
        assert get_intent_style_instructions(IntentType.GENERAL, 0.95, "CONVERSATIONAL") == ""

    def test_unknown_intent_returns_empty(self):
        assert get_intent_style_instructions("no_such_intent", 0.95, "CONVERSATIONAL") == ""

    def test_all_blocks_start_on_new_paragraph(self):
        # Blocks are appended to a rstrip()'d system prompt — they must supply
        # their own separation so they don't fuse with the preceding section.
        for key, block in _INTENT_STYLE_BLOCKS.items():
            assert block.startswith("\n\n"), key
            assert "## INTENT STYLE:" in block, key

    def test_covered_intents_match_expectation(self):
        expected = {
            "factual_recall", "temporal_recall", "technical_help",
            "project_work", "creative_exploration", "meta_conversational",
            "casual_social",
        }
        assert set(_INTENT_STYLE_BLOCKS.keys()) == expected
