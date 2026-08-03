"""
Tests for conditional section-usage instructions (2026-08-02 prompt trim).

The per-section guidance blocks (Obsidian notes / self-docs / temporal
grounding, ~1.2K tokens) moved from the static operating principles into
core/prompt/section_instructions.py and are appended to the system prompt
ONLY on turns whose context actually renders those sections.
"""

from pathlib import Path

from core.prompt.section_instructions import (
    SECTION_INSTRUCTIONS,
    conditional_instruction_tail,
)


class TestConditionalTail:
    def test_empty_ctx_injects_nothing(self):
        assert conditional_instruction_tail({}) == ""
        assert conditional_instruction_tail(None) == ""

    def test_absent_sections_inject_nothing(self):
        ctx = {"memories": ["m"], "personal_notes": [], "reference_docs": None,
               "narrative_state": ""}
        assert conditional_instruction_tail(ctx) == ""

    def test_single_section_injects_only_its_block(self):
        tail = conditional_instruction_tail({"personal_notes": ["note"]})
        assert "[USER'S PERSONAL NOTES]" in tail
        assert "[DAEMON DOCUMENTATION]" not in tail
        assert "[TEMPORAL GROUNDING]" not in tail

    def test_all_sections_inject_all_blocks(self):
        ctx = {"personal_notes": ["n"], "reference_docs": ["d"],
               "narrative_state": "life state"}
        tail = conditional_instruction_tail(ctx)
        for marker in ("[USER'S PERSONAL NOTES]", "[DAEMON DOCUMENTATION]",
                       "[TEMPORAL GROUNDING]"):
            assert marker in tail

    def test_moved_blocks_gone_from_static_principles(self):
        # The whole point: these must not ALSO ride in the static prompt.
        principles = Path("config/prompts/operating_principles.txt").read_text()
        for header in ("### [USER'S PERSONAL NOTES] — Obsidian Vault",
                       "### [DAEMON DOCUMENTATION] — Self-Knowledge",
                       "### [TEMPORAL GROUNDING] — Life Context Synthesis"):
            assert header not in principles

    def test_scoring_example_uses_live_weights(self):
        # The moved self-docs example had gone stale (0.35/0.25/0.20 era);
        # it must state the live weight vector (config.yaml gating.score_weights).
        docs_block = dict(SECTION_INSTRUCTIONS)["reference_docs"]
        assert "0.30×relevance" in docs_block
        assert "0.35×relevance" not in docs_block
