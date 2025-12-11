"""
tests/test_system_prompt_placeholders.py

Tests for system prompt placeholder substitution.
Validates that system_prompt.txt contains placeholders and orchestrator substitutes them correctly.
"""

import pytest
from pathlib import Path


class TestSystemPromptPlaceholders:
    """Test that system_prompt.txt contains required placeholders."""

    def test_system_prompt_contains_user_name_placeholder(self):
        """System prompt should contain {USER_NAME} placeholder."""
        prompt_path = Path("core/system_prompt.txt")
        assert prompt_path.exists(), "system_prompt.txt not found"

        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "{USER_NAME}" in content, "system_prompt.txt should contain {USER_NAME} placeholder"

        # Should appear multiple times (at least 2-3 occurrences)
        count = content.count("{USER_NAME}")
        assert count >= 2, f"Expected at least 2 {USER_NAME} placeholders, found {count}"

    def test_system_prompt_contains_user_pronouns_placeholder(self):
        """System prompt should contain {USER_PRONOUNS} placeholder."""
        prompt_path = Path("core/system_prompt.txt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "{USER_PRONOUNS}" in content, "system_prompt.txt should contain {USER_PRONOUNS} placeholder"

    def test_system_prompt_contains_pronoun_variants(self):
        """System prompt should reference pronoun variant placeholders."""
        prompt_path = Path("core/system_prompt.txt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should mention PRONOUN_SUBJ, PRONOUN_OBJ, PRONOUN_POSS in instructions
        assert "PRONOUN_SUBJ" in content, "Should reference {PRONOUN_SUBJ}"
        assert "PRONOUN_OBJ" in content, "Should reference {PRONOUN_OBJ}"
        assert "PRONOUN_POSS" in content, "Should reference {PRONOUN_POSS}"

    def test_system_prompt_no_hardcoded_luke(self):
        """System prompt should not contain hardcoded 'Luke' references (except in examples)."""
        prompt_path = Path("core/system_prompt.txt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Check each line for hardcoded "Luke" (case-sensitive)
        # Allow "Luke" in comment lines or specific contexts
        problematic_lines = []
        for i, line in enumerate(lines, start=1):
            # Skip comment lines
            if line.strip().startswith("#"):
                continue

            # Check for standalone "Luke" (not in a placeholder context)
            if " Luke" in line or "Luke " in line or "(Luke)" in line:
                # Make sure it's not part of {USER_NAME} documentation
                if "{USER_NAME}" not in line:
                    problematic_lines.append((i, line.strip()))

        # Should have no hardcoded Luke references (all should use {USER_NAME})
        assert len(problematic_lines) == 0, \
            f"Found hardcoded 'Luke' references:\n" + \
            "\n".join([f"Line {num}: {text}" for num, text in problematic_lines])

    def test_system_prompt_preserves_structure(self):
        """System prompt should preserve key sections after placeholder addition."""
        prompt_path = Path("core/system_prompt.txt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Key sections should still exist
        required_sections = [
            "## Core Style",
            "## Interaction Rules",
            "## Response Approach",
            "## Facts Handling",
            "## Memory & Context Integration",
            "## Guardrails"
        ]

        for section in required_sections:
            assert section in content, f"Missing required section: {section}"

    def test_system_prompt_readable_and_valid(self):
        """System prompt should be readable and non-empty."""
        prompt_path = Path("core/system_prompt.txt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert len(content) > 1000, "System prompt seems too short"
        assert "Daemon" in content, "System prompt should reference Daemon"


class TestPlaceholderIntegration:
    """Test placeholder integration with orchestrator (already covered in STEP 4, but sanity check)."""

    def test_orchestrator_has_placeholder_substitution_logic(self):
        """Orchestrator prepare_prompt should have placeholder substitution code."""
        from pathlib import Path
        orchestrator_path = Path("core/orchestrator.py")

        with open(orchestrator_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should have substitution logic
        assert 'replace("{USER_NAME}"' in content
        assert 'replace("{USER_PRONOUNS}"' in content
        assert 'PRONOUN_SUBJ' in content
        assert 'PRONOUN_OBJ' in content
        assert 'PRONOUN_POSS' in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
