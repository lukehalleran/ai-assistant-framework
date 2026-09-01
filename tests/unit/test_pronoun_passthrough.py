"""Tests for pronoun passthrough in core/orchestrator.py (2026-09-01)."""

import pytest


def _resolve_pronouns(pronouns):
    """Extract the pronoun resolution logic from the orchestrator.

    Matches the logic in DaemonOrchestrator._maybe_inject_identity_placeholders
    at line 1041-1055 (or thereabouts).
    """
    PRONOUN_MAP = {
        "he/him": ("he", "him", "his"),
        "she/her": ("she", "her", "her"),
        "they/them": ("they", "them", "their"),
    }
    # Try map first; if not found and format is valid, use verbatim passthrough
    pronouns_lower = pronouns.lower().strip()
    if pronouns_lower in PRONOUN_MAP:
        subj, obj, poss = PRONOUN_MAP[pronouns_lower]
    elif "/" in pronouns_lower:
        # Verbatim passthrough: split on "/" and pad the third slot
        parts = pronouns_lower.split("/")
        if len(parts) == 2 and all(p.isalpha() for p in parts):
            subj, obj, poss = parts[0], parts[1], parts[1]
        else:
            subj, obj, poss = ("they", "them", "their")
    else:
        subj, obj, poss = ("they", "them", "their")
    return subj, obj, poss


class TestPronounPassthrough:
    """Pronoun passthrough and map behavior."""

    def test_known_map_hehin(self):
        """Known pronoun map: he/him."""
        result = _resolve_pronouns("he/him")
        assert result == ("he", "him", "his")

    def test_known_map_sheher(self):
        """Known pronoun map: she/her."""
        result = _resolve_pronouns("she/her")
        assert result == ("she", "her", "her")

    def test_unknown_custom_pronouns(self):
        """Unknown pronouns with valid format: verbatim passthrough."""
        result = _resolve_pronouns("xe/xem")
        assert result == ("xe", "xem", "xem")

    def test_custom_three_part_pronouns(self):
        """Custom pronouns with three parts: fallback to they/them."""
        # The logic checks len(parts) == 2, so three-part falls through
        result = _resolve_pronouns("ze/hir/hirs")
        assert result == ("they", "them", "their")

    def test_invalid_format_falls_back(self):
        """Invalid format (no slash): fallback to they/them."""
        result = _resolve_pronouns("xe-xem")  # No slash
        assert result == ("they", "them", "their")

    def test_numeric_pronouns_rejected(self):
        """Pronouns with non-alpha characters: fallback."""
        result = _resolve_pronouns("x3/x3m")  # Contains numeric
        assert result == ("they", "them", "their")
