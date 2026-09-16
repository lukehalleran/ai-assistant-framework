"""
Unit tests for utils/notes_common.py — shared fallback-model roster and
frontmatter parser extracted from the daily/weekly/monthly note generators
(2026-09-16 compaction). Behavior must match the pre-extraction copies
exactly: same roster/order, same primary-first-dedupe composition line, and
the same frontmatter/body split-and-strip semantics (including on malformed
YAML, where the old per-file methods swallowed the exception and kept
`frontmatter = {}`).
"""

from utils.notes_common import FALLBACK_MODELS, parse_frontmatter


# =============================================================================
# FALLBACK_MODELS
# =============================================================================

def test_fallback_models_exact_roster():
    """The roster is the exact tuple that was duplicated across the three
    note generators, in the exact same order."""
    assert FALLBACK_MODELS == (
        "claude-opus-4.8",
        "sonnet-4.5",
        "gpt-4o-mini",
        "deepseek-v3.1",
        "gpt-4o",
        "claude-opus-4.5",
        "gemini-3-pro",
        "gpt-5",
        "deepseek-r1",
        "glm-4.6",
    )


def test_fallback_models_is_immutable_tuple():
    assert isinstance(FALLBACK_MODELS, tuple)


def test_models_to_try_composition_dedupes_primary_already_in_roster():
    """The generators' local composition line:
    `[self.model_name] + [m for m in FALLBACK_MODELS if m != self.model_name]`
    must put the primary first and never repeat it, even when the primary
    is already a member of the fallback roster."""
    primary = "gpt-4o-mini"
    models_to_try = [primary] + [m for m in FALLBACK_MODELS if m != primary]
    assert models_to_try[0] == primary
    assert models_to_try.count(primary) == 1
    assert len(models_to_try) == len(FALLBACK_MODELS)
    # Every remaining roster member (bar the primary) survives, in order.
    assert models_to_try[1:] == [m for m in FALLBACK_MODELS if m != primary]


def test_models_to_try_composition_primary_outside_roster():
    """A primary model not in the roster is simply prepended; the full
    roster follows unchanged."""
    primary = "some-other-model"
    models_to_try = [primary] + [m for m in FALLBACK_MODELS if m != primary]
    assert models_to_try == [primary] + list(FALLBACK_MODELS)


# =============================================================================
# parse_frontmatter
# =============================================================================

def test_parse_frontmatter_valid_document():
    content = "---\ndate: '2026-09-16'\nusage_intensity: 5\n---\nBody text here."
    frontmatter, body = parse_frontmatter(content)
    assert frontmatter == {"date": "2026-09-16", "usage_intensity": 5}
    assert body == "Body text here."


def test_parse_frontmatter_no_frontmatter():
    """A document that doesn't start with '---' is returned as-is: empty
    frontmatter dict, body unchanged (not even stripped)."""
    content = "Just a plain markdown document.\nNo frontmatter here."
    frontmatter, body = parse_frontmatter(content)
    assert frontmatter == {}
    assert body == content


def test_parse_frontmatter_malformed_yaml():
    """On a YAML parse error, the exception is swallowed and frontmatter
    stays the initial empty dict, but body is still split and stripped from
    the raw content exactly as when parsing succeeds."""
    content = "---\nfoo: [1, 2\n---\nBody survives anyway."
    frontmatter, body = parse_frontmatter(content)
    assert frontmatter == {}
    assert body == "Body survives anyway."


def test_parse_frontmatter_body_is_stripped():
    content = "---\nkey: value\n---\n\n  Body with surrounding whitespace.  \n\n"
    frontmatter, body = parse_frontmatter(content)
    assert frontmatter == {"key": "value"}
    assert body == "Body with surrounding whitespace."


def test_parse_frontmatter_incomplete_delimiter_only_two_parts():
    """Starts with '---' but there's no closing delimiter (split yields
    fewer than 3 parts) — frontmatter stays empty, body is the ORIGINAL
    unstripped content (the `body = parts[2].strip()` line never runs)."""
    content = "---\nno closing delimiter at all"
    frontmatter, body = parse_frontmatter(content)
    assert frontmatter == {}
    assert body == content
