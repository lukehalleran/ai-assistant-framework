# utils/notes_common.py
"""
Module Contract
- Purpose: shared roster/parser for the three note generators (daily/weekly/
  monthly). Extracted 2026-09-16 (compaction) — `FALLBACK_MODELS` and the
  frontmatter parser were copy-pasted verbatim across
  utils/daily_notes_generator.py, utils/weekly_notes_generator.py and
  utils/monthly_notes_generator.py.
- Inputs:
  - FALLBACK_MODELS: Tuple[str, ...]  (ordered fallback-model roster; callers
    build `[primary] + [m for m in FALLBACK_MODELS if m != primary]`)
  - parse_frontmatter(content) -> (frontmatter_dict, body_str)
- Dependencies: none beyond stdlib + PyYAML (a hard dependency; imported at
  module level — the original per-file methods imported it lazily, hoisted
  2026-09-16 by the import-hygiene batch).
- Side effects: none; pure data + a pure function.
"""

from typing import Any, Dict, Tuple

import yaml

FALLBACK_MODELS: Tuple[str, ...] = (
    "claude-opus-4.8",  # Anthropic Claude (best)
    "sonnet-4.5",       # Anthropic Claude (fast)
    "gpt-4o-mini",      # Fast, cheap OpenAI
    "deepseek-v3.1",    # DeepSeek
    "gpt-4o",           # Standard OpenAI
    "claude-opus-4.5",  # Anthropic Claude
    "gemini-3-pro",     # Google Gemini
    "gpt-5",            # Newer OpenAI
    "deepseek-r1",      # DeepSeek reasoning
    "glm-4.6",          # GLM
)


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content."""
    frontmatter = {}
    body = content

    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
            except Exception:
                pass
            body = parts[2].strip()

    return frontmatter, body
