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

from pathlib import Path
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


# ---------------------------------------------------------------------------
# Daily-notes base directory (single resolver for all three generators)
# ---------------------------------------------------------------------------

def daily_notes_base() -> Path:
    """Directory that holds daily notes (and the weekly/monthly folders under it).

    Obsidian enabled AND the vault directory exists -> <vault>/<DAILY_NOTES_FOLDER>
    (the historical behaviour). Otherwise -> <data root>/notes/<DAILY_NOTES_FOLDER>,
    where the data root is the directory of CORPUS_FILE (relative paths resolve
    against the repo root, never the launch CWD). Reads live config values.
    """
    from config import app_config  # lazy import: live-config

    folder = app_config.DAILY_NOTES_FOLDER
    # An empty path would resolve to "." (the launch CWD) and pass is_dir().
    if app_config.OBSIDIAN_ENABLED and str(app_config.OBSIDIAN_VAULT_PATH or "").strip():
        vault = Path(app_config.OBSIDIAN_VAULT_PATH).expanduser()
        if vault.is_dir():
            return vault / folder
    corpus = Path(app_config.CORPUS_FILE or "./data/corpus.json").expanduser()
    if not corpus.is_absolute():
        corpus = Path(__file__).resolve().parent.parent / corpus
    return corpus.parent / "notes" / folder


# Speaker-authority rules shared by the daily-note and narrative-synthesis
# prompts (2026-10-12, B4; class: BC-46, BC-75, BC-91). Live incident: the user
# said "I just started a new job at Northwind as a data analyst"; Daemon later
# called Northwind "its own guess"; the daily note then wrote the user's
# statement in as "later confirmed this was test data, not fact - actual
# workplace remains unknown", the narrative inherited it, and the note also
# credited Wolfram with a tip calculation no tool ran. NO braces here: both
# prompts pass through str.format().
SPEAKER_AUTHORITY_RULES = (
    "SPEAKER AUTHORITY - these rules outrank everything above:\n"
    "- Lines starting \"User:\" are the user's own words; lines starting \"Daemon:\" are "
    "the assistant's. A fact the user stated about their own life (job, home, health, "
    "plans) is the user's own account: record it as stated. NEVER describe a user "
    "statement as unconfirmed, unverified, test data, or not fact.\n"
    "- If the assistant later doubted, disowned, or contradicted something the user "
    "stated, that is an assistant error, not evidence against the user. Note the "
    "assistant's slip if it matters; keep the user's statement standing unless the USER "
    "retracted it.\n"
    "- Text marked \"Personal-claim check\" concerns claims the ASSISTANT added; it says "
    "nothing about the user's statements.\n"
    "- Name a tool or service (calculator, web search, Wolfram, calendar, email) as having "
    "been used ONLY if the source text explicitly says that tool ran. Otherwise describe "
    "the result without naming any tool.\n"
)
