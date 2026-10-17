"""B1 (2026-10-12): shipped prompts must not gender the user. class: BC-83.

Original batch label: oct12_b1 (renamed 2026-10-16; old path in tests/RENAMES.tsv)."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_FILES = sorted(
    {p for p in (ROOT / "config" / "prompts").glob("*.txt")} | {ROOT / "core" / "system_prompt.txt"}
)
_PRONOUN = re.compile(r"\b(he|him|his|himself|she|her|hers|herself)\b", re.IGNORECASE)


def test_prompt_files_found():
    assert len(_FILES) >= 4


@pytest.mark.parametrize("path", _FILES, ids=lambda p: p.name)
def test_no_gendered_pronouns(path):
    hits = [(i, l.strip()) for i, l in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if _PRONOUN.search(l)]
    assert not hits, hits
