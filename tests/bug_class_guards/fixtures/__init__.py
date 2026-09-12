"""Reusable synthetic source snippets for the bug-class scanner controls.

Every scanner gets a RED control (a tree that must yield a finding) and a
GREEN control (the same shape, closed), both driven through the same
``scan()`` the gate calls — a control that exercised a private helper instead
would be exactly the fixture-contract drift BC-64 names.

Snippets are kept as strings rather than importable ``.py`` files so the
fixture tree is assembled per test in ``tmp_path`` and can never be picked up
by collection, linting, or a scanner run against the real repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

# ---------------------------------------------------------------- DM-01 ----

# Both incident shapes: a bare keyword constant and a keyword-list `any(...)`.
DM01_RED = '''"""Module with raw substring keyword tests."""

HEAVY_KEYWORDS = ["ice", "crisis"]


def is_heavy(text):
    text_lower = text.lower()
    if "ice" in text_lower:
        return True
    return any(word in text_lower for word in HEAVY_KEYWORDS)
'''

# Same module, chokepoint adopted — never a candidate.
DM01_GREEN_CHOKEPOINT = '''"""Module that routes through the chokepoint."""

from utils.trigger_match import compile_keyword_matcher, has_non_negated_hit

HEAVY_KEYWORDS = ["ice", "crisis"]
_MATCHER = compile_keyword_matcher(HEAVY_KEYWORDS)


def is_heavy(text):
    text_lower = text.lower()
    if "ice" in text_lower:
        return True
    return has_non_negated_hit(text, _MATCHER)
'''

# Ordinary membership tests: no lowered-text right-hand side, no vocabulary.
DM01_GREEN_ORDINARY = '''"""Module with ordinary membership tests."""

ROWS = {"a": 1}


def lookup(key, payload):
    if key in ROWS:
        return ROWS[key]
    if "x" in payload:
        return payload["x"]
    return None
'''

# ---------------------------------------------------------------- DM-17 ----

DM17_RED_SCRIPT = '''"""Curation script with --apply and no daemon guard."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()
'''

DM17_GREEN_SCRIPT = '''"""Curation script with --apply behind the daemon guard."""

import argparse

from utils.daemon_guard import daemon_running


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply and daemon_running():
        raise SystemExit("daemon is live")
    return args
'''

DM17_RED_TEST = '''"""Test naming a production store path."""

STORE = "data/knowledge_graph.json"


def test_reads_store():
    assert STORE
'''

DM17_GREEN_TEST = '''"""Test using only sandboxed paths."""


def test_reads_store(tmp_path):
    assert (tmp_path / "knowledge_graph.json").parent.exists()
'''

# ---------------------------------------------------------------- DM-18 ----

DM18_RED = '''"""Retrieval module that swallows a store failure into an empty list."""

import logging

logger = logging.getLogger(__name__)


class Store:
    def __init__(self, collection):
        self.collection = collection

    def get_rows(self, query):
        try:
            return self.collection.query(query_texts=[query])
        except Exception:
            logger.warning("query failed")
            return []
'''

DM18_GREEN_RERAISE = '''"""Same store call, but the handler re-raises."""

import logging

logger = logging.getLogger(__name__)


class Store:
    def __init__(self, collection):
        self.collection = collection

    def get_rows(self, query):
        try:
            return self.collection.query(query_texts=[query])
        except Exception:
            raise

    def get_other(self, query):
        try:
            return self.collection.query(query_texts=[query])
        except Exception:
            logger.warning("query failed")
            raise
'''

DM18_GREEN_NARROW = '''"""Same store call, narrow handler and a discriminating return."""

class Store:
    def __init__(self, collection):
        self.collection = collection

    def get_rows(self, query):
        try:
            return self.collection.query(query_texts=[query])
        except KeyError:
            return []

    def get_tri_state(self, query):
        try:
            return self.collection.query(query_texts=[query])
        except Exception:
            return {"source": "fallback", "rows": []}
'''

DM18_GREEN_NO_STORE = '''"""Broad handler returning empty, but no store call in the function."""

def parse_ints(raw):
    try:
        return [int(part) for part in raw.split(",")]
    except Exception:
        return []
'''

# ---------------------------------------------------------------- DM-16 ----

# `reached_key` is read directly outside config/; `dead_key` is declared in
# app_config but its constant has no consumer; the block scalar's continuation
# lines look like keys and must not be read as any.
DM16_YAML = """section:
  reached_key: 1
  dead_key: 2
  prose: 'Summary of the thing:

    Input: "{text}"
    '
  nested:
    deep_dead_key: 3
list_section:
  - item_one: 1
"""

DM16_APP_CONFIG = '''"""Synthetic app_config."""

REACHED = config.get("section", {}).get("reached_key", 0)
DEAD = config.get("section", {}).get("dead_key", 0)
DEEP_DEAD = config.get("section", {}).get("deep_dead_key", 0)
'''

DM16_CONSUMER = '''"""Synthetic consumer module."""

from config.app_config import REACHED

PROSE_KEY = "prose"


def use():
    return REACHED
'''

DM16_CONSUMER_ALL = '''"""Synthetic consumer reading every constant."""

from config.app_config import DEAD, DEEP_DEAD, REACHED

PROSE_KEY = "prose"


def use():
    return REACHED + DEAD + DEEP_DEAD
'''


# ---------------------------------------------------------------- DM-29 ----


def changelog(batches: list[tuple[str, str]]) -> str:
    """Render a minimal changelog: ``## <date> <title>`` + one body line."""
    parts = ["# CLAUDE.md fix-history archive", ""]
    for date, body in batches:
        parts.extend([f"## {date} synthetic batch", "", body, ""])
    return "\n".join(parts) + "\n"


DM29_WIDENING_LINE = "`_INFO_SEEKING_CUES` gained one more phrase for the missed shape."


# ------------------------------------------------------------------ tree ----


def build_tree(root: Path, files: Mapping[str, str]) -> Path:
    """Write ``{relative path: text}`` under ``root``, creating parents."""
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return root


# -------------------------------------------------------------- catalog ----


def catalog_doc(
    index: list[tuple[str, str]],
    body: list[tuple[str, str]] | None = None,
    *,
    singletons: int = 1,
) -> str:
    """Render a minimal VALID catalog from ``(BC id, status)`` pairs.

    ``body`` defaults to ``index``, so a disagreement is expressed by passing
    both.  Every structure the parser requires — the index table, the DM and
    CM tables, a singleton bullet, and all five entry fields — is synthesised
    here so the catalog controls never depend on the real 94 KB document:
    using today's live debt as a red control is the moving-oracle shape BC-65
    names, and it makes reconciling that debt break the tests.
    """
    body = index if body is None else body
    lines = [
        "# Bug classes",
        "",
        "## Index",
        "",
        "| ID | Class | Family | Status |",
        "|---|---|---|---|",
    ]
    for ident, status in index:
        lines.append(f"| {ident} | synthetic mechanism | A | {status} |")
    lines += [
        "",
        "## Detection methods (DM) — find instances without a full read",
        "",
        "| ID | Method | Runs as | Classes |",
        "|---|---|---|---|",
        "| DM-01 | synthetic detection method | grep | all |",
        "",
        "## Closure methods (CM) — what has actually stopped a class",
        "",
        "| ID | Method | Why it holds | Classes |",
        "|---|---|---|---|",
        "| CM-01 | synthetic closure method | synthetic | all |",
        "",
        "## A. Synthetic family",
        "",
    ]
    for ident, status in body:
        lines += [
            f"### {ident} Synthetic mechanism heading",
            "- Mechanism: synthetic mechanism text.",
            "- Incidents: 2026-01-01 synthetic incident text.",
            "- Find: DM-01 over the synthetic tree.",
            "- Closure: CM-01 at the synthetic chokepoint.",
            f"- Status: {status} — synthetic status text.",
            "",
        ]
    lines += ["## Unclassified singletons", ""]
    for number in range(1, singletons + 1):
        lines.append(f"- SG-{number:03d} synthetic singleton.")
    return "\n".join(lines) + "\n"
