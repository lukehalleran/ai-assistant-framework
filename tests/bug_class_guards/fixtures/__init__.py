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


# ------------------------------------------------------------- dm31 ----

DM31_RED = '''"""The 2026-09-11 shape: a public classifier asserts the world."""


async def analyze_for_web_search_llm(
    query,
    model_manager=None,
    web_search_enabled: bool = True,
    remaining_credits: float = 100,
):
    return (query, web_search_enabled, remaining_credits)
'''

DM31_GREEN_RESOLVED = '''"""None means "resolve it", which is the fix."""


def _resolve_remaining_credits(value):
    return 100.0 if value is None else float(value)


async def analyze_for_web_search_llm(
    query,
    model_manager=None,
    web_search_enabled=None,
    remaining_credits=None,
):
    return (query, web_search_enabled, _resolve_remaining_credits(remaining_credits))
'''

DM31_GREEN_CALLER_SIZES = '''"""Caller-chosen sizes and a fail-closed toggle are not assertions."""


def gather(query, limit: int = 30, max_tokens: int = 4000, estimated_credits: float = 1.0,
           verbose_enabled: bool = False):
    return (query, limit, max_tokens, estimated_credits, verbose_enabled)
'''

DM31_GREEN_PRIVATE_HELPER = '''"""A private helper's default is an internal convenience."""


def _build_prompt(query, remaining_credits: float = 100, web_search_enabled: bool = True):
    return f"{query}{remaining_credits}{web_search_enabled}"
'''


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


# ------------------------------------------------- DM-01 contract v2 ----

# An import of the chokepoint is not proof that every raw test uses it.
DM01_RED_IMPORT_ONLY = '''"""Imports the chokepoint but still tests a raw substring."""

from utils.trigger_match import compile_keyword_matcher  # noqa: F401


def is_heavy(text):
    text_lower = text.lower()
    return "ice" in text_lower
'''

# Real matcher use beside a raw test: the raw test is still a candidate. This
# is the former green chokepoint fixture, which contained the blind spot.
DM01_RED_BESIDE_MATCHER = DM01_GREEN_CHOKEPOINT

DM01_GREEN_MATCHER_ONLY = '''"""Routes every keyword test through the chokepoint."""

from utils.trigger_match import compile_keyword_matcher, has_non_negated_hit

HEAVY_KEYWORDS = ["ice", "crisis"]
_MATCHER = compile_keyword_matcher(HEAVY_KEYWORDS)


def is_heavy(text):
    return has_non_negated_hit(text, _MATCHER)
'''

# Lowered-text shapes the removed lexical prefilter could not see: a call
# receiver with parentheses, and an attribute ending ``_lower``.
DM01_RED_OUTSIDE_OLD_PREFILTER = '''"""Lowered-text membership the old prefilter skipped."""


def normalize(text):
    return " ".join(text.split())


class Detector:
    def __init__(self, text):
        self.text_lower = text.lower()

    def is_heavy(self, text):
        if "ice" in normalize(text).lower():
            return True
        return "crisis" in self.text_lower
'''

# ------------------------------------------------- DM-17 contract v2 ----

DM17_RED_GUARD_IN_COMMENT = '''"""Store script whose guard exists only as prose."""

import argparse

# daemon_guard.daemon_running() is checked by the operator before --apply.


def write_store():
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        write_store()
'''

DM17_RED_GUARD_IN_STRING = '''"""Store script whose guard exists only inside a string."""

import argparse

NOTE = "run daemon_running() from utils.daemon_guard before applying"


def write_store():
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        write_store()
'''

DM17_RED_GUARD_IMPORT_ONLY = '''"""Store script that imports the guard and never calls it."""

import argparse

from utils.daemon_guard import daemon_running  # noqa: F401


def write_store():
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        write_store()
'''

DM17_RED_GUARD_LATE = '''"""Store script that checks the daemon only after writing."""

import argparse

from utils.daemon_guard import daemon_running


def write_store():
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        write_store()
    if daemon_running():
        raise SystemExit("daemon is live")
'''

DM17_RED_GUARD_UNREACHABLE = '''"""Store script whose guard sits in a branch that never runs."""

import argparse

from utils.daemon_guard import daemon_running


def write_store():
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if False:
        if daemon_running():
            raise SystemExit("daemon is live")
    if args.apply:
        write_store()
'''

DM17_RED_GUARD_UNCALLED_HELPER = '''"""Store script whose guard lives in a helper nothing calls."""

import argparse

from utils.daemon_guard import daemon_running


def refuse_if_live():
    if daemon_running():
        raise SystemExit("daemon is live")


def write_store():
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        write_store()
'''

# The reviewed shapes the repository's own --apply scripts use.
DM17_GREEN_WRAPPER = '''"""Store script using the wrapped guard and a --force override."""

import argparse


def _daemon_running() -> bool:
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception:
        return False


def write_store(apply):
    return apply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if _daemon_running() and not args.force:
        return 1
    if not args.apply:
        return 0
    write_store(args.apply)
    return 0
'''

DM17_GREEN_TRY_IMPORT = '''"""Store script that imports and runs the guard after its dry-run exit."""

import argparse
import sys


def write_store(apply):
    return apply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        print("dry run")
        return
    try:
        from utils.daemon_guard import daemon_running
        if daemon_running():
            sys.exit(2)
    except ImportError:
        pass
    write_store(args.apply)
'''

DM17_GREEN_GUARDED_CALLEE = '''"""Store script whose entry point hands args to a self-guarded runner."""

import argparse


def _daemon_running() -> bool:
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception:
        return False


def write_store(apply):
    return apply


def run(args):
    if not args.apply:
        return 0
    if _daemon_running():
        return 1
    write_store(args.apply)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    return run(args)
'''

# Documented scope boundary: prose that mentions the flag is not a flag.
DM17_SCOPE_PROSE_ONLY_APPLY = '''"""Read-only report. No --apply flag exists; nothing is written."""


def main():
    return 0
'''

DM17_RED_TEST_CONCATENATED = '''"""Test naming a store path through implicit string concatenation."""

STORE = "data" "/knowledge_graph.json"


def test_reads_store():
    assert STORE
'''

# The narrowly named root-conftest exception: only tests/conftest.py owns the
# sandbox redirects, so only that exact path is exempt.
DM17_ROOT_CONFTEST_SANDBOX = '''"""Root conftest redirecting a production store path into tmp."""

import pytest

LIVE_STORE = "data/pending_actions.json"


@pytest.fixture(autouse=True)
def sandbox_store(tmp_path, monkeypatch):
    monkeypatch.setenv("PENDING_ACTIONS_PATH", str(tmp_path / "pending_actions.json"))
    return LIVE_STORE
'''

# ------------------------------------------------- DM-18 multiplicity ----

# Two identical broad handlers in one function share one anchor, so the
# baseline needs two occurrences (the four legacy duplicate keys).
DM18_TWIN_HANDLERS = '''"""Retrieval function with two identical swallowing handlers."""


class Store:
    def __init__(self, collection):
        self.collection = collection

    def get_rows(self, query):
        try:
            ids = self.collection.query(query_texts=[query])
        except Exception:
            return []
        try:
            return self.collection.get(ids=ids)
        except Exception:
            return []
'''

DM18_ONE_OF_TWINS = '''"""Retrieval function after one of the twin handlers was fixed."""


class Store:
    def __init__(self, collection):
        self.collection = collection

    def get_rows(self, query):
        try:
            ids = self.collection.query(query_texts=[query])
        except Exception:
            return []
        return self.collection.get(ids=ids)
'''

DM18_THIRD_HANDLER = DM18_TWIN_HANDLERS + '''
    def get_more(self, query):
        try:
            return self.collection.query(query_texts=[query])
        except Exception:
            return []
'''


# ------------------------------------------------------- contract pins ----

import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
from collections import Counter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = "config/bug_class_policy.json"
BASELINE_PATH = "config/bug_class_baseline.json"
DISPOSITIONS_PATH = "config/bug_class_dispositions.json"
ANCHOR = "bug-class-anchor/1"
REVIEW_DATE = "2026-09-13"

# The eight reviewed scanners. A different mode, class list or scanner set is
# a contract change: this pin, the policy and the registry move together.
EXPECTED_SCANNERS = {
    "dm01_raw_substring": ("gate", ("BC-01", "BC-02")),
    "dm16_config_key_reachability": ("report", ("BC-12", "BC-10")),
    "dm17_apply_without_guard": ("gate", ("BC-37",)),
    "dm18_except_returns_empty": ("gate", ("BC-20", "BC-47")),
    "dm29_phrase_append_signature": ("report", ("BC-76",)),
    "dm31_live_state_default": ("gate", ("BC-78", "BC-11", "BC-12")),
    "dm38_machinery_consumers": ("gate", ("BC-91", "BC-20", "BC-45")),
    "catalog": ("gate", ("BC-71",)),
}

EXPECTED_SCANNER_LEGS = {
    "dm01_raw_substring": ("python_source",),
    "dm16_config_key_reachability": (
        "dm16_config_yaml", "dm16_app_config", "dm16_schema", "dm16_consumers",
    ),
    "dm17_apply_without_guard": ("dm17_scripts", "dm17_tests"),
    "dm18_except_returns_empty": ("dm18_retrieval",),
    "dm29_phrase_append_signature": ("dm29_changelog",),
    "dm31_live_state_default": ("python_source",),
    "dm38_machinery_consumers": ("dm38_consumers",),
    "catalog": ("catalog_document",),
}

COMMON_SOURCE_ROOTS = (
    "core", "memory", "knowledge", "utils", "gui", "api", "models",
    "processing", "config", "scripts", "main.py",
)
DM16_CONSUMER_ROOTS = tuple(root for root in COMMON_SOURCE_ROOTS if root != "config")
DM18_ROOTS = ("memory", "knowledge", "core/prompt", "api")

EXPECTED_LEGS = {
    "python_source": ("python_tree", COMMON_SOURCE_ROOTS, True),
    "dm16_config_yaml": ("file", ("config/config.yaml",), True),
    "dm16_app_config": ("file", ("config/app_config.py",), True),
    "dm16_schema": ("file", ("config/schema.py",), False),
    "dm16_consumers": ("python_tree", DM16_CONSUMER_ROOTS, True),
    "dm17_scripts": ("python_flat", ("scripts",), True),
    "dm17_tests": ("python_tree", ("tests",), True),
    "dm18_retrieval": ("python_tree", DM18_ROOTS, True),
    "dm29_changelog": ("file", ("CLAUDE_CHANGELOG.md",), False),
    "dm38_consumers": ("python_tree", ("core", "gui", "utils"), True),
    "catalog_document": ("file", ("docs/BUG_CLASSES.md",), True),
}

EXPECTED_REPO_WIDE_GUARDS = [
    "tests/unit/test_no_git_state_in_tests.py",
    "tests/unit/test_ordered_slice_guard.py",
    "tests/unit/test_budget_meters_rendered_sections.py",
    "tests/unit/test_tool_wiring_parity.py",
    "tests/unit/test_model_capability_wiring.py",
]

BENIGN = '"""Synthetic module."""\n\nVALUE = 1\n'


def repo_files() -> dict[str, str]:
    """One benign file in every declared root: the smallest full-scan-green tree.

    Every coverage control removes or breaks exactly one piece of it.
    """
    return {
        "main.py": BENIGN,
        "api/__init__.py": BENIGN,
        "config/__init__.py": BENIGN,
        "config/app_config.py": DM16_APP_CONFIG,
        "config/config.yaml": DM16_YAML,
        "core/__init__.py": BENIGN,
        "core/prompt/__init__.py": BENIGN,
        "gui/__init__.py": BENIGN,
        "knowledge/__init__.py": BENIGN,
        "memory/__init__.py": BENIGN,
        "models/__init__.py": BENIGN,
        "processing/__init__.py": BENIGN,
        "scripts/tool.py": BENIGN,
        "tests/__init__.py": BENIGN,
        "utils/__init__.py": BENIGN,
        "docs/BUG_CLASSES.md": catalog_doc(
            [("BC-01", "partial"), ("BC-03", "open"), ("BC-37", "partial")]
        ),
    }


def build_repo(
    root: Path,
    extra: Mapping[str, str] | None = None,
    *,
    drop: tuple[str, ...] | list[str] = (),
    ledgers: bool = True,
) -> Path:
    """Minimal tree + the REAL policy + empty, valid ledgers."""
    build_tree(root, repo_files())
    if extra:
        build_tree(root, extra)
    for rel in drop:
        target = root / rel
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    (root / POLICY_PATH).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / POLICY_PATH, root / POLICY_PATH)
    if ledgers:
        write_ledgers(root)
    return root


def _entry_key(entry: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(entry[field] for field in ("scanner", "path", "symbol", "kind", "digest"))


def write_ledgers(root: Path, entries=(), records=(), *, legacy=None) -> None:
    baseline = {"schema": 2, "anchor": ANCHOR, "entries": sorted(entries, key=_entry_key)}
    ledger = {
        "schema": 1,
        "anchor": ANCHOR,
        "legacy_baseline": legacy,
        "records": list(records),
    }
    for rel, payload in ((BASELINE_PATH, baseline), (DISPOSITIONS_PATH, ledger)):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_ledgers(root: Path) -> tuple[dict, dict]:
    return (
        json.loads((root / BASELINE_PATH).read_text(encoding="utf-8")),
        json.loads((root / DISPOSITIONS_PATH).read_text(encoding="utf-8")),
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchor_of(scanner: str, finding: Mapping[str, object]) -> dict:
    return {
        "scanner": scanner,
        "path": finding["path"],
        "symbol": finding["symbol"],
        "kind": finding["kind"],
        "digest": finding["digest"],
    }


def record(
    root: Path,
    anchor: Mapping[str, str],
    ordinal: int,
    *,
    status: str = "accepted_debt",
    request: str | None = None,
    source_sha256: str | None = None,
) -> dict:
    return {
        "anchor": dict(anchor),
        "ordinal": ordinal,
        "legacy": None,
        "source_sha256": source_sha256 or sha256_file(root / anchor["path"]),
        "status": status,
        "assessment": "uncertain",
        "rationale": "Synthetic control; not a product review.",
        "reviewer": "synthetic-test",
        "review_date": REVIEW_DATE,
        "evidence": "tests/bug_class_guards",
        "request": request,
    }


def admit(root: Path, payload: Mapping[str, object]) -> None:
    """Record every gate finding of a scan report as reviewed accepted debt.

    The synthetic stand-in for a human review: tests use it to reach a green
    tree, then break one thing.
    """
    entries, records, seen = [], [], Counter()
    for scanner, data in payload["scanners"].items():
        if data["mode"] != "gate":
            continue
        for finding in data["findings"]:
            anchor = anchor_of(scanner, finding)
            seen[_entry_key(anchor)] += 1
            entries.append({**anchor, "excerpt": finding["excerpt"]})
            records.append(record(root, anchor, seen[_entry_key(anchor)]))
    write_ledgers(root, entries, records)


def run_scan(capsys, root: Path, *args: str):
    """Deployed CLI with --json: (exit, parsed report or None, stdout, stderr)."""
    import check_bug_classes

    code = check_bug_classes.main(["scan", "--root", str(root), "--json", *args])
    captured = capsys.readouterr()
    try:
        payload = json.loads(captured.out)
    except ValueError:
        payload = None
    return code, payload, captured.out, captured.err


def failure_codes(payload: Mapping[str, object] | None) -> set[str]:
    return {failure["code"] for failure in (payload or {}).get("failures", [])}


def leg_receipt(payload: Mapping[str, object], scanner: str, leg_id: str) -> dict:
    for leg in payload["scanners"][scanner]["legs"]:
        if leg["id"] == leg_id:
            return leg
    raise KeyError(f"{scanner} has no receipt for leg {leg_id}")


def root_status(leg: Mapping[str, object], root_path: str) -> str:
    for entry in leg["roots"]:
        if entry["path"] == root_path:
            return entry["status"]
    raise KeyError(f"leg {leg['id']} has no root {root_path}")
