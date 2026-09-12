"""DM-17 — store-writing surfaces that can fight the live daemon (BC-37).

Two legs, both from the catalog's own Find line:

Leg 1 ``scripts/*.py`` carrying ``--apply`` but referencing neither
``daemon_guard`` nor ``daemon_running``.  A live Daemon holds its JSON stores
in memory and re-saves them on shutdown, so an ``--apply`` run against a
running instance is silently clobbered (2026-08-05 curated profile writes;
2026-09-05 ``graph_junk_cleanup.py`` still had no guard).

Leg 2 string constants starting with ``data/`` inside ``tests/`` (outside
``tests/conftest.py``, which owns the sandbox redirects).  A test that names
a production store path writes the owner's live data (2026-08-22 pytest seeded
the live tone floor; 2026-09-01 a lazily-importing test wrote prod
``pending_actions.json``).
"""

from __future__ import annotations

import ast
from pathlib import Path

from .common import (
    Finding,
    ScanResult,
    clip,
    function_spans,
    iter_python_files,
    parse_module,
    read_source,
    relpath,
    scope_for,
    source_line,
)

SCANNER_ID = "dm17_apply_without_guard"
CLASS_IDS = ("BC-37",)

GUARD_TOKENS = ("daemon_guard", "daemon_running")
APPLY_TOKEN = "--apply"

TEST_ROOTS = ("tests",)
CONFTEST_EXEMPT = "tests/conftest.py"
DATA_PREFIX = "data/"


def _scan_apply_scripts(root: Path) -> tuple[list[Finding], int]:
    findings: list[Finding] = []
    scripts = root / "scripts"
    processed = 0
    if not scripts.is_dir():
        return findings, processed
    for path in sorted(scripts.glob("*.py"), key=lambda p: p.name):
        source = read_source(path)
        processed += 1
        if APPLY_TOKEN not in source:
            continue
        if any(token in source for token in GUARD_TOKENS):
            continue
        first = next(
            (line for line in source.splitlines() if APPLY_TOKEN in line), APPLY_TOKEN
        )
        findings.append(
            Finding(SCANNER_ID, CLASS_IDS, relpath(root, path), "", 1, clip(first))
        )
    return findings, processed


def _scan_test_data_paths(root: Path) -> tuple[list[Finding], int]:
    findings: list[Finding] = []
    processed = 0
    for path in iter_python_files(root, TEST_ROOTS):
        rel = relpath(root, path)
        processed += 1
        if rel == CONFTEST_EXEMPT:
            continue
        source = read_source(path)
        if DATA_PREFIX not in source:
            continue
        tree = parse_module(path, source)
        spans = function_spans(tree)
        lines = source.splitlines()
        seen: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if not node.value.startswith(DATA_PREFIX):
                continue
            if node.lineno in seen:
                continue
            seen.add(node.lineno)
            findings.append(
                Finding(
                    SCANNER_ID,
                    CLASS_IDS,
                    rel,
                    scope_for(spans, node.lineno),
                    node.lineno,
                    source_line(lines, node.lineno),
                )
            )
    return findings, processed


def scan(root: Path) -> ScanResult:
    script_findings, script_files = _scan_apply_scripts(root)
    test_findings, test_files = _scan_test_data_paths(root)
    findings = script_findings + test_findings
    findings.sort(key=lambda f: (f.path, f.line, f.text))
    return ScanResult(findings, script_files + test_files)
