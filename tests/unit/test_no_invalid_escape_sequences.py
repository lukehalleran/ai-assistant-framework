"""Guard: no tracked-style Python source may contain an invalid escape sequence.

A non-raw string or docstring with a bare backslash before a non-escape
character (``\\/``, ``\\ ``) is a DeprecationWarning/SyntaxWarning today and a
SyntaxError in a future Python. 2026-09-19 found two (a module docstring and a
test docstring); the warning only ever appeared once per process in a test-run
summary nobody read. This compiles every source file with warnings as errors.
"""
from __future__ import annotations

import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("core", "memory", "knowledge", "utils", "gui", "api", "models", "processing",
             "config", "agent_branch", "eval", "scripts", "tests")
EXTRA_FILES = ("main.py",)
_SKIP_PARTS = {"__pycache__", "node_modules", "venv", ".venv", "data", "integration.bak"}


def _sources():
    for name in EXTRA_FILES:
        path = REPO_ROOT / name
        if path.exists():
            yield path
    for d in SCAN_DIRS:
        base = REPO_ROOT / d
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if not _SKIP_PARTS.intersection(path.parts):
                yield path


def invalid_escape_sites(paths):
    sites = []
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                compile(source, str(path), "exec")
            except SyntaxError as exc:  # a future Python: the warning is already an error
                if "invalid escape" in str(exc):
                    sites.append(f"{path.relative_to(REPO_ROOT)}:{exc.lineno}: {exc.msg}")
                continue
        for w in caught:
            if "invalid escape sequence" in str(w.message):
                sites.append(f"{path.relative_to(REPO_ROOT)}:{w.lineno}: {w.message}")
    return sites


def test_detector_flags_a_bare_backslash(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text('X = "a\\/b"\n', encoding="utf-8")
    good = tmp_path / "good.py"
    good.write_text('X = r"a\\/b"\nY = "c\\\\d"\n', encoding="utf-8")
    global REPO_ROOT
    original, REPO_ROOT = REPO_ROOT, tmp_path
    try:
        sites = invalid_escape_sites([bad, good])
    finally:
        REPO_ROOT = original
    assert len(sites) == 1 and sites[0].startswith("bad.py:1")


def test_no_invalid_escape_sequences_in_the_repo():
    sites = invalid_escape_sites(_sources())
    assert not sites, (
        "invalid escape sequence(s) — use a raw string or double the backslash:\n  "
        + "\n  ".join(sites)
    )
