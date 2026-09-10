"""Repo-wide guard (2026-09-09, audit T01 closure): a test that cannot fail is
not a test.

The 2026-09-09 audit found eleven bodies in one file wrapped in
``except Exception: assert True`` and one that ``return False``-ed on failure;
a corrupt-provider probe showed 27 undetected cases across 13 bodies — the
tests had stayed green through an interface change they never exercised.
This guard makes CI reject the pattern at introduction:

* a bare ``assert True`` anywhere in a test module;
* an ``except Exception`` / bare ``except:`` handler whose body only
  SWALLOWS (``pass``, ``assert True``, ``return``/``return False``/``None``,
  ``continue``, a bare ``...``) with no ``raise``, ``pytest.fail``,
  ``pytest.skip``/``xfail``, real assertion, or logging of the failure.

Legitimate handlers (cleanup, resource-availability skips, explicit
``pytest.fail``) pass because they do something with the exception.
A justified exception goes in ALLOWLIST keyed by (path, enclosing function)
with a reason; stale entries fail so the list cannot rot.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_DIRS = [ROOT / "tests"]
EXCLUDE_PARTS = {"integration.bak", "node_modules", "__pycache__"}

# (relative path, enclosing function name) -> reason
#
# T01 backlog (2026-09-09) closed the same day: all 31 pre-existing vacuous
# bodies found on this guard's first run were repaired, rewritten as explicit
# contracts, or deleted as fossils. See CLAUDE_CHANGELOG.md / the T01 entry in
# docs/PLAN_20260909_audit_repairs.md for the per-function disposition table.
ALLOWLIST: dict[tuple[str, str], str] = {}

_SWALLOW_STMTS = (ast.Pass, ast.Continue)


def _is_assert_true(node: ast.AST) -> bool:
    return (isinstance(node, ast.Assert)
            and isinstance(node.test, ast.Constant)
            and node.test.value is True)


def _handler_swallows(handler: ast.ExceptHandler) -> bool:
    """True when the handler body does nothing observable with the failure."""
    for stmt in handler.body:
        if isinstance(stmt, _SWALLOW_STMTS):
            continue
        if _is_assert_true(stmt):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue  # docstring / bare literal / Ellipsis
        if isinstance(stmt, ast.Return):
            v = stmt.value
            if v is None or (isinstance(v, ast.Constant) and v.value in (False, None)):
                continue
            return False
        return False  # anything else (raise, pytest.fail, assert x, log, call…)
    return True


def _catches_broad(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    names = []
    for n in ast.walk(handler.type):
        if isinstance(n, ast.Name):
            names.append(n.id)
        elif isinstance(n, ast.Attribute):
            names.append(n.attr)
    return any(n in ("Exception", "BaseException") for n in names)


def _test_function_names(tree: ast.AST) -> set[str]:
    """Names of functions that are TESTS (``test_*``). Helpers, fixtures and
    availability probes may swallow — an embedder that is not installed is a
    skip condition, not a failure — so only test bodies are policed."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            names.add(node.name)
    return names


def _enclosing_function(tree: ast.AST) -> dict[int, str]:
    """Map every line number to the innermost enclosing function name."""
    spans: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((node.lineno, node.end_lineno or node.lineno, node.name))
    spans.sort(key=lambda s: (s[0], -s[1]))
    out: dict[int, str] = {}
    for start, end, name in spans:
        for ln in range(start, end + 1):
            out[ln] = name  # later (inner) spans overwrite outer ones
    return out


def scan(paths=TEST_DIRS):
    findings: list[tuple[str, str, int, str]] = []
    for base in paths:
        for path in sorted(base.rglob("*.py")):
            if EXCLUDE_PARTS & set(path.parts):
                continue
            if path.name == Path(__file__).name:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            try:
                rel = str(path.relative_to(ROOT))
            except ValueError:
                rel = str(path.relative_to(base.parent))
            func_at = _enclosing_function(tree)
            test_names = _test_function_names(tree)
            for node in ast.walk(tree):
                func = func_at.get(getattr(node, "lineno", -1), "<module>")
                if func not in test_names:
                    continue  # helpers / fixtures / module level: not policed
                if _is_assert_true(node):
                    findings.append((rel, func, node.lineno, "bare `assert True`"))
                elif isinstance(node, ast.ExceptHandler) and _catches_broad(node) \
                        and _handler_swallows(node):
                    findings.append((rel, func, node.lineno, "broad except that only swallows"))
    return findings


def test_no_vacuous_assertions_in_tests():
    findings = scan()
    used: set[tuple[str, str]] = set()
    offenders = []
    for rel, func, lineno, kind in findings:
        key = (rel, func)
        if key in ALLOWLIST:
            used.add(key)
            continue
        offenders.append(f"{rel}:{lineno} in {func}(): {kind}")
    stale = set(ALLOWLIST) - used
    assert not offenders, (
        "Tests that cannot fail (fix the test — do not add to ALLOWLIST without a reason):\n  "
        + "\n  ".join(offenders)
    )
    assert not stale, f"Remove stale ALLOWLIST entries: {sorted(stale)}"


def test_guard_recognises_the_audit_shapes(tmp_path):
    bad = tmp_path / "tests"; bad.mkdir()
    (bad / "test_bad.py").write_text(
        "import pytest\n"
        "async def test_a(gen):\n"
        "    try:\n        r = await gen.go()\n        assert r\n"
        "    except Exception:\n        assert True\n"
        "def test_b():\n"
        "    try:\n        1/0\n    except:\n        return False\n"
        "def test_ok():\n"
        "    try:\n        1/0\n    except Exception as e:\n        pytest.fail(str(e))\n"
        "def test_cleanup():\n"
        "    try:\n        1/0\n    except Exception:\n        raise\n"
    )
    found = scan([bad])
    kinds = sorted((f[1], f[3]) for f in found)
    assert kinds == [("test_a", "bare `assert True`"),
                     ("test_a", "broad except that only swallows"),
                     ("test_b", "broad except that only swallows")]
