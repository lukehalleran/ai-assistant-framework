"""
Repo-wide guard (2026-10-15, class: BC-30): no ``.switch_model(`` call may sit
inside an ``async def``. switch_model writes the process-wide active model, so a
call inside a coroutine changes the model for every concurrent coroutine.
Per-call routing goes through ``generate_async(route_model=...)`` /
``generate_once(model_name=...)``.

Allowed (the user's own model choice, not a per-turn side effect): api/routes/models.py,
gui/launch.py, main.py. tests/, data/, venv/ and scripts/ (single-threaded
startup tools) are not scanned. Ceiling 0.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKIP_TOP = {"tests", "data", "venv", "scripts", "node_modules", "web", "integration.bak"}
SKIP_PARTS = {"__pycache__", "node_modules", "venv", ".venv"}
ALLOWED = {"api/routes/models.py", "gui/launch.py", "main.py"}


def _hits(tree: ast.Module):
    out = []

    def visit(node, in_async):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.AsyncFunctionDef):
                visit(child, True)
            elif isinstance(child, (ast.FunctionDef, ast.Lambda)):
                # a sync def nested in a coroutine still runs on the loop when called
                visit(child, in_async)
            else:
                if (
                    in_async
                    and isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "switch_model"
                ):
                    out.append(child.lineno)
                visit(child, in_async)

    visit(tree, False)
    return out


def _scan():
    found = []
    for p in sorted(REPO_ROOT.rglob("*.py")):
        rel = p.relative_to(REPO_ROOT)
        if rel.parts[0] in SKIP_TOP or not SKIP_PARTS.isdisjoint(rel.parts):
            continue
        if rel.as_posix() in ALLOWED:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        found.extend(f"{rel.as_posix()}:{ln}" for ln in _hits(tree))
    return found


def test_no_switch_model_inside_coroutines():
    found = _scan()
    assert not found, (
        "switch_model() inside an async def mutates the process-wide model "
        "(BC-30); route per call with generate_async(route_model=...): " + ", ".join(found)
    )


def test_scanner_detects_a_coroutine_switch():
    tree = ast.parse("async def f(m):\n    m.switch_model('x')\n\ndef g(m):\n    m.switch_model('y')\n")
    assert _hits(tree) == [2]
