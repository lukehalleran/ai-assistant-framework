"""No scripts/ module may mutate the process environment at import time.

Incident 2026-10-23: scripts/model_bakeoff.py and scripts/budget_experiment.py
set ``os.environ["WEB_SEARCH_ENABLED"] = "0"`` at module level; a test that
imported them switched web search off for the whole pytest process and the
pre-push unit pass failed hundreds of web tests. Both were moved under
``if __name__ == "__main__"``; this AST scan keeps the class closed.
Re-checked 2026-10-26 (lane H1): zero module-level writes remain.

Class: BC-58 (a side effect reachable from import rather than from the
entry point); sibling of the BC-87 import-order startup side effects.

Module level = anything not inside a def/class body. ``if __name__ == "__main__"``
bodies are exempt (the entry point). Detected: ``os.environ[...] = ``/``del``,
``os.environ.setdefault/update/pop/clear``, ``os.putenv``/``os.unsetenv``.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"

# (relative path, stripped source line) -> reason. Empty on purpose: add an
# entry only with a reason a reviewer can check (e.g. a script that is never
# imported and says so).
ALLOWLIST: dict[tuple[str, str], str] = {}

_MUTATORS = {"setdefault", "update", "pop", "clear", "popitem"}


def _is_environ(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "environ"
            and isinstance(node.value, ast.Name) and node.value.id == "os")


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    t = node.test
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
            and t.left.id == "__name__" and len(t.comparators) == 1
            and isinstance(t.comparators[0], ast.Constant)
            and t.comparators[0].value == "__main__")


def _mutation(node: ast.AST) -> bool:
    targets = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        targets = [node.target]
    elif isinstance(node, ast.Delete):
        targets = node.targets
    if any(isinstance(t, ast.Subscript) and _is_environ(t.value) for t in targets):
        return True
    if isinstance(node, ast.Call):
        fn = node.func
        if isinstance(fn, ast.Attribute):
            if _is_environ(fn.value) and fn.attr in _MUTATORS:
                return True
            if (isinstance(fn.value, ast.Name) and fn.value.id == "os"
                    and fn.attr in ("putenv", "unsetenv")):
                return True
    return False


def _walk_module_level(node: ast.AST):
    """Yield nodes reachable without entering a def/class or a __main__ guard."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef, ast.Lambda)):
            continue
        if _is_main_guard(child):
            continue
        yield child
        yield from _walk_module_level(child)


def find_import_time_writes(source: str, rel: str):
    tree = ast.parse(source)
    lines = source.splitlines()
    return [(rel, n.lineno, lines[n.lineno - 1].strip())
            for n in _walk_module_level(tree) if _mutation(n)]


def test_no_script_mutates_environ_at_import():
    bad = []
    for p in sorted(SCRIPTS.glob("*.py")):
        rel = str(p.relative_to(REPO_ROOT))
        for r, ln, text in find_import_time_writes(p.read_text(encoding="utf-8"), rel):
            if (r, text) not in ALLOWLIST:
                bad.append(f"{r}:{ln}: {text}")
    assert not bad, (
        "scripts/ module mutates os.environ at import time (move it into "
        "main() or under `if __name__ == \"__main__\":`):\n  " + "\n  ".join(bad))


def test_allowlist_entries_have_reasons():
    assert all(len(r) > 20 for r in ALLOWLIST.values())


class TestDetector:
    def test_flags_module_level_assignment(self):
        assert find_import_time_writes('import os\nos.environ["X"] = "0"\n', "s.py")

    def test_flags_nested_in_module_level_if_and_try(self):
        src = 'import os\ntry:\n    os.environ.setdefault("X", "1")\nexcept Exception:\n    pass\n'
        assert find_import_time_writes(src, "s.py")

    def test_flags_putenv_and_update(self):
        assert find_import_time_writes('import os\nos.putenv("A", "1")\n', "s.py")
        assert find_import_time_writes('import os\nos.environ.update({"A": "1"})\n', "s.py")

    def test_main_guard_and_functions_exempt(self):
        src = ('import os\n'
               'def f():\n    os.environ["X"] = "0"\n'
               'if __name__ == "__main__":\n    os.environ["Y"] = "1"\n')
        assert find_import_time_writes(src, "s.py") == []

    def test_reads_are_not_writes(self):
        assert find_import_time_writes('import os\nv = os.environ.get("X")\nw = os.environ["Y"]\n', "s.py") == []
