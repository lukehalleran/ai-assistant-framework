"""Persistent writers must not independently derive a shared temp filename."""
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("api", "core", "gui", "knowledge", "memory", "models", "utils", "scripts", "config", "eval", "agent_branch")
# An exception needs (relative path, qualified function), exact expression and
# a reason. No exceptions are currently needed; stale ones must fail too.
ALLOWLIST = {}


def temp_sites(source):
    sites = []

    class Visitor(ast.NodeVisitor):
        scope = []

        def visit_FunctionDef(self, node):
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

        def visit_Expr(self, node):
            if not isinstance(node.value, ast.Constant):  # ignore docstrings
                self.generic_visit(node)

        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value.endswith(".tmp"):
                sites.append((".".join(self.scope) or "<module>", node.value))

    Visitor().visit(ast.parse(source))
    return sites


def test_no_shared_temp_filename_writers_outside_the_shared_helper():
    paths = list(ROOT.glob("*.py"))
    for directory in SOURCE_DIRS:
        paths.extend((ROOT / directory).rglob("*.py"))
    unexpected, used = [], set()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if relative == "utils/safe_json.py":
            continue
        for function, anchor in temp_sites(path.read_text(encoding="utf-8")):
            key = (relative, function)
            exception = ALLOWLIST.get(key)
            if exception and exception["anchor"] == anchor and exception["reason"]:
                used.add(key)
            else:
                unexpected.append((relative, function, anchor))
    assert not unexpected, f"Use the shared atomic writer: {unexpected}"
    assert used == set(ALLOWLIST), "Remove stale atomic-writer exceptions"


def test_guard_recognizes_concat_fstring_and_suffix_forms():
    source = '''
def example(path):
    one = path + ".tmp"
    two = f"{path}.tmp"
    three = path.with_suffix(".md.tmp")
'''
    assert temp_sites(source) == [("example", ".tmp"), ("example", ".tmp"), ("example", ".md.tmp")]
