"""DM-17 — store-writing surfaces that can fight the live daemon (BC-37).

Two legs, both from the catalog's own Find line, counted independently:

Leg ``dm17_scripts`` (``scripts/*.py``): a script that declares or parses the
``--apply`` flag in code must refuse while a live Daemon holds its stores.  A
live Daemon keeps its JSON stores in memory and re-saves them on shutdown, so
an ``--apply`` run against a running instance is silently clobbered
(2026-08-05 curated profile writes; 2026-09-05 ``graph_junk_cleanup.py`` still
had no guard).

Leg ``dm17_tests`` (``tests/**/*.py``): string constants starting with
``data/``.  A test that names a production store path writes the owner's live
data (2026-08-22 pytest seeded the live tone floor; 2026-09-01 a
lazily-importing test wrote prod ``pending_actions.json``).  Only the exact
root ``tests/conftest.py`` is exempt: it owns the sandbox redirects.

Contract v2 (2026-09-13): guard TEXT is not guard EVIDENCE.  A comment, a
string, a docstring, or an unused import mentioning ``daemon_guard`` proves
nothing.  The reviewed shapes this scanner accepts are:

* a guard call — ``daemon_running()`` imported from ``utils.daemon_guard``,
  or a module-level wrapper whose body returns that call (directly or inside
  a ``try``);
* inside a guard ``if`` whose test is the call (alone or as an operand of a
  top-level ``and``) and whose body refuses (``return``/``raise``/an exit call);
* reachable (not under a constant-false branch, not after an unconditional
  exit in its block), and structurally BEFORE every statement in its function
  that consumes the ``--apply`` value: a positive ``.apply`` test, a call
  passing the value, or a call handing the args object to a module function
  that is not itself guarded.  A guard inside a ``try`` whose body before it
  holds only imports also covers the statements after that ``try`` (the
  repository's ImportError-tolerant shape).

A script with no guard call at all is ``apply_without_guard``.  A script with
a guard call that does not satisfy those shapes is ``apply_guard_unresolved``
— an unresolved candidate for a human, never a proof of safety.

Documented scope boundaries: prose-only mentions of ``--apply`` and flag names
built dynamically are not apply flags; ``os.path.join("data", ...)`` and
``Path("data") / ...`` in tests are not ``data/`` literals.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .common import (
    Exemption,
    Finding,
    Leg,
    ScanResult,
    canonical,
    clip,
    digest_of,
    docstring_ids,
    line_group_findings,
    parse_module,
    read_source,
    relpath,
    resolve_leg,
    source_line,
)

SCANNER_ID = "dm17_apply_without_guard"
CLASS_IDS = ("BC-37",)
CONTRACT_VERSION = 2

CONFTEST_EXEMPT = "tests/conftest.py"
CONFTEST_REASON = (
    "The root conftest owns the sandbox redirects that point production store "
    "paths at tmp directories, so its data/ literals are the isolation "
    "mechanism rather than a test writing a live store. Exact path only; "
    "nested conftest files are scanned."
)
SCRIPTS_LEG = Leg("dm17_scripts", "python_flat", ("scripts",), True)
TESTS_LEG = Leg(
    "dm17_tests", "python_tree", ("tests",), True, (Exemption(CONFTEST_EXEMPT, CONFTEST_REASON),)
)
LEGS = (SCRIPTS_LEG, TESTS_LEG)

KIND_UNGUARDED = "apply_without_guard"
KIND_UNRESOLVED = "apply_guard_unresolved"
KIND_TEST_LITERAL = "test_data_path_literal"
KINDS = (KIND_UNGUARDED, KIND_UNRESOLVED, KIND_TEST_LITERAL)

GUARD_MODULE = "utils.daemon_guard"
GUARD_FUNCTIONS = frozenset({"daemon_running"})
APPLY_FLAG = "--apply"
DATA_PREFIX = "data/"

_EXIT_ATTRS = frozenset({"exit", "_exit", "error"})
_EXIT_NAMES = frozenset({"exit", "quit"})
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _dotted(node.value)
        return f"{head}.{node.attr}" if head else ""
    return ""


def _is_exit_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in _EXIT_NAMES or func.id == "SystemExit"
    return isinstance(func, ast.Attribute) and func.attr in _EXIT_ATTRS


def _refuses(stmt: ast.stmt) -> bool:
    if isinstance(stmt, (ast.Return, ast.Raise)):
        return True
    return isinstance(stmt, ast.Expr) and _is_exit_call(stmt.value)


def _ends_block(stmt: ast.stmt) -> bool:
    return _refuses(stmt) or isinstance(stmt, (ast.Break, ast.Continue))


def _walk_expr(node: ast.AST):
    """ast.walk that does not descend into nested scopes."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        for child in ast.iter_child_nodes(current):
            if not isinstance(child, _SCOPE_NODES):
                stack.append(child)


def _child_blocks(stmt: ast.stmt):
    """(statement list, reachable) for each nested block of a statement."""
    if isinstance(stmt, ast.If):
        test = stmt.test
        if isinstance(test, ast.Constant):
            return [(stmt.body, bool(test.value)), (stmt.orelse, not test.value)]
        return [(stmt.body, True), (stmt.orelse, True)]
    if isinstance(stmt, ast.While):
        live = not (isinstance(stmt.test, ast.Constant) and not stmt.test.value)
        return [(stmt.body, live), (stmt.orelse, True)]
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return [(stmt.body, True), (stmt.orelse, True)]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return [(stmt.body, True)]
    if isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
        blocks = [(stmt.body, True)]
        blocks += [(handler.body, True) for handler in stmt.handlers]
        return blocks + [(stmt.orelse, True), (stmt.finalbody, True)]
    if type(stmt).__name__ == "Match":
        return [(case.body, True) for case in stmt.cases]
    return []


def _headers(stmt: ast.stmt) -> list[ast.AST]:
    """The expressions a compound statement evaluates itself."""
    if isinstance(stmt, (ast.If, ast.While)):
        return [stmt.test]
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return [stmt.iter]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return [item.context_expr for item in stmt.items]
    if isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
        return []
    if type(stmt).__name__ == "Match":
        return [stmt.subject]
    return [stmt]


class _ScriptAnalysis:
    """Guard-evidence verdict for one ``scripts/*.py`` module."""

    def __init__(self, tree: ast.Module):
        self.tree = tree
        docs = docstring_ids(tree)
        self.apply_flags = sorted(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and node.value == APPLY_FLAG
                and id(node) not in docs
            ),
            key=lambda node: (node.lineno, node.col_offset),
        )
        self.guard_names: set[str] = set()
        self.module_aliases: set[str] = {GUARD_MODULE}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == GUARD_MODULE:
                for alias in node.names:
                    if alias.name in GUARD_FUNCTIONS:
                        self.guard_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "utils":
                for alias in node.names:
                    if alias.name == "daemon_guard":
                        self.module_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == GUARD_MODULE and alias.asname:
                        self.module_aliases.add(alias.asname)
        self.functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.wrappers = {
            name for name, func in self.functions.items() if self._returns_guard(func.body)
        }
        self.carriers = {
            node.value.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "apply"
            and isinstance(node.value, ast.Name)
        }
        self.apply_aliases = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and self._is_apply_read(node.value)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self._self_guarded: dict[str, bool] = {}

    # -- guard recognition ------------------------------------------------

    def _is_direct_guard_call(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if isinstance(func, ast.Name):
            return func.id in self.guard_names
        if isinstance(func, ast.Attribute) and func.attr in GUARD_FUNCTIONS:
            return _dotted(func.value) in self.module_aliases
        return False

    def _returns_guard(self, body: list[ast.stmt]) -> bool:
        for stmt in body:
            if isinstance(stmt, ast.Return) and self._is_direct_guard_call(stmt.value):
                return True
            if isinstance(stmt, ast.Try) and self._returns_guard(stmt.body):
                return True
        return False

    def is_guard_call(self, node: ast.AST) -> bool:
        if self._is_direct_guard_call(node):
            return True
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self.wrappers
        )

    def has_guard_call(self) -> bool:
        return any(self.is_guard_call(node) for node in ast.walk(self.tree))

    def _is_guard_stmt(self, stmt: ast.stmt) -> bool:
        if not isinstance(stmt, ast.If):
            return False
        test = stmt.test
        operands = test.values if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) else [test]
        if not any(self.is_guard_call(operand) for operand in operands):
            return False
        return any(_refuses(inner) for inner in stmt.body)

    # -- apply consumption --------------------------------------------------

    def _is_apply_read(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute):
            return node.attr == "apply" and isinstance(node.ctx, ast.Load)
        if isinstance(node, ast.Name):
            return node.id in getattr(self, "apply_aliases", ()) and isinstance(node.ctx, ast.Load)
        if isinstance(node, ast.Call):
            return (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "apply"
            )
        if isinstance(node, ast.Subscript):
            return isinstance(node.slice, ast.Constant) and node.slice.value == "apply"
        if isinstance(node, ast.Compare):
            constants = [node.left, *node.comparators]
            return any(isinstance(c, ast.Constant) and c.value == APPLY_FLAG for c in constants)
        return False

    def _contains_apply_read(self, node: ast.AST) -> bool:
        return any(self._is_apply_read(child) for child in _walk_expr(node))

    def _positive_apply_read(self, node: ast.AST, negated: bool = False) -> bool:
        if self._is_apply_read(node):
            return not negated
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self._positive_apply_read(node.operand, not negated)
        return any(
            self._positive_apply_read(child, negated)
            for child in ast.iter_child_nodes(node)
            if not isinstance(child, _SCOPE_NODES)
        )

    def _consumption(self, stmt: ast.stmt) -> tuple[bool, list[str]]:
        """(consumes directly, module callees handed the args object)."""
        direct = False
        callees: list[str] = []
        if isinstance(stmt, (ast.If, ast.While)):
            if self._positive_apply_read(stmt.test):
                direct = True
            elif stmt.orelse and self._contains_apply_read(stmt.test):
                direct = True  # `if not args.apply: ... else: <writes>`
        for header in _headers(stmt):
            for node in _walk_expr(header):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                    continue
                arguments = list(node.args) + [keyword.value for keyword in node.keywords]
                if any(self._contains_apply_read(argument) for argument in arguments):
                    direct = True
                if any(isinstance(a, ast.Name) and a.id in self.carriers for a in arguments):
                    name = node.func.id if isinstance(node.func, ast.Name) else ""
                    if name in self.functions:
                        callees.append(name)
                    else:
                        direct = True
        return direct, callees

    # -- structural dominance ---------------------------------------------

    def _collect(self, body, prefix, reachable, guards, consumers):
        ended = False
        for index, stmt in enumerate(body):
            path = prefix + ((id(body), index),)
            live = reachable and not ended
            if isinstance(stmt, _SCOPE_NODES):
                continue
            if live and self._is_guard_stmt(stmt):
                guards.append(path)
            else:
                direct, callees = self._consumption(stmt)
                if direct or callees:
                    consumers.append((path, direct, callees))
            for child, child_live in _child_blocks(stmt):
                if not child:
                    continue
                before = len(guards)
                self._collect(child, path, live and child_live, guards, consumers)
                if (
                    isinstance(stmt, ast.Try)
                    and child is stmt.body
                    and any(
                        len(g) == len(path) + 1
                        and g[-1][0] == id(child)
                        and all(isinstance(s, (ast.Import, ast.ImportFrom)) for s in child[: g[-1][1]])
                        for g in guards[before:]
                    )
                ):
                    guards.append(path)  # covers the statements after this try
            if _ends_block(stmt):
                ended = True

    @staticmethod
    def _dominates(guard, consumer) -> bool:
        depth = len(guard) - 1
        if len(consumer) <= depth or guard[:depth] != consumer[:depth]:
            return False
        return guard[depth][0] == consumer[depth][0] and consumer[depth][1] > guard[depth][1]

    def _scope_facts(self, body):
        guards: list = []
        consumers: list = []
        self._collect(body, (), True, guards, consumers)
        return guards, consumers

    def self_guarded(self, name: str) -> bool:
        if name in self._self_guarded:
            return self._self_guarded[name]
        self._self_guarded[name] = False  # recursion guard
        guards, consumers = self._scope_facts(self.functions[name].body)
        verdict = bool(guards) and all(
            self._covered(guards, consumer) for consumer in consumers
        )
        self._self_guarded[name] = verdict
        return verdict

    def _covered(self, guards, consumer) -> bool:
        path, direct, callees = consumer
        if any(self._dominates(guard, path) for guard in guards):
            return True
        return not direct and all(self.self_guarded(callee) for callee in callees)

    def verdict(self) -> str | None:
        if not self.apply_flags:
            return None
        if not self.has_guard_call():
            return KIND_UNGUARDED
        scopes = [[stmt for stmt in self.tree.body if not isinstance(stmt, _SCOPE_NODES)]]
        scopes += [
            node.body
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        any_guard = False
        for body in scopes:
            guards, consumers = self._scope_facts(body)
            any_guard = any_guard or bool(guards)
            if not all(self._covered(guards, consumer) for consumer in consumers):
                return KIND_UNRESOLVED
        return None if any_guard else KIND_UNRESOLVED

    def flag_expression(self) -> tuple[int, str, tuple[int, int]]:
        """(line, canonical rendering, span) of the first ``--apply`` flag's call."""
        flag = self.apply_flags[0]
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Call, ast.Compare)) and any(
                child is flag for child in ast.walk(node)
            ):
                enclosing = node
                break
        else:
            enclosing = flag
        # Smallest enclosing Call/Compare: walk again from that node inward.
        for node in ast.walk(enclosing):
            if node is not enclosing and isinstance(node, (ast.Call, ast.Compare)) and any(
                child is flag for child in ast.walk(node)
            ):
                enclosing = node
        span = (enclosing.lineno, getattr(enclosing, "end_lineno", None) or enclosing.lineno)
        return flag.lineno, canonical(enclosing), span


def _scan_apply_scripts(root: Path) -> tuple[list[Finding], object]:
    resolved = resolve_leg(root, SCRIPTS_LEG)
    findings: list[Finding] = []
    for path in resolved.files:
        source = read_source(path)
        analysis = _ScriptAnalysis(parse_module(path, source))
        kind = analysis.verdict()
        if kind is None:
            continue
        line, expression, span = analysis.flag_expression()
        lines = source.splitlines()
        legacy_first = next((raw for raw in lines if APPLY_FLAG in raw), APPLY_FLAG)
        findings.append(
            Finding(
                SCANNER_ID,
                CLASS_IDS,
                relpath(root, path),
                "<module>",
                line,
                kind,
                digest_of([expression]),
                source_line(lines, line),
                SCRIPTS_LEG.id,
                unresolved=kind == KIND_UNRESOLVED,
                legacy_symbol="",
                legacy_text=clip(legacy_first),
                span=span,
            )
        )
    unresolved = sum(1 for finding in findings if finding.unresolved)
    return findings, resolved.receipt(unresolved)


def _scan_test_data_paths(root: Path) -> tuple[list[Finding], object]:
    resolved = resolve_leg(root, TESTS_LEG)
    findings: list[Finding] = []
    for path in resolved.files:
        rel = relpath(root, path)
        if rel in TESTS_LEG.exempt_paths:
            continue
        source = read_source(path)
        tree = parse_module(path, source)
        groups: dict[int, list[tuple[int, str, str]]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith(DATA_PREFIX):
                    groups.setdefault(node.lineno, []).append(
                        (node.col_offset, KIND_TEST_LITERAL, canonical(node), node.lineno, node.end_lineno or node.lineno)
                    )
        findings.extend(
            line_group_findings(
                SCANNER_ID, CLASS_IDS, rel, tree, source.splitlines(), groups, TESTS_LEG.id
            )
        )
    return findings, resolved.receipt()


def scan(root: Path) -> ScanResult:
    script_findings, script_receipt = _scan_apply_scripts(root)
    test_findings, test_receipt = _scan_test_data_paths(root)
    findings = script_findings + test_findings
    findings.sort(key=lambda f: (f.path, f.line, f.excerpt))
    return ScanResult(findings, (script_receipt, test_receipt))
