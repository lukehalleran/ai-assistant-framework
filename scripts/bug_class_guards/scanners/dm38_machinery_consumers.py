"""DM-38 — BC-91 read-time machinery at audited consumer sites + emitter registry.

Bounded structural gate, four rules, all AST (a comment or a bare import
cannot satisfy them). Mutation controls: ``tests/bug_class_guards/test_scanners.py``.

1. builder — ``core/prompt/builder.py``: both memory filler functions
   (``_topup_filler``, ``_recency_floor_filler``) canonicalize EVERY candidate
   through ``_canonical_turn_key`` (raw ``.get("query")`` identity re-admitted
   a 14 K paste three turns running, 2026-09-22).
2. delivery pipeline — ``gui/handlers.py``: each audited delivery site
   (``_run_agentic_search``, ``_run_enhanced``) hands a PRE-SUFFIX snapshot
   (never a name that a ``*_suffix`` append rebound) to ONE ordered pipeline
   ``_apply_delivery_revisions`` and never calls a checker directly; inside the
   pipeline both checkers run on clean text and the personal-claim check reads
   the grounding-REVISED body (sequential revisions must compose — BC-45).
3. render — ``core/prompt/formatter.py``: the three assistant-response render
   sites rebind their text through ``strip_delivery_notices`` before
   annotating it.
4. emitter registry — everywhere in the leg except ``utils/read_time_markers.py``:
   no ``"> ⚠️"`` literal, and every ``delivery_notice(...)`` call names a
   registered ``NOTICE_*`` constant as its opening. A notice family that a
   consumer can strip is by construction one an emitter registered; a new
   emitter cannot bypass the leaf.

This is not whole-program taint analysis: a consumer shape outside these
sites still needs the BC-58 sibling review.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from .common import Finding, Leg, ScanResult, digest_of, parse_module, read_source, relpath, resolve_leg

SCANNER_ID = "dm38_machinery_consumers"
CLASS_IDS = ("BC-91", "BC-20", "BC-45")
CONTRACT_VERSION = 2
LEG = Leg("dm38_consumers", "python_tree", ("core", "gui", "utils"), True)
LEGS = (LEG,)
KIND = "machinery_consumer_bypass"
KINDS = (KIND,)

_BUILDER_PATH = "core/prompt/builder.py"
_HANDLERS_PATH = "gui/handlers.py"
_FORMATTER_PATH = "core/prompt/formatter.py"
_LEAF_PATH = "utils/read_time_markers.py"

_CANONICAL = "_canonical_turn_key"
_STRIPPERS = frozenset({"strip_delivery_notices", "strip_machinery"})
_PIPELINE = "_apply_delivery_revisions"
_GROUNDING = "_apply_grounding_check_for_delivery"
_PERSONAL = "_apply_personal_claim_check_for_delivery"
_CHECKERS = frozenset({_GROUNDING, _PERSONAL})
_SITE_FUNCS = ("_run_agentic_search", "_run_enhanced")
_NOTICE_FACTORY = "delivery_notice"
_NOTICE_PREFIX = "> ⚠️"
_NOTICE_CONST_PREFIX = "NOTICE_"
# A name that carries appended machinery: the action-guard suffix, a notice.
_SUFFIX_NAME_RE = re.compile(r"(?:_suffix|_notice|NOTICE)$")
_MAX_CHAIN = 8

_BUILDER_SITES = {
    "_topup_filler": frozenset({"recents", "mems", "extra_recent"}),
    "_recency_floor_filler": frozenset({"recent_convos", "stored_recent"}),
}
_FORMATTER_SITES = {
    "_format_memory": "response",
    "mem_parts": "r",
    "_assemble_prompt": "last_a",
}


# ------------------------------------------------------------------ helpers


def _dotted_tail(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _store_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def _unwrap(node: ast.AST) -> ast.AST:
    return node.value if isinstance(node, ast.Await) else node


def _func_nodes(tree: ast.AST) -> dict[str, list[ast.AST]]:
    out: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(node.name, []).append(node)
    return out


def _param_names(function: ast.AST) -> set[str]:
    a = function.args
    names = [p.arg for p in a.posonlyargs + a.args + a.kwonlyargs]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    return set(names)


def _assignments_to(function: ast.AST, name: str) -> list[ast.stmt]:
    out = []
    for stmt in ast.walk(function):
        if isinstance(stmt, ast.Assign) and any(name in _store_names(t) for t in stmt.targets):
            out.append(stmt)
        elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)) and name in _store_names(stmt.target):
            out.append(stmt)
        elif isinstance(stmt, ast.NamedExpr) and name in _store_names(stmt.target):
            out.append(stmt)
    return out


def _suffix_expr(node: ast.AST) -> bool:
    """A concatenation that glues a suffix/notice name or a notice factory call."""
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and _SUFFIX_NAME_RE.search(n.id):
            return True
        if isinstance(n, ast.Call) and _dotted_tail(n.func) == _NOTICE_FACTORY:
            return True
    return False


def _arg_is_clean(function: ast.AST, arg: ast.AST, at_line: int, depth: int = 0) -> bool:
    """True when ``arg`` provably carries no delivery suffix at ``at_line``:
    a stripper/checker/pipeline result, a string literal, a parameter, or a
    name whose LAST binding before ``at_line`` chains (through plain
    ``x = y`` copies, ``a or b`` fallbacks, or suffix-free sanitizer calls)
    to one of those, with no ``+ *_suffix``/notice rebinding on the way. Anything else is unproven → not clean."""
    if depth > _MAX_CHAIN:
        return False
    arg = _unwrap(arg)
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return True
    if isinstance(arg, ast.Call):
        return not _suffix_expr(arg)
    if not isinstance(arg, ast.Name):
        return False
    prior = [s for s in _assignments_to(function, arg.id) if getattr(s, "lineno", 0) < at_line]
    if not prior:
        return arg.id in _param_names(function)
    last = max(prior, key=lambda s: s.lineno)
    if isinstance(last, ast.AugAssign):
        # ``x += <card render>`` keeps x clean only when the appended
        # expression names no suffix/notice and x was clean before it.
        if _suffix_expr(last.value):
            return False
        return _arg_is_clean(function, ast.Name(id=arg.id, ctx=ast.Load()), last.lineno, depth + 1)
    value = _unwrap(last.value)
    if isinstance(value, ast.BinOp):
        return False
    if isinstance(value, ast.Call):
        # A sanitizer/stripper/checker call that names no suffix, notice or
        # notice factory anywhere in its expression keeps the text clean. A
        # helper that hides a suffix under an unrelated name is outside this
        # bounded gate (BC-58 sibling review), by design.
        return not _suffix_expr(value)
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return True
    if isinstance(value, ast.Name):
        return _arg_is_clean(function, value, last.lineno, depth + 1)
    if isinstance(value, ast.BoolOp):
        return all(
            isinstance(v, ast.Name) and _arg_is_clean(function, v, last.lineno, depth + 1)
            for v in value.values
        )
    return False


def _text_arg(call: ast.Call) -> ast.AST | None:
    return call.args[1] if len(call.args) >= 2 else None


def _calls_named(function: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(function) if isinstance(n, ast.Call) and _dotted_tail(n.func) == name]


def _make_finding(path: str, symbol: str, node: ast.AST | None, detail: str) -> Finding:
    line = getattr(node, "lineno", 1)
    return Finding(
        SCANNER_ID,
        CLASS_IDS,
        path,
        symbol,
        line,
        KIND,
        digest_of([path, symbol, detail]),
        detail,
        LEG.id,
        span=(line, getattr(node, "end_lineno", None) or line),
    )


# --------------------------------------------------------- rule 1: builder


def _canonical_for_target(node: ast.AST, target: str) -> bool:
    return any(
        isinstance(n, ast.Call)
        and _dotted_tail(n.func) == _CANONICAL
        and len(n.args) == 1
        and isinstance(n.args[0], ast.Name)
        and n.args[0].id == target
        for n in ast.walk(node)
    )


def _comparison_uses_target(node: ast.AST, target: str) -> bool:
    return any(isinstance(n, ast.Compare) and _canonical_for_target(n, target) for n in ast.walk(node))


def _builder_complete(node: ast.AST, expected_sources: frozenset[str]) -> bool:
    seen: set[str] = set()
    parents = {child: parent for parent in ast.walk(node) for child in ast.iter_child_nodes(parent)}
    for child in ast.walk(node):
        if isinstance(child, ast.comprehension):
            sources = _names(child.iter) & expected_sources
            targets = _store_names(child.target)
            parent = parents.get(child)
            if sources and isinstance(parent, (ast.SetComp, ast.GeneratorExp)) and any(
                _canonical_for_target(parent.elt, target) for target in targets
            ):
                seen.update(sources)
        elif isinstance(child, ast.For):
            sources = _names(child.iter) & expected_sources
            targets = _store_names(child.target)
            if not sources or not targets:
                continue
            for source in sources:
                for target in targets:
                    # Either the loop compares the canonical key directly, or it
                    # binds the canonical key to a name the loop then compares.
                    if any(isinstance(s, ast.If) and _comparison_uses_target(s.test, target) for s in ast.walk(child)):
                        seen.add(source)
                        break
                    bound = {
                        dest
                        for assign in ast.walk(child)
                        if isinstance(assign, ast.Assign) and _canonical_for_target(assign.value, target)
                        for dest in _store_names(assign.targets[0])
                    }
                    if bound and any(
                        isinstance(s, ast.If) and (_names(s.test) & bound) for s in ast.walk(child)
                    ):
                        seen.add(source)
                        break
    return seen == set(expected_sources)


def _scan_builder(tree: ast.AST, path: str) -> list[Finding]:
    findings = []
    functions = _func_nodes(tree)
    for name, sources in _BUILDER_SITES.items():
        candidates = functions.get(name, [])
        if len(candidates) != 1 or not _builder_complete(candidates[0], sources):
            findings.append(_make_finding(path, name, candidates[0] if candidates else None,
                f"{name} must canonicalize every candidate read from {', '.join(sorted(sources))}"))
    return findings


# ------------------------------------------------ rule 2: delivery pipeline


def _scan_delivery_sites(tree: ast.AST, path: str) -> list[Finding]:
    findings = []
    functions = _func_nodes(tree)
    for site in _SITE_FUNCS:
        candidates = functions.get(site, [])
        if len(candidates) != 1:
            findings.append(_make_finding(path, site, candidates[0] if candidates else None,
                f"expected one audited delivery site {site}"))
            continue
        function = candidates[0]
        pipeline_calls = _calls_named(function, _PIPELINE)
        if not pipeline_calls:
            findings.append(_make_finding(path, site, function,
                f"{site} no longer routes delivery revisions through {_PIPELINE}"))
        for call in pipeline_calls:
            arg = _text_arg(call)
            if arg is None or not _arg_is_clean(function, arg, call.lineno):
                findings.append(_make_finding(path, site, call,
                    f"{site} passes delivery-suffixed or unproven text to {_PIPELINE}; pass the pre-suffix snapshot"))
        for checker in sorted(_CHECKERS):
            for call in _calls_named(function, checker):
                findings.append(_make_finding(path, site, call,
                    f"{site} calls {checker} directly, bypassing the ordered {_PIPELINE} pipeline"))
    return findings


def _scan_pipeline(tree: ast.AST, path: str) -> list[Finding]:
    findings = []
    candidates = _func_nodes(tree).get(_PIPELINE, [])
    if len(candidates) != 1:
        findings.append(_make_finding(path, _PIPELINE, candidates[0] if candidates else None,
            f"expected exactly one ordered delivery pipeline {_PIPELINE}"))
        return findings
    function = candidates[0]
    grounding = _calls_named(function, _GROUNDING)
    personal = _calls_named(function, _PERSONAL)
    for checker, calls in ((_GROUNDING, grounding), (_PERSONAL, personal)):
        if not calls:
            findings.append(_make_finding(path, _PIPELINE, function,
                f"{_PIPELINE} no longer runs the audited {checker} check"))
        for call in calls:
            arg = _text_arg(call)
            if arg is None or not _arg_is_clean(function, arg, call.lineno):
                findings.append(_make_finding(path, _PIPELINE, call,
                    f"{_PIPELINE} passes delivery-suffixed or unproven text to {checker}"))
    if grounding and personal:
        g_call = min(grounding, key=lambda c: c.lineno)
        # Names bound from the grounding result (``grounded, _ = await ...``).
        result_names: set[str] = set()
        for stmt in ast.walk(function):
            if isinstance(stmt, ast.Assign) and any(n is g_call for n in ast.walk(stmt.value)):
                for t in stmt.targets:
                    result_names |= _store_names(t)
        for p_call in personal:
            arg = _text_arg(p_call)
            ok = False
            if isinstance(arg, ast.Name) and p_call.lineno > g_call.lineno:
                if arg.id in result_names:
                    ok = True
                else:
                    for stmt in _assignments_to(function, arg.id):
                        value = _unwrap(getattr(stmt, "value", stmt))
                        if (g_call.lineno < stmt.lineno < p_call.lineno
                                and isinstance(value, ast.Name) and value.id in result_names):
                            ok = True
                            break
            if not ok:
                findings.append(_make_finding(path, _PIPELINE, p_call,
                    f"{_PERSONAL} in {_PIPELINE} does not read the body revised by {_GROUNDING}; sequential revisions must compose"))
    return findings


# --------------------------------------------------------- rule 3: render


def _has_strip_rebind(function: ast.AST, variable: str) -> bool:
    for stmt in ast.walk(function):
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            continue
        value = stmt.value
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        if not isinstance(value, ast.Call) or _dotted_tail(value.func) not in _STRIPPERS:
            continue
        if not value.args or not isinstance(value.args[0], ast.Name) or value.args[0].id != variable:
            continue
        if any(variable in _store_names(target) for target in targets):
            return True
    return False


def _scan_formatter(tree: ast.AST, path: str) -> list[Finding]:
    findings = []
    functions = _func_nodes(tree)
    for name, variable in _FORMATTER_SITES.items():
        candidates = functions.get(name, [])
        if len(candidates) != 1 or not _has_strip_rebind(candidates[0], variable):
            findings.append(_make_finding(path, name, candidates[0] if candidates else None,
                f"{name} must rebind {variable} through strip_delivery_notices before rendering/annotating it"))
    return findings


# ------------------------------------------------ rule 4: emitter registry


def _enclosing_symbols(tree: ast.AST) -> dict[ast.AST, str]:
    out: dict[ast.AST, str] = {}

    def visit(node: ast.AST, symbol: str) -> None:
        for child in ast.iter_child_nodes(node):
            child_symbol = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else symbol
            out[child] = child_symbol
            visit(child, child_symbol)

    visit(tree, "<module>")
    return out


def _scan_emitters(tree: ast.AST, path: str) -> list[Finding]:
    findings = []
    symbols = _enclosing_symbols(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _NOTICE_PREFIX in node.value:
            findings.append(_make_finding(path, symbols.get(node, "<module>"), node,
                f"delivery-notice literal outside {_LEAF_PATH}; emit through {_NOTICE_FACTORY}(NOTICE_*)"))
        elif isinstance(node, ast.Call) and _dotted_tail(node.func) == _NOTICE_FACTORY:
            opening = node.args[0] if node.args else None
            if not (opening is not None and _dotted_tail(opening).startswith(_NOTICE_CONST_PREFIX)):
                findings.append(_make_finding(path, symbols.get(node, "<module>"), node,
                    f"{_NOTICE_FACTORY} opening must be a registered NOTICE_* constant from {_LEAF_PATH}"))
    return findings


# ------------------------------------------------------------------- scan


def scan(root: Path) -> ScanResult:
    resolved = resolve_leg(root, LEG)
    findings: list[Finding] = []
    for source_path in resolved.files:
        path = relpath(root, source_path)
        source = read_source(source_path)
        tree = parse_module(source_path, source)
        if path == _BUILDER_PATH:
            findings.extend(_scan_builder(tree, path))
        elif path == _HANDLERS_PATH:
            findings.extend(_scan_delivery_sites(tree, path))
            findings.extend(_scan_pipeline(tree, path))
        elif path == _FORMATTER_PATH:
            findings.extend(_scan_formatter(tree, path))
        if path != _LEAF_PATH:
            findings.extend(_scan_emitters(tree, path))
    findings.sort(key=lambda f: (f.path, f.line, f.symbol, f.excerpt))
    return ScanResult(findings, (resolved.receipt(),))
