"""DM-31 — a PUBLIC function asserts live external state through a default.

BC-78 (exhausted budget enforced only at the leaf call), BC-11 (live setting
never reaches an already-built consumer) and BC-12 (config key never reaches
its runtime reader) keep arriving through the same code shape: a function
whose parameter carries live external state — a credit balance, a quota, a
Settings toggle — gives that parameter a LITERAL default. A caller that does
not pass it then silently inherits an assertion about the world.

The 2026-09-11 incident is the reference case. `analyze_for_web_search_llm`
declared `remaining_credits: float = 100` and `web_search_enabled: bool =
True`; the agentic gate passed neither. For three hours after Tavily's daily
limit was hit the gate classified as if 100 credits remained, which ALSO put
its call in a different cache bucket from the prompt builder's (87 classifier
calls / 12 cache hits in one session, contradictory verdicts 3 s apart on the
same message), and nothing in the routing path ever read the real budget.

Structural rule (all three must hold):
  1. the function is public — its name does not start with "_";
  2. a parameter's NAME is in the live-state vocabulary: the budget family
     (credits / budget / quota / remaining_*) or the availability family
     (*enabled / *available);
  3. its default is a literal that ASSERTS state — any number for the budget
     family, `True` for the availability family. `None` is not a finding:
     None is how this codebase spells "resolve it" (see
     `web_search_trigger._resolve_remaining_credits`), and `False` for a
     toggle is the fail-closed direction.

Deliberately NOT flagged (documented scope boundaries): private helpers
(their callers are in the same module and the resolving wrapper supplies the
value), caller-chosen sizes (`limit`, `max_*`, `estimated_*` are a request,
not a fact about the world), any non-literal default (a module constant such
as `DEFAULT_CREDITS`), and a live value read inside the body instead of
through a parameter.

Contract v2: one candidate per parameter, anchored on the parameter node
(name and annotation) plus its default.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from .common import (
    PYTHON_SOURCE_LEG,
    Finding,
    ScanResult,
    canonical,
    digest_of,
    function_spans,
    parse_module,
    read_source,
    relpath,
    resolve_leg,
    scope_for,
    source_line,
)

SCANNER_ID = "dm31_live_state_default"
CLASS_IDS = ("BC-78", "BC-11", "BC-12")
CONTRACT_VERSION = 2
LEGS = (PYTHON_SOURCE_LEG,)
KIND = "live_state_literal_default"
KINDS = (KIND,)

# Budget/quota family: a number here asserts how much of a metered external
# resource remains. `limit`/`max_*`/`estimated_*` are excluded by design —
# they are a caller's request, not a claim about the world.
_BUDGET_NAME_RE = re.compile(r"credits|budget|quota|^remaining_|_remaining$")

# Request-shaped qualifiers: the parameter names an amount the CALLER is
# asking for, not a fact about the world ("estimated_credits", "max_budget",
# "per_query_limit"). Any name carrying one of these is out of scope.
_REQUEST_QUALIFIER_RE = re.compile(
    r"^(?:estimated|max|min|default|per|requested|target)_|limit")

# Availability family: `True` here asserts a capability is on.
_TOGGLE_NAME_RE = re.compile(r"^(?:[a-z0-9_]*_)?(?:enabled|available)$")


def _asserts_state(name: str, default: ast.expr) -> bool:
    if not isinstance(default, ast.Constant):
        return False
    value = default.value
    if _REQUEST_QUALIFIER_RE.search(name):
        return False
    if _BUDGET_NAME_RE.search(name):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if _TOGGLE_NAME_RE.match(name):
        return value is True
    return False


def _params_with_defaults(node: ast.AST):
    """(arg, default) for every parameter that HAS a default."""
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    pairs = list(zip(positional[len(positional) - len(args.defaults):], args.defaults))
    pairs += [
        (arg, default)
        for arg, default in zip(args.kwonlyargs, args.kw_defaults)
        if default is not None
    ]
    return pairs


def scan(root: Path) -> ScanResult:
    resolved = resolve_leg(root, PYTHON_SOURCE_LEG)
    findings: list[Finding] = []
    for path in resolved.files:
        source = read_source(path)
        rel = relpath(root, path)
        tree = parse_module(path, source)
        spans = function_spans(tree)
        lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_") and node.name != "__init__":
                continue
            for arg, default in _params_with_defaults(node):
                if not _asserts_state(arg.arg, default):
                    continue
                line = getattr(arg, "lineno", node.lineno)
                findings.append(
                    Finding(
                        SCANNER_ID,
                        CLASS_IDS,
                        rel,
                        scope_for(spans, node.lineno),
                        line,
                        KIND,
                        digest_of([canonical(arg), canonical(default)]),
                        source_line(lines, line),
                        PYTHON_SOURCE_LEG.id,
                        span=(line, getattr(default, "end_lineno", None) or line),
                    )
                )
    findings.sort(key=lambda f: (f.path, f.line, f.excerpt))
    return ScanResult(findings, (resolved.receipt(),))
