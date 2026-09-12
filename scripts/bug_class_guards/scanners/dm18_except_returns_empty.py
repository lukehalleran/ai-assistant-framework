"""DM-18 — a broad ``except`` returning an empty result beside a store call.

BC-20 (writer/reader shape mismatch swallowed to an empty result) and BC-47
(failure or not-run collapsed into a valid empty result) share one code
shape: a retrieval function wraps a store/API call in ``except Exception``
and returns ``[]``/``{}``/``None``.  The caller — and, downstream, the model —
then cannot tell "there are no rows" from "the read raised".
``get_ids_by_timestamp_range`` raised on EVERY call from the day it shipped
and nobody noticed, because the caller saw an ordinary empty list.

Structural rule (all three must hold):
  1. the handler is broad — bare ``except:``, ``except Exception``, or
     ``except BaseException``;
  2. its LAST statement returns an empty literal (so a handler that re-raises,
     or logs then re-raises, is never flagged);
  3. the enclosing function makes a store-shaped call — an attribute call whose
     name contains query/get_/search/collection/retrieve/load/fetch.

Scoped to the retrieval layers where the incidents happened.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .common import (
    Finding,
    ScanResult,
    function_spans,
    iter_python_files,
    parse_module,
    read_source,
    relpath,
    scope_for,
    source_line,
)

SCANNER_ID = "dm18_except_returns_empty"
CLASS_IDS = ("BC-20", "BC-47")

ROOTS = ("memory", "knowledge", "core/prompt", "api")

_STORE_CALL_FRAGMENTS = (
    "query",
    "get_",
    "search",
    "collection",
    "retrieve",
    "load",
    "fetch",
)

_BROAD_EXCEPTION_NAMES = {"Exception", "BaseException"}


def _is_broad_handler(handler: ast.ExceptHandler) -> bool:
    node = handler.type
    if node is None:
        return True
    if isinstance(node, ast.Name):
        return node.id in _BROAD_EXCEPTION_NAMES
    if isinstance(node, ast.Tuple):
        return any(
            isinstance(elt, ast.Name) and elt.id in _BROAD_EXCEPTION_NAMES
            for elt in node.elts
        )
    return False


def _is_empty_literal(node: ast.expr | None) -> bool:
    """``[]``, ``{}``, ``""``, ``()``, ``None``, ``set()`` (or a bare return)."""
    if node is None:
        return True  # bare `return` — returns None, same swallow
    if isinstance(node, ast.Constant):
        return node.value is None or node.value == ""
    if isinstance(node, (ast.List, ast.Tuple)):
        return not node.elts
    if isinstance(node, ast.Dict):
        return not node.keys
    if isinstance(node, ast.Set):
        return False  # `{x}` is never empty; `set()` is a Call
    if isinstance(node, ast.Call):
        return (
            isinstance(node.func, ast.Name)
            and node.func.id in {"set", "dict", "list", "tuple"}
            and not node.args
            and not node.keywords
        )
    return False


def _has_store_call(func: ast.AST) -> bool:
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if name and any(part in name for part in _STORE_CALL_FRAGMENTS):
            return True
    return False


def _functions(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def scan(root: Path) -> ScanResult:
    findings: list[Finding] = []
    files = iter_python_files(root, ROOTS)
    processed = 0
    for path in files:
        source = read_source(path)
        processed += 1
        if "except" not in source:
            continue
        rel = relpath(root, path)
        tree = parse_module(path, source)
        spans = function_spans(tree)
        lines = source.splitlines()
        seen: set[int] = set()
        for func in _functions(tree):
            if not _has_store_call(func):
                continue
            for node in ast.walk(func):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                if not _is_broad_handler(node) or not node.body:
                    continue
                last = node.body[-1]
                if not isinstance(last, ast.Return) or not _is_empty_literal(last.value):
                    continue
                if last.lineno in seen:
                    continue
                seen.add(last.lineno)
                findings.append(
                    Finding(
                        SCANNER_ID,
                        CLASS_IDS,
                        rel,
                        scope_for(spans, last.lineno),
                        last.lineno,
                        source_line(lines, last.lineno),
                    )
                )
    findings.sort(key=lambda f: (f.path, f.line, f.text))
    return ScanResult(findings, processed)
