"""DM-01 — raw-substring keyword tests outside the ``utils.trigger_match`` chokepoint.

BC-01 (bare-substring keyword matching) and BC-02 (negation-blind trigger
matching) have nine recorded incidents between them: ``"crisis"`` inside
``str(CrisisLevel.CONVERSATIONAL)``, ``'solve'`` inside "resolution",
``'how'`` inside "shower", ``'actions'`` inside "not taking any actions",
``"ice"`` inside "Price".  Every one was a keyword list matched by plain
``in`` against lowered text instead of through
``utils/trigger_match.compile_keyword_matcher`` / ``has_non_negated_hit``.

This scanner reports CANDIDATES, not verdicts: a module that already imports
the chokepoint is skipped entirely (it has adopted the closure and its
remaining raw tests are its own business), and the two AST shapes below are
the ones the incidents actually took.

Shape A  ``"literal" in <text>.lower()``            (a bare keyword constant)
Shape B  ``any(k in text_lower for k in HEAVY_KEYWORDS)``  (a keyword list)

Both require the right-hand side to be a lowered-text expression
(``<expr>.lower()`` or a name ending ``_lower`` / ``_lc``) — the shape that
says "I am matching keywords against user text", not an ordinary membership
test over a dict or set.
"""

from __future__ import annotations

import ast
import re
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

SCANNER_ID = "dm01_raw_substring"
CLASS_IDS = ("BC-01", "BC-02")

CHOKEPOINT = "trigger_match"

# A Name that reads as a keyword/cue vocabulary rather than a data container.
_LIST_NAME_RE = re.compile(
    r"(KEYWORDS?|CUES?|WORDS?|PHRASES?|MARKERS?|TERMS?|STARTERS?|SIGNALS?)$"
)
# Cheap prefilter: skip files with no lowered-text membership test at all.
_PREFILTER = re.compile(r"\bin\s+[A-Za-z_][A-Za-z_0-9.\[\]]*\s*\.lower\(\)|\bin\s+\w*_(lower|lc)\b")


def _is_lowered_text(node: ast.expr) -> bool:
    """``<expr>.lower()`` or a name ending ``_lower`` / ``_lc``."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr == "lower" and not node.args and not node.keywords
    if isinstance(node, ast.Name):
        return node.id.endswith("_lower") or node.id.endswith("_lc")
    if isinstance(node, ast.Attribute):
        return node.attr.endswith("_lower") or node.attr.endswith("_lc")
    return False


def _vocabulary_name(node: ast.expr) -> bool:
    """A Name/Attribute that looks like a keyword vocabulary constant."""
    if isinstance(node, ast.Name):
        return bool(_LIST_NAME_RE.search(node.id))
    if isinstance(node, ast.Attribute):
        return bool(_LIST_NAME_RE.search(node.attr))
    return False


def _membership_hit(node: ast.AST, vocabulary_vars: set[str]) -> bool:
    """True when ``node`` is ``<const|vocab-var> in <lowered text>``."""
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return False
    if not isinstance(node.ops[0], (ast.In, ast.NotIn)):
        return False
    if not _is_lowered_text(node.comparators[0]):
        return False
    left = node.left
    if isinstance(left, ast.Constant) and isinstance(left.value, str):
        return True
    return isinstance(left, ast.Name) and left.id in vocabulary_vars


def _comprehension_vars(node: ast.AST) -> set[str]:
    """Names bound by a comprehension over a keyword-vocabulary iterable."""
    bound: set[str] = set()
    generators = getattr(node, "generators", None)
    if not generators:
        return bound
    for generator in generators:
        if not _vocabulary_name(generator.iter):
            continue
        target = generator.target
        if isinstance(target, ast.Name):
            bound.add(target.id)
    return bound


def _imports_chokepoint(tree: ast.Module) -> bool:
    """Structural check: an actual import of ``utils.trigger_match``.

    A mention in a comment or docstring is not adoption; only an import is.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and CHOKEPOINT in (node.module or ""):
            return True
        if isinstance(node, ast.Import):
            if any(CHOKEPOINT in alias.name for alias in node.names):
                return True
    return False


def scan(root: Path) -> ScanResult:
    findings: list[Finding] = []
    files = iter_python_files(root)
    processed = 0
    for path in files:
        source = read_source(path)
        processed += 1
        if not _PREFILTER.search(source):
            continue
        rel = relpath(root, path)
        tree = parse_module(path, source)
        if _imports_chokepoint(tree):
            continue  # module has adopted the chokepoint
        spans = function_spans(tree)
        lines = source.splitlines()
        seen_lines: set[int] = set()
        for node in ast.walk(tree):
            bound = _comprehension_vars(node)
            candidates: list[ast.AST] = []
            if bound:
                candidates.extend(
                    child for child in ast.walk(node) if isinstance(child, ast.Compare)
                )
            elif isinstance(node, ast.Compare):
                candidates.append(node)
            for candidate in candidates:
                if not _membership_hit(candidate, bound):
                    continue
                lineno = candidate.lineno
                if lineno in seen_lines:
                    continue
                seen_lines.add(lineno)
                findings.append(
                    Finding(
                        SCANNER_ID,
                        CLASS_IDS,
                        rel,
                        scope_for(spans, lineno),
                        lineno,
                        source_line(lines, lineno),
                    )
                )
    findings.sort(key=lambda f: (f.path, f.line, f.text))
    return ScanResult(findings, processed)
