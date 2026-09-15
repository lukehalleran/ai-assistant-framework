"""DM-01 — raw-substring keyword tests outside the ``utils.trigger_match`` chokepoint.

BC-01 (bare-substring keyword matching) and BC-02 (negation-blind trigger
matching) have nine recorded incidents between them: ``"crisis"`` inside
``str(CrisisLevel.CONVERSATIONAL)``, ``'solve'`` inside "resolution",
``'how'`` inside "shower", ``'actions'`` inside "not taking any actions",
``"ice"`` inside "Price".  Every one was a keyword list matched by plain
``in`` against lowered text instead of through
``utils/trigger_match.compile_keyword_matcher`` / ``has_non_negated_hit``.

This scanner reports CANDIDATES, not verdicts, in the two AST shapes the
incidents actually took:

Shape A  ``"literal" in <text>.lower()``            (a bare keyword constant)
Shape B  ``any(k in text_lower for k in HEAVY_KEYWORDS)``  (a keyword list)

Both require the right-hand side to be a lowered-text expression
(``<expr>.lower()`` or a name/attribute ending ``_lower`` / ``_lc``) — the
shape that says "I am matching keywords against user text", not an ordinary
membership test over a dict or set.

Contract v2 (2026-09-13): no module exemption and no lexical prefilter.
Importing — or even calling — the chokepoint says nothing about the other raw
tests in the same module, and the old ``in <name>.lower()`` regex prefilter
missed ``in normalize(text).lower()`` and ``in self.text_lower``.  Every file
in the leg is parsed (the CLI's syntax preflight has already parsed it).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from .common import (
    PYTHON_SOURCE_LEG,
    ScanResult,
    canonical,
    line_group_findings,
    parse_module,
    read_source,
    relpath,
    resolve_leg,
)

SCANNER_ID = "dm01_raw_substring"
CLASS_IDS = ("BC-01", "BC-02")
CONTRACT_VERSION = 2
LEGS = (PYTHON_SOURCE_LEG,)
KIND = "raw_membership"
KINDS = (KIND,)

# A Name that reads as a keyword/cue vocabulary rather than a data container.
_LIST_NAME_RE = re.compile(
    r"(KEYWORDS?|CUES?|WORDS?|PHRASES?|MARKERS?|TERMS?|STARTERS?|SIGNALS?)$"
)


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
    for generator in getattr(node, "generators", None) or ():
        if _vocabulary_name(generator.iter) and isinstance(generator.target, ast.Name):
            bound.add(generator.target.id)
    return bound


def scan(root: Path) -> ScanResult:
    resolved = resolve_leg(root, PYTHON_SOURCE_LEG)
    findings = []
    for path in resolved.files:
        source = read_source(path)
        tree = parse_module(path, source)
        groups: dict[int, list[tuple[int, str, str]]] = {}
        claimed: set[int] = set()
        for node in ast.walk(tree):
            bound = _comprehension_vars(node)
            if bound:
                # Shape B: the candidate is the whole comprehension, so the
                # vocabulary it iterates is part of the anchor.
                for child in ast.walk(node):
                    if id(child) in claimed or not _membership_hit(child, bound):
                        continue
                    claimed.add(id(child))
                    groups.setdefault(child.lineno, []).append(
                        (child.col_offset, KIND, canonical(node), node.lineno, node.end_lineno or node.lineno)
                    )
            elif id(node) not in claimed and _membership_hit(node, set()):
                claimed.add(id(node))
                groups.setdefault(node.lineno, []).append(
                    (node.col_offset, KIND, canonical(node), node.lineno, node.end_lineno or node.lineno)
                )
        findings.extend(
            line_group_findings(
                SCANNER_ID, CLASS_IDS, relpath(root, path), tree,
                source.splitlines(), groups, PYTHON_SOURCE_LEG.id,
            )
        )
    findings.sort(key=lambda f: (f.path, f.line, f.excerpt))
    return ScanResult(findings, (resolved.receipt(),))
