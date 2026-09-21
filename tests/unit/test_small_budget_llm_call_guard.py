"""
Repo-wide guard for the small-budget-vs-reasoning-model class (see
docs/BUG_CLASSES.md BC-89).

Mechanism: a one-word/one-letter structured LLM call (a classifier, a
pairwise-ranking vote, a category label) is dialed to a tiny ``max_tokens``
on the assumption the model will spend it all on the answer. A
reasoning-capable model (kimi-k3 and others behind ``config.yaml
models.active``, which is user-selected and changed frequently — see
CLAUDE.md's Embedding model / model registry notes) spends the budget in its
internal reasoning channel FIRST; with nothing left for the visible answer it
returns empty, and the caller's exception/fallback handling silently routes
to the WRONG branch instead of raising loudly.

Incidents (docs/BUG_CLASSES.md BC-89): 2026-09-06 agentic decision calls sent
no reasoning key at all (kimi-k3 reasoned 45s/round) — fixed via
``generate_once(disable_reasoning=True)``; 2026-09-20
``DocumentGenerator.classify_deliverable`` was called with ``max_tokens=8``
and reasoning ON — kimi-k3 spent the whole budget reasoning, returned an
empty string, and the fail-safe silently chose the wrong deliverable branch
(fixed: ``max_tokens=64`` + ``disable_reasoning=True``).

Mode: CEILING RATCHET, mirroring ``tests/unit/test_import_hygiene_guard.py``.
``MAX_SMALL_BUDGET_SITES`` is the count measured when this guard was written
(2026-09-20); it only ever goes DOWN. A new site is either (a) given
``disable_reasoning=True``, (b) given more headroom (``max_tokens`` > 64), or
(c) reviewed and ratcheted in explicitly.

The scan is self-contained (``ast`` over the working tree; no application
import, no git state, mirrors ``test_import_hygiene_guard.py``'s directory
list). It flags a ``Call`` whose callee attribute/name is ``generate_once``
carrying a literal ``max_tokens`` keyword <= ``MAX_TOKENS_THRESHOLD`` and NO
``disable_reasoning=True`` keyword. A dynamic (non-literal) ``max_tokens``
value is not flagged — the scan only catches the structural landmine, not
every possible risky call.

``knowledge/document_generator.py`` is asserted separately: tonight's
incident lived there, so it gets a dedicated, tighter ceiling rather than
riding the whole-repo number. NOTE (2026-09-20 audit): this file was NOT
measured at zero. ``classify_deliverable`` (the site the incident was about)
carries ``disable_reasoning=True`` and is clean. ``_refine_topic`` (a topic
cleanup call around line ~571) still has ``max_tokens=30`` with no
``disable_reasoning`` keyword — a structurally identical, previously
unflagged site in the SAME file, found by this guard rather than fixed by
it (this test file may not edit knowledge/document_generator.py — a
concurrent batch owns that file). The document-generator ceiling below is
set to the MEASURED count (1), not the originally-requested zero, and this
docstring plus the test's own assertion message name the outstanding site so
it is not lost.
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATED_DIRS = ("core", "gui", "api", "memory", "utils", "knowledge", "models", "processing", "config")
GATED_FILES = ("main.py",)
SKIP_PARTS = {"__pycache__", "node_modules", "integration.bak"}

MAX_TOKENS_THRESHOLD = 64
TARGET_CALLEE = "generate_once"

# Lowered whenever a site is fixed (disable_reasoning=True added, or the
# budget raised past the threshold); raised by nobody. See module docstring.
MAX_SMALL_BUDGET_SITES = 4  # ratchet: only goes down (5 measured 2026-09-20; _refine_topic fixed same night)
DOCUMENT_GENERATOR_MAX = 0  # _refine_topic fixed 2026-09-20 (disable_reasoning=True)
CEILING_SLACK = 0  # a ratchet with slack lets new sites in


def _python_files(root: Path, dirs: tuple[str, ...], files: tuple[str, ...] = ()):
    for d in dirs:
        base = root / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            if SKIP_PARTS.isdisjoint(p.parts):
                yield p
    for f in files:
        p = root / f
        if p.is_file():
            yield p


def _callee_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _literal_int(node: ast.AST | None) -> int | float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    return None


def _is_disable_reasoning_true(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _small_budget_calls(tree: ast.Module):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _callee_name(node.func) != TARGET_CALLEE:
            continue
        max_tokens_val = None
        disable_reasoning_true = False
        for kw in node.keywords:
            if kw.arg == "max_tokens":
                max_tokens_val = _literal_int(kw.value)
            elif kw.arg == "disable_reasoning":
                disable_reasoning_true = _is_disable_reasoning_true(kw.value)
        if max_tokens_val is None or max_tokens_val > MAX_TOKENS_THRESHOLD:
            continue
        if disable_reasoning_true:
            continue
        yield node


def scan(root: Path, dirs: tuple[str, ...], files: tuple[str, ...] = ()):
    """Return [(relpath, lineno, source_line)] for every risky call site."""
    hits: list[tuple[str, int, str]] = []
    for path in _python_files(root, dirs, files):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        lines = source.splitlines()
        rel = path.relative_to(root).as_posix()
        for node in _small_budget_calls(tree):
            line = lines[node.lineno - 1].strip() if 0 <= node.lineno - 1 < len(lines) else ""
            hits.append((rel, node.lineno, line))
    return hits


def _report(hits, label: str) -> str:
    per_file = Counter(rel for rel, _, _ in hits)
    top = "\n".join(f"  {n:4d}  {rel}" for rel, n in per_file.most_common(15))
    sites = "\n".join(f"  {rel}:{ln}  {src}" for rel, ln, src in hits)
    return f"[{label}] small-budget generate_once() calls without disable_reasoning=True: {len(hits)}\n{top}\n{sites}\n"


def test_small_budget_generate_once_calls_do_not_grow():
    hits = scan(REPO_ROOT, GATED_DIRS, GATED_FILES)
    print(_report(hits, "gated"))
    sites = "\n".join(f"  {rel}:{ln}  {src}" for rel, ln, src in hits)
    assert len(hits) <= MAX_SMALL_BUDGET_SITES, (
        f"{len(hits)} generate_once() calls pass a literal max_tokens<={MAX_TOKENS_THRESHOLD} "
        f"with no disable_reasoning=True keyword, ceiling is {MAX_SMALL_BUDGET_SITES}. "
        "Failure mode (BC-89, docs/BUG_CLASSES.md): a reasoning-capable model spends the whole "
        "token budget in its reasoning channel and returns an EMPTY answer; the caller's "
        "fail-safe then silently routes to the wrong branch instead of raising. Fix by adding "
        "disable_reasoning=True, or raising max_tokens, or reviewing and ratcheting the ceiling "
        f"up explicitly. Offending sites:\n{sites}"
    )


def test_ceiling_is_lowered_after_each_batch():
    hits = scan(REPO_ROOT, GATED_DIRS, GATED_FILES)
    slack = MAX_SMALL_BUDGET_SITES - len(hits)
    assert slack <= CEILING_SLACK, (
        f"MAX_SMALL_BUDGET_SITES ({MAX_SMALL_BUDGET_SITES}) is {slack} above the measured count "
        f"({len(hits)}); lower it to {len(hits)} in this batch so the ratchet only ever tightens."
    )


def test_document_generator_has_no_new_small_budget_calls():
    """knowledge/document_generator.py is where tonight's incident lived
    (classify_deliverable, BC-89). It is asserted on its own, tighter ceiling
    rather than folded into the whole-repo number so a regression there is
    never masked by slack elsewhere. See module docstring: this is NOT zero
    yet — _refine_topic is a known, unfixed sibling site in this same file,
    left for the batch that owns knowledge/document_generator.py tonight."""
    hits = scan(REPO_ROOT, ("knowledge",), ())
    hits = [h for h in hits if h[0] == "knowledge/document_generator.py"]
    sites = "\n".join(f"  {rel}:{ln}  {src}" for rel, ln, src in hits)
    assert len(hits) <= DOCUMENT_GENERATOR_MAX, (
        f"{len(hits)} small-budget generate_once() calls without disable_reasoning=True in "
        f"knowledge/document_generator.py, ceiling is {DOCUMENT_GENERATOR_MAX} (the known "
        f"_refine_topic site). A NEW site appeared beyond the tracked one:\n{sites}"
    )
