"""
Repo-wide guard for BC-86 (undocumented function-body import): every import
inside a function body in the application packages must carry a
``# lazy import: <reason>`` marker from the closed vocabulary, on the import
line or on a comment-only line within the two lines above it. An unmarked function-body import is
either a load-bearing placement nobody recorded (cycle, patch point, live
config read, heavy load, optional dependency) or an accident — and nothing
distinguishes the two, so a later hoist breaks a contract and a later lazy
hides a dependency (a broken import fails mid-conversation instead of at
startup, and the code is read from disk after the branch may have moved).

Mode (2026-09-16): CEILING RATCHET. ``MAX_UNMARKED`` is the count on the
tree at the time it was last lowered. The first test fails when the count
GROWS past it; the second fails when the ceiling is left slack after a
hygiene batch (lower it to the new count). The import-hygiene plan
(``~/daemon_checkpoints/import_hygiene/PLAN_20260914_import_hygiene_v2.md``
§5 Phase 6) flips this to a content-anchored allowlist once the count is
zero. ``scripts/`` is REPORTED (printed under ``-s``), never gated.

The scan is self-contained (``ast`` over the working tree; no application
import, no git state). It mirrors the inventory tool's marker rule: reasons
are comma-separated, a free-text parenthetical may follow, and the legacy
shorthands in ``REASON_ALIASES`` are accepted until the MARK batches
normalize them.
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATED_DIRS = ("core", "gui", "api", "memory", "utils", "knowledge", "models", "processing", "config")
GATED_FILES = ("main.py",)
REPORTED_DIRS = ("scripts",)
SKIP_PARTS = {"__pycache__", "node_modules", "integration.bak"}

REASONS = frozenset({
    "startup-cost", "import-side-effect", "cycle", "live-config", "layering",
    "patch-point", "optional-dependency", "platform",
})
REASON_ALIASES = {
    "startup": "startup-cost", "patch": "patch-point", "patch point": "patch-point",
    "call-time": "live-config", "live": "live-config", "circular": "cycle",
    "optional": "optional-dependency", "side-effect": "import-side-effect",
}
MARKER_RE = re.compile(r"#\s*lazy import:\s*(?P<reasons>[^\n]*)")
LOOKBACK_LINES = 2

# Lowered by every import-hygiene batch; raised by nobody. See module docstring.
MAX_UNMARKED = 262  # 2026-09-19: the six try-only config imports resolved (five live app_config reads, one marked optional-dependency); the remainder is in accepted-debt files (plan v2 phase 5d)
MAX_INVALID = 0  # 0 since the 2026-09-16 MARK batches; a marker outside the vocabulary now fails
CEILING_SLACK = 25


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


def _function_imports(tree: ast.Module):
    seen: set[int] = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(fn):
                if isinstance(n, (ast.Import, ast.ImportFrom)) and id(n) not in seen:
                    seen.add(id(n))
                    yield fn, n


def _marker_reasons(lines: list[str], lineno: int) -> list[str] | None:
    """Reasons of the marker on the import line or up to two lines above; None when absent."""
    lo = max(0, lineno - 1 - LOOKBACK_LINES)
    for idx in range(lineno - 1, lo - 1, -1):
        if idx < lineno - 1 and not lines[idx].strip().startswith("#"):
            break  # only the import line itself or COMMENT-ONLY lines above may carry the marker;
                   # an adjacent import's trailing marker must not be inherited (2026-09-16 tool defect)
        m = MARKER_RE.search(lines[idx])
        if m:
            text = m.group("reasons").strip()
            text = re.sub(r"\(.*?\)", " ", text)  # free-text explanation in parentheses
            out = []
            for piece in text.split(","):
                piece = piece.strip().lower()
                if not piece:
                    continue
                token = piece.split()[0]
                out.append(REASON_ALIASES.get(piece, REASON_ALIASES.get(token, token)))
            return out
    return None


def scan(root: Path, dirs: tuple[str, ...], files: tuple[str, ...] = ()):
    """Return (unmarked, invalid) lists of (relpath, lineno, source_line)."""
    unmarked: list[tuple[str, int, str]] = []
    invalid: list[tuple[str, int, str]] = []
    for path in _python_files(root, dirs, files):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        lines = source.splitlines()
        rel = path.relative_to(root).as_posix()
        for _fn, node in _function_imports(tree):
            reasons = _marker_reasons(lines, node.lineno)
            line = lines[node.lineno - 1].strip()
            if reasons is None:
                unmarked.append((rel, node.lineno, line))
            elif not reasons or any(r not in REASONS for r in reasons):
                invalid.append((rel, node.lineno, line))
    return unmarked, invalid


def _report(unmarked, invalid, label: str) -> str:
    per_file = Counter(rel for rel, _, _ in unmarked)
    top = "\n".join(f"  {n:4d}  {rel}" for rel, n in per_file.most_common(15))
    bad = "\n".join(f"  {rel}:{ln}  {src}" for rel, ln, src in invalid[:20])
    return (f"[{label}] unmarked function-body imports: {len(unmarked)} in {len(per_file)} files; "
            f"invalid markers: {len(invalid)}\n{top}\n"
            + (f"invalid markers:\n{bad}\n" if invalid else ""))


def test_unmarked_function_body_imports_do_not_grow():
    unmarked, invalid = scan(REPO_ROOT, GATED_DIRS, GATED_FILES)
    scripts_unmarked, scripts_invalid = scan(REPO_ROOT, REPORTED_DIRS)
    print(_report(unmarked, invalid, "gated"))
    print(_report(scripts_unmarked, scripts_invalid, "scripts (reported only)"))
    offenders = "\n".join(f"  {rel}:{ln}  {src}" for rel, ln, src in unmarked[:40])
    assert len(unmarked) <= MAX_UNMARKED, (
        f"{len(unmarked)} unmarked function-body imports in app code, ceiling is {MAX_UNMARKED}. "
        "Either hoist the new import, convert it to a module-attribute read, or mark it "
        "`# lazy import: <reason>` with a reason from REASONS (see CLAUDE.md import doctrine). "
        f"First offenders:\n{offenders}"
    )
    assert len(invalid) <= MAX_INVALID, (
        f"{len(invalid)} function-body imports carry a marker outside the closed vocabulary "
        f"({sorted(REASONS)}), ceiling is {MAX_INVALID}:\n"
        + "\n".join(f"  {rel}:{ln}  {src}" for rel, ln, src in invalid)
    )


def test_ceiling_is_lowered_after_each_batch():
    unmarked, _ = scan(REPO_ROOT, GATED_DIRS, GATED_FILES)
    slack = MAX_UNMARKED - len(unmarked)
    assert slack <= CEILING_SLACK, (
        f"MAX_UNMARKED ({MAX_UNMARKED}) is {slack} above the measured count ({len(unmarked)}); "
        f"lower it to {len(unmarked)} in this batch so the ratchet only ever tightens."
    )
