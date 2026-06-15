# agent_branch/eval_static.py
"""
Layer 1a — the STATIC gate. Runs FIRST and gates everything else.

Module Contract
- Purpose: decide whether a worker's diff may proceed, using ONLY static
  analysis — never importing, running, or evaluating any worker code. This is
  the layer that protects the host: if a diff touches a test / conftest / safety
  path, or strays outside its allowed scope, it is killed here BEFORE the dynamic
  layer (eval_sandboxed) ever applies or executes it.
- Inputs:
  - ``diff_text``: a unified diff the SUPERVISOR computed itself from the worker
    clone (never a worker-authored patch file).
  - ``manifest``: the BranchManifest (allowed/forbidden paths, caps).
  - ``run_stats`` (optional): commits/cycles/tokens/wallclock actually consumed.
  - ``guard_log`` / ``audit_log`` (optional): untrusted log TEXT surfaced from the
    container, scanned for destructive-op / secrets-access markers (corroborating
    evidence only — the container is the real boundary).
- Output: a ``GateResult`` (killed?, high_risk?, reasons, flags, parsed paths).
  All violations are collected (not short-circuited) so the report is complete.
- Side effects: none. Pure functions; safe to call on untrusted input.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from agent_branch.manifest import BranchManifest
from utils.logging_utils import get_logger

logger = get_logger("agent_branch.eval_static")


# Paths that are not merely forbidden but indicate an attempt to weaken the
# evaluation itself — touching any of these is HIGH RISK, not just a kill.
_HIGH_RISK_PATTERNS: List[str] = [
    "tests/",
    "conftest.py",
    "eval/",
    "utils/destructive_op_guard.py",
    "utils/shell_cmd_guard.py",
    "utils/python_fs_guard.py",
    "utils/fs_snapshot.py",
    "scripts/safe_git.sh",
    "scripts/safe_cmd.sh",
    "scripts/bin/",
    "pytest.ini",
    "agent_branch/",
]

# Vendored / generated noise. A diff dominated by these is killed (workers must
# not bury a real change inside generated churn).
_VENDORED_PATTERNS: List[str] = [
    "vendor/",
    "node_modules/",
    "dist/",
    "build/",
    "__pycache__/",
    ".egg-info/",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "*.min.js",
    "*.min.css",
    "*.pyc",
    "*.map",
]

_VENDORED_DOMINANCE = 0.5   # >50% of changed lines in vendored paths → kill
_VENDORED_FLOOR = 10        # ...only once the diff is non-trivial

# "Fake success" surface: config defaults and benchmark/eval thresholds a worker
# could edit to make the supervisor's check pass without doing the real work
# (M2 D2). Reported as a distinct CONFIG_TAMPER reason, always high-risk.
_CONFIG_TAMPER_PATTERNS: List[str] = [
    "config/",
    "config.yaml",
    "config.local.yaml",
    # Every file pytest resolves config from — editing any can forge a passing
    # proof (e.g. ``addopts = --collect-only`` runs zero tests but exits 0, or
    # ``-p <worker_plugin>`` loads worker code into the test session).
    "pytest.ini",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    # .gitattributes can set ``*.py -diff`` to hide added lines from the
    # supervisor's diff (showing "Binary files differ"), or wire a clean/smudge
    # filter — a diff-content/line-count evasion surface, not just a config one.
    ".gitattributes",
    "feature_registry.yaml",
    "tests/benchmarks/",
    "tests/fixtures/",
    "*.threshold",
]

# Best-effort markers in untrusted container logs.
_DESTRUCTIVE_MARKERS = ("BLOCKED destructive", "destructive op", "git restore", "rm -rf", "rmtree")
_SECRETS_MARKERS = ("secrets-access", "read .env", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "/.env")


class GateFlag(str, Enum):
    EMPTY_DIFF = "empty_diff"
    WHITESPACE_ONLY = "whitespace_only"
    FORBIDDEN_PATH = "forbidden_path"
    SAFETY_TOUCHED = "safety_touched"
    CONFIG_TAMPER = "config_tamper"
    OUT_OF_SCOPE = "out_of_scope"
    PROTECTED_STORE = "protected_store"
    DIFF_TOO_LARGE = "diff_too_large"
    VENDORED_NOISE = "vendored_noise"
    DESTRUCTIVE_OP = "destructive_op"
    SECRETS_ACCESS = "secrets_access"
    BUDGET_EXCEEDED = "budget_exceeded"


class RunStats(BaseModel):
    """What the worker actually consumed (measured supervisor-side)."""
    commits: int = 0
    cycles: int = 0
    tokens_spent: int = 0
    wallclock_elapsed_s: float = 0.0


class GateResult(BaseModel):
    killed: bool = False
    high_risk: bool = False
    reasons: List[str] = Field(default_factory=list)
    flags: List[GateFlag] = Field(default_factory=list)
    touched_paths: List[str] = Field(default_factory=list)
    added_lines: int = 0
    removed_lines: int = 0

    def _add(self, flag: GateFlag, reason: str, *, high_risk: bool = False) -> None:
        self.killed = True
        self.flags.append(flag)
        self.reasons.append(reason)
        if high_risk:
            self.high_risk = True


# ---------------------------------------------------------------------------
# glob matching
# ---------------------------------------------------------------------------

def _glob_to_regex(pattern: str) -> str:
    """Translate a path glob to an anchored regex.

    Supports ``**`` (any depth, incl. across ``/``), ``*`` (within a segment),
    ``?`` (one non-slash char), and a trailing ``/`` meaning "this dir and
    everything under it".
    """
    if pattern.endswith("/"):
        pattern = pattern + "**"
    out: List[str] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern[i : i + 2] == "**":
                j = i + 2
                if pattern[j : j + 1] == "/":
                    out.append("(?:.*/)?")
                    i = j + 1
                else:
                    out.append(".*")
                    i = j
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return "^" + "".join(out) + "$"


def matches_allowed(path: str, patterns: List[str]) -> bool:
    """Strict: ``path`` must match one of the supervisor-authored allow globs."""
    return any(re.match(_glob_to_regex(p), path) for p in patterns)


def matches_forbidden(path: str, patterns: List[str]) -> bool:
    """Liberal (defensive): a forbidden entry matches if the glob matches, OR if
    a slash-free entry equals the path's basename (so ``conftest.py`` catches
    ``anything/conftest.py``)."""
    base = path.rsplit("/", 1)[-1]
    for p in patterns:
        if re.match(_glob_to_regex(p), path):
            return True
        if "/" not in p and not p.startswith("*") and p == base:
            return True
    return False


# ---------------------------------------------------------------------------
# diff parsing
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")
_RENAME_RE = re.compile(r"^rename (?:from|to) (.+)$")
_NEWDEL_RE = re.compile(r"^(?:\+\+\+|---) (?:[ab]/)?(.+)$")
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")


def parse_diff(diff_text: str) -> Tuple[List[str], int, int, bool, bool]:
    """Return (touched_paths, added_lines, removed_lines, is_empty, whitespace_only).

    Touched paths include BOTH sides of a ``diff --git a/X b/Y`` and rename
    endpoints, so a rename can't smuggle a forbidden file past the gate.

    Header vs. content is tracked with a hunk state machine: ``+++``/``---`` lines
    are file headers ONLY in the header section (before the first ``@@`` of a
    file). Inside a hunk body every line carries a one-char prefix, so an *added*
    line whose content begins with ``++ `` (rendered ``+++ ...``) is real content
    — counted, never mistaken for a header path. Without this, such a line would
    be uncounted (bypassing ``max_diff_lines``) and its text recorded as a touched
    path (skewing scope checks and the worker's own diff-economy score).
    """
    touched: set[str] = set()
    added: List[str] = []
    removed: List[str] = []
    in_hunk = False  # True once we've seen this file's first @@ hunk header

    for raw in diff_text.splitlines():
        m = _DIFF_GIT_RE.match(raw)
        if m:
            touched.add(m.group(1).strip())
            touched.add(m.group(2).strip())
            in_hunk = False
            continue
        if _HUNK_RE.match(raw):
            in_hunk = True
            continue
        if not in_hunk:
            # File-header section: parse rename/+++/--- path lines; ignore the
            # rest (index, mode, "new file", etc.). No content is counted here.
            m = _RENAME_RE.match(raw)
            if m:
                touched.add(m.group(1).strip())
                continue
            if raw.startswith("+++ ") or raw.startswith("--- "):
                m = _NEWDEL_RE.match(raw)
                if m and m.group(1).strip() != "/dev/null":
                    touched.add(m.group(1).strip())
            continue
        # Inside a hunk body: +/- are content, leading space is context, a
        # leading backslash is the "\ No newline at end of file" marker.
        if raw.startswith("+"):
            added.append(raw[1:])
        elif raw.startswith("-"):
            removed.append(raw[1:])

    touched_paths = sorted(p for p in touched if p and p != "/dev/null")
    is_empty = not added and not removed and not touched_paths

    # Whitespace-only = the non-blank content is identical once ALL whitespace is
    # removed (``git diff -w`` semantics — catches reindents and operator spacing).
    non_ws_added = sorted(s for s in (_WS_RE.sub("", l) for l in added) if s)
    non_ws_removed = sorted(s for s in (_WS_RE.sub("", l) for l in removed) if s)
    whitespace_only = bool(added or removed) and non_ws_added == non_ws_removed

    return touched_paths, len(added), len(removed), is_empty, whitespace_only


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

def run_static_gate(
    diff_text: str,
    manifest: BranchManifest,
    *,
    run_stats: Optional[RunStats] = None,
    guard_log: str = "",
    audit_log: str = "",
) -> GateResult:
    """Apply every static check. Collects all violations; ``killed`` is True if
    any fired. NEVER executes worker code."""
    manifest.assert_intact()  # a tampered manifest can't authorize anything

    touched, added, removed, is_empty, ws_only = parse_diff(diff_text)
    result = GateResult(touched_paths=touched, added_lines=added, removed_lines=removed)

    # 1. empty / whitespace-only
    if is_empty:
        result._add(GateFlag.EMPTY_DIFF, "diff is empty — no work performed")
        # nothing else to check
        return _finalize(result)
    if ws_only:
        result._add(GateFlag.WHITESPACE_ONLY, "diff is whitespace-only — no real change")

    # 2. per-path scope checks
    vendored_lines = 0
    for path in touched:
        if _is_config_tamper(path):
            result._add(GateFlag.CONFIG_TAMPER,
                        f"config/threshold tamper (fake-success surface): {path}",
                        high_risk=True)
        elif matches_forbidden(path, manifest.forbidden_paths) or _is_high_risk(path):
            high = _is_high_risk(path)
            flag = GateFlag.SAFETY_TOUCHED if high else GateFlag.FORBIDDEN_PATH
            result._add(flag, f"touched forbidden path: {path}", high_risk=high)
        elif manifest.allowed_paths and not matches_allowed(path, manifest.allowed_paths):
            result._add(GateFlag.OUT_OF_SCOPE, f"touched path outside allowed scope: {path}")
        elif not manifest.allowed_paths:
            result._add(GateFlag.OUT_OF_SCOPE, f"no allowed_paths declared — nothing may change: {path}")
        if _is_protected_store(path, manifest.protected_stores):
            result._add(GateFlag.PROTECTED_STORE, f"touched protected store: {path}", high_risk=True)
        if matches_forbidden(path, _VENDORED_PATTERNS):
            vendored_lines += 1

    # 3. diff size
    total_changed = added + removed
    if total_changed > manifest.max_diff_lines:
        result._add(
            GateFlag.DIFF_TOO_LARGE,
            f"diff {total_changed} lines exceeds max_diff_lines={manifest.max_diff_lines}",
        )

    # 4. vendored noise dominance (by touched-file count as a proxy)
    if total_changed >= _VENDORED_FLOOR and touched:
        if vendored_lines / len(touched) > _VENDORED_DOMINANCE:
            result._add(GateFlag.VENDORED_NOISE, "diff dominated by vendored/generated files")

    # 5. untrusted log scan (corroborating evidence)
    if guard_log and any(m.lower() in guard_log.lower() for m in _DESTRUCTIVE_MARKERS):
        result._add(GateFlag.DESTRUCTIVE_OP, "destructive op recorded in guard log", high_risk=True)
    blob = f"{guard_log}\n{audit_log}"
    if blob.strip() and any(m.lower() in blob.lower() for m in _SECRETS_MARKERS):
        result._add(GateFlag.SECRETS_ACCESS, "secrets-access attempt recorded in audit log", high_risk=True)

    # 6. budgets
    if run_stats is not None:
        _check_budgets(result, run_stats, manifest)

    return _finalize(result)


def _check_budgets(result: GateResult, stats: RunStats, manifest: BranchManifest) -> None:
    checks = [
        (stats.commits, manifest.max_commits, "commits"),
        (stats.cycles, manifest.max_cycles, "cycles"),
        (stats.tokens_spent, manifest.token_budget, "tokens"),
        (stats.wallclock_elapsed_s, manifest.wallclock_seconds, "wallclock_seconds"),
    ]
    for actual, cap, label in checks:
        if cap and actual > cap:
            result._add(GateFlag.BUDGET_EXCEEDED, f"{label} {actual} exceeded cap {cap}")


def _is_high_risk(path: str) -> bool:
    return matches_forbidden(path, _HIGH_RISK_PATTERNS)


def _is_config_tamper(path: str) -> bool:
    return matches_forbidden(path, _CONFIG_TAMPER_PATTERNS)


def _is_protected_store(path: str, stores: List[str]) -> bool:
    return any(store and store in path for store in stores)


def _finalize(result: GateResult) -> GateResult:
    if result.killed:
        logger.warning(
            "static gate KILL (high_risk=%s): %s", result.high_risk, "; ".join(result.reasons)
        )
    else:
        logger.info("static gate PASS (%d paths, +%d/-%d)", len(result.touched_paths),
                    result.added_lines, result.removed_lines)
    return result
