#!/usr/bin/env python3
"""READ-ONLY proposer: draft map for folding date-named tests into the
``test_<module>_<behaviour>.py`` convention (plan v2, docs/PLAN_20261016_test_naming_by_domain.md).

It re-runs three inventories on the tree it is pointed at and writes a DRAFT map
(TSV, columns per plan section 4.2):

1. the dated-file inventory (``test_<month><digit>...`` under ``tests/``);
2. the case-insensitive citation inventory, split CURRENT-REFERENCE vs HISTORICAL;
3. the AST import inventory (which production modules each dated file touches,
   and which other test files import it).

Evidence only, never a verdict (plan section 2): static imports and reference counts,
DYNAMIC loads (importlib / spec_from_file_location / runpy / subprocess of a script),
monkeypatched boundaries, route-handler entry points, fixtures, and the adding commit's
touched sources.  A row whose evidence does not converge is ``confidence=owner`` and
gets ``new_path='?'``: there is no forced tie-break.

Nothing here writes inside the checkout unless ``--out`` points there.  Usage::

    python scripts/propose_test_renames.py --root . --out ~/daemon_exec/oct16n_runs/RENAMES.draft.tsv
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
DATED_FILE_RE = re.compile(rf"^test_(?:{MONTHS})\d", re.I)
CITATION_RE = re.compile(rf"\btest_(?:{MONTHS})\d[A-Za-z0-9_]*", re.I)

# The approved exception set (plan 3.1a): path -> reason.
EXEMPT = {
    "tests/unit/test_sep10_calendar_forced_action_loop.py": "class-guard ledger anchor",
}

COLUMNS = (
    "kind", "old_path", "new_path", "subject_module", "other_subjects",
    "confidence", "evidence", "flags", "reason",
)
COLUMN_LINE = "\t".join(COLUMNS)

SKIP_DIRS = frozenset({
    ".git", "venv", ".venv", "node_modules", "data", "__pycache__", "dist", "build",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "wiki", "logs", "site-packages",
})
# The tooling's own files cite sample names; they are not migration consumers.
TOOLING_EXCLUDES = frozenset({
    "tests/RENAMES.tsv",
    "tests/unit/test_test_file_naming_guard.py",
    "tests/unit/test_propose_test_renames.py",
    "tests/unit/test_apply_test_renames.py",
    "scripts/propose_test_renames.py",
    "scripts/apply_test_renames.py",
})
TEXT_SUFFIXES = frozenset({
    ".py", ".md", ".txt", ".yaml", ".yml", ".json", ".sh", ".toml", ".cfg", ".ini",
    ".tsv", ".patch", ".ts", ".tsx", ".js", ".spec", ".rst", ".csv", "",
})
MAX_TEXT_BYTES = 5_000_000

BATCH_TOKEN_RE = re.compile(r"^[a-z]\d+[a-z]*$")
SESSION_WORDS = frozenset({"followups", "followup", "session", "batch", "referee"})

_HISTORICAL_PREFIXES = (
    "docs/execution/", "docs/history/", "docs/audits/", "docs/postmortems/", "docs/generalization/",
)
_HISTORICAL_DOC_RE = re.compile(r"^(?:AUDIT|HANDOFF|PLAN|REVIEW|INDEPENDENT_AUDIT|BUG_RETROSPECTIVE)_", re.I)


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


def classify_location(relpath: str) -> str:
    """Return ``"historical"`` or ``"current"`` for a repo-relative path (plan section 1)."""
    p = relpath.replace("\\", "/")
    base = p.rsplit("/", 1)[-1]
    if p.startswith(_HISTORICAL_PREFIXES):
        return "historical"
    if p == "CLAUDE_CHANGELOG.md" or p.startswith("config/bug_class_"):
        return "historical"
    if p.endswith(".patch") or re.search(r"receipt", base, re.I):
        return "historical"
    if p.startswith("docs/") and (_HISTORICAL_DOC_RE.match(base) or re.search(r"\d{8}", base)):
        return "historical"
    return "current"


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True, text=True, timeout=60, check=True
        )
    except (OSError, subprocess.SubprocessError):  # degrades: no git evidence, rows say so
        return None
    return out.stdout


def head_sha(root: Path) -> str:
    out = _git(root, "rev-parse", "HEAD")
    return out.strip() if out else "unknown"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Inventory 1: dated files
# ---------------------------------------------------------------------------


def _walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            yield Path(dirpath) / name


def list_dated_files(root: Path) -> list[str]:
    """Every file under tests/ whose basename is date-named, repo-relative, sorted."""
    tests = root / "tests"
    out = []
    if tests.is_dir():
        for p in _walk(tests):
            if DATED_FILE_RE.match(p.name):
                out.append(p.relative_to(root).as_posix())
    return sorted(out)


# ---------------------------------------------------------------------------
# Production module index (+ ambiguity)
# ---------------------------------------------------------------------------


def production_files(root: Path) -> list[str]:
    out = []
    for p in _walk(root):
        if p.suffix != ".py":
            continue
        rel = p.relative_to(root).as_posix()
        top = rel.split("/", 1)[0]
        if top in {"tests", "web", "docs"}:
            continue
        out.append(rel)
    return sorted(out)


def module_index(prod: list[str]) -> dict[str, str]:
    """dotted module name -> repo-relative path (``a/b.py`` -> ``a.b``; ``a/b/__init__.py`` -> ``a.b``)."""
    idx: dict[str, str] = {}
    for rel in prod:
        parts = rel[:-3].split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            idx[".".join(parts)] = rel
    return idx


def ambiguity_set(prod: list[str]) -> set[str]:
    """Basenames (without ``.py``, excluding ``__init__``) that exist in more than one production file."""
    seen: dict[str, int] = defaultdict(int)
    for rel in prod:
        stem = rel.rsplit("/", 1)[-1][:-3]
        if stem != "__init__":
            seen[stem] += 1
    return {s for s, n in seen.items() if n > 1}


def subject_label(path: str, prod: list[str], ambiguous: set[str]) -> tuple[str, bool]:
    """Filename label for a subject module path; qualified when the basename is ambiguous."""
    parts = path[:-3].split("/")
    stem = parts[-1]
    pkg_parts = parts[:-1]
    if stem == "__init__":
        if not pkg_parts:
            return "init", False
        stem, pkg_parts = pkg_parts[-1], pkg_parts[:-1]
        label_ambiguous = True
    else:
        label_ambiguous = stem in ambiguous
    if not label_ambiguous:
        return stem.lower(), False
    # add parent directories until the label is unique among same-basename files
    siblings = []
    for rel in prod:
        q = rel[:-3].split("/")
        if q[-1] == "__init__":
            q = q[:-1]
        if q and q[-1] == stem:
            siblings.append(q)
    depth = 1
    while depth <= len(pkg_parts):
        label = pkg_parts[-depth:] + [stem]
        clashes = [s for s in siblings if s[-len(label):] == label]
        if len(clashes) <= 1:
            break
        depth += 1
    label = pkg_parts[-min(depth, len(pkg_parts)):] + [stem] if pkg_parts else [stem]
    return "_".join(label).lower(), True


# ---------------------------------------------------------------------------
# Inventory 3: AST evidence per dated test file
# ---------------------------------------------------------------------------


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _resolve_module(dotted: str, idx: dict[str, str]) -> str | None:
    parts = dotted.split(".")
    for n in range(len(parts), 0, -1):
        cand = ".".join(parts[:n])
        if cand in idx:
            return idx[cand]
    return None


def _leaves(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _leaves(node.left) + _leaves(node.right)
    if isinstance(node, ast.Call):
        name = _dotted(node.func) or ""
        if name.endswith(("join", "joinpath")) or name.rsplit(".", 1)[-1] == "Path":
            out: list[str] = []
            for a in node.args:
                out += _leaves(a)
            return out
    return []


_MONKEYPATCH_NAMES = frozenset({"monkeypatch", "mp", "m", "mpatch"})
_DYNAMIC_NAMES = frozenset({
    "spec_from_file_location", "run_path", "run_module", "import_module", "__import__", "load_source",
})
_SUBPROCESS_FUNCS = frozenset({"run", "Popen", "check_call", "check_output", "call"})


def analyze_test_file(root: Path, rel: str, idx: dict[str, str], prod: list[str]) -> dict:
    """Static/dynamic/patched/route/fixture evidence for one test file (never a verdict)."""
    text = (root / rel).read_text(encoding="utf-8", errors="replace")
    ev: dict = {
        "static": Counter(), "patched": Counter(), "dynamic": Counter(), "routes": Counter(),
        "fixtures": 0, "fixture_mods": set(), "entry_calls": 0, "helper_imports": set(),
        "parse_error": False,
    }
    try:
        tree = ast.parse(text)
    except SyntaxError:  # degrades: file gets owner confidence + parse_error flag
        ev["parse_error"] = True
        return ev

    alias: dict[str, str] = {}  # local name -> production path
    import_nodes: list[tuple[str, str, str]] = []  # (kind, local, path) kind: mod|sym
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                path = _resolve_module(a.name, idx)
                if path is None:
                    if a.name.startswith("tests"):
                        ev["helper_imports"].add(a.name)
                    continue
                local = a.asname or a.name.split(".")[0]
                alias[local] = path
                import_nodes.append(("mod" if a.asname else f"pkg:{a.name}", local, path))
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.startswith("tests"):
                ev["helper_imports"].add(node.module)
                continue
            for a in node.names:
                sub = f"{node.module}.{a.name}"
                if sub in idx:
                    path = idx[sub]
                    kind = "mod"
                else:
                    path = _resolve_module(node.module, idx)
                    kind = "sym"
                if path is None:
                    continue
                local = a.asname or a.name
                alias[local] = path
                import_nodes.append((kind, local, path))

    name_loads: Counter = Counter()
    attr_chains: Counter = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            name_loads[node.id] += 1
        elif isinstance(node, ast.Attribute):
            d = _dotted(node)
            if d:
                attr_chains[d] += 1
    for kind, local, path in import_nodes:
        if kind.startswith("pkg:"):
            refs = attr_chains.get(kind[4:], 0)
        else:
            refs = name_loads.get(local, 0)
        ev["static"][path] += max(1, refs)

    # patched boundaries and dynamic loads
    has_dynamic = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = _dotted(node.func) or ""
        last = fn.rsplit(".", 1)[-1]
        if last in _DYNAMIC_NAMES or (fn.startswith("subprocess.") and last in _SUBPROCESS_FUNCS):
            has_dynamic = True
        if last == "main":
            ev["entry_calls"] += 1
        is_setattr = last in {"setattr", "delattr"} and any(
            part in _MONKEYPATCH_NAMES for part in fn.split(".")[:-1]
        )
        is_patch = fn in {"patch", "mock.patch", "unittest.mock.patch"} or fn.endswith(
            (".patch", "patch.object", "patch.dict")
        ) or fn in {"patch.object", "patch.dict"}
        if (is_setattr or is_patch) and node.args:
            first = node.args[0]
            target: str | None = None
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                target = _resolve_module(first.value, idx)
            else:
                d = _dotted(first)
                if d:
                    target = alias.get(d.split(".")[0])
            if target:
                ev["patched"][target] += 1
    if has_dynamic:
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.BinOp, ast.Call, ast.Constant)):
                joined = "/".join(_leaves(node)).replace("\\", "/")
                if not joined:
                    continue
                if joined.endswith(".py"):
                    hits = [p for p in prod if p == joined or p.endswith("/" + joined)]
                    if len(hits) == 1:
                        found.add(hits[0])
                elif joined in idx and isinstance(node, ast.Constant):
                    found.add(idx[joined])
        for p in found:
            ev["dynamic"][p] += 1

    # routes
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            m = re.match(r"^/api/([a-z_]+)", node.value)
            if m:
                cand = f"api/routes/{m.group(1)}.py"
                if cand in prod:
                    ev["routes"][cand] += 1

    # fixtures
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any("fixture" in (_dotted(d.func if isinstance(d, ast.Call) else d) or "")
                   for d in node.decorator_list):
                ev["fixtures"] += 1
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Name) and sub.id in alias:
                        ev["fixture_mods"].add(alias[sub.id])
    return ev


def candidate_scores(ev: dict) -> Counter:
    scores: Counter = Counter()
    for p, n in ev["static"].items():
        scores[p] += n
    for p, n in ev["patched"].items():
        scores[p] += n
    for p, n in ev["dynamic"].items():
        scores[p] += 3 * n
    for p, n in ev["routes"].items():
        scores[p] += n
    return scores


# ---------------------------------------------------------------------------
# Adding-commit evidence
# ---------------------------------------------------------------------------


def adding_commit_sources(root: Path, rel: str) -> tuple[str, list[str], int]:
    """(short sha, non-test touched sources, total files) of the commit that added ``rel``."""
    out = _git(root, "log", "--diff-filter=A", "--format=%H", "--", rel)
    if not out or not out.strip():
        return "", [], 0
    sha = out.strip().splitlines()[-1]
    shown = _git(root, "show", "--name-only", "--format=", sha)
    if shown is None:
        return sha[:9], [], 0
    files = [f for f in shown.splitlines() if f.strip()]
    sources = [f for f in files if not f.startswith(("tests/", "docs/")) and f.endswith(".py")]
    return sha[:9], sources, len(files)


# ---------------------------------------------------------------------------
# Safety-name rule (tests/conftest.py)
# ---------------------------------------------------------------------------


def load_safety_regex(root: Path):
    """The conftest ``_SAFETY_TEST_RE`` pattern, read from the tree itself (None if absent)."""
    conf = root / "tests" / "conftest.py"
    if not conf.is_file():
        return None
    try:
        tree = ast.parse(conf.read_text(encoding="utf-8"))
    except SyntaxError:  # degrades: no safety check possible; callers see None
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_SAFETY_TEST_RE" for t in node.targets
        ):
            call = node.value
            if isinstance(call, ast.Call) and call.args and isinstance(call.args[0], ast.Constant):
                return re.compile(call.args[0].value)
    return None


def safety_name_changes(regex, old_path: str, new_path: str) -> bool:
    """True when the safety-skip rule would treat old and new basenames differently."""
    if regex is None:
        return False
    old_b, new_b = old_path.rsplit("/", 1)[-1], new_path.rsplit("/", 1)[-1]
    return bool(regex.search(old_b)) != bool(regex.search(new_b))


# ---------------------------------------------------------------------------
# Inventory 2: citations
# ---------------------------------------------------------------------------


def _text_files(root: Path):
    listed = _git(root, "ls-files")
    if listed is not None:
        paths = [root / line for line in listed.splitlines() if line.strip()]
    else:
        paths = list(_walk(root))
    for p in paths:
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            continue
        if rel in TOOLING_EXCLUDES or p.suffix not in TEXT_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in rel.split("/")[:-1]):
            continue
        try:
            if not p.is_file() or p.stat().st_size > MAX_TEXT_BYTES:
                continue
        except OSError:  # degrades: unreadable file is not scanned
            continue
        yield rel, p


def citation_inventory(root: Path, stems: set[str]) -> dict[str, dict]:
    """Per old stem: ``{"current": [(path, line)], "historical": [...], "case_variants": [...]}``.

    Case-insensitive token match (v1 missed ``test_sep27b_B5`` with a lowercase-only regex).
    Tokens that are not a mapped stem are collected under the pseudo-key ``"?unmapped"``.
    """
    lower = {s.lower(): s for s in stems}
    inv: dict[str, dict] = {s: {"current": [], "historical": [], "case_variants": []} for s in stems}
    inv["?unmapped"] = {"current": [], "historical": [], "case_variants": []}
    for rel, p in _text_files(root):
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # degrades: binary/odd file is not scanned
            continue
        if not CITATION_RE.search(text):
            continue
        loc = classify_location(rel)
        for ln, line in enumerate(text.splitlines(), 1):
            for m in CITATION_RE.finditer(line):
                tok = m.group(0)
                if tok in inv:
                    inv[tok][loc].append((rel, ln))
                elif tok.lower() in lower:
                    key = lower[tok.lower()]
                    inv[key]["case_variants"].append((rel, ln, tok))
                else:
                    inv["?unmapped"][loc].append((rel, ln, tok))
    return inv


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def residual_tokens(old_stem: str, subject_label_: str) -> tuple[list[str], bool]:
    """Behaviour tokens left after removing the date prefix, batch labels and session words.

    Returns ``(tokens, batch_residual)``; ``batch_residual`` is True when nothing meaningful remains.
    """
    s = re.sub(rf"^test_(?:{MONTHS})\d+[a-z]?_?", "", old_stem, flags=re.I)
    toks = [t for t in s.lower().split("_") if t]
    toks = [t for t in toks if not BATCH_TOKEN_RE.match(t) and t not in SESSION_WORDS]
    subj = subject_label_.split("_")
    # drop a leading overlap with the subject label ("pattern_engine_utc" under pattern_engine -> "utc")
    n = 0
    while n < len(subj) and n < len(toks) and toks[n] == subj[n]:
        n += 1
    if n == len(subj) and n > 0:
        toks = toks[n:]
    return toks, not toks


def build_rows(root: Path, base_sha: str | None = None, with_git: bool = True) -> tuple[list[dict], dict]:
    prod = production_files(root)
    idx = module_index(prod)
    ambiguous = ambiguity_set(prod)
    safety = load_safety_regex(root)
    dated = list_dated_files(root)
    stems = {Path(d).stem for d in dated}
    inventory = citation_inventory(root, stems)
    existing = {p.relative_to(root).as_posix() for p in _walk(root / "tests")} if (root / "tests").is_dir() else set()

    rows: list[dict] = []
    importers: dict[str, list[str]] = defaultdict(list)
    analyses: dict[str, dict] = {}
    for rel in dated:
        if rel in EXEMPT:
            continue
        analyses[rel] = analyze_test_file(root, rel, idx, prod)
    for rel, ev in analyses.items():
        for h in ev["helper_imports"]:
            tail = h.rsplit(".", 1)[-1]
            for other in dated:
                if Path(other).stem == tail and other != rel:
                    importers[other].append(rel)
    # test files (dated or not) that import a dated file, from the whole tests/ tree
    for p in _walk(root / "tests") if (root / "tests").is_dir() else []:
        if p.suffix != ".py":
            continue
        rel = p.relative_to(root).as_posix()
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # degrades: unparseable file contributes no importer evidence
            continue
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                last = m.rsplit(".", 1)[-1]
                if last in stems and m.startswith("tests"):
                    key = next(d for d in dated if Path(d).stem == last)
                    if rel not in importers[key]:
                        importers[key].append(rel)

    seen_new: Counter = Counter()
    for rel in dated:
        if rel in EXEMPT:
            rows.append({
                "kind": "exempt", "old_path": rel, "new_path": "-", "subject_module": "-",
                "other_subjects": "-", "confidence": "-", "evidence": "-", "flags": "-",
                "reason": EXEMPT[rel], "sha256": sha256_file(root / rel),
            })
            continue
        ev = analyses[rel]
        scores = candidate_scores(ev)
        ranked = scores.most_common()
        flags: list[str] = []
        conf = "owner"
        subject = "?"
        others: list[str] = []
        new_path = "?"
        if ev["parse_error"]:
            flags.append("parse_error")
        if ranked:
            top, top_n = ranked[0]
            second_n = ranked[1][1] if len(ranked) > 1 else 0
            dyn = set(ev["dynamic"])
            if dyn and top not in dyn:
                conf = "owner"
            elif len(ranked) == 1:
                conf = "mechanical"
            elif second_n and top_n == second_n:
                conf = "owner"
            elif top_n >= 1.5 * second_n:
                conf = "review"
            else:
                conf = "owner"
            if conf != "owner":
                subject = top
            others = [p for p, n in ranked[1:] if n >= 2 and n >= 0.25 * top_n][:6]
            if len([p for p, n in ranked[1:] if n >= 0.5 * top_n]) >= 1:
                flags.append("multi_subject")
        old_stem = Path(rel).stem
        parent = rel.rsplit("/", 1)[0]
        if subject != "?":
            label, qualified = subject_label(subject, prod, ambiguous)
            if qualified:
                flags.append("qualified_basename")
            toks, batch_residual = residual_tokens(old_stem, label)
            if batch_residual:
                flags.append("batch_residual")
            if toks:
                new_path = f"{parent}/test_{label}_{'_'.join(toks)}.py"
        sha, srcs, nfiles = adding_commit_sources(root, rel) if with_git else ("", [], 0)
        parts = []
        if ev["static"]:
            parts.append("static:" + ",".join(f"{p}({n})" for p, n in ev["static"].most_common(6)))
        if ev["patched"]:
            parts.append("patched:" + ",".join(f"{p}({n})" for p, n in ev["patched"].most_common(6)))
        if ev["dynamic"]:
            parts.append("dynamic:" + ",".join(sorted(ev["dynamic"])))
        if ev["routes"]:
            parts.append("routes:" + ",".join(sorted(ev["routes"])))
        if ev["entry_calls"]:
            parts.append(f"entry_calls:{ev['entry_calls']}")
        if ev["fixtures"]:
            parts.append(f"fixtures:{ev['fixtures']}")
        if ev["helper_imports"]:
            parts.append("test_helpers:" + ",".join(sorted(ev["helper_imports"])))
        if importers.get(rel):
            parts.append("imported_by:" + ",".join(sorted(importers[rel])))
        if sha:
            tail = f"commit:{sha}"
            if nfiles > 40:
                tail += f"[bulk:{nfiles}]"
            elif srcs:
                tail += "[" + ",".join(srcs[:6]) + "]"
            parts.append(tail)
        if conf == "owner" and ranked:
            parts.append("competing:" + ",".join(f"{p}({n})" for p, n in ranked[:4]))
        if not ranked:
            parts.append("no production module found")
        row = {
            "kind": "rename", "old_path": rel, "new_path": new_path, "subject_module": subject,
            "other_subjects": ",".join(others) if others else "-", "confidence": conf,
            "evidence": "; ".join(parts) or "-", "flags": ",".join(flags) or "-", "reason": "-",
            "sha256": sha256_file(root / rel),
        }
        rows.append(row)
        if new_path != "?":
            seen_new[new_path] += 1

    for row in rows:
        if row["kind"] != "rename":
            continue
        npath = row["new_path"]
        fl = [] if row["flags"] == "-" else row["flags"].split(",")
        if npath != "?":
            if seen_new[npath] > 1 or npath in existing:
                fl.append("collision")
                if row["confidence"] == "mechanical":
                    row["confidence"] = "review"
            if safety_name_changes(safety, row["old_path"], npath):
                fl.append("safety_name")
                row["confidence"] = "owner"
        row["flags"] = ",".join(sorted(set(fl))) or "-"
        row["evidence"] = row["evidence"].replace("\t", " ").replace("\n", " ")
    rows.sort(key=lambda r: r["old_path"])
    meta = {
        "base_sha": base_sha or head_sha(root),
        "inventory": inventory,
        "ambiguous_basenames": sorted(ambiguous),
        "importers": {k: sorted(v) for k, v in importers.items()},
        "dated_count": len(dated),
    }
    return rows, meta


# ---------------------------------------------------------------------------
# Map file I/O (shared with the applier)
# ---------------------------------------------------------------------------


def render_map(rows: list[dict], meta: dict, status: str = "draft") -> str:
    lines = [f"#status\t{status}", f"#base_sha\t{meta['base_sha']}"]
    for r in rows:
        if r["kind"] == "rename":
            lines.append(f"#sha256\t{r['old_path']}\t{r['sha256']}")
    lines.append(COLUMN_LINE)
    for r in rows:
        lines.append("\t".join(r[c] for c in COLUMNS))
    return "\n".join(lines) + "\n"


def body_sha256(text: str) -> str:
    body = [ln for ln in text.splitlines() if not ln.startswith("#")]
    return hashlib.sha256("\n".join(body).encode("utf-8")).hexdigest()


def parse_map(text: str) -> tuple[dict, list[dict], list[str]]:
    """Parse a map into ``(header, rows, errors)``; header holds status/base_sha/hashes/approval."""
    header: dict = {"hashes": {}}
    rows: list[dict] = []
    errors: list[str] = []
    seen_columns = False
    for n, ln in enumerate(text.splitlines(), 1):
        if not ln.strip():
            continue
        if ln.startswith("#"):
            parts = ln[1:].split("\t")
            if parts[0] == "sha256" and len(parts) == 3:
                header["hashes"][parts[1]] = parts[2]
            elif len(parts) == 2:
                header[parts[0]] = parts[1]
            else:
                errors.append(f"line {n}: malformed header line")
            continue
        if not seen_columns:
            if ln != COLUMN_LINE:
                errors.append(f"line {n}: column header mismatch")
            seen_columns = True
            continue
        cells = ln.split("\t")
        if len(cells) != len(COLUMNS):
            errors.append(f"line {n}: expected {len(COLUMNS)} columns, got {len(cells)}")
            continue
        rows.append(dict(zip(COLUMNS, cells)))
    if not seen_columns:
        errors.append("missing column header line")
    return header, rows, errors


def summarize(rows: list[dict], meta: dict) -> str:
    ren = [r for r in rows if r["kind"] == "rename"]
    conf = Counter(r["confidence"] for r in ren)
    flags: Counter = Counter()
    for r in ren:
        for f in r["flags"].split(","):
            if f != "-":
                flags[f] += 1
    out = [
        f"dated files: {meta['dated_count']} ({len(ren)} rename rows, {len(rows) - len(ren)} exempt)",
        "confidence: " + ", ".join(f"{k}={v}" for k, v in sorted(conf.items())),
        "flags: " + (", ".join(f"{k}={v}" for k, v in sorted(flags.items())) or "none"),
        f"unresolved new_path ('?'): {sum(1 for r in ren if r['new_path'] == '?')}",
        "ambiguous production basenames (qualify when used as subject): "
        + ", ".join(meta["ambiguous_basenames"]),
    ]
    coll = [r for r in ren if "collision" in r["flags"].split(",")]
    if coll:
        out.append("collisions:")
        out += [f"  {r['old_path']} -> {r['new_path']}" for r in coll]
    out.append("owner rows:")
    for r in ren:
        if r["confidence"] == "owner":
            m = re.search(r"competing:([^;]*)", r["evidence"])
            out.append(f"  {r['old_path']}: {m.group(1) if m else r['evidence'][:160]}")
    inv = meta["inventory"]
    cur = sum(len(v["current"]) for k, v in inv.items() if k != "?unmapped")
    his = sum(len(v["historical"]) for k, v in inv.items() if k != "?unmapped")
    cv = sum(len(v["case_variants"]) for k, v in inv.items() if k != "?unmapped")
    out.append(f"citations: current={cur} historical={his} case_variants={cv} "
               f"unmapped_tokens={len(inv['?unmapped']['current']) + len(inv['?unmapped']['historical'])}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".", help="tree to inventory (default: cwd)")
    ap.add_argument("--out", required=True, help="draft map path (not inside the checkout by default)")
    ap.add_argument("--summary", help="optional summary text path")
    ap.add_argument("--inventory-out", help="optional JSON citation inventory path")
    ap.add_argument("--no-git", action="store_true", help="skip adding-commit evidence")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    rows, meta = build_rows(root, with_git=not args.no_git)
    Path(args.out).write_text(render_map(rows, meta, "draft"), encoding="utf-8")
    text = summarize(rows, meta)
    print(text)
    if args.summary:
        Path(args.summary).write_text(text + "\n", encoding="utf-8")
    if args.inventory_out:
        Path(args.inventory_out).write_text(
            json.dumps({"inventory": meta["inventory"], "importers": meta["importers"]}, indent=1),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
