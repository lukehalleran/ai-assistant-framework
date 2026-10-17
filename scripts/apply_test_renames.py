#!/usr/bin/env python3
"""Applier for the approved test-rename map (plan v2 sections 4.2 and 4.5).

Modes (exactly one; ``--dry-run`` is the default):

* ``--dry-run``  compute the full EDIT SET (renames, import rewrites, current-reference line
  edits, the docstring batch-label line, BUG_CLASSES / TEST_LANES policy, the permanent
  ``tests/RENAMES.tsv``, the guard's allowlist removal) as a reviewable diff file and report the
  preflight.  Touches nothing.  Run against an unapproved draft it labels the output DRAFT.
* ``--apply``    the same edit set, performed only after the preflight passes (approved header,
  every old file present with its approved hash, no target exists, unique and acyclic map, safety-name
  rule unchanged, every hunk applies, clean tree).  Any failure touches nothing.  Renames use
  ``git mv``; progress goes to ``--journal``.
* ``--resume``   re-validate the journal against the tree and finish an interrupted ``--apply``
  (the only operation allowed on a dirty tree).
* ``--abort-report`` list exactly what was moved or edited and what is pending.  Deletes nothing and
  runs no destructive git.
* ``--seal OUT`` turn a reviewed draft into an approved map (hash-bound header).

CLAUDE.md / FOLLOWUPS.md edits are only PRINTED as a patch, never written.

Example::

    python scripts/apply_test_renames.py --root . --map RENAMES.approved.tsv --out EDITSET.diff
    python scripts/apply_test_renames.py --root . --map RENAMES.approved.tsv --apply --journal J.jsonl
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import propose_test_renames as P  # noqa: E402  (sibling module; same-directory import)
from utils.daemon_guard import daemon_running  # noqa: E402

MIGRATION_DATE = "2026-10-16"
GUARD_PATH = "tests/unit/test_test_file_naming_guard.py"
RENAMES_PATH = "tests/RENAMES.tsv"
BUG_CLASSES_PATH = "docs/BUG_CLASSES.md"
TEST_LANES_PATH = "docs/TEST_LANES.md"
PRINTED_ONLY = ("CLAUDE.md", "FOLLOWUPS.md")

_TOKEN_RE = re.compile(
    rf"\b(?P<tok>test_(?:{P.MONTHS})\d[A-Za-z0-9_]*)(?P<py>\.py)?(?::(?P<ln>\d+))?", re.I
)
_POINTER_LINE_RE = re.compile(r"^\s*(?:[-*]\s+)?(?:\*\*)?(Closure|Find|Guard|Test|Tests|Detector)\b", re.I)
_ALLOWLIST_BLOCK_RE = re.compile(
    r"# BEGIN PRE-MIGRATION ALLOWLIST.*?# END PRE-MIGRATION ALLOWLIST", re.S
)
_TEST_LANES_PROSE_RE = re.compile(
    r"Existing\s+dated files \(.*?\) are NOT renamed: they are\s+cited by path across .*?CGR files\.", re.S
)


class EditError(Exception):
    """A hunk that cannot be applied; the whole batch then touches nothing."""


# ---------------------------------------------------------------------------
# Edit-set construction (pure; reads the tree, never writes it)
# ---------------------------------------------------------------------------


def batch_label(old_stem: str) -> str:
    """``sep09`` / ``oct12_b4`` style label: the date token plus batch tokens, without the ``test_`` prefix."""
    toks = old_stem.split("_")[1:]
    if not toks:
        return old_stem
    keep = [toks[0]] + [t for t in toks[1:] if P.BATCH_TOKEN_RE.match(t.lower()) or t.lower() in P.SESSION_WORDS]
    return "_".join(keep)


def insert_docstring_line(text: str, line: str) -> str:
    """Add ``line`` to the module docstring (creating one if absent); raises EditError if unsupported."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise EditError(f"unparseable source: {exc.msg}") from exc
    lines = text.splitlines(keepends=True)
    body = tree.body
    first = body[0] if body else None
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        end_line = first.end_lineno
        raw = lines[end_line - 1].encode("utf-8")
        head = raw[: first.end_col_offset].decode("utf-8")
        rest = raw[first.end_col_offset:].decode("utf-8")
        if not (head.endswith('"""') or head.endswith("'''")):
            raise EditError("module docstring is not triple-quoted")
        quote, before = head[-3:], head[:-3]
        if before.strip() == "":
            prev = lines[end_line - 2] if end_line >= 2 else ""
            sep = "\n" if prev.strip() else ""
            lines[end_line - 1] = f"{sep}{before}{line}\n{before}{quote}{rest}"
        else:
            lines[end_line - 1] = f"{before.rstrip()}\n\n{line}{quote}{rest}"
        new = "".join(lines)
    else:
        if first is None:
            raise EditError("file has no statements")
        at = first.lineno - 1
        # keep decorators' / leading comments' position: insert before the first statement line
        lines.insert(at, f'"""{line}"""\n\n')
        new = "".join(lines)
    try:
        doc = ast.get_docstring(ast.parse(new)) or ""
    except SyntaxError as exc:
        raise EditError(f"docstring insertion broke the file: {exc.msg}") from exc
    if line not in doc:
        raise EditError("docstring line not present after insertion")
    return new


def _relocate(old_text: str, new_text: str, n: int) -> int:
    old_lines = old_text.splitlines()
    new_lines = [x.strip() for x in new_text.splitlines()]
    if not 1 <= n <= len(old_lines):
        raise EditError(f"locator :{n} is outside the old file")
    target = old_lines[n - 1].strip()
    hits = [i + 1 for i, x in enumerate(new_lines) if x == target]
    if not hits:
        raise EditError(f"locator :{n} line {target[:60]!r} not found in the renamed file")
    delta = len(new_lines) - len(old_lines)
    return min(hits, key=lambda i: abs(i - (n + delta)))


def _substitute(rel: str, text: str, mapping: dict[str, str], old_texts: dict[str, str],
                new_texts: dict[str, str], *, line_policy: bool, unmapped: list) -> str:
    """Replace mapped stems in ``text``.  ``line_policy`` = BUG_CLASSES rule (pointer lines replaced,
    narrative lines keep the old name and gain ``(now `new`)``)."""
    out_lines = []
    for ln, line in enumerate(text.splitlines(keepends=True), 1):
        pointer = (not line_policy) or bool(_POINTER_LINE_RE.match(line)) or line.lstrip().startswith("|") \
            or "pytest" in line
        noted: set[str] = set()

        def repl(m: re.Match) -> str:
            tok = m.group("tok")
            new_stem = mapping.get(tok)
            if new_stem is None:
                if tok not in {Path(e).stem for e in P.EXEMPT}:
                    unmapped.append((rel, ln, tok))
                return m.group(0)
            py = m.group("py") or ""
            ln_part = m.group("ln")
            suffix = ""
            if ln_part is not None:
                if py:
                    if tok not in old_texts or tok not in new_texts:
                        raise EditError(f"{rel}:{ln} locator for {tok} but no renamed file text")
                    suffix = f":{_relocate(old_texts[tok], new_texts[tok], int(ln_part))}"
                else:
                    suffix = f":{ln_part}"
            if pointer:
                return f"{new_stem}{py}{suffix}"
            noted.add(tok)
            return m.group(0)

        new_line = _TOKEN_RE.sub(repl, line)
        if noted:
            # narrative line: annotate the first mention of each renamed stem with a pointer
            for tok in sorted(noted):
                m = re.search(rf"\b{re.escape(tok)}(?:\.py)?`?", new_line)
                if m:
                    note = f" (now `{mapping[tok]}.py`)"
                    new_line = new_line[: m.end()] + note + new_line[m.end():]
        out_lines.append(new_line)
    return "".join(out_lines)


def build_editset(root: Path, header: dict, rows: list[dict], *, map_text: str,
                  migration_date: str = MIGRATION_DATE) -> dict:
    """Compute the edit set.  Returns a dict with renames, writes, printed, unresolved, unmapped, errors."""
    errors: list[str] = []
    unmapped: list = []
    ren_rows = [r for r in rows if r["kind"] == "rename"]
    unresolved = [r["old_path"] for r in ren_rows if r["new_path"] in ("?", "")]
    ren_rows = [r for r in ren_rows if r["new_path"] not in ("?", "")]
    path_map = {r["old_path"]: r["new_path"] for r in ren_rows}
    mapping = {Path(o).stem: Path(n).stem for o, n in path_map.items()}
    old_texts: dict[str, str] = {}
    new_texts: dict[str, str] = {}
    writes: dict[str, dict] = {}  # final path -> write record

    def pre_text(rel: str) -> str:
        return (root / rel).read_text(encoding="utf-8")

    # stage 1: the renamed files themselves (their own mentions + the docstring batch-label line)
    for old, new in sorted(path_map.items()):
        try:
            text = pre_text(old)
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"{old}: unreadable ({exc.__class__.__name__})")
            continue
        old_stem = Path(old).stem
        old_texts[old_stem] = text
        try:
            body = _substitute(old, text, mapping, old_texts, new_texts, line_policy=False, unmapped=unmapped)
        except EditError as exc:
            errors.append(f"{old}: {exc}")
            body = text
        try:
            line = f"Original batch label: {batch_label(old_stem)} (renamed {migration_date}; old path in tests/RENAMES.tsv)."
            body = insert_docstring_line(body, line)
        except EditError as exc:
            errors.append(f"{old}: docstring line cannot be added ({exc})")
        new_texts[old_stem] = body
        writes[new] = {"path": new, "pre_path": old, "pre_text": text, "text": body}

    # stage 2: every other CURRENT-REFERENCE file
    guard_seen = False
    for rel, p in P._text_files(root):
        if rel in path_map or rel in (RENAMES_PATH, GUARD_PATH):
            continue
        if rel in PRINTED_ONLY or P.classify_location(rel) == "historical":
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # degrades: unreadable file cannot be edited; reported as unmapped
            continue
        if not P.CITATION_RE.search(text):
            continue
        try:
            body = _substitute(rel, text, mapping, old_texts, new_texts,
                               line_policy=(rel == BUG_CLASSES_PATH), unmapped=unmapped)
        except EditError as exc:
            errors.append(f"{rel}: {exc}")
            continue
        if body != text:
            writes[rel] = {"path": rel, "pre_path": rel, "pre_text": text, "text": body}

    # TEST_LANES prose (the "NOT renamed" paragraph is false after the migration)
    tl = root / TEST_LANES_PATH
    if tl.is_file():
        rec = writes.get(TEST_LANES_PATH, {"path": TEST_LANES_PATH, "pre_path": TEST_LANES_PATH,
                                           "pre_text": tl.read_text(encoding="utf-8")})
        cur = rec.get("text", rec["pre_text"])
        new_par = (
            "Dated files (`test_sep*.py`, `test_oct*.py`, ...) were renamed by module and behaviour on "
            f"{migration_date}; `tests/RENAMES.tsv` maps every old path to its new one (and records the one "
            "exempt class-guard ledger anchor), `git log --follow <new path>` keeps the history, and "
            "`tests/unit/test_test_file_naming_guard.py` rejects new dated names."
        )
        if _TEST_LANES_PROSE_RE.search(cur):
            rec["text"] = _TEST_LANES_PROSE_RE.sub(lambda _m: new_par, cur, count=1)
            writes[TEST_LANES_PATH] = rec
        elif "were renamed by module and behaviour" not in cur:
            errors.append(f"{TEST_LANES_PATH}: the section 6 'NOT renamed' paragraph was not found")
    else:
        errors.append(f"{TEST_LANES_PATH}: missing")

    # the permanent resolver and the allowlist removal
    rt = root / RENAMES_PATH
    writes[RENAMES_PATH] = {"path": RENAMES_PATH, "pre_path": RENAMES_PATH,
                            "pre_text": rt.read_text(encoding="utf-8") if rt.is_file() else None,
                            "text": map_text}
    gp = root / GUARD_PATH
    if gp.is_file():
        g_text = gp.read_text(encoding="utf-8")
        if _ALLOWLIST_BLOCK_RE.search(g_text):
            g_new = _ALLOWLIST_BLOCK_RE.sub(
                lambda _m: "# (pre-migration allowlist removed by the migration commit)\n"
                           "PRE_MIGRATION_ALLOWLIST: frozenset[str] = frozenset()", g_text, count=1)
            writes[GUARD_PATH] = {"path": GUARD_PATH, "pre_path": GUARD_PATH, "pre_text": g_text, "text": g_new}
            guard_seen = True
        else:
            errors.append(f"{GUARD_PATH}: allowlist markers not found")
    else:
        errors.append(f"{GUARD_PATH}: missing")
    del guard_seen

    # printed-only patch (CLAUDE.md / FOLLOWUPS.md)
    printed = []
    for rel in PRINTED_ONLY:
        fp = root / rel
        if fp.is_file():
            t = fp.read_text(encoding="utf-8")
            if P.CITATION_RE.search(t):
                try:
                    nt = _substitute(rel, t, mapping, old_texts, new_texts, line_policy=False, unmapped=[])
                except EditError as exc:
                    errors.append(f"{rel}: {exc}")
                    continue
                if nt != t:
                    printed.append((rel, t, nt))

    for rec in writes.values():
        rec["post_sha"] = hashlib.sha256(rec["text"].encode("utf-8")).hexdigest()
        rec["pre_sha"] = (hashlib.sha256(rec["pre_text"].encode("utf-8")).hexdigest()
                          if rec["pre_text"] is not None else None)
    return {
        "renames": sorted(path_map.items()), "writes": [writes[k] for k in sorted(writes)],
        "printed": printed, "unresolved": unresolved, "unmapped": unmapped, "errors": errors,
        "exempt": [r["old_path"] for r in rows if r["kind"] == "exempt"],
    }


def render_diff(root: Path, es: dict, *, draft: bool) -> str:
    parts: list[str] = []
    if draft:
        parts.append("# DRAFT: the map is not approved; this edit set is for review only and cannot be applied.")
    parts.append(f"# renames: {len(es['renames'])}; content edits: {len(es['writes'])}; "
                 f"unresolved rows excluded: {len(es['unresolved'])}")
    body: list[str] = []
    for old, new in es["renames"]:
        a = (root / old).read_text(encoding="utf-8")
        b = next((w["text"] for w in es["writes"] if w["path"] == new), a)
        ratio = difflib.SequenceMatcher(None, a, b, autojunk=False).quick_ratio()
        body.append(f"rename from {old}\nrename to {new}\nsimilarity ~{int(ratio * 100)}%")
    for w in es["writes"]:
        pre = w["pre_text"] if w["pre_text"] is not None else ""
        if pre == w["text"]:
            continue
        diff = difflib.unified_diff(
            pre.splitlines(keepends=True), w["text"].splitlines(keepends=True),
            fromfile=f"a/{w['pre_path']}", tofile=f"b/{w['path']}", n=2)
        body.append("".join(x if x.endswith("\n") else x + "\n" for x in diff).rstrip("\n"))
    body_text = "\n\n".join(body) + "\n"
    parts.append(body_text)
    if es["printed"]:
        parts.append("# PRINTED ONLY (owner applies by hand; never written by the applier):")
        for rel, t, nt in es["printed"]:
            d = difflib.unified_diff(t.splitlines(keepends=True), nt.splitlines(keepends=True),
                                     fromfile=f"a/{rel}", tofile=f"b/{rel}", n=0)
            parts.append("".join(x if x.endswith("\n") else x + "\n" for x in d))
    if es["unmapped"]:
        parts.append("# unmapped dated-name tokens in CURRENT locations (left as is):")
        parts += [f"#   {r}:{n} {t}" for r, n, t in es["unmapped"][:200]]
    parts.append(f"# editset_sha256: {hashlib.sha256(body_text.encode('utf-8')).hexdigest()}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def _tree_clean(root: Path) -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=str(root), capture_output=True,
                             text=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):  # degrades: unknown tree state counts as not clean
        return False
    return out.strip() == ""


def preflight(root: Path, header: dict, rows: list[dict], es: dict, *, map_text: str,
              require_approval: bool, require_clean: bool, allow_base_mismatch: bool = False) -> list[str]:
    errs: list[str] = list(es["errors"])
    status = header.get("status", "draft")
    if require_approval:
        if status != "approved":
            errs.append(f"map status is {status!r}, not 'approved'")
        want = header.get("approved_body_sha256")
        if not want or want != P.body_sha256(map_text):
            errs.append("approved_body_sha256 missing or does not match the map body")
        if es["unresolved"]:
            errs.append(f"{len(es['unresolved'])} rows still have new_path '?'")
        if not allow_base_mismatch and header.get("base_sha") not in (None, P.head_sha(root)):
            errs.append(f"base_sha {header.get('base_sha')} != HEAD {P.head_sha(root)} "
                        "(re-seal on the current base or pass --allow-base-mismatch)")
    safety = P.load_safety_regex(root)
    seen_old: set[str] = set()
    seen_new: set[str] = set()
    olds = {r["old_path"] for r in rows if r["kind"] == "rename"}
    for r in rows:
        if r["kind"] == "exempt":
            if r["old_path"] not in P.EXEMPT:
                errs.append(f"exempt row {r['old_path']} is not in the approved exception set")
            if not (root / r["old_path"]).is_file():
                errs.append(f"exempt row {r['old_path']}: file missing")
            continue
        if r["kind"] != "rename":
            errs.append(f"unknown row kind {r['kind']!r}")
            continue
        old, new = r["old_path"], r["new_path"]
        if new in ("?", ""):
            continue
        if old in seen_old:
            errs.append(f"duplicate old_path {old}")
        if new in seen_new:
            errs.append(f"duplicate new_path {new}")
        seen_old.add(old)
        seen_new.add(new)
        if not (root / old).is_file():
            errs.append(f"{old}: old file missing")
        else:
            want = header.get("hashes", {}).get(old)
            if want is None:
                errs.append(f"{old}: no approved hash in the header")
            elif P.sha256_file(root / old) != want:
                errs.append(f"{old}: content hash differs from the approved hash")
        if (root / new).exists():
            errs.append(f"{new}: target already exists")
        if new in olds and new != old:
            errs.append(f"{new}: is also an old_path in the map (cycle/chain)")
        base = new.rsplit("/", 1)[-1]
        if not new.startswith("tests/") or not re.fullmatch(r"test_[A-Za-z0-9_]+\.py", base):
            errs.append(f"{new}: not a tests/**/test_*.py path")
        if P.DATED_FILE_RE.match(base):
            errs.append(f"{new}: still a dated name")
        if P.safety_name_changes(safety, old, new):
            errs.append(f"{old} -> {new}: changes whether the conftest safety-skip rule applies (owner decision)")
    if require_clean and not _tree_clean(root):
        errs.append("working tree is not clean (only --resume may run on a dirty tree)")
    return errs


# ---------------------------------------------------------------------------
# Apply / resume / abort-report
# ---------------------------------------------------------------------------


def _write_text(path: Path, text: str, mode_from: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-apply")
    tmp.write_text(text, encoding="utf-8")
    if mode_from is not None and mode_from.exists():
        os.chmod(tmp, mode_from.stat().st_mode & 0o777)
    os.replace(tmp, path)


def _git_mv(root: Path, old: str, new: str) -> None:
    (root / new).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "mv", old, new], cwd=str(root), check=True, capture_output=True, timeout=60)


def _journal_append(journal: Path, rec: dict) -> None:
    with journal.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _load_journal(journal: Path) -> list[dict]:
    if not journal.is_file():
        return []
    return [json.loads(ln) for ln in journal.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _plan_path(journal: Path) -> Path:
    return journal.with_name(journal.name + ".plan.json")


def _save_plan(journal: Path, es: dict, rows: list[dict], header: dict) -> None:
    old_sha = {r["old_path"]: header["hashes"].get(r["old_path"]) for r in rows if r["kind"] == "rename"}
    plan = {
        "renames": [{"old": o, "new": n, "old_sha": old_sha.get(o)} for o, n in es["renames"]],
        "writes": [{"path": w["path"], "pre_sha": w["pre_sha"], "post_sha": w["post_sha"], "text": w["text"]}
                   for w in es["writes"] if w["pre_text"] != w["text"]],
    }
    _plan_path(journal).write_text(json.dumps(plan), encoding="utf-8")


def _execute(root: Path, plan: dict, journal: Path, done_steps: set[tuple]) -> None:
    for r in plan["renames"]:
        key = ("rename", r["old"], r["new"])
        old_exists, new_exists = (root / r["old"]).exists(), (root / r["new"]).exists()
        if key in done_steps:
            if old_exists or not new_exists:
                raise RuntimeError(f"journal says {r['old']} was moved but the tree disagrees")
            continue
        if old_exists and not new_exists:
            _git_mv(root, r["old"], r["new"])
        elif new_exists and not old_exists:
            if r["old_sha"] and P.sha256_file(root / r["new"]) != r["old_sha"]:
                raise RuntimeError(f"{r['new']} exists but is not the approved content of {r['old']}")
        else:
            raise RuntimeError(f"cannot move {r['old']} -> {r['new']}: unexpected tree state")
        _journal_append(journal, {"step": "rename", "old": r["old"], "new": r["new"]})
    for w in plan["writes"]:
        key = ("write", w["path"], w["post_sha"])
        path = root / w["path"]
        cur = P.sha256_file(path) if path.is_file() else None
        if cur == w["post_sha"]:
            if key not in done_steps:
                _journal_append(journal, {"step": "write", "path": w["path"], "post_sha": w["post_sha"]})
            continue
        if key in done_steps:
            raise RuntimeError(f"journal says {w['path']} was written but its content changed since")
        if cur != w["pre_sha"]:
            raise RuntimeError(f"{w['path']}: content is neither the approved pre-state nor the post-state")
        _write_text(path, w["text"], mode_from=path)
        _journal_append(journal, {"step": "write", "path": w["path"], "post_sha": w["post_sha"]})
    _journal_append(journal, {"step": "done"})


def do_apply(root: Path, map_path: Path, journal: Path, *, editset_file: Path | None,
             allow_base_mismatch: bool, migration_date: str) -> int:
    map_text = map_path.read_text(encoding="utf-8")
    header, rows, perrs = P.parse_map(map_text)
    if perrs:
        print("MAP ERRORS:\n  " + "\n  ".join(perrs))
        return 2
    if journal.exists():
        print(f"refusing: journal {journal} already exists (use --resume or --abort-report)")
        return 2
    es = build_editset(root, header, rows, map_text=map_text, migration_date=migration_date)
    errs = preflight(root, header, rows, es, map_text=map_text, require_approval=True, require_clean=True,
                     allow_base_mismatch=allow_base_mismatch)
    if editset_file is not None:
        rendered = render_diff(root, es, draft=False)
        want = re.search(r"# editset_sha256: (\w+)", editset_file.read_text(encoding="utf-8"))
        have = re.search(r"# editset_sha256: (\w+)", rendered)
        if not want or not have or want.group(1) != have.group(1):
            errs.append("regenerated edit set differs from the reviewed --editset file")
    if errs:
        print("PREFLIGHT FAILED; nothing was touched:\n  " + "\n  ".join(errs))
        return 1
    _save_plan(journal, es, rows, header)
    _journal_append(journal, {"step": "start", "map_sha256": hashlib.sha256(map_text.encode()).hexdigest()})
    plan = json.loads(_plan_path(journal).read_text(encoding="utf-8"))
    _execute(root, plan, journal, set())
    print(f"applied: {len(plan['renames'])} renames, {len(plan['writes'])} content writes; journal {journal}")
    return 0


def _done_steps(entries: list[dict]) -> set[tuple]:
    out: set[tuple] = set()
    for e in entries:
        if e["step"] == "rename":
            out.add(("rename", e["old"], e["new"]))
        elif e["step"] == "write":
            out.add(("write", e["path"], e["post_sha"]))
    return out


def do_resume(root: Path, journal: Path) -> int:
    entries = _load_journal(journal)
    if not entries or not _plan_path(journal).is_file():
        print("nothing to resume: journal or plan missing")
        return 2
    if entries[-1]["step"] == "done":
        print("journal already complete; nothing to do")
        return 0
    plan = json.loads(_plan_path(journal).read_text(encoding="utf-8"))
    try:
        _execute(root, plan, journal, _done_steps(entries))
    except RuntimeError as exc:
        print(f"RESUME REFUSED: {exc}")
        return 1
    print(f"resumed and completed; journal {journal}")
    return 0


def do_abort_report(root: Path, journal: Path) -> int:
    entries = _load_journal(journal)
    if not entries:
        print("no journal: nothing was applied by this tool")
        return 0
    done = _done_steps(entries)
    plan = json.loads(_plan_path(journal).read_text(encoding="utf-8")) if _plan_path(journal).is_file() else {
        "renames": [], "writes": []}
    print("ABORT REPORT (read-only; deletes nothing, runs no destructive git)")
    print("moved:")
    for r in plan["renames"]:
        if ("rename", r["old"], r["new"]) in done:
            print(f"  {r['old']} -> {r['new']}   (undo by hand: git mv {r['new']} {r['old']})")
    print("edited:")
    for w in plan["writes"]:
        if ("write", w["path"], w["post_sha"]) in done:
            print(f"  {w['path']}   (inspect: git diff -- {w['path']})")
    print("pending:")
    for r in plan["renames"]:
        if ("rename", r["old"], r["new"]) not in done:
            print(f"  rename {r['old']} -> {r['new']}")
    for w in plan["writes"]:
        if ("write", w["path"], w["post_sha"]) not in done:
            print(f"  write {w['path']}")
    print(f"complete: {'yes' if entries[-1]['step'] == 'done' else 'no'}")
    return 0


# ---------------------------------------------------------------------------
# Seal
# ---------------------------------------------------------------------------


def do_seal(root: Path, map_path: Path, out: Path, accept_changed: bool) -> int:
    text = map_path.read_text(encoding="utf-8")
    header, rows, perrs = P.parse_map(text)
    if perrs:
        print("MAP ERRORS:\n  " + "\n  ".join(perrs))
        return 2
    open_rows = [r["old_path"] for r in rows if r["kind"] == "rename" and r["new_path"] in ("?", "")]
    if open_rows:
        print(f"refusing to seal: {len(open_rows)} rows still have new_path '?': " + ", ".join(open_rows[:5]))
        return 1
    changed = []
    hashes = {}
    for r in rows:
        if r["kind"] == "rename":
            if not (root / r["old_path"]).is_file():
                print(f"refusing to seal: {r['old_path']} is missing")
                return 1
            h = P.sha256_file(root / r["old_path"])
            hashes[r["old_path"]] = h
            if header["hashes"].get(r["old_path"]) not in (None, h):
                changed.append(r["old_path"])
    if changed and not accept_changed:
        print("refusing to seal: files changed since the draft: " + ", ".join(changed))
        return 1
    lines = ["#status\tapproved", f"#base_sha\t{P.head_sha(root)}"]
    lines += [f"#sha256\t{o}\t{h}" for o, h in sorted(hashes.items())]
    body = [ln for ln in text.splitlines() if not ln.startswith("#")]
    lines.append(f"#approved_body_sha256\t{hashlib.sha256(chr(10).join(body).encode()).hexdigest()}")
    out.write_text("\n".join(lines + body) + "\n", encoding="utf-8")
    print(f"sealed {len(hashes)} rename rows -> {out}")
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--map", help="approved (or draft) map TSV")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="default")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--abort-report", action="store_true")
    mode.add_argument("--seal", metavar="OUT", help="write an approved copy of --map to OUT")
    ap.add_argument("--out", help="dry-run: edit-set diff file")
    ap.add_argument("--journal", help="journal path for --apply/--resume/--abort-report")
    ap.add_argument("--editset", help="--apply: reviewed edit-set diff that must match the regenerated one")
    ap.add_argument("--allow-base-mismatch", action="store_true")
    ap.add_argument("--accept-changed", action="store_true", help="--seal: accept files changed since the draft")
    ap.add_argument("--migration-date", default=MIGRATION_DATE)
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    if args.apply and daemon_running(root):
        print("REFUSED: a live Daemon main.py process is running from this tree (or could not be ruled out); "
              "apply the migration in a clone with the Daemon down.")
        return 1

    if args.resume or args.abort_report:
        if not args.journal:
            ap.error("--journal is required")
        j = Path(args.journal)
        return do_resume(root, j) if args.resume else do_abort_report(root, j)
    if not args.map:
        ap.error("--map is required")
    map_path = Path(args.map)
    if args.seal:
        return do_seal(root, map_path, Path(args.seal), args.accept_changed)
    if args.apply:
        if not args.journal:
            ap.error("--journal is required with --apply")
        return do_apply(root, map_path, Path(args.journal),
                        editset_file=Path(args.editset) if args.editset else None,
                        allow_base_mismatch=args.allow_base_mismatch, migration_date=args.migration_date)

    map_text = map_path.read_text(encoding="utf-8")
    header, rows, perrs = P.parse_map(map_text)
    if perrs:
        print("MAP ERRORS:\n  " + "\n  ".join(perrs))
        return 2
    draft = header.get("status", "draft") != "approved"
    es = build_editset(root, header, rows, map_text=map_text, migration_date=args.migration_date)
    errs = preflight(root, header, rows, es, map_text=map_text, require_approval=False, require_clean=False,
                     allow_base_mismatch=True)
    diff = render_diff(root, es, draft=draft)
    if args.out:
        Path(args.out).write_text(diff, encoding="utf-8")
    print(("DRAFT " if draft else "") + f"dry-run: {len(es['renames'])} renames, {len(es['writes'])} content edits, "
          f"{len(es['unresolved'])} unresolved rows, {len(es['unmapped'])} unmapped tokens")
    print("preflight: " + ("OK" if not errs else f"{len(errs)} problems"))
    for e in errs[:80]:
        print("  " + e)
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
