#!/usr/bin/env python3
"""Naming policy: a name describes CONTENT, never a date (stdlib only; the ONE definition).

A date may be metadata (docstring, commit body, changelog heading, record id ``CGR-YYYYMMDD-NNN``, backup or
log-archive filename, an archived point-in-time document inside an archival folder).  A date may NOT be the NAME of a
durable, living artifact: a source/test/script file (or a directory on its path), a living doc, a branch, a commit
subject or a PR title.  See docs/TEST_LANES.md section 6 and the bug-class draft "Date used as an artifact's identifier".

CLI (exit 0 clean / 1 violations / 2 git or usage error)::

    naming_policy.py paths --code     [--root .]   # filesystem walk, CODE scope
    naming_policy.py paths --tracked  [--root .]   # git ls-files: CODE + DOCS + ROOT scope
    naming_policy.py subject "<commit subject>"
    naming_policy.py branch <name>
    naming_policy.py pr-title "<title>"
    naming_policy.py range <base_sha> <head_sha> [--root .]   # non-merge commits in base..head

Used by tests/unit/test_dated_names_guard.py, tests/unit/test_test_file_naming_guard.py and (drafted, owner-applied)
.github/workflows/naming.yml, hooks/commit-msg and hooks/pre-push.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------------------------------------- date detection
_MONTHS = r"jan|feb|mar|apr|may|jun|jul|aug|sept|sep|oct|nov|dec"
_CAP_MONTHS = "|".join(m.capitalize() for m in _MONTHS.split("|"))
_DAY = r"(?:0?[1-9]|[12]\d|3[01])(?!\d)"

# A month name that begins its own word (not preceded by a letter) -- or, capitalised, a camelCase word start
# (``sep30HideSurfaces``, ``testOct27``) -- followed by a day number.  The separator may be '-', '_' or (except for
# "may", which is an ordinary English word) a space.
_MONTH_DAY_RE = re.compile(
    rf"(?:(?<![A-Za-z])(?i:{_MONTHS})|(?<=[a-z])(?:{_CAP_MONTHS}))"
    rf"(?:[-_]?|(?<!may)(?<!May)(?<!MAY) ?){_DAY}"
)
_ISO_RE = re.compile(r"(?<!\d)20\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?!\d)")
_COMPACT_RE = re.compile(r"(?<!\d)20\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?!\d)")
# Record ids carry a compact date by design (CGR-20261014-002, CGR-20260913-005-2).
_RECORD_ID_RE = re.compile(r"CGR-\d{8}-\d{3}(?:-\d+)?")


def date_tokens(text: str) -> list[str]:
    """Return the date tokens in ``text`` (month+day forms, ISO dates, compact dates), in order; [] when none.

    Record ids (``CGR-YYYYMMDD-NNN``) are metadata and are removed first.  Bare numbers, versions, model names and
    ports never match (``gpt-4o``, ``bge-small-en-v1.5``, ``MGT6203``, ``phase5d``, ``v2``, ``ISO8601``, ``#111``).
    """
    cleaned = _RECORD_ID_RE.sub(" ", text)
    found: list[tuple[int, str]] = []
    for rx in (_ISO_RE, _COMPACT_RE, _MONTH_DAY_RE):
        for m in rx.finditer(cleaned):
            found.append((m.start(), m.group(0)))
    found.sort()
    out: list[str] = []
    last_end = -1
    for start, tok in found:  # drop a token wholly inside an earlier one
        if start >= last_end:
            out.append(tok)
            last_end = start + len(tok)
    return out


# ----------------------------------------------------------------------------------------------- allow rules (data)
# Folders whose files are archived point-in-time records: dates in their names are metadata.
ARCHIVAL_PREFIXES: tuple[str, ...] = ("docs/audits/", "docs/archive/", "docs/execution/")
# The single ledger-anchored exception (anchors records in config/bug_class_dispositions.json and
# config/bug_class_baseline.json; re-anchoring is a separate authorized class-guard change).
APPROVED_PATHS: frozenset[str] = frozenset({"tests/unit/test_sep10_calendar_forced_action_loop.py"})

# ----------------------------------------------------------------------------------------------- traversal contract
# Every top-level directory that holds source.  A NEW top-level directory containing a source file fails the guard
# until it is added here or to SKIP_TOP_LEVEL (so a new tree cannot silently escape).
CODE_INCLUDE: tuple[str, ...] = (
    "agent_branch", "api", "config", "core", "eval", "gui", "hooks", "installer", "integrations", "knowledge",
    "memory", "models", "processing", "scripts", "tests", "utils", "web",
)
DOCS_DIR = "docs"
# Data / runtime directories (top level only) -- never source, never checked.
SKIP_TOP_LEVEL: frozenset[str] = frozenset(
    {"data", "logs", "documents", "conversation_logs", "daemon_notes", "chroma_db", "assets"}
)
SKIP_SUBTREES: tuple[str, ...] = ("assets/generated",)
_SKIP_ANYWHERE_NAMES = frozenset({"__pycache__", "node_modules", "dist", "build", "civenv"})
SOURCE_SUFFIXES: frozenset[str] = frozenset({".py", ".ts", ".tsx", ".js", ".sh"})


def _skipped_dir_name(name: str) -> bool:
    """True for a directory name that is never descended into (hidden dirs, caches, build output, venvs)."""
    return name in _SKIP_ANYWHERE_NAMES or name.startswith("venv") or name.startswith(".")


def _is_skipped(relpath: str) -> bool:
    parts = relpath.split("/")
    if parts[0] in SKIP_TOP_LEVEL and len(parts) > 1:
        return True
    if any(relpath == s or relpath.startswith(s + "/") for s in SKIP_SUBTREES):
        return True
    return any(_skipped_dir_name(p) for p in parts[:-1])


def _top(relpath: str) -> str:
    return relpath.split("/", 1)[0]


def _in_scope(relpath: str, *, tracked: bool) -> bool:
    if _is_skipped(relpath):
        return False
    top = _top(relpath)
    if "/" not in relpath:  # root-level file
        return tracked
    if top in CODE_INCLUDE:
        return True
    if top == DOCS_DIR:
        return tracked
    return False


def iter_candidate_paths(root: str | os.PathLike, *, file_list: list[str] | None = None) -> list[str]:
    """The ONE traversal.  Returns sorted repo-relative posix paths to check.

    ``file_list=None``: filesystem walk of CODE scope (the pytest guard / ``paths --code``).
    ``file_list=[...]``: a tracked-file list -> CODE + DOCS + ROOT scope (``paths --tracked``).
    """
    root_p = Path(root)
    if file_list is not None:
        return sorted(p for p in (f.replace("\\", "/") for f in file_list) if _in_scope(p, tracked=True))
    out: list[str] = []
    for top in CODE_INCLUDE:
        base = root_p / top
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            rel_dir = Path(dirpath).relative_to(root_p).as_posix()
            dirnames[:] = sorted(d for d in dirnames if not _is_skipped(f"{rel_dir}/{d}/x"))
            for name in filenames:
                rel = f"{rel_dir}/{name}"
                if not _is_skipped(rel):
                    out.append(rel)
    return sorted(out)


def unclassified_top_level_dirs(root: str | os.PathLike, *, file_list: list[str] | None = None) -> list[str]:
    """Top-level directories holding a source file that are neither in CODE_INCLUDE, docs/, nor a skipped data dir."""
    known = set(CODE_INCLUDE) | SKIP_TOP_LEVEL | {DOCS_DIR}
    found: set[str] = set()
    if file_list is not None:
        for f in file_list:
            f = f.replace("\\", "/")
            if "/" in f and _top(f) not in known and not _top(f).startswith(".") and not _is_skipped(f) \
                    and Path(f).suffix in SOURCE_SUFFIXES:
                found.add(_top(f))
        return sorted(found)
    root_p = Path(root)
    for entry in sorted(os.listdir(root_p)):
        if entry in known or _skipped_dir_name(entry) or not (root_p / entry).is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root_p / entry):
            dirnames[:] = [d for d in dirnames if not _skipped_dir_name(d)]
            if any(Path(n).suffix in SOURCE_SUFFIXES for n in filenames):
                found.add(entry)
                break
    return sorted(found)


# ----------------------------------------------------------------------------------------------- surface checks
Violation = tuple[str, str]  # (subject, reason)


def check_path(relpath: str) -> list[Violation]:
    """Violations for one repo-relative path: every component (directories included) is checked."""
    rel = relpath.replace("\\", "/")
    if rel.startswith("./"):
        rel = rel[2:]
    if rel.startswith(ARCHIVAL_PREFIXES) or rel in APPROVED_PATHS:
        return []
    out: list[Violation] = []
    for comp in rel.split("/"):
        toks = date_tokens(comp)
        if toks:
            out.append((rel, f"date in name ({', '.join(toks)}) -- name by content; put the date in the file body "
                             "(archival docs belong in docs/archive/)"))
            break
    return out


def check_commit_subject(subject: str) -> list[Violation]:
    """A commit subject describes the change; a merge commit's subject is generated by git/GitHub and is skipped."""
    first = subject.strip().splitlines()[0] if subject.strip() else ""
    if first.startswith("Merge "):
        return []
    toks = date_tokens(first)
    if toks:
        return [(first, f"date in commit subject ({', '.join(toks)}) -- describe the change; dates go in the body")]
    return []


def check_branch(name: str) -> list[Violation]:
    toks = date_tokens(name)
    if toks:
        return [(name, f"date in branch name ({', '.join(toks)}) -- use fix/<area>-<what>")]
    return []


def check_pr_title(title: str) -> list[Violation]:
    toks = date_tokens(title)
    if toks:
        return [(title, f"date in PR title ({', '.join(toks)}) -- describe the change")]
    return []


def check_paths(paths: list[str]) -> list[Violation]:
    out: list[Violation] = []
    for p in paths:
        out.extend(check_path(p))
    return out


def check_tree(root: str | os.PathLike, *, file_list: list[str] | None = None) -> list[Violation]:
    """Path violations + unclassified-top-level-directory violations for the tree (the guard's whole job)."""
    out = check_paths(iter_candidate_paths(root, file_list=file_list))
    for d in unclassified_top_level_dirs(root, file_list=file_list):
        out.append((d + "/", "top-level directory holds source but is not classified in naming_policy.CODE_INCLUDE "
                             "or SKIP_TOP_LEVEL"))
    return out


# ----------------------------------------------------------------------------------------------- git helpers / CLI
class GitError(RuntimeError):
    pass


def _git(root: str | os.PathLike, *args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitError(f"git {' '.join(args)}: {exc}") from exc
    if r.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {r.stderr.strip() or 'exit ' + str(r.returncode)}")
    return r.stdout


def tracked_files(root: str | os.PathLike) -> list[str]:
    return [p for p in _git(root, "ls-files", "-z").split("\0") if p]


def commits_in_range(root: str | os.PathLike, base: str, head: str) -> list[tuple[str, str]]:
    """Non-merge commits reachable from ``head`` and not from ``base`` as (sha, subject).  Raises GitError when a
    sha is unknown, or when the merge base is unavailable (shallow history) -- never silently 'no commits'."""
    _git(root, "cat-file", "-e", f"{base}^{{commit}}")
    _git(root, "cat-file", "-e", f"{head}^{{commit}}")
    _git(root, "merge-base", base, head)
    raw = _git(root, "log", "--no-merges", "--format=%H%x1f%s%x1e", f"{base}..{head}")
    out = []
    for rec in raw.split("\x1e"):
        rec = rec.strip("\n")
        if rec:
            sha, _, subj = rec.partition("\x1f")
            out.append((sha, subj))
    return out


def _report(violations: list[Violation]) -> int:
    for subj, reason in violations:
        print(f"{subj}: {reason}")
    return 1 if violations else 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = "."
    if "--root" in args:
        i = args.index("--root")
        try:
            root = args[i + 1]
        except IndexError:
            print("usage error: --root needs a value", file=sys.stderr)
            return 2
        del args[i:i + 2]
    if not args:
        print(__doc__, file=sys.stderr)
        return 2
    cmd, rest = args[0], args[1:]
    try:
        if cmd == "paths" and rest in (["--code"], ["--tracked"]):
            if rest == ["--code"]:
                return _report(check_tree(root))
            return _report(check_tree(root, file_list=tracked_files(root)))
        if cmd == "subject" and len(rest) == 1:
            return _report(check_commit_subject(rest[0]))
        if cmd == "branch" and len(rest) == 1:
            return _report(check_branch(rest[0]))
        if cmd == "pr-title" and len(rest) == 1:
            return _report(check_pr_title(rest[0]))
        if cmd == "range" and len(rest) == 2:
            bad: list[Violation] = []
            for sha, subj in commits_in_range(root, rest[0], rest[1]):
                bad.extend((f"{sha[:10]} {s}", r) for s, r in check_commit_subject(subj))
            return _report(bad)
    except GitError as exc:
        print(f"git error: {exc}", file=sys.stderr)
        return 2
    print(f"usage error: {' '.join(args)}", file=sys.stderr)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
