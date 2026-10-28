"""Archived dated docs keep working links (2026-10-27 move of 12 dated docs into docs/archive/; BC-63).

Checks the working tree only: (a) every relative markdown link inside docs/archive/ resolves (links into the
gitignored runtime ``data/`` tree are exempt - they never resolve in a clean checkout); (b) no living file still
names a moved doc by its old path.  docs/execution/ (frozen point-in-time records with quoted hashes and patches) and
scripts/bug_class_guards/ (reserved; the owner applies its one docstring fix) are deliberately not rewritten.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = REPO_ROOT / "docs" / "archive"

MOVED_STEMS = (
    "AUDIT_20260906_CONVERSATION", "AUDIT_20260915_personal_event_grounding", "BUG_RETROSPECTIVE_20260715_20260904",
    "CLAIM_SUPPORT_TRACE_20260902", "GENERALIZATION_AUDIT_20260901", "GENERALIZATION_CI_REVIEW_20260913",
    "INDEPENDENT_AUDIT_20260905", "PLAN_20260909_audit_repairs", "PLAN_20260912_session_defects",
    "PLAN_20260913_generalization_execution", "PLAN_20260915_personal_claim_support", "SITUATION_COVERAGE_AUDIT_20260901",
)
_LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
_OLD_REF_RE = re.compile(r"(?<!archive/)(?<![A-Za-z0-9_])(?:" + "|".join(MOVED_STEMS) + r")")
_LIVING_ROOTS = ("core", "api", "gui", "knowledge", "memory", "models", "utils", "scripts", "tests", "eval", "config",
                 "processing", "integrations", "agent_branch")
_EXEMPT_PREFIXES = ("scripts/bug_class_guards/",)
_TEXT_SUFFIXES = {".py", ".md", ".sh", ".yml", ".yaml", ".ts", ".tsx"}


def _links(path: Path):
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for m in _LINK_RE.finditer(line):
            yield n, m.group(1)


def test_every_moved_doc_is_in_the_archive_and_gone_from_docs():
    for stem in MOVED_STEMS:
        assert (ARCHIVE / f"{stem}.md").is_file(), stem
        assert not (REPO_ROOT / "docs" / f"{stem}.md").exists(), stem


def test_relative_links_in_archived_docs_resolve():
    broken = []
    for doc in sorted(ARCHIVE.glob("*.md")):
        for n, target in _links(doc):
            if re.match(r"[A-Za-z][A-Za-z0-9+.-]*:|#", target):
                continue
            rel = target.split("#", 1)[0]
            if not rel:
                continue
            resolved = (doc.parent / rel).resolve()
            try:
                top = resolved.relative_to(REPO_ROOT).parts[0]
            except ValueError:
                broken.append(f"{doc.name}:{n}: {target} (outside repo)")
                continue
            if top == "data":
                continue
            if not resolved.exists():
                broken.append(f"{doc.name}:{n}: {target}")
    assert broken == []


def test_link_checker_catches_an_un_rebased_link(tmp_path):
    (tmp_path / "archive").mkdir()
    (tmp_path / "other.md").write_text("x")
    doc = tmp_path / "archive" / "d.md"
    doc.write_text("[ok](../other.md) [stale](other.md)")
    resolved = {t: (doc.parent / t).exists() for _, t in _links(doc)}
    assert resolved == {"../other.md": True, "other.md": False}


def test_no_living_file_names_a_moved_doc_by_its_old_path():
    stale = []
    this = Path(__file__).resolve()
    files = [p for p in (REPO_ROOT / "docs").glob("*.md")] + [p for p in (REPO_ROOT / "docs" / "generalization").glob("*.md")]
    for root in _LIVING_ROOTS:
        files += [p for p in (REPO_ROOT / root).rglob("*") if p.is_file() and p.suffix in _TEXT_SUFFIXES]
    for p in files:
        rel = p.relative_to(REPO_ROOT).as_posix()
        if p.resolve() == this or rel.startswith(_EXEMPT_PREFIXES) or "node_modules" in rel:
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if _OLD_REF_RE.search(line):
                stale.append(f"{rel}:{n}")
    assert stale == []
