"""Single source of truth for which docs are Daemon's self-knowledge (class: BC-59, BC-58).

Incident (2026-10-27): the frozen build was fixed (audit A01, 2026-10-23) to ship and
seed only ``config/reference_docs_manifest.txt``, but the dev/live path still seeded
every top-level ``docs/*.md`` (90 files incl. gitignored handoffs, a private review and
untracked drafts) into ``reference_docs``. The read side admitted any chunk under the
repo ``docs/`` dir, so those could render as [DAEMON DOCUMENTATION] and surface through
insight mode's "files" channel.

This module owns the manifest parse, its validation, path resolution, and the
predicates every consumer uses:
  * ``is_listed_self_doc``  - read allow-list ([DAEMON DOCUMENTATION]): the chunk's file
    is a valid manifest entry IN THIS INSTALL's docs dir (provenance, not a filename
    match). A moved install re-matches once the seed re-uploads the docs
    (``ReferenceDocsManager.sync_file`` re-syncs when the stored path changed).
  * ``is_unlisted_repo_doc`` - conservative deny rule for the purge: inside this
    install's docs dir and not listed. Deletion never guesses about other paths.
  * ``is_unreviewed_doc``   - broader read-side deny rule (insight "files" channel): the
    purge rule plus doc-shaped files (``.../docs/<name>.md``) from any OTHER path, e.g.
    a store restored from an older install. Filtering a read loses nothing.

Failure policy: reads and the seed use only VALID manifest entries (a repo-relative
``docs/`` path without ``..`` whose file exists); an unreadable or empty manifest yields
no self-docs. The purge refuses whenever ``manifest_problems()`` is non-empty.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path, PurePosixPath
from typing import FrozenSet, List, Mapping, NamedTuple, Optional, Tuple

from utils.bootstrap import resolve_bundled_path

logger = logging.getLogger(__name__)

MANIFEST_RELPATH = "config/reference_docs_manifest.txt"


class _Manifest(NamedTuple):
    entries: Tuple[str, ...]        # valid repo-relative entries, manifest order
    files: FrozenSet[str]           # their resolved absolute paths
    problems: Tuple[str, ...]       # empty when the manifest is fully valid


_cache: Optional[_Manifest] = None


def parse_manifest(text: str) -> List[str]:
    paths: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and line not in paths:
            paths.append(line)
    return paths


def entry_shape_problem(rel: str) -> Optional[str]:
    """Shape rule shared with the build validator (scripts/check_bundle_manifest.py)."""
    pp = PurePosixPath(rel)
    if pp.is_absolute() or ".." in pp.parts or pp.parts[:1] != ("docs",) or len(pp.parts) < 2:
        return f"{rel}: must be a repo-relative path under docs/"
    return None


def _resolved(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def clear_cache() -> None:
    """Test hook: forget the cached manifest."""
    global _cache
    _cache = None


def _load() -> _Manifest:
    global _cache
    if _cache is not None:
        return _cache
    problems: List[str] = []
    entries: List[str] = []
    files: List[str] = []
    try:
        with open(resolve_bundled_path(MANIFEST_RELPATH), encoding="utf-8") as fh:
            raw = parse_manifest(fh.read())
    except Exception as exc:  # degrades: no self-docs rather than unreviewed ones (fail closed)
        raw = []
        problems.append(f"manifest unreadable ({type(exc).__name__})")
    if not raw and not problems:
        problems.append("manifest is empty")
    for rel in raw:
        shape = entry_shape_problem(rel)
        if shape:
            problems.append(shape)
            continue
        path = resolve_bundled_path(rel)
        if not os.path.isfile(path):
            problems.append(f"{rel}: file does not exist")
            continue
        entries.append(rel)
        files.append(_resolved(path))
    if problems:
        logger.warning("[SelfDocs] reference docs manifest problems (%s): %s - only valid entries "
                       "are seeded/rendered; the purge refuses", MANIFEST_RELPATH, "; ".join(problems))
    _cache = _Manifest(tuple(entries), frozenset(files), tuple(problems))
    return _cache


def manifest_entries() -> Tuple[str, ...]:
    """VALID repo-relative posix paths of the reviewed self-docs (may be empty)."""
    return _load().entries


def manifest_problems() -> Tuple[str, ...]:
    """Every reason the manifest is not fully valid; empty when it is."""
    return _load().problems


def manifest_files() -> List[str]:
    """Absolute paths of the valid manifest entries (for the seed)."""
    return [resolve_bundled_path(e) for e in manifest_entries()]


def self_docs_dir() -> Path:
    return Path(_resolved(resolve_bundled_path("docs")))


def manifest_files_under(path: str) -> List[str]:
    """Valid manifest files equal to ``path`` or inside it (seed paths inside docs/)."""
    p = Path(_resolved(path))
    out = []
    for f in manifest_files():
        rf = Path(_resolved(f))
        if rf == p or rf.is_relative_to(p):
            out.append(f)
    return out


def is_inside_self_docs_dir(path: str) -> bool:
    """True when ``path`` (resolved) is the self-docs dir or anything inside it."""
    try:
        p, d = Path(_resolved(path)), self_docs_dir()
        return p == d or p.is_relative_to(d)
    except Exception:  # degrades: an unresolvable path is treated as outside docs/
        return False


def _file_path(meta: Mapping) -> str:
    return str(meta.get("file_path") or "")


def is_listed_self_doc(metadata: Optional[Mapping]) -> bool:
    meta = metadata or {}
    if meta.get("type") == "user_upload":
        return False
    fp = _file_path(meta)
    if not fp:
        return False  # no recorded origin => no provenance (the live store has none)
    try:
        return _resolved(fp) in _load().files
    except Exception:  # degrades: an unresolvable path is not self-knowledge
        return False


def is_unlisted_repo_doc(metadata: Optional[Mapping]) -> bool:
    """Purge rule: inside THIS install's docs dir and not a valid manifest doc."""
    meta = metadata or {}
    if meta.get("type") == "user_upload":
        return False
    fp = _file_path(meta)
    return bool(fp) and is_inside_self_docs_dir(fp) and not is_listed_self_doc(meta)


def is_foreign_repo_doc(metadata: Optional[Mapping]) -> bool:
    """A doc-shaped file (``.../docs/<name>.md``) from some OTHER path (old install, restore)."""
    meta = metadata or {}
    if meta.get("type") == "user_upload":
        return False
    fp = _file_path(meta)
    if not fp:
        return False
    pp = PurePosixPath(fp.replace("\\", "/"))
    return pp.parent.name == "docs" and pp.suffix.lower() == ".md" and not is_inside_self_docs_dir(fp)


def is_unreviewed_doc(metadata: Optional[Mapping]) -> bool:
    """Read-side deny rule (insight "files" channel): unlisted here, or doc-shaped elsewhere."""
    return is_unlisted_repo_doc(metadata) or is_foreign_repo_doc(metadata)
