#!/usr/bin/env python3
"""Reviewed docs manifest for the frozen build (audit A01, 2026-10-23, class: BC-59 nearest).

``daemon.spec`` used to bundle the WHOLE working-directory ``docs`` tree, including
untracked and gitignored owner files; git ignore rules and the privacy hook do not
filter PyInstaller data entries. The build now ships exactly the files listed in
``config/reference_docs_manifest.txt`` (one repo-relative path per line, ``#``
comments): one data entry per file, never a directory. A listed path that does not
exist or is not git-tracked FAILS the build.

``docs_data_entries`` takes the tracked set as an argument so tests never read real
git state; ``git_tracked_files`` is the spec-side source of that set.

    python scripts/check_bundle_manifest.py     # validate the real manifest
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import PurePosixPath
from typing import Iterable, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.self_docs_manifest import MANIFEST_RELPATH, entry_shape_problem, parse_manifest  # noqa: E402,F401


class ManifestError(SystemExit):
    """Raised (as a loud build failure) for an unreviewable manifest."""


def git_tracked_files(root: str) -> Set[str]:
    out = subprocess.run(["git", "-C", root, "ls-files", "-z"], check=True,
                         capture_output=True, text=True).stdout
    return {p for p in out.split("\0") if p}


def docs_data_entries(root: str, manifest_text: str,
                      tracked: Iterable[str]) -> List[Tuple[str, str]]:
    """PyInstaller ``datas`` tuples (src, dest_dir), one per manifest file."""
    tracked = set(tracked)
    entries: List[Tuple[str, str]] = []
    problems: List[str] = []
    for rel in parse_manifest(manifest_text):
        pp = PurePosixPath(rel)
        shape = entry_shape_problem(rel)  # the same rule the runtime seed/read gate applies
        if shape:
            problems.append(shape)
            continue
        if not os.path.isfile(os.path.join(root, *pp.parts)):
            problems.append(f"{rel}: file does not exist")
        elif rel not in tracked:
            problems.append(f"{rel}: not git-tracked (never ship an unreviewed file)")
        else:
            entries.append((rel, str(pp.parent)))
    if problems:
        raise ManifestError(
            "reference docs manifest is invalid (" + MANIFEST_RELPATH + "):\n  "
            + "\n  ".join(problems))
    return entries


def load_docs_entries(root: str) -> List[Tuple[str, str]]:
    with open(os.path.join(root, *MANIFEST_RELPATH.split("/")), encoding="utf-8") as fh:
        text = fh.read()
    return docs_data_entries(root, text, git_tracked_files(root))


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ents = load_docs_entries(here)
    print(f"manifest OK: {len(ents)} docs files")
    sys.exit(0)
