"""Filesystem snapshot utility for agent session safety.

Purpose: Create deterministic manifests of repo file state (path, size,
mtime, sha256) so that pre- and post-agent diffs can be computed without
relying on git alone.

Inputs:  Repository root path.
Outputs: Manifest dict  {rel_path: {size, mtime, sha256}}, JSON on disk.
Key functions: create_manifest, diff_manifests, save_manifest, load_manifest.
Side effects: Reads filesystem; writes JSON when save_manifest is called.
Dependencies: Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# ============================================================================
# Exclusion rules
# ============================================================================

_EXCLUDED_DIRS: set[str] = {
    ".git",
    ".agent_snapshots",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
}

_EXCLUDED_PREFIXES: tuple[str, ...] = (
    "RECOVERY_",
)

_EXCLUDED_NAMES: set[str] = {
    ".Rhistory",
}

_EXCLUDED_EXTENSIONS: set[str] = {
    ".pyc",
    ".pyo",
    ".rpm",
    ".7z",
}


def should_exclude(rel: str) -> bool:
    """Return True if *rel* (a POSIX-style relative path) should be skipped."""
    parts = Path(rel).parts

    # Any path component is an excluded dir?
    for part in parts:
        if part in _EXCLUDED_DIRS:
            return True

    basename = parts[-1] if parts else ""

    # Excluded prefixes (e.g. RECOVERY_*)
    for prefix in _EXCLUDED_PREFIXES:
        if basename.startswith(prefix):
            return True

    # Excluded exact names
    if basename in _EXCLUDED_NAMES:
        return True

    # Excluded extensions
    _, ext = os.path.splitext(basename)
    if ext.lower() in _EXCLUDED_EXTENSIONS:
        return True

    return False


# ============================================================================
# Hashing
# ============================================================================

_BUF_SIZE = 1 << 16  # 64 KiB


def sha256_file(path: str | Path) -> str:
    """Return the hex sha256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_BUF_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# Manifest creation
# ============================================================================

def iter_snapshot_files(root: str | Path) -> list[Path]:
    """Walk *root* and return relative paths of non-excluded files, sorted."""
    root = Path(root).resolve()
    results: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place so os.walk skips them.
        dirnames[:] = [
            d for d in dirnames if d not in _EXCLUDED_DIRS
        ]
        for fname in filenames:
            full = Path(dirpath) / fname
            rel = full.relative_to(root)
            if not should_exclude(str(rel)):
                results.append(rel)
    results.sort()
    return results


def create_manifest(root: str | Path) -> dict[str, dict[str, Any]]:
    """Build a manifest dict: {relative_posix_path: {size, mtime, sha256}}."""
    root = Path(root).resolve()
    manifest: dict[str, dict[str, Any]] = {}
    for rel in iter_snapshot_files(root):
        full = root / rel
        try:
            stat = full.stat()
            manifest[rel.as_posix()] = {
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "sha256": sha256_file(full),
            }
        except OSError:
            # File vanished between walk and stat — skip silently.
            continue
    return manifest


# ============================================================================
# Persistence
# ============================================================================

def save_manifest(manifest: dict[str, dict[str, Any]], path: str | Path) -> None:
    """Write *manifest* to *path* as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def load_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read a manifest JSON file."""
    with open(path) as f:
        return json.load(f)


# ============================================================================
# Diffing
# ============================================================================

def diff_manifests(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """Compare two manifests. Returns {added, deleted, modified} lists of paths."""
    before_keys = set(before)
    after_keys = set(after)

    added = sorted(after_keys - before_keys)
    deleted = sorted(before_keys - after_keys)
    modified: list[str] = []
    for key in sorted(before_keys & after_keys):
        if before[key].get("sha256") != after[key].get("sha256"):
            modified.append(key)

    return {
        "added": added,
        "deleted": deleted,
        "modified": modified,
    }


# ============================================================================
# CLI
# ============================================================================

def _cli_create(args: argparse.Namespace) -> None:
    manifest = create_manifest(args.root)
    save_manifest(manifest, args.output)
    print(f"Manifest saved: {args.output} ({len(manifest)} files)")


def _cli_diff(args: argparse.Namespace) -> None:
    before = load_manifest(args.before)
    after = load_manifest(args.after)
    result = diff_manifests(before, after)

    for category in ("added", "deleted", "modified"):
        items = result[category]
        if items:
            print(f"\n{category.upper()} ({len(items)}):")
            for item in items:
                print(f"  {item}")

    total = sum(len(v) for v in result.values())
    if total == 0:
        print("No changes detected.")
    else:
        print(f"\nTotal changes: {total}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fs_snapshot",
        description="Filesystem snapshot utility for agent session safety.",
    )
    sub = parser.add_subparsers(dest="command")

    p_create = sub.add_parser("create", help="Create a manifest")
    p_create.add_argument("root", help="Repository root directory")
    p_create.add_argument("output", help="Output manifest JSON path")
    p_create.set_defaults(func=_cli_create)

    p_diff = sub.add_parser("diff", help="Diff two manifests")
    p_diff.add_argument("before", help="Before manifest JSON path")
    p_diff.add_argument("after", help="After manifest JSON path")
    p_diff.set_defaults(func=_cli_diff)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
