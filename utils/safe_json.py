"""
Module Contract — utils/safe_json.py

Purpose:
    Shared JSON persistence safety for the critical single-file stores
    (knowledge graph, entity aliases, user profile, corpus, claim index).
    Two guarantees:
      1. atomic_write_json(): a crash mid-write can never truncate an
         existing store (temp file in the same directory + os.replace).
      2. load_critical_json(): an existing-but-corrupt store is NEVER
         silently replaced with empty state. The corrupt file is copied
         to <path>.corrupt-<timestamp> (preserved, never deleted) and
         CorruptStoreError is raised so startup fails with an actionable
         message instead of quietly overwriting user data.

Inputs:  file paths + JSON-serializable data.
Outputs: parsed JSON (load) / files on disk (write).
Side effects: temp-file creation, os.replace, quarantine copies.

Key decisions:
    - Missing file is NOT an error (fresh install) → load returns None.
    - Corrupt/unreadable existing file is fatal by default. Callers that
      wrap store construction in broad `except Exception` must re-raise
      CorruptStoreError explicitly (see memory_coordinator.py).
"""

import json
import os
import shutil
from datetime import datetime
from typing import Any, Optional

from utils.logging_utils import get_logger

logger = get_logger("safe_json")


class CorruptStoreError(RuntimeError):
    """An existing persistent store failed to load.

    Raised instead of silently continuing with empty state (which would
    overwrite the user's data on the next save). The original file is
    preserved — quarantine_path points at a copy when one could be made.
    """

    def __init__(self, path: str, label: str, original: Exception,
                 quarantine_path: Optional[str] = None):
        self.path = path
        self.label = label
        self.original = original
        self.quarantine_path = quarantine_path
        preserved = (
            f"A copy was preserved at: {quarantine_path}"
            if quarantine_path
            else "The original file was left in place."
        )
        super().__init__(
            f"{label} at '{path}' exists but could not be loaded ({original}). "
            f"Refusing to start with empty state — that would overwrite your data. "
            f"{preserved} "
            f"To recover: restore the file from a backup, or move it aside to "
            f"start fresh deliberately."
        )


class StoreVersionError(RuntimeError):
    """A persistent store was written by a NEWER app version than this one.

    Loading (and later saving) it with old code could silently drop fields —
    so startup must stop with an actionable message instead.
    """

    def __init__(self, path: str, label: str, found: int, supported: int):
        self.path = path
        self.label = label
        self.found = found
        self.supported = supported
        super().__init__(
            f"{label} at '{path}' has schema_version {found}, but this build "
            f"supports up to {supported}. It was written by a newer Daemon "
            f"version — loading it here could silently drop data. "
            f"Run the newer version, or restore a backup that matches this build."
        )


def check_schema_version(payload: Any, *, current: int, path: str,
                         label: str) -> int:
    """Validate a dict payload's schema_version. Missing → 1 (pre-versioning).

    Returns the found version (older versions are the caller's migration
    concern); raises StoreVersionError for versions newer than `current`.
    """
    found = 1
    if isinstance(payload, dict):
        try:
            found = int(payload.get("schema_version", 1))
        except (TypeError, ValueError):
            found = 1
    if found > current:
        logger.critical(
            f"[SafeJSON] {label} at {path}: schema_version {found} > supported {current}"
        )
        raise StoreVersionError(path, label, found, current)
    return found


def atomic_write_json(path: str, data: Any, *, indent: int = 2,
                      ensure_ascii: bool = False, default=None,
                      fsync: bool = True) -> None:
    """Write JSON via temp file + os.replace so a crash can't truncate `path`.

    Raises OSError on failure (after cleaning up the temp file); the
    existing file at `path` is untouched in that case.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii,
                      default=default)
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def load_critical_json(path: str, label: str) -> Optional[Any]:
    """Load a critical JSON store. Missing file → None (fresh start).

    Existing but unparseable/unreadable → quarantine a copy and raise
    CorruptStoreError. Never returns empty state for an existing file.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        if os.path.getsize(path) == 0:
            # Nothing recoverable in an empty file — treat as fresh start.
            # (Pre-created empty files are legitimate; strictness protects
            # data, and a 0-byte file has none to protect.)
            logger.warning(f"[SafeJSON] {label} at {path} is empty; starting fresh")
            return None
    except OSError:
        pass
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError, OSError, UnicodeDecodeError) as e:
        raise corrupt_store(path, label, e) from e


def corrupt_store(path: str, label: str, original: Exception) -> CorruptStoreError:
    """Quarantine a corrupt store file and build the error to raise.

    For load paths that can't use load_critical_json directly (e.g. a
    non-stdlib JSON parser): call this in the except block and raise it.
    """
    quarantine_path = quarantine_corrupt_file(path)
    logger.critical(
        f"[SafeJSON] {label} at {path} is corrupt or unreadable: {original}. "
        f"Quarantine copy: {quarantine_path or 'FAILED (original left in place)'}"
    )
    return CorruptStoreError(path, label, original, quarantine_path)


def quarantine_corrupt_file(path: str) -> Optional[str]:
    """Copy a corrupt file to <path>.corrupt-<timestamp>. Never deletes."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = f"{path}.corrupt-{ts}"
    try:
        shutil.copy2(path, dest)
        return dest
    except OSError as e:
        logger.error(f"[SafeJSON] Could not quarantine {path}: {e}")
        return None
