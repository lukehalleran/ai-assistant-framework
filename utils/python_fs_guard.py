"""Python-level filesystem guard for agent session safety.

Purpose: Monkey-patch destructive Python filesystem operations (os.remove,
         os.unlink, os.rmdir, os.rename, os.replace, shutil.rmtree, shutil.move)
         to protect critical repo paths from agent-originated deletion, move,
         or overwrite during in-process agentic tool dispatch.

Scope:   This closes the in-process Python delete/move/replace bypass for
         agentic tool dispatch.  It does NOT protect arbitrary file writes,
         copy-overwrites, subprocesses, or separate Python interpreters.

Inputs:  activate(repo_root), deactivate(), agent_mode() context manager.
Outputs: PermissionError on blocked operations; warning logs on blocks/unlocks.
Dependencies: Standard library + shell_cmd_guard (protected path sets) +
             destructive_op_guard (unlock_allowed).
"""

from __future__ import annotations

import contextvars
import logging
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Generator

from utils.destructive_op_guard import unlock_allowed
from utils.shell_cmd_guard import (
    ALWAYS_BLOCKED_TARGETS,
    PROTECTED_DIRS,
    PROTECTED_FILES,
)

logger = logging.getLogger("python_fs_guard")

# ============================================================================
# Module-level state
# ============================================================================

_active: bool = False
_repo_root: Path | None = None
_originals: dict[str, Any] = {}

# Async-safe agent-mode flag.  Propagates through asyncio.gather automatically.
_agent_mode: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "python_fs_guard_agent_mode", default=False
)


# ============================================================================
# Public API
# ============================================================================

def activate(repo_root: str | Path | None = None) -> None:
    """Install monkey-patches on destructive filesystem functions.

    Idempotent — calling twice is a no-op.  In frozen mode, attempts
    activation with the executable directory as root and logs a warning
    if no valid root can be resolved.
    """
    global _active, _repo_root

    if _active:
        return

    # Resolve repo root
    if repo_root is not None:
        _repo_root = Path(repo_root).resolve()
    elif getattr(sys, "frozen", False):
        # Frozen mode: use executable directory, but warn
        _repo_root = Path(sys.executable).resolve().parent
        logger.warning(
            "[PythonFSGuard] Running in frozen mode. Using executable directory "
            f"as repo root: {_repo_root}. Protection may be limited."
        )
    else:
        # Infer from this file's location: utils/python_fs_guard.py -> repo root
        _repo_root = Path(__file__).resolve().parent.parent

    if not _repo_root.is_dir():
        logger.error(
            f"[PythonFSGuard] Repo root is not a directory: {_repo_root}. "
            "Python filesystem guard NOT activated."
        )
        return

    # Save originals
    _originals["os.remove"] = os.remove
    _originals["os.unlink"] = os.unlink
    _originals["os.rmdir"] = os.rmdir
    _originals["os.rename"] = os.rename
    _originals["os.replace"] = os.replace
    _originals["shutil.rmtree"] = shutil.rmtree
    _originals["shutil.move"] = shutil.move

    # Install patches
    os.remove = _guarded_remove
    os.unlink = _guarded_unlink
    os.rmdir = _guarded_rmdir
    os.rename = _guarded_rename
    os.replace = _guarded_replace
    shutil.rmtree = _guarded_rmtree
    shutil.move = _guarded_move

    _active = True
    logger.info(
        f"[PythonFSGuard] Activated. Repo root: {_repo_root}. "
        "Guarding: os.remove, os.unlink, os.rmdir, os.rename, os.replace, "
        "shutil.rmtree, shutil.move"
    )


def deactivate() -> None:
    """Restore original filesystem functions.  Idempotent."""
    global _active, _repo_root

    if not _active:
        return

    os.remove = _originals["os.remove"]
    os.unlink = _originals["os.unlink"]
    os.rmdir = _originals["os.rmdir"]
    os.rename = _originals["os.rename"]
    os.replace = _originals["os.replace"]
    shutil.rmtree = _originals["shutil.rmtree"]
    shutil.move = _originals["shutil.move"]

    _originals.clear()
    _active = False
    _repo_root = None
    logger.info("[PythonFSGuard] Deactivated. Original functions restored.")


def is_active() -> bool:
    """Return whether the guard patches are currently installed."""
    return _active


def set_agent_mode(active: bool = True) -> None:
    """Set the agent-mode flag on the current async context."""
    _agent_mode.set(active)


@contextmanager
def agent_mode() -> Generator[None, None, None]:
    """Context manager that enables agent-mode checks for the enclosed block.

    Uses ContextVar tokens so nested contexts and exceptions reset correctly.
    """
    token = _agent_mode.set(True)
    try:
        yield
    finally:
        _agent_mode.reset(token)


# ============================================================================
# Path resolution and protection checks
# ============================================================================

def _normalize_path_str(target: Any) -> str:
    """Convert a target to a string for raw checks."""
    return str(target).rstrip("/").rstrip("\\") if target is not None else ""


def _resolve_to_repo_relative(target: Any) -> str | None:
    """Resolve a target path to a repo-root-relative string.

    Returns None if the target resolves outside the repo root.
    """
    if _repo_root is None:
        return None

    try:
        resolved = Path(str(target)).resolve()
        rel = resolved.relative_to(_repo_root)
        return str(rel)
    except (ValueError, OSError):
        return None


def _is_protected_path(rel_path: str) -> bool:
    """Check if a repo-relative path is protected or inside a protected dir."""
    parts = PurePosixPath(rel_path).parts
    if not parts:
        return False

    # First component is a protected directory
    if parts[0] in PROTECTED_DIRS:
        return True

    # Full path is a protected root-level file
    if len(parts) == 1 and parts[0] in PROTECTED_FILES:
        return True

    return False


def _is_always_blocked(raw: str, resolved_rel: str | None) -> bool:
    """Check if a target matches always-blocked patterns.

    Checks both the raw input string and the resolved repo-relative path.
    Also blocks operations targeting the repo root itself.
    """
    # Check raw input — check both as-is and stripped
    if raw in ALWAYS_BLOCKED_TARGETS:
        return True
    raw_normalized = raw.rstrip("/").rstrip("\\")
    if raw_normalized and raw_normalized in ALWAYS_BLOCKED_TARGETS:
        return True
    # "/" strips to "" — handle explicitly
    if raw_normalized == "" and raw in ("/", "\\"):
        return True

    # Check resolved path
    if resolved_rel is not None:
        if resolved_rel in ALWAYS_BLOCKED_TARGETS:
            return True
        rel_normalized = resolved_rel.rstrip("/").rstrip("\\")
        if rel_normalized and rel_normalized in ALWAYS_BLOCKED_TARGETS:
            return True
        # Repo root itself (resolves to "." or "")
        if rel_normalized in (".", ""):
            return True

    # Check if resolved absolute path IS the repo root
    if _repo_root is not None:
        try:
            resolved_abs = Path(raw).resolve()
            if resolved_abs == _repo_root:
                return True
        except (OSError, ValueError):
            pass

    return False


# ============================================================================
# Core guard logic
# ============================================================================

def _check_and_maybe_block(operation: str, target: Any) -> None:
    """Check whether a filesystem operation on target should be blocked.

    Corrected ordering:
    1. Not active? → pass through
    2. Not in agent mode? → pass through
    3. Raw target is always-blocked? → raise PermissionError, no unlock
    4. Resolve path
    5. Resolved path is always-blocked or equals repo root? → raise
    6. Outside repo? → pass through
    7. Not protected? → pass through
    8. unlock_allowed()? → log warning, pass through
    9. Otherwise → raise PermissionError with unlock instructions
    """
    # 1. Guard not active
    if not _active:
        return

    # 2. Not in agent mode — Daemon's own runtime is unguarded
    if not _agent_mode.get(False):
        return

    raw_original = str(target) if target is not None else ""
    raw = _normalize_path_str(target)
    if not raw and not raw_original:
        return

    # 3. Raw target is always-blocked — no unlock possible
    # Check both original and normalized forms (e.g. "/" normalizes to "")
    if raw_original in ALWAYS_BLOCKED_TARGETS or _is_always_blocked(raw_original, None):
        msg = (
            f"[PythonFSGuard] BLOCKED {operation}: '{raw_original}' is always blocked "
            f"(catastrophic target). No unlock available."
        )
        logger.error(msg)
        raise PermissionError(msg)

    # 4. Resolve path to repo-relative
    resolved_rel = _resolve_to_repo_relative(target)

    # 5. Resolved path is always-blocked or equals repo root
    if _is_always_blocked(raw_original, resolved_rel):
        msg = (
            f"[PythonFSGuard] BLOCKED {operation}: '{raw}' resolves to "
            f"always-blocked target or repo root. No unlock available."
        )
        logger.error(msg)
        raise PermissionError(msg)

    # 6. Outside repo — not our business
    if resolved_rel is None:
        return

    # 7. Not a protected path — allow
    if not _is_protected_path(resolved_rel):
        return

    # 8. Explicit unlock in effect
    if unlock_allowed(root=_repo_root):
        logger.warning(
            f"[PythonFSGuard] ALLOWED (unlocked) {operation}: '{raw}' "
            f"(resolved: '{resolved_rel}'). Explicit unlock is in effect."
        )
        return

    # 9. Blocked — protected path, no unlock
    msg = (
        f"[PythonFSGuard] BLOCKED {operation}: '{raw}' targets protected "
        f"path '{resolved_rel}'.\n"
        f"To proceed, use one of:\n"
        f"  ALLOW_DESTRUCTIVE_OPS=1 (environment variable)\n"
        f"  touch {_repo_root / '.agent_allow_destructive_once'} (one-shot lockfile)"
    )
    logger.error(msg)
    raise PermissionError(msg)


# ============================================================================
# Guarded wrappers — preserve *args/**kwargs for platform compatibility
# ============================================================================

def _guarded_remove(path, *args, **kwargs):
    """Guarded os.remove — checks single target path."""
    _check_and_maybe_block("os.remove", path)
    return _originals["os.remove"](path, *args, **kwargs)


def _guarded_unlink(path, *args, **kwargs):
    """Guarded os.unlink — checks single target path. Covers Path.unlink()."""
    _check_and_maybe_block("os.unlink", path)
    return _originals["os.unlink"](path, *args, **kwargs)


def _guarded_rmdir(path, *args, **kwargs):
    """Guarded os.rmdir — checks single target path. Covers Path.rmdir()."""
    _check_and_maybe_block("os.rmdir", path)
    return _originals["os.rmdir"](path, *args, **kwargs)


def _guarded_rename(src, dst, *args, **kwargs):
    """Guarded os.rename — checks both source and destination. Covers Path.rename()."""
    _check_and_maybe_block("os.rename (source)", src)
    _check_and_maybe_block("os.rename (destination)", dst)
    return _originals["os.rename"](src, dst, *args, **kwargs)


def _guarded_replace(src, dst, *args, **kwargs):
    """Guarded os.replace — checks both source and destination. Covers Path.replace()."""
    _check_and_maybe_block("os.replace (source)", src)
    _check_and_maybe_block("os.replace (destination)", dst)
    return _originals["os.replace"](src, dst, *args, **kwargs)


def _guarded_rmtree(path, *args, **kwargs):
    """Guarded shutil.rmtree — checks top-level path."""
    _check_and_maybe_block("shutil.rmtree", path)
    return _originals["shutil.rmtree"](path, *args, **kwargs)


def _guarded_move(src, dst, *args, **kwargs):
    """Guarded shutil.move — checks both source and destination."""
    _check_and_maybe_block("shutil.move (source)", src)
    _check_and_maybe_block("shutil.move (destination)", dst)
    return _originals["shutil.move"](src, dst, *args, **kwargs)
