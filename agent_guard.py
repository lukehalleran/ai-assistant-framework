"""
agent_guard.py - Standalone agent safety guard for Python projects.

Module Contract
- Purpose: Protect critical repository paths from AI agent operations by
  monkey-patching destructive Python filesystem functions and classifying
  git/shell commands as safe or destructive. Single-file, drop-in, zero
  external dependencies.
- Inputs:
  - activate(repo_root, protected_dirs, protected_files, lockfile_name):
    configures guard scope and installs monkey-patches
  - agent_mode() context manager: marks code blocks as agent-originated
  - classify_git_command(args): git arg list (without leading 'git')
  - classify_shell_command(args, repo_root, protected_dirs, protected_files):
    shell arg list with optional config overrides
  - unlock_allowed(env, root, lockfile_name): explicit unlock check
- Outputs:
  - PermissionError raised on blocked operations (in agent_mode only)
  - Classification dicts: {subcmd, destructive, reason} for git;
    {command, destructive, severity, reason, targets} for shell
  - Boolean convenience: is_destructive_git(), is_destructive_shell()
- Key behaviors:
  - Three-layer guard: git commands, shell commands, Python fs calls
  - Python fs guard monkey-patches 10 functions: os.remove, os.unlink,
    os.rmdir, os.rename, os.replace, shutil.rmtree, shutil.move,
    shutil.copyfile, shutil.copy, shutil.copy2
  - Guards only fire inside agent_mode() context — normal app code unaffected
  - Two-tier blocking: always-blocked (., .., /, ~, *) never unlockable;
    protected paths unlockable via env var or lockfile
  - ContextVar-based agent_mode is async-safe (asyncio.gather compatible)
  - Path resolution uses PurePosixPath (no filesystem access) for
    shell classifier; Path.resolve() for Python fs guard
  - All classifiers are stateless and independently usable without activate()
- Side effects:
  - activate() replaces 10 stdlib functions with guarded wrappers
  - deactivate() restores originals
  - Logging via stdlib logging (logger name: "agent_guard")
- Dependencies: Python 3.10+ standard library only

Usage:
    from agent_guard import activate, agent_mode

    # 1. Activate the guard (call once at startup)
    activate(
        repo_root=".",
        protected_dirs={".git", "src", "config", "data"},
        protected_files={".env", "main.py"},
    )

    # 2. Wrap agent operations in agent_mode()
    with agent_mode():
        # These will raise PermissionError if targeting protected paths:
        os.remove("config/settings.yaml")   # BLOCKED
        shutil.rmtree("src/")               # BLOCKED
        os.remove("/tmp/scratch.txt")        # ALLOWED (outside repo)

    # Outside agent_mode(), everything passes through normally:
    os.remove("config/old_backup.yaml")      # ALLOWED (not in agent mode)

    # Classify commands without activating the monkey-patches:
    from agent_guard import classify_git_command, classify_shell_command
    classify_git_command(["push"])
    # -> {"subcmd": "push", "destructive": True, "reason": "git push is always destructive"}

    classify_shell_command(["rm", "-rf", "src/"], repo_root=".")
    # -> {"command": "rm", "destructive": True, "severity": "protected", ...}

Unlock mechanism:
    Protected-path blocks can be overridden when explicitly needed:
      - Set ALLOW_DESTRUCTIVE_OPS=1 in environment, or
      - Create a .agent_allow_destructive_once file in repo root (configurable)
    Always-blocked targets (., .., /, ~, *) can NEVER be unlocked.

License: MIT
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

__version__ = "0.1.0"

logger = logging.getLogger("agent_guard")

# ============================================================================
# Constants
# ============================================================================

# Targets that are ALWAYS blocked regardless of unlock or config.
# These represent catastrophic operations with no legitimate agent use case.
ALWAYS_BLOCKED_TARGETS: frozenset[str] = frozenset({".", "..", "/", "~", "*"})

# Commands whose args are inspected by the shell classifier.
_DESTRUCTIVE_COMMANDS: frozenset[str] = frozenset({
    "rm", "rmdir", "mv", "chmod", "truncate", "find",
})

# ============================================================================
# Module state (set by activate())
# ============================================================================

_active: bool = False
_repo_root: Path | None = None
_protected_dirs: frozenset[str] = frozenset()
_protected_files: frozenset[str] = frozenset()
_lockfile_name: str = ".agent_allow_destructive_once"
_originals: dict[str, Any] = {}

# Async-safe agent-mode flag.  Propagates through asyncio.gather automatically.
_agent_mode: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "agent_guard_agent_mode", default=False
)


# ############################################################################
#
#  SECTION 1 — Git command classifier (stateless, no config needed)
#
# ############################################################################

_ALWAYS_DESTRUCTIVE_GIT: set[str] = {"restore", "clean", "push"}

_CONDITIONAL_GIT_FLAGS: dict[str, set[str]] = {
    "reset": {"--hard", "--merge", "--keep"},
    "switch": {"-C"},
    "branch": {"-D"},
}


def classify_git_command(args: list[str]) -> dict[str, Any]:
    """Classify a git argument list (without the leading ``git``).

    Returns a dict with keys:
        subcmd:      the git subcommand (e.g. ``restore``, ``status``)
        destructive: ``True`` if the command is considered destructive
        reason:      human-readable reason, or ``None`` if safe
    """
    if not args:
        return {"subcmd": None, "destructive": False, "reason": None}

    subcmd = args[0]

    # Always destructive
    if subcmd in _ALWAYS_DESTRUCTIVE_GIT:
        return {
            "subcmd": subcmd,
            "destructive": True,
            "reason": f"git {subcmd} is always destructive",
        }

    # Conditionally destructive (flag-dependent)
    if subcmd in _CONDITIONAL_GIT_FLAGS:
        dangerous_flags = _CONDITIONAL_GIT_FLAGS[subcmd]
        for arg in args[1:]:
            if arg in dangerous_flags:
                return {
                    "subcmd": subcmd,
                    "destructive": True,
                    "reason": f"git {subcmd} {arg} is destructive",
                }
        return {"subcmd": subcmd, "destructive": False, "reason": None}

    # checkout: block ``--`` separator (file restore) but allow ``-b`` (branch)
    if subcmd == "checkout":
        rest = args[1:]
        for arg in rest:
            if arg in ("-b", "-B"):
                return {"subcmd": subcmd, "destructive": False, "reason": None}
        if "--" in rest:
            return {
                "subcmd": subcmd,
                "destructive": True,
                "reason": "git checkout -- <path> restores files",
            }
        return {"subcmd": subcmd, "destructive": False, "reason": None}

    return {"subcmd": subcmd, "destructive": False, "reason": None}


def is_destructive_git(args: list[str]) -> bool:
    """Return ``True`` if the git argument list is destructive."""
    return classify_git_command(args)["destructive"]


# ############################################################################
#
#  SECTION 2 — Unlock check
#
# ############################################################################

def unlock_allowed(
    env: dict[str, str] | None = None,
    root: str | Path | None = None,
    lockfile_name: str | None = None,
) -> bool:
    """Return ``True`` if destructive ops are explicitly unlocked.

    Checks (in order):
        1. ``ALLOW_DESTRUCTIVE_OPS=1`` in *env*
        2. *lockfile_name* file exists in *root*
    """
    if env is None:
        env = dict(os.environ)
    if env.get("ALLOW_DESTRUCTIVE_OPS") == "1":
        return True

    if root is not None:
        lf = lockfile_name or _lockfile_name
        lockfile = Path(root) / lf
        if lockfile.exists():
            return True

    return False


# ############################################################################
#
#  SECTION 3 — Path resolution helpers
#
# ############################################################################

def _resolve_target(target: str, repo_root: Path) -> str | None:
    """Normalize a target path to a repo-root-relative string.

    Returns ``None`` if the target resolves outside the repo root.
    Uses ``PurePosixPath`` to avoid filesystem access (testability).
    """
    if not target or target in ("--", "-"):
        return None
    if target.startswith("~"):
        return None

    target_path = PurePosixPath(target)

    # Absolute paths: check if under repo root
    if target_path.is_absolute():
        try:
            rel = PurePosixPath(target).relative_to(repo_root)
            return str(rel)
        except ValueError:
            return None

    # Relative paths: resolve via pure path math (no symlink following)
    combined = PurePosixPath(repo_root) / target_path
    parts: list[str] = []
    for part in combined.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != ".":
            parts.append(part)
    normalized = PurePosixPath(*parts) if parts else PurePosixPath(".")

    try:
        rel = normalized.relative_to(repo_root)
        result = str(rel)
        return result if result != "." else "."
    except ValueError:
        return None


def _is_protected(
    rel_path: str,
    protected_dirs: frozenset[str],
    protected_files: frozenset[str],
) -> bool:
    """Check if a repo-relative path is a protected path or inside one."""
    if rel_path in ALWAYS_BLOCKED_TARGETS:
        return True

    parts = PurePosixPath(rel_path).parts
    if not parts:
        return False

    # First component is a protected directory
    if parts[0] in protected_dirs:
        return True

    # Full path is a protected root-level file
    if len(parts) == 1 and parts[0] in protected_files:
        return True

    return False


# ############################################################################
#
#  SECTION 4 — Shell command classifiers (stateless — accept config as params)
#
# ############################################################################

def _safe_result(command: str | None, targets: list[str] | None = None) -> dict[str, Any]:
    return {
        "command": command,
        "destructive": False,
        "severity": None,
        "reason": None,
        "targets": targets or [],
    }


def _blocked_result(
    command: str, severity: str, reason: str, targets: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "command": command,
        "destructive": True,
        "severity": severity,
        "reason": reason,
        "targets": targets or [],
    }


# --- Flag parsing helpers ---------------------------------------------------

def _parse_rm_flags(args: list[str]) -> tuple[bool, bool]:
    """Parse rm-style flags. Returns ``(recursive, force)``."""
    recursive = False
    force = False
    for arg in args:
        if arg.startswith("--"):
            if arg == "--recursive":
                recursive = True
            elif arg == "--force":
                force = True
            continue
        if arg.startswith("-") and not arg.startswith("--"):
            flags = arg[1:]
            if "r" in flags or "R" in flags:
                recursive = True
            if "f" in flags:
                force = True
    return recursive, force


def _extract_targets(args: list[str]) -> list[str]:
    """Extract non-flag arguments from a command's arg list."""
    targets: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--":
            idx = args.index(arg)
            targets.extend(a for a in args[idx + 1:] if a)
            break
        if arg.startswith("-"):
            if arg in ("-s", "--size", "--reference", "--mode"):
                skip_next = True
            continue
        targets.append(arg)
    return targets


# --- Per-command classifiers -------------------------------------------------

def _classify_rm(
    args: list[str], repo_root: Path,
    p_dirs: frozenset[str], p_files: frozenset[str],
) -> dict[str, Any]:
    recursive, force = _parse_rm_flags(args)
    targets = _extract_targets(args)

    if not targets:
        return _safe_result("rm")

    resolved_targets = []
    for t in targets:
        if t in ALWAYS_BLOCKED_TARGETS:
            if recursive:
                return _blocked_result(
                    "rm", "always",
                    f"rm -r on '{t}' is always blocked (catastrophic)",
                    targets=[t],
                )
        rel = _resolve_target(t, repo_root)
        if rel is not None:
            resolved_targets.append((t, rel))

    for original, rel in resolved_targets:
        if rel == ".":
            if recursive:
                return _blocked_result(
                    "rm", "always",
                    "rm -r on repo root is always blocked",
                    targets=[original],
                )
        if _is_protected(rel, p_dirs, p_files):
            return _blocked_result(
                "rm", "protected",
                f"rm targets protected path '{rel}'",
                targets=[original],
            )

    return _safe_result("rm", [t for t, _ in resolved_targets])


def _classify_mv(
    args: list[str], repo_root: Path,
    p_dirs: frozenset[str], p_files: frozenset[str],
) -> dict[str, Any]:
    targets = _extract_targets(args)
    if len(targets) < 2:
        return _safe_result("mv")

    sources = targets[:-1]
    for src in sources:
        if src in ALWAYS_BLOCKED_TARGETS:
            return _blocked_result(
                "mv", "always",
                f"mv from '{src}' is always blocked (catastrophic)",
                targets=[src],
            )
        rel = _resolve_target(src, repo_root)
        if rel is not None and _is_protected(rel, p_dirs, p_files):
            return _blocked_result(
                "mv", "protected",
                f"mv targets protected path '{rel}'",
                targets=[src],
            )

    return _safe_result("mv", targets)


def _classify_rmdir(
    args: list[str], repo_root: Path,
    p_dirs: frozenset[str], p_files: frozenset[str],
) -> dict[str, Any]:
    targets = _extract_targets(args)
    for t in targets:
        if t in ALWAYS_BLOCKED_TARGETS:
            return _blocked_result(
                "rmdir", "always",
                f"rmdir on '{t}' is always blocked",
                targets=[t],
            )
        rel = _resolve_target(t, repo_root)
        if rel is not None and _is_protected(rel, p_dirs, p_files):
            return _blocked_result(
                "rmdir", "protected",
                f"rmdir targets protected path '{rel}'",
                targets=[t],
            )
    return _safe_result("rmdir", targets)


def _classify_chmod(
    args: list[str], repo_root: Path,
    p_dirs: frozenset[str], p_files: frozenset[str],
) -> dict[str, Any]:
    recursive = False
    mode = None
    targets: list[str] = []

    for arg in args:
        if arg in ("-R", "--recursive"):
            recursive = True
        elif arg.startswith("-"):
            continue
        elif mode is None:
            mode = arg
        else:
            targets.append(arg)

    if not targets or mode is None:
        return _safe_result("chmod")

    is_restrictive = mode in ("000", "0000", "100", "0100", "200", "0200")

    for t in targets:
        rel = _resolve_target(t, repo_root)
        if rel is not None and _is_protected(rel, p_dirs, p_files):
            if recursive or is_restrictive:
                return _blocked_result(
                    "chmod", "protected",
                    f"chmod {mode}{' -R' if recursive else ''} on protected path '{rel}'",
                    targets=[t],
                )

    return _safe_result("chmod", targets)


def _classify_truncate(
    args: list[str], repo_root: Path,
    p_dirs: frozenset[str], p_files: frozenset[str],
) -> dict[str, Any]:
    targets: list[str] = []
    skip_next = False

    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in ("-s", "--size", "-r", "--reference"):
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        targets.append(arg)

    for t in targets:
        rel = _resolve_target(t, repo_root)
        if rel is not None and _is_protected(rel, p_dirs, p_files):
            return _blocked_result(
                "truncate", "protected",
                f"truncate targets protected path '{rel}'",
                targets=[t],
            )

    return _safe_result("truncate", targets)


def _classify_find(
    args: list[str], repo_root: Path,
    p_dirs: frozenset[str], p_files: frozenset[str],
) -> dict[str, Any]:
    has_delete = "-delete" in args
    has_exec_rm = False

    for i, arg in enumerate(args):
        if arg == "-exec" and i + 1 < len(args) and "rm" in args[i + 1]:
            has_exec_rm = True
            break

    if not has_delete and not has_exec_rm:
        return _safe_result("find")

    search_paths: list[str] = []
    for arg in args:
        if arg.startswith("-") or arg.startswith("(") or arg.startswith("!"):
            break
        search_paths.append(arg)

    action = "-delete" if has_delete else "-exec rm"

    for p in search_paths:
        if p in ALWAYS_BLOCKED_TARGETS:
            return _blocked_result(
                "find", "always",
                f"find with {action} on '{p}' is always blocked",
                targets=[p],
            )
        rel = _resolve_target(p, repo_root)
        if rel is not None and _is_protected(rel, p_dirs, p_files):
            return _blocked_result(
                "find", "protected",
                f"find with {action} on protected path '{rel}'",
                targets=[p],
            )

    return _safe_result("find", search_paths)


# --- Main shell classifier ---------------------------------------------------

_SHELL_CLASSIFIERS = {
    "rm": _classify_rm,
    "rmdir": _classify_rmdir,
    "mv": _classify_mv,
    "chmod": _classify_chmod,
    "truncate": _classify_truncate,
    "find": _classify_find,
}


def classify_shell_command(
    args: list[str],
    repo_root: str | Path | None = None,
    protected_dirs: set[str] | frozenset[str] | None = None,
    protected_files: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Classify a shell command as safe or destructive.

    Can be used standalone (pass all params) or after ``activate()``
    (falls back to stored config for any param not provided).

    Args:
        args: Command argument list, e.g. ``["rm", "-rf", "src/"]``.
        repo_root: Repository root path.  Falls back to stored config or cwd.
        protected_dirs: Directory names to protect.  Falls back to stored config.
        protected_files: File names to protect.  Falls back to stored config.

    Returns:
        dict with keys: ``command``, ``destructive``, ``severity``, ``reason``, ``targets``

        Severity is ``"always"`` (never unlockable), ``"protected"`` (unlockable),
        or ``None`` (safe).
    """
    if not args:
        return _safe_result(None)

    command = os.path.basename(args[0])
    rest = args[1:]

    root = Path(repo_root) if repo_root else (_repo_root or Path.cwd())
    p_dirs = frozenset(protected_dirs) if protected_dirs is not None else _protected_dirs
    p_files = frozenset(protected_files) if protected_files is not None else _protected_files

    classifier = _SHELL_CLASSIFIERS.get(command)
    if classifier is None:
        return _safe_result(command)

    return classifier(rest, root, p_dirs, p_files)


def is_destructive_shell(
    args: list[str],
    repo_root: str | Path | None = None,
    protected_dirs: set[str] | frozenset[str] | None = None,
    protected_files: set[str] | frozenset[str] | None = None,
) -> bool:
    """Return ``True`` if the shell command is destructive."""
    return classify_shell_command(args, repo_root, protected_dirs, protected_files)["destructive"]


# ############################################################################
#
#  SECTION 5 — Python filesystem guard (monkey-patching)
#
# ############################################################################

def activate(
    repo_root: str | Path = ".",
    protected_dirs: set[str] | frozenset[str] | None = None,
    protected_files: set[str] | frozenset[str] | None = None,
    lockfile_name: str = ".agent_allow_destructive_once",
) -> None:
    """Install monkey-patches on destructive filesystem functions.

    Call once at startup.  Idempotent - calling twice is a no-op.

    Args:
        repo_root: Path to the repository root.  Defaults to current directory.
        protected_dirs: Set of directory names (repo-root-relative) to protect.
            Defaults to ``{".git"}``.
        protected_files: Set of file names (repo-root-relative) to protect.
            Defaults to empty set.
        lockfile_name: Name of the one-shot unlock file.
            Defaults to ``".agent_allow_destructive_once"``.

    After activation, the following functions are guarded:
        ``os.remove``, ``os.unlink``, ``os.rmdir``, ``os.rename``, ``os.replace``,
        ``shutil.rmtree``, ``shutil.move``, ``shutil.copyfile``, ``shutil.copy``,
        ``shutil.copy2``

    Guarded functions only block when **both** conditions are true:
        1. The guard is active (``activate()`` was called)
        2. Code is running inside an ``agent_mode()`` context

    Outside ``agent_mode()``, all operations pass through unmodified.
    """
    global _active, _repo_root, _protected_dirs, _protected_files, _lockfile_name

    if _active:
        return

    _repo_root = Path(repo_root).resolve()

    if not _repo_root.is_dir():
        logger.error(
            f"[AgentGuard] Repo root is not a directory: {_repo_root}. "
            "Guard NOT activated."
        )
        return

    _protected_dirs = frozenset(protected_dirs) if protected_dirs is not None else frozenset({".git"})
    _protected_files = frozenset(protected_files) if protected_files is not None else frozenset()
    _lockfile_name = lockfile_name

    # Save originals
    _originals["os.remove"] = os.remove
    _originals["os.unlink"] = os.unlink
    _originals["os.rmdir"] = os.rmdir
    _originals["os.rename"] = os.rename
    _originals["os.replace"] = os.replace
    _originals["shutil.rmtree"] = shutil.rmtree
    _originals["shutil.move"] = shutil.move
    _originals["shutil.copyfile"] = shutil.copyfile
    _originals["shutil.copy"] = shutil.copy
    _originals["shutil.copy2"] = shutil.copy2

    # Install patches
    os.remove = _guarded_remove
    os.unlink = _guarded_unlink
    os.rmdir = _guarded_rmdir
    os.rename = _guarded_rename
    os.replace = _guarded_replace
    shutil.rmtree = _guarded_rmtree
    shutil.move = _guarded_move
    shutil.copyfile = _guarded_copyfile
    shutil.copy = _guarded_copy
    shutil.copy2 = _guarded_copy2

    _active = True
    logger.info(
        f"[AgentGuard] Activated. root={_repo_root} "
        f"dirs={sorted(_protected_dirs)} files={sorted(_protected_files)}"
    )


def deactivate() -> None:
    """Restore original filesystem functions.  Idempotent."""
    global _active, _repo_root, _protected_dirs, _protected_files

    if not _active:
        return

    os.remove = _originals["os.remove"]
    os.unlink = _originals["os.unlink"]
    os.rmdir = _originals["os.rmdir"]
    os.rename = _originals["os.rename"]
    os.replace = _originals["os.replace"]
    shutil.rmtree = _originals["shutil.rmtree"]
    shutil.move = _originals["shutil.move"]
    shutil.copyfile = _originals["shutil.copyfile"]
    shutil.copy = _originals["shutil.copy"]
    shutil.copy2 = _originals["shutil.copy2"]

    _originals.clear()
    _active = False
    _repo_root = None
    _protected_dirs = frozenset()
    _protected_files = frozenset()
    logger.info("[AgentGuard] Deactivated. Original functions restored.")


def is_active() -> bool:
    """Return whether the guard patches are currently installed."""
    return _active


def set_agent_mode(active: bool = True) -> None:
    """Set the agent-mode flag on the current async context."""
    _agent_mode.set(active)


@contextmanager
def agent_mode() -> Generator[None, None, None]:
    """Context manager that enables agent-mode checks for the enclosed block.

    Uses ``ContextVar`` tokens so nested contexts and exceptions reset correctly.
    Safe for use with ``asyncio.gather()`` - each task gets its own copy.

    Example::

        with agent_mode():
            await run_agent_tool(...)  # guarded
        # unguarded again here
    """
    token = _agent_mode.set(True)
    try:
        yield
    finally:
        _agent_mode.reset(token)


# --- Path checking for the Python guard ------------------------------------

def _normalize_path_str(target: Any) -> str:
    return str(target).rstrip("/").rstrip("\\") if target is not None else ""


def _resolve_to_repo_relative(target: Any) -> str | None:
    """Resolve a target path to a repo-root-relative string via the real filesystem."""
    if _repo_root is None:
        return None
    try:
        resolved = Path(str(target)).resolve()
        rel = resolved.relative_to(_repo_root)
        return str(rel)
    except (ValueError, OSError):
        return None


def _is_protected_path(rel_path: str) -> bool:
    """Check if a repo-relative path is protected (uses stored config)."""
    parts = PurePosixPath(rel_path).parts
    if not parts:
        return False
    if parts[0] in _protected_dirs:
        return True
    if len(parts) == 1 and parts[0] in _protected_files:
        return True
    return False


def _is_always_blocked(raw: str, resolved_rel: str | None) -> bool:
    """Check if a target matches always-blocked patterns."""
    if raw in ALWAYS_BLOCKED_TARGETS:
        return True
    raw_normalized = raw.rstrip("/").rstrip("\\")
    if raw_normalized and raw_normalized in ALWAYS_BLOCKED_TARGETS:
        return True
    if raw_normalized == "" and raw in ("/", "\\"):
        return True

    if resolved_rel is not None:
        if resolved_rel in ALWAYS_BLOCKED_TARGETS:
            return True
        rel_normalized = resolved_rel.rstrip("/").rstrip("\\")
        if rel_normalized and rel_normalized in ALWAYS_BLOCKED_TARGETS:
            return True
        if rel_normalized in (".", ""):
            return True

    if _repo_root is not None:
        try:
            resolved_abs = Path(raw).resolve()
            if resolved_abs == _repo_root:
                return True
        except (OSError, ValueError):
            pass

    return False


# --- Core guard logic -------------------------------------------------------

def _check_and_maybe_block(operation: str, target: Any) -> None:
    """Check whether a filesystem operation on target should be blocked.

    Decision order:
        1. Guard not active -> pass
        2. Not in agent mode -> pass
        3. Raw target is always-blocked -> raise (no unlock)
        4. Resolve path
        5. Resolved is always-blocked or repo root -> raise (no unlock)
        6. Outside repo -> pass
        7. Not protected -> pass
        8. Unlock in effect -> warn + pass
        9. Otherwise -> raise with unlock instructions
    """
    if not _active:
        return
    if not _agent_mode.get(False):
        return

    raw_original = str(target) if target is not None else ""
    raw = _normalize_path_str(target)
    if not raw and not raw_original:
        return

    # Always-blocked (raw form)
    if raw_original in ALWAYS_BLOCKED_TARGETS or _is_always_blocked(raw_original, None):
        msg = (
            f"[AgentGuard] BLOCKED {operation}: '{raw_original}' is always blocked "
            f"(catastrophic target). No unlock available."
        )
        logger.error(msg)
        raise PermissionError(msg)

    # Resolve
    resolved_rel = _resolve_to_repo_relative(target)

    # Always-blocked (resolved form)
    if _is_always_blocked(raw_original, resolved_rel):
        msg = (
            f"[AgentGuard] BLOCKED {operation}: '{raw}' resolves to "
            f"always-blocked target or repo root. No unlock available."
        )
        logger.error(msg)
        raise PermissionError(msg)

    # Outside repo
    if resolved_rel is None:
        return

    # Not protected
    if not _is_protected_path(resolved_rel):
        return

    # Explicit unlock
    if unlock_allowed(root=_repo_root, lockfile_name=_lockfile_name):
        logger.warning(
            f"[AgentGuard] ALLOWED (unlocked) {operation}: '{raw}' "
            f"(resolved: '{resolved_rel}'). Explicit unlock is in effect."
        )
        return

    # Blocked
    msg = (
        f"[AgentGuard] BLOCKED {operation}: '{raw}' targets protected "
        f"path '{resolved_rel}'.\n"
        f"To proceed, use one of:\n"
        f"  ALLOW_DESTRUCTIVE_OPS=1 (environment variable)\n"
        f"  touch {_repo_root / _lockfile_name} (one-shot lockfile)"
    )
    logger.error(msg)
    raise PermissionError(msg)


# --- Guarded wrappers -------------------------------------------------------

def _guarded_remove(path, *args, **kwargs):
    _check_and_maybe_block("os.remove", path)
    return _originals["os.remove"](path, *args, **kwargs)


def _guarded_unlink(path, *args, **kwargs):
    _check_and_maybe_block("os.unlink", path)
    return _originals["os.unlink"](path, *args, **kwargs)


def _guarded_rmdir(path, *args, **kwargs):
    _check_and_maybe_block("os.rmdir", path)
    return _originals["os.rmdir"](path, *args, **kwargs)


def _guarded_rename(src, dst, *args, **kwargs):
    _check_and_maybe_block("os.rename (source)", src)
    _check_and_maybe_block("os.rename (destination)", dst)
    return _originals["os.rename"](src, dst, *args, **kwargs)


def _guarded_replace(src, dst, *args, **kwargs):
    _check_and_maybe_block("os.replace (source)", src)
    _check_and_maybe_block("os.replace (destination)", dst)
    return _originals["os.replace"](src, dst, *args, **kwargs)


def _guarded_rmtree(path, *args, **kwargs):
    _check_and_maybe_block("shutil.rmtree", path)
    return _originals["shutil.rmtree"](path, *args, **kwargs)


def _guarded_move(src, dst, *args, **kwargs):
    _check_and_maybe_block("shutil.move (source)", src)
    _check_and_maybe_block("shutil.move (destination)", dst)
    return _originals["shutil.move"](src, dst, *args, **kwargs)


def _guarded_copyfile(src, dst, *args, **kwargs):
    _check_and_maybe_block("shutil.copyfile (destination)", dst)
    return _originals["shutil.copyfile"](src, dst, *args, **kwargs)


def _guarded_copy(src, dst, *args, **kwargs):
    _check_and_maybe_block("shutil.copy (destination)", dst)
    return _originals["shutil.copy"](src, dst, *args, **kwargs)


def _guarded_copy2(src, dst, *args, **kwargs):
    _check_and_maybe_block("shutil.copy2 (destination)", dst)
    return _originals["shutil.copy2"](src, dst, *args, **kwargs)
