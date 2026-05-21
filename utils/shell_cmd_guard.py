"""Destructive shell command classifier for agent session safety.

Purpose: Classify shell command argument lists as safe or destructive based
on command type, flags, and whether targets are protected paths. Used by
safe_cmd.sh wrapper and by tests.

Inputs:  Command argument list, optional repo root path.
Outputs: Classification dict, boolean destructive check.
Dependencies: Standard library only. Imports unlock_allowed from
             destructive_op_guard.py.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

# ============================================================================
# Protected paths (repo-root-relative)
# ============================================================================

# Directories that should never be deleted/moved/clobbered by an agent.
PROTECTED_DIRS: frozenset[str] = frozenset({
    "data",
    "config",
    ".git",
    "scripts",
    "memory",
    "core",
    "knowledge",
    "utils",
    "models",
    "gui",
    "eval",
    "integrations",
    "processing",
    "docs",
    "tests",
    "conversation_logs",
})

# Individual root-level files that should never be deleted/moved.
PROTECTED_FILES: frozenset[str] = frozenset({
    "main.py",
    "CLAUDE.md",
    "requirements.txt",
    ".env",
    "pytest.ini",
    "daemon.spec",
})

# Targets that are always blocked regardless of unlock.
# These represent catastrophic operations with no legitimate agent use case.
ALWAYS_BLOCKED_TARGETS: frozenset[str] = frozenset({
    ".",
    "..",
    "/",
    "~",
    "*",
})

# Commands that can be destructive depending on args/targets.
_DESTRUCTIVE_COMMANDS: frozenset[str] = frozenset({
    "rm", "rmdir", "mv", "chmod", "truncate", "find",
})


# ============================================================================
# Result helpers
# ============================================================================

def _safe_result(command: str | None, targets: list[str] | None = None) -> dict[str, Any]:
    return {
        "command": command,
        "destructive": False,
        "severity": None,
        "reason": None,
        "targets": targets or [],
    }


def _blocked_result(
    command: str,
    severity: str,
    reason: str,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "command": command,
        "destructive": True,
        "severity": severity,
        "reason": reason,
        "targets": targets or [],
    }


# ============================================================================
# Path resolution
# ============================================================================

def _resolve_target(target: str, repo_root: Path) -> str | None:
    """Normalize a target path to a repo-root-relative string.

    Returns None if the target resolves outside the repo root.
    Does NOT follow symlinks (avoids filesystem access for testability).
    """
    if not target or target in ("--", "-"):
        return None

    # Expand ~ but don't resolve (keeps testability)
    if target.startswith("~"):
        return None  # outside repo

    target_path = PurePosixPath(target)

    # Absolute paths: check if under repo root
    if target_path.is_absolute():
        repo_str = str(repo_root)
        # Normalize both for comparison
        try:
            rel = PurePosixPath(target).relative_to(repo_root)
            return str(rel)
        except ValueError:
            return None  # outside repo

    # Relative paths: resolve against repo root using pure path math
    combined = PurePosixPath(repo_root) / target_path
    # Normalize ".." components
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
        return None  # traversed outside repo


def _is_protected(rel_path: str, protected_dirs: frozenset[str], protected_files: frozenset[str]) -> bool:
    """Check if a repo-relative path is a protected path or inside one."""
    if rel_path in ALWAYS_BLOCKED_TARGETS:
        return True

    parts = PurePosixPath(rel_path).parts
    if not parts:
        return False

    # Check if the first component is a protected directory
    if parts[0] in protected_dirs:
        return True

    # Check if the full path is a protected root-level file
    if len(parts) == 1 and parts[0] in protected_files:
        return True

    return False


# ============================================================================
# Flag parsing
# ============================================================================

def _parse_rm_flags(args: list[str]) -> tuple[bool, bool]:
    """Parse rm-style flags from args. Returns (recursive, force).

    Handles combined flags like -rf, -Rf, -fr, as well as long-form
    --recursive and --force.
    """
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
    """Extract non-flag arguments from a command's args (everything after the command)."""
    targets: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--":
            # Everything after -- is a target
            idx = args.index(arg)
            targets.extend(a for a in args[idx + 1:] if a)
            break
        if arg.startswith("-"):
            # Some flags take a value argument
            if arg in ("-s", "--size", "--reference", "--mode"):
                skip_next = True
            continue
        targets.append(arg)
    return targets


# ============================================================================
# Per-command classifiers
# ============================================================================

def _classify_rm(args: list[str], repo_root: Path) -> dict[str, Any]:
    """Classify an rm command."""
    recursive, force = _parse_rm_flags(args)
    targets = _extract_targets(args)

    if not targets:
        return _safe_result("rm")

    resolved_targets = []
    for t in targets:
        # Check always-blocked targets first (before resolution)
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
        if _is_protected(rel, PROTECTED_DIRS, PROTECTED_FILES):
            # Block recursive rm on any protected path, and also block
            # non-recursive rm on protected files or files inside protected dirs
            return _blocked_result(
                "rm", "protected",
                f"rm targets protected path '{rel}'",
                targets=[original],
            )

    return _safe_result("rm", [t for t, _ in resolved_targets])


def _classify_mv(args: list[str], repo_root: Path) -> dict[str, Any]:
    """Classify an mv command. Blocks if source is a protected path."""
    targets = _extract_targets(args)

    if len(targets) < 2:
        return _safe_result("mv")

    # Sources are all targets except the last (destination)
    sources = targets[:-1]

    for src in sources:
        if src in ALWAYS_BLOCKED_TARGETS:
            return _blocked_result(
                "mv", "always",
                f"mv from '{src}' is always blocked (catastrophic)",
                targets=[src],
            )
        rel = _resolve_target(src, repo_root)
        if rel is not None and _is_protected(rel, PROTECTED_DIRS, PROTECTED_FILES):
            return _blocked_result(
                "mv", "protected",
                f"mv targets protected path '{rel}'",
                targets=[src],
            )

    return _safe_result("mv", targets)


def _classify_rmdir(args: list[str], repo_root: Path) -> dict[str, Any]:
    """Classify an rmdir command."""
    targets = _extract_targets(args)

    for t in targets:
        if t in ALWAYS_BLOCKED_TARGETS:
            return _blocked_result(
                "rmdir", "always",
                f"rmdir on '{t}' is always blocked",
                targets=[t],
            )
        rel = _resolve_target(t, repo_root)
        if rel is not None and _is_protected(rel, PROTECTED_DIRS, PROTECTED_FILES):
            return _blocked_result(
                "rmdir", "protected",
                f"rmdir targets protected path '{rel}'",
                targets=[t],
            )

    return _safe_result("rmdir", targets)


def _classify_chmod(args: list[str], repo_root: Path) -> dict[str, Any]:
    """Classify a chmod command. Blocks chmod 000 or chmod -R on protected paths."""
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

    # Block restrictive modes on protected paths
    is_restrictive = mode in ("000", "0000", "100", "0100", "200", "0200")

    for t in targets:
        rel = _resolve_target(t, repo_root)
        if rel is not None and _is_protected(rel, PROTECTED_DIRS, PROTECTED_FILES):
            if recursive or is_restrictive:
                return _blocked_result(
                    "chmod", "protected",
                    f"chmod {mode}{' -R' if recursive else ''} on protected path '{rel}'",
                    targets=[t],
                )

    return _safe_result("chmod", targets)


def _classify_truncate(args: list[str], repo_root: Path) -> dict[str, Any]:
    """Classify a truncate command."""
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
        if rel is not None and _is_protected(rel, PROTECTED_DIRS, PROTECTED_FILES):
            return _blocked_result(
                "truncate", "protected",
                f"truncate targets protected path '{rel}'",
                targets=[t],
            )

    return _safe_result("truncate", targets)


def _classify_find(args: list[str], repo_root: Path) -> dict[str, Any]:
    """Classify a find command. Blocks find with -delete or -exec rm on protected paths."""
    has_delete = "-delete" in args
    has_exec_rm = False

    for i, arg in enumerate(args):
        if arg == "-exec" and i + 1 < len(args) and "rm" in args[i + 1]:
            has_exec_rm = True
            break

    if not has_delete and not has_exec_rm:
        return _safe_result("find")

    # Find the search paths (args before the first flag/expression)
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
        if rel is not None and _is_protected(rel, PROTECTED_DIRS, PROTECTED_FILES):
            return _blocked_result(
                "find", "protected",
                f"find with {action} on protected path '{rel}'",
                targets=[p],
            )

    return _safe_result("find", search_paths)


# ============================================================================
# Main classifier
# ============================================================================

_CLASSIFIERS = {
    "rm": _classify_rm,
    "rmdir": _classify_rmdir,
    "mv": _classify_mv,
    "chmod": _classify_chmod,
    "truncate": _classify_truncate,
    "find": _classify_find,
}


def classify_shell_cmd(
    args: list[str],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Classify a shell command (split into args list).

    Returns a dict with:
        command:     the base command (e.g. 'rm', 'mv')
        destructive: True if the command is considered destructive
        severity:    'always' | 'protected' | None
        reason:      human-readable reason, or None if safe
        targets:     list of resolved target paths
    """
    if not args:
        return _safe_result(None)

    command = os.path.basename(args[0])  # handle /usr/bin/rm etc.
    rest = args[1:]

    if repo_root is None:
        repo_root = Path.cwd()
    else:
        repo_root = Path(repo_root)

    classifier = _CLASSIFIERS.get(command)
    if classifier is None:
        return _safe_result(command)

    return classifier(rest, repo_root)


def is_destructive_shell_cmd(
    args: list[str],
    repo_root: str | Path | None = None,
) -> bool:
    """Return True if the shell command is destructive."""
    return classify_shell_cmd(args, repo_root)["destructive"]
