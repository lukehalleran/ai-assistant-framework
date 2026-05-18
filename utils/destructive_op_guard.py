"""Destructive git command classifier for agent session safety.

Purpose: Classify git argument lists as safe or destructive, and check
whether an explicit unlock is in effect. Used by safe_git.sh logic and
by tests.

Inputs:  Git argument list (after 'git'), optional env dict / repo root.
Outputs: Classification dict, boolean destructive check, unlock status.
Dependencies: Standard library only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# ============================================================================
# Destructive patterns
# ============================================================================

# Subcommands that are always destructive
_ALWAYS_DESTRUCTIVE: set[str] = {"restore", "clean", "push"}

# Subcommands with conditionally destructive flags
_CONDITIONAL_FLAGS: dict[str, set[str]] = {
    "reset": {"--hard", "--merge", "--keep"},
    "switch": {"-C"},
    "branch": {"-D"},
}


def classify_git_args(args: list[str]) -> dict[str, Any]:
    """Classify a git argument list (without the leading 'git').

    Returns a dict with:
        subcmd:      the git subcommand (e.g. 'restore', 'status')
        destructive: True if the command is considered destructive
        reason:      human-readable reason, or None if safe
    """
    if not args:
        return {"subcmd": None, "destructive": False, "reason": None}

    subcmd = args[0]

    # Always destructive
    if subcmd in _ALWAYS_DESTRUCTIVE:
        return {
            "subcmd": subcmd,
            "destructive": True,
            "reason": f"git {subcmd} is always destructive",
        }

    # Conditionally destructive (flag-dependent)
    if subcmd in _CONDITIONAL_FLAGS:
        dangerous_flags = _CONDITIONAL_FLAGS[subcmd]
        for arg in args[1:]:
            if arg in dangerous_flags:
                return {
                    "subcmd": subcmd,
                    "destructive": True,
                    "reason": f"git {subcmd} {arg} is destructive",
                }
        return {"subcmd": subcmd, "destructive": False, "reason": None}

    # checkout: block -- separator or ambiguous file targets
    if subcmd == "checkout":
        rest = args[1:]
        # -b / -B = branch creation, safe
        for arg in rest:
            if arg in ("-b", "-B"):
                return {"subcmd": subcmd, "destructive": False, "reason": None}
        # -- separator = file restore
        if "--" in rest:
            return {
                "subcmd": subcmd,
                "destructive": True,
                "reason": "git checkout -- <path> restores files",
            }
        return {"subcmd": subcmd, "destructive": False, "reason": None}

    # Everything else is safe
    return {"subcmd": subcmd, "destructive": False, "reason": None}


def is_destructive_git_args(args: list[str]) -> bool:
    """Return True if the git argument list is destructive."""
    return classify_git_args(args)["destructive"]


# ============================================================================
# Unlock check
# ============================================================================

_LOCKFILE_NAME = ".agent_allow_destructive_once"


def unlock_allowed(
    env: dict[str, str] | None = None,
    root: str | Path | None = None,
) -> bool:
    """Return True if destructive ops are explicitly unlocked.

    Checks:
      1. ALLOW_DESTRUCTIVE_OPS=1 in env
      2. .agent_allow_destructive_once file exists in root
    """
    if env is None:
        env = dict(os.environ)
    if env.get("ALLOW_DESTRUCTIVE_OPS") == "1":
        return True

    if root is not None:
        lockfile = Path(root) / _LOCKFILE_NAME
        if lockfile.exists():
            return True

    return False
