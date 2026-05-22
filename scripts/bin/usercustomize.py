"""Auto-activate Python filesystem guard in subprocesses.

This file is picked up by Python's site module when scripts/bin/ is on
PYTHONPATH (set by activate_guards.sh).  It activates the Python filesystem
guard so that child interpreters spawned from a guarded shell session
inherit protection.

Skipped during pytest/coverage runs to avoid interference.
Set DISABLE_FS_GUARD=1 to skip explicitly.
"""

import os
import sys


def _should_skip() -> bool:
    markers = (
        "PYTEST_CURRENT_TEST",
        "PYTEST_ADDOPTS",
        "COVERAGE_PROCESS_START",
        "PYTEST_XDIST_WORKER",
    )
    return (
        any(os.getenv(k) for k in markers)
        or os.getenv("DISABLE_FS_GUARD") == "1"
    )


if not _should_skip():
    try:
        # Resolve repo root: scripts/bin/usercustomize.py -> scripts/bin -> scripts -> repo root
        _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        from utils.python_fs_guard import activate, is_active
        if not is_active():
            activate(repo_root=_repo_root)
    except Exception:
        # Guard activation is best-effort in subprocesses — don't crash the child
        pass
