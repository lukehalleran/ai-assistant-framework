"""Detect a live Daemon main.py process.

Module Contract:
  Purpose: single source of truth for the "is Daemon running?" check that
           store-writing scripts use to refuse --apply. The live instance
           holds JSON stores (profile, adaptive exemplars, graph, …) in
           memory and re-saves them on its next write, silently clobbering
           anything a script changed on disk (2026-08-05 profile-clobber;
           2026-08-21 the exemplar purge ran against a live store).
  Inputs:  daemon_running(repo_root=None) -> bool
  Side effects: none (reads /proc).

Why not grep the command line for "Daemon_v1"? That was the old per-script
check, and a relative-path launch defeats it: `python main.py` from inside
the repo has no repo name anywhere in its cmdline, so the guard silently
passed and an --apply ran against a live store (2026-08-21). This check
resolves each main.py candidate's /proc/<pid>/cwd against the repo root
instead — launch style can't hide the working directory.
"""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _looks_like_daemon(args: list) -> bool:
    """argv shape check: a python interpreter running main.py (or a frozen
    main.py executable) — excludes editors/tools that merely mention it."""
    if not args:
        return False
    prog = os.path.basename(args[0])
    if prog.endswith("main.py"):
        return True
    if "python" in prog:
        return any(a.endswith("main.py") for a in args[1:])
    return False


def daemon_running(repo_root=None) -> bool:
    """True if a main.py process is running WITH this repo as its cwd."""
    root = Path(repo_root).resolve() if repo_root else REPO_ROOT
    try:
        out = subprocess.run(
            ["pgrep", "-f", "main.py"], capture_output=True, text=True
        ).stdout
    except Exception:
        return False
    for pid in out.split():
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            cwd = Path(os.readlink(f"/proc/{pid}/cwd"))
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except Exception:
            continue  # process exited or unreadable
        if cwd != root:
            continue
        args = [a for a in raw.decode(errors="replace").split("\x00") if a]
        if _looks_like_daemon(args):
            return True
    return False
