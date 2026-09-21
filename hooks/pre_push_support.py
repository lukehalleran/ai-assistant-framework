#!/usr/bin/env python3
"""Small, dependency-free probes used by ``hooks/pre-push``.

The hook is commonly run from a clone while the real Daemon runs from a
different checkout.  This module resolves that checkout from an explicit
operator setting or a bounded chain of local ``origin`` URLs, then performs a
fail-closed process-table probe before the hook considers the non-unit lane.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


MAX_ROOTS = 16
LIVE_ROOT_ENV = "DAEMON_LIVE_REPO_ROOT"


class ProbeUnknown(RuntimeError):
    """The live checkout or process table could not be established safely."""


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ProbeUnknown(f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _checkout_root(path: Path) -> Path:
    try:
        raw = _git(path, "rev-parse", "--show-toplevel")
        root = Path(raw).resolve()
    except (OSError, ProbeUnknown, ValueError) as exc:
        raise ProbeUnknown("checkout root is not resolvable") from exc
    if not (root / "main.py").is_file():
        raise ProbeUnknown("checkout has no main.py")
    return root


def _local_origin_path(root: Path) -> Path | None:
    try:
        url = _git(root, "config", "--get", "remote.origin.url")
    except ProbeUnknown:
        return None
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            return None
        value = unquote(parsed.path)
        return Path(value).expanduser().resolve() if value else None
    if parsed.scheme:
        return None
    # A relative local remote is relative to the checkout that owns it.  SCP
    # style URLs contain a colon and are remote, not filesystem paths.
    if ":" in url and not url.startswith(("./", "../", "/")):
        return None
    return (root / url).expanduser().resolve()


def resolve_live_root(repo_root: str | os.PathLike[str]) -> Path:
    """Resolve the checkout whose process owns the live stores.

    ``DAEMON_LIVE_REPO_ROOT`` is authoritative but still validated.  Without
    it, only a local-origin chain is safe: a network origin gives no evidence
    about which checkout on this machine is live, so callers must block.
    """

    caller = _checkout_root(Path(repo_root).resolve())
    configured = os.environ.get(LIVE_ROOT_ENV, "").strip()
    if configured:
        return _checkout_root(Path(configured).expanduser().resolve())

    seen: set[Path] = set()
    current = caller
    local_roots: list[Path] = []
    for _ in range(MAX_ROOTS):
        if current in seen:
            break
        seen.add(current)
        local_roots.append(current)
        remote = _local_origin_path(current)
        if remote is None:
            break
        try:
            current = _checkout_root(remote)
        except ProbeUnknown:
            raise ProbeUnknown("origin does not resolve to a local checkout")
    else:
        raise ProbeUnknown("local origin chain exceeds probe bound")

    # A clone with a local origin normally has the live checkout as its final
    # distinct root.  A checkout with a network origin has no trustworthy
    # global live-root answer and must be configured explicitly.
    if len(local_roots) > 1:
        return local_roots[-1]
    raise ProbeUnknown(f"{LIVE_ROOT_ENV} is required for a standalone checkout")


def _process_ids() -> list[str]:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "main.py"],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ProbeUnknown("cannot inspect the process table") from exc
    if result.returncode not in (0, 1):
        raise ProbeUnknown("process-table probe failed")
    return [line for line in result.stdout.splitlines() if line.isdigit()]


def _preflight_process_table(pids: list[str]) -> None:
    for pid in pids:
        try:
            Path(f"/proc/{pid}/cwd").readlink()
            Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError as exc:
            raise ProbeUnknown("process-table entry became unreadable") from exc


def _own_checkout_root() -> Path:
    """The checkout that HOLDS this script, not the resolved live root.

    ``pre_push_support.py`` is invoked with ``env -u PYTHONPATH python -s``
    (D1), which drops both the ambient PYTHONPATH and any implicit cwd entry.
    When it is run as ``python hooks/pre_push_support.py`` (not ``-c``),
    ``sys.path[0]`` is the script's own directory (``.../hooks``), so a bare
    ``import utils`` finds nothing and raises ModuleNotFoundError. We always
    want THIS checkout's own ``utils.daemon_guard`` under test — never the
    live root's copy, which is a data argument (``repo_root=``), not an
    import source.
    """

    return Path(__file__).resolve().parent.parent


def daemon_state(repo_root: str | os.PathLike[str]) -> tuple[str, Path]:
    """Return ``(up|down|unknown, live_root)`` with no false-down result."""

    live_root = resolve_live_root(repo_root)
    pids = _process_ids()
    _preflight_process_table(pids)
    own_root = str(_own_checkout_root())
    if own_root not in sys.path:
        sys.path.insert(0, own_root)
    try:
        from utils.daemon_guard import daemon_running

        running = daemon_running(repo_root=live_root)
    except Exception as exc:  # guard probe must fail closed
        raise ProbeUnknown("daemon guard failed") from exc
    return ("up" if running else "down"), live_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("live-root", "daemon-state"))
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        if args.command == "live-root":
            print(resolve_live_root(args.repo_root))
        else:
            state, root = daemon_state(args.repo_root)
            print(f"{state} {root}")
    except ProbeUnknown as exc:
        print(f"unknown: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
