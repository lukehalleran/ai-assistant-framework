"""
tests/unit/test_log_rotation_respects_instance_lock.py

Regression test for the 2026-09-19 "refused launch steals the running
Daemon's log" bug: main.py calls configure_logging() at IMPORT time (before
the single-instance lock is acquired at main.py:826), so a refused second
`python main.py` used to rename the LIVE instance's non-empty
daemon_debug.log to an archive name before anyone checked whether another
instance was actually running. The running process then kept appending to
the now-archived path while the "live" filename held only the refused
launcher's few startup lines, and utils/log_rotation.py later gzipped/
pruned the archive by name.

Fix: utils/single_instance.instance_lock_held_by_other() is a read-only
probe of the same instance lock; utils/logging_utils.configure_logging()
skips the rotate-by-rename step whenever another live process holds it.

All tests use tmp_path lock dirs and tmp log paths ONLY — never the repo's
real data/ dir or the repo-root daemon_debug.log (a real Daemon instance is
running on this machine while this test executes).
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from utils.logging_utils import configure_logging
from utils.single_instance import (
    acquire_single_instance_lock,
    instance_lock_held_by_other,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """Snapshot and restore the root logger so later tests are unaffected.

    configure_logging() clears root.handlers and re-adds console/file
    handlers pointed at whatever file_path it was given; without this a test
    here would leave the root logger holding an open handle into tmp_path
    after the test (and other tests' log lines would land there instead of
    wherever they expect).
    """
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    for h in root.handlers:
        try:
            h.close()
        except Exception:
            pass
    root.handlers = original_handlers
    root.setLevel(original_level)


def _clean_subprocess_env() -> dict:
    """Env for a spawned Python process: no PYTHONPATH (a clone's
    scripts/bin/usercustomize.py on PYTHONPATH would import the LIVE repo's
    utils package instead of this clone's — see project memory
    reference_pythonpath_usercustomize_shadowing.md).
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


# ---------------------------------------------------------------------------
# 1. No lock file at all -> not held.
# ---------------------------------------------------------------------------


def test_no_lock_file_returns_false(tmp_path):
    assert instance_lock_held_by_other(lock_dir=str(tmp_path)) is False


# ---------------------------------------------------------------------------
# 2. Held by a real, separate OS process; released once that process exits.
# ---------------------------------------------------------------------------


def test_held_by_subprocess_then_released_after_exit(tmp_path):
    lock_dir = tmp_path / "lockdir"
    lock_dir.mkdir()

    script = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from utils.single_instance import acquire_single_instance_lock\n"
        f"fh = acquire_single_instance_lock(lock_dir={str(lock_dir)!r})\n"
        "print('ACQUIRED', flush=True)\n"
        "time.sleep(30)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-s", "-c", script],
        cwd=str(REPO_ROOT),
        env=_clean_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = proc.stdout.readline()
        assert line.strip() == "ACQUIRED", (
            f"subprocess never signalled that it acquired the lock; "
            f"stdout={line!r} stderr={proc.stderr.read() if proc.stderr else ''}"
        )
        assert instance_lock_held_by_other(lock_dir=str(lock_dir)) is True
    finally:
        proc.kill()
        proc.wait(timeout=10)

    # The kernel releases flock automatically when the holder dies — even on
    # SIGKILL (see utils/single_instance.py module docstring) — so this
    # should clear promptly; poll briefly to absorb scheduling jitter.
    deadline = time.time() + 5
    held = True
    while time.time() < deadline:
        held = instance_lock_held_by_other(lock_dir=str(lock_dir))
        if not held:
            break
        time.sleep(0.05)
    assert held is False


# ---------------------------------------------------------------------------
# 3. configure_logging must NOT rotate an existing non-empty log while the
#    instance lock is held.
#
#    The lock is held here via a second file descriptor in THIS process
#    rather than a subprocess: instance_lock_held_by_other()'s own docstring
#    records that POSIX flock denies a second fd's lock attempt even from
#    the SAME process (two independent open() calls on one path are
#    independent open file descriptions), so this is a faithful stand-in for
#    "another live process holds it" and keeps the test fast/deterministic.
#    Test 2 above already proves the real-subprocess case end to end.
# ---------------------------------------------------------------------------


def test_configure_logging_skips_rotation_when_lock_held(tmp_path, monkeypatch):
    monkeypatch.delenv("DAEMON_TEST_MODE", raising=False)
    lock_dir = tmp_path / "lockdir"
    lock_dir.mkdir()
    log_path = tmp_path / "daemon_debug.log"
    log_path.write_text("pre-existing session content\n")

    import utils.single_instance as single_instance_module

    monkeypatch.setattr(single_instance_module, "_default_lock_dir", lambda: str(lock_dir))

    fh = acquire_single_instance_lock(lock_dir=str(lock_dir))
    try:
        configure_logging(file_path=str(log_path))
    finally:
        fh.close()

    siblings = sorted(p.name for p in tmp_path.iterdir())
    rotated = [n for n in siblings if n.startswith("daemon_debug_") and n != log_path.name]
    assert rotated == [], f"log was rotated while the instance lock was held: {siblings}"
    assert log_path.exists()
    assert log_path.read_text().startswith("pre-existing session content"), (
        "the live log's original content must be preserved (append mode, no rename)"
    )


# ---------------------------------------------------------------------------
# 4. Today's behaviour is preserved when nobody holds the lock: rotate.
# ---------------------------------------------------------------------------


def test_configure_logging_rotates_when_no_holder(tmp_path, monkeypatch):
    monkeypatch.delenv("DAEMON_TEST_MODE", raising=False)
    lock_dir = tmp_path / "lockdir"
    lock_dir.mkdir()
    log_path = tmp_path / "daemon_debug.log"
    log_path.write_text("pre-existing session content\n")

    import utils.single_instance as single_instance_module

    monkeypatch.setattr(single_instance_module, "_default_lock_dir", lambda: str(lock_dir))

    configure_logging(file_path=str(log_path))

    rotated = [
        p for p in tmp_path.iterdir()
        if p.name.startswith("daemon_debug_") and p.name != log_path.name
    ]
    assert len(rotated) == 1, f"expected exactly one rotated sibling, found {rotated}"
    assert rotated[0].read_text() == "pre-existing session content\n"
    assert log_path.exists()
