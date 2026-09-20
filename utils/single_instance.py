"""
# utils/single_instance.py

Module Contract
- Purpose: Prevent two Daemon instances from running against the same data
  directory at once (two orchestrators sharing ChromaDB/corpus caused the
  duplicate-threads incident). OS-level advisory file lock — atomic, and
  released automatically by the kernel when the process dies, even on SIGKILL,
  so a stale lock file can never block a fresh launch.
- Inputs: lock_dir (defaults to the ChromaDB parent directory, so the lock
  guards the same data the instance would open; frozen builds inherit their
  relocated data dir automatically).
- Outputs: acquire_single_instance_lock() returns the open lock file handle —
  the CALLER MUST KEEP A REFERENCE for the process lifetime (GC closing the
  handle releases the lock). Raises SingleInstanceError (with the holder's PID)
  if another instance holds the lock.
- Platform: fcntl.flock on POSIX; msvcrt.locking fallback on Windows.
"""

import os
import sys

from utils.logging_utils import get_logger

logger = get_logger("single_instance")

LOCK_FILENAME = "daemon.lock"


class SingleInstanceError(RuntimeError):
    """Another Daemon instance already holds the data-directory lock."""


def _default_lock_dir() -> str:
    try:
        from config.app_config import CHROMA_PATH  # lazy import: live-config
        parent = os.path.dirname(os.path.abspath(CHROMA_PATH))
        if parent:
            return parent
    except Exception as e:
        logger.debug(f"[SingleInstance] CHROMA_PATH unavailable ({e}); using ./data")
    return os.path.abspath("data")


def _try_lock(fh) -> bool:
    if sys.platform == "win32":
        import msvcrt  # lazy import: platform
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    else:
        import fcntl  # lazy import: platform
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False


def instance_lock_held_by_other(lock_dir: str | None = None) -> bool:
    """Return True iff another live process currently holds the instance lock.

    This is a read-only probe: it never writes the PID and never truncates
    the lock file (unlike acquire_single_instance_lock()). It opens the same
    lock path, tries a non-blocking exclusive lock, and immediately releases
    it if acquired — so calling this never itself holds the lock afterward.

    Used by log rotation (utils/logging_utils.configure_logging) to avoid
    stealing a running instance's log: a refused second `python main.py`
    used to rename the LIVE instance's daemon_debug.log to an archive name
    before the single-instance lock was even checked (main.py acquires the
    lock well after configure_logging() runs at import time), so the running
    process kept appending to the dead archive while the "live" path held
    only the refused launcher's few startup lines (2026-09-19).

    NOTE: POSIX flock is associated with the open file description, not the
    process — a second independent open()+flock() call from the SAME
    process that already holds the lock will also fail to acquire here and
    this function will report True for "held by other" even though it is
    the caller's own lock. That is acceptable: the only caller today is log
    rotation, and a process that already holds the lock has no reason to
    rotate its own log out from under itself.
    """
    path = os.path.join(lock_dir or _default_lock_dir(), LOCK_FILENAME)
    if not os.path.exists(path):
        return False
    try:
        fh = open(path, "a+", encoding="utf-8")
    except OSError:
        return False
    try:
        if _try_lock(fh):
            try:
                if sys.platform == "win32":
                    import msvcrt  # lazy import: platform
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl  # lazy import: platform
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            return False
        return True
    except OSError:
        return False
    finally:
        try:
            fh.close()
        except OSError:
            pass


def acquire_single_instance_lock(lock_dir: str | None = None):
    """Acquire the exclusive instance lock, or raise SingleInstanceError.

    Returns the open file handle; keep it referenced for the process lifetime.
    """
    lock_dir = lock_dir or _default_lock_dir()
    os.makedirs(lock_dir, exist_ok=True)
    path = os.path.join(lock_dir, LOCK_FILENAME)

    fh = open(path, "a+", encoding="utf-8")
    if not _try_lock(fh):
        try:
            fh.seek(0)
            holder = fh.read().strip() or "unknown"
        except OSError:
            holder = "unknown"
        fh.close()
        raise SingleInstanceError(
            f"Another Daemon instance is already running (PID {holder}). "
            f"Lock: {path}. If that instance is a zombie, find it with "
            f"`pgrep -af 'python.*main.py'` and stop it first."
        )

    # Record our PID for the error message a second launch will show.
    try:
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
    except OSError as e:
        logger.debug(f"[SingleInstance] Could not write PID to lock file: {e}")

    logger.info(f"[SingleInstance] Acquired instance lock: {path} (pid {os.getpid()})")
    return fh
