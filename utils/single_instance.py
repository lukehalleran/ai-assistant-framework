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
        from config.app_config import CHROMA_PATH
        parent = os.path.dirname(os.path.abspath(CHROMA_PATH))
        if parent:
            return parent
    except Exception as e:
        logger.debug(f"[SingleInstance] CHROMA_PATH unavailable ({e}); using ./data")
    return os.path.abspath("data")


def _try_lock(fh) -> bool:
    if sys.platform == "win32":
        import msvcrt
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    else:
        import fcntl
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False


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
