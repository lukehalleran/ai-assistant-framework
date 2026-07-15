"""
Module Contract — utils/log_rotation.py

Purpose:
    Bound on-disk log growth (production-grade audit 2026-07-14: turn_records/
    actions_audit/daily_notes grew unbounded; 798 daemon_debug archives = 733MB).
    Runs once at startup (main.py gui/cli, after preflight) — never touches a
    file a live handler is writing.

Strategies by file:
    - turn_records.jsonl, daily_notes.log → numbered rotation when over the
      size cap (file → file.1 → file.2 …, oldest beyond `keep` dropped).
    - actions_audit.jsonl → timestamped archive rename when over the cap
      (audit history is never deleted, only split).
    - daemon_debug_<ts>.log archives → gzip when older than compress_age_days,
      delete (.log/.log.gz) when older than keep_days. The live
      daemon_debug.log (no timestamp) is never touched.

Inputs:  config LOG_MAINTENANCE_* constants (YAML section log_maintenance).
Outputs: rotated/compressed/pruned files; returns a summary dict for logging.
Side effects: renames, gzip-compression, deletion of expired debug archives.
"""

import glob
import gzip
import os
import re
import shutil
import time
from datetime import datetime
from typing import Dict

from utils.logging_utils import get_logger

logger = get_logger("log_rotation")

# Timestamped archives only — the live daemon_debug.log never matches.
_DEBUG_ARCHIVE_RE = re.compile(r"daemon_debug_\d{8}_\d{6}\.log(\.gz)?$")


def rotate_if_large(path: str, max_bytes: int, keep: int = 3) -> bool:
    """Numbered rotation: path → path.1 → … → path.keep (oldest dropped)."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) <= max_bytes:
            return False
        oldest = f"{path}.{keep}"
        if os.path.exists(oldest):
            os.remove(oldest)
        for i in range(keep - 1, 0, -1):
            src = f"{path}.{i}"
            if os.path.exists(src):
                os.rename(src, f"{path}.{i + 1}")
        os.rename(path, f"{path}.1")
        logger.info(f"[LogRotation] Rotated {path} (> {max_bytes} bytes)")
        return True
    except OSError as e:
        logger.warning(f"[LogRotation] Rotation failed for {path}: {e}")
        return False


def archive_if_large(path: str, max_bytes: int) -> bool:
    """Timestamped archive rename — history preserved, never deleted."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) <= max_bytes:
            return False
        base, ext = os.path.splitext(path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.rename(path, f"{base}-{ts}{ext}")
        logger.info(f"[LogRotation] Archived {path} → {base}-{ts}{ext}")
        return True
    except OSError as e:
        logger.warning(f"[LogRotation] Archive failed for {path}: {e}")
        return False


def _debug_archives(directory: str):
    for p in glob.glob(os.path.join(directory, "daemon_debug_*.log*")):
        if _DEBUG_ARCHIVE_RE.search(os.path.basename(p)):
            yield p


def maintain_debug_archives(directory: str, compress_age_days: float,
                            keep_days: float) -> Dict[str, int]:
    """Gzip old daemon_debug_<ts>.log archives; delete expired ones."""
    now = time.time()
    compressed = pruned = 0
    for path in list(_debug_archives(directory)):
        try:
            age_days = (now - os.path.getmtime(path)) / 86400
            if age_days > keep_days:
                os.remove(path)
                pruned += 1
            elif age_days > compress_age_days and not path.endswith(".gz"):
                with open(path, "rb") as f_in, gzip.open(path + ".gz", "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                # Preserve the archive's original mtime so keep_days still
                # counts from when the log was written, not when compressed.
                stat = os.stat(path)
                os.utime(path + ".gz", (stat.st_atime, stat.st_mtime))
                os.remove(path)
                compressed += 1
        except OSError as e:
            logger.warning(f"[LogRotation] Debug-archive maintenance failed for {path}: {e}")
    return {"compressed": compressed, "pruned": pruned}


def run_startup_log_maintenance() -> Dict[str, int]:
    """One startup pass over all managed logs. Config-driven; never raises."""
    from config.app_config import (
        LOG_MAINTENANCE_AUDIT_MAX_MB,
        LOG_MAINTENANCE_DAILY_NOTES_MAX_MB,
        LOG_MAINTENANCE_DEBUG_COMPRESS_AGE_DAYS,
        LOG_MAINTENANCE_DEBUG_KEEP_DAYS,
        LOG_MAINTENANCE_ENABLED,
        LOG_MAINTENANCE_TURN_RECORDS_MAX_MB,
        TURN_TELEMETRY_PATH,
    )
    summary: Dict[str, int] = {"rotated": 0, "archived": 0, "compressed": 0, "pruned": 0}
    if not LOG_MAINTENANCE_ENABLED:
        return summary
    try:
        mb = 1024 * 1024
        if rotate_if_large(TURN_TELEMETRY_PATH,
                           int(LOG_MAINTENANCE_TURN_RECORDS_MAX_MB * mb)):
            summary["rotated"] += 1
        if rotate_if_large(os.path.join("logs", "daily_notes.log"),
                           int(LOG_MAINTENANCE_DAILY_NOTES_MAX_MB * mb)):
            summary["rotated"] += 1
        if archive_if_large(os.path.join("logs", "actions_audit.jsonl"),
                            int(LOG_MAINTENANCE_AUDIT_MAX_MB * mb)):
            summary["archived"] += 1
        dbg = maintain_debug_archives(
            ".",
            compress_age_days=LOG_MAINTENANCE_DEBUG_COMPRESS_AGE_DAYS,
            keep_days=LOG_MAINTENANCE_DEBUG_KEEP_DAYS,
        )
        summary["compressed"] = dbg["compressed"]
        summary["pruned"] = dbg["pruned"]
        if any(summary.values()):
            logger.info(f"[LogRotation] Startup maintenance: {summary}")
    except Exception as e:  # maintenance must never block startup
        logger.error(f"[LogRotation] Startup maintenance failed: {e}")
    return summary
