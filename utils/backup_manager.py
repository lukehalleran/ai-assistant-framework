"""
Module Contract — utils/backup_manager.py

Purpose:
    Automated local backups of the user's memory data — the JSON stores
    (knowledge graph, entity aliases, user profile, corpus, claim index,
    surfacing history) plus the ChromaDB directory. Runs as the final
    shutdown phase (main._do_shutdown_async) and on demand
    (scripts/restore_backup.py --backup-now, scripts/export_user_data.py).

Layout:
    <BACKUP_DIR>/<YYYYMMDD_HHMMSS>/
        manifest.json          — {ts, reason, includes_chroma, files}
        <store files>          — flat copies, original filenames
        chroma_db_v4/          — full tree copy (only when included)

Behavior:
    - JSON stores are copied on every backup (a few MB — cheap).
    - The ChromaDB tree (~600MB) is included only when the newest
      chroma-including backup is older than BACKUP_MIN_INTERVAL_HOURS.
    - *.sqlite3 files are copied via the sqlite3 backup API (consistent
      even if a connection is still open); everything else via copy2.
    - Retention: newest BACKUP_RETENTION backups are kept; older ones are
      pruned — but the newest chroma-including backup is always kept.
      Pruning only ever deletes directories under BACKUP_DIR that contain
      a manifest.json we wrote (never arbitrary paths).
    - The Google OAuth token is deliberately NOT backed up (bearer
      secrets; losing it costs one re-auth).

Inputs:  config BACKUP_* constants; store paths from app_config/UserProfile.
Outputs: BackupResult; backup directories on disk.
Side effects: file copies; prunes old backup dirs (see above).
"""

import json
import os
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

from utils.logging_utils import get_logger

logger = get_logger("backup_manager")

MANIFEST_NAME = "manifest.json"


@dataclass
class BackupResult:
    ok: bool
    path: Optional[str] = None
    files_copied: List[str] = field(default_factory=list)
    chroma_included: bool = False
    skipped_reason: Optional[str] = None
    error: Optional[str] = None


def _config():
    from config.app_config import (
        BACKUP_DIR, BACKUP_ENABLED, BACKUP_INCLUDE_CHROMA,
        BACKUP_MIN_INTERVAL_HOURS, BACKUP_RETENTION,
    )
    return {
        "enabled": BACKUP_ENABLED,
        "dir": BACKUP_DIR,
        "retention": BACKUP_RETENTION,
        "min_interval_hours": BACKUP_MIN_INTERVAL_HOURS,
        "include_chroma": BACKUP_INCLUDE_CHROMA,
    }


def backup_targets() -> List[str]:
    """The JSON stores worth backing up. Only existing files are returned."""
    from config.app_config import (
        CORPUS_FILE, KNOWLEDGE_GRAPH_ALIASES_PATH, KNOWLEDGE_GRAPH_PERSIST_PATH,
        PROACTIVE_SURFACING_HISTORY_PATH, STALENESS_INDEX_PATH,
    )
    from memory.user_profile import UserProfile
    from utils.adaptive_exemplars import _STORE_PATH as adaptive_exemplars_path
    from memory.learned_relations import _STORE_PATH as learned_relations_path

    # Narrative staleness flag path — resolved at call time for test sandboxing
    narrative_stale_path = os.getenv("NARRATIVE_STALE_FLAG_PATH", os.path.join("data", "narrative_stale.json"))

    candidates = [
        KNOWLEDGE_GRAPH_PERSIST_PATH,
        KNOWLEDGE_GRAPH_ALIASES_PATH,
        UserProfile.DEFAULT_PATH,
        CORPUS_FILE,
        STALENESS_INDEX_PATH,
        PROACTIVE_SURFACING_HISTORY_PATH,
        # Post-07-14 stores (2026-07-15+)
        adaptive_exemplars_path,
        learned_relations_path,
        os.path.join("data", "tone_state.json"),  # context_pipeline._TONE_STATE_PATH
        os.path.join("data", "pending_actions.json"),  # core.actions.types.PendingActionsStore
        os.path.join("data", "curation_queue.json"),  # memory.curation.engine._DEFAULT_QUEUE_PATH
        narrative_stale_path,  # utils.narrative_staleness._DEFAULT_FLAG_PATH
    ]
    return [p for p in candidates if p and os.path.isfile(p)]


def chroma_path() -> str:
    from config.app_config import CHROMA_PATH
    return CHROMA_PATH


def _copy_sqlite(src: str, dest: str) -> None:
    """Consistent sqlite copy via the backup API (safe with open readers)."""
    src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        dest_conn = sqlite3.connect(dest)
        try:
            src_conn.backup(dest_conn)
        finally:
            dest_conn.close()
    finally:
        src_conn.close()


def _copy_chroma_tree(src_dir: str, dest_dir: str) -> None:
    """Copy the ChromaDB tree; sqlite files go through the backup API."""
    for root, _dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        target_root = os.path.join(dest_dir, rel) if rel != "." else dest_dir
        os.makedirs(target_root, exist_ok=True)
        for name in files:
            src = os.path.join(root, name)
            dest = os.path.join(target_root, name)
            if name.endswith(".sqlite3"):
                _copy_sqlite(src, dest)
            else:
                shutil.copy2(src, dest)


def _list_backups(backup_dir: str) -> List[dict]:
    """Existing backups (dirs containing our manifest), newest first."""
    if not os.path.isdir(backup_dir):
        return []
    found = []
    for name in sorted(os.listdir(backup_dir), reverse=True):
        path = os.path.join(backup_dir, name)
        manifest_path = os.path.join(path, MANIFEST_NAME)
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            manifest = {}
        found.append({"name": name, "path": path, "manifest": manifest})
    return found


def _chroma_due(backups: List[dict], min_interval_hours: float) -> bool:
    """True when no sufficiently recent chroma-including backup exists."""
    cutoff = datetime.now() - timedelta(hours=min_interval_hours)
    for b in backups:
        if not b["manifest"].get("includes_chroma"):
            continue
        try:
            ts = datetime.fromisoformat(b["manifest"]["ts"])
        except (KeyError, ValueError):
            continue
        if ts >= cutoff:
            return False
    return True


def _prune(backup_dir: str, retention: int) -> List[str]:
    """Keep the newest `retention` backups + the newest chroma backup."""
    backups = _list_backups(backup_dir)  # newest first
    keep = {b["path"] for b in backups[:retention]}
    for b in backups:
        if b["manifest"].get("includes_chroma"):
            keep.add(b["path"])  # newest chroma backup (list is newest-first)
            break
    removed = []
    for b in backups:
        if b["path"] in keep:
            continue
        # Safety: only ever remove a dir under backup_dir that holds our manifest.
        if os.path.isfile(os.path.join(b["path"], MANIFEST_NAME)):
            try:
                shutil.rmtree(b["path"])
                removed.append(b["path"])
            except OSError as e:
                logger.warning(f"[Backup] Could not prune {b['path']}: {e}")
    return removed


def run_backup(reason: str = "manual",
               include_chroma: Optional[bool] = None) -> BackupResult:
    """Create one backup. include_chroma=None → decide from config + interval."""
    cfg = _config()
    if not cfg["enabled"]:
        return BackupResult(ok=True, skipped_reason="disabled")

    backup_dir = cfg["dir"]
    backups = _list_backups(backup_dir)
    if include_chroma is None:
        include_chroma = (
            cfg["include_chroma"]
            and os.path.isdir(chroma_path())
            and _chroma_due(backups, cfg["min_interval_hours"])
        )

    ts = datetime.now()
    dest = os.path.join(backup_dir, ts.strftime("%Y%m%d_%H%M%S"))
    try:
        os.makedirs(dest, exist_ok=False)
        os.chmod(backup_dir, 0o700)

        copied = []
        for src in backup_targets():
            shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
            copied.append(os.path.basename(src))

        if include_chroma:
            _copy_chroma_tree(
                chroma_path(),
                os.path.join(dest, os.path.basename(chroma_path().rstrip("/"))),
            )

        manifest = {
            "ts": ts.isoformat(),
            "reason": reason,
            "includes_chroma": bool(include_chroma),
            "files": copied,
        }
        with open(os.path.join(dest, MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        pruned = _prune(backup_dir, cfg["retention"])
        if pruned:
            logger.info(f"[Backup] Pruned {len(pruned)} old backup(s)")
        logger.info(
            f"[Backup] Wrote {dest} ({len(copied)} stores"
            f"{', + chroma' if include_chroma else ''})"
        )
        return BackupResult(ok=True, path=dest, files_copied=copied,
                            chroma_included=bool(include_chroma))
    except Exception as e:
        logger.error(f"[Backup] Failed: {e}")
        return BackupResult(ok=False, error=str(e), path=dest)


def run_shutdown_backup() -> BackupResult:
    """Shutdown entry point — config-driven chroma inclusion."""
    return run_backup(reason="shutdown", include_chroma=None)
