#!/usr/bin/env python3
"""
Restore memory-store backups written by utils/backup_manager.py.

Usage:
    python scripts/restore_backup.py --list
    python scripts/restore_backup.py --backup-now [--with-chroma | --no-chroma]
    python scripts/restore_backup.py --restore 20260714_183000            # DRY RUN
    python scripts/restore_backup.py --restore 20260714_183000 --apply

Safety:
    - Refuses to restore while Daemon is running (single-instance lock).
    - Dry-run by default; --apply performs the restore.
    - Never deletes current data: every file/dir being replaced is moved
      aside to <path>.pre-restore-<timestamp> first.
"""

import argparse
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.backup_manager import (  # noqa: E402
    MANIFEST_NAME, _list_backups, backup_targets, chroma_path, run_backup,
)


def _backup_dir() -> str:
    from config.app_config import BACKUP_DIR
    return BACKUP_DIR


def cmd_list() -> int:
    backups = _list_backups(_backup_dir())
    if not backups:
        print(f"No backups found in {_backup_dir()}")
        return 0
    print(f"Backups in {_backup_dir()} (newest first):")
    for b in backups:
        m = b["manifest"]
        chroma = " +chroma" if m.get("includes_chroma") else ""
        print(f"  {b['name']}  reason={m.get('reason', '?'):9s}"
              f" files={len(m.get('files', []))}{chroma}")
    return 0


def cmd_backup_now(with_chroma) -> int:
    result = run_backup(reason="manual", include_chroma=with_chroma)
    if result.ok and result.path:
        print(f"Backup written: {result.path}"
              f"{' (+chroma)' if result.chroma_included else ''}")
        return 0
    print(f"Backup failed: {result.error or result.skipped_reason}")
    return 1


def _require_daemon_stopped() -> None:
    from utils.single_instance import SingleInstanceError, acquire_single_instance_lock
    try:
        lock = acquire_single_instance_lock()
    except SingleInstanceError as e:
        print(f"REFUSED: Daemon appears to be running — stop it first.\n  {e}")
        sys.exit(1)
    # Hold the lock for the life of this process so Daemon can't start mid-restore.
    globals()["_restore_lock"] = lock


def cmd_restore(name: str, apply: bool) -> int:
    backups = {b["name"]: b for b in _list_backups(_backup_dir())}
    if name not in backups:
        print(f"Backup '{name}' not found. Use --list.")
        return 1
    src_dir = backups[name]["path"]
    manifest = backups[name]["manifest"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Map backup files → live destinations by basename.
    dest_by_name = {os.path.basename(p): p for p in backup_targets()}
    # Targets that don't exist live yet still restore to their configured path.
    from config.app_config import (
        CORPUS_FILE, KNOWLEDGE_GRAPH_ALIASES_PATH, KNOWLEDGE_GRAPH_PERSIST_PATH,
        PROACTIVE_SURFACING_HISTORY_PATH, STALENESS_INDEX_PATH,
    )
    from memory.user_profile import UserProfile
    for p in (KNOWLEDGE_GRAPH_PERSIST_PATH, KNOWLEDGE_GRAPH_ALIASES_PATH,
              UserProfile.DEFAULT_PATH, CORPUS_FILE, STALENESS_INDEX_PATH,
              PROACTIVE_SURFACING_HISTORY_PATH):
        if p:
            dest_by_name.setdefault(os.path.basename(p), p)

    plan = []
    for fname in manifest.get("files", []):
        src = os.path.join(src_dir, fname)
        dest = dest_by_name.get(fname)
        if dest and os.path.isfile(src):
            plan.append(("file", src, dest))
    chroma_base = os.path.basename(chroma_path().rstrip("/"))
    if manifest.get("includes_chroma") and os.path.isdir(os.path.join(src_dir, chroma_base)):
        plan.append(("tree", os.path.join(src_dir, chroma_base), chroma_path()))

    if not plan:
        print("Nothing restorable in this backup.")
        return 1

    print(f"{'RESTORING' if apply else 'DRY RUN — would restore'} from {src_dir}:")
    for kind, src, dest in plan:
        aside = f"{dest}.pre-restore-{ts}"
        print(f"  {kind:4s} {src}\n       → {dest}   (current moved to {aside})")

    if not apply:
        print("\nRe-run with --apply to perform the restore.")
        return 0

    _require_daemon_stopped()
    for kind, src, dest in plan:
        aside = f"{dest}.pre-restore-{ts}"
        if os.path.exists(dest):
            os.rename(dest, aside)
        if kind == "file":
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            shutil.copy2(src, dest)
        else:
            shutil.copytree(src, dest)
    print("\nRestore complete. Previous data preserved at the *.pre-restore-* paths.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List backups")
    parser.add_argument("--backup-now", action="store_true", help="Run a backup now")
    parser.add_argument("--with-chroma", dest="with_chroma", action="store_true",
                        default=None, help="Force-include the ChromaDB tree")
    parser.add_argument("--no-chroma", dest="with_chroma", action="store_false",
                        help="Exclude the ChromaDB tree")
    parser.add_argument("--restore", metavar="NAME", help="Restore a backup (dry-run)")
    parser.add_argument("--apply", action="store_true", help="Actually perform --restore")
    args = parser.parse_args()

    if args.list:
        return cmd_list()
    if args.backup_now:
        return cmd_backup_now(args.with_chroma)
    if args.restore:
        return cmd_restore(args.restore, apply=args.apply)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
