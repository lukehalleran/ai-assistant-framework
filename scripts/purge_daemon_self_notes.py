#!/usr/bin/env python3
"""Remove poisoned Daemon self-notes (DRY-RUN FIRST).

2026-09-11 (round 4, docs/BUG_CLASSES.md BC-75): a confabulated calendar-
state claim written into a daemon self-note gets read back the following
turn by [DAEMON SELF-NOTES] as apparent established fact — a contamination
loop that keeps re-asserting itself until the note is actually deleted.
This is the owner tool that closes it: it targets ALL THREE copies of a
note (the `daemon_self_notes` ChromaDB collection, the markdown files
under `daemon_notes/`, AND `daemon_notes/index.json`), since a note only
stops surfacing — and index.json only stops lying about a note that no
longer exists — once no copy remains.

Safety model (same shape as scripts/purge_adaptive_exemplars.py):
  * Default is DRY RUN — lists exactly which chroma docs, files, and
    index.json entries match and exits. Nothing is written or deleted.
  * --apply writes a pre-image JSONL backup of every matched chroma doc
    (content + metadata), every matched file's content, AND every matched
    index.json entry to
    data/backups/purge_daemon_self_notes_preimage_<ts>.jsonl FIRST.
  * Matched chroma docs are then deleted from the `daemon_self_notes`
    collection; matched FILES are MOVED into
    data/backups/daemon_self_notes_removed_<ts>/ — never unlinked; matched
    index.json ENTRIES are dropped and the file is rewritten atomically
    (utils.safe_json.atomic_write_json — temp file + os.replace, same
    directory). The pre-image JSONL is a second, independent copy of all
    three.
  * Refuses --apply while a live Daemon main.py is detected
    (utils.daemon_guard.daemon_running) — the running instance holds
    ChromaDB open and (for autonomous notes) tracks its own session-count
    state; a purge underneath it would not stick.

Selectors: --title-contains and/or --path (case-insensitive substrings).
At least one is required — an unscoped purge is refused (a typo must not
silently select everything).

Usage:
    python scripts/purge_daemon_self_notes.py --title-contains "TA sessions"
    python scripts/purge_daemon_self_notes.py --path ta-sessions-schedule
    python scripts/purge_daemon_self_notes.py --title-contains "TA sessions" --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _daemon_running() -> bool:
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception as exc:  # fail CLOSED: without the real guard we cannot prove the Daemon is down
        print(f"[daemon-guard] utils.daemon_guard unavailable ({exc!r}) — treating the Daemon as RUNNING; "
              f"--apply is refused. Run from the repo root with the project interpreter.", file=sys.stderr)
        return True


def _note_title_from_content(content: str) -> str:
    """The stored embed text is f"{title}\\n{summary}[...]" — first line."""
    return (content or "").split("\n", 1)[0].strip()


def match_notes(docs: list, title_contains: str | None, path_contains: str | None) -> list:
    """Chroma daemon_self_notes docs whose title (content's first line) and/or
    note_id (== the .md filename stem, per DaemonNotesManager.create_note)
    match the given case-insensitive substrings. Both filters, when given,
    must hit (AND); at least one filter must be given by the caller."""
    tc = title_contains.lower() if title_contains else None
    pc = path_contains.lower() if path_contains else None
    out = []
    for d in docs:
        title = _note_title_from_content(d.get("content") or "")
        note_id = (d.get("metadata") or {}).get("note_id", "") or ""
        title_hit = tc is None or tc in title.lower()
        path_hit = pc is None or pc in note_id.lower()
        if title_hit and path_hit:
            out.append(d)
    return out


def _extract_file_title(text: str, fallback: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("title:"):
            return stripped.split(":", 1)[1].strip().strip('"')
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def scan_files(notes_dir: Path) -> list:
    if not notes_dir.exists():
        return []
    return sorted(p for p in notes_dir.glob("*.md") if p.is_file())


def match_files(paths: list, title_contains: str | None, path_contains: str | None) -> list:
    tc = title_contains.lower() if title_contains else None
    pc = path_contains.lower() if path_contains else None
    out = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        title = _extract_file_title(text, p.stem)
        title_hit = tc is None or tc in title.lower()
        path_hit = pc is None or pc in str(p).lower()
        if title_hit and path_hit:
            out.append(p)
    return out


def scan_index_entries(notes_dir: Path) -> list:
    """Read daemon_notes/index.json — tolerant of a missing or corrupt file,
    mirroring DaemonNotesManager._update_index's own lenient read (this is a
    read-only report/removal tool, not the store's critical-load path; a
    corrupt index.json here just reports zero entries rather than aborting
    a dry run)."""
    index_path = notes_dir / "index.json"
    if not index_path.exists():
        return []
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def match_index_entries(entries: list, title_contains: str | None, path_contains: str | None) -> list:
    """index.json entries (2026-09-11, round 5, A17b) whose "title" and/or
    "path" field match the given case-insensitive substrings — same
    AND-when-both-given semantics as match_notes/match_files above."""
    tc = title_contains.lower() if title_contains else None
    pc = path_contains.lower() if path_contains else None
    out = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        title = str(e.get("title") or "")
        path = str(e.get("path") or "")
        title_hit = tc is None or tc in title.lower()
        path_hit = pc is None or pc in path.lower()
        if title_hit and path_hit:
            out.append(e)
    return out


def write_preimage(backup_dir: Path, chroma_hits: list, file_hits: list,
                    index_hits: list, ts: str) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    preimage = backup_dir / f"purge_daemon_self_notes_preimage_{ts}.jsonl"
    with open(preimage, "w", encoding="utf-8") as f:
        for d in chroma_hits:
            f.write(json.dumps({"store": "chroma", **d}, default=str) + "\n")
        for p in file_hits:
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                content = ""
            f.write(json.dumps({"store": "file", "path": str(p), "content": content}, default=str) + "\n")
        for e in index_hits:
            f.write(json.dumps({"store": "index", **e}, default=str) + "\n")
    return preimage


def apply_index_purge(notes_dir: Path, all_entries: list, index_hits: list) -> int:
    """Rewrite index.json with the matched entries dropped, keeping the
    rest byte-for-byte equivalent. Atomic write (utils.safe_json —
    same-directory temp file + os.replace); index.json is a plain list, so
    it fits atomic_write_json's contract directly. Returns the count
    removed; a no-op (0, file untouched) when nothing matched."""
    if not index_hits:
        return 0
    removed_ids = {e.get("id") for e in index_hits if e.get("id")}
    if removed_ids:
        kept = [e for e in all_entries if e.get("id") not in removed_ids]
    else:
        # No "id" field to key on (older/hand-edited entries) — fall back
        # to excluding by identity within the already-scanned list so a
        # coincidentally-identical OTHER entry is never dropped instead.
        hit_ids = {id(e) for e in index_hits}
        kept = [e for e in all_entries if id(e) not in hit_ids]
    from utils.safe_json import atomic_write_json
    atomic_write_json(str(notes_dir / "index.json"), kept)
    return len(all_entries) - len(kept)


def apply_purge(store, chroma_hits: list, file_hits: list, moved_dir: Path) -> tuple[int, int]:
    """Delete matched chroma docs by id; MOVE (never unlink) matched files
    into moved_dir. Returns (chroma_deleted, files_moved)."""
    deleted = 0
    ids = [d["id"] for d in chroma_hits if d.get("id")]
    if ids:
        coll = store._get_collection("daemon_self_notes")
        for i in range(0, len(ids), 200):
            coll.delete(ids=ids[i:i + 200])
        deleted = len(ids)

    moved = 0
    if file_hits:
        moved_dir.mkdir(parents=True, exist_ok=True)
        for p in file_hits:
            if not p.exists():
                continue
            dest = moved_dir / p.name
            if dest.exists():
                dest = moved_dir / f"{p.stem}.{int(time.time() * 1000)}{p.suffix}"
            shutil.move(str(p), str(dest))
            moved += 1
    return deleted, moved


def run(args, *, store, notes_dir: Path, backup_root: Path | None = None) -> int:
    """Selection + report + (optionally) apply. `store` needs .list_all(name)
    and ._get_collection(name).delete(ids=...); `notes_dir` is the daemon
    self-notes markdown directory. `backup_root` defaults to data/backups
    (tests should always pass a tmp_path)."""
    if not args.title_contains and not args.path:
        print("Give --title-contains and/or --path — an unscoped purge is refused.")
        return 1

    chroma_docs = store.list_all("daemon_self_notes")
    chroma_hits = match_notes(chroma_docs, args.title_contains, args.path)

    file_paths = scan_files(notes_dir)
    file_hits = match_files(file_paths, args.title_contains, args.path)

    index_entries = scan_index_entries(notes_dir)
    index_hits = match_index_entries(index_entries, args.title_contains, args.path)

    print(f"Scanned {len(chroma_docs)} daemon_self_notes chroma docs, "
          f"{len(file_paths)} files in {notes_dir}, "
          f"{len(index_entries)} index.json entries.")
    print(f"\nChroma matches: {len(chroma_hits)}")
    for d in chroma_hits:
        print(f"  - [{d.get('id')}] {_note_title_from_content(d.get('content') or '')!r}")
    print(f"\nFile matches: {len(file_hits)}")
    for p in file_hits:
        print(f"  - {p}")
    print(f"\nIndex.json matches: {len(index_hits)}")
    for e in index_hits:
        print(f"  - [{e.get('id')}] {e.get('title')!r} -> {e.get('path')}")

    if not chroma_hits and not file_hits and not index_hits:
        print("\nNothing matched — nothing to do.")
        return 0

    if not args.apply:
        print("\nDRY RUN — nothing was written, moved, or deleted. Re-run with "
              "--apply to back up + remove the matched entries.")
        return 0

    if _daemon_running():
        print("Refusing to run --apply: a live Daemon main.py is detected — it "
              "holds daemon_self_notes open in ChromaDB (and tracks its own "
              "session-note-count state). Shut Daemon down first.")
        return 1

    backup_root = Path(backup_root) if backup_root else Path("data") / "backups"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    preimage = write_preimage(backup_root, chroma_hits, file_hits, index_hits, ts)
    print(f"\nPre-image backup: {preimage}")

    moved_dir = backup_root / f"daemon_self_notes_removed_{ts}"
    deleted, moved = apply_purge(store, chroma_hits, file_hits, moved_dir)
    print(f"Deleted {deleted} chroma docs from daemon_self_notes.")
    print(f"Moved {moved} file(s) into {moved_dir} (never unlinked).")

    index_removed = apply_index_purge(notes_dir, index_entries, index_hits)
    print(f"Removed {index_removed} entr{'y' if index_removed == 1 else 'ies'} from index.json.")
    return 0


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--title-contains", default=None,
                     help="case-insensitive substring of the note title")
    ap.add_argument("--path", default=None,
                     help="case-insensitive substring of the note_id / file path")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    from config.app_config import CHROMA_PATH, DAEMON_NOTES_OUTPUT_DIR
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    notes_dir = Path(DAEMON_NOTES_OUTPUT_DIR)
    return run(args, store=store, notes_dir=notes_dir)


if __name__ == "__main__":
    sys.exit(main())
