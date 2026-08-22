"""
Deduplicate the reference_docs collection (DRY-RUN FIRST).

Background (2026-08-02): ReferenceDocsManager read the raw
`chroma_store.collections` dict (None placeholder before first access), so at
startup the auto-seed saw "no stored hash" for the alphabetically-first doc,
re-uploaded it, and never deleted the previous copy — one duplicate upload
per startup. AGENTIC_SEARCH accumulated ~15K chunks (93% of the collection).
The access bug is fixed in reference_docs_manager._collection(); this script
removes the historical pile.

Selection: chunks are grouped per title by upload batch (each upload stamps
all its chunks with one `timestamp`). The NEWEST batch per title is kept;
every older batch is deleted. Chunks with no timestamp sort oldest.

Safety:
  - Default is DRY RUN: prints a per-title report. Nothing is modified.
  - --apply writes a pre-image backup JSONL of every chunk it is about to
    remove (data/backups/dedup_reference_docs_preimage_<ts>.jsonl), THEN
    deletes from Chroma.
  - Refuses to run while a live Daemon main.py is detected (--force to
    override a stale pgrep match).

Usage:
    python scripts/dedup_reference_docs.py            # dry run (report only)
    python scripts/dedup_reference_docs.py --apply    # backup + delete
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _daemon_running() -> bool:
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception:
        pass
    # Fallback (guard module unavailable): old cmdline heuristic. Known hole:
    # a relative-path launch has no repo name in its cmdline (2026-08-21).
    try:
        out = subprocess.run(
            ["pgrep", "-af", "main.py"], capture_output=True, text=True
        ).stdout
        return any("Daemon_v1" in line for line in out.splitlines())
    except Exception:
        return False


def _scan():
    from config.app_config import CHROMA_PATH
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    coll = store._get_collection("reference_docs")
    got = coll.get(include=["metadatas", "documents"])
    ids = got.get("ids") or []
    metas = got.get("metadatas") or []
    docs = got.get("documents") or []

    by_title = defaultdict(lambda: defaultdict(list))  # title -> batch_ts -> chunks
    for i, meta, doc in zip(ids, metas, docs):
        meta = meta or {}
        title = str(meta.get("title", "") or "(untitled)")
        batch = str(meta.get("timestamp", "") or "")  # one value per upload batch
        by_title[title][batch].append({"id": i, "metadata": meta, "content": doc})
    return store, coll, by_title


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete (after a pre-image backup). Default: dry run.")
    ap.add_argument("--force", action="store_true",
                    help="Run even if a live Daemon process is detected.")
    args = ap.parse_args()

    if _daemon_running() and not args.force:
        print("ABORT: a live Daemon main.py process is running — it holds ChromaDB "
              "open. Close it first (or --force if the match is stale).")
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    store, coll, by_title = _scan()

    doomed = []
    total_chunks = 0
    print(f"{'title':<32} {'batches':>7} {'chunks':>7} {'keep':>5} {'delete':>7}")
    for title in sorted(by_title):
        batches = by_title[title]
        n_chunks = sum(len(v) for v in batches.values())
        total_chunks += n_chunks
        # Empty-string timestamps sort first → treated as oldest.
        keep_batch = max(batches.keys())
        kill = [c for b, chunks in batches.items() if b != keep_batch for c in chunks]
        doomed.extend(kill)
        print(f"{title[:32]:<32} {len(batches):>7} {n_chunks:>7} "
              f"{len(batches[keep_batch]):>5} {len(kill):>7}")

    print(f"\nTotal: {total_chunks} chunks; {len(doomed)} duplicates to delete, "
          f"{total_chunks - len(doomed)} kept (newest batch per title).")

    if not doomed:
        print("Nothing to dedupe.")
        return 0

    if not args.apply:
        print("\nDRY RUN — nothing was deleted. Re-run with --apply to dedupe "
              "(a pre-image backup is written first).")
        return 0

    backup_dir = Path("data") / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    preimage = backup_dir / f"dedup_reference_docs_preimage_{ts}.jsonl"
    with open(preimage, "w") as f:
        for c in doomed:
            f.write(json.dumps(c, default=str) + "\n")
    print(f"Pre-image backup written to {preimage}")

    ids = [c["id"] for c in doomed]
    for i in range(0, len(ids), 500):
        coll.delete(ids=ids[i : i + 500])
    print(f"Deleted {len(ids)} duplicate chunks from reference_docs.")
    print("Restore path: the pre-image JSONL above contains every removed chunk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
