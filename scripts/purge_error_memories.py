"""
Purge historical junk conversation memories (DRY-RUN FIRST).

Targets docs stored BEFORE the 2026-07-03 storage-time guards and seen
competing for top-10 retrieval slots (2026-07-15):
  - API-error sentinel turns ("[API unavailable] ...", "[API Error] ...",
    every prefix in models.model_manager.API_ERROR_PREFIXES) persisted as
    assistant replies (Feb–March 2026 era)
  - Bare connectivity-test exchanges (query literally "test"/"testing")

Selection uses THE deployed retrieval predicate
(memory.utils.is_junk_conversation_doc) — the same function that now hides
these docs at retrieval time — never a re-derivation.

Scans the stores:
  - ChromaDB `conversations` collection (CHROMA_PATH)
  - ChromaDB `summaries` collection [2026-08-03: the cross-dedup queue review
    found ~a dozen "[API Error] 402" docs stored as summaries — long and
    letter-rich, they evaded the length/letter-based junk guard; selection
    uses the deployed memory.utils.is_junk_summary, which now rejects
    API-error sentinels]
  - the corpus JSON (CORPUS_FILE)

Safety:
  - Default is DRY RUN: prints a report and writes the full candidate list
    to data/purge_candidates_<ts>.jsonl. Nothing is modified.
  - --apply writes a pre-image backup JSONL of every doc/entry it is about
    to remove (data/backups/purge_error_memories_preimage_<ts>.jsonl), THEN
    deletes from Chroma and rewrites the corpus via its atomic save path.
  - Refuses to run while a live Daemon main.py is detected (--force to
    override, e.g. under a stale pgrep match).

Usage:
    python scripts/purge_error_memories.py            # dry run (report only)
    python scripts/purge_error_memories.py --apply    # backup + delete
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.utils import is_junk_conversation_doc, is_junk_summary  # THE deployed predicates


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


def _scan_chroma():
    from config.app_config import CHROMA_PATH
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    docs = store.list_all("conversations")
    hits = [d for d in docs if is_junk_conversation_doc(content=d.get("content", ""))]
    return store, docs, hits


def _scan_summaries(store):
    docs = store.list_all("summaries")
    hits = [d for d in docs if is_junk_summary(d.get("content", ""))]
    return docs, hits


def _scan_corpus():
    from memory.corpus_manager import CorpusManager

    cm = CorpusManager()
    entries = cm.corpus
    hits = [
        e for e in entries
        if isinstance(e, dict)
        and is_junk_conversation_doc(
            query=e.get("query", ""), response=e.get("response", "")
        )
    ]
    return cm, entries, hits


def _preview(tag: str, items, key):
    print(f"\n{tag}: {len(items)} candidate(s)")
    for it in items[:8]:
        txt = " ".join((key(it) or "").split())[:110]
        ts = (it.get("metadata") or {}).get("timestamp") or it.get("timestamp") or "?"
        print(f"  - [{ts}] {txt!r}")
    if len(items) > 8:
        print(f"  ... and {len(items) - 8} more (full list in the candidates file)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete (after writing a pre-image backup). Default: dry run.")
    ap.add_argument("--force", action="store_true",
                    help="Run even if a live Daemon process is detected.")
    args = ap.parse_args()

    if _daemon_running() and not args.force:
        print("ABORT: a live Daemon main.py process is running — it holds ChromaDB "
              "and the corpus open. Close it first (or --force if the match is stale).")
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    store, chroma_docs, chroma_hits = _scan_chroma()
    summary_docs, summary_hits = _scan_summaries(store)
    cm, corpus_entries, corpus_hits = _scan_corpus()

    print(f"Scanned {len(chroma_docs)} chroma conversation docs, "
          f"{len(summary_docs)} chroma summaries, "
          f"{len(corpus_entries)} corpus entries.")
    _preview("Chroma `conversations`", chroma_hits, lambda d: d.get("content"))
    _preview("Chroma `summaries`", summary_hits, lambda d: d.get("content"))
    _preview("Corpus JSON", corpus_hits,
             lambda e: f"User: {e.get('query','')} / Assistant: {e.get('response','')}")

    candidates_path = Path("data") / f"purge_candidates_{ts}.jsonl"
    with open(candidates_path, "w") as f:
        for d in chroma_hits:
            f.write(json.dumps({"store": "chroma", **d}, default=str) + "\n")
        for d in summary_hits:
            f.write(json.dumps({"store": "chroma_summaries", **d}, default=str) + "\n")
        for e in corpus_hits:
            f.write(json.dumps({"store": "corpus", "entry": e}, default=str) + "\n")
    print(f"\nFull candidate list written to {candidates_path}")

    if not chroma_hits and not summary_hits and not corpus_hits:
        print("Nothing to purge.")
        return 0

    if not args.apply:
        print("\nDRY RUN — nothing was deleted. Re-run with --apply to purge "
              "(a pre-image backup is written first).")
        return 0

    # ---- APPLY: pre-image backup, then delete ----
    backup_dir = Path("data") / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    preimage = backup_dir / f"purge_error_memories_preimage_{ts}.jsonl"
    with open(preimage, "w") as f:
        for d in chroma_hits:
            f.write(json.dumps({"store": "chroma", **d}, default=str) + "\n")
        for d in summary_hits:
            f.write(json.dumps({"store": "chroma_summaries", **d}, default=str) + "\n")
        for e in corpus_hits:
            f.write(json.dumps({"store": "corpus", "entry": e}, default=str) + "\n")
    print(f"Pre-image backup written to {preimage}")

    # Chroma: delete by id in batches
    ids = [d["id"] for d in chroma_hits if d.get("id")]
    if ids:
        coll = store._get_collection("conversations")
        for i in range(0, len(ids), 200):
            coll.delete(ids=ids[i : i + 200])
        print(f"Deleted {len(ids)} docs from chroma `conversations`.")

    sum_ids = [d["id"] for d in summary_hits if d.get("id")]
    if sum_ids:
        coll = store._get_collection("summaries")
        for i in range(0, len(sum_ids), 200):
            coll.delete(ids=sum_ids[i : i + 200])
        print(f"Deleted {len(sum_ids)} docs from chroma `summaries`.")

    # Corpus: drop hit entries, save through the manager's atomic write
    if corpus_hits:
        hit_ids = {id(e) for e in corpus_hits}
        cm.corpus = [e for e in corpus_entries if id(e) not in hit_ids]
        cm.save_corpus()
        print(f"Removed {len(corpus_hits)} corpus entries "
              f"({len(cm.corpus)} remain); corpus saved atomically.")

    print("\nDone. Restore path: the pre-image JSONL above contains every "
          "removed doc/entry verbatim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
