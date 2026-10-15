#!/usr/bin/env python3
"""
Quarantine (or un-quarantine) conversations / summaries / reflections in BOTH copies -
REVERSIBLE, DRY-RUN FIRST.  (2026-10-15, class: BC-26, BC-58)

Every conversation, summary and reflection lives twice: a Chroma document and a dict in
the rolling corpus JSON.  `scripts/quarantine_facts.py` covers `facts` (Chroma only - facts
have no corpus copy).  This tool flips the quarantine flag on the Chroma doc through THE
deployed `MultiCollectionChromaStore.update_metadata` AND on the matching corpus entry
(matched by `memory.quarantine_sync.match_corpus_index`, saved once through THE deployed
`CorpusManager.save_corpus`).  Corpus readers hide flagged entries via
`CorpusManager.visible_entries()`.

Input (`--from-file`): TSV lines `collection<TAB>id` ('#' comments allowed).

Safety:
  - Default is DRY RUN: prints per target the collection, id, Chroma state, corpus outcome
    (match / no_corpus_copy / ambiguous) and corpus state (60-char prefix only).
  - --apply REFUSES while a live Daemon main.py is detected (fails closed), writes ONE
    pre-image JSONL (Chroma metadata + matched corpus entry, before any write) to
    backups/quarantine_memory_preimage_<ts>.jsonl beside the corpus file (data/), then flips.
  - A missing or ambiguous corpus copy never blocks the Chroma flip; it is reported.
  - A target already in the requested state is skipped per copy (so the tool can finish only
    the corpus half of an earlier Chroma-only quarantine).
  - --undo (with --apply) clears both flags.  Nothing is ever deleted.

Usage:
    python scripts/quarantine_memory.py --from-file targets.tsv
    python scripts/quarantine_memory.py --from-file targets.tsv --reason "owner review" --apply
    python scripts/quarantine_memory.py --from-file targets.tsv --undo --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.quarantine_facts import (  # noqa: E402
    QUARANTINE_KEY,
    metadata_update,
)
from utils.daemon_guard import daemon_running  # noqa: E402  (same guard quarantine_facts wraps)

# None = "backups/" beside the corpus file (the live data dir, whatever the cwd or
# checkout the script runs from); tests point it at a tmp dir.
BACKUP_DIR = None


def parse_targets(text: str) -> list[tuple[str, str]]:
    """`collection<TAB>id` rows ('#' comments); malformed rows are skipped, order kept, deduped."""
    out: list[tuple[str, str]] = []
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            continue
        row = (parts[0].strip(), parts[1].strip())
        if row not in out:
            out.append(row)
    return out


def _open_store():
    from config.app_config import CHROMA_PATH
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    return MultiCollectionChromaStore(persist_directory=CHROMA_PATH)


def _open_corpus():
    from config.app_config import CORPUS_FILE
    from memory.corpus_manager import CorpusManager
    return CorpusManager(CORPUS_FILE)


def _read_chroma_doc(store, collection: str, doc_id: str):
    if collection not in getattr(store, "collections", {}):
        return None
    res = store._get_collection(collection).get(ids=[doc_id], include=["documents", "metadatas"])
    if not res or not res.get("ids"):
        return None
    return {"id": doc_id, "content": (res.get("documents") or [""])[0] or "",
            "metadata": (res.get("metadatas") or [{}])[0] or {}}


def plan_targets(store, cm, targets):
    """One row per target: doc, corpus index/reason, current flag state of each copy."""
    from memory.quarantine_sync import match_corpus_index
    rows = []
    for collection, doc_id in targets:
        doc = _read_chroma_doc(store, collection, doc_id)
        row = {"collection": collection, "id": doc_id, "doc": doc, "index": None,
               "reason": "chroma_missing" if doc is None else None,
               "chroma_flagged": None, "corpus_flagged": None}
        if doc is not None:
            row["chroma_flagged"] = bool(doc["metadata"].get(QUARANTINE_KEY))
            idx, reason = match_corpus_index(collection, doc, cm.corpus)
            row["index"], row["reason"] = idx, reason
            if idx is not None:
                row["corpus_flagged"] = bool(cm.corpus[idx].get(QUARANTINE_KEY))
        rows.append(row)
    return rows


def _print_plan(rows, undo: bool) -> None:
    verb = "un-quarantine" if undo else "quarantine"
    print(f"{len(rows)} target(s); action = {verb}")
    for r in rows:
        prefix = repr(str((r["doc"] or {}).get("content", ""))[:60]) if r["doc"] else "-"
        print(f"  {r['collection']}\t{r['id']}\tchroma={'MISSING' if r['doc'] is None else ('flagged' if r['chroma_flagged'] else 'clear')}"
              f"\tcorpus={r['reason']}"
              f"{'' if r['corpus_flagged'] is None else (' flagged' if r['corpus_flagged'] else ' clear')}\t{prefix}")


def run(rows, store, cm, *, undo: bool, reason: str, backup_dir: Path) -> dict:
    """Apply a plan: pre-image first, then Chroma flips, then ONE corpus save."""
    from memory.quarantine_sync import set_corpus_flag

    ts = datetime.now()
    stamp = ts.strftime("%Y%m%d_%H%M%S")
    update = metadata_update(undo=undo, reason=reason, ts=ts.isoformat())
    backup_dir.mkdir(parents=True, exist_ok=True)
    preimage = backup_dir / f"quarantine_memory_preimage_{stamp}.jsonl"
    with open(preimage, "w") as f:  # write-once audit artifact, not a store
        for r in rows:
            f.write(json.dumps({
                "collection": r["collection"], "id": r["id"],
                "chroma_metadata": (r["doc"] or {}).get("metadata"),
                "corpus_index": r["index"],
                "corpus_entry": cm.corpus[r["index"]] if r["index"] is not None else None,
            }, default=str) + "\n")

    target_state = not undo
    chroma_changed = corpus_changed = 0
    flipped_entries: set[int] = set()
    for r in rows:
        if r["doc"] is not None and r["chroma_flagged"] != target_state:
            if store.update_metadata(r["collection"], r["id"], dict(update)):
                chroma_changed += 1
        if r["index"] is not None and r["index"] not in flipped_entries \
                and bool(cm.corpus[r["index"]].get(QUARANTINE_KEY)) != target_state:
            set_corpus_flag(cm.corpus[r["index"]], dict(update))
            flipped_entries.add(r["index"])
            corpus_changed += 1
    if corpus_changed:
        cm.save_corpus()
    cm.invalidate_read_cache()
    return {"preimage": preimage, "chroma_changed": chroma_changed, "corpus_changed": corpus_changed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--from-file", required=True, help="TSV: collection<TAB>id per line")
    ap.add_argument("--reason", default="owner_quarantine", help="stored as curation_quarantine_reason")
    ap.add_argument("--undo", action="store_true", help="clear the flag on both copies")
    ap.add_argument("--apply", action="store_true", help="actually flip (pre-image backup first)")
    args = ap.parse_args(argv)

    if args.apply and daemon_running():
        print("ABORT: a live Daemon main.py process is running (or could not be ruled out) - it holds "
              "ChromaDB and the corpus in memory and would re-save over this change. Shut it down first.")
        return 1

    targets = parse_targets(Path(args.from_file).read_text())
    if not targets:
        print("No targets in file; nothing to do.")
        return 0

    store = _open_store()
    cm = _open_corpus()
    rows = plan_targets(store, cm, targets)
    _print_plan(rows, args.undo)
    if not args.apply:
        print("\nDRY RUN - nothing was modified. Re-run with --apply (Daemon DOWN).")
        return 0

    backup_dir = BACKUP_DIR or Path(cm.corpus_file).resolve().parent / "backups"
    result = run(rows, store, cm, undo=args.undo, reason=args.reason, backup_dir=backup_dir)
    print(f"Pre-image: {result['preimage']}")
    print(f"Chroma flips: {result['chroma_changed']}, corpus flips: {result['corpus_changed']}. "
          f"Reversible: re-run with {'--apply' if args.undo else '--undo --apply'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
