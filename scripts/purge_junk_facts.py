"""
Purge junk facts from the `facts` collection (DRY-RUN FIRST).

Targets facts stored before the 2026-08-02 extraction-time guards:
  - Adverbial/temporal/negation fragment objects ("for a bit", "with food",
    "yesterday", "not good", "not eaten yet") and bare feeling adjectives
    ("unhappy") — no durable factual content.
  - Polarity-inverted preferences: a positive relation (likes/loves/enjoys/
    favorite_*) whose own source excerpt negates the object ("Feel like I
    hate my fucking life" was stored as user | likes | my fucking life).

Selection uses THE deployed extraction guards
(memory.fact_extractor._is_junk_object / _polarity_conflict) — the same
functions that now block these at extraction time — never a re-derivation.

Also writes (report only, never deletes) a graph-node candidate list of
junk-shaped entity ids for review with scripts/graph_junk_cleanup.py.

Safety:
  - Default is DRY RUN: prints a report and writes the candidate list to
    data/junk_fact_candidates_<ts>.jsonl. Nothing is modified.
  - --apply writes a pre-image backup JSONL of every fact it is about to
    remove (data/backups/purge_junk_facts_preimage_<ts>.jsonl), THEN deletes
    from Chroma. Graph nodes are NEVER touched by this script.
  - Refuses to run while a live Daemon main.py is detected (--force to
    override a stale pgrep match).

Usage:
    python scripts/purge_junk_facts.py            # dry run (report only)
    python scripts/purge_junk_facts.py --apply    # backup + delete facts
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# THE deployed extraction-time guards
from memory.fact_extractor import _is_junk_object, _polarity_conflict


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


def _classify(doc: dict):
    """Return a reason string when the stored fact is junk, else None.

    Older facts carry no subject/relation/object metadata — parse the
    canonical "subject | relation | object" content form. Anything that
    doesn't parse is left alone (fail-safe: never junk by absence).
    """
    md = doc.get("metadata") or {}
    rel = str(md.get("relation", "") or "")
    obj = str(md.get("object", "") or "")
    if not rel or not obj:
        parts = [p.strip() for p in str(doc.get("content", "")).split(" | ", 2)]
        if len(parts) != 3 or not all(parts):
            return None
        _, rel, obj = parts
    src = str(md.get("source_excerpt", "") or md.get("source_text", "") or "")
    if _is_junk_object(obj, rel):
        return "junk_object"
    if src and _polarity_conflict(src, rel, obj):
        return "polarity_inverted"
    return None


def _scan_facts():
    from config.app_config import CHROMA_PATH
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    docs = store.list_all("facts")
    hits = []
    for d in docs:
        reason = _classify(d)
        if reason:
            hits.append({**d, "purge_reason": reason})
    return store, docs, hits


def _graph_node_candidates():
    """Junk-shaped graph node ids (report only; review via graph_junk_cleanup.py)."""
    try:
        path = Path("data") / "knowledge_graph.json"
        g = json.loads(path.read_text())
        nodes = g.get("nodes", [])
        ids = (list(nodes.keys()) if isinstance(nodes, dict)
               else [n.get("id", "") for n in nodes])
        return [i for i in ids
                if i and i != "user" and _is_junk_object(i.replace("_", " "), "")]
    except Exception as e:
        print(f"(graph scan skipped: {e})")
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete facts (after a pre-image backup). Default: dry run.")
    ap.add_argument("--from-file", metavar="JSONL",
                    help="With --apply: delete ONLY the ids in this (hand-edited) "
                         "candidates JSONL instead of everything the rescan flags. "
                         "Delete lines you want to KEEP before running.")
    ap.add_argument("--force", action="store_true",
                    help="Run even if a live Daemon process is detected.")
    args = ap.parse_args()

    if _daemon_running() and not args.force:
        print("ABORT: a live Daemon main.py process is running — it holds ChromaDB "
              "open. Close it first (or --force if the match is stale).")
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    store, docs, hits = _scan_facts()

    print(f"Scanned {len(docs)} facts.")
    by_reason = {}
    for h in hits:
        by_reason.setdefault(h["purge_reason"], []).append(h)
    for reason, items in sorted(by_reason.items()):
        print(f"\n{reason}: {len(items)} candidate(s)")
        for it in items[:10]:
            print(f"  - {it.get('content', '')[:110]!r}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more (full list in the candidates file)")

    candidates_path = Path("data") / f"junk_fact_candidates_{ts}.jsonl"
    with open(candidates_path, "w") as f:
        for h in hits:
            f.write(json.dumps(h, default=str) + "\n")
    print(f"\nFull candidate list written to {candidates_path}")

    graph_junk = _graph_node_candidates()
    if graph_junk:
        gpath = Path("data") / f"junk_graph_node_candidates_{ts}.txt"
        gpath.write_text("\n".join(graph_junk) + "\n")
        print(f"Graph-node candidates (NOT deleted; review with "
              f"scripts/graph_junk_cleanup.py): {len(graph_junk)} → {gpath}")

    if not hits:
        print("Nothing to purge.")
        return 0

    if not args.apply:
        print("\nDRY RUN — nothing was deleted. Re-run with --apply to purge "
              "(a pre-image backup is written first), or edit the candidates "
              "file to remove KEEP lines and use --apply --from-file <path>.")
        return 0

    if args.from_file:
        keep_ids = set()
        with open(args.from_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    keep_ids.add(json.loads(line).get("id"))
        before = len(hits)
        hits = [h for h in hits if h.get("id") in keep_ids]
        print(f"--from-file: restricting to {len(hits)}/{before} rescan hits "
              f"present in {args.from_file}")

    backup_dir = Path("data") / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    preimage = backup_dir / f"purge_junk_facts_preimage_{ts}.jsonl"
    with open(preimage, "w") as f:
        for h in hits:
            f.write(json.dumps(h, default=str) + "\n")
    print(f"Pre-image backup written to {preimage}")

    ids = [h["id"] for h in hits if h.get("id")]
    coll = store._get_collection("facts")
    for i in range(0, len(ids), 200):
        coll.delete(ids=ids[i : i + 200])
    print(f"Deleted {len(ids)} junk facts from chroma `facts`.")
    print("\nDone. Restore path: the pre-image JSONL above contains every "
          "removed fact verbatim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
