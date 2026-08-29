#!/usr/bin/env python3
"""Backfill stance metadata onto historical facts + graph edges (DRY-RUN FIRST).

2026-08-23 stance/epistemic-tagging layer (Phase B4): historical facts and
graph edges carry no stance tag, so every read-side consumer treats them as
"unknown" (conservative but blind — the casey--is-->evil edge still renders
as a bare world-fact until tagged). This script classifies EVERY stored fact
triple and graph edge with THE DEPLOYED deterministic classifier
(memory/stance_classifier.classify_triple_stance — never a re-derivation)
and writes the stance into metadata. capture_tone is NOT backfilled — the
capture-time tone regime is unrecoverable historically, and settledness
deliberately requires explicit non-elevated evidence (absent = never counts).

Safety model (store-writing script contract):
  * Default is DRY RUN — prints the planned updates table and exits.
  * HARD SENTINEL: the dry run exits nonzero unless the known appraisal fact
    (fact_e1f5f920_20260818_135210570, `casey | is | evil`) classifies as
    "appraisal". If the deployed classifier can't get the sentinel right,
    nothing may be written.
  * --apply refuses while a live Daemon main.py is detected
    (utils/daemon_guard — the running instance holds stores in memory and
    would clobber the writes on its next save).
  * --apply writes pre-image backups first to
    data/backups/stance_backfill_<ts>/: a copy of knowledge_graph.json and a
    JSONL dump of every fact's (id, metadata) via safe_json atomic write.

Usage:
    python scripts/backfill_stance.py            # dry run + sentinel check
    python scripts/backfill_stance.py --apply    # write (Daemon must be down)
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SENTINEL_FACT_ID = "fact_e1f5f920_20260818_135210570"

VALID_TARGET_STANCES = ("objective", "appraisal", "reported")


def _daemon_running() -> bool:
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception:
        return False


def classify_fact_doc(doc: dict) -> str | None:
    """Classify one fact document's stance with THE deployed classifier.
    Returns None when no triple can be parsed (nothing to write)."""
    from memory.cross_deduplicator import CrossCollectionDeduplicator
    from memory.stance_classifier import classify_triple_stance

    subj, pred, obj = CrossCollectionDeduplicator._extract_triple(doc)
    if not subj or not pred:
        return None
    return classify_triple_stance(subj, pred, obj).stance


def plan_fact_updates(fact_docs: list) -> tuple:
    """Return (updates, stats, sentinel_ok). updates = [(doc_id, stance)].
    Docs already carrying a valid stance are skipped (idempotent re-runs)."""
    from memory.stance_classifier import VALID_STANCES

    updates = []
    stats = {"total": 0, "already_tagged": 0, "unparseable": 0}
    per_stance = {}
    sentinel_seen = False
    sentinel_ok = False

    for doc in fact_docs:
        stats["total"] += 1
        doc_id = doc.get("id") or ""
        md = doc.get("metadata", {}) or {}
        if md.get("stance") in VALID_STANCES:
            stats["already_tagged"] += 1
            if doc_id == SENTINEL_FACT_ID:
                sentinel_seen = True
                sentinel_ok = md.get("stance") == "appraisal"
            continue
        stance = classify_fact_doc(doc)
        if stance is None:
            stats["unparseable"] += 1
            continue
        per_stance[stance] = per_stance.get(stance, 0) + 1
        updates.append((doc_id, stance))
        if doc_id == SENTINEL_FACT_ID:
            sentinel_seen = True
            sentinel_ok = stance == "appraisal"

    stats["per_stance"] = per_stance
    stats["sentinel_seen"] = sentinel_seen
    return updates, stats, sentinel_ok


def plan_graph_updates(graph_memory) -> list:
    """Return [(edge_key, stance)] for edges lacking a valid stance tag."""
    from memory.stance_classifier import VALID_STANCES, classify_triple_stance

    updates = []
    for ekey, edge in graph_memory._edge_index.items():
        if (edge.metadata or {}).get("stance") in VALID_STANCES:
            continue
        stance = classify_triple_stance(
            edge.source_id, edge.relation, edge.target_id
        ).stance
        updates.append((ekey, stance))
    return updates


def _backup(graph_path: Path, fact_docs: list, backup_dir: Path) -> None:
    from utils.safe_json import atomic_write_json

    backup_dir.mkdir(parents=True, exist_ok=True)
    if graph_path.exists():
        shutil.copy2(graph_path, backup_dir / graph_path.name)
    atomic_write_json(
        str(backup_dir / "facts_metadata_preimage.json"),
        [{"id": d.get("id"), "metadata": d.get("metadata", {})} for d in fact_docs],
    )


def main(argv=None) -> int:
    global SENTINEL_FACT_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Write the updates (default: dry run)")
    parser.add_argument("--sentinel-id", default=SENTINEL_FACT_ID,
                        help="Override the sentinel fact id (tests only)")
    args = parser.parse_args(argv)

    SENTINEL_FACT_ID = args.sentinel_id

    if args.apply and _daemon_running():
        print("REFUSED: a live Daemon main.py is running. Its in-memory stores "
              "would clobber these writes on the next save. Shut it down first.")
        return 2

    from config.app_config import CHROMA_PATH
    from memory.graph_memory import GraphMemory
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    # THE deployed store path (config memory.chroma_path / CHROMA_PATH env) —
    # the class default ("data/chroma_multi") is a stale artifact dir and
    # scanning it would silently backfill the wrong store.
    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    fact_docs = store.list_all("facts") or []
    graph_memory = GraphMemory()

    fact_updates, stats, sentinel_ok = plan_fact_updates(fact_docs)
    graph_updates = plan_graph_updates(graph_memory)

    print(f"Facts scanned:        {stats['total']}")
    print(f"  already tagged:     {stats['already_tagged']}")
    print(f"  unparseable:        {stats['unparseable']}")
    print(f"  to update:          {len(fact_updates)}")
    for stance, n in sorted(stats["per_stance"].items()):
        print(f"    {stance:<12} {n}")
    print(f"Graph edges to tag:   {len(graph_updates)}")
    _appraisal_edges = [k for k, s in graph_updates if s == "appraisal"]
    for k in _appraisal_edges[:20]:
        print(f"    appraisal edge: {k}")

    if not stats["sentinel_seen"]:
        print(f"\nSENTINEL MISSING: fact {SENTINEL_FACT_ID} not found in the "
              "facts collection — refusing (wrong store, or the fact was "
              "purged; verify before writing).")
        return 1
    if not sentinel_ok:
        print(f"\nSENTINEL FAILED: fact {SENTINEL_FACT_ID} did not classify "
              "as 'appraisal' — the deployed classifier is wrong for the "
              "known case; nothing may be written.")
        return 1
    print(f"\nSentinel OK: {SENTINEL_FACT_ID} -> appraisal")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply (Daemon down).")
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("data/backups") / f"stance_backfill_{ts}"
    graph_path = Path(getattr(graph_memory, "persist_path", "data/knowledge_graph.json"))
    _backup(graph_path, fact_docs, backup_dir)
    print(f"Pre-image backup written to {backup_dir}")

    written = 0
    for doc_id, stance in fact_updates:
        if not doc_id:
            continue
        try:
            store.update_metadata("facts", doc_id, {"stance": stance})
            written += 1
        except Exception as e:
            print(f"  update failed for {doc_id}: {e}")
    print(f"Facts updated: {written}/{len(fact_updates)}")

    for ekey, stance in graph_updates:
        edge = graph_memory._edge_index.get(ekey)
        if edge is None:
            continue
        edge.metadata["stance"] = stance
        src, rel, tgt = ekey.split("|", 2)
        try:
            graph_memory.graph[src][tgt]["metadata"] = edge.metadata
        except Exception:
            pass
    graph_memory._mark_dirty()
    graph_memory.save()
    print(f"Graph edges updated: {len(graph_updates)} (graph saved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
