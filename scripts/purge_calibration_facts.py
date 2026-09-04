#!/usr/bin/env python3
"""
Purge SYNTHETIC calibration facts from the live stores (DRY-RUN FIRST).

scripts/generate_test_facts.py used to write ~80 invented facts straight into
the live CHROMA_PATH `facts` collection and knowledge graph (metadata
`source=test_calibration`: a brewery job, half-marathon running, D&D, a cat
named Mochi, a partner named Sarah…). The 2026-09-02 provenance audit found 48
of them live — the knowledge graph "knew" the owner played D&D and ran. They
are indistinguishable from real memory at retrieval time.

Selection is by the synthetic-source marker ONLY (`metadata.source ==
"test_calibration"`) — never by content, so a real fact that happens to say
"running" is untouched.

Graph handling (edges carry the Chroma fact ids that created them):
  - an edge whose source_fact_ids are ALL calibration facts is a removal
    candidate;
  - an edge with MIXED sources (calibration + real) is REPORTED, never removed;
  - nodes are NEVER removed here — nodes left with degree 0 after the edge
    removal are listed in a candidate file for scripts/graph_junk_cleanup.py
    (curated, dry-run-first), because "Sam"/"Sarah"/"Portland" may also be
    real entities.

Safety:
  - Default is DRY RUN (read-only): prints a report, writes
    data/calibration_fact_candidates_<ts>.jsonl + graph candidate files.
    Reading Chroma/graph while the daemon is live is fine.
  - --apply REFUSES to run while a live Daemon main.py is detected (it holds
    the stores in memory and would re-save over us). Writes pre-image backups
    (facts JSONL + full graph JSON copy) under data/backups/ BEFORE deleting.
  - Facts are deleted from Chroma; graph edges are removed via GraphMemory;
    nothing else is touched.

Usage:
    python scripts/purge_calibration_facts.py            # dry run
    python scripts/purge_calibration_facts.py --apply    # Daemon must be DOWN
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CALIBRATION_SOURCE = "test_calibration"


def _daemon_running() -> bool:
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pure selection helpers (unit-tested without Chroma)
# ---------------------------------------------------------------------------

def select_calibration_facts(docs):
    """Facts whose metadata carries the synthetic-source marker."""
    hits = []
    for d in docs or []:
        md = d.get("metadata") or {}
        if str(md.get("source", "")).strip() == CALIBRATION_SOURCE:
            hits.append({**d, "purge_reason": "synthetic_calibration_source"})
    return hits


def select_calibration_edges(graph_json, fact_ids):
    """Split graph edges into (fully_synthetic, mixed_provenance) lists."""
    fact_ids = set(fact_ids or [])
    full, mixed = [], []
    for edge in (graph_json or {}).get("edges", []) or []:
        sources = [s for s in (edge.get("source_fact_ids") or []) if s]
        if not sources:
            continue
        overlap = fact_ids.intersection(sources)
        if not overlap:
            continue
        if overlap == set(sources):
            full.append(edge)
        else:
            mixed.append({**edge, "synthetic_fact_ids": sorted(overlap)})
    return full, mixed


def orphan_node_candidates(graph_json, edges_to_remove):
    """Nodes that would have degree 0 once `edges_to_remove` are gone."""
    removed = {(e["source_id"], e["relation"], e["target_id"]) for e in edges_to_remove}
    degree = {}
    for edge in (graph_json or {}).get("edges", []) or []:
        key = (edge.get("source_id"), edge.get("relation"), edge.get("target_id"))
        if key in removed:
            continue
        for nid in (edge.get("source_id"), edge.get("target_id")):
            degree[nid] = degree.get(nid, 0) + 1
    touched = set()
    for e in edges_to_remove:
        touched.add(e["source_id"])
        touched.add(e["target_id"])
    nodes = (graph_json or {}).get("nodes", {}) or {}
    return sorted(n for n in touched if n != "user" and n in nodes and degree.get(n, 0) == 0)


# ---------------------------------------------------------------------------
# Store access
# ---------------------------------------------------------------------------

def _scan():
    from config.app_config import CHROMA_PATH, KNOWLEDGE_GRAPH_PERSIST_PATH
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    docs = store.list_all("facts")
    hits = select_calibration_facts(docs)
    graph_path = Path(KNOWLEDGE_GRAPH_PERSIST_PATH)
    graph_json = json.loads(graph_path.read_text()) if graph_path.exists() else {}
    full, mixed = select_calibration_edges(graph_json, [h.get("id") for h in hits])
    orphans = orphan_node_candidates(graph_json, full)
    return store, docs, hits, graph_path, graph_json, full, mixed, orphans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--apply", action="store_true",
                    help="Delete the facts + fully-synthetic edges (pre-image backups first). "
                         "Default: dry run.")
    ap.add_argument("--force", action="store_true",
                    help="With --apply: run even if a live Daemon process is detected.")
    args = ap.parse_args()

    if args.apply and _daemon_running() and not args.force:
        print("ABORT: a live Daemon main.py process is running — it holds the stores in "
              "memory and would re-save over this purge. Shut it down first "
              "(or --force if the match is stale).")
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    store, docs, hits, graph_path, graph_json, full, mixed, orphans = _scan()

    print(f"Scanned {len(docs)} facts; {len(hits)} carry source={CALIBRATION_SOURCE!r}.")
    for h in hits[:12]:
        print(f"  - {str(h.get('content', ''))[:100]!r}")
    if len(hits) > 12:
        print(f"  ... and {len(hits) - 12} more (full list in the candidates file)")
    print(f"\nGraph: {len(full)} edge(s) sourced ONLY by calibration facts (removal candidates)")
    for e in full[:12]:
        print(f"  - {e['source_id']} --{e['relation']}--> {e['target_id']}")
    if len(full) > 12:
        print(f"  ... and {len(full) - 12} more")
    print(f"Graph: {len(mixed)} edge(s) with MIXED provenance (reported only, never removed)")
    for e in mixed[:8]:
        print(f"  - {e['source_id']} --{e['relation']}--> {e['target_id']} "
              f"(synthetic {len(e['synthetic_fact_ids'])}/{len(e.get('source_fact_ids') or [])})")
    print(f"Graph: {len(orphans)} node(s) would be left with degree 0 (NOT removed; "
          f"review with scripts/graph_junk_cleanup.py)")
    for n in orphans[:20]:
        print(f"  - {n}")

    out_dir = Path("data")
    cand = out_dir / f"calibration_fact_candidates_{ts}.jsonl"
    with open(cand, "w") as f:
        for h in hits:
            f.write(json.dumps(h, default=str) + "\n")
    print(f"\nFact candidates written to {cand}")
    if full or mixed:
        gcand = out_dir / f"calibration_graph_edge_candidates_{ts}.json"
        gcand.write_text(json.dumps({"remove": full, "mixed_report_only": mixed}, indent=2, default=str))
        print(f"Graph edge candidates written to {gcand}")
    if orphans:
        ocand = out_dir / f"calibration_orphan_node_candidates_{ts}.txt"
        ocand.write_text("\n".join(orphans) + "\n")
        print(f"Orphan-node candidates (for graph_junk_cleanup.py) written to {ocand}")

    if not hits and not full:
        print("Nothing to purge.")
        return 0
    if not args.apply:
        print("\nDRY RUN — nothing was modified. Re-run with --apply (Daemon DOWN) to purge; "
              "pre-image backups are written first.")
        return 0

    backup_dir = Path("data") / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    preimage = backup_dir / f"purge_calibration_facts_preimage_{ts}.jsonl"
    with open(preimage, "w") as f:
        for h in hits:
            f.write(json.dumps(h, default=str) + "\n")
    print(f"Pre-image fact backup written to {preimage}")
    if graph_path.exists():
        gbackup = backup_dir / f"purge_calibration_graph_preimage_{ts}.json"
        shutil.copy2(graph_path, gbackup)
        print(f"Pre-image graph backup written to {gbackup}")

    ids = [h["id"] for h in hits if h.get("id")]
    if ids:
        coll = store._get_collection("facts")
        for i in range(0, len(ids), 200):
            coll.delete(ids=ids[i:i + 200])
        print(f"Deleted {len(ids)} calibration facts from chroma `facts`.")

    if full:
        from memory.graph_memory import GraphMemory
        g = GraphMemory(persist_path=str(graph_path))
        removed = 0
        for e in full:
            ekey = f"{e['source_id']}|{e['relation']}|{e['target_id']}"
            if ekey in g._edge_index:
                del g._edge_index[ekey]
                removed += 1
            if g.graph.has_edge(e["source_id"], e["target_id"]):
                edge_rel = g.graph[e["source_id"]][e["target_id"]].get("relation")
                if edge_rel == e["relation"]:
                    g.graph.remove_edge(e["source_id"], e["target_id"])
        # Re-sync the relation-level adjacency maps (and restore any nx pair
        # a surviving sibling relation still needs) after the direct
        # _edge_index mutation above (2026-09-03).
        g.rebuild_edge_indexes()
        g._mark_dirty()
        g.save()
        print(f"Removed {removed} fully-synthetic graph edge(s). Nodes untouched "
              f"({len(orphans)} orphan candidates listed above).")

    print("\nDone. Restore path: the pre-image files above contain every removed record verbatim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
