#!/usr/bin/env python3
"""
Quarantine (or un-quarantine) knowledge-graph EDGES by key — REVERSIBLE, DRY-RUN FIRST.

Graph counterpart of scripts/quarantine_facts.py. A quarantined edge stays on
disk but never renders: GraphMemory.edge_is_suppressed() (metadata
`curation_quarantined=True`) is honoured by the [KNOWLEDGE GRAPH] prompt
section, the proactive-insight surfacer and the insight sweep. Nothing is
deleted; `--undo --apply` clears the flag.

2026-09-03 motivating case: `user|has_dog|biscuit`, `user|has_dog|mochi`,
`user|has_dog|bean` (all three nodes carry curated `species: cat`) rendered
"User has dog Mochi" into every turn and the reply called the cat a dog; the
`facts`-collection copies were quarantined on 09-02 but the graph copies were
not. The read-time species guard now neutralizes them regardless — this
script records the decision in the store so the edges stop showing up in
graph exports/tools too.

Edge keys are `source|relation|target` (node ids, lower-case), one per line
('#' comments allowed), or passed with repeated --edge.

Safety:
  - Default is DRY RUN: resolves each key, prints the edge + current state.
  - --apply REFUSES to run while a live Daemon main.py is detected (it holds
    the graph in memory and would re-save the old state). Writes a pre-image
    JSONL of every touched edge to data/backups/quarantine_graph_edges_preimage_<ts>.jsonl
    BEFORE flipping, then saves through GraphMemory.save() (atomic).

Usage:
    python scripts/quarantine_graph_edges.py --edge "user|has_dog|mochi" --edge "user|has_dog|biscuit"
    python scripts/quarantine_graph_edges.py --from-file data/graph_edge_quarantine_candidates_20260903.txt --reason "species conflict" --apply
    python scripts/quarantine_graph_edges.py --from-file ... --undo --apply
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUARANTINE_KEY = "curation_quarantined"


def _daemon_running() -> bool:
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception:
        return False


def parse_edge_keys(text: str) -> list:
    keys = []
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip().lower() for p in line.split("|")]
        if len(parts) != 3 or not all(parts):
            print(f"  ! skipping malformed key: {raw!r}")
            continue
        keys.append("|".join(parts))
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edge", action="append", default=[], help="edge key source|relation|target (repeatable)")
    ap.add_argument("--from-file", help="file of edge keys, one per line")
    ap.add_argument("--graph", default=None, help="graph JSON path (default: config KNOWLEDGE_GRAPH_PERSIST_PATH)")
    ap.add_argument("--reason", default="owner quarantine", help="recorded in edge metadata")
    ap.add_argument("--undo", action="store_true", help="clear the flag instead of setting it")
    ap.add_argument("--apply", action="store_true", help="actually flip metadata (pre-image backup first)")
    ap.add_argument("--force", action="store_true", help="with --apply: run even if a live Daemon is detected")
    args = ap.parse_args()

    keys = parse_edge_keys("\n".join(args.edge))
    if args.from_file:
        keys += parse_edge_keys(Path(args.from_file).read_text(encoding="utf-8"))
    keys = list(dict.fromkeys(keys))
    if not keys:
        print("No edge keys given."); return 2

    if args.apply and _daemon_running() and not args.force:
        print("REFUSED: a live Daemon main.py is running — it holds the graph in memory and "
              "would re-save the pre-flip state. Shut it down, then re-run with --apply.")
        return 3

    graph_path = args.graph
    if not graph_path:
        try:
            from config.app_config import KNOWLEDGE_GRAPH_PERSIST_PATH as graph_path  # noqa: N813
        except Exception:
            graph_path = "data/knowledge_graph.json"
    from memory.graph_memory import GraphMemory
    gm = GraphMemory(persist_path=graph_path)
    gm.load()
    print(f"Graph: {graph_path} ({gm.node_count()} nodes, {gm.edge_count()} edges)")

    verb = "UN-quarantine" if args.undo else "quarantine"
    touched = []
    for key in keys:
        edge = gm._edge_index.get(key)
        if edge is None:
            print(f"  ? not found: {key}")
            continue
        md = dict(edge.metadata or {})
        state = bool(md.get(QUARANTINE_KEY))
        print(f"  {key}  w={edge.weight} first={str(edge.first_seen)[:16]} quarantined={state}"
              f"  -> would {verb}")
        touched.append((key, edge, md))

    if not args.apply:
        print(f"\nDRY RUN — nothing was modified. Re-run with --apply (Daemon DOWN) to {verb} "
              f"{len(touched)} edge(s).")
        return 0
    if not touched:
        print("Nothing to do."); return 0

    os.makedirs("data/backups", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pre = Path("data/backups") / f"quarantine_graph_edges_preimage_{ts}.jsonl"
    with open(pre, "w", encoding="utf-8") as fh:
        for key, edge, md in touched:
            fh.write(json.dumps({"edge_key": key, "metadata": md}, default=str) + "\n")
    print(f"Pre-image written: {pre}")

    for key, edge, md in touched:
        md = dict(edge.metadata or {})
        if args.undo:
            for k in (QUARANTINE_KEY, "curation_reason", "curation_ts"):
                md.pop(k, None)
        else:
            md[QUARANTINE_KEY] = True
            md["curation_reason"] = args.reason
            md["curation_ts"] = datetime.now().isoformat()
        edge.metadata = md
        gm._edge_index[key] = edge
        try:
            gm.graph[edge.source_id][edge.target_id]["metadata"] = md
        except Exception:
            pass
    if hasattr(gm, "_mark_dirty"):
        gm._mark_dirty()
    else:
        gm._dirty = True
    gm.save()
    print(f"Applied: {verb} on {len(touched)} edge(s). Undo with --undo --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
