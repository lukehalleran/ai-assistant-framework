#!/usr/bin/env python3
"""
scripts/graph_relation_normalize.py

One-time migration: re-canonicalize edge relations in the knowledge graph
(data/knowledge_graph.json) through the DEPLOYED normalize_relation()
(memory/entity_resolver.py) — never a re-derivation (CLAUDE.md rule #3).

Historical edges were ingested before the family-collapse patterns existed
("asked_about_wakeup_time", "class_start_date", "inquire_about", ...), so the
live graph carries variant relations that new ingestion would now collapse.
This rewrites them to match, merging edges that collide on the same
(source, canonical_relation, target): weights sum, source_fact_ids union,
first_seen takes the earliest, last_seen the latest.

Safety model (per CLAUDE.md "NEVER auto-delete user data"):
  * Default run is DRY-RUN: prints the full old->new mapping and merge
    collisions, touches nothing.
  * --apply backs up the graph JSON first (timestamped byte-copy).

Usage:
    python scripts/graph_relation_normalize.py            # dry-run
    python scripts/graph_relation_normalize.py --apply    # rewrite (backs up first)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.entity_resolver import normalize_relation  # noqa: E402
from memory.graph_memory import GraphMemory  # noqa: E402

DEFAULT_GRAPH = os.path.join("data", "knowledge_graph.json")


def plan(gm: GraphMemory):
    """Return (renames, merges): edges whose relation changes, and canonical
    keys that more than one existing edge collapses into."""
    renames = []  # (old_key, edge, canon_rel)
    targets = {}  # canonical key -> [old_key, ...]
    for key, edge in gm._edge_index.items():
        canon = normalize_relation(edge.relation)
        canon_key = f"{edge.source_id}|{canon}|{edge.target_id}"
        targets.setdefault(canon_key, []).append(key)
        if canon != edge.relation:
            renames.append((key, edge, canon))
    merges = {ck: keys for ck, keys in targets.items() if len(keys) > 1}
    return renames, merges


def backup(src: str) -> str:
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = f"{src}.bak-{stamp}"
    with open(src, "rb") as r, open(dst, "wb") as w:
        w.write(r.read())
    return dst


def apply_plan(gm: GraphMemory, renames, merges) -> tuple[int, int]:
    """Rewrite relations in the edge index + NetworkX graph. Returns
    (renamed_count, merged_count)."""
    renamed = 0
    merged = 0
    # Process merges first so rename collisions fold instead of clobbering.
    merge_members = {k for keys in merges.values() for k in keys}

    for canon_key, keys in merges.items():
        edges = [gm._edge_index[k] for k in keys]
        survivor = edges[0]
        canon_rel = canon_key.split("|")[1]
        survivor.relation = canon_rel
        survivor.weight = sum(e.weight for e in edges)
        fact_ids = []
        for e in edges:
            for fid in e.source_fact_ids:
                if fid not in fact_ids:
                    fact_ids.append(fid)
        survivor.source_fact_ids = fact_ids
        firsts = [e.first_seen for e in edges if e.first_seen]
        lasts = [e.last_seen for e in edges if e.last_seen]
        if firsts:
            survivor.first_seen = min(firsts)
        if lasts:
            survivor.last_seen = max(lasts)
        for k in keys:
            del gm._edge_index[k]
        gm._edge_index[canon_key] = survivor
        _sync_nx_edge(gm, survivor)
        merged += len(keys) - 1

    for old_key, edge, canon in renames:
        if old_key in merge_members:
            continue  # already folded above
        del gm._edge_index[old_key]
        edge.relation = canon
        gm._edge_index[f"{edge.source_id}|{canon}|{edge.target_id}"] = edge
        _sync_nx_edge(gm, edge)
        renamed += 1

    gm._mark_dirty()
    return renamed, merged


def _sync_nx_edge(gm: GraphMemory, edge) -> None:
    if gm.graph.has_edge(edge.source_id, edge.target_id):
        data = gm.graph[edge.source_id][edge.target_id]
        data["relation"] = edge.relation
        data["weight"] = edge.weight
        data["source_fact_ids"] = edge.source_fact_ids


def main():
    ap = argparse.ArgumentParser(description="Canonicalize graph edge relations.")
    ap.add_argument("--graph", default=DEFAULT_GRAPH, help="graph JSON path")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite relations (backs up the graph JSON first)")
    args = ap.parse_args()

    gm = GraphMemory(persist_path=args.graph)
    print(f"[graph] {gm.node_count()} nodes, {gm.edge_count()} edges @ {args.graph}")

    renames, merges = plan(gm)
    print(f"\nrelation renames: {len(renames)}")
    for _, edge, canon in sorted(renames, key=lambda r: r[1].relation):
        print(f"  {edge.source_id} --{edge.relation}--> {edge.target_id}   =>   {canon}")
    print(f"\nmerge collisions (same src|canon|tgt): {len(merges)}")
    for ck, keys in merges.items():
        print(f"  {ck}  <=  {keys}")

    if not args.apply:
        print("\n=== DRY-RUN (no changes made) — re-run with --apply to rewrite ===")
        return

    if not renames and not merges:
        print("\nNothing to rewrite.")
        return

    bak = backup(args.graph)
    print(f"\n[backup] {bak}")
    renamed, merged = apply_plan(gm, renames, merges)
    gm.save()
    print(f"[applied] renamed {renamed} edges, merged {merged} duplicates; "
          f"graph now {gm.node_count()} nodes, {gm.edge_count()} edges.")
    print(f"Restore with: cp {bak} {args.graph}")


if __name__ == "__main__":
    main()
