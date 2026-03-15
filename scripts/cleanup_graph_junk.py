#!/usr/bin/env python3
"""
Cleanup junk nodes from the knowledge graph.

Phrases (4+ words) and generic words are migrated to subject node
metadata (preserving the info), then their nodes and edges are removed.

Usage:
    python scripts/cleanup_graph_junk.py              # dry-run (preview)
    python scripts/cleanup_graph_junk.py --execute     # actually delete
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GRAPH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "knowledge_graph.json",
)


def is_junk_node(node_id: str) -> bool:
    """Matches MemoryStorage._is_graph_worthy_object (inverted).

    4+ word phrases and generic words are junk nodes.
    """
    o = node_id.strip().lower()

    if len(o) < 2:
        return True

    # 4+ words (underscores as separators) = descriptive phrase
    if len(o.replace("_", " ").split()) >= 4:
        return True

    # Generic/meaningless
    generic = {
        "a_lot", "some", "none", "true", "false",
        "yes", "no", "many", "few", "lots", "not_sure", "unknown",
        "graph",
    }
    if o in generic:
        return True

    return False


def cleanup(execute: bool = False):
    if not os.path.exists(GRAPH_PATH):
        print(f"Graph file not found: {GRAPH_PATH}")
        return

    with open(GRAPH_PATH, "r") as f:
        data = json.load(f)

    nodes = data.get("nodes", {})
    edges = data.get("edges", [])

    # Identify junk nodes
    junk_ids = set()
    for nid in nodes:
        if is_junk_node(nid):
            junk_ids.add(nid)

    # Migrate junk target values into source node metadata, then remove
    migrated = 0
    removed_edges = 0
    clean_edges = []

    for edge in edges:
        src = edge.get("source_id", "")
        tgt = edge.get("target_id", "")
        rel = edge.get("relation", "")

        if tgt in junk_ids:
            # Migrate: store the display_name as metadata on the source node
            tgt_display = nodes.get(tgt, {}).get("display_name", tgt)
            if src in nodes:
                meta = nodes[src].get("metadata", {}) or {}
                meta[rel] = tgt_display
                nodes[src]["metadata"] = meta
                migrated += 1
                if not execute:
                    print(f"  [MIGRATE] {src}.metadata[\"{rel}\"] = \"{tgt_display}\"")
            removed_edges += 1
        elif src in junk_ids:
            removed_edges += 1
            if not execute:
                print(f"  [REMOVE EDGE] {src} --{rel}--> {tgt}")
        else:
            clean_edges.append(edge)

    print(f"\n{'='*60}")
    print(f"Junk nodes to remove ({len(junk_ids)}):")
    for nid in sorted(junk_ids):
        display = nodes[nid].get("display_name", nid)
        print(f"  - {nid:55s}  ({display})")

    print(f"\n{'='*60}")
    print(f"{'PREVIEW' if not execute else 'EXECUTED'}:")
    print(f"  Nodes before:  {len(nodes)}")
    print(f"  Junk nodes:    {len(junk_ids)}")
    print(f"  Nodes after:   {len(nodes) - len(junk_ids)}")
    print(f"  Edges before:  {len(edges)}")
    print(f"  Edges removed: {removed_edges}")
    print(f"  Edges after:   {len(clean_edges)}")
    print(f"  Migrated to metadata: {migrated}")

    if execute:
        backup_path = GRAPH_PATH + ".bak"
        shutil.copy2(GRAPH_PATH, backup_path)
        print(f"\n  Backup saved: {backup_path}")

        for nid in junk_ids:
            del nodes[nid]
        data["edges"] = clean_edges

        with open(GRAPH_PATH, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Graph saved: {GRAPH_PATH}")
    else:
        print(f"\n  Run with --execute to apply changes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean junk nodes from knowledge graph")
    parser.add_argument("--execute", action="store_true", help="Apply changes (default is dry-run preview)")
    args = parser.parse_args()
    cleanup(execute=args.execute)
