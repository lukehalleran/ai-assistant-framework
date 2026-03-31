#!/usr/bin/env python3
"""
Import Wikidata cache into the knowledge graph (Phase 4).

Loads data/wikidata_cache.json and:
  1. Creates entity nodes for each Wikidata item (source="wikidata")
  2. Creates edges for inter-entity Wikidata relations
  3. Runs entity resolution to find personal→wikidata bridges
  4. Reports bridge density for go/no-go gate

Usage:
    python scripts/import_wikidata_to_graph.py
    python scripts/import_wikidata_to_graph.py --dry-run
    python scripts/import_wikidata_to_graph.py --cache data/wikidata_cache.json
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_utils import get_logger

logger = get_logger("import_wikidata")


def _slugify(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s


def import_wikidata(cache_path: str, dry_run: bool = False):
    from config.app_config import (
        KNOWLEDGE_GRAPH_PERSIST_PATH,
        KNOWLEDGE_GRAPH_ALIASES_PATH,
        WIKIDATA_EMBEDDING_MATCH_THRESHOLD,
        GRAPH_WALK_MIN_BRIDGE_EDGES,
    )
    from memory.graph_memory import GraphMemory
    from memory.graph_models import GraphEdge, GraphNode
    from memory.entity_resolver import EntityResolver

    print("=" * 70)
    print("WIKIDATA IMPORT TO KNOWLEDGE GRAPH")
    print(f"Cache: {cache_path}")
    print(f"Dry run: {dry_run}")
    print("=" * 70)

    # Load cache
    print("\nLoading Wikidata cache...")
    with open(cache_path, "r") as f:
        cache = json.load(f)

    entities = cache.get("entities", {})
    relations = cache.get("relations", [])
    print(f"  Entities in cache: {len(entities)}")
    print(f"  Relations in cache: {len(relations)}")

    # Load graph
    print("\nLoading knowledge graph...")
    graph = GraphMemory(persist_path=KNOWLEDGE_GRAPH_PERSIST_PATH)
    resolver = EntityResolver(graph, aliases_path=KNOWLEDGE_GRAPH_ALIASES_PATH)

    initial_nodes = graph.node_count()
    initial_edges = graph.edge_count()
    print(f"  Existing nodes: {initial_nodes}")
    print(f"  Existing edges: {initial_edges}")

    if dry_run:
        print("\n--- DRY RUN: showing what would happen ---")

    # Phase 1: Import entity nodes (bulk mode suppresses per-50 auto-saves)
    print(f"\nPhase 1: Importing {len(entities)} entity nodes...")
    t0 = time.time()
    nodes_added = 0
    nodes_skipped = 0
    now = datetime.now()

    bulk_ctx = graph.bulk_import()
    bulk_ctx.__enter__()

    for qid, ent in entities.items():
        label = ent.get("label", "")
        if not label:
            continue

        slug = _slugify(label)

        # Skip if already exists
        if graph.graph.has_node(slug):
            nodes_skipped += 1
            continue

        if not dry_run:
            node = GraphNode(
                entity_id=slug,
                display_name=label,
                entity_type="concept",
                aliases=[a.lower() for a in ent.get("aliases", [])[:10]],
                first_seen=now,
                last_seen=now,
                mention_count=0,
                metadata={
                    "source": "wikidata",
                    "wikidata_qid": qid,
                    "wikidata_description": ent.get("description", "")[:200],
                    "domain_category": ent.get("domain_category", ""),
                },
            )
            graph.add_entity(node)

            # Register aliases
            for alias in ent.get("aliases", [])[:10]:
                resolver.learn_alias(alias.lower(), slug, context="wikidata_import")

        nodes_added += 1

    elapsed = time.time() - t0
    print(f"  Added: {nodes_added}, Skipped (existing): {nodes_skipped} ({elapsed:.1f}s)")

    # Phase 2: Import inter-entity relations
    print(f"\nPhase 2: Importing {len(relations)} inter-entity relations...")
    t0 = time.time()
    edges_added = 0
    edges_skipped = 0

    # Build QID → slug mapping
    qid_to_slug = {}
    for qid, ent in entities.items():
        qid_to_slug[qid] = _slugify(ent.get("label", ""))

    for rel in relations:
        src_slug = qid_to_slug.get(rel["source_qid"], "")
        tgt_slug = qid_to_slug.get(rel["target_qid"], "")

        if not src_slug or not tgt_slug:
            edges_skipped += 1
            continue

        # Both nodes must exist in graph
        if not dry_run:
            if not graph.graph.has_node(src_slug) or not graph.graph.has_node(tgt_slug):
                edges_skipped += 1
                continue

            edge = GraphEdge(
                source_id=src_slug,
                relation=rel["relation_label"],
                target_id=tgt_slug,
                weight=1.0,
                truth_score=0.95,
                first_seen=now,
                last_seen=now,
                metadata={"source": "wikidata", "property_id": rel["property_id"]},
            )
            graph.add_relation(edge, fact_id="")

        edges_added += 1

    elapsed = time.time() - t0
    print(f"  Added: {edges_added}, Skipped: {edges_skipped} ({elapsed:.1f}s)")

    # Phase 3: Entity resolution — find personal→wikidata bridges
    print(f"\nPhase 3: Finding personal→wikidata bridges...")
    print(f"  Embedding threshold: {WIKIDATA_EMBEDDING_MATCH_THRESHOLD}")
    t0 = time.time()

    from knowledge.wikidata_resolver import WikidataEntityMapper

    mapper = WikidataEntityMapper(graph, resolver, entities)
    bridge_matches = mapper.map_personal_to_wikidata(
        embedding_threshold=WIKIDATA_EMBEDDING_MATCH_THRESHOLD,
    )

    # Create bridge edges
    bridges_created = 0
    for match in bridge_matches:
        personal_id = match["personal_id"]
        wikidata_slug = _slugify(match["wikidata_label"])
        confidence = match.get("confidence", 0.7)

        if not dry_run and graph.graph.has_node(wikidata_slug):
            # Bidirectional bridge edge
            edge = GraphEdge(
                source_id=personal_id,
                relation="related_to",
                target_id=wikidata_slug,
                weight=confidence,
                truth_score=confidence,
                first_seen=now,
                last_seen=now,
                metadata={
                    "source": "wikidata_bridge",
                    "bridge_confidence": confidence,
                    "match_type": match["match_type"],
                },
            )
            graph.add_relation(edge, fact_id="")

            # Register alias cross-reference
            resolver.learn_alias(
                match["wikidata_label"].lower(),
                personal_id,
                context="wikidata_bridge",
            )

            bridges_created += 1

    elapsed = time.time() - t0
    print(f"  Matches found: {len(bridge_matches)}")
    print(f"  Bridge edges created: {bridges_created}")
    print(f"  ({elapsed:.1f}s)")

    # Show all matches
    if bridge_matches:
        print(f"\n  Bridge details:")
        for m in bridge_matches:
            sim = f" sim={m['similarity']}" if 'similarity' in m else ""
            print(f"    {m['personal_display']:25s} → {m['wikidata_label']:30s} "
                  f"[{m['match_type']}] conf={m['confidence']}{sim} ({m['domain']})")

    # Phase 4: Save (exit bulk mode → single save)
    if not dry_run:
        print("\nSaving graph...")
        bulk_ctx.__exit__(None, None, None)
        resolver.save_external_aliases()
    else:
        graph._bulk_mode = False

    # Summary
    final_nodes = graph.node_count() if not dry_run else initial_nodes + nodes_added
    final_edges = graph.edge_count() if not dry_run else initial_edges + edges_added + bridges_created
    bridge_count = graph.count_bridge_edges() if not dry_run else bridges_created

    print(f"\n{'='*70}")
    print(f"IMPORT COMPLETE")
    print(f"{'='*70}")
    print(f"  Nodes: {initial_nodes} → {final_nodes} (+{nodes_added})")
    print(f"  Edges: {initial_edges} → {final_edges} (+{edges_added + bridges_created})")
    print(f"  Bridge edges: {bridge_count}")

    print(f"\n--- GO/NO-GO GATE ---")
    if bridge_count >= GRAPH_WALK_MIN_BRIDGE_EDGES:
        print(f"  PASS: {bridge_count} bridges >= {GRAPH_WALK_MIN_BRIDGE_EDGES} threshold")
        print(f"  Graph walk generator will activate at shutdown.")
    else:
        print(f"  BELOW THRESHOLD: {bridge_count} bridges < {GRAPH_WALK_MIN_BRIDGE_EDGES}")
        print(f"  Walk generator will use fallback (existing SynthesisGenerator).")
        print(f"  Wiki enrichment will grow bridges organically over time.")


def main():
    parser = argparse.ArgumentParser(description="Import Wikidata cache into knowledge graph")
    parser.add_argument("--cache", type=str, default="data/wikidata_cache.json",
                       help="Path to wikidata_cache.json")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would happen without modifying the graph")
    args = parser.parse_args()

    if not Path(args.cache).exists():
        print(f"ERROR: Cache file not found: {args.cache}")
        print(f"Run 'python scripts/extract_wikidata.py' first.")
        sys.exit(1)

    import_wikidata(cache_path=args.cache, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
