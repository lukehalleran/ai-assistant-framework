#!/usr/bin/env python3
"""
Generate synthetic personal facts for synthesis pipeline calibration (Phase 1.3).

Populates the `facts` ChromaDB collection and knowledge graph with ~80
realistic personal facts spanning multiple domains. Includes intentional
noise: ambiguous entities, near-duplicates, hedged confidence, and sub-threshold
facts that should be gated.

Usage:
    python scripts/generate_test_facts.py              # populate facts + graph
    python scripts/generate_test_facts.py --dry-run    # show what would be added
    python scripts/generate_test_facts.py --clear      # wipe test facts first

Requires: ChromaDB initialized, knowledge graph paths writable.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_utils import get_logger

logger = get_logger("generate_test_facts")


# ---------------------------------------------------------------------------
# Fact definitions: (subject, predicate, object, confidence, domain_hint)
# ---------------------------------------------------------------------------

# Clean facts — core personal profile
CLEAN_FACTS = [
    # Career / work
    ("user", "works_at", "Acme Brewery", 0.95, "career"),
    ("user", "job_title", "head brewer", 0.90, "career"),
    ("user", "career_years", "6 years in brewing", 0.85, "career"),
    ("user", "previous_job", "homebrewing instructor", 0.80, "career"),
    ("user", "work_goal", "open own brewery", 0.75, "career"),

    # Relationships
    ("user", "has_brother", "Auggie", 0.95, "relationships"),
    ("user", "has_cat", "Paczki", 0.95, "relationships"),
    ("user", "mother_name", "Mom", 0.85, "relationships"),
    ("user", "relationship_status", "dating Sarah", 0.80, "relationships"),
    ("Auggie", "studies", "computer science", 0.85, "relationships"),
    ("Auggie", "hobby", "rock climbing", 0.80, "relationships"),
    ("Sarah", "works_as", "veterinarian", 0.80, "relationships"),

    # Fitness
    ("user", "trains", "weightlifting 4x/week", 0.90, "fitness"),
    ("user", "bench_press_max", "225 lbs", 0.85, "fitness"),
    ("user", "squat_max", "315 lbs", 0.85, "fitness"),
    ("user", "running", "trains for half marathon", 0.80, "fitness"),
    ("user", "fitness_goal", "run sub-1:45 half marathon", 0.75, "fitness"),

    # Health
    ("user", "takes_medication", "Adderall for ADHD", 0.90, "health"),
    ("user", "allergy", "seasonal pollen", 0.85, "health"),
    ("user", "sleep_hours", "7 hours average", 0.80, "health"),
    ("user", "diet", "high protein, moderate carbs", 0.75, "health"),

    # Education
    ("user", "studying", "actuarial science", 0.90, "education"),
    ("user", "learning", "statistics and probability", 0.85, "education"),
    ("user", "completed_course", "linear algebra", 0.80, "education"),
    ("user", "interested_in", "machine learning", 0.75, "education"),

    # Geography
    ("user", "lives_in", "Portland", 0.95, "geography"),
    ("user", "hometown", "Eugene, Oregon", 0.85, "geography"),
    ("user", "wants_to_visit", "Japan", 0.70, "geography"),

    # Hobbies
    ("user", "hobby", "board games", 0.90, "hobbies"),
    ("user", "hobby", "sourdough baking", 0.90, "hobbies"),
    ("user", "plays", "Dungeons and Dragons", 0.85, "hobbies"),
    ("user", "reads", "science fiction", 0.80, "hobbies"),
    ("user", "favorite_game", "Terraforming Mars", 0.75, "hobbies"),

    # Projects
    ("user", "building", "Daemon", 0.95, "projects"),
    ("user", "project_language", "Python", 0.90, "projects"),
    ("user", "project_goal", "conversational RAG system", 0.85, "projects"),
    ("Daemon", "uses", "ChromaDB", 0.85, "projects"),
    ("Daemon", "uses", "FAISS", 0.85, "projects"),
    ("Daemon", "feature", "knowledge graph", 0.80, "projects"),

    # Beliefs / values
    ("user", "values", "intellectual honesty", 0.80, "values"),
    ("user", "values", "continuous learning", 0.75, "values"),

    # Food / drink
    ("user", "favorite_beer", "Belgian tripel", 0.85, "food"),
    ("user", "favorite_food", "ramen", 0.80, "food"),
    ("user", "drinks", "black coffee daily", 0.85, "food"),
    ("Paczki", "named_after", "Polish donut", 0.90, "food"),
]

# Intentional noise — tests entity resolution, confidence gating, dedup
NOISY_FACTS = [
    # Ambiguous entity: "August" vs "Auggie" (entity resolver test)
    ("August", "mentioned", "he might visit next month", 0.60, "relationships"),

    # Hedged, no explicit name (should still link to brother context)
    ("user", "brother_activity", "also runs sometimes I think", 0.55, "relationships"),

    # Sub-threshold confidence — should be gated by 0.50 threshold
    ("user", "maybe_allergic_to", "shellfish", 0.45, "health"),
    ("user", "might_have_visited", "Barcelona once", 0.40, "geography"),

    # Near-duplicates (dedup test)
    ("user", "hobby", "running", 0.80, "fitness"),  # overlaps with "trains for half marathon"
    ("user", "enjoys", "jogging", 0.75, "fitness"),  # synonym of running

    # Ambiguous domain: career or geography?
    ("user", "commutes_to", "the brewery downtown", 0.75, "career"),

    # Vague, low-value
    ("user", "read_about", "stoicism", 0.60, "education"),
    ("user", "heard_of", "some philosophy podcast", 0.55, "hobbies"),

    # Entity fact (non-user subject)
    ("Paczki", "breed", "domestic shorthair", 0.85, "relationships"),
    ("Paczki", "age", "3 years old", 0.80, "relationships"),
    ("Portland", "known_for", "craft beer scene", 0.75, "geography"),
]


def build_fact_text(subj, pred, obj):
    """Build the pipe-delimited fact string for ChromaDB."""
    return f"{subj} | {pred} | {obj}"


def run(dry_run=False, clear=False):
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    from memory.graph_memory import GraphMemory
    from memory.entity_resolver import EntityResolver
    from memory.graph_models import GraphNode, GraphEdge
    from config.app_config import (
        CHROMA_PATH,
        KNOWLEDGE_GRAPH_PERSIST_PATH,
        KNOWLEDGE_GRAPH_ALIASES_PATH,
        KNOWLEDGE_GRAPH_MIN_CONFIDENCE,
    )

    all_facts = CLEAN_FACTS + NOISY_FACTS

    print(f"Facts to add: {len(CLEAN_FACTS)} clean + {len(NOISY_FACTS)} noisy = {len(all_facts)} total")
    print(f"Dry run: {dry_run}")

    if dry_run:
        print("\n--- FACTS ---")
        for subj, pred, obj, conf, domain in all_facts:
            gate = "GATED" if conf < 0.50 else "ok"
            print(f"  [{gate:5s}] ({conf:.2f}) {subj} | {pred} | {obj}  [{domain}]")

        sub_threshold = [f for f in all_facts if f[3] < 0.50]
        print(f"\nSub-threshold (< 0.50, should be gated): {len(sub_threshold)}")
        domains = set(f[4] for f in all_facts)
        print(f"Domains represented: {len(domains)} — {sorted(domains)}")

        # Count expected graph entities
        entities = set()
        for subj, pred, obj, conf, _ in all_facts:
            if conf >= 0.50:
                entities.add(subj.lower())
                if len(obj.split()) < 4:  # rough graph-worthiness check
                    entities.add(obj.lower())
        print(f"Expected graph entities (rough): ~{len(entities)}")
        return

    # Init stores
    print("\nInitializing ChromaDB...")
    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)

    if clear:
        print("Clearing existing facts collection...")
        coll = store.collections.get("facts")
        if coll and coll.count() > 0:
            # Get all IDs and delete
            all_ids = coll.get()["ids"]
            if all_ids:
                coll.delete(ids=all_ids)
                print(f"  Deleted {len(all_ids)} existing facts")

    print("Initializing knowledge graph...")
    graph = GraphMemory(persist_path=KNOWLEDGE_GRAPH_PERSIST_PATH)
    resolver = EntityResolver(graph, aliases_path=KNOWLEDGE_GRAPH_ALIASES_PATH)

    # Add facts
    added = 0
    gated = 0
    duped = 0

    for subj, pred, obj, conf, domain in all_facts:
        fact_text = build_fact_text(subj, pred, obj)

        # Build metadata
        metadata = {
            "source": "test_calibration",
            "confidence": conf,
            "subject": subj,
            "predicate": pred,
            "object": obj,
            "fact_scope": "user" if subj.lower() == "user" else "entity",
            "domain_hint": domain,
        }

        # Add to ChromaDB
        try:
            doc_id = store.add_fact(fact_text, metadata)
            if doc_id is None:
                duped += 1
                print(f"  DUP:   {fact_text}")
                continue
            added += 1
        except Exception as e:
            print(f"  ERROR: {fact_text} — {e}")
            continue

        # Ingest to graph (respecting confidence gate)
        if conf < KNOWLEDGE_GRAPH_MIN_CONFIDENCE:
            gated += 1
            print(f"  GATED: {fact_text} (conf={conf:.2f} < {KNOWLEDGE_GRAPH_MIN_CONFIDENCE})")
            continue

        # Graph ingestion (simplified version of _ingest_fact_to_graph)
        try:
            from memory.entity_resolver import normalize_relation
            canon_rel = normalize_relation(pred)

            subj_display = subj if subj.lower() != "user" else "User"
            subj_type = "person" if subj.lower() == "user" else "other"
            subj_id = resolver.resolve_or_create(subj, entity_type=subj_type, display_name=subj_display)

            # Only create object nodes for short, entity-like objects
            obj_words = obj.split()
            if len(obj_words) < 4 and not any(c.isdigit() for c in obj):
                obj_id = resolver.resolve_or_create(obj, display_name=obj)
                graph.add_relation(
                    GraphEdge(source_id=subj_id, relation=canon_rel, target_id=obj_id),
                    fact_id=doc_id,
                )
            else:
                # Store as node metadata
                node = graph.get_entity(subj_id)
                if node:
                    graph.add_entity(GraphNode(
                        entity_id=subj_id,
                        display_name=node.display_name,
                        entity_type=node.entity_type,
                        metadata={canon_rel: obj},
                    ))
        except Exception as e:
            print(f"  GRAPH ERR: {fact_text} — {e}")

    # Save graph
    graph.save()
    resolver_aliases = resolver.graph._alias_index
    print(f"\nResults:")
    print(f"  Added to ChromaDB: {added}")
    print(f"  Duplicates skipped: {duped}")
    print(f"  Sub-threshold gated: {gated}")
    print(f"  Graph nodes: {graph.node_count()}")
    print(f"  Graph edges: {graph.edge_count()}")
    print(f"  Alias index entries: {len(resolver_aliases)}")

    # Verify domain coverage
    facts_coll = store.collections.get("facts")
    if facts_coll:
        print(f"  Facts collection total: {facts_coll.count()}")

    # Check success criteria
    print(f"\n--- SUCCESS CRITERIA ---")
    nodes_ok = graph.node_count() >= 30
    edges_ok = graph.edge_count() >= 25
    print(f"  Graph nodes >= 30: {'PASS' if nodes_ok else 'FAIL'} ({graph.node_count()})")
    print(f"  Graph edges >= 25: {'PASS' if edges_ok else 'FAIL'} ({graph.edge_count()})")
    print(f"  Sub-threshold gated: {'PASS' if gated >= 1 else 'FAIL'} ({gated})")
    print(f"  Domains: {len(set(f[4] for f in all_facts))}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic facts for calibration")
    parser.add_argument("--dry-run", action="store_true", help="Show facts without adding")
    parser.add_argument("--clear", action="store_true", help="Clear existing facts first")
    args = parser.parse_args()

    run(dry_run=args.dry_run, clear=args.clear)


if __name__ == "__main__":
    main()
