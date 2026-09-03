#!/usr/bin/env python3
"""
Generate synthetic personal facts for synthesis pipeline calibration (Phase 1.3).

Populates the `facts` ChromaDB collection and knowledge graph with ~80
realistic personal facts spanning multiple domains. Includes intentional
noise: ambiguous entities, near-duplicates, hedged confidence, and sub-threshold
facts that should be gated.

Usage:
    python scripts/generate_test_facts.py --dry-run                       # show what would be added
    python scripts/generate_test_facts.py --sandbox-dir /tmp/daemon_cal   # populate a SANDBOX
    python scripts/generate_test_facts.py --sandbox-dir DIR --clear       # wipe the sandbox facts first

SANDBOX ONLY (2026-09-02): this script used to write straight into the live
CHROMA_PATH / knowledge graph — 48 synthetic `source=test_calibration` facts
(D&D, half-marathon running, a brewery job, a cat named Mochi…) were found in
the OWNER's live facts collection and graph, indistinguishable from real memory
at retrieval. It now REQUIRES --sandbox-dir and refuses any resolved path that
coincides with the live stores. Live cleanup: scripts/purge_calibration_facts.py.
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
    ("user", "has_brother", "Sam", 0.95, "relationships"),
    ("user", "has_cat", "Mochi", 0.95, "relationships"),
    ("user", "mother_name", "Mom", 0.85, "relationships"),
    ("user", "relationship_status", "dating Sarah", 0.80, "relationships"),
    ("Sam", "studies", "computer science", 0.85, "relationships"),
    ("Sam", "hobby", "rock climbing", 0.80, "relationships"),
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
    ("Mochi", "named_after", "Polish donut", 0.90, "food"),
]

# Intentional noise — tests entity resolution, confidence gating, dedup
NOISY_FACTS = [
    # Ambiguous entity: "August" vs "Sam" (entity resolver test)
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
    ("Mochi", "breed", "domestic shorthair", 0.85, "relationships"),
    ("Mochi", "age", "3 years old", 0.80, "relationships"),
    ("Portland", "known_for", "craft beer scene", 0.75, "geography"),
]


def build_fact_text(subj, pred, obj):
    """Build the pipe-delimited fact string for ChromaDB."""
    return f"{subj} | {pred} | {obj}"


class LiveStoreRefused(RuntimeError):
    """Raised when a resolved sandbox path coincides with a live store."""


def resolve_sandbox_paths(sandbox_dir):
    """Chroma dir + graph/alias JSON paths, all UNDER the sandbox directory."""
    if not sandbox_dir:
        raise LiveStoreRefused(
            "--sandbox-dir is required: calibration facts never go into the live stores"
        )
    root = Path(sandbox_dir).expanduser().resolve()
    return {
        "chroma": root / "chroma",
        "graph": root / "knowledge_graph.json",
        "aliases": root / "entity_aliases.json",
    }


def refuse_live_paths(paths, *, live_chroma=None, live_graph=None, live_aliases=None):
    """Abort when any sandbox path resolves onto a live store path."""
    if live_chroma is None or live_graph is None or live_aliases is None:
        from config.app_config import (
            CHROMA_PATH,
            KNOWLEDGE_GRAPH_ALIASES_PATH,
            KNOWLEDGE_GRAPH_PERSIST_PATH,
        )
        live_chroma = CHROMA_PATH if live_chroma is None else live_chroma
        live_graph = KNOWLEDGE_GRAPH_PERSIST_PATH if live_graph is None else live_graph
        live_aliases = KNOWLEDGE_GRAPH_ALIASES_PATH if live_aliases is None else live_aliases
    live = {
        "chroma": Path(str(live_chroma)).expanduser().resolve(),
        "graph": Path(str(live_graph)).expanduser().resolve(),
        "aliases": Path(str(live_aliases)).expanduser().resolve(),
    }
    for key, path in paths.items():
        resolved = Path(path).expanduser().resolve()
        for live_key, live_path in live.items():
            if resolved == live_path or live_path in resolved.parents or resolved in live_path.parents:
                raise LiveStoreRefused(
                    f"refusing: sandbox {key} path {resolved} coincides with live {live_key} store {live_path}"
                )
    return paths


def run(dry_run=False, clear=False, sandbox_dir=None):
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

        entities = set()
        for subj, pred, obj, conf, _ in all_facts:
            if conf >= 0.50:
                entities.add(subj.lower())
                if len(obj.split()) < 4:
                    entities.add(obj.lower())
        print(f"Expected graph entities (rough): ~{len(entities)}")
        return

    paths = refuse_live_paths(resolve_sandbox_paths(sandbox_dir))
    paths["chroma"].mkdir(parents=True, exist_ok=True)

    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    from memory.graph_memory import GraphMemory
    from memory.entity_resolver import EntityResolver
    from memory.graph_models import GraphNode, GraphEdge
    from config.app_config import KNOWLEDGE_GRAPH_MIN_CONFIDENCE

    print(f"\nSandbox: {paths['chroma'].parent}")
    print("Initializing sandbox ChromaDB...")
    store = MultiCollectionChromaStore(persist_directory=str(paths["chroma"]))

    if clear:
        print("Clearing sandbox facts collection...")
        coll = store._get_collection("facts")
        if coll and coll.count() > 0:
            all_ids = coll.get()["ids"]
            if all_ids:
                coll.delete(ids=all_ids)
                print(f"  Deleted {len(all_ids)} sandbox facts")

    print("Initializing sandbox knowledge graph...")
    graph = GraphMemory(persist_path=str(paths["graph"]))
    resolver = EntityResolver(graph, aliases_path=str(paths["aliases"]))

    added = gated = duped = 0
    for subj, pred, obj, conf, domain in all_facts:
        fact_text = build_fact_text(subj, pred, obj)
        metadata = {
            "source": "test_calibration",
            "confidence": conf,
            "subject": subj,
            "predicate": pred,
            "object": obj,
            "fact_scope": "user" if subj.lower() == "user" else "entity",
            "domain_hint": domain,
        }
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

        if conf < KNOWLEDGE_GRAPH_MIN_CONFIDENCE:
            gated += 1
            print(f"  GATED: {fact_text} (conf={conf:.2f} < {KNOWLEDGE_GRAPH_MIN_CONFIDENCE})")
            continue

        try:
            from memory.entity_resolver import normalize_relation
            canon_rel = normalize_relation(pred)
            subj_display = subj if subj.lower() != "user" else "User"
            subj_type = "person" if subj.lower() == "user" else "other"
            subj_id = resolver.resolve_or_create(subj, entity_type=subj_type, display_name=subj_display)
            obj_words = obj.split()
            if len(obj_words) < 4 and not any(c.isdigit() for c in obj):
                obj_id = resolver.resolve_or_create(obj, display_name=obj)
                graph.add_relation(
                    GraphEdge(source_id=subj_id, relation=canon_rel, target_id=obj_id),
                    fact_id=doc_id,
                )
            else:
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

    graph.save()
    print(f"\nResults:")
    print(f"  Added to sandbox ChromaDB: {added}")
    print(f"  Duplicates skipped: {duped}")
    print(f"  Sub-threshold gated: {gated}")
    print(f"  Graph nodes: {graph.node_count()}")
    print(f"  Graph edges: {graph.edge_count()}")

    facts_coll = store._get_collection("facts")
    if facts_coll:
        print(f"  Sandbox facts collection total: {facts_coll.count()}")

    print(f"\n--- SUCCESS CRITERIA ---")
    nodes_ok = graph.node_count() >= 30
    edges_ok = graph.edge_count() >= 25
    print(f"  Graph nodes >= 30: {'PASS' if nodes_ok else 'FAIL'} ({graph.node_count()})")
    print(f"  Graph edges >= 25: {'PASS' if edges_ok else 'FAIL'} ({graph.edge_count()})")
    print(f"  Sub-threshold gated: {'PASS' if gated >= 1 else 'FAIL'} ({gated})")
    print(f"  Domains: {len(set(f[4] for f in all_facts))}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic facts for calibration (SANDBOX ONLY)")
    parser.add_argument("--dry-run", action="store_true", help="Show facts without adding")
    parser.add_argument("--clear", action="store_true", help="Clear the sandbox facts collection first")
    parser.add_argument("--sandbox-dir", default=None,
                        help="Directory to hold the sandbox chroma/ + knowledge_graph.json "
                             "(REQUIRED unless --dry-run; never a live store path)")
    args = parser.parse_args()

    try:
        run(dry_run=args.dry_run, clear=args.clear, sandbox_dir=args.sandbox_dir)
    except LiveStoreRefused as e:
        print(f"ABORT: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
