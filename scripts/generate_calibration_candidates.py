#!/usr/bin/env python3
"""
Generate synthesis candidates from the real wiki corpus for manual labeling.

Usage:
    python scripts/generate_calibration_candidates.py --count 50
    python scripts/generate_calibration_candidates.py --count 50 --output data/unlabeled_candidates.json

Runs the real SynthesisGenerator against the wiki_knowledge + facts collections,
then outputs candidates in calibration fixture format (minus the tier/expected_outcome
fields, which you fill in manually).

After labeling, merge into tests/fixtures/calibration_candidates.json.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_utils import get_logger

logger = get_logger("generate_calibration")

DEFAULT_OUTPUT = "data/unlabeled_candidates.json"


async def generate(count: int, output_path: str):
    """Generate candidates and write to JSON for manual labeling."""
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    from models.model_manager import ModelManager
    from knowledge.synthesis_generator import SynthesisGenerator

    # Init real stores
    print("Initializing ChromaDB store...")
    store = MultiCollectionChromaStore()

    # Check wiki_knowledge has content
    coll = store.collections.get("wiki_knowledge")
    if not coll or coll.count() == 0:
        print("ERROR: wiki_knowledge collection is empty. Load Wikipedia embeddings first.")
        print("  Run: python scripts/load_parquet_to_chromadb.py --parquet-dir <path>")
        sys.exit(1)
    print(f"  wiki_knowledge: {coll.count()} documents")

    facts_coll = store.collections.get("facts")
    facts_count = facts_coll.count() if facts_coll else 0
    print(f"  facts: {facts_count} documents")

    # Init model manager for bridge articulation
    print("Initializing model manager...")
    model_manager = ModelManager()

    # Try to load graph memory for distance computation
    graph_memory = None
    entity_resolver = None
    try:
        from memory.graph_memory import GraphMemory
        from memory.entity_resolver import EntityResolver
        graph_memory = GraphMemory()
        entity_resolver = EntityResolver()
        print(f"  Graph: {graph_memory.graph.number_of_nodes()} nodes, {graph_memory.graph.number_of_edges()} edges")
    except Exception as e:
        print(f"  Graph not available ({e}), using default distances")

    # Init generator
    generator = SynthesisGenerator(
        chroma_store=store,
        model_manager=model_manager,
        graph_memory=graph_memory,
        entity_resolver=entity_resolver,
    )

    # Generate candidates
    print(f"\nGenerating {count} candidates...")
    candidates = await generator.generate_candidates(count=count)
    print(f"  Generated {len(candidates)} candidates")

    if not candidates:
        print("No candidates generated. Check wiki_knowledge content and model config.")
        sys.exit(1)

    # Convert to labeling format
    unlabeled = []
    for c in candidates:
        unlabeled.append({
            "tier": "UNLABELED",
            "expected_outcome": "UNLABELED",
            "concept_a": c.concept_a,
            "concept_b": c.concept_b,
            "connection_claim": c.connection_claim,
            "source_domains": sorted(c.source_domains),
            "endpoint_distance": round(c.endpoint_distance, 3),
            "notes": "",
        })

    # Write output
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "_description": (
            f"Unlabeled synthesis candidates generated {datetime.now().isoformat()[:19]}. "
            f"Label each with tier and expected_outcome, then merge into "
            f"tests/fixtures/calibration_candidates.json."
        ),
        "_labeling_guide": {
            "tiers": {
                "trivial": "Well-known connection, should be rejected via novelty",
                "noise": "Forced/incoherent metaphor, should be rejected via coherence",
                "noise_borderline": "Plausible-sounding but scientifically wrong",
                "interesting_known": "Genuinely insightful but already published",
                "interesting_novel": "Non-obvious AND new — should be accepted",
                "boundary": "Hard to call — use for diagnostic only",
            },
            "expected_outcome": "accepted | rejected",
        },
        "candidates": unlabeled,
    }

    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote {len(unlabeled)} candidates to {output}")
    print(f"\nNext steps:")
    print(f"  1. Open {output}")
    print(f"  2. For each candidate, set 'tier' and 'expected_outcome'")
    print(f"  3. Add 'notes' explaining your reasoning")
    print(f"  4. Merge labeled candidates into tests/fixtures/calibration_candidates.json")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthesis candidates from real corpus for manual labeling"
    )
    parser.add_argument(
        "--count", type=int, default=50,
        help="Number of candidates to generate (default: 50)",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    asyncio.run(generate(args.count, args.output))


if __name__ == "__main__":
    main()
