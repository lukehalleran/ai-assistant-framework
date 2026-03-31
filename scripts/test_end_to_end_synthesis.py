#!/usr/bin/env python3
"""
End-to-end synthesis dreaming test (Phase 3.1).

Runs the full generator → filter → storage pipeline against real data:
- Facts from ChromaDB (personal facts + test facts)
- Wiki from FAISS (40M vectors)
- Real LLM for bridge articulation + coherence judging

Usage:
    python scripts/test_end_to_end_synthesis.py
    python scripts/test_end_to_end_synthesis.py --candidates 15
    python scripts/test_end_to_end_synthesis.py --candidates 15 --runs 3
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_utils import get_logger

logger = get_logger("test_e2e_synthesis")


async def run_dreaming(candidate_count: int = 15, num_runs: int = 1):
    from config.app_config import (
        CHROMA_PATH,
        KNOWLEDGE_GRAPH_PERSIST_PATH,
        KNOWLEDGE_GRAPH_ALIASES_PATH,
    )
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    from memory.graph_memory import GraphMemory
    from memory.entity_resolver import EntityResolver
    from knowledge.synthesis_generator import SynthesisGenerator
    from knowledge.synthesis_filter import SynthesisFilter
    from memory.synthesis_memory import SynthesisMemory
    from models.model_manager import ModelManager

    print("=" * 70)
    print("END-TO-END SYNTHESIS DREAMING TEST")
    print("=" * 70)

    # Init
    print("\nInitializing...")
    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    model_manager = ModelManager()
    graph = GraphMemory(persist_path=KNOWLEDGE_GRAPH_PERSIST_PATH)
    resolver = EntityResolver(graph, aliases_path=KNOWLEDGE_GRAPH_ALIASES_PATH)

    # Verify prerequisites
    from knowledge.semantic_search import get_index
    faiss_idx = get_index()
    faiss_idx.load()
    if not faiss_idx.loaded:
        print("ERROR: FAISS wiki index not available.")
        sys.exit(1)

    facts_coll = store.collections.get("facts")
    facts_count = facts_coll.count() if facts_coll else 0

    print(f"  FAISS wiki vectors: {faiss_idx._total_rows:,}")
    print(f"  Facts in ChromaDB: {facts_count:,}")
    print(f"  Graph nodes: {graph.node_count()}")
    print(f"  Graph edges: {graph.edge_count()}")
    print(f"  Candidates per run: {candidate_count}")
    print(f"  Runs: {num_runs}")

    if graph.node_count() < 20:
        print("WARNING: Graph has < 20 nodes. Generator sparsity guard may skip.")

    # Run
    all_run_results = []
    total_accepted = 0
    total_generated = 0
    total_by_stage = {}

    for run_idx in range(num_runs):
        print(f"\n{'='*70}")
        print(f"RUN {run_idx + 1}/{num_runs}")
        print(f"{'='*70}")

        t0 = time.time()

        # Generator
        generator = SynthesisGenerator(
            chroma_store=store,
            model_manager=model_manager,
            graph_memory=graph,
            entity_resolver=resolver,
        )

        print(f"\nGenerating {candidate_count} candidates...")
        candidates = await generator.generate_candidates(count=candidate_count)
        gen_elapsed = time.time() - t0

        if not candidates:
            print("  No candidates generated. Check facts collection and graph sparsity.")
            all_run_results.append({"run": run_idx + 1, "generated": 0, "accepted": 0})
            continue

        print(f"  Generated {len(candidates)} candidates in {gen_elapsed:.1f}s")
        total_generated += len(candidates)

        # Show candidates
        for i, c in enumerate(candidates):
            print(f"  [{i+1}] {c.concept_a} ↔ {c.concept_b} (dist={c.endpoint_distance:.2f})")
            print(f"      domains: {c.source_domains}")
            print(f"      claim: {c.connection_claim[:120]}...")

        # Filter
        print(f"\nFiltering through 8-stage pipeline...")
        synthesis_memory = SynthesisMemory(store)
        filter_pipeline = SynthesisFilter(
            chroma_store=store,
            model_manager=model_manager,
            synthesis_memory=synthesis_memory,
        )

        t1 = time.time()
        batch_results = await filter_pipeline.process_batch(candidates)
        filter_elapsed = time.time() - t1

        accepted = batch_results.get("accepted", 0)
        rejected = batch_results.get("rejected", 0)
        breakdown = batch_results.get("rejection_breakdown", {})
        avg_times = batch_results.get("avg_stage_times_ms", {})
        total_accepted += accepted

        for stage, count in breakdown.items():
            total_by_stage[stage] = total_by_stage.get(stage, 0) + count

        print(f"\n  Results: {accepted} accepted, {rejected} rejected ({filter_elapsed:.1f}s)")
        print(f"  Rejection breakdown: {breakdown}")
        print(f"  Avg stage times: {', '.join(f'{k}={v:.0f}ms' for k, v in avg_times.items())}")

        # Show accepted insights
        accepted_results = batch_results.get("accepted_results", [])
        if accepted_results:
            print(f"\n  --- ACCEPTED INSIGHTS ---")
            for r in accepted_results:
                c = r.candidate
                print(f"  {c.concept_a} ↔ {c.concept_b}")
                print(f"    claim: {c.connection_claim[:150]}")
                print(f"    composite: {r.composite_score:.3f}, coherence: {r.coherence_level.name}")
                print(f"    novelty_ext: {r.novelty_score_external:.3f}, cooccurrence: {r.cooccurrence_similarity:.3f}")
                print()
        else:
            print(f"\n  No insights accepted this run.")

        all_run_results.append({
            "run": run_idx + 1,
            "generated": len(candidates),
            "accepted": accepted,
            "rejected": rejected,
            "breakdown": breakdown,
            "elapsed_gen": round(gen_elapsed, 1),
            "elapsed_filter": round(filter_elapsed, 1),
            "accepted_insights": [
                {
                    "concept_a": r.candidate.concept_a,
                    "concept_b": r.candidate.concept_b,
                    "claim": r.candidate.connection_claim,
                    "composite": round(r.composite_score, 3),
                    "coherence": r.coherence_level.name if r.coherence_level else "N/A",
                }
                for r in accepted_results
            ],
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY ({num_runs} run{'s' if num_runs > 1 else ''})")
    print(f"{'='*70}")
    print(f"  Total candidates generated: {total_generated}")
    print(f"  Total accepted: {total_accepted}")
    acceptance_rate = total_accepted / total_generated * 100 if total_generated > 0 else 0
    print(f"  Acceptance rate: {acceptance_rate:.0f}%")
    print(f"  Rejection by stage: {total_by_stage}")

    # Evaluate against targets
    print(f"\n--- PHASE 3.1 CRITERIA ---")
    print(f"  Candidates generated >= 10: {'PASS' if total_generated >= 10 else 'FAIL'} ({total_generated})")
    rate_ok = 10 <= acceptance_rate <= 30
    print(f"  Acceptance rate 10-30%: {'PASS' if rate_ok else 'CHECK'} ({acceptance_rate:.0f}%)")
    if acceptance_rate == 0:
        print(f"  WARNING: 0% acceptance — filter may be too strict or sampling too narrow")
    elif acceptance_rate > 60:
        print(f"  WARNING: >{60}% acceptance — filter may be too loose")
    print(f"  At least 1 interesting insight: REVIEW ABOVE")

    # Save results
    output_path = Path("data/synthesis_e2e_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_run_results, f, indent=2, default=str)
    print(f"\n  Detailed results: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="End-to-end synthesis dreaming test")
    parser.add_argument("--candidates", type=int, default=15, help="Candidates per run (default 15)")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (default 1)")
    args = parser.parse_args()

    asyncio.run(run_dreaming(candidate_count=args.candidates, num_runs=args.runs))


if __name__ == "__main__":
    main()
