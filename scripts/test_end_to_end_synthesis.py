#!/usr/bin/env python3
"""
End-to-end synthesis dreaming test.

Runs all three generators head-to-head with independent quotas:
- RetrievalSynthesisGenerator (structural query → FAISS → adversarial eval)
- GraphWalkGenerator (biased Markov walks across personal↔wikidata boundary)
- SynthesisGenerator (random cross-store sampling, baseline)

All candidates pooled → single filter pass → compare acceptance by source.

Usage:
    python scripts/test_end_to_end_synthesis.py
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


def _candidate_source(c) -> str:
    """Label a candidate by its generator source."""
    if c.walk_path and len(c.walk_path) >= 3 and c.walk_path[0] == "retrieval":
        return "RETRIEVAL"
    elif c.walk_path and len(c.walk_path) > 3:
        return "WALK"
    else:
        return "XSTORE"


async def run_dreaming(candidate_count: int = 15, num_runs: int = 1):
    from config.app_config import (
        CHROMA_PATH,
        KNOWLEDGE_GRAPH_PERSIST_PATH,
        KNOWLEDGE_GRAPH_ALIASES_PATH,
        GRAPH_WALK_MIN_BRIDGE_EDGES,
        SYNTHESIS_RETRIEVAL_ENABLED,
    )
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    from memory.graph_memory import GraphMemory
    from memory.entity_resolver import EntityResolver
    from knowledge.synthesis_generator import SynthesisGenerator
    from knowledge.synthesis_filter import SynthesisFilter
    from memory.synthesis_memory import SynthesisMemory
    from models.model_manager import ModelManager

    print("=" * 70)
    print("END-TO-END SYNTHESIS DREAMING TEST (HEAD-TO-HEAD)")
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
    print(f"  Bridge edges: {graph.count_bridge_edges()}")
    print(f"  Candidates per run: {candidate_count}")
    print(f"  Runs: {num_runs}")

    # Quota split: 5 each for 3 generators at candidates=15
    retrieval_quota = candidate_count // 3
    walk_quota = candidate_count // 3
    xstore_quota = candidate_count - retrieval_quota - walk_quota

    # Run
    all_run_results = []
    total_accepted = 0
    total_generated = 0
    total_by_stage = {}
    total_by_source = {"RETRIEVAL": {"gen": 0, "acc": 0}, "WALK": {"gen": 0, "acc": 0}, "XSTORE": {"gen": 0, "acc": 0}}

    for run_idx in range(num_runs):
        print(f"\n{'='*70}")
        print(f"RUN {run_idx + 1}/{num_runs}")
        print(f"{'='*70}")

        t0 = time.time()
        candidates = []

        # --- Tier 0: RetrievalSynthesisGenerator ---
        if SYNTHESIS_RETRIEVAL_ENABLED:
            from knowledge.synthesis_retriever import RetrievalSynthesisGenerator
            retrieval_gen = RetrievalSynthesisGenerator(
                chroma_store=store,
                model_manager=model_manager,
                graph_memory=graph,
                entity_resolver=resolver,
            )
            print(f"\n[RETRIEVAL] Generating {retrieval_quota} candidates...")
            retrieval_candidates = await retrieval_gen.generate_candidates(count=retrieval_quota)
            candidates.extend(retrieval_candidates)
            print(f"  Retrieval candidates: {len(retrieval_candidates)}")
        else:
            print(f"\n[RETRIEVAL] Disabled via config")

        # --- Tier 1: GraphWalkGenerator ---
        from knowledge.graph_walk_generator import GraphWalkGenerator
        bridge_count = graph.count_bridge_edges()
        if bridge_count >= GRAPH_WALK_MIN_BRIDGE_EDGES:
            walker = GraphWalkGenerator(
                graph_memory=graph,
                entity_resolver=resolver,
                model_manager=model_manager,
            )
            print(f"\n[WALK] {bridge_count} bridge edges, generating {walk_quota} candidates...")
            walk_candidates = await walker.generate_candidates(count=walk_quota)
            candidates.extend(walk_candidates)
            print(f"  Walk candidates: {len(walk_candidates)}")
        else:
            print(f"\n[WALK] Skipped ({bridge_count} < {GRAPH_WALK_MIN_BRIDGE_EDGES} bridge edges)")

        # --- Tier 2: SynthesisGenerator (baseline) ---
        generator = SynthesisGenerator(
            chroma_store=store,
            model_manager=model_manager,
            graph_memory=graph,
            entity_resolver=resolver,
        )
        print(f"\n[XSTORE] Generating {xstore_quota} cross-store candidates...")
        cross_candidates = await generator.generate_candidates(count=xstore_quota)
        candidates.extend(cross_candidates)
        print(f"  Cross-store candidates: {len(cross_candidates)}")

        gen_elapsed = time.time() - t0

        if not candidates:
            print("  No candidates generated from any generator.")
            all_run_results.append({"run": run_idx + 1, "generated": 0, "accepted": 0})
            continue

        print(f"\n  Total: {len(candidates)} candidates in {gen_elapsed:.1f}s")
        total_generated += len(candidates)

        # Show candidates
        for i, c in enumerate(candidates):
            src = _candidate_source(c)
            total_by_source[src]["gen"] += 1
            print(f"  [{i+1}] [{src}] {c.concept_a} ↔ {c.concept_b} (dist={c.endpoint_distance:.2f})")
            print(f"      domains: {c.source_domains}")
            print(f"      claim: {c.connection_claim[:120]}...")

        # Filter — pass graph_memory for bridge creation
        print(f"\nFiltering through 8-stage pipeline...")
        synthesis_memory = SynthesisMemory(store)
        filter_pipeline = SynthesisFilter(
            chroma_store=store,
            model_manager=model_manager,
            synthesis_memory=synthesis_memory,
            graph_memory=graph,
            entity_resolver=resolver,
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

        # Show accepted insights with source label
        accepted_results = batch_results.get("accepted_results", [])
        if accepted_results:
            print(f"\n  --- ACCEPTED INSIGHTS ---")
            for r in accepted_results:
                c = r.candidate
                src = _candidate_source(c)
                total_by_source[src]["acc"] += 1
                print(f"  [{src}] {c.concept_a} ↔ {c.concept_b}")
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
                    "source": _candidate_source(r.candidate),
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

    # Head-to-head comparison
    print(f"\n--- HEAD-TO-HEAD BY GENERATOR ---")
    for src in ["RETRIEVAL", "WALK", "XSTORE"]:
        gen = total_by_source[src]["gen"]
        acc = total_by_source[src]["acc"]
        rate = (acc / gen * 100) if gen > 0 else 0
        print(f"  {src:10s}: {gen:3d} generated, {acc:3d} accepted ({rate:.0f}%)")

    print(f"\n--- CRITERIA ---")
    print(f"  Candidates generated >= 10: {'PASS' if total_generated >= 10 else 'FAIL'} ({total_generated})")
    if acceptance_rate == 0:
        print(f"  WARNING: 0% acceptance — check structural query diversity")
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
