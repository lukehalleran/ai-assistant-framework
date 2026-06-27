#!/usr/bin/env python3
"""
Wiki<->wiki synthesis experiment (validation Test B on real generated content).

The personal-anchored generators produce weak candidates (sparse graph, vague nodes,
random fact<->wiki pairings -> 0/13 accepted). Hypothesis: a denser, richer concept
space (Wikipedia <-> Wikipedia) yields stronger candidates. This script tests it AND
produces the validation signal from docs/SYNTHESIS_VALIDATION.md Test B:

  1. Sample pairs of Wikipedia concepts (diverse topics).
  2. Articulate a bridge between each (the generator's own LLM articulation).
  3. Run each bridge through:
       - novelty_external (FAISS): is this connection already KNOWN in literature?
       - the coherence judge (now fixed): is it a real structural isomorphism?
  4. Report:
       - rediscovery rate = KNOWN / generated  (generator finds real structure)
       - coherence pass-rate = MODERATE+ / generated
       - KNOWN & coherence-pass = literature-confirmed real AND the judge agrees
         (the strongest single signal — no human grading)

The personal-centric gates (domain_crossing, semantic_distance) are intentionally
skipped — they aren't meaningful for wiki<->wiki.

Usage:
  python scripts/wiki_synthesis_experiment.py --pairs 12

Makes real LLM calls (articulation + judge). Needs the DB free (Daemon stopped).
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.app_config import CHROMA_PATH
from models.model_manager import ModelManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from memory.synthesis_memory import SynthesisMemory
from knowledge.synthesis_generator import SynthesisGenerator
from knowledge.synthesis_filter import SynthesisFilter
from knowledge.synthesis_models import SynthesisCandidate, SynthesisResult, CoherenceLevel

_PASS = {CoherenceLevel.MODERATE, CoherenceLevel.STRONG}


def _candidate(name_a, name_b, claim):
    return SynthesisCandidate(
        concept_a=name_a, concept_b=name_b, connection_claim=claim,
        walk_path=["wiki", name_a.lower(), name_b.lower(), "wiki"],
        source_domains={"wikipedia_a", "wikipedia_b"},
        endpoint_distance=0.6, timestamp=datetime.now(),
    )


async def main():
    ap = argparse.ArgumentParser(description="Wiki<->wiki synthesis experiment")
    ap.add_argument("--pairs", type=int, default=12, help="Concept pairs to test (default 12)")
    args = ap.parse_args()

    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    mm = ModelManager()
    sm = SynthesisMemory(store)
    gen = SynthesisGenerator(chroma_store=store, model_manager=mm)
    filt = SynthesisFilter(chroma_store=store, model_manager=mm, synthesis_memory=sm)

    from knowledge.semantic_search import get_index
    idx = get_index(); idx.load()
    if not idx.loaded:
        print("ERROR: FAISS wiki index not available."); sys.exit(1)
    print(f"FAISS wiki vectors: {idx._total_rows:,}\n")

    # 1) Sample 2N diverse wiki concepts, split into two pools, pair across.
    arts = gen._sample_wiki_articles(args.pairs * 2)
    if len(arts) < 4:
        print(f"Too few wiki samples ({len(arts)})."); sys.exit(1)
    half = len(arts) // 2
    pool_a, pool_b = arts[:half], arts[half:half * 2]

    print(f"Articulating {min(len(pool_a), len(pool_b))} wiki<->wiki bridges...\n")
    rows = []
    for ia, ib in zip(pool_a, pool_b):
        name_a = gen._extract_concept_name(ia, source="wiki")
        name_b = gen._extract_concept_name(ib, source="wiki")
        if not name_a or not name_b or name_a.lower() == name_b.lower():
            continue
        ctx_a = (ia.get("content", ""))[:300]
        ctx_b = (ib.get("content", ""))[:300]
        # 2) articulate
        claim = await gen._articulate_bridge(name_a, name_b, "Wikipedia", "Wikipedia", ctx_a, ctx_b)
        if not claim or "NO_CONNECTION" in claim.upper():
            print(f"  [no connection] {name_a} <-> {name_b}")
            continue

        result = SynthesisResult(candidate=_candidate(name_a, name_b, claim))
        # 3a) known-in-literature check (FAISS)
        nov = await filt._stage_3_novelty_external(result)
        known = not nov.passed  # rejected here == already in literature
        # 3b) coherence judge (fixed)
        await filt._stage_5_coherence_judge(result)
        coh = result.coherence_level
        coh_pass = coh in _PASS

        rows.append((name_a, name_b, known, coh.name if coh else "N/A", coh_pass,
                     result.cooccurrence_similarity, result.novelty_score_external))
        print(f"  {'KNOWN' if known else 'novel':6} | coh={coh.name if coh else 'N/A':9} "
              f"{'PASS' if coh_pass else 'fail'} | cooccur={result.cooccurrence_similarity:.2f} "
              f"| {name_a} <-> {name_b}")

    n = len(rows)
    if not n:
        print("\nNo bridges articulated — try --pairs higher or check sampling.")
        return
    known_n = sum(r[2] for r in rows)
    coh_n = sum(r[4] for r in rows)
    known_and_coh = sum(1 for r in rows if r[2] and r[4])

    print("\n" + "=" * 70)
    print("WIKI<->WIKI EXPERIMENT — Test B (rediscovery) + judge on real content")
    print("=" * 70)
    print(f"  bridges articulated:        {n}")
    print(f"  KNOWN (in literature):      {known_n}/{n} ({known_n/n:.0%})  <- rediscovery rate")
    print(f"  coherence pass (MODERATE+): {coh_n}/{n} ({coh_n/n:.0%})")
    print(f"  KNOWN & coherence-pass:     {known_and_coh}/{n}  <- literature-confirmed AND judge agrees")
    print("=" * 70)
    print("Read: a meaningful KNOWN rate (vs ~0 for random noise) = the generator finds real\n"
          "structure. KNOWN & coherence-pass items are rediscoveries the judge also accepts —\n"
          "the cleanest no-human-grading proof the generate+filter loop works. (Compare the\n"
          "coherence pass-rate to the personal-anchored pass, which got 0/13.)")


if __name__ == "__main__":
    asyncio.run(main())
