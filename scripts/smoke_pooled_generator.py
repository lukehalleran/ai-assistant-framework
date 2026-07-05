#!/usr/bin/env python3
"""Smoke: the production PooledConceptSynthesisGenerator -> real SynthesisFilter.
Confirms the wired discovery path produces candidates that survive the judge.
Isolated temp chroma; no production writes. Usage: python scripts/smoke_pooled_generator.py [count]"""
import asyncio, os, sys, tempfile
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 6


async def main():
    from models.model_manager import ModelManager
    from knowledge.synthesis_pooled_generator import PooledConceptSynthesisGenerator
    from knowledge.synthesis_filter import SynthesisFilter
    from memory.synthesis_memory import SynthesisMemory
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    mm = ModelManager()
    gen = PooledConceptSynthesisGenerator(model_manager=mm, seed=7)
    print(f"Generating {COUNT} pooled discovery candidates...", flush=True)
    cands = await gen.generate_candidates(count=COUNT)
    print(f"Generated {len(cands)} candidates:")
    for c in cands:
        print(f"  {c.concept_a} <-> {c.concept_b}  (dist={c.endpoint_distance})")
        print(f"     {c.connection_claim[:130]}")

    tmp = tempfile.mkdtemp(prefix="pooled_smoke_")
    store = MultiCollectionChromaStore(persist_directory=tmp)
    filt = SynthesisFilter(chroma_store=store, model_manager=mm, synthesis_memory=SynthesisMemory(store))
    filt.graph_memory = None
    filt.entity_resolver = None
    print("\nFiltering through the real SynthesisFilter...", flush=True)
    results = await asyncio.gather(*(filt.process_candidate(c) for c in cands))
    lvl = Counter((r.coherence_level.name if r.coherence_level else "—") for r in results)
    stage = Counter((r.rejection_stage or "ACCEPTED") for r in results)
    print(f"\ncoherence levels: {dict(lvl)}")
    print(f"outcome by stage: {dict(stage)}")
    acc = sum(1 for r in results if (r.rejection_stage or 'ACCEPTED') == 'ACCEPTED')
    modstrong = sum(1 for r in results if r.coherence_level and r.coherence_level.name in ('MODERATE', 'STRONG'))
    print(f"\nMOD+STRONG: {modstrong}/{len(cands)}   ACCEPTED: {acc}/{len(cands)}")


if __name__ == "__main__":
    asyncio.run(main())
