#!/usr/bin/env python3
"""
Diagnose WHY the top discovery accepts are famous textbook identities: is the world-novelty
oracle BLIND to them (scores documented identities as novel) or is COHERENCE floating them
(low novelty, but STRONG outvotes it at weights 0.35/0.60)?

Re-runs specific (a,b,claim) candidates from a grading-batch CSV through the REAL filter and
prints the per-term breakdown the composite hides: concept cos(A,B), cooccurrence_similarity,
claim novelty (novelty_external), specificity, the novelty AGGREGATE, coherence, distance,
composite, and crucially whether stage-3 (novelty_external / known gate) FIRED. Low cooccurrence
+ low cos + "novel" verdict on a known textbook identity == oracle blind spot (wiki corpus gap).

Isolated temp chroma; reads the real wiki index/corpus for the novelty stages.
Usage: python scripts/probe_top_discovery_novelty.py <batch.csv> [--top 5 --bottom 5]
"""
import argparse
import asyncio
import csv
import os
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--bottom", type=int, default=5)
    args = ap.parse_args()
    path = args.csv if os.path.isabs(args.csv) else os.path.join(_REPO, "data", "synthesis_discovery", args.csv)
    rows = sorted(csv.DictReader(open(path)), key=lambda r: -float(r["composite"]))
    pick = rows[: args.top] + rows[-args.bottom:]

    import numpy as np
    from models.model_manager import ModelManager
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    from memory.synthesis_memory import SynthesisMemory
    from knowledge.synthesis_filter import SynthesisFilter
    from knowledge.synthesis_models import SynthesisCandidate
    from knowledge.semantic_search import get_index

    idx = get_index(); idx.load()
    emb = idx.embedder
    mm = ModelManager()
    tmp = tempfile.mkdtemp(prefix="probe_nov_")
    store = MultiCollectionChromaStore(persist_directory=tmp)
    filt = SynthesisFilter(chroma_store=store, model_manager=mm, synthesis_memory=SynthesisMemory(store))
    filt.graph_memory = None
    filt.entity_resolver = None

    print(f"{'pair':<44}{'cosAB':>6}{'cooc':>6}{'claimN':>7}{'spec':>6}{'NOVagg':>7}{'coh':>5}{'dist':>6}{'comp':>6}  gate")
    print("-" * 104)
    for r in pick:
        a, b = r["concept_a"], r["concept_b"]
        va, vb = emb.encode([a, b], normalize_embeddings=True)
        cosab = float(np.dot(va, vb))
        cand = SynthesisCandidate(
            concept_a=a, concept_b=b, connection_claim=r["connection_claim"],
            walk_path=["pooled", a.lower(), b.lower()], source_domains={a, b},
            endpoint_distance=round(1.0 - cosab, 4))
        res = await filt.process_candidate(cand)
        comp_md = next((s.metadata for s in res.stage_results if s.stage_name == "composite_scoring"), {}) or {}
        cooc = res.cooccurrence_similarity
        claimN = res.novelty_score_external
        spec = 1.0 - res.template_similarity
        novagg = comp_md.get("novelty")
        coh = res.coherence_level.name[:4] if res.coherence_level else "—"
        dist = comp_md.get("distance")
        gate = res.rejection_stage or "ACCEPT"
        novs = f"{novagg:.3f}" if novagg is not None else "  —  "
        dists = f"{dist:.3f}" if dist is not None else "  —  "
        print(f"{a[:20]+' <-> '+b[:18]:<44}{cosab:>6.3f}{cooc:>6.3f}{claimN:>7.3f}{spec:>6.3f}{novs:>7}{coh:>5}{dists:>6}"
              f"{(res.composite_score or 0):>6.3f}  {gate}")
    print("-" * 104)
    print("READ: high cosAB or high cooc -> oracle SEES the link (should be queued/known).")
    print("      low cosAB AND low cooc on a famous textbook identity -> oracle BLIND (wiki-corpus gap).")
    print("      if novelty AGG is high for known identities -> oracle problem; if low but ACCEPT -> coherence floated it.")


if __name__ == "__main__":
    asyncio.run(main())
