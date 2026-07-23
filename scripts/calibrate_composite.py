#!/usr/bin/env python3
"""
Calibrate the Stage-6 composite for the PRODUCTION pooled discovery generator.

Stage 6 is `composite = w_coh·coherence + w_nov·novelty + w_dist·distance + w_struct·structural
>= composite_min_score`. The 2026-06-28 diagnosis (on the OLD generator) found it behaves as
"MODERATE pair that's far enough apart": structural is a hard constant 0.5 (every pair has 2
domains), coherence is quantized & flat at the MODERATE decision point, so accept is
distance-dominated and ~2/3 of MODERATE real-structure candidates are dropped.

This version measures the SAME components but on the shipped PooledConceptSynthesisGenerator
(knowledge/synthesis_pooled_generator.py) so the calibration reflects what production now emits.
It captures, for every Stage-6 reacher (accepted OR composite-rejected): the 4 components, the 4
novelty sub-signals, the composite, the coherence level, and the claim (so the rows are gradeable).

Diagnostics (no human grading needed to READ):
  - per-component min/median/max/range across Stage-6 reachers -> which components are DEGENERATE
    (tiny range -> add no discrimination, cap the composite);
  - composite by coherence LEVEL (MODERATE vs STRONG -> only these reach Stage 6);
  - threshold-sensitivity curve (accepts at 0.55..0.75) -> where the gate sits, how sharp it is.

ISOLATED temp ChromaDB; no production writes. Real LLM calls (articulation + opus judge).
Usage: python scripts/calibrate_composite.py [--n 50 --seed 11]
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import tempfile
from collections import Counter
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from config.app_config import (SYNTHESIS_COHERENCE_MODEL, SYNTHESIS_COMPOSITE_MIN_SCORE,
                               SYNTHESIS_WEIGHT_COHERENCE, SYNTHESIS_WEIGHT_NOVELTY,
                               SYNTHESIS_WEIGHT_DISTANCE, SYNTHESIS_WEIGHT_STRUCTURAL)
from models.model_manager import ModelManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from memory.synthesis_memory import SynthesisMemory
from knowledge.synthesis_pooled_generator import PooledConceptSynthesisGenerator
from knowledge.synthesis_filter import SynthesisFilter
from knowledge.synthesis_models import CandidateStatus

_COMPONENTS = ["coherence", "novelty", "distance", "structural"]
_NOV_SUB = ["claim_novelty", "cooccurrence_novelty", "specificity", "internal_novelty"]
_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


async def main():
    ap = argparse.ArgumentParser(description="Stage-6 composite calibration (pooled generator, isolated)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    mm = ModelManager()
    gen = PooledConceptSynthesisGenerator(model_manager=mm, seed=args.seed)
    pairs = gen._in_band_pairs()
    if not pairs:
        print("ERROR: no in-band pairs (FAISS index unavailable?)."); sys.exit(1)
    pairs = pairs[: args.n]
    print(f"Pooled in-band pairs: sweeping {len(pairs)} (band {gen.min_cos}-{gen.max_cos})")
    print(f"Composite weights: coh={SYNTHESIS_WEIGHT_COHERENCE} nov={SYNTHESIS_WEIGHT_NOVELTY} "
          f"dist={SYNTHESIS_WEIGHT_DISTANCE} struct={SYNTHESIS_WEIGHT_STRUCTURAL} | "
          f"gate={SYNTHESIS_COMPOSITE_MIN_SCORE} | judge={SYNTHESIS_COHERENCE_MODEL}\n")

    tmp = tempfile.mkdtemp(prefix="calib_chroma_")
    store = MultiCollectionChromaStore(persist_directory=tmp)
    filt = SynthesisFilter(chroma_store=store, model_manager=mm, synthesis_memory=SynthesisMemory(store))
    filt.graph_memory = None
    filt.entity_resolver = None
    print(f"(isolated temp chroma: {tmp})\n")

    sem = asyncio.Semaphore(5)

    async def articulate(a, b, cos):
        return await gen._articulate(sem, a, b, cos)

    cands = [c for c in await asyncio.gather(*(articulate(a, b, c) for a, b, c in pairs)) if c]
    print(f"Articulated {len(cands)}/{len(pairs)} (NO_CONNECTION: {len(pairs) - len(cands)})", flush=True)

    from knowledge.synthesis_filter import is_defaulted_coherence

    stage6, funnel = [], Counter()
    results = await asyncio.gather(*(filt.process_candidate(c) for c in cands))
    for c, res in zip(cands, results):
        # A fail-open MODERATE (judge LLM error/empty) is a defaulted verdict, not a
        # rating — including it would inflate the measured distributions.
        if is_defaulted_coherence(getattr(res, "coherence_justification", None)):
            funnel["excluded@defaulted_coherence"] += 1
            continue
        comp_stage = next((s for s in res.stage_results if s.stage_name == "composite_scoring"), None)
        if comp_stage is None:
            funnel[f"reject@{res.rejection_stage}"] += 1
            continue
        md = comp_stage.metadata or {}
        row = {"a": c.concept_a, "b": c.concept_b, "claim": c.connection_claim,
               "composite": res.composite_score, "accepted": res.status == CandidateStatus.ACCEPTED,
               "coh_level": res.coherence_level.name if res.coherence_level else "?"}
        for k in _COMPONENTS + _NOV_SUB:
            row[k] = md.get(k)
        stage6.append(row)
        funnel["accepted" if row["accepted"] else "reject@composite"] += 1

    print("\n" + "=" * 80)
    print(f"COMPOSITE CALIBRATION (pooled) — {len(cands)} articulated, {len(stage6)} reached Stage 6")
    print("=" * 80)
    print("  funnel:", dict(funnel))
    if not stage6:
        print("\nNo candidates reached Stage 6 — increase --n."); return

    def stats(key):
        vals = sorted(r[key] for r in stage6 if r.get(key) is not None)
        return (vals[0], statistics.median(vals), vals[-1], vals[-1] - vals[0]) if vals else None

    print(f"\n  component distributions across Stage-6 reachers (n={len(stage6)}):")
    print(f"    {'component':<22}{'min':>7}{'median':>9}{'max':>7}{'range':>8}  note")
    for key in _COMPONENTS + ["—"] + _NOV_SUB:
        if key == "—":
            print(f"    {'· novelty sub-signals':<22}"); continue
        s = stats(key)
        if s is None:
            continue
        note = "DEGENERATE (tiny range)" if s[3] < 0.08 else ("compressed" if s[3] < 0.20 else "")
        print(f"    {key:<22}{s[0]:>7.3f}{s[1]:>9.3f}{s[2]:>7.3f}{s[3]:>8.3f}  {note}")

    comps = sorted(r["composite"] for r in stage6 if r["composite"] is not None)
    print(f"\n  composite distribution: min={comps[0]:.3f} median={comps[len(comps)//2]:.3f} max={comps[-1]:.3f}")
    print("  threshold sensitivity (accepts among Stage-6 reachers):")
    for t in _THRESHOLDS:
        acc = sum(1 for c in comps if c >= t)
        mark = "  <- current" if abs(t - SYNTHESIS_COMPOSITE_MIN_SCORE) < 1e-9 else ""
        print(f"    >={t:.2f}:  {acc:>3}/{len(comps)} ({acc/len(comps):.0%}) {'#'*acc}{mark}")

    print("\n  composite by coherence LEVEL (only MODERATE/STRONG reach Stage 6):")
    by_lvl = {}
    for r in stage6:
        by_lvl.setdefault(r["coh_level"], []).append(r["composite"])
    for lvl, vs in sorted(by_lvl.items()):
        vs = sorted(v for v in vs if v is not None)
        if vs:
            print(f"    {lvl:<10} n={len(vs):>2}  composite min={vs[0]:.3f} median={vs[len(vs)//2]:.3f} "
                  f"max={vs[-1]:.3f}  accept@{SYNTHESIS_COMPOSITE_MIN_SCORE}="
                  f"{sum(1 for v in vs if v >= SYNTHESIS_COMPOSITE_MIN_SCORE)}/{len(vs)}")
    print("=" * 80)

    ts = datetime.now()
    out = os.path.join(_REPO_ROOT, "data", "synthesis_discovery",
                       f"calibrate_composite_pooled_{ts:%Y%m%d_%H%M%S}.json")
    json.dump({"timestamp": ts.isoformat(timespec="seconds"), "n_articulated": len(cands),
               "n_stage6": len(stage6), "funnel": dict(funnel),
               "weights": {"coherence": SYNTHESIS_WEIGHT_COHERENCE, "novelty": SYNTHESIS_WEIGHT_NOVELTY,
                           "distance": SYNTHESIS_WEIGHT_DISTANCE, "structural": SYNTHESIS_WEIGHT_STRUCTURAL},
               "gate": SYNTHESIS_COMPOSITE_MIN_SCORE, "stage6_rows": stage6},
              open(out, "w"), indent=2, default=str)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
