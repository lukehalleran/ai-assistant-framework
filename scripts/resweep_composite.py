#!/usr/bin/env python3
"""
Re-sweep the Stage-6 composite acceptance on post-fix volume with the LIVE judge.

WHY (the "doubly stale" problem): the composite scorer (0.30·coherence + 0.40·novelty +
0.15·distance + 0.15·structural ≥ 0.65) was tuned while (a) the coherence judge was DEAD —
every candidate defaulted WEAK, so coherence ≈ floor and almost nothing cleared 0.65 — and
(b) Stage-3 novelty was inverted, so the wrong candidate population reached the composite at
all. Both are fixed now (judge RATING-first + reasoning-off; Stage-3 cos(A,B); retrieval
metric-aware). So ANY pre-fix "accepted N" count is meaningless. This runs a realistic
anchored candidate batch through the FULL corrected pipeline (live opus-4.8 judge) and reports
the per-stage funnel + composite distribution + acceptance — the corrected numbers.

ISOLATION: the filter is wired to a TEMP ChromaDB dir, so this NEVER reads or writes the
production synthesis_results audit queue or the knowledge graph. Pure measurement.

CANDIDATES: reuses the anchored pairs already generated + saved by the validated PoC
(data/synthesis_discovery/poc_anchored_vs_random_*.json) — realistic novel cross-domain pairs
— so we avoid re-running idx.search(k=1500), which re-triggers the wiki text-column OOM.

Makes real LLM calls (claim articulation + opus-4.8 judge). Needs Daemon stopped (FAISS/embedder).
Usage: python scripts/resweep_composite.py [--n 20 --seed 7]
"""
import argparse
import asyncio
import glob
import json
import os
import random
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
from knowledge.synthesis_generator import SynthesisGenerator
from knowledge.synthesis_filter import SynthesisFilter
from knowledge.synthesis_models import SynthesisCandidate, CandidateStatus
from knowledge.semantic_search import semantic_search_with_neighbors, get_index


def _ctx(concept):
    try:
        r = semantic_search_with_neighbors(concept, k=3)
        return " ".join((x.get("content") or x.get("text") or "") for x in r)[:300]
    except Exception:
        return ""


async def main():
    ap = argparse.ArgumentParser(description="Stage-6 composite re-sweep (live judge, isolated)")
    ap.add_argument("--n", type=int, default=20, help="Candidates to sweep")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pairs-json", default=None, help="PoC run JSON (default: most recent)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    jf = args.pairs_json or sorted(glob.glob(os.path.join(
        _REPO_ROOT, "data", "synthesis_discovery", "poc_anchored_vs_random_*.json")))[-1]
    run = json.load(open(jf))
    pairs = run["arm_a_pairs"]  # anchored novel cross-domain pairs
    rng.shuffle(pairs)
    pairs = pairs[: args.n]
    print(f"Source pairs: {os.path.basename(jf)} | sweeping {len(pairs)} anchored candidates")
    print(f"Composite: {SYNTHESIS_WEIGHT_COHERENCE}·coh + {SYNTHESIS_WEIGHT_NOVELTY}·nov + "
          f"{SYNTHESIS_WEIGHT_DISTANCE}·dist + {SYNTHESIS_WEIGHT_STRUCTURAL}·struct ≥ "
          f"{SYNTHESIS_COMPOSITE_MIN_SCORE} | judge={SYNTHESIS_COHERENCE_MODEL}\n")

    idx = get_index(); idx.load()
    if not idx.loaded:
        print("ERROR: FAISS index unavailable."); sys.exit(1)
    try:
        idx.index.nprobe = max(int(getattr(idx.index, "nprobe", 1)), 48)
    except Exception:
        pass

    # ISOLATION: temp chroma so nothing touches the production audit queue / graph.
    tmp = tempfile.mkdtemp(prefix="resweep_chroma_")
    store = MultiCollectionChromaStore(persist_directory=tmp)
    mm = ModelManager()
    sm = SynthesisMemory(store)
    gen = SynthesisGenerator(chroma_store=store, model_manager=mm)
    filt = SynthesisFilter(chroma_store=store, model_manager=mm, synthesis_memory=sm)
    filt.graph_memory = None; filt.entity_resolver = None  # no bridge-edge side effect
    print(f"(isolated temp chroma: {tmp})\n")

    rows = []
    for n, p in enumerate(pairs, 1):
        a, b = p["a"], p["b"]
        ctx_a, ctx_b = _ctx(a), _ctx(b)
        claim = await gen._articulate_bridge(a, b, "Wikipedia", "Wikipedia", ctx_a, ctx_b)
        if not claim or "NO_CONNECTION" in claim.upper():
            print(f"  [{n:2}/{len(pairs)}] {a} <-> {b}: no-connection (not a candidate)")
            rows.append({"a": a, "b": b, "articulated": False})
            continue
        cand = SynthesisCandidate(
            concept_a=a, concept_b=b, connection_claim=claim,
            walk_path=["wiki", a.lower(), b.lower(), "wiki"],
            source_domains={"wikipedia_a", "wikipedia_b"},
            endpoint_distance=float(max(0.0, min(1.0, 1.0 - p["cos"]))),
            timestamp=datetime.now(),
        )
        res = await filt.process_candidate(cand)
        accepted = res.status == CandidateStatus.ACCEPTED
        from knowledge.synthesis_filter import is_defaulted_coherence
        defaulted = is_defaulted_coherence(getattr(res, "coherence_justification", None))
        rows.append({
            "a": a, "b": b, "articulated": True, "accepted": accepted,
            "rejection_stage": res.rejection_stage,
            "composite": res.composite_score,
            "coherence": res.coherence_level.name if res.coherence_level else None,
            "coherence_defaulted": defaulted,  # fail-open MODERATE — exclude from rates
            "reached_composite": any(s.stage_name == "composite_scoring" for s in res.stage_results),
        })
        tag = "ACCEPT" if accepted else f"reject@{res.rejection_stage}"
        comp = f"comp={res.composite_score:.3f}" if res.composite_score else "comp=—"
        coh = res.coherence_level.name if res.coherence_level else "—"
        print(f"  [{n:2}/{len(pairs)}] {tag:18} {comp} coh={coh:9} | {a} <-> {b}", flush=True)

    # ---- aggregate funnel ----
    # Fail-open MODERATE rows (judge LLM error/empty) are defaulted verdicts, not
    # ratings — exclude them from measured rates, but report the count loudly.
    n_defaulted = sum(1 for r in rows if r.get("coherence_defaulted"))
    if n_defaulted:
        print(f"\n  !! {n_defaulted} rows had DEFAULTED coherence (judge error/empty) — "
              f"excluded from the rates below")
    rows_measured = [r for r in rows if not r.get("coherence_defaulted")]
    artic = [r for r in rows_measured if r.get("articulated")]
    accepted = [r for r in artic if r.get("accepted")]
    reached = [r for r in artic if r.get("reached_composite")]
    rej = Counter(r["rejection_stage"] for r in artic if not r.get("accepted"))

    print("\n" + "=" * 76)
    print("STAGE-6 COMPOSITE RE-SWEEP — corrected pipeline, live judge")
    print("=" * 76)
    print(f"  candidates swept:            {len(rows)}")
    print(f"  articulated (real claims):   {len(artic)}")
    print(f"  reached composite (stage 6): {len(reached)}")
    print(f"  ACCEPTED (composite≥{SYNTHESIS_COMPOSITE_MIN_SCORE}):     {len(accepted)} "
          f"({len(accepted)/max(len(artic),1):.0%} of articulated)")
    print("  rejection breakdown by stage:")
    for stage, c in rej.most_common():
        print(f"    {stage:24} {c}")
    if reached:
        comps = sorted(r["composite"] for r in reached if r["composite"] is not None)
        print(f"  composite scores @ stage 6 (n={len(comps)}): "
              f"min={comps[0]:.3f} med={comps[len(comps)//2]:.3f} max={comps[-1]:.3f}")
        print(f"    threshold {SYNTHESIS_COMPOSITE_MIN_SCORE} — "
              f"{sum(1 for c in comps if c>=SYNTHESIS_COMPOSITE_MIN_SCORE)}/{len(comps)} clear it")
    print("=" * 76)
    print("Read: with the dead judge, coherence≈floor → ~nothing cleared 0.65. If accepted>0\n"
          "now and the composites @stage6 straddle 0.65 sensibly, the corrected pipeline\n"
          "produces real acceptances; the stale pre-fix 'accepted N' is replaced by this.")

    ts = datetime.now()
    out = os.path.join(_REPO_ROOT, "data", "synthesis_discovery",
                       f"resweep_composite_{ts:%Y%m%d_%H%M%S}.json")
    json.dump({"timestamp": ts.isoformat(timespec="seconds"), "source": os.path.basename(jf),
               "composite_min": SYNTHESIS_COMPOSITE_MIN_SCORE, "judge": SYNTHESIS_COHERENCE_MODEL,
               "n_swept": len(rows), "n_articulated": len(artic),
               "n_reached_composite": len(reached), "n_accepted": len(accepted),
               "rejection_breakdown": dict(rej), "rows": rows},
              open(out, "w"), indent=2, default=str)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    asyncio.run(main())
