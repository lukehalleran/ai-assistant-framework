#!/usr/bin/env python3
"""
Generate a GRADEABLE batch of discovery candidates (pooled generator -> recalibrated filter),
so the owner can hand-grade them and we LOCK the Stage-6 composite threshold from real grades.

generate mode (default): run the production PooledConceptSynthesisGenerator through the REAL
SynthesisFilter under the CURRENT config, and emit every Stage-6 reacher (ACCEPT or QUEUE) as a
CSV row with its coherence level, novelty, composite, and the live accept/queue status, plus
BLANK grade columns. Candidates that die earlier (coherence WEAK / text_sanity) are correct
rejects and are NOT put up for grading (only counted).

analyze mode (--analyze <graded.csv>): read the grades (grade 1-5; 4-5 = a real discovery) and:
  - report precision/recall of the CURRENT threshold (accepted-and-good vs good-but-queued),
  - sweep composite thresholds and recommend the one that best separates grade>=4 from grade<=3
    (max Youden's J), flagging false-negatives (QUEUED but graded 4-5) and false-positives
    (ACCEPTED but graded 1-3).

Isolated temp chroma; no production writes. ~ (#pairs) articulation + filter calls.
Usage:
  python scripts/grade_discovery_batch.py [--n 44 --seed 23]
  python scripts/grade_discovery_batch.py --analyze <graded.csv>
"""
import argparse
import asyncio
import csv
import glob
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

_CSV_COLS = [
    "pair_id", "concept_a", "concept_b", "coherence_level", "novelty", "composite",
    "status", "connection_claim",
    # blank hand-grade columns:
    "grade", "mechanism_real", "heard_before", "notes",
]


async def generate(n, seed):
    from config.app_config import (SYNTHESIS_COMPOSITE_MIN_SCORE, SYNTHESIS_WEIGHT_COHERENCE,
                                   SYNTHESIS_WEIGHT_NOVELTY, SYNTHESIS_WEIGHT_DISTANCE,
                                   SYNTHESIS_WEIGHT_STRUCTURAL)
    from models.model_manager import ModelManager
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    from memory.synthesis_memory import SynthesisMemory
    from knowledge.synthesis_pooled_generator import PooledConceptSynthesisGenerator
    from knowledge.synthesis_filter import SynthesisFilter
    from knowledge.synthesis_models import CandidateStatus

    print(f"Discovery grading batch | weights coh={SYNTHESIS_WEIGHT_COHERENCE} nov={SYNTHESIS_WEIGHT_NOVELTY} "
          f"dist={SYNTHESIS_WEIGHT_DISTANCE} struct={SYNTHESIS_WEIGHT_STRUCTURAL} | gate={SYNTHESIS_COMPOSITE_MIN_SCORE}\n")
    mm = ModelManager()
    gen = PooledConceptSynthesisGenerator(model_manager=mm, seed=seed)
    pairs = gen._in_band_pairs()[:n]
    if not pairs:
        print("ERROR: no in-band pairs (FAISS index unavailable?)."); sys.exit(1)
    print(f"Articulating {len(pairs)} in-band pairs...", flush=True)
    sem = asyncio.Semaphore(5)
    cands = [c for c in await asyncio.gather(*(gen._articulate(sem, a, b, c) for a, b, c in pairs)) if c]
    print(f"  articulated {len(cands)}/{len(pairs)}", flush=True)

    tmp = tempfile.mkdtemp(prefix="grade_batch_")
    store = MultiCollectionChromaStore(persist_directory=tmp)
    filt = SynthesisFilter(chroma_store=store, model_manager=mm, synthesis_memory=SynthesisMemory(store))
    filt.graph_memory = None
    filt.entity_resolver = None
    print("Filtering through the recalibrated SynthesisFilter...", flush=True)
    results = await asyncio.gather(*(filt.process_candidate(c) for c in cands))

    rows, funnel = [], Counter()
    for c, res in zip(cands, results):
        comp_stage = next((s for s in res.stage_results if s.stage_name == "composite_scoring"), None)
        if comp_stage is None:
            funnel[f"reject@{res.rejection_stage}"] += 1
            continue
        accepted = res.status == CandidateStatus.ACCEPTED
        funnel["ACCEPT" if accepted else "QUEUE"] += 1
        rows.append({
            "concept_a": c.concept_a, "concept_b": c.concept_b,
            "coherence_level": res.coherence_level.name if res.coherence_level else "?",
            "novelty": round((comp_stage.metadata or {}).get("novelty", 0.0), 3),
            "composite": round(res.composite_score or 0.0, 3),
            "status": "ACCEPT" if accepted else "QUEUE",
            "connection_claim": c.connection_claim,
        })
    rows.sort(key=lambda r: -r["composite"])
    for i, r in enumerate(rows):
        r["pair_id"] = i

    print(f"\nfunnel: {dict(funnel)}")
    print(f"Gradeable (reached Stage 6): {len(rows)}  | ACCEPT={funnel['ACCEPT']}  QUEUE={funnel['QUEUE']}\n")
    print(f"{'#':>2} {'comp':>5} {'lvl':<9} {'nov':>5} {'status':<7} pair")
    for r in rows:
        print(f"{r['pair_id']:>2} {r['composite']:>5.3f} {r['coherence_level']:<9} {r['novelty']:>5.3f} "
              f"{r['status']:<7} {r['concept_a']} <-> {r['concept_b']}")

    ts = datetime.now()
    outdir = os.path.join(_REPO, "data", "synthesis_discovery")
    os.makedirs(outdir, exist_ok=True)
    stem = f"discovery_grading_batch_{ts:%Y%m%d_%H%M%S}"
    csv_path = os.path.join(outdir, stem + ".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**{k: "" for k in ("grade", "mechanism_real", "heard_before", "notes")}, **r})
    print(f"\nHand-grade this CSV -> {csv_path}")
    print("  grade 1-5 (4-5 = a real discovery worth surfacing); optional mechanism_real (y/n/unsure),")
    print("  heard_before (y/n). Then: python scripts/grade_discovery_batch.py --analyze "
          f"{os.path.basename(csv_path)}")


def analyze(path):
    if not os.path.isabs(path):
        path = os.path.join(_REPO, "data", "synthesis_discovery", path)
    from config.app_config import SYNTHESIS_COMPOSITE_MIN_SCORE
    rows = [r for r in csv.DictReader(open(path)) if str(r.get("grade", "")).strip()]
    print(f"Analyze {os.path.basename(path)} | graded {len(rows)}\n")
    if len(rows) < 6:
        print("Grade at least ~6 rows (column `grade`, 1-5) to calibrate."); return
    data = []
    for r in rows:
        try:
            g = int(float(r["grade"]))
        except ValueError:
            continue
        data.append({"comp": float(r["composite"]), "good": g >= 4,
                     "status": r["status"], "a": r["concept_a"], "b": r["concept_b"], "grade": g})
    cur = SYNTHESIS_COMPOSITE_MIN_SCORE
    good = [d for d in data if d["good"]]
    bad = [d for d in data if not d["good"]]
    print(f"good (grade>=4): {len(good)}   weak (<=3): {len(bad)}")
    # current threshold confusion
    fn = [d for d in good if d["comp"] < cur]   # good but queued
    fp = [d for d in bad if d["comp"] >= cur]    # weak but accepted
    print(f"\nCURRENT threshold {cur}:")
    print(f"  accept-and-good (TP): {sum(1 for d in good if d['comp']>=cur)}   "
          f"queue-and-weak (TN): {sum(1 for d in bad if d['comp']<cur)}")
    print(f"  FALSE NEGATIVES (good but QUEUED): {len(fn)}  " + ("(threshold too HIGH)" if fn else ""))
    for d in sorted(fn, key=lambda d: -d["comp"]):
        print(f"     comp={d['comp']:.3f} grade={d['grade']}  {d['a']} <-> {d['b']}")
    print(f"  FALSE POSITIVES (weak but ACCEPTED): {len(fp)}  " + ("(threshold too LOW)" if fp else ""))
    for d in sorted(fp, key=lambda d: -d["comp"]):
        print(f"     comp={d['comp']:.3f} grade={d['grade']}  {d['a']} <-> {d['b']}")
    # sweep: best Youden's J
    print("\nthreshold sweep (TPR=good accepted, FPR=weak accepted):")
    best = None
    for t in [x / 100 for x in range(60, 86, 2)]:
        tp = sum(1 for d in good if d["comp"] >= t)
        fpr_n = sum(1 for d in bad if d["comp"] >= t)
        tpr = tp / len(good) if good else 0
        fpr = fpr_n / len(bad) if bad else 0
        j = tpr - fpr
        mark = "  <- current" if abs(t - cur) < 1e-9 else ""
        if best is None or j > best[1]:
            best = (t, j)
        print(f"  >={t:.2f}: TPR={tpr:.2f} FPR={fpr:.2f} J={j:+.2f}{mark}")
    print(f"\nRECOMMENDED threshold (max Youden's J): {best[0]:.2f}  (current {cur})")
    print("  set in config.yaml: synthesis.composite_min_score")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        if len(sys.argv) < 3:
            print("usage: --analyze <graded.csv>"); return
        analyze(sys.argv[2]); return
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=44)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    asyncio.run(generate(args.n, args.seed))


if __name__ == "__main__":
    main()
