#!/usr/bin/env python3
"""
Re-score the human-graded synthesis candidates through the CORRECTED pipeline.

The 28 graded candidates in the production synthesis_results queue are slider-only (no
two-layer binaries) and almost all grade 4-5 (27/28) — a clean POSITIVE set of insights the
owner personally validated, NOT a balanced classifier-training set. So the useful question is
an agreement / recall check, not threshold separation:

  Of the connections you rated 4-5 (real insight / breakthrough), does the CORRECTED pipeline
  (fixed opus judge + fixed Stage-3 novelty + metric-aware retrieval) still ACCEPT them, and
  where do they land on the composite? Plus OLD composite (as generated) vs NEW (re-scored).

Claims are stored, so NO re-articulation — only the fixed judge re-runs (~28 opus calls).
ISOLATED: reads the production queue read-only; the filter writes to a TEMP chroma (no
re-pollution of the audit queue / graph).

CAVEAT printed in the output: these are PERSONAL concepts (e.g. "good_to_go", "get a 90"),
and Stage-3/distance are wiki-oriented — early-stage rejections may reflect personal-vs-wiki
mismatch, not candidate quality. The judge (on the stored claim) is the meaningful signal.

Usage: python scripts/rescore_graded.py
"""
import asyncio
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from config.app_config import CHROMA_PATH, SYNTHESIS_COMPOSITE_MIN_SCORE, SYNTHESIS_COHERENCE_MODEL
from models.model_manager import ModelManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from memory.synthesis_memory import SynthesisMemory
from knowledge.synthesis_filter import SynthesisFilter
from knowledge.synthesis_models import SynthesisCandidate, CandidateStatus
from knowledge.semantic_search import get_index


def _load_graded():
    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    col = store._get_collection("synthesis_results")
    data = col.get(include=["metadatas", "documents"])
    out = []
    for i, m, d in zip(data["ids"], data["metadatas"], data["documents"]):
        if not m or str(m.get("human_grade", "")).strip() in ("", "None"):
            continue
        claim = str(m.get("connection_claim") or d or "").strip()
        sd = m.get("source_domains")
        domains = {x.strip() for x in str(sd).split(",") if x.strip()} if sd else set()
        out.append({
            "id": i, "grade": int(float(m.get("human_grade"))),
            "a": m.get("concept_a"), "b": m.get("concept_b"), "claim": claim,
            "domains": domains, "endpoint_distance": float(m.get("endpoint_distance") or 0.2),
            "old_composite": (float(m["composite_score"]) if m.get("composite_score") is not None else None),
            "old_coh": m.get("coherence_level"),
        })
    return out


async def main():
    graded = _load_graded()
    print(f"Loaded {len(graded)} human-graded candidates from production (read-only).")
    print(f"  grade distribution: {dict(Counter(g['grade'] for g in graded))}")
    print(f"  judge={SYNTHESIS_COHERENCE_MODEL} | accept gate={SYNTHESIS_COMPOSITE_MIN_SCORE}\n")

    idx = get_index(); idx.load()
    if not idx.loaded:
        print("ERROR: FAISS unavailable."); sys.exit(1)
    try:
        idx.index.nprobe = max(int(getattr(idx.index, "nprobe", 1)), 48)
    except Exception:
        pass

    tmp = tempfile.mkdtemp(prefix="rescore_chroma_")
    temp_store = MultiCollectionChromaStore(persist_directory=tmp)
    mm = ModelManager()
    sm = SynthesisMemory(temp_store)
    filt = SynthesisFilter(chroma_store=temp_store, model_manager=mm, synthesis_memory=sm)
    filt.graph_memory = None; filt.entity_resolver = None
    print(f"(isolated temp chroma: {tmp})\n")

    rows = []
    for n, g in enumerate(graded, 1):
        if not g["claim"]:
            rows.append({**g, "articulated": False}); continue
        domains = g["domains"] if len(g["domains"]) >= 2 else {"domain_a", "domain_b"}
        cand = SynthesisCandidate(
            concept_a=g["a"], concept_b=g["b"], connection_claim=g["claim"],
            walk_path=["wiki", str(g["a"]).lower(), str(g["b"]).lower(), "wiki"],
            source_domains=domains, endpoint_distance=g["endpoint_distance"],
            timestamp=datetime.now())
        res = await filt.process_candidate(cand)
        accepted = res.status == CandidateStatus.ACCEPTED
        from knowledge.synthesis_filter import is_defaulted_coherence
        rows.append({
            **g, "articulated": True, "accepted": accepted,
            "rejection_stage": res.rejection_stage, "new_composite": res.composite_score,
            "new_coh": res.coherence_level.name if res.coherence_level else None,
            "coherence_defaulted": is_defaulted_coherence(
                getattr(res, "coherence_justification", None)),
        })
        tag = "ACCEPT" if accepted else f"reject@{res.rejection_stage}"
        nc = f"{res.composite_score:.3f}" if res.composite_score else "—"
        oc = f"{g['old_composite']:.3f}" if g["old_composite"] is not None else "—"
        print(f"  [{n:2}/{len(graded)}] g{g['grade']} | OLD {oc} -> NEW {nc} | {tag:22} "
              f"coh={res.coherence_level.name if res.coherence_level else '—':9} | "
              f"{str(g['a'])[:24]} <-> {str(g['b'])[:24]}", flush=True)

    # ---- aggregate ----
    # Fail-open MODERATE rows (judge error/empty) are defaulted verdicts, not ratings —
    # exclude from measured recall, report the count.
    n_defaulted = sum(1 for r in rows if r.get("coherence_defaulted"))
    if n_defaulted:
        print(f"\n  !! {n_defaulted} rows had DEFAULTED coherence (judge error/empty) — "
              f"excluded from the recall numbers below")
    artic = [r for r in rows if r.get("articulated") and not r.get("coherence_defaulted")]
    pos = [r for r in artic if r["grade"] >= 4]   # the human-validated insights
    pos_acc = [r for r in pos if r.get("accepted")]
    rej_stage = Counter(r["rejection_stage"] for r in pos if not r.get("accepted"))

    print("\n" + "=" * 78)
    print("RE-SCORE OF HUMAN-GRADED INSIGHTS THROUGH THE CORRECTED PIPELINE")
    print("=" * 78)
    print(f"  graded candidates re-scored:        {len(artic)}")
    print(f"  grade 4-5 (your validated insights): {len(pos)}")
    print(f"  ...that the CORRECTED pipeline ACCEPTS: {len(pos_acc)}/{len(pos)} "
          f"({len(pos_acc)/max(len(pos),1):.0%})  <- recall on YOUR positives")
    print("  where the rejected 4-5s die (corrected pipeline):")
    for stage, c in rej_stage.most_common():
        print(f"    {stage:24} {c}")
    # composite shift for those reaching stage 6
    reached = [r for r in pos if r.get("new_composite") is not None]
    if reached:
        comps = sorted(r["new_composite"] for r in reached)
        print(f"  NEW composite for 4-5s reaching stage 6 (n={len(reached)}): "
              f"min={comps[0]:.3f} med={comps[len(comps)//2]:.3f} max={comps[-1]:.3f}; "
              f"{sum(1 for c in comps if c>=SYNTHESIS_COMPOSITE_MIN_SCORE)}/{len(comps)} clear {SYNTHESIS_COMPOSITE_MIN_SCORE}")
        paired = [(r["old_composite"], r["new_composite"]) for r in reached if r["old_composite"] is not None]
        if paired:
            do = sum(o for o, _ in paired) / len(paired)
            dn = sum(nw for _, nw in paired) / len(paired)
            print(f"  OLD vs NEW composite (paired, n={len(paired)}): "
                  f"mean OLD {do:.3f} -> NEW {dn:.3f} ({dn-do:+.3f})")
    print("  coherence level shift (OLD -> NEW) on grade 4-5:")
    shift = Counter((r.get("old_coh"), r.get("new_coh")) for r in pos)
    for (o, nw), c in shift.most_common():
        print(f"    {str(o):10} -> {str(nw):10}  {c}")
    print("=" * 78)
    print("CAVEAT: personal concepts + wiki-oriented Stage-3/distance — early-stage rejects may\n"
          "be personal-vs-wiki mismatch, not quality. The judge (on the stored claim) is the\n"
          "meaningful signal. This is a RECALL/agreement check on a positive-only set, not a\n"
          "balanced threshold calibration (you have no graded negatives).")

    ts = datetime.now()
    out = os.path.join(_REPO_ROOT, "data", "synthesis_discovery",
                       f"rescore_graded_{ts:%Y%m%d_%H%M%S}.json")
    json.dump({"timestamp": ts.isoformat(timespec="seconds"), "n_graded": len(graded),
               "grade_dist": dict(Counter(g["grade"] for g in graded)),
               "pos_accept": len(pos_acc), "pos_total": len(pos), "rows": rows},
              open(out, "w"), indent=2, default=str)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    asyncio.run(main())
