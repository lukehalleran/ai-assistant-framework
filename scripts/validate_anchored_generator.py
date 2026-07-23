#!/usr/bin/env python3
"""
Validate ANCHORED discovery generation through the REAL filter (the missing link).

The audit (2026-06-30) showed discovery generation is a GENERATION-QUALITY problem, not
a filter problem: of 139 synthesis_results, ~78 die as "Coherence WEAK < MODERATE" with
de-jargon snippets like "two things weakly linked" / "both terms can serve as one
endpoint" — the signature of random-ish pairing with no real shared mechanism. The filter
is high-precision (accepted+graded = 27/28 grade 4-5). So the fix is better GENERATION.

`scripts/synthesis_poc_anchored_vs_random.py` already proved ANCHORED pair-selection
(anchor concept -> FAISS-neighbourhood partner in the non-obvious band cos<0.45) beats
distance-matched random on the doc-co-occurrence KNOWN oracle. But that validated against
the oracle, NOT the production coherence judge. THIS script closes that gap: it takes the
SAME validated anchored vs distance-matched-random selection, ARTICULATES each pair into a
SynthesisCandidate, and runs BOTH arms through the ACTUAL SynthesisFilter, then compares:
  - coherence MODERATE+STRONG rate (does the judge see real structure more often?),
  - accept rate, and
  - rejected-known rate (rediscoveries = real-but-documented = also "found real structure").

PRE-REGISTERED success: anchored's (MODERATE+STRONG)+known rate > random's, i.e. anchored
candidates carry real structure the judge recognizes more often than random ones. If so,
productionizing the anchored selection as a generator is justified.

Reuses the validated selection verbatim from the PoC (memory-safe FAISS: title-only reads,
never idx.search over the 33GB text column). Isolated temp chroma; no production writes.
~ (|A|+|B|) articulation calls + judge calls on candidates reaching the coherence stage.
Usage: python scripts/validate_anchored_generator.py [--anchors N] [--per-anchor M] [--seed S]
"""
import argparse
import asyncio
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
sys.path.insert(0, _SCRIPTS)

import random

from config.app_config import SYNTHESIS_COHERENCE_MODEL
from knowledge.semantic_search import get_index
from knowledge.synthesis_models import SynthesisCandidate
from knowledge.synthesis_filter import SynthesisFilter
from memory.synthesis_memory import SynthesisMemory
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from models.model_manager import ModelManager
# the VALIDATED anchored/random selection machinery (imported, not rebuilt)
from scripts.synthesis_poc_anchored_vs_random import build_arm_a, build_arm_b, CosCache, two_prop_z

DISCOVERY_SYS = (
    "You identify non-obvious STRUCTURAL connections between concepts for a discovery "
    "engine. A real connection is a shared mechanism, dynamic, feedback loop, trade-off, or "
    "mathematical form — NOT a shared topic or vague 'both are systems'. A separate "
    "adversarial judge rejects weak ones, so propose only a genuine shared structure; if "
    "there is none, say NO_CONNECTION."
)
DISCOVERY_PROMPT = """Two concepts:
  A: {a}
  B: {b}

Name the SINGLE most specific STRUCTURAL connection they share: a mechanism / dynamic /
feedback loop / trade-off / mathematical form such that a non-obvious feature of one
predicts a feature of the other. One or two plain-prose sentences — concrete, mechanistic,
falsifiable (someone could argue against it). NOT "both involve change" / "both are
important". If they share no real transferable structure, respond exactly: NO_CONNECTION."""


async def articulate(mm, sem, pair):
    """Pair -> SynthesisCandidate (or None if the model finds no real structure)."""
    async with sem:
        raw = await mm.generate_once(
            DISCOVERY_PROMPT.format(a=pair["a"], b=pair["b"]),
            model_name=SYNTHESIS_COHERENCE_MODEL, system_prompt=DISCOVERY_SYS,
            max_tokens=220, temperature=0.6, disable_reasoning=True)
    claim = (raw or "").strip()
    if not claim or "NO_CONNECTION" in claim.upper() or len(claim.split()) < 5:
        return None
    return SynthesisCandidate(
        concept_a=pair["a"], concept_b=pair["b"], connection_claim=claim,
        # walk_path[0] != "retrieval" -> midrange-peak distance scoring; two distinct
        # concept names as source_domains clear the >=2 domain gate (the coherence
        # judge, not domain_crossing, is what we're measuring).
        walk_path=["anchored", pair["a"].lower(), pair["b"].lower()],
        source_domains={pair["a"], pair["b"]},
        endpoint_distance=round(1.0 - float(pair["cos"]), 4),
        timestamp=datetime.now())


async def run_arm(label, pairs, mm, filt, sem):
    cands = [c for c in await asyncio.gather(*(articulate(mm, sem, p) for p in pairs)) if c]
    print(f"  [{label}] articulated {len(cands)}/{len(pairs)} (NO_CONNECTION: {len(pairs)-len(cands)})", flush=True)
    results = await asyncio.gather(*(filt.process_candidate(c) for c in cands))
    return cands, results


def tally(label, n_pairs, cands, results):
    lvl = Counter((r.coherence_level.name if r.coherence_level else "—") for r in results)
    stages = Counter((r.rejection_stage or "accepted") for r in results)
    n_art = len(cands)
    mod_strong = sum(1 for r in results if r.coherence_level and r.coherence_level.name in ("MODERATE", "STRONG"))
    accepted = sum(1 for r in results if (r.rejection_stage or "accepted") == "accepted")
    known = sum(1 for r in results if r.rejection_stage == "novelty_external")
    # "found real structure" = judge saw it (MOD/STRONG) OR it's a documented rediscovery
    real = mod_strong + known
    return {
        "label": label, "n_pairs": n_pairs, "n_articulated": n_art,
        "coherence": dict(lvl), "rejection_stages": dict(stages),
        "mod_strong": mod_strong, "accepted": accepted, "known": known, "real": real,
        "real_rate": real / n_art if n_art else 0.0,
        "mod_strong_rate": mod_strong / n_art if n_art else 0.0,
    }


def show(t):
    print(f"\n{t['label']}: pairs={t['n_pairs']} articulated={t['n_articulated']}")
    print(f"  coherence levels   : {t['coherence']}")
    print(f"  rejection stages   : {t['rejection_stages']}")
    print(f"  MODERATE+STRONG    : {t['mod_strong']}  ({t['mod_strong_rate']:.0%} of articulated)")
    print(f"  accepted           : {t['accepted']}")
    print(f"  known rediscoveries: {t['known']}")
    print(f"  REAL structure (MOD/STRONG + known): {t['real']}  ({t['real_rate']:.0%} of articulated)")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=int, default=14)
    ap.add_argument("--per-anchor", type=int, default=2)
    ap.add_argument("--k", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("Validate ANCHORED discovery generation through the REAL filter")
    print(f"model={SYNTHESIS_COHERENCE_MODEL} anchors={args.anchors} per_anchor={args.per_anchor}\n")

    idx = get_index(); idx.load()
    if not idx.loaded:
        print("ERROR: FAISS wiki index not available — cannot run."); sys.exit(1)
    try:
        idx.index.nprobe = max(int(getattr(idx.index, "nprobe", 1)), 48)
    except Exception:
        pass

    from synthesis_controlled_distance import ANCHORS
    from synthesis_discovery_miner import POOL
    cc = CosCache(idx)
    anchors = ANCHORS[: args.anchors]
    universe = sorted(set(POOL) | set(ANCHORS))

    print("--- Arm A: anchored (FAISS-neighbourhood partners, cos<0.45) ---", flush=True)
    arm_a_pairs, used = build_arm_a(idx, cc, anchors, args.k, args.per_anchor, rng)
    print(f"Arm A: {len(arm_a_pairs)} pairs", flush=True)
    print("--- Arm B: random pool pairs, distance-matched to Arm A ---", flush=True)
    arm_b_pairs, *_ = build_arm_b(cc, universe, arm_a_pairs, used, rng)
    print(f"Arm B: {len(arm_b_pairs)} pairs\n", flush=True)

    mm = ModelManager()
    tmp = tempfile.mkdtemp(prefix="anchored_gen_val_")
    store = MultiCollectionChromaStore(persist_directory=tmp)
    filt = SynthesisFilter(chroma_store=store, model_manager=mm, synthesis_memory=SynthesisMemory(store))
    filt.graph_memory = None
    filt.entity_resolver = None
    sem = asyncio.Semaphore(4)

    print("--- Articulating + filtering both arms (real SynthesisFilter) ---", flush=True)
    a_cands, a_res = await run_arm("anchored", arm_a_pairs, mm, filt, sem)
    b_cands, b_res = await run_arm("random", arm_b_pairs, mm, filt, sem)

    A = tally("ANCHORED", len(arm_a_pairs), a_cands, a_res)
    B = tally("RANDOM (dist-matched)", len(arm_b_pairs), b_cands, b_res)
    print("\n" + "=" * 78)
    show(A); show(B)
    print("\n" + "=" * 78)
    # significance on the "found real structure" rate (per articulated candidate)
    st = two_prop_z(A["real"], A["n_articulated"], B["real"], B["n_articulated"])
    stc = two_prop_z(A["mod_strong"], A["n_articulated"], B["mod_strong"], B["n_articulated"])
    print(f"REAL-structure rate:  anchored {A['real_rate']:.0%}  vs  random {B['real_rate']:.0%}"
          f"   gap {st['diff']:+.0%}  z={st['z']:.2f}  p={st['p']:.4f}")
    print(f"MOD+STRONG rate    :  anchored {A['mod_strong_rate']:.0%}  vs  random {B['mod_strong_rate']:.0%}"
          f"   gap {stc['diff']:+.0%}  z={stc['z']:.2f}  p={stc['p']:.4f}")
    better = A["real_rate"] > B["real_rate"]
    print("\nVERDICT:", "ANCHORED carries real structure the judge recognizes more often than random"
          " — productionizing the anchored selection is justified." if better else
          "anchored does NOT beat random through the real filter — inspect before wiring.")

    import json
    ts = datetime.now()
    outdir = os.path.join(_REPO, "data", "synthesis_discovery")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"anchored_gen_validation_{ts:%Y%m%d_%H%M%S}.json")
    json.dump({"timestamp": ts.isoformat(timespec="seconds"), "params": vars(args),
               "anchored": A, "random": B, "stats_real": st, "stats_mod_strong": stc,
               "anchored_claims": [{"a": c.concept_a, "b": c.concept_b, "claim": c.connection_claim,
                                    "level": r.coherence_level.name if r.coherence_level else None,
                                    "stage": r.rejection_stage or "accepted"}
                                   for c, r in zip(a_cands, a_res)],
               "random_claims": [{"a": c.concept_a, "b": c.concept_b, "claim": c.connection_claim,
                                  "level": r.coherence_level.name if r.coherence_level else None,
                                  "stage": r.rejection_stage or "accepted"}
                                 for c, r in zip(b_cands, b_res)]},
              open(out, "w"), indent=2, default=str)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
