#!/usr/bin/env python3
"""
The COMPLETE reflection scorer, end-to-end: GROUNDED-TRUTH × DOMAIN-DISTANCE × SELF-NOVELTY
(R = the user's own life-graph). Integrates the three validated axes into one pipeline that
surfaces final reflections.

  [1] ANCHORED GENERATION  (reflection_anchored_generator): mine recurring SELF-mechanisms that
      span >=2 distinct life-domains -> pick one anchor entity per domain -> cross-domain pairs.
      (Observed: anchored 3/11 truth-passes vs a prior random run's 0/30 — suggestive, small n,
      baseline not re-run side-by-side; the bridge is the anchor.)
  [2] GROUNDED-TRUTH GATE  (run_pair): grounded generator + de-jargon/counterfactual + DUAL
      GROUNDING (each half traceable to that entity's real memory). truth_pass = mechanism valid
      AND both halves grounded. (Lenient FLOOR — discrimination is NOT here.)
  [3] DOMAIN-DISTANCE      (inverted-U, PROVISIONAL — pending hand-label validation): value appears
      to peak at moderate entity-cosine distance (~0.18-0.25, n=11); near = thin (same life-region),
      far = unrelated. distance_value().
  [4] SELF-NOVELTY         (SelfNoveltyOracle): has the user already RECOGNIZED this mechanism?
      cooldown + growable mechanism registry, claim-level (NOT embedding) match.

  SURFACED  iff  truth_pass AND distance_value>0 AND self_novel.
  reflection_score = truth_pass · distance_value · self_novel.
  Surfaced reflections are recorded (registry grows) so the same mechanism isn't re-surfaced.

Read-only on production memory; writes the mechanism registry + a JSON artifact.
Usage:
  python scripts/reflection_pipeline.py [k_mechanisms]
  python scripts/reflection_pipeline.py --seed-known   # demo: pre-seed a mechanism, watch self-novelty reject it
"""
import asyncio
import json
import os
import sys
from datetime import datetime

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from scripts.reflection_validation_harness import load_substrate, eligible_life_entities, run_pair
from scripts.reflection_anchored_generator import mine_anchors, to_pairs
from scripts.reflection_self_novelty import SelfNoveltyOracle

# PROVISIONAL constants — eyeballed from n=11 anchored pairs, single mining seed
# (docs/REFLECTION_SYNTHESIS_UNIFIED_SCORER.md §7f). NOT yet validated against hand
# labels: re-fit after the 16-claim hand-label via the harness's CHECK 2 (which now
# scores this deployed distance_value(), not a monotone AUC on raw distance).
SWEET_PEAK = 0.215   # inverted-U peak (apparent sweet spot 0.18-0.25)
SWEET_HALF = 0.13    # zero below ~0.085 / above ~0.345


def distance_value(d: float) -> float:
    """Inverted-U value of entity-cosine distance: thin when near, unrelated when far."""
    v = 1.0 - abs(d - SWEET_PEAK) / SWEET_HALF
    return round(max(0.0, v), 3)


async def main():
    seed_known = "--seed-known" in sys.argv
    k = next((int(a) for a in sys.argv[1:] if a.isdigit()), 12)
    print(f"REFLECTION pipeline (truth × distance × self-novelty) | k={k}"
          + ("  [demo: seeded registry]" if seed_known else "") + "\n")

    from models.model_manager import ModelManager
    sub = load_substrate()
    life = eligible_life_entities(sub)
    mm = ModelManager()
    oracle = SelfNoveltyOracle(mm)

    if seed_known:
        # Demo: pre-register a mechanism so self-novelty has something to reject against.
        oracle.registry.append({"mechanism": "I outsource the verdict on my own body/state to an "
                                "external authority, and when I feel judged I reach for a way to "
                                "regulate myself", "entities": ["dad", "kavarin"]})
        print(f"[demo] seeded registry with 1 recognized mechanism\n")

    # [1] anchored generation
    print("[1] mining anchored cross-domain self-mechanisms...", flush=True)
    mined, _ = await mine_anchors(mm, sub, life)
    pairs, dropped = to_pairs(sub, mined)
    print(f"    {len(pairs)} anchored cross-domain pairs (dropped {len(dropped)})")
    if not pairs:
        print("no anchored pairs — aborting."); return

    # [2] grounded-truth gate
    print("[2] grounded-truth + dual-grounding gate...", flush=True)
    sem = asyncio.Semaphore(4)
    gated = await asyncio.gather(*(run_pair(mm, sem, sub, p) for p in pairs))

    # [3]+[4] distance value + self-novelty (only for truth-passers, sequential so the registry
    # updates as we surface — a later candidate with the same mechanism is then self-known)
    print("[3]+[4] distance value + self-novelty...", flush=True)
    final = []
    for p in gated:
        p["distance_value"] = distance_value(p["ent_dist"])
        p["self_novel"] = None
        p["self_novel_reason"] = ""
        if p["truth_pass"] and p["distance_value"] > 0:
            sn = await oracle.score(p.get("mechanism_seed") or p.get("mechanism", ""),
                                    p["name_a"], p["name_b"])
            p["self_novel"] = sn["self_novel"]
            p["self_novel_reason"] = sn["reason"]
            p["nearest_known"] = sn.get("nearest_known", "")
        surfaced = bool(p["truth_pass"] and p["distance_value"] > 0 and p["self_novel"])
        p["surfaced"] = surfaced
        p["reflection_score"] = round((1.0 if p["truth_pass"] else 0.0) * p["distance_value"]
                                      * (1.0 if p["self_novel"] else 0.0), 3)
        if surfaced:
            oracle.record_surfaced(p.get("mechanism_seed") or p.get("mechanism", ""), p["name_a"], p["name_b"])
        final.append(p)

    # report
    print("\n" + "=" * 100)
    surfaced = [p for p in final if p["surfaced"]]
    for p in sorted(final, key=lambda x: -x["reflection_score"]):
        if p["declined"]:
            tag = "declined"
        elif not p["truth_pass"]:
            tag = f"truth-fail({p['rating']})"
        elif p["distance_value"] == 0:
            tag = f"thin/far d={p['ent_dist']:.2f}"
        elif not p["self_novel"]:
            tag = f"self-known ({p['self_novel_reason']})"
        else:
            tag = "★ SURFACED"
        print(f"[{p['domain_a'][:13]:13}⟂{p['domain_b'][:13]:13}] d={p['ent_dist']:.2f} "
              f"dv={p['distance_value']:.2f} score={p['reflection_score']:.2f}  {tag}")
        if p["surfaced"] or (not p["declined"] and p["truth_pass"]):
            print(f"      {p['name_a']}↔{p['name_b']}: {p['claim'][:150]}")
    print("=" * 100)
    n = len(final)
    print(f"pairs={n}  declined={sum(p['declined'] for p in final)}  "
          f"truth_pass={sum(p['truth_pass'] for p in final)}  "
          f"self-known={sum(1 for p in final if p['self_novel'] is False)}  "
          f"★SURFACED={len(surfaced)}")
    print("\nFull three-axis scorer is live: truth (floor) × distance (sweet-spot) × self-novelty "
          "(registry). Surfaced reflections recorded -> registry grows -> won't re-surface.")

    ts = datetime.now()
    out = os.path.join(_REPO, "data", "synthesis_discovery", f"reflection_pipeline_{ts:%Y%m%d_%H%M%S}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"timestamp": ts.isoformat(timespec="seconds"), "k": k, "seed_known": seed_known,
               "n_surfaced": len(surfaced), "results": final}, open(out, "w"), indent=2, default=str)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
