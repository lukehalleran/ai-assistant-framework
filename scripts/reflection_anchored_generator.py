#!/usr/bin/env python3
"""
The ANCHORED-selection reflection generator (R = the user's own life-graph).

WHY (the validation harness result, 2026-06-30): random cross-domain pairing is
~90-100% declines — randomly-chosen distant entity pairs share no real mechanism, so
distance over random pairs predicts DECLINE, not value. Reflections are RARE ANCHORED
bridges: the only cross-domain claims that survived were the hand-picked gold anchors,
where a real recurring SELF-mechanism (e.g. dad→authority/evaluation) happens to bridge
two distant areas. This is the reflection analogue of the discovery arm's anchored-vs-
random result (anchored ≫ random). So the generator must ANCHOR on the bridge, not the
pair: mine the recurring mechanism FIRST, then find which distant entities it connects.

PIPELINE (inverts the random harness — fix the bridge, find the pair):
  Stage 1 MINE  — one LLM pass over the person's LIFE-DOMAIN structure (real entities
                  grouped by domain): name recurring SELF-mechanisms that operate across
                  >=2 DISTINCT domains, and for each pick one anchor ENTITY from each of
                  two DIFFERENT domains it governs. The mechanism is the anchor.
  Stage 2 GROUND+GATE — for each anchored (entity_a, entity_b) candidate, run the SAME
                  pair-conditioned generator + truth/dual-grounding gate as the harness
                  (reused verbatim): write the first-person claim grounded ONLY in those
                  two entities' REAL facts, then gate mechanism-validity + dual-grounding.
  DISTANCE FILTER — keep only cross-domain candidates (different clusters); rank by
                  entity-cosine distance (the locked metric). Same-domain anchors = thin.

YIELD COMPARISON is the validation: cross-domain truth-PASS rate here vs the random
harness's ~0. If anchoring lifts cross-domain yield well above random, anchored selection
is the right generator and distance is confirmed as a same-domain FILTER, not a ranker.

Substrate LOCKED (reflection_domain_clustering.py mcs=5 artifact). Read-only on production
memory; LLM calls only; writes a JSON artifact. ~1 + 2*K model calls.
Usage: python scripts/reflection_anchored_generator.py [k_mechanisms]
"""
import asyncio
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

# Reuse the LOCKED substrate + the harness's pair-builder, gate, and per-pair runner.
from scripts.reflection_validation_harness import (
    load_substrate, eligible_life_entities, build_pair, run_pair, resolve,
)

K_MECHANISMS = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].isdigit()) else 12
MAX_PER_DOMAIN = 10   # entities listed per domain in the mining prompt

MINE_SYSTEM = (
    "You are a perceptive reflective observer for ONE person, working ONLY from the real "
    "structure of their own life. Your job is to find RECURRING underlying MECHANISMS — "
    "tunable psychological/behavioral patterns (how they handle authority, evaluation, "
    "control, reward, depletion, belonging, risk…) — that show up in MORE THAN ONE distinct "
    "area of their life. A real mechanism is a factor whose change would alter behavior; it "
    "is NOT a mood, a category, or a surface theme. You are mining anchors for later "
    "verification, so propose your strongest, most genuinely cross-cutting patterns."
)
MINE_PROMPT = """Here are the distinct DOMAINS of one person's life, each with real entities from their memory:

{domains}

Find the recurring SELF-MECHANISMS that bridge DIFFERENT domains. For each mechanism:
  - name the mechanism as a tunable factor (e.g. "how much an authority I depend on controls
    a resource I need", "whether action is self-initiated vs externally evaluated",
    "spending tomorrow's reserve for today's output");
  - pick ONE specific anchor ENTITY from each of TWO DIFFERENT domains above where the SAME
    mechanism operates. The two entities MUST come from two different domains — never two
    entities from the same domain (that would be thin, same-area consistency, not a bridge).
  - in one line, say how the mechanism shows up in each of the two entities' area.

Only use entities that actually appear above. Prefer pairs whose two domains are far apart
(family vs coursework, health vs career) over near ones. Quality over quantity — propose only
mechanisms that genuinely recur across a domain boundary; if you can find few, return few.

Return ONLY a JSON array of up to {k} items, each:
  {{"mechanism": "<tunable factor>", "domain_a": "<domain label>", "entity_a": "<entity>",
    "domain_b": "<domain label>", "entity_b": "<entity>", "how": "<one line, both sides>"}}
"""


def build_domain_listing(sub, life):
    by_cluster = defaultdict(list)
    for e in life:
        by_cluster[e["cluster"]].append(e)
    lines = []
    for cid, es in sorted(by_cluster.items(), key=lambda kv: -len(kv[1])):
        label = sub["cluster_meta"].get(cid, {}).get("label", f"c{cid}")
        names = [e["name"] for e in es][:MAX_PER_DOMAIN]
        lines.append(f"DOMAIN {label}: " + ", ".join(names))
    return "\n".join(lines)


def _parse_mined(raw):
    """Lenient: parse the whole array, else SALVAGE complete {...} objects from a
    truncated array (the mining call can hit max_tokens mid-array)."""
    if not raw:
        return []
    candidates = []
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                candidates = data
        except Exception:
            candidates = []
    if not candidates:  # salvage object-by-object from a truncated/maformed array
        for om in re.finditer(r"\{[^{}]*\}", raw, re.DOTALL):
            try:
                candidates.append(json.loads(om.group(0)))
            except Exception:
                continue
    out = []
    for d in candidates:
        if isinstance(d, dict) and d.get("entity_a") and d.get("entity_b") and d.get("mechanism"):
            out.append(d)
    return out


async def mine_anchors(mm, sub, life):
    from config.app_config import SYNTHESIS_COHERENCE_MODEL
    prompt = MINE_PROMPT.format(domains=build_domain_listing(sub, life), k=K_MECHANISMS)
    raw = await mm.generate_once(
        prompt, model_name=SYNTHESIS_COHERENCE_MODEL, system_prompt=MINE_SYSTEM,
        max_tokens=2600, temperature=0.5, disable_reasoning=True)
    return _parse_mined(raw), raw


def to_pairs(sub, mined):
    """Resolve each mined (entity_a, entity_b) to substrate entities; keep CROSS-domain only."""
    pairs, dropped = [], []
    for d in mined:
        ra, rb = resolve(sub, d["entity_a"]), resolve(sub, d["entity_b"])
        a, b = sub["by_id"].get(ra), sub["by_id"].get(rb)
        if not a or not b:
            dropped.append((d, "unresolved")); continue
        if a["entity_id"] == b["entity_id"]:
            dropped.append((d, "same_entity")); continue
        p = build_pair(sub, a, b, "anchored")
        p["mechanism_seed"] = d.get("mechanism", "")
        p["how_seed"] = d.get("how", "")
        if p["same_domain"]:
            dropped.append((d, f"same_domain({p['domain_a']})")); continue
        pairs.append(p)
    return pairs, dropped


async def main():
    from models.model_manager import ModelManager
    print(f"ANCHORED reflection generator (R=self) | k={K_MECHANISMS}\n")
    sub = load_substrate()
    life = eligible_life_entities(sub)
    mm = ModelManager()

    print("\nStage 1 — mining recurring cross-domain self-mechanisms...", flush=True)
    mined, raw = await mine_anchors(mm, sub, life)
    print(f"Mined {len(mined)} candidate mechanisms.")
    pairs, dropped = to_pairs(sub, mined)
    print(f"Anchored CROSS-domain pairs after resolve+cross-filter: {len(pairs)} "
          f"(dropped {len(dropped)}: {Counter(r for _, r in dropped)})")
    for p in pairs:
        print(f"  [{p['domain_a']} ⟂ {p['domain_b']}] {p['name_a']} ↔ {p['name_b']}  d={p['ent_dist']:.2f}")
        print(f"      mech: {p['mechanism_seed'][:110]}")
    if not pairs:
        print("\nNo anchored cross-domain pairs resolved — inspect raw:\n" + (raw or "")[:600])
        return

    print(f"\nStage 2 — ground + gate {len(pairs)} anchored pairs...", flush=True)
    sem = asyncio.Semaphore(4)
    done = await asyncio.gather(*(run_pair(mm, sem, sub, p) for p in pairs))

    print("\n" + "=" * 100)
    for p in sorted(done, key=lambda x: -x["ent_dist"]):
        verdict = "DECLINED" if p["declined"] else ("TRUTH-PASS" if p["truth_pass"] else f"truth-fail({p['rating']})")
        print(f"[{p['domain_a'][:14]:14} ⟂ {p['domain_b'][:14]:14}] d={p['ent_dist']:.2f} "
              f"{p['name_a'][:14]:14}↔{p['name_b'][:14]:14} {verdict}")
        if not p["declined"]:
            print(f"      {p['claim'][:160]}")
    print("=" * 100)

    n = len(done)
    decl = sum(p["declined"] for p in done)
    tp = sum(p["truth_pass"] for p in done)
    print(f"\nANCHORED yield over {n} cross-domain pairs: declined={decl}  truth_pass={tp}  "
          f"({100*tp/n:.0f}% pass, {100*decl/n:.0f}% decline)")
    # NOTE: this baseline is a HARDCODED historical reference (harness cross-strata run,
    # 2026-06-29), NOT recomputed this run — a live comparison requires re-running the
    # random arm side-by-side. tp>0 against a static string is suggestive, not a test.
    print("RANDOM baseline (STATIC reference, harness cross strata 2026-06-29): "
          "cross truth_pass=0/30, decline~93% — not recomputed this run")
    if n:
        print(f"=> anchored cross-domain pass rate {100*tp/n:.0f}%  vs  static random ~0% "
              + ("— suggestive that anchoring helps (small n, static baseline — "
                 "re-run the random arm for a live comparison before citing)."
                 if tp > 0 else "— no pass; inspect grounding strictness / mining quality."))

    ts = datetime.now()
    outdir = os.path.join(_REPO, "data", "synthesis_discovery")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"reflection_anchored_{ts:%Y%m%d_%H%M%S}.json")
    json.dump({"timestamp": ts.isoformat(timespec="seconds"), "artifact": sub["artifact"],
               "k": K_MECHANISMS, "n_mined": len(mined), "raw_mining": raw,
               "dropped": [{"cand": d, "why": w} for d, w in dropped],
               "pairs": done}, open(out, "w"), indent=2, default=str)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
