#!/usr/bin/env python3
"""
Validate text-mention gates vs default / bidirectional on the n=99 set.

The discovery miner over-fired (28%) on common-word concept stems (contro, social, supply,
demand, networ) colliding via stem-substring with unrelated articles. We need a gate that
kills those collisions WITHOUT cratering recall the way bidirectionality does (hard 96%→36%).

Two ideas, both measured here against default (recall ceiling) and bidirectional (FP floor):

  distinctive@frac  — a mention counts only if ≥1 matched stem is RARE across the working
                      concept set (DF/N ≤ frac). FAILED in the first pass: DF over a
                      topically-clustered set inflates domain-central terms (thermo, networ),
                      so it kills real same-domain links. Kept here for the comparison.

  toptext@K         — POSITION not rarity: the cross-mention must land in the OTHER concept's
                      top-K most-relevant chunks (where its core topic lives), not buried in a
                      tangential chunk. Immune to topical-DF bias AND to common-word concept
                      names. The hypothesis: 'supply' in mitochondria's tangential chunks dies,
                      'anneal' in metallurgy's top chunks survives.

shared-title always counts (strong) in every variant. Reuses KNOWN/UNRELATED/RANDOM_POOL +
the oracle's _stems/_norm; one cached retrieval pass. Free — FAISS only, no LLM.

Usage: python scripts/validate_distinctive_mention.py [--depth 40 --topks 5,10 --df-fracs 0.20]
"""
import argparse
import os
import random
import sys
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)

import numpy as np
from knowledge.doc_cooccurrence import _stems, _norm
from knowledge.semantic_search import semantic_search_with_neighbors, get_index
from synthesis_oracle_validation import KNOWN, UNRELATED, RANDOM_POOL


def main():
    ap = argparse.ArgumentParser(description="Validate mention gates @ n=99")
    ap.add_argument("--depth", type=int, default=40)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--random-pairs", type=int, default=40)
    ap.add_argument("--df-fracs", default="0.20")
    ap.add_argument("--topks", default="5,10")
    args = ap.parse_args()
    random.seed(args.seed)
    fracs = [float(x) for x in args.df_fracs.split(",") if x]
    topks = [int(x) for x in args.topks.split(",") if x]

    idx = get_index(); idx.load()
    if not idx.loaded:
        print("ERROR: FAISS wiki index not available."); sys.exit(1)
    try:
        idx.index.nprobe = max(int(getattr(idx.index, "nprobe", 1)), 48)
    except Exception:
        pass
    emb = idx.embedder
    print(f"FAISS rows={idx._total_rows:,} | depth={args.depth} | topks={topks} fracs={fracs}\n")

    rand_pairs, seen = [], set()
    while len(rand_pairs) < args.random_pairs and len(seen) < 4000:
        a, b = random.sample(RANDOM_POOL, 2)
        key = frozenset((a, b))
        if key in seen:
            seen.add(key); continue
        seen.add(key); rand_pairs.append((a, b))

    concepts = sorted({c for pair in (KNOWN + UNRELATED + rand_pairs) for c in pair}
                      | set(RANDOM_POOL))
    cache = {}
    for i, c in enumerate(concepts, 1):
        r = semantic_search_with_neighbors(c, k=args.depth)  # similarity-sorted desc (post metric-fix)
        titles = {_norm(x.get("title")) for x in r if x.get("title")}; titles.discard("")
        full = " ".join((x.get("content") or x.get("text") or "") for x in r).lower()
        top = {K: " ".join((x.get("content") or x.get("text") or "") for x in r[:K]).lower()
               for K in topks}
        cache[c] = {"titles": titles, "full": full, "top": top, "stems": _stems(c)}
        if i % 25 == 0:
            print(f"  retrieved {i}/{len(concepts)}", flush=True)
    N = len(concepts)

    all_stems = set().union(*(cache[c]["stems"] for c in concepts))
    stem_df = Counter({s: sum(1 for c in concepts if s in cache[c]["full"]) for s in all_stems})

    vecs = {c: v for c, v in zip(concepts,
            emb.encode(concepts, convert_to_numpy=True, normalize_embeddings=True))}
    cos = lambda a, b: float(np.dot(vecs[a], vecs[b]))

    def matched(a, b, field):
        ca, cb = cache[a], cache[b]
        ta = cb["top"][field] if isinstance(field, int) else cb["full"]
        tb = ca["top"][field] if isinstance(field, int) else ca["full"]
        a2b = [s for s in ca["stems"] if s in ta]   # A's stem in B's (top-K or full) text
        b2a = [s for s in cb["stems"] if s in tb]   # B's stem in A's text
        return a2b, b2a

    def shared(a, b):
        return bool(cache[a]["titles"] & cache[b]["titles"])

    # ---- variant predicates (a, b) -> known bool ----
    def v_default(a, b):
        a2b, b2a = matched(a, b, "full"); return shared(a, b) or bool(a2b) or bool(b2a)

    def v_bidir(a, b):
        a2b, b2a = matched(a, b, "full"); return shared(a, b) or (bool(a2b) and bool(b2a))

    def v_distinctive(a, b, frac):
        a2b, b2a = matched(a, b, "full")
        return shared(a, b) or any(stem_df[s] / N <= frac for s in set(a2b) | set(b2a))

    def v_toptext(a, b, K):
        a2b, b2a = matched(a, b, K); return shared(a, b) or bool(a2b) or bool(b2a)

    variants = [("default", v_default), ("bidirectional", v_bidir)]
    for f in fracs:
        variants.append((f"distinctive@{f:.2f}", lambda a, b, f=f: v_distinctive(a, b, f)))
    for K in topks:
        variants.append((f"toptext@{K}", lambda a, b, K=K: v_toptext(a, b, K)))

    rows, fn_detail, fp_detail = {}, {}, {}
    for name, fn in variants:
        rec_all = rec_hard = rec_easy = hard_tot = easy_tot = clean_fp = scale_fp = 0
        fns, fps = [], []
        for a, b in KNOWN:
            k = fn(a, b); c = cos(a, b); rec_all += int(k)
            if c < 0.45:
                hard_tot += 1; rec_hard += int(k)
            else:
                easy_tot += 1; rec_easy += int(k)
            if not k:
                fns.append((a, b, c))
        for a, b in UNRELATED:
            if fn(a, b):
                clean_fp += 1; fps.append((a, b, "clean"))
        for a, b in rand_pairs:
            if fn(a, b):
                scale_fp += 1; fps.append((a, b, "scale"))
        rows[name] = (rec_all, rec_hard, hard_tot, rec_easy, easy_tot, clean_fp, scale_fp)
        fn_detail[name], fp_detail[name] = fns, fps

    nK = len(KNOWN)
    print("\n" + "=" * 86)
    print("MENTION-GATE VARIANTS @ n=99  (target: hard recall near 96% AND FP near 0)")
    print("=" * 86)
    print(f"  {'variant':<18}{'recall(all)':>13}{'recall(hard)':>14}{'recall(easy)':>14}"
          f"{'cleanFP':>9}{'scaleFP':>9}")
    for name, _ in variants:
        ra, rh, ht, re, et, cf, sf = rows[name]
        print(f"  {name:<18}{ra}/{nK} ({ra/nK:.0%}){'':<2}"
              f"{rh}/{ht} ({rh/max(ht,1):.0%}){'':<3}{re}/{et} ({re/max(et,1):.0%}){'':<3}"
              f"{cf}/{len(UNRELATED)} ({cf/len(UNRELATED):.0%}){'':<1}"
              f"{sf}/{len(rand_pairs)} ({sf/max(len(rand_pairs),1):.0%})")
    print("=" * 86)

    # detail for each toptext variant (the candidate win)
    for name, _ in variants:
        if not name.startswith("toptext"):
            continue
        print(f"\n--- {name} ---")
        print("  KNOWN lost:", end=" ")
        if fn_detail[name]:
            print()
            for a, b, c in fn_detail[name]:
                print(f"    cos={c:.2f} | {a} + {b}")
        else:
            print("none")
        print("  FPs surviving:", end=" ")
        if fp_detail[name]:
            print()
            for a, b, kind in fp_detail[name]:
                print(f"    [{kind}] {a} + {b}")
        else:
            print("none")


if __name__ == "__main__":
    main()
