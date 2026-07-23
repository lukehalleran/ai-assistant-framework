#!/usr/bin/env python3
"""
Re-validate the doc-co-occurrence oracle's BIDIRECTIONAL mode on the n=99 labeled set.

Question this answers: requiring the text-mention in BOTH directions strips one-way stem
collisions (precision up) but will lose genuinely-known pairs the corpus only discusses
one-way (recall down). HOW MUCH recall does it cost, and is the FP reduction worth it? That
number decides whether bidirectionality becomes the oracle's default / gets used downstream
(miner, live filter) or stays a selective precision knob.

Reuses the labeled sets + design from `synthesis_oracle_validation.py` (KNOWN / UNRELATED /
RANDOM_POOL) — imported, not duplicated. Every pair is scored under BOTH modes in one pass;
retrieval is memoized so the second mode is ~free (identical chunks; only the AND/OR of the
two directions differs). Lists exactly which KNOWN pairs flip known→novel (the recall cost)
and which FPs get eliminated (the precision gain).

Free — FAISS only, no LLM. Usage: python scripts/validate_oracle_bidirectional.py [--depth 40 --seed 13]
"""
import argparse
import functools
import os
import random
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)

import numpy as np
import knowledge.doc_cooccurrence as _docco
from knowledge.doc_cooccurrence import doc_cooccurrence
from knowledge.semantic_search import get_index
from synthesis_oracle_validation import KNOWN, UNRELATED, RANDOM_POOL  # reuse labeled sets


def main():
    ap = argparse.ArgumentParser(description="Re-validate bidirectional oracle mode @ n=99")
    ap.add_argument("--depth", type=int, default=40)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--random-pairs", type=int, default=40)
    args = ap.parse_args()
    random.seed(args.seed)

    idx = get_index(); idx.load()
    if not idx.loaded:
        print("ERROR: FAISS wiki index not available."); sys.exit(1)
    try:
        idx.index.nprobe = max(int(getattr(idx.index, "nprobe", 1)), 48)
    except Exception:
        pass
    emb = idx.embedder

    # Memoize retrieval so scoring each pair under both modes costs ~1x FAISS work.
    _orig = _docco.semantic_search_with_neighbors
    _memo = functools.lru_cache(maxsize=8192)(lambda q, k=8: _orig(q, k))
    _docco.semantic_search_with_neighbors = lambda q, k=8: _memo(q, k)

    print(f"FAISS rows={idx._total_rows:,} | depth={args.depth} | comparing default vs bidirectional\n")

    def cos(a, b):
        v = emb.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
        return float(np.dot(v[0], v[1]))

    def both(a, b):
        rd = doc_cooccurrence(a, b, depth=args.depth, bidirectional=False)
        rb = doc_cooccurrence(a, b, depth=args.depth, bidirectional=True)
        return rd, rb

    # ---- KNOWN: recall under both modes, split by cosine; track flips (recall lost) ----
    rec = {"def": {"all": 0, "hard": 0, "easy": 0}, "bi": {"all": 0, "hard": 0, "easy": 0}}
    hard_total = easy_total = 0
    flips, fn_both = [], []   # flips: known under default, novel under bidirectional
    for a, b in KNOWN:
        rd, rb = both(a, b)
        c = cos(a, b)
        bucket = "hard" if c < 0.45 else "easy"
        if bucket == "hard":
            hard_total += 1
        else:
            easy_total += 1
        rec["def"]["all"] += int(rd.known); rec["def"][bucket] += int(rd.known)
        rec["bi"]["all"] += int(rb.known); rec["bi"][bucket] += int(rb.known)
        if rd.known and not rb.known:
            # only mention-based knowns can flip (shared-title is unaffected by the flag)
            flips.append((a, b, c, "via shared-title?" if rd.shared else "mention-only"))
        if not rd.known and not rb.known:
            fn_both.append((a, b, c))

    # ---- UNRELATED: clean FP under both modes ----
    fp_def, fp_bi, fp_eliminated = [], [], []
    for a, b in UNRELATED:
        rd, rb = both(a, b)
        if rd.known:
            fp_def.append((a, b, cos(a, b), rd.shared, rd.mention))
        if rb.known:
            fp_bi.append((a, b, cos(a, b), rb.shared, rb.mention))
        if rd.known and not rb.known:
            fp_eliminated.append((a, b))

    # ---- RANDOM: FP at scale under both modes ----
    rand_pairs, seen = [], set()
    while len(rand_pairs) < args.random_pairs and len(seen) < 4000:
        a, b = random.sample(RANDOM_POOL, 2)
        key = frozenset((a, b))
        if key in seen:
            seen.add(key); continue
        seen.add(key); rand_pairs.append((a, b))
    rand_def, rand_bi, rand_eliminated = [], [], []
    for a, b in rand_pairs:
        rd, rb = both(a, b)
        if rd.known:
            rand_def.append((a, b, cos(a, b)))
        if rb.known:
            rand_bi.append((a, b, cos(a, b)))
        if rd.known and not rb.known:
            rand_eliminated.append((a, b, cos(a, b)))

    nK = len(KNOWN)
    # ---- report ----
    print("=" * 76)
    print("ORACLE BIDIRECTIONAL RE-VALIDATION — default (either-dir) vs bidirectional")
    print("=" * 76)
    print(f"  {'metric':<28}{'default':>14}{'bidirectional':>16}{'delta':>10}")

    def row(label, d, b, denom, good_down=False):
        dr, br = d / denom if denom else 0, b / denom if denom else 0
        arrow = ""
        if good_down:
            arrow = "  (FP↓ good)" if b < d else ("  (FP↑)" if b > d else "")
        else:
            arrow = "  (recall↓)" if b < d else ""
        print(f"  {label:<28}{d}/{denom} ({dr:.0%}){'':<2}{b}/{denom} ({br:.0%}){'':<2}"
              f"{(br-dr):+.0%}{arrow}")

    row("KNOWN recall (all)", rec["def"]["all"], rec["bi"]["all"], nK)
    row("  hard (cos<0.45)", rec["def"]["hard"], rec["bi"]["hard"], hard_total)
    row("  easy (cos>=0.45)", rec["def"]["easy"], rec["bi"]["easy"], easy_total)
    row("UNRELATED FP (clean)", len(fp_def), len(fp_bi), len(UNRELATED), good_down=True)
    row("RANDOM FP (scale)", len(rand_def), len(rand_bi), len(rand_pairs), good_down=True)
    print("=" * 76)
    print(f"Recall cost: {len(flips)} known pair(s) lost.  FP eliminated: clean "
          f"{len(fp_eliminated)}, scale {len(rand_eliminated)}.")

    if flips:
        print("\nKNOWN pairs LOST under bidirectional (the recall cost):")
        for a, b, c, note in flips:
            print(f"  cos={c:.2f} [{note}] {a} + {b}")
    if fn_both:
        print("\nFalse negatives in BOTH modes (already missed by default):")
        for a, b, c in fn_both:
            print(f"  cos={c:.2f} | {a} + {b}")
    if fp_eliminated:
        print("\nClean FPs ELIMINATED by bidirectional (the precision gain):")
        for a, b in fp_eliminated:
            print(f"  {a} + {b}")
    if rand_eliminated:
        print("\nRandom-pair FPs eliminated by bidirectional:")
        for a, b, c in rand_eliminated:
            print(f"  cos={c:.2f} | {a} + {b}")

    print("\nREAD: bidirectional is worth making the default if recall (esp. hard subset)")
    print("holds while clean/scale FP drops. If hard recall craters, keep it a selective")
    print("precision knob (e.g. unsupervised mining) rather than the oracle's default.")


if __name__ == "__main__":
    main()
