#!/usr/bin/env python3
"""
Harden the document co-occurrence oracle on a bigger labeled set (free — FAISS only).

n=15 was a smoke test. The number that decides whether this oracle is trustworthy is the
FALSE-POSITIVE rate at scale — chiefly the risk that a common-term stem ("optimi", "networ")
coincidentally appears in an unrelated concept's article text. Three groups:

  KNOWN     — genuinely connected pairs (recall; split into cos<0.45 "hard / cross-domain"
              vs cos>=0.45 "easy / same-field" so we see the recall cosine CAN'T provide).
  UNRELATED — hand-curated confidently-unrelated pairs (the clean FP rate).
  RANDOM    — random pairings from a concept pool (FP estimate at scale; an upper bound,
              since a few random pairs are genuinely related).

Reports recall (overall + hard subset), clean FP, scale FP, and lists every FN/FP for
inspection so the matcher can be tuned. No LLM.

Usage: python scripts/synthesis_oracle_validation.py --depth 40 [--seed 13]
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from knowledge.doc_cooccurrence import doc_cooccurrence
from knowledge.semantic_search import get_index

# Genuinely connected (literature-documented) pairs.
KNOWN = [
    # same-field / obvious (cosine catches these too)
    ("entropy", "thermodynamics"), ("diffusion", "transport phenomena"),
    ("natural selection", "evolution"), ("equilibrium", "steady state"),
    ("random walk", "Brownian motion"), ("eigenvalue", "eigenvector"),
    ("neuron", "synapse"), ("renormalization group", "critical phenomena"),
    ("Turing machine", "computability"), ("entropy", "information"),
    # cross-domain (cosine MISSES these — the valuable recall)
    ("simulated annealing", "metallurgy"), ("genetic algorithm", "natural selection"),
    ("Kalman filter", "Bayesian inference"), ("percolation theory", "epidemic"),
    ("game theory", "evolution"), ("information theory", "thermodynamics"),
    ("network science", "epidemiology"), ("Hopfield network", "Ising model"),
    ("reaction-diffusion system", "morphogenesis"), ("power law", "phase transition"),
    ("small-world network", "social network"), ("queueing theory", "traffic flow"),
    ("control theory", "homeostasis"), ("Markov chain", "statistical mechanics"),
    ("maximum entropy", "statistical inference"), ("spin glass", "combinatorial optimization"),
    ("fractal", "coastline"), ("chaos theory", "weather forecasting"),
    ("replicator dynamics", "population genetics"), ("catastrophe theory", "phase transition"),
    ("cellular automaton", "self-organization"), ("neural network", "brain"),
    ("epidemic", "rumor spreading"), ("Bayesian inference", "machine learning"),
]

# Confidently unrelated — the clean false-positive control. Some A-terms are common
# ("optimization", "topology") on purpose, to stress the common-stem FP risk.
UNRELATED = [
    ("entropy", "cricket"), ("homeostasis", "jazz"), ("optimization", "Renaissance painting"),
    ("diffusion", "medieval castle"), ("game theory", "ceramics"), ("percolation", "ballet"),
    ("metabolism", "typography"), ("Bayesian inference", "surfing"),
    ("network science", "Gothic cathedral"), ("immune system", "volcano"),
    ("gradient descent", "Baroque music"), ("plate tectonics", "grammar"),
    ("photosynthesis", "stock market"), ("topology", "horse racing"),
    ("enzyme", "skyscraper"), ("black hole", "etiquette"), ("supply and demand", "mitochondria"),
    ("relativity", "cuisine"), ("quantum entanglement", "agriculture"),
    ("vaccine", "calligraphy"), ("glacier", "violin"), ("polynomial", "perfume"),
    ("antibody", "chess opening"), ("sediment", "saxophone"), ("capacitor", "sonnet"),
]

# Pool for random pairings (FP at scale). Real concepts, randomly paired.
RANDOM_POOL = [
    "entropy", "diffusion", "evolution", "percolation", "homeostasis", "topology",
    "metabolism", "relativity", "photosynthesis", "capacitor", "glacier", "enzyme",
    "polynomial", "antibody", "ballet", "cricket", "jazz", "volcano", "grammar",
    "cuisine", "agriculture", "saxophone", "sonnet", "perfume", "etiquette",
    "skyscraper", "ceramics", "typography", "violin", "calculus", "tectonics",
    "immunology", "rhetoric", "cartography", "thermodynamics", "ecology",
    "metallurgy", "epidemiology", "cryptography", "linguistics",
]


def main():
    ap = argparse.ArgumentParser(description="Doc co-occurrence oracle hardening")
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
    print(f"FAISS rows={idx._total_rows:,} | depth={args.depth}\n")

    def cos(a, b):
        v = emb.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
        return float(np.dot(v[0], v[1]))

    # ---- KNOWN: recall, split by cosine ----
    known_hit = hard_total = hard_hit = easy_total = easy_hit = 0
    fns = []
    for a, b in KNOWN:
        r = doc_cooccurrence(a, b, depth=args.depth)
        c = cos(a, b)
        known_hit += int(r.known)
        if c < 0.45:
            hard_total += 1; hard_hit += int(r.known)
        else:
            easy_total += 1; easy_hit += int(r.known)
        if not r.known:
            fns.append((a, b, c))

    # ---- UNRELATED: clean FP ----
    fps = []
    for a, b in UNRELATED:
        r = doc_cooccurrence(a, b, depth=args.depth)
        if r.known:
            fps.append((a, b, cos(a, b), r.shared, r.mention))

    # ---- RANDOM: FP at scale ----
    rand_pairs, seen = [], set()
    while len(rand_pairs) < args.random_pairs and len(seen) < 4000:
        a, b = random.sample(RANDOM_POOL, 2)
        key = frozenset((a, b))
        if key in seen:
            seen.add(key); continue
        seen.add(key); rand_pairs.append((a, b))
    rand_known = []
    for a, b in rand_pairs:
        if doc_cooccurrence(a, b, depth=args.depth).known:
            rand_known.append((a, b, cos(a, b)))

    # ---- Report ----
    print("=" * 72)
    print("DOC CO-OCCURRENCE ORACLE — hardening on a bigger labeled set")
    print("=" * 72)
    print(f"KNOWN recall:        {known_hit}/{len(KNOWN)} ({known_hit/len(KNOWN):.0%})")
    print(f"  hard (cos<0.45):   {hard_hit}/{hard_total} "
          f"({hard_hit/max(hard_total,1):.0%})  <- recall cosine CANNOT provide")
    print(f"  easy (cos>=0.45):  {easy_hit}/{easy_total} ({easy_hit/max(easy_total,1):.0%})")
    print(f"UNRELATED FP (clean):{len(fps)}/{len(UNRELATED)} ({len(fps)/len(UNRELATED):.0%})")
    print(f"RANDOM FP (scale):   {len(rand_known)}/{len(rand_pairs)} "
          f"({len(rand_known)/max(len(rand_pairs),1):.0%})  <- upper bound (some are real)")
    print("=" * 72)
    if fns:
        print("FALSE NEGATIVES (known, oracle said novel):")
        for a, b, c in fns:
            print(f"  cos={c:.2f} | {a} + {b}")
    if fps:
        print("FALSE POSITIVES (unrelated, oracle said known):")
        for a, b, c, sh, mn in fps:
            print(f"  cos={c:.2f} shared={sh} mention={int(mn)} | {a} + {b}")
    if rand_known:
        print("RANDOM flagged known (inspect — real or FP):")
        for a, b, c in rand_known:
            print(f"  cos={c:.2f} | {a} + {b}")


if __name__ == "__main__":
    main()
