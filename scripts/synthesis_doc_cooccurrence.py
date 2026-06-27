#!/usr/bin/env python3
"""
Document co-occurrence oracle — the Test-B instrument cosine can't be.

The fixed stage-3 gate uses direct cos(A,B), which equates "known" with "topically
close" — so it MISSES non-obvious cross-domain known connections (the discovery target)
and is circular with the controlled-distance band selection. This oracle is independent
of embedding distance: it asks **do A and B get discussed in the same wiki articles?**

For each pair (A, B):
  - retrieve the top-`depth` wiki article TITLES for query A and for query B (FAISS),
  - shared titles  = |titles(A) ∩ titles(B)|        (co-discussed in the same articles)
  - direct mention = B's name is a retrieved title for A, or vice versa (stronger)
  - known iff shared >= `min_shared` OR direct mention.

Validated here on a labeled set with THREE groups, the middle one being the whole point:
  KNOWN_OBVIOUS     — high cos, same-field (cosine already catches these)
  KNOWN_CROSSDOMAIN — documented together but embedding-DISTANT (cosine MISSES these)
  UNRELATED         — neither (should stay novel)

Success = doc-co-occurrence flags BOTH known groups (incl. the cross-domain one cosine
misses) while leaving UNRELATED novel. cos(A,B) is printed alongside to show the contrast.

Free — FAISS + embedder only, no LLM. Needs the DB free is NOT required (FAISS only).

Usage: python scripts/synthesis_doc_cooccurrence.py --depth 30 --min-shared 1
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from knowledge.semantic_search import semantic_search_with_neighbors, get_index

# Generic tokens that shouldn't count as a distinctive cross-mention.
_GENERIC_TOK = {"theory", "system", "systems", "model", "models", "method", "process",
                "science", "general", "number", "function", "problem", "effect"}

# (A, B) labeled pairs.
KNOWN_OBVIOUS = [
    ("entropy", "thermodynamics"), ("diffusion", "transport phenomena"),
    ("natural selection", "evolution"), ("equilibrium", "steady state"),
]
KNOWN_CROSSDOMAIN = [  # documented together, but embedding-distant — cosine misses these
    ("percolation theory", "epidemic"), ("simulated annealing", "metallurgy"),
    ("game theory", "evolution"), ("Kalman filter", "Bayesian inference"),
    ("network science", "epidemiology"), ("information theory", "thermodynamics"),
]
UNRELATED = [
    ("entropy", "cricket"), ("homeostasis", "bridge"),
    ("optimization", "marina"), ("diffusion", "Kepler"),
    ("natural selection", "Delta Goodrem"),
]


def _norm(t):
    return (t or "").strip().lower()


def _stems(phrase):
    """Distinctive 6-char stems of a concept's content tokens (len>=6, non-generic)."""
    toks = [t for t in re.findall(r"[a-z]+", phrase.lower())
            if len(t) >= 6 and t not in _GENERIC_TOK]
    return {t[:6] for t in toks} or {t[:6] for t in re.findall(r"[a-z]+", phrase.lower())
                                     if len(t) >= 4 and t not in _GENERIC_TOK}


def doc_cooccurrence(a, b, depth):
    """Independent-of-cosine known signal: do A and B get discussed together?

    Two signals over each concept's top-`depth` retrieved chunks:
      - shared TITLES (co-listed in the same articles), and
      - cross-MENTION: B's distinctive term appears in the TEXT of A's articles, or
        vice versa (the crossover analogy is in the body, not the title).
    """
    ra = semantic_search_with_neighbors(a, k=depth)
    rb = semantic_search_with_neighbors(b, k=depth)
    ta = {_norm(r.get("title")) for r in ra if r.get("title")}; ta.discard("")
    tb = {_norm(r.get("title")) for r in rb if r.get("title")}; tb.discard("")
    shared = ta & tb

    a_text = " ".join((r.get("content") or r.get("text") or "") for r in ra).lower()
    b_text = " ".join((r.get("content") or r.get("text") or "") for r in rb).lower()
    a_stems, b_stems = _stems(a), _stems(b)
    b_in_a = any(s in a_text for s in b_stems)   # B discussed inside A's articles
    a_in_b = any(s in b_text for s in a_stems)   # A discussed inside B's articles
    mention = b_in_a or a_in_b
    return len(shared), sorted(shared)[:4], mention


def main():
    ap = argparse.ArgumentParser(description="Document co-occurrence oracle validation")
    ap.add_argument("--depth", type=int, default=30, help="Top-K titles per concept")
    ap.add_argument("--min-shared", type=int, default=1, help="Shared titles to call KNOWN")
    args = ap.parse_args()

    idx = get_index(); idx.load()
    if not idx.loaded:
        print("ERROR: FAISS wiki index not available."); sys.exit(1)
    try:
        idx.index.nprobe = max(int(getattr(idx.index, "nprobe", 1)), 48)
    except Exception:
        pass
    emb = idx.embedder
    print(f"FAISS rows={idx._total_rows:,} | depth={args.depth} min_shared={args.min_shared}\n")

    def cos(a, b):
        v = emb.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
        return float(np.dot(v[0], v[1]))

    def run(label, pairs, want_known):
        print(f"== {label} (want known={want_known}) ==")
        hits = 0
        for a, b in pairs:
            shared, examples, mention = doc_cooccurrence(a, b, args.depth)
            known = shared >= args.min_shared or mention
            hits += int(known == want_known)
            c = cos(a, b)
            flag = "KNOWN" if known else "novel"
            tag = "cos-MISS" if (want_known and c < 0.45 and known) else ""
            ex = f" e.g. {examples[:2]}" if examples else ""
            print(f"  [{flag:5}] shared={shared:2} mention={int(mention)} cos={c:5.2f} "
                  f"{tag:8} | {a} + {b}{ex}")
        print(f"  -> {hits}/{len(pairs)} correct\n")
        return hits, len(pairs)

    ko = run("KNOWN_OBVIOUS", KNOWN_OBVIOUS, True)
    kc = run("KNOWN_CROSSDOMAIN", KNOWN_CROSSDOMAIN, True)
    ur = run("UNRELATED", UNRELATED, False)

    tot_correct = ko[0] + kc[0] + ur[0]
    tot = ko[1] + kc[1] + ur[1]
    print("=" * 70)
    print(f"DOC-CO-OCCURRENCE ORACLE accuracy: {tot_correct}/{tot} ({tot_correct/tot:.0%})")
    print(f"  cross-domain known caught: {kc[0]}/{kc[1]}  <- the pairs cosine MISSES")
    print(f"  unrelated kept novel:      {ur[0]}/{ur[1]}  <- false-positive control")
    print("=" * 70)
    print("Win = cross-domain knowns caught (cosine can't) AND unrelated stays novel.\n"
          "If so, this is a non-circular Test-B oracle independent of embedding distance.")


if __name__ == "__main__":
    main()
