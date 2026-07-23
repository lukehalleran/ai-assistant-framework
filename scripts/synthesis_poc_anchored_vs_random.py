#!/usr/bin/env python3
"""
Anchored-vs-random proof-of-concept for the synthesis generator.

ONE falsifiable claim, measured (not tuned):
    "Anchored pair-selection surfaces literature-known cross-domain connections at a
     HIGHER rate than random pairing AT THE SAME SEMANTIC DISTANCE."
If anchored does NOT beat a distance-matched random baseline, the generator does not
find real non-obvious structure. That is a valid, reportable result.

This script ADDS nothing to the core synthesis stack — it imports and orchestrates the
already-validated pieces:
  - knowledge/doc_cooccurrence.py        -> doc_cooccurrence() is the SCORER (the
        validated 97%-recall / ~4%-FP known-oracle). Scoring is doc-co-occurrence, NOT
        cosine — cosine is circular with band selection and blind to cross-domain pairs.
  - scripts/synthesis_controlled_distance -> ANCHORS + concept-quality filters + true-cos.
  - scripts/synthesis_discovery_miner     -> the diverse concept POOL + the --save layout.
  - knowledge/semantic_search             -> the FAISS index (get_index / true cosine).

DESIGN
  * Whole experiment is restricted to the NON-OBVIOUS band: true-cosine < 0.45 (the
    cross-domain / discovery quadrant). Close pairs are trivially known; uninteresting.
  * Arm A (anchored): for each prominent ANCHOR, take partners from the anchor's FAISS
    retrieval neighbourhood (concept-only filter), recompute TRUE cosine, keep those
    < 0.45. The "anchored signal" under test = *being in the anchor's retrieval
    neighbourhood* on top of being at this distance.
  * Arm B (random): random concept pairs from the curated pool, RESAMPLED so Arm B's
    true-cosine distribution matches Arm A's per 0.05 bin. This distance-matching is the
    whole experiment; without it any gap is a distance artifact. Where a bin can't be
    fully matched we take what's available and PRINT both histograms so the residual
    confound is visible.
  * Pairs are de-duped across arms (by unordered concept set) — no pair in both.
  * Every pair is scored by doc_cooccurrence(); for each KNOWN hit we log the match TYPE
    (shared-title vs text-mention) so a skeptic can check the gap isn't driven by
    common-word text-mention collisions (the 28%-miner-overfire failure mode).

PRE-REGISTERED SUCCESS CRITERION (fixed before looking at results):
    PASS = rate_A > rate_B at matched distance, the gap is significant (p < 0.05), AND
    the gap SURVIVES excluding text-mention-only hits (shared-title-only re-test still
    positive and significant). Anything else => generator does not beat baseline.

DISCLOSED METHODOLOGY CHOICES / CONFOUNDS (reported, not worked around):
  1. Arm B's random universe is POOL u ANCHORS (both curated prominent-concept lists) so
     distance-matching in the moderate band has enough candidates. Documented in output.
  2. Partner-type asymmetry: Arm A partners are FAISS-neighbourhood wiki titles (one
     prominent anchor + a possibly-less-prominent neighbour); Arm B pairs are two curated
     prominent concepts. More-prominent concepts have more wiki coverage -> more
     co-occurrence opportunity, so this biases the baseline UP, i.e. AGAINST Arm A. A
     positive Arm A result is therefore conservative; a negative one is partly confounded.
  3. A retrieval memo (pure-function cache of identical FAISS queries) is applied to speed
     the run. It is result-preserving (deterministic query->result); it does not alter
     the oracle's logic or any output. Disable with --no-cache.

Free of LLMs and ChromaDB — FAISS + embedder only. Does NOT require Daemon stopped.

Usage:
  python scripts/synthesis_poc_anchored_vs_random.py
  python scripts/synthesis_poc_anchored_vs_random.py --per-anchor 3 --k 1500 --depth 40
  python scripts/synthesis_poc_anchored_vs_random.py --save data/synthesis_discovery/poc
"""
import argparse
import functools
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)  # import sibling scripts (controlled_distance, discovery_miner)

# --- SCORER: the validated doc-co-occurrence known-oracle (imported, not rebuilt) ---
import knowledge.doc_cooccurrence as _docco
from knowledge.doc_cooccurrence import doc_cooccurrence
from knowledge.semantic_search import get_index
# --- Anchored sampler pieces + the diverse pool (imported, not rebuilt) ---
from synthesis_controlled_distance import ANCHORS, _looks_like_concept, _same_concept  # noqa: E402
from synthesis_discovery_miner import POOL  # noqa: E402

MAX_COS = 0.45   # non-obvious / cross-domain band ceiling — fixed by the design
BIN_W = 0.05     # distance-matching bin width


# ----------------------------------------------------------------------------------
# Statistics (dependency-free; scipy used only as an optional Fisher cross-check)
# ----------------------------------------------------------------------------------
def _phi(x):
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_prop_z(kA, nA, kB, nB):
    """Two-proportion z-test (pooled) + Wald 95% CI on the difference (pA - pB)."""
    if nA == 0 or nB == 0:
        return {"pA": 0.0, "pB": 0.0, "diff": 0.0, "z": 0.0, "p": 1.0, "ci": (0.0, 0.0)}
    pA, pB = kA / nA, kB / nB
    pool = (kA + kB) / (nA + nB)
    se_pool = math.sqrt(pool * (1 - pool) * (1 / nA + 1 / nB)) if 0 < pool < 1 else 0.0
    z = (pA - pB) / se_pool if se_pool > 0 else 0.0
    p = 2 * (1 - _phi(abs(z)))
    se_diff = math.sqrt(pA * (1 - pA) / nA + pB * (1 - pB) / nB)
    ci = ((pA - pB) - 1.96 * se_diff, (pA - pB) + 1.96 * se_diff)
    return {"pA": pA, "pB": pB, "diff": pA - pB, "z": z, "p": p, "ci": ci}


def _fisher(kA, nA, kB, nB):
    """Optional Fisher's exact p (two-sided) if scipy is present, else None."""
    try:
        from scipy.stats import fisher_exact  # type: ignore
        _, p = fisher_exact([[kA, nA - kA], [kB, nB - kB]], alternative="two-sided")
        return float(p)
    except Exception:
        return None


# ----------------------------------------------------------------------------------
# Cosine cache — true cosine = normalized-embedding dot (the metric _cos / _pick_partners
# use; FAISS inner product is NOT used to define distance, per controlled_distance).
# ----------------------------------------------------------------------------------
class CosCache:
    def __init__(self, idx):
        self.idx = idx
        self._vec = {}

    def vecs(self, names):
        missing = [n for n in names if n not in self._vec]
        if missing:
            embs = self.idx.embedder.encode(missing, convert_to_numpy=True,
                                            normalize_embeddings=True)
            for n, e in zip(missing, embs):
                self._vec[n] = e
        return np.array([self._vec[n] for n in names])

    def cos(self, a, b):
        va, vb = self.vecs([a, b])
        return float(np.dot(va, vb))


def _binof(c):
    return min(int(c / BIN_W), int(MAX_COS / BIN_W) - 1)


# ----------------------------------------------------------------------------------
# Arm A — anchored selection (FAISS neighbourhood partners, controlled distance < 0.45)
# ----------------------------------------------------------------------------------
def build_arm_a(idx, cc, anchors, k, per_anchor, rng):
    pairs, used = [], set()
    for anchor in anchors:
        # MEMORY-SAFE retrieval (16GB box): idx.search() reads the full 'text' column for
        # all k rows — a multi-GB spike at k=1500 x 24 anchors, which OOM'd the machine.
        # We only need partner TITLES here (cosine is recomputed from titles; the oracle
        # re-reads text itself later, bounded + memoized). So: raw FAISS search -> ids,
        # then read ONLY the small 'title' column. The huge text column is never touched.
        q = idx._encode_query(anchor)
        try:
            _D, I = idx.index.search(q, int(k))
        except Exception as e:
            print(f"  [armA] FAISS search failed for '{anchor}': {e}", flush=True)
            continue
        ids = [int(i) for i in I[0] if i >= 0 and int(i) < idx._total_rows]
        meta = idx._read_rows(ids, columns=["title"])  # title-only -> tiny read
        seen, titles = set(), []
        for i in ids:
            t = (meta.get(i, {}).get("title") or "").strip()
            tl = t.lower()
            if not t or tl in seen:
                continue
            if _same_concept(anchor, t) or not _looks_like_concept(t):
                continue
            seen.add(tl)
            titles.append(t)
        if not titles:
            print(f"  [armA] no concept-like neighbours for '{anchor}'", flush=True)
            continue
        # true cosine of every concept-like neighbour vs the anchor (one batch encode)
        embs = idx.embedder.encode([anchor] + titles, convert_to_numpy=True,
                                   normalize_embeddings=True)
        qa = embs[0]
        inband = [(t, float(np.dot(qa, embs[i + 1])))
                  for i, t in enumerate(titles)
                  if float(np.dot(qa, embs[i + 1])) < MAX_COS]
        rng.shuffle(inband)
        taken = 0
        for t, c in inband:
            if taken >= per_anchor:
                break
            key = frozenset((anchor.lower(), t.lower()))
            if key in used:
                continue
            used.add(key)
            # seed the cos cache so Arm B uses an identical vector for shared concepts
            cc._vec.setdefault(anchor, qa)
            pairs.append({"anchor": anchor, "a": anchor, "b": t, "cos": c})
            taken += 1
        print(f"  [armA] {anchor:28} -> {taken} partner(s) in band "
              f"(of {len(inband)} <{MAX_COS})", flush=True)
    return pairs, used


# ----------------------------------------------------------------------------------
# Arm B — random concept pairs from the curated pool, distance-matched to Arm A
# ----------------------------------------------------------------------------------
def build_arm_b(cc, universe, arm_a, used, rng):
    concepts = sorted(set(universe))
    cc.vecs(concepts)  # warm cache
    # Arm A target counts per bin
    target = defaultdict(int)
    for p in arm_a:
        target[_binof(p["cos"])] += 1
    # candidate random pairs by bin (exclude any pair already used by Arm A)
    candbin = defaultdict(list)
    avail_total = 0
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            a, b = concepts[i], concepts[j]
            c = cc.cos(a, b)
            if c >= MAX_COS:
                continue
            if frozenset((a.lower(), b.lower())) in used:
                continue
            candbin[_binof(c)].append({"a": a, "b": b, "cos": c})
            avail_total += 1
    armb, matched, available = [], defaultdict(int), {}
    for b in sorted(target):
        cands = list(candbin.get(b, []))
        available[b] = len(cands)
        rng.shuffle(cands)
        take = cands[: target[b]]
        matched[b] = len(take)
        for row in take:
            used.add(frozenset((row["a"].lower(), row["b"].lower())))
            armb.append(row)
    return armb, target, matched, available, avail_total


# ----------------------------------------------------------------------------------
# Scoring — the validated oracle; record known + match type per pair
# ----------------------------------------------------------------------------------
def score_arm(label, pairs, depth, bidirectional=False):
    scored = []
    for n, p in enumerate(pairs, 1):
        r = doc_cooccurrence(p["a"], p["b"], depth=depth, bidirectional=bidirectional)
        mtype = "shared-title" if r.shared >= 1 else ("text-mention" if r.mention else "none")
        scored.append({**p, "known": bool(r.known), "shared": r.shared,
                       "shared_titles": list(r.shared_titles), "mention": bool(r.mention),
                       "match_type": mtype})
        if n % 10 == 0:
            print(f"  [{label}] scored {n}/{len(pairs)}", flush=True)
    return scored


def arm_stats(scored):
    n = len(scored)
    known = [s for s in scored if s["known"]]
    shared_title = [s for s in known if s["shared"] >= 1]
    text_only = [s for s in known if s["shared"] == 0 and s["mention"]]
    return {
        "n": n,
        "known": len(known),
        "known_strict": len(shared_title),   # shared-title only
        "shared_title": len(shared_title),
        "text_mention_only": len(text_only),
        "rate": len(known) / n if n else 0.0,
        "rate_strict": len(shared_title) / n if n else 0.0,
        "mean_cos": (sum(s["cos"] for s in scored) / n) if n else 0.0,
    }


def hist(pairs):
    h = defaultdict(int)
    for p in pairs:
        h[_binof(p["cos"])] += 1
    return h


def _save(base, payload, arm_a, arm_b):
    if base.endswith((".json", ".md")):
        base = base.rsplit(".", 1)[0]
    os.makedirs(os.path.dirname(os.path.abspath(base)) or ".", exist_ok=True)
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    A, B = payload["arm_a"], payload["arm_b"]
    md = [
        f"# Anchored-vs-random synthesis PoC — {payload['timestamp']}",
        "",
        f"**Claim:** anchored pair-selection beats distance-matched random pairing on "
        f"literature-known rate, in the non-obvious band (cos < {MAX_COS}).",
        "",
        f"- Arm A (anchored): n={A['n']}, known {A['known']}/{A['n']} = **{A['rate']:.1%}** "
        f"(shared-title {A['shared_title']}, text-mention-only {A['text_mention_only']}), "
        f"mean cos {A['mean_cos']:.3f}",
        f"- Arm B (random, distance-matched): n={B['n']}, known {B['known']}/{B['n']} = "
        f"**{B['rate']:.1%}** (shared-title {B['shared_title']}, text-mention-only "
        f"{B['text_mention_only']}), mean cos {B['mean_cos']:.3f}",
        "",
        f"- **Gap** = {payload['stats']['diff']:+.1%}  |  z={payload['stats']['z']:.2f}  "
        f"p={payload['stats']['p']:.4f}  95% CI [{payload['stats']['ci'][0]:+.1%}, "
        f"{payload['stats']['ci'][1]:+.1%}]"
        + (f"  Fisher p={payload['stats']['fisher_p']:.4f}"
           if payload['stats'].get('fisher_p') is not None else ""),
        f"- **Shared-title-only re-test** (text-mention excluded): A {A['rate_strict']:.1%} "
        f"vs B {B['rate_strict']:.1%}, gap {payload['stats_strict']['diff']:+.1%}, "
        f"p={payload['stats_strict']['p']:.4f}",
        "",
        f"## VERDICT: {payload['verdict']}",
        "",
        "### Distance-match histograms (count per 0.05 cosine bin)",
        "",
        "| bin | Arm A | Arm B target | Arm B matched | Arm B available |",
        "|-----|------:|------:|------:|------:|",
    ]
    for b in range(int(MAX_COS / BIN_W)):
        lo, hi = b * BIN_W, (b + 1) * BIN_W
        md.append(f"| {lo:.2f}-{hi:.2f} | {payload['hist_a'].get(str(b), 0)} | "
                  f"{payload['target'].get(str(b), 0)} | {payload['matched'].get(str(b), 0)} "
                  f"| {payload['available'].get(str(b), 0)} |")
    md += ["", "### Arm A known pairs", "", "| cos | match | pair |", "|----:|----|----|"]
    for s in sorted([s for s in arm_a if s["known"]], key=lambda s: s["cos"]):
        via = f" (via {', '.join(s['shared_titles'][:2])})" if s["shared_titles"] else ""
        md.append(f"| {s['cos']:.2f} | {s['match_type']} | {s['a']} ↔ {s['b']}{via} |")
    md += ["", "### Arm B known pairs", "", "| cos | match | pair |", "|----:|----|----|"]
    for s in sorted([s for s in arm_b if s["known"]], key=lambda s: s["cos"]):
        via = f" (via {', '.join(s['shared_titles'][:2])})" if s["shared_titles"] else ""
        md.append(f"| {s['cos']:.2f} | {s['match_type']} | {s['a']} ↔ {s['b']}{via} |")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return base + ".json"


def main():
    ap = argparse.ArgumentParser(description="Anchored-vs-random synthesis PoC")
    ap.add_argument("--anchors", type=int, default=len(ANCHORS),
                    help="How many ANCHORS to use for Arm A (default: all)")
    ap.add_argument("--per-anchor", type=int, default=3,
                    help="Max in-band partners per anchor (Arm A size lever)")
    ap.add_argument("--k", type=int, default=1500, help="FAISS neighbour depth per anchor")
    ap.add_argument("--depth", type=int, default=40, help="Oracle retrieval depth (validated=40)")
    ap.add_argument("--bidirectional", action="store_true",
                    help="Require text-mention in BOTH directions (strips one-way stem collisions)")
    ap.add_argument("--seed", type=int, default=13, help="RNG seed")
    ap.add_argument("--no-cache", action="store_true",
                    help="Disable the result-preserving retrieval memo")
    ap.add_argument("--save", nargs="?", const="AUTO", default="AUTO", metavar="PATH",
                    help="Persist run (default: timestamped under data/synthesis_discovery/)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    idx = get_index()
    idx.load()
    if not idx.loaded:
        print("ERROR: FAISS wiki index not available — cannot run. (Methodology blocker.)")
        sys.exit(1)
    try:
        idx.index.nprobe = max(int(getattr(idx.index, "nprobe", 1)), 48)
    except Exception:
        pass

    # Result-preserving retrieval memo: identical (query, k) -> identical FAISS result.
    # Does NOT change the oracle's logic or outputs; purely avoids re-reading wiki rows.
    if not args.no_cache:
        _orig = _docco.semantic_search_with_neighbors
        _memo = functools.lru_cache(maxsize=8192)(lambda q, k=8: _orig(q, k))
        _docco.semantic_search_with_neighbors = lambda q, k=8: _memo(q, k)

    cc = CosCache(idx)
    anchors = ANCHORS[: args.anchors]
    universe = sorted(set(POOL) | set(ANCHORS))  # Arm B random universe (disclosed choice)

    print("=" * 78)
    print("ANCHORED-vs-RANDOM SYNTHESIS PoC")
    print("=" * 78)
    print(f"FAISS rows={idx._total_rows:,} nprobe={getattr(idx.index,'nprobe','?')} | "
          f"band: cos<{MAX_COS} | k={args.k} depth={args.depth} "
          f"per_anchor={args.per_anchor} seed={args.seed}")
    print(f"Arm A anchors={len(anchors)} | Arm B universe={len(universe)} concepts "
          f"(POOL u ANCHORS)\n")
    print("PRE-REGISTERED PASS = rate_A>rate_B AND p<0.05 AND survives shared-title-only re-test.\n")

    print("--- Building Arm A (anchored, FAISS-neighbourhood partners) ---", flush=True)
    arm_a_pairs, used = build_arm_a(idx, cc, anchors, args.k, args.per_anchor, rng)
    print(f"\nArm A: {len(arm_a_pairs)} pairs (target >=50)\n", flush=True)

    print("--- Building Arm B (random pool pairs, distance-matched) ---", flush=True)
    arm_b_pairs, target, matched, available, avail_total = build_arm_b(
        cc, universe, arm_a_pairs, used, rng)
    print(f"Arm B: {len(arm_b_pairs)} pairs matched (of {avail_total} <{MAX_COS} candidates)\n",
          flush=True)

    mode = "BIDIRECTIONAL (text-mention requires both directions)" if args.bidirectional \
           else "either-direction (validated default)"
    print(f"--- Scoring with doc_cooccurrence oracle [{mode}] ---", flush=True)
    arm_a = score_arm("armA", arm_a_pairs, args.depth, args.bidirectional)
    arm_b = score_arm("armB", arm_b_pairs, args.depth, args.bidirectional)

    A, B = arm_stats(arm_a), arm_stats(arm_b)
    st = two_prop_z(A["known"], A["n"], B["known"], B["n"])
    st["fisher_p"] = _fisher(A["known"], A["n"], B["known"], B["n"])
    st_strict = two_prop_z(A["known_strict"], A["n"], B["known_strict"], B["n"])

    # ---- Pre-registered verdict ----
    pass_gap = A["rate"] > B["rate"]
    pass_sig = st["p"] < 0.05
    pass_strict = (A["rate_strict"] > B["rate_strict"]) and (st_strict["p"] < 0.05)
    PASS = pass_gap and pass_sig and pass_strict
    verdict = "PASS — anchored beats distance-matched random" if PASS else \
              "FAIL — anchored does NOT beat distance-matched random baseline"

    ha, tgt = hist(arm_a_pairs), {str(k): v for k, v in target.items()}

    # ---------------- headline ----------------
    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"  {'arm':<26}{'n':>5}{'known':>8}{'rate':>8}{'shared-T':>10}{'text-only':>11}{'meanCos':>9}")
    print(f"  {'A anchored':<26}{A['n']:>5}{A['known']:>8}{A['rate']:>7.1%}"
          f"{A['shared_title']:>10}{A['text_mention_only']:>11}{A['mean_cos']:>9.3f}")
    print(f"  {'B random (dist-matched)':<26}{B['n']:>5}{B['known']:>8}{B['rate']:>7.1%}"
          f"{B['shared_title']:>10}{B['text_mention_only']:>11}{B['mean_cos']:>9.3f}")
    print("-" * 78)
    print(f"  GAP (A - B) = {st['diff']:+.1%}   z={st['z']:.2f}   p={st['p']:.4f}"
          f"   95% CI [{st['ci'][0]:+.1%}, {st['ci'][1]:+.1%}]"
          + (f"   Fisher p={st['fisher_p']:.4f}" if st["fisher_p"] is not None else ""))
    print(f"  SHARED-TITLE-ONLY re-test: A {A['rate_strict']:.1%} vs B {B['rate_strict']:.1%}"
          f"  gap {st_strict['diff']:+.1%}  p={st_strict['p']:.4f}")
    print("-" * 78)
    print("  Distance-match histogram (count per 0.05 bin):")
    print(f"    {'bin':<12}{'A':>5}{'B target':>10}{'B matched':>11}{'B avail':>9}")
    undermatched = False
    for b in range(int(MAX_COS / BIN_W)):
        a_n, t_n, m_n, v_n = ha.get(b, 0), target.get(b, 0), matched.get(b, 0), available.get(b, 0)
        if t_n != m_n:
            undermatched = True
        if a_n or v_n:
            flag = "  <-- under-matched" if t_n != m_n else ""
            print(f"    {b*BIN_W:.2f}-{(b+1)*BIN_W:.2f} {a_n:>5}{t_n:>10}{m_n:>11}{v_n:>9}{flag}")
    print("=" * 78)
    if undermatched:
        print("NOTE: some bins could not be fully distance-matched (Arm B pool too thin in "
              "the moderate band). Residual distance confound is visible above — interpret "
              "the gap with the histograms in view.")
    if A["n"] < 50:
        print(f"NOTE: Arm A n={A['n']} < 50 target — fewer in-band concept partners than hoped; "
              "statistical power is limited. Reported as-is (not worked around).")
    print(f"\n  VERDICT: {verdict}")
    print(f"    rate_A>rate_B={pass_gap} | p<0.05={pass_sig} | survives-strict={pass_strict}")

    # ---------------- save ----------------
    ts = datetime.now()
    if args.save and args.save != "AUTO":
        base = args.save
    else:
        out_dir = os.path.join(_REPO_ROOT, "data", "synthesis_discovery")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, f"poc_anchored_vs_random_{ts:%Y%m%d_%H%M%S}")
    payload = {
        "timestamp": ts.isoformat(timespec="seconds"),
        "claim": "anchored pair-selection beats distance-matched random pairing on "
                 "literature-known rate in the non-obvious band (cos<0.45)",
        "params": {"max_cos": MAX_COS, "bin_w": BIN_W, "k": args.k, "depth": args.depth,
                   "per_anchor": args.per_anchor, "seed": args.seed,
                   "bidirectional": args.bidirectional,
                   "anchors_used": len(anchors), "armB_universe": len(universe)},
        "methodology_notes": [
            "Scorer is doc_cooccurrence (validated known-oracle), NOT cosine.",
            "Arm A partners = FAISS-neighbourhood concept titles at true-cos<0.45.",
            "Arm B = random pairs from POOL u ANCHORS, resampled to match Arm A's per-bin cosine.",
            "Partner-type asymmetry biases the baseline UP (against Arm A) -> positive result conservative.",
            "Retrieval memo is result-preserving (pure-function cache).",
        ],
        "arm_a": A, "arm_b": B,
        "stats": {**st, "ci": list(st["ci"])},
        "stats_strict": {**st_strict, "ci": list(st_strict["ci"])},
        "hist_a": {str(k): v for k, v in ha.items()},
        "target": tgt,
        "matched": {str(k): v for k, v in matched.items()},
        "available": {str(k): v for k, v in available.items()},
        "undermatched": undermatched,
        "verdict": verdict,
        "pass_components": {"rate_gt": pass_gap, "significant": pass_sig, "survives_strict": pass_strict},
        "arm_a_pairs": arm_a,
        "arm_b_pairs": arm_b,
    }
    path = _save(base, payload, arm_a, arm_b)
    print(f"\nSaved run -> {path}\n  + readable table -> {path[:-5]}.md")


if __name__ == "__main__":
    main()
