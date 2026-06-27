#!/usr/bin/env python3
"""
Controlled-distance, core-concept synthesis experiment (validation next-lever).

The wiki<->wiki experiment (docs/SYNTHESIS_VALIDATION.md) proved the bottleneck is
PAIR SELECTION, not the corpus or the judge: random obscure pairs -> ~0% known (the
null baseline). This script implements the two fixes that experiment pointed to:

  FIX 1 — core-concept anchors (not random obscure stubs). Pair A is drawn from a
          curated list of PROMINENT cross-domain concepts. The list is curated for
          prominence, NOT for connectedness (that would game the base-rate); the
          PAIRING is algorithmic, so the known-rate stays an honest signal.

  FIX 2 — moderate-distance partner (not random-far). Pair B is drawn from concept
          A's FAISS neighborhood at a controlled cosine band: related enough to
          plausibly connect, far enough to be non-obvious / cross-topic.

It runs THREE bands per anchor in one pass so the output interprets itself:

  near     — top FAISS neighbors (high cosine, same subtopic): trivially related,
             expected high-but-obvious known-rate. The "too close" failure mode.
  moderate — the controlled sweet spot (FIX 2): the hypothesis says this is where
             non-obvious-yet-real connections live -> known-rate should beat random.
  random   — a random corpus article (true far baseline). This IS the wiki<->wiki
             null baseline (~0% known). Any lift of moderate over random = signal.

For each (A, B): articulate a bridge (the generator's own LLM), then run
  - novelty_external (FAISS): is this connection already KNOWN in literature?
  - the coherence judge (fixed): is it a real structural isomorphism?

The read: if moderate's known-rate (and coherence-pass) clears the random baseline,
controlled-distance sampling finds real, non-obvious structure — the thing the
generators were missing. If moderate ~= random, distance isn't the lever and we look
elsewhere. Either way the experiment is decisive, unlike a bare 0%.

Usage:
  python scripts/synthesis_controlled_distance.py --anchors 10
  python scripts/synthesis_controlled_distance.py --anchors 24 --bands moderate,random
  python scripts/synthesis_controlled_distance.py --band-lo 0.30 --band-hi 0.50 --k 1500

Makes real LLM calls (articulation + judge). Needs the DB free (Daemon stopped).
"""

import argparse
import asyncio
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.app_config import CHROMA_PATH
from models.model_manager import ModelManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from memory.synthesis_memory import SynthesisMemory
from knowledge.synthesis_generator import SynthesisGenerator
from knowledge.synthesis_filter import SynthesisFilter
from knowledge.synthesis_models import SynthesisCandidate, SynthesisResult, CoherenceLevel
from knowledge.semantic_search import get_index

_PASS = {CoherenceLevel.MODERATE, CoherenceLevel.STRONG}

# FIX 1: prominent cross-domain concept anchors. Each is a substantive, well-documented
# Wikipedia concept with genuine cross-field reach — chosen for PROMINENCE, blind to which
# pairs connect. (Contrast the wiki<->wiki run's "Palio di Castellanza", "Zubulake I".)
ANCHORS = [
    "entropy", "natural selection", "feedback loop", "phase transition",
    "diffusion", "resonance", "equilibrium", "optimization", "percolation theory",
    "homeostasis", "game theory", "oscillation", "evolution", "supply and demand",
    "immune system", "recursion", "chaos theory", "symmetry", "epidemic",
    "gradient descent", "tragedy of the commons", "metabolism",
    "error detection and correction", "network theory",
]


def _candidate(name_a, name_b, claim, dist):
    return SynthesisCandidate(
        concept_a=name_a, concept_b=name_b, connection_claim=claim,
        walk_path=["wiki", name_a.lower(), name_b.lower(), "wiki"],
        source_domains={"wikipedia_a", "wikipedia_b"},
        endpoint_distance=float(max(0.0, min(1.0, dist))), timestamp=datetime.now(),
    )


def _same_concept(anchor: str, title: str) -> bool:
    """True if `title` is just the anchor itself (or a trivial substring variant)."""
    a, t = anchor.lower().strip(), (title or "").lower().strip()
    return not t or a in t or t in a


def _cos(idx, a: str, b: str) -> float:
    """Cosine between two short strings via the shared embedder (normalized -> dot)."""
    va = idx._encode_query(a)[0]
    vb = idx._encode_query(b)[0]
    return float(np.dot(va, vb))


# Partner-quality filter — the FAISS neighborhood of a concept is full of people,
# tools, lists and dated events ("Harry Kesten", "Velvet assembler", "List of screamo
# bands"), which aren't concepts and pollute the pairing. Keep concept-like titles only.
_BLOCK_SUBSTR = (
    "list of", "(journal)", "(disambiguation)", "(magazine)", "(newspaper)",
    "(album)", "(song)", "(film)", "(tv series)", "(band)", "awards", "discography",
    "filmography", "cricket team", "football club", " f.c.", " a.f.c.", " atv",
    "championship", "season)", "election", "census",
)
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
# Likely a person: "Firstname Lastname" / "Firstname M. Lastname", no concept lowercase word.
_PERSON_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+$")


def _looks_like_concept(title: str) -> bool:
    t = title.strip()
    tl = t.lower()
    if len(t) < 3:
        return False
    if any(s in tl for s in _BLOCK_SUBSTR):
        return False
    if _YEAR_RE.search(t) and len(t.split()) <= 4:  # dated event, not a concept
        return False
    if _PERSON_RE.match(t):                          # bare person name
        return False
    return True


def _pick_partners(idx, anchor, neighbors, bands, band_lo, band_hi, near_thr):
    """Choose one B per requested band from the anchor's FAISS neighborhood.

    Distance is scored with TRUE cosine (FAISS returns un-normalized inner product —
    values can exceed 1.0 — so it cannot define the band). Concept-only titles are kept.
    Returns dict band -> (title, context, cosine) or None if that band has no member.
    """
    # Clean: drop same-concept, de-dup, keep concept-like titles only.
    seen = set()
    cand = []  # (title, context)
    for r in neighbors:
        title = (r.get("title") or "").strip()
        tl = title.lower()
        if _same_concept(anchor, title) or tl in seen:
            continue
        if not _looks_like_concept(title):
            continue
        seen.add(tl)
        cand.append((title, (r.get("content") or "")[:300]))
    if not cand:
        return {}

    # Batch true-cosine of every candidate title vs the anchor (one encode call).
    titles = [c[0] for c in cand]
    embs = idx.embedder.encode([anchor] + titles, convert_to_numpy=True,
                               normalize_embeddings=True)
    qa = embs[0]
    scored = [(t, ctx, float(np.dot(qa, embs[i + 1])))
              for i, (t, ctx) in enumerate(cand)]

    out = {}
    if "near" in bands:
        pool = [c for c in scored if c[2] >= near_thr]
        out["near"] = random.choice(pool) if pool else None
    if "moderate" in bands:
        pool = [c for c in scored if band_lo <= c[2] < band_hi]
        out["moderate"] = random.choice(pool) if pool else None
    return out


def _random_corpus_b(idx, anchor):
    """True far baseline: a random corpus article + its cosine to the anchor."""
    for _ in range(8):
        ridx = random.randrange(idx._total_rows)
        data = idx._read_rows([ridx])
        row = data.get(ridx)
        if not row:
            continue
        title = (row.get("title") or "").strip()
        text = (row.get("text") or row.get("content") or "")
        if not (title or text):
            continue
        name = title or text.split(".")[0][:80].strip()
        if _same_concept(anchor, name):
            continue
        return name, text[:300], _cos(idx, anchor, name)
    return None


async def _run_pair(gen, filt, anchor, ctx_a, name_b, ctx_b, sim, band):
    """Articulate + novelty(known) + judge for one (anchor, B) pair. Returns a row dict."""
    claim = await gen._articulate_bridge(anchor, name_b, "Wikipedia", "Wikipedia", ctx_a, ctx_b)
    if not claim or "NO_CONNECTION" in claim.upper():
        print(f"  [{band:8}] no-connection            sim={sim:.2f} | {anchor} <-> {name_b}")
        return {"band": band, "articulated": False}

    result = SynthesisResult(candidate=_candidate(anchor, name_b, claim, 1.0 - sim))
    nov = await filt._stage_3_novelty_external(result)
    known = not nov.passed  # rejected by novelty == already in literature
    await filt._stage_5_coherence_judge(result)
    coh = result.coherence_level
    coh_pass = coh in _PASS

    print(f"  [{band:8}] {'KNOWN' if known else 'novel':5} coh={coh.name if coh else 'N/A':9}"
          f"{'PASS' if coh_pass else 'fail'} sim={sim:.2f} | {anchor} <-> {name_b}")
    return {"band": band, "articulated": True, "known": known,
            "coh_pass": coh_pass, "coh": coh.name if coh else "N/A"}


async def main():
    ap = argparse.ArgumentParser(description="Controlled-distance synthesis experiment")
    ap.add_argument("--anchors", type=int, default=10, help="Anchor concepts to test (default 10)")
    ap.add_argument("--bands", default="near,moderate,random",
                    help="Comma list of bands to run (near,moderate,random)")
    ap.add_argument("--k", type=int, default=1500, help="FAISS neighbor depth per anchor")
    ap.add_argument("--band-lo", type=float, default=0.30, help="Moderate band lower cosine")
    ap.add_argument("--band-hi", type=float, default=0.55, help="Moderate band upper cosine")
    # near-thr == band-hi keeps the bands contiguous (near >= 0.55, moderate [0.30,0.55)),
    # both now on the TRUE-cosine scale (not FAISS inner product).
    ap.add_argument("--near-thr", type=float, default=0.55, help="Near band min cosine")
    ap.add_argument("--seed", type=int, default=13, help="RNG seed for reproducibility")
    args = ap.parse_args()

    random.seed(args.seed)
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]

    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    mm = ModelManager()
    sm = SynthesisMemory(store)
    gen = SynthesisGenerator(chroma_store=store, model_manager=mm)
    filt = SynthesisFilter(chroma_store=store, model_manager=mm, synthesis_memory=sm)

    idx = get_index()
    idx.load()
    if not idx.loaded:
        print("ERROR: FAISS wiki index not available."); sys.exit(1)
    # Deeper IVF probing so the moderate band is actually reachable.
    try:
        idx.index.nprobe = max(int(getattr(idx.index, "nprobe", 1)), 48)
    except Exception:
        pass
    print(f"FAISS wiki vectors: {idx._total_rows:,} | nprobe={getattr(idx.index, 'nprobe', '?')}")
    print(f"bands={bands} | moderate=[{args.band_lo},{args.band_hi}) near>={args.near_thr} "
          f"k={args.k}\n")

    anchors = ANCHORS[:args.anchors]
    rows = []
    for anchor in anchors:
        neighbors = idx.search(anchor, k=args.k)
        if not neighbors:
            print(f"  (no neighbors for '{anchor}', skipping)")
            continue
        ctx_a = (neighbors[0].get("content") or "")[:300]  # anchor's own article ~ top hit

        partners = _pick_partners(idx, anchor, neighbors, bands,
                                  args.band_lo, args.band_hi, args.near_thr)
        for band in bands:
            if band == "random":
                rb = _random_corpus_b(idx, anchor)
                if not rb:
                    continue
                name_b, ctx_b, sim = rb
            else:
                p = partners.get(band)
                if not p:
                    continue
                name_b, ctx_b, sim = p
            row = await _run_pair(gen, filt, anchor, ctx_a, name_b, ctx_b, sim, band)
            rows.append(row)

    # ── Aggregate per band ──
    by_band = defaultdict(list)
    for r in rows:
        by_band[r["band"]].append(r)

    print("\n" + "=" * 74)
    print("CONTROLLED-DISTANCE EXPERIMENT — known-rate & coherence by distance band")
    print("=" * 74)
    print(f"  {'band':10} {'pairs':>6} {'artic':>6} {'KNOWN':>10} {'coh-PASS':>10} {'K&coh':>7}")
    for band in bands:
        rs = by_band.get(band, [])
        art = [r for r in rs if r.get("articulated")]
        n = len(art)
        known_n = sum(1 for r in art if r.get("known"))
        coh_n = sum(1 for r in art if r.get("coh_pass"))
        kc = sum(1 for r in art if r.get("known") and r.get("coh_pass"))
        kr = f"{known_n}/{n} ({known_n/n:.0%})" if n else "0/0"
        cr = f"{coh_n}/{n} ({coh_n/n:.0%})" if n else "0/0"
        print(f"  {band:10} {len(rs):>6} {n:>6} {kr:>10} {cr:>10} {kc:>7}")
    print("=" * 74)
    print("Read: MODERATE known-rate clearing the RANDOM baseline = controlled-distance\n"
          "sampling finds real, non-obvious structure (the generator's missing lever).\n"
          "NEAR known but obvious; MODERATE ~= RANDOM means distance isn't the lever.")


if __name__ == "__main__":
    asyncio.run(main())
