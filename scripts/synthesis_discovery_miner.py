#!/usr/bin/env python3
"""
Discovery miner — surface non-obvious real connections from Wikipedia.

The validated doc co-occurrence oracle (knowledge/doc_cooccurrence.py) detects whether two
concepts are discussed together in literature, independent of embedding distance. The
DISCOVERY QUADRANT is pairs that are doc-co-occurring (real) AND low-cosine (non-obvious):
e.g. `simulated annealing ↔ metallurgy` at cos 0.09. This script mines that quadrant.

Efficient: each concept is retrieved ONCE (the slow part — on-demand parquet reads), its
articles cached, then every pairwise co-occurrence check is a free string op. ~N retrievals
for an N-concept pool, not N² .

Output: the low-cosine + doc-co-occurring pairs, sorted most-non-obvious first. These are
real connections the system found that cosine alone would call "novel" — the proof the
instrument works as a discovery engine. (Novelty-to-you remains the human last mile.)

Free — FAISS + embedder only, no LLM.
Usage: python scripts/synthesis_discovery_miner.py --depth 30 --max-cos 0.40 --top 25
       python scripts/synthesis_discovery_miner.py --save   # also persist the run to data/
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from knowledge.doc_cooccurrence import _stems, _norm
from knowledge.semantic_search import semantic_search_with_neighbors, get_index

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Diverse, prominent concepts across many fields — so any co-occurring low-cosine pair is a
# genuine cross-domain link, not same-field trivia.
POOL = [
    "entropy", "thermodynamics", "percolation theory", "phase transition", "resonance",
    "chaos theory", "fractal", "topology", "renormalization group", "statistical mechanics",
    "diffusion", "turbulence", "crystallography", "natural selection", "evolution",
    "epidemic", "metabolism", "homeostasis", "morphogenesis", "ecology", "immune system",
    "photosynthesis", "enzyme", "information theory", "optimization", "neural network",
    "simulated annealing", "Markov chain", "cryptography", "game theory", "error correction",
    "supply and demand", "social network", "auction theory", "control theory", "oscillation",
    "network science", "metallurgy", "queueing theory", "linguistics",
]


def _save_run(args, idx, low_pairs, found):
    """Persist the FULL co-occurring set (not just --top) as JSON + a readable .md table.

    The miner otherwise only prints to stdout, so a run that isn't piped is lost. Bare
    --save writes a timestamped pair under data/synthesis_discovery/; an explicit path
    prefix overrides that. Returns the .json path.
    """
    ts = datetime.now()
    if args.save and args.save != "AUTO":
        base = args.save
        if base.endswith((".json", ".md")):
            base = base.rsplit(".", 1)[0]
        os.makedirs(os.path.dirname(os.path.abspath(base)) or ".", exist_ok=True)
    else:
        out_dir = os.path.join(_REPO_ROOT, "data", "synthesis_discovery")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, f"discovery_{ts:%Y%m%d_%H%M%S}")

    pairs = [
        {"cos": round(cs, 4), "a": a, "b": b,
         "signal": "shared-title" if sh else "text-mention",
         "shared_count": sh, "shared_titles": ex}
        for cs, a, b, sh, ex, mn in found
    ]
    payload = {
        "timestamp": ts.isoformat(timespec="seconds"),
        "params": {"depth": args.depth, "max_cos": args.max_cos, "top": args.top,
                   "top_chunks": args.top_chunks},
        "faiss_rows": idx._total_rows,
        "pool_size": len(POOL),
        "low_pairs_scanned": low_pairs,
        "cooccurring_found": len(found),
        "cooccurrence_rate": round(len(found) / max(low_pairs, 1), 4),
        "pairs": pairs,  # FULL set, sorted most-non-obvious first — not truncated to --top
    }
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md = [
        f"# Discovery miner run — {payload['timestamp']}",
        "",
        f"- params: depth={args.depth} max_cos={args.max_cos} pool={len(POOL)} "
        f"faiss_rows={idx._total_rows:,}",
        f"- non-obvious pairs scanned: {low_pairs} | co-occurring: {len(found)} "
        f"({payload['cooccurrence_rate']:.0%})",
        "",
        "| cos | signal | pair |",
        "|----:|--------|------|",
    ]
    for p in pairs:
        via = f" (via {', '.join(p['shared_titles'])})" if p["shared_titles"] else ""
        md.append(f"| {p['cos']:.2f} | {p['signal']} | {p['a']} ↔ {p['b']}{via} |")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    return base + ".json"


def main():
    ap = argparse.ArgumentParser(description="Discovery miner")
    ap.add_argument("--depth", type=int, default=30, help="Retrieval depth per concept")
    ap.add_argument("--max-cos", type=float, default=0.40, help="Keep pairs below this cosine (non-obvious)")
    ap.add_argument("--top", type=int, default=25, help="How many discoveries to show")
    ap.add_argument("--top-chunks", type=int, default=10,
                    help="Mention must land in the other concept's top-K chunks (0=full text, "
                         "the noisy old behaviour). Default 10 = validated precision gate.")
    ap.add_argument("--save", nargs="?", const="AUTO", default=None, metavar="PATH",
                    help="Persist the full run. Bare --save → timestamped file under "
                         "data/synthesis_discovery/; or pass an explicit path prefix.")
    args = ap.parse_args()

    idx = get_index(); idx.load()
    if not idx.loaded:
        print("ERROR: FAISS wiki index not available."); sys.exit(1)
    try:
        idx.index.nprobe = max(int(getattr(idx.index, "nprobe", 1)), 48)
    except Exception:
        pass
    gate = (f"top-{args.top_chunks}-chunk mention gate (precision-first)"
            if args.top_chunks > 0 else "FULL-text mention (noisy/legacy)")
    print(f"FAISS rows={idx._total_rows:,} | pool={len(POOL)} depth={args.depth} "
          f"max_cos={args.max_cos} | {gate}\n")

    # 1) Retrieve each concept ONCE; cache titles + text + stems. (The slow part.)
    cache = {}
    for i, c in enumerate(POOL):
        r = semantic_search_with_neighbors(c, k=args.depth)  # similarity-sorted desc
        titles = {_norm(x.get("title")) for x in r if x.get("title")}; titles.discard("")
        full = " ".join((x.get("content") or x.get("text") or "") for x in r).lower()
        # Mention text: top-`--top-chunks` chunks only (the concept's CORE articles). A cross-
        # mention there is real co-occurrence; one buried in a tangential chunk is a common-word
        # collision. Validated @ n=99 (validate_distinctive_mention.py): full-text mention
        # over-fires (28% on the pool); top-10 gating zeroes the common-word FPs at ~0 cost to
        # the archetypes. --top-chunks 0 restores the old noisy full-text behaviour.
        mtext = (" ".join((x.get("content") or x.get("text") or "")
                          for x in r[:args.top_chunks]).lower()
                 if args.top_chunks > 0 else full)
        cache[c] = {"titles": titles, "text": full, "mtext": mtext, "stems": _stems(c)}
        print(f"  retrieved {i + 1:2}/{len(POOL)}: {c}", flush=True)

    # 2) Embed all concepts once (cheap); all-pairs cosine + cached co-occurrence (free).
    embs = idx.embedder.encode(POOL, convert_to_numpy=True, normalize_embeddings=True)

    low_pairs = 0
    found = []  # (cos, a, b, shared_count, shared_titles, mention)
    for i in range(len(POOL)):
        for j in range(i + 1, len(POOL)):
            cs = float(np.dot(embs[i], embs[j]))
            if cs >= args.max_cos:
                continue  # too obvious / same-topic
            low_pairs += 1
            a, b = POOL[i], POOL[j]
            ca, cb = cache[a], cache[b]
            shared = ca["titles"] & cb["titles"]
            mention = (any(s in ca["mtext"] for s in cb["stems"])
                       or any(s in cb["mtext"] for s in ca["stems"]))
            if shared or mention:
                found.append((cs, a, b, len(shared), sorted(shared)[:2], mention))

    found.sort(key=lambda t: t[0])  # most non-obvious (lowest cosine) first

    print("\n" + "=" * 74)
    print(f"DISCOVERY QUADRANT — low-cosine ({'<'}{args.max_cos}) + doc-co-occurring")
    print("=" * 74)
    print(f"non-obvious pairs scanned: {low_pairs}  |  flagged real (co-occurring): "
          f"{len(found)} ({len(found)/max(low_pairs,1):.0%})\n")
    strong = "shared-title (strong)"; weak = "text-mention"
    for cs, a, b, sh, ex, mn in found[:args.top]:
        sig = strong if sh else weak
        exs = f" via {ex}" if ex else ""
        print(f"  cos={cs:5.2f} [{sig:20}] {a}  ↔  {b}{exs}")
    print("\nRead: each is a pair Wikipedia discusses together but embeddings call distant —\n"
          "a real, non-obvious connection the instrument mined. Shared-title = stronger\n"
          "(co-listed in an article); text-mention = one concept appears in the other's body.\n"
          "Whether any is novel TO YOU is the human last mile.")

    if args.save is not None:
        path = _save_run(args, idx, low_pairs, found)
        print(f"\nSaved full run ({len(found)} co-occurring pairs) → {path}\n"
              f"  + readable table → {path[:-5]}.md")


if __name__ == "__main__":
    main()
