#!/usr/bin/env python3
"""
THROWAWAY PROBE — directionality of text-mention hits in the anchored-vs-random PoC.

MEASURE ONLY. Builds nothing, fixes nothing, modifies no core code. Loads the most recent
poc_anchored_vs_random_*.json from data/synthesis_discovery/ and, for every TEXT-MENTION-ONLY
known pair (both arms), asks whether the stem-collision is BIDIRECTIONAL (real co-occurrence)
or UNIDIRECTIONAL (one concept's distinctive term leaking into the other's articles without
the reverse — the common-word / neighbor-vocabulary contamination the PoC gap may be made of).

HYPOTHESIS: anchored partners are semantic neighbours of the anchor, so the anchor's term
shows up in the partner's articles WITHOUT the reverse → anchored should be MORE unidirectional.
Real literature co-occurrence is bidirectional. If requiring bidirectionality shrinks the PoC
gap toward parity, the anchored edge was largely artifact.

EXACT-REPLICATION GUARANTEES (a mismatch here would invalidate the probe):
  * Distinctive-term extraction is the oracle's own `_stems`/`_norm`, IMPORTED — not reimplemented.
  * Article text is built exactly as the oracle builds it:
        " ".join((r.get("content") or r.get("text") or "") for r in chunks).lower()
  * Retrieval is the oracle's `semantic_search_with_neighbors` at the run's depth (40), with
    FAISS nprobe forced to 48 — the value the PoC set at runtime. (At the default nprobe the
    index returns different chunks and a PoC text-mention could fail to reproduce.)
  Direction is just the oracle's `mention` disjunction evaluated per-side:
        A->B = any(s in b_text for s in a_stems)   # A's distinctive term in B's articles
        B->A = any(s in a_text for s in b_stems)    # B's distinctive term in A's articles
        oracle mention == (A->B or B->A)
  So any text-mention-only known MUST be A->B and/or B->A. A NEITHER means the probe diverged
  from the oracle's code path → comparison invalid → flagged + STOP (no conclusions drawn).

The only reimplemented thing is the two-proportion z-test (pure stats, same formula as the
PoC's helper) — it has nothing to do with the distinctive-term extraction.

Free — FAISS + embedder only, no LLM, no ChromaDB. REPORT and STOP.
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

# Oracle's OWN extraction + retrieval — imported, never reimplemented.
from knowledge.doc_cooccurrence import _stems, _norm  # noqa: E402
from knowledge.semantic_search import semantic_search_with_neighbors, get_index  # noqa: E402

DEPTH = 40   # the depth the PoC run used (params.depth)
NPROBE = 48  # the nprobe the PoC set at runtime


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_prop_z(kA, nA, kB, nB):
    """Two-proportion z-test (pooled) + Wald 95% CI on (pA - pB). Same formula as the PoC."""
    if nA == 0 or nB == 0:
        return {"pA": 0.0, "pB": 0.0, "diff": 0.0, "z": 0.0, "p": 1.0, "ci": (0.0, 0.0)}
    pA, pB = kA / nA, kB / nB
    pool = (kA + kB) / (nA + nB)
    se_pool = math.sqrt(pool * (1 - pool) * (1 / nA + 1 / nB)) if 0 < pool < 1 else 0.0
    z = (pA - pB) / se_pool if se_pool > 0 else 0.0
    p = 2 * (1 - _phi(abs(z)))
    se_diff = math.sqrt(pA * (1 - pA) / nA + pB * (1 - pB) / nB)
    return {"pA": pA, "pB": pB, "diff": pA - pB, "z": z, "p": p,
            "ci": ((pA - pB) - 1.96 * se_diff, (pA - pB) + 1.96 * se_diff)}


def retrieve_concept(concept, cache):
    """Retrieve + cache a concept's joined article text and its oracle stems (exact replica)."""
    if concept in cache:
        return cache[concept]
    chunks = semantic_search_with_neighbors(concept, k=DEPTH)
    text = " ".join((r.get("content") or r.get("text") or "") for r in chunks).lower()
    titles = {_norm(r.get("title")) for r in chunks if r.get("title")}
    titles.discard("")
    cache[concept] = {"text": text, "stems": _stems(concept), "titles": titles}
    return cache[concept]


def classify(a, b, cache):
    """Return (label, a2b_stems, b2a_stems) using the oracle's exact per-direction logic."""
    ca, cb = retrieve_concept(a, cache), retrieve_concept(b, cache)
    a2b = sorted(s for s in ca["stems"] if s in cb["text"])  # A's term in B's text
    b2a = sorted(s for s in cb["stems"] if s in ca["text"])  # B's term in A's text
    if a2b and b2a:
        label = "BIDIRECTIONAL"
    elif a2b:
        label = "A->B"
    elif b2a:
        label = "B->A"
    else:
        label = "NEITHER"
    return label, a2b, b2a


def main():
    ap = argparse.ArgumentParser(description="Directionality probe (measure only)")
    ap.add_argument("--json", default=None, help="PoC run JSON (default: most recent)")
    ap.add_argument("--save", default="AUTO", help="output prefix (default timestamped)")
    args = ap.parse_args()

    jf = args.json or sorted(glob.glob(os.path.join(
        _REPO_ROOT, "data", "synthesis_discovery", "poc_anchored_vs_random_*.json")))[-1]
    run = json.load(open(jf))
    print(f"Loaded PoC run: {os.path.basename(jf)} (depth={run['params']['depth']})")
    assert run["params"]["depth"] == DEPTH, "PoC depth != probe depth — abort"

    idx = get_index(); idx.load()
    if not idx.loaded:
        print("ERROR: FAISS index unavailable."); sys.exit(1)
    try:
        idx.index.nprobe = max(int(getattr(idx.index, "nprobe", 1)), NPROBE)
    except Exception:
        pass
    print(f"FAISS rows={idx._total_rows:,} nprobe={getattr(idx.index,'nprobe','?')} depth={DEPTH}\n")

    arms = {"A": run["arm_a_pairs"], "B": run["arm_b_pairs"]}
    cache = {}
    results = {"A": [], "B": []}
    neither_flags = []
    stem_df = Counter()  # how many concept texts each matched stem appears in (collision proxy)

    for arm, pairs in arms.items():
        tm = [p for p in pairs if p.get("known") and p.get("match_type") == "text-mention"]
        print(f"--- Arm {arm}: classifying {len(tm)} text-mention-only knowns ---", flush=True)
        for n, p in enumerate(tm, 1):
            a, b = p["a"], p["b"]
            label, a2b, b2a = classify(a, b, cache)
            for s in set(a2b) | set(b2a):
                stem_df[s] += 1
            row = {"a": a, "b": b, "cos": p.get("cos"), "label": label,
                   "a2b_stems": a2b, "b2a_stems": b2a, "anchor": p.get("anchor")}
            results[arm].append(row)
            if label == "NEITHER":
                neither_flags.append((arm, a, b))
            if n % 10 == 0:
                print(f"    {n}/{len(tm)}", flush=True)

    # ---- VALIDITY GATE ----
    if neither_flags:
        print("\n" + "!" * 74)
        print(f"INVALID: {len(neither_flags)} text-mention-only known(s) classified NEITHER — "
              "the probe diverged from the oracle's retrieval/extraction path. Comparison is "
              "not trustworthy. STOPPING without conclusions.")
        for arm, a, b in neither_flags[:10]:
            print(f"  [{arm}] {a}  <->  {b}")
        print("!" * 74)
        sys.exit(2)

    # ---- per-arm aggregates ----
    st_known = {"A": run["arm_a"]["shared_title"], "B": run["arm_b"]["shared_title"]}
    n_arm = {"A": run["arm_a"]["n"], "B": run["arm_b"]["n"]}
    agg = {}
    for arm in ("A", "B"):
        rs = results[arm]
        c = Counter(r["label"] for r in rs)
        tm_n = len(rs)
        bidir = c["BIDIRECTIONAL"]
        uni = c["A->B"] + c["B->A"]
        agg[arm] = {
            "tm_known": tm_n, "bidir": bidir, "a2b_only": c["A->B"], "b2a_only": c["B->A"],
            "uni": uni,
            "bidir_share": bidir / tm_n if tm_n else 0.0,
            "uni_share": uni / tm_n if tm_n else 0.0,
            # bidirectional-only known-rate: shared-title + bidirectional text-mention count
            # as known; unidirectional text-mention reclassified NOT known.
            "known_bidironly": st_known[arm] + bidir,
            "rate_bidironly": (st_known[arm] + bidir) / n_arm[arm],
            "orig_known": run[f"arm_{arm.lower()}"]["known"],
            "orig_rate": run[f"arm_{arm.lower()}"]["rate"],
        }

    z_orig = two_prop_z(agg["A"]["orig_known"], n_arm["A"], agg["B"]["orig_known"], n_arm["B"])
    z_bidir = two_prop_z(agg["A"]["known_bidironly"], n_arm["A"],
                         agg["B"]["known_bidironly"], n_arm["B"])
    # the headline directionality comparison: bidirectional share among tm-only knowns
    z_share = two_prop_z(agg["A"]["bidir"], agg["A"]["tm_known"],
                         agg["B"]["bidir"], agg["B"]["tm_known"])

    # ---------------- report ----------------
    print("\n" + "=" * 78)
    print("DIRECTIONALITY OF TEXT-MENTION-ONLY KNOWNS")
    print("=" * 78)
    print(f"  {'arm':<6}{'tm-known':>9}{'BIDIR':>7}{'A->B':>7}{'B->A':>7}{'uni':>6}"
          f"{'bidir%':>9}{'uni%':>8}")
    for arm in ("A", "B"):
        g = agg[arm]
        print(f"  {arm:<6}{g['tm_known']:>9}{g['bidir']:>7}{g['a2b_only']:>7}{g['b2a_only']:>7}"
              f"{g['uni']:>6}{g['bidir_share']:>8.0%}{g['uni_share']:>8.0%}")
    print("-" * 78)
    print("KEY NUMBER — bidirectional share among text-mention-only knowns:")
    print(f"  Arm A {agg['A']['bidir_share']:.0%} ({agg['A']['bidir']}/{agg['A']['tm_known']})"
          f"   vs   Arm B {agg['B']['bidir_share']:.0%} "
          f"({agg['B']['bidir']}/{agg['B']['tm_known']})"
          f"   diff {z_share['diff']:+.0%}  p={z_share['p']:.3f}")
    print("  Hypothesis predicts anchored (A) has the HIGHER UNIdirectional share "
          "(more collision).")
    print("-" * 78)
    print("Bidirectional-only known-rate (shared-title + bidir text-mention; "
          "unidirectional reclassified NOT known):")
    print(f"  original : A {agg['A']['orig_rate']:.1%} ({agg['A']['orig_known']}/{n_arm['A']})"
          f"  vs B {agg['B']['orig_rate']:.1%} ({agg['B']['orig_known']}/{n_arm['B']})"
          f"  gap {z_orig['diff']:+.1%}  p={z_orig['p']:.4f}")
    print(f"  bidir-only: A {agg['A']['rate_bidironly']:.1%} "
          f"({agg['A']['known_bidironly']}/{n_arm['A']})"
          f"  vs B {agg['B']['rate_bidironly']:.1%} ({agg['B']['known_bidironly']}/{n_arm['B']})"
          f"  gap {z_bidir['diff']:+.1%}  p={z_bidir['p']:.4f}  "
          f"95% CI [{z_bidir['ci'][0]:+.1%}, {z_bidir['ci'][1]:+.1%}]")
    print("=" * 78)

    # ---- eyeball aid: anchored unidirectional (collision suspects) vs bidirectional ----
    def commonness(row):
        stems = set(row["a2b_stems"]) | set(row["b2a_stems"])
        # higher df = more concept texts share the stem = more collision-like; shorter = worse
        return (max((stem_df[s] for s in stems), default=0), -min((len(s) for s in stems), default=99))

    uni_A = sorted([r for r in results["A"] if r["label"] in ("A->B", "B->A")],
                   key=commonness, reverse=True)
    bidir_A = [r for r in results["A"] if r["label"] == "BIDIRECTIONAL"]

    def stems_str(r):
        a2b = f"A→B:{','.join(r['a2b_stems'])}" if r["a2b_stems"] else ""
        b2a = f"B→A:{','.join(r['b2a_stems'])}" if r["b2a_stems"] else ""
        return " ".join(x for x in (a2b, b2a) if x)

    print("\nEYEBALL — Arm A UNIDIRECTIONAL (collision suspects, most-common-stem first):")
    for r in uni_A[:10]:
        df = max((stem_df[s] for s in set(r["a2b_stems"]) | set(r["b2a_stems"])), default=0)
        print(f"  [{r['label']:5} df={df:>2}] {r['a']}  ↔  {r['b']}    ({stems_str(r)})")
    print("\nEYEBALL — Arm A BIDIRECTIONAL (plausible-real):")
    for r in bidir_A[:10]:
        print(f"  [BIDIR] {r['a']}  ↔  {r['b']}    ({stems_str(r)})")

    # ---------------- save ----------------
    ts = datetime.now()
    base = (args.save if args.save != "AUTO"
            else os.path.join(_REPO_ROOT, "data", "synthesis_discovery",
                              f"probe_directionality_{ts:%Y%m%d_%H%M%S}"))
    if base.endswith((".json", ".md")):
        base = base.rsplit(".", 1)[0]
    payload = {
        "timestamp": ts.isoformat(timespec="seconds"),
        "source_run": os.path.basename(jf),
        "depth": DEPTH, "nprobe": NPROBE,
        "replication_notes": [
            "_stems/_norm imported from doc_cooccurrence (oracle's own extraction).",
            "text join + per-direction substring match replicate the oracle line-for-line.",
            "nprobe=48 + depth=40 match the PoC run.",
            "two_prop_z reimplemented (pure stats, same formula) — unrelated to extraction.",
        ],
        "neither_count": len(neither_flags),
        "per_arm": agg,
        "stats": {"original_known_gap": {**z_orig, "ci": list(z_orig["ci"])},
                  "bidir_only_gap": {**z_bidir, "ci": list(z_bidir["ci"])},
                  "bidir_share_gap": {**z_share, "ci": list(z_share["ci"])}},
        "pairs": {arm: results[arm] for arm in ("A", "B")},
    }
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md = [
        f"# Directionality probe — {payload['timestamp']}",
        f"Source: `{os.path.basename(jf)}` | depth {DEPTH} | nprobe {NPROBE}",
        "",
        "## Text-mention-only knowns by direction",
        "",
        "| arm | tm-known | BIDIR | A→B | B→A | unidir | bidir% | unidir% |",
        "|-----|---------:|------:|----:|----:|-------:|-------:|--------:|",
    ]
    for arm in ("A", "B"):
        g = agg[arm]
        md.append(f"| {arm} | {g['tm_known']} | {g['bidir']} | {g['a2b_only']} | "
                  f"{g['b2a_only']} | {g['uni']} | {g['bidir_share']:.0%} | {g['uni_share']:.0%} |")
    md += [
        "",
        f"**Bidirectional share (tm-only knowns):** A {agg['A']['bidir_share']:.0%} vs "
        f"B {agg['B']['bidir_share']:.0%} (diff {z_share['diff']:+.0%}, p={z_share['p']:.3f})",
        "",
        "## Known-rate: original vs bidirectional-only",
        "",
        "| | Arm A | Arm B | gap | p |",
        "|--|------:|------:|----:|--:|",
        f"| original | {agg['A']['orig_rate']:.1%} | {agg['B']['orig_rate']:.1%} | "
        f"{z_orig['diff']:+.1%} | {z_orig['p']:.4f} |",
        f"| bidir-only | {agg['A']['rate_bidironly']:.1%} | {agg['B']['rate_bidironly']:.1%} | "
        f"{z_bidir['diff']:+.1%} | {z_bidir['p']:.4f} |",
        "",
        "## Arm A unidirectional (collision suspects, most-common-stem first)",
        "",
        "| dir | df | pair | stems |",
        "|-----|---:|------|-------|",
    ]
    for r in uni_A:
        df = max((stem_df[s] for s in set(r["a2b_stems"]) | set(r["b2a_stems"])), default=0)
        md.append(f"| {r['label']} | {df} | {r['a']} ↔ {r['b']} | {stems_str(r)} |")
    md += ["", "## Arm A bidirectional (plausible-real)", "",
           "| pair | stems |", "|------|-------|"]
    for r in bidir_A:
        md.append(f"| {r['a']} ↔ {r['b']} | {stems_str(r)} |")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nSaved → {base}.json\n  + {base}.md")


if __name__ == "__main__":
    main()
