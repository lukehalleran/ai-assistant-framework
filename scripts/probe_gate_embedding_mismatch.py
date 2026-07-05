#!/usr/bin/env python
"""
Probe: quantify the gate's embedding-space mismatch.

ChromaDB retrieves candidates in bge-small-en-v1.5 space; the cosine gate
(processing/gate_system.py) then re-scores them with all-MiniLM-L6-v2 against
thresholds that were tuned on MiniLM's score distribution. This script measures
how much the two spaces actually disagree on live data, and derives the
quantile-matched bge thresholds needed if the gate switches to the retrieval
embedder.

For each probe query:
  1. Retrieve top-K candidates per collection via ChromaDB (bge space, like the
     real HNSW candidate stage).
  2. Re-embed query + gate-extracted content with BOTH models.
  3. Compare: correlation, pass/fail disagreement at the live threshold,
     top-N selection overlap, and the pass-rate-preserving bge threshold.

Read-only: only queries ChromaDB, never writes.

Usage:
    venv/bin/python scripts/probe_gate_embedding_mismatch.py [--per-query-k 50]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.app_config import CHROMA_PATH, GATE_REL_THRESHOLD  # noqa: E402
from eval.corpus import SEED_CORPUS  # noqa: E402

# Live gate parameters (mirror processing/gate_system.py batch_gate_memories)
COSINE_WEIGHT = float(os.getenv("GATE_COSINE_WEIGHT", "0.85"))
TRUTH_WEIGHT = 1.0 - COSINE_WEIGHT
TOP_N = 20  # gate caps final memories at 20


def _extract_gate_content(item: dict) -> str:
    """Mirror gate_system._extract_gate_content for chroma query results."""
    content = (item.get("content") or "").strip()
    if content:
        return content[:800]
    meta = item.get("metadata", {}) or {}
    return str(meta.get("content", ""))[:500]


def _encode(model, texts):
    return model.encode(
        list(texts), convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False, batch_size=64,
    ).astype(np.float32)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if len(a) < 3 or np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query-k", type=int, default=50)
    ap.add_argument("--out", default="data/probe_gate_embedding_mismatch.json")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    print(f"[probe] chroma path: {CHROMA_PATH}")
    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    bge = store.embedding_fn._model  # exact instance retrieval uses
    minilm = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"[probe] bge model: {store.embedding_model_name}")

    queries = [q["query_text"] for q in SEED_CORPUS]
    print(f"[probe] {len(queries)} probe queries (eval seed corpus)")

    collections = {
        "conversations": args.per_query_k,
        "summaries": 15,
        "reflections": 15,
    }

    # thresholds the live paths use (memories = blended vs GATE_REL_THRESHOLD)
    report = {"chroma_path": CHROMA_PATH, "gate_rel_threshold": GATE_REL_THRESHOLD,
              "cosine_weight": COSINE_WEIGHT, "collections": {}}

    for coll, k in collections.items():
        rows = []  # (query_idx, minilm_cos, bge_cos, truth)
        per_query = defaultdict(list)
        q_minilm = _encode(minilm, queries)
        q_bge = _encode(bge, queries)

        for qi, query in enumerate(queries):
            try:
                results = store.query_collection(coll, query, n_results=k)
            except Exception as e:
                print(f"[probe] {coll} query failed: {e}")
                continue
            texts = [_extract_gate_content(r) for r in results]
            texts = [t for t in texts if t]
            if not texts:
                continue
            m_vecs = _encode(minilm, texts)
            b_vecs = _encode(bge, texts)
            m_sims = m_vecs @ q_minilm[qi]
            b_sims = b_vecs @ q_bge[qi]
            for r, ms, bs in zip(results, m_sims, b_sims):
                truth = float((r.get("metadata") or {}).get("truth_score", 0.5))
                rows.append((qi, float(ms), float(bs), truth))
                per_query[qi].append((float(ms), float(bs), truth))

        if not rows:
            print(f"[probe] {coll}: no data")
            continue

        m = np.array([r[1] for r in rows])
        b = np.array([r[2] for r in rows])
        truth = np.array([r[3] for r in rows])

        pearson = float(np.corrcoef(m, b)[0, 1])
        spear_all = _spearman(m, b)
        # per-query rank agreement + top-N overlap (what the gate actually decides)
        spears, overlaps = [], []
        for qi, items in per_query.items():
            if len(items) < 5:
                continue
            mm = np.array([x[0] for x in items])
            bb = np.array([x[1] for x in items])
            s = _spearman(mm, bb)
            if not np.isnan(s):
                spears.append(s)
            n = min(TOP_N, len(items))
            top_m = set(np.argsort(-mm)[:n].tolist())
            top_b = set(np.argsort(-bb)[:n].tolist())
            overlaps.append(len(top_m & top_b) / n)

        stats = {}
        for name, arr in (("minilm", m), ("bge", b)):
            stats[name] = {
                "mean": float(arr.mean()), "std": float(arr.std()),
                "p05": float(np.percentile(arr, 5)), "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)), "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
            }

        entry = {
            "n_pairs": len(rows),
            "score_stats": stats,
            "pearson": pearson,
            "spearman_pooled": spear_all,
            "spearman_per_query_mean": float(np.mean(spears)) if spears else None,
            "top20_overlap_mean": float(np.mean(overlaps)) if overlaps else None,
        }

        if coll == "conversations":
            # live decision: blended = 0.85*cos + 0.15*truth >= threshold
            # (0.18 = GATE_REL_THRESHOLD, 0.20 = GATE_DEICTIC_MIN floor)
            blended_m = COSINE_WEIGHT * m + TRUTH_WEIGHT * truth
            blended_b = COSINE_WEIGHT * b + TRUTH_WEIGHT * truth
            analysis = {}
            for label, thr in (("gate_rel", GATE_REL_THRESHOLD), ("deictic_min", 0.20)):
                pass_m = blended_m >= thr
                pass_rate = float(pass_m.mean())
                bge_thr = float(np.percentile(blended_b, 100 * (1 - pass_rate))) if 0 < pass_rate < 1 else None
                item = {
                    "minilm_blended_threshold": thr,
                    "minilm_pass_rate": pass_rate,
                    "quantile_matched_bge_blended_threshold": bge_thr,
                }
                if bge_thr is not None:
                    item["disagreement_rate"] = float(((blended_b >= bge_thr) != pass_m).mean())
                analysis[label] = item
            entry["live_threshold_analysis"] = analysis
        else:
            # summaries/reflections: report matched thresholds for a range of
            # candidate MiniLM cutoffs (these paths are currently uncalled, but
            # keep the mapping for reference)
            mapping = {}
            for thr in (0.15, 0.20, 0.25, 0.30):
                pr = float((m >= thr).mean())
                mapping[str(thr)] = {
                    "minilm_pass_rate": pr,
                    "bge_threshold_same_rate": float(np.percentile(b, 100 * (1 - pr))) if 0 < pr < 1 else None,
                }
            entry["threshold_mapping"] = mapping

        report["collections"][coll] = entry
        print(f"\n[probe] === {coll} (n={len(rows)}) ===")
        print(json.dumps(entry, indent=2))

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
