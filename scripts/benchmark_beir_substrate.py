"""
# scripts/benchmark_beir_substrate.py

Module Contract
- Purpose: Measure Daemon's *retrieval substrate* (the BAAI/bge-small-en-v1.5 embedder,
  and optionally the ms-marco-MiniLM-L-6-v2 cross-encoder reranker) on standard BEIR
  datasets, producing nDCG@10 / Recall@10 / MRR@10 that are DIRECTLY COMPARABLE to the
  public MTEB / BEIR leaderboard.

  This deliberately BYPASSES the full Daemon memory scorer (recency / truth / importance /
  continuity / graph_bonus / intent overrides). Those terms are meaningless on BEIR (no
  recency, no user profile, no graph) and would make the number uninterpretable. What this
  benchmarks is the generic dense-retrieval layer only — the one piece of the pipeline that
  the outside world also measures. The memory-specific scoring is covered by the in-house
  synthetic + production suites in tests/benchmarks/ (and, prospectively, LongMemEval —
  see the memory note "benchmark-longmemeval").

  Secondary value: it's a metric-direction sanity check. If BGE-small lands near its
  published nDCG@10 on SciFact/NFCorpus, the embed→similarity→rank direction is correct
  end-to-end (the same class of bug as the wiki FAISS L2/cosine mismatch that once inverted
  retrieval).

- Usage:
    python scripts/benchmark_beir_substrate.py                       # nfcorpus + scifact, dense
    python scripts/benchmark_beir_substrate.py --datasets scifact    # one dataset
    python scripts/benchmark_beir_substrate.py --rerank              # + cross-encoder rerank top-100
    python scripts/benchmark_beir_substrate.py --query-instruction   # add BGE query prefix (leaderboard parity)
  Cache dir: $BEIR_CACHE_DIR (default .cache/beir). Datasets are small public zips.

- No new pip deps: urllib + zipfile + numpy + sentence_transformers (already required).
- Memory: only small BEIR corpora are allowed by default; a corpus over --max-corpus docs
  aborts with a message (16GB box guard). NFCorpus ~3.6K docs, SciFact ~5.2K docs.
- Side effects: downloads dataset zips to the cache dir; loads models from the HF cache.
"""

import argparse
import json
import logging
import math
import os
import sys
import urllib.request
import zipfile
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("beir_bench")

_BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
_CACHE = os.environ.get("BEIR_CACHE_DIR", os.path.join(".cache", "beir"))

# Substrate under test — must match config/app_config live values.
_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# BGE v1.5 retrieval query instruction (optional; off by default to match Daemon usage).
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Published nDCG@10 for bge-small-en-v1.5 (MTEB leaderboard, approximate — verify there).
# Anchor only: a locally-computed number within a few points validates the integration.
_PUBLISHED_NDCG10 = {"nfcorpus": 0.34, "scifact": 0.71, "arguana": 0.59, "scidocs": 0.21}


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
def _download(name: str) -> str:
    """Download+extract a BEIR dataset zip into the cache; return its folder."""
    os.makedirs(_CACHE, exist_ok=True)
    folder = os.path.join(_CACHE, name)
    if os.path.isdir(folder) and os.path.exists(os.path.join(folder, "corpus.jsonl")):
        return folder
    zip_path = os.path.join(_CACHE, f"{name}.zip")
    url = _BEIR_URL.format(name=name)
    logger.info(f"[{name}] downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(zip_path, "wb") as f:
        f.write(r.read())
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(_CACHE)
    os.remove(zip_path)
    return folder


def _load_dataset(name: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Return (corpus{id->title+text}, queries{id->text}, qrels{qid->{did->rel}})."""
    folder = _download(name)

    corpus: Dict[str, str] = {}
    with open(os.path.join(folder, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = (d.get("title", "") + " " + d.get("text", "")).strip()
            corpus[d["_id"]] = text

    queries: Dict[str, str] = {}
    with open(os.path.join(folder, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries[d["_id"]] = d["text"]

    # qrels/test.tsv: header row then  query-id \t corpus-id \t score
    qrels: Dict[str, Dict[str, int]] = {}
    qrels_path = os.path.join(folder, "qrels", "test.tsv")
    with open(qrels_path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) <= 0:
                continue
            qrels.setdefault(qid, {})[did] = int(score)

    # Restrict queries to those with test judgements.
    queries = {q: t for q, t in queries.items() if q in qrels}
    return corpus, queries, qrels


# --------------------------------------------------------------------------- #
# Metrics (standard BEIR definitions)
# --------------------------------------------------------------------------- #
def _dcg(gains: List[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _ndcg_at_k(ranked: List[str], qrel: Dict[str, int], k: int) -> float:
    gains = [qrel.get(d, 0) for d in ranked[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def _recall_at_k(ranked: List[str], qrel: Dict[str, int], k: int) -> float:
    rel = {d for d, g in qrel.items() if g > 0}
    if not rel:
        return 0.0
    return len(rel & set(ranked[:k])) / len(rel)


def _mrr_at_k(ranked: List[str], qrel: Dict[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if qrel.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def _embed(model, texts: List[str], batch_size: int) -> np.ndarray:
    return np.asarray(
        model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # cosine == dot product
            show_progress_bar=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )


def evaluate(
    name: str,
    embedder,
    reranker=None,
    query_instruction: bool = False,
    max_corpus: int = 100_000,
    batch_size: int = 128,
    rerank_depth: int = 100,
) -> Dict[str, float]:
    corpus, queries, qrels = _load_dataset(name)
    if len(corpus) > max_corpus:
        raise SystemExit(
            f"[{name}] corpus has {len(corpus)} docs > --max-corpus {max_corpus}; "
            f"pick a smaller dataset or raise the cap deliberately."
        )
    logger.info(f"[{name}] {len(corpus)} docs, {len(queries)} test queries")

    doc_ids = list(corpus.keys())
    doc_mat = _embed(embedder, [corpus[d] for d in doc_ids], batch_size)

    q_ids = list(queries.keys())
    q_texts = [queries[q] for q in q_ids]
    if query_instruction:
        q_texts = [_BGE_QUERY_INSTRUCTION + t for t in q_texts]
    q_mat = _embed(embedder, q_texts, batch_size)

    ndcg, recall, mrr = [], [], []
    top_n = max(rerank_depth, 10)
    # Score queries in blocks to bound peak memory (Q_block x N floats).
    for start in range(0, len(q_ids), 64):
        block = q_mat[start:start + 64]
        sims = block @ doc_mat.T                      # (b, N) cosine
        # top candidates per query
        cand_idx = np.argpartition(-sims, min(top_n, sims.shape[1] - 1), axis=1)[:, :top_n]
        for row, qi in enumerate(range(start, start + block.shape[0])):
            qid = q_ids[qi]
            cands = cand_idx[row]
            cand_scores = sims[row, cands]
            order = np.argsort(-cand_scores)
            ranked = [doc_ids[cands[o]] for o in order]

            if reranker is not None:
                depth = ranked[:rerank_depth]
                pairs = [[queries[qid], corpus[d]] for d in depth]
                ce = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
                ranked = [d for _, d in sorted(zip(ce, depth), key=lambda x: -x[0])]

            ndcg.append(_ndcg_at_k(ranked, qrels[qid], 10))
            recall.append(_recall_at_k(ranked, qrels[qid], 10))
            mrr.append(_mrr_at_k(ranked, qrels[qid], 10))

    return {
        "ndcg@10": float(np.mean(ndcg)),
        "recall@10": float(np.mean(recall)),
        "mrr@10": float(np.mean(mrr)),
        "n_queries": len(q_ids),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Daemon retrieval-substrate benchmark on BEIR")
    ap.add_argument("--datasets", nargs="+", default=["nfcorpus", "scifact"],
                    help="BEIR dataset names (small ones only by default)")
    ap.add_argument("--rerank", action="store_true",
                    help="apply the cross-encoder reranker on top-N dense candidates")
    ap.add_argument("--query-instruction", action="store_true",
                    help="prepend the BGE retrieval query prefix (leaderboard parity; "
                         "Daemon does NOT use it, so default off)")
    ap.add_argument("--max-corpus", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--rerank-depth", type=int, default=100)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer, CrossEncoder

    logger.info(f"Loading embedder {_EMBED_MODEL} ...")
    embedder = SentenceTransformer(_EMBED_MODEL)
    reranker = None
    if args.rerank:
        logger.info(f"Loading reranker {_RERANK_MODEL} ...")
        reranker = CrossEncoder(_RERANK_MODEL)

    rows = []
    for name in args.datasets:
        res = evaluate(
            name, embedder, reranker,
            query_instruction=args.query_instruction,
            max_corpus=args.max_corpus,
            batch_size=args.batch_size,
            rerank_depth=args.rerank_depth,
        )
        rows.append((name, res))

    mode = "dense+rerank" if args.rerank else "dense"
    qi = "with-instruction" if args.query_instruction else "no-instruction"
    print("\n" + "=" * 72)
    print(f"BEIR substrate benchmark — {_EMBED_MODEL}  [{mode}, {qi}]")
    print("=" * 72)
    print(f"{'dataset':<12} {'nDCG@10':>9} {'Recall@10':>10} {'MRR@10':>8} "
          f"{'nQ':>5} {'pub nDCG@10':>12}")
    print("-" * 72)
    for name, r in rows:
        pub = _PUBLISHED_NDCG10.get(name)
        pub_s = f"~{pub:.2f}" if pub is not None else "n/a"
        print(f"{name:<12} {r['ndcg@10']:>9.4f} {r['recall@10']:>10.4f} "
              f"{r['mrr@10']:>8.4f} {r['n_queries']:>5} {pub_s:>12}")
    print("-" * 72)
    print("Published = MTEB/BEIR leaderboard (approx; verify). Substrate only — the Daemon")
    print("memory scorer is intentionally bypassed. See module docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
