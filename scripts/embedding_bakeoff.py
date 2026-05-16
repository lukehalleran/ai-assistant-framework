#!/usr/bin/env python3
"""
Shadow embedding bakeoff: compare retrieval quality across embedding models
without touching production ChromaDB.

Tests each model against the same benchmark seeds and cases, using temporary
ChromaDB collections per model.

Usage:
    python scripts/embedding_bakeoff.py

Models tested:
    - all-MiniLM-L6-v2     (baseline, current production)
    - BAAI/bge-small-en-v1.5
    - intfloat/e5-small-v2

Output:
    Per-model MRR, R@1, R@3, R@10 by intent, plus rank distribution.
"""

import asyncio
import gc
import os
import sys
import tempfile
import time
import yaml
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Models to test
# ---------------------------------------------------------------------------

MODELS = [
    {
        "name": "all-MiniLM-L6-v2",
        "chroma_name": "all-MiniLM-L6-v2",
    },
    {
        "name": "BAAI/bge-small-en-v1.5",
        "chroma_name": "BAAI/bge-small-en-v1.5",
    },
    {
        "name": "thenlper/gte-small",
        "chroma_name": "thenlper/gte-small",
    },
]

REFERENCE_TIME = datetime(2026, 2, 15, 12, 0, 0)


# ---------------------------------------------------------------------------
# MockTimeManager (copied from conftest)
# ---------------------------------------------------------------------------

class MockTimeManager:
    def __init__(self, ref: datetime):
        self._ref = ref

    def current(self) -> datetime:
        return self._ref

    def calculate_active_day_decay(self, timestamp, decay_rate):
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        age_hours = max(0.0, (self._ref - timestamp).total_seconds() / 3600.0)
        return 1.0 / (1.0 + decay_rate * age_hours)


# ---------------------------------------------------------------------------
# Seed & run
# ---------------------------------------------------------------------------

def load_config():
    yaml_path = Path(__file__).parent.parent / "tests" / "fixtures" / "retrieval_benchmarks.yaml"
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def seed_store(chroma_store, seed_memories, corpus_mgr):
    """Seed a ChromaDB store + corpus manager with benchmark data."""
    id_mapping = {}
    for mem in seed_memories:
        abs_ts = REFERENCE_TIME + timedelta(hours=mem["timestamp_offset_hours"])
        ts_iso = abs_ts.isoformat()

        if mem.get("query") and mem.get("response"):
            chroma_content = f"User: {mem['query']}\nAssistant: {mem['response']}"
        else:
            chroma_content = mem.get("content", "")

        metadata = {
            "timestamp": ts_iso,
            "truth_score": float(mem.get("truth_score", 0.6)),
            "importance_score": float(mem.get("importance_score", 0.5)),
            "tags": mem.get("tags", ""),
            "benchmark_id": mem["id"],
            "type": mem.get("memory_type", "episodic"),
        }
        if mem.get("query"):
            metadata["query"] = mem["query"]
        if mem.get("response"):
            metadata["response"] = mem["response"]

        doc_id = chroma_store.add_to_collection(mem["collection"], chroma_content, metadata)
        id_mapping[mem["id"]] = doc_id

        if mem.get("also_in_corpus") and mem.get("query") and mem.get("response"):
            corpus_mgr.add_entry(
                query=mem["query"],
                response=mem["response"],
                tags=(mem.get("tags", "").split(",") if mem.get("tags") else []),
                timestamp=abs_ts,
            )

    return id_mapping


def identify_memory(memory, seed_memories):
    """Map returned memory to seed ID."""
    bid = (memory.get("metadata") or {}).get("benchmark_id")
    if bid:
        return bid
    text = " ".join([
        memory.get("content", ""),
        memory.get("query", ""),
        memory.get("response", ""),
    ]).lower()
    for seed in seed_memories:
        marker = seed.get("marker", "")
        if marker and marker.lower() in text:
            return seed["id"]
    return None


async def run_case(retriever, scorer, intent_classifier, seed_memories, test_case):
    """Run a single benchmark case, return metrics dict."""
    query = test_case["query"]
    top_k = test_case.get("top_k", 10)
    must_retrieve = set(test_case.get("must_retrieve") or [])

    intent_result = intent_classifier.classify(query)
    scorer._intent_weight_overrides = intent_result.weight_overrides or None

    try:
        memories = await retriever.get_memories(query=query, limit=top_k)
    finally:
        scorer._intent_weight_overrides = None

    retrieved_ids = []
    for mem in memories:
        sid = identify_memory(mem, seed_memories)
        if sid and sid not in retrieved_ids:
            retrieved_ids.append(sid)

    retrieved_set = set(retrieved_ids)

    if must_retrieve:
        hits = must_retrieve & retrieved_set
        recall_at_k = len(hits) / len(must_retrieve)
        recall_at_1 = len(must_retrieve & set(retrieved_ids[:1])) / len(must_retrieve)
        recall_at_3 = len(must_retrieve & set(retrieved_ids[:3])) / len(must_retrieve)
        mrr = 0.0
        for i, rid in enumerate(retrieved_ids):
            if rid in must_retrieve:
                mrr = 1.0 / (i + 1)
                break
        rank_positions = []
        for eid in sorted(must_retrieve):
            if eid in retrieved_ids:
                rank_positions.append(retrieved_ids.index(eid) + 1)
            else:
                rank_positions.append(-1)
    else:
        recall_at_k = recall_at_1 = recall_at_3 = mrr = 0.0
        rank_positions = []

    return {
        "case_id": test_case["id"],
        "intent": test_case.get("expected_intent", ""),
        "has_retrieval": bool(must_retrieve),
        "recall_at_k": recall_at_k,
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "mrr": mrr,
        "ranks": rank_positions,
    }


async def run_model(model_cfg, config):
    """Run all benchmark cases for a single embedding model."""
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    from memory.corpus_manager import CorpusManager
    from memory.memory_retriever import MemoryRetriever
    from memory.memory_scorer import MemoryScorer
    from core.intent_classifier import IntentClassifier

    model_name = model_cfg["name"]
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    # Set model via env var (ChromaDB store reads this)
    os.environ["CHROMA_ST_MODEL"] = model_cfg["chroma_name"]

    with tempfile.TemporaryDirectory() as tmpdir:
        chroma_dir = os.path.join(tmpdir, "chroma")
        corpus_file = os.path.join(tmpdir, "corpus.json")

        t0 = time.perf_counter()
        chroma_store = MultiCollectionChromaStore(persist_directory=chroma_dir)
        model_load_s = time.perf_counter() - t0
        print(f"  Model load: {model_load_s:.1f}s")

        corpus_mgr = CorpusManager(corpus_file=corpus_file)
        time_mgr = MockTimeManager(REFERENCE_TIME)

        # Seed
        t0 = time.perf_counter()
        seed_memories = config["seed_memories"]
        id_mapping = seed_store(chroma_store, seed_memories, corpus_mgr)
        seed_s = time.perf_counter() - t0
        print(f"  Seed {len(seed_memories)} memories: {seed_s:.1f}s")

        scorer = MemoryScorer(
            time_manager=time_mgr,
            conversation_context=deque(maxlen=50),
        )
        retriever = MemoryRetriever(
            corpus_manager=corpus_mgr,
            chroma_store=chroma_store,
            gate_system=None,
            scorer=scorer,
            hybrid_retriever=None,
            time_manager=time_mgr,
        )
        intent_classifier = IntentClassifier()

        # Run cases
        test_cases = config["test_cases"]
        results = []
        t0 = time.perf_counter()
        for tc in test_cases:
            r = await run_case(retriever, scorer, intent_classifier, seed_memories, tc)
            results.append(r)
        run_s = time.perf_counter() - t0
        print(f"  Run {len(test_cases)} cases: {run_s:.1f}s")

    # Free GPU memory
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(model_name: str, results: List[Dict]):
    """Print per-intent metrics for a model."""
    by_intent: Dict[str, List[Dict]] = {}
    for r in results:
        by_intent.setdefault(r["intent"], []).append(r)

    print(f"\n  {'Intent':<25} {'Cases':>5} {'R@1':>6} {'R@3':>6} {'R@10':>6} {'MRR':>8}")
    print(f"  {'-'*58}")

    total_mrr = 0.0
    total_r1 = 0.0
    total_r3 = 0.0
    total_r10 = 0.0
    n_retrieval = 0

    for intent in sorted(by_intent.keys()):
        cases = by_intent[intent]
        ret_cases = [c for c in cases if c["has_retrieval"]]
        if not ret_cases:
            print(f"  {intent:<25} {len(cases):>5} {'--':>6} {'--':>6} {'--':>6} {'--':>8}")
            continue

        n = len(ret_cases)
        avg_r1 = sum(c["recall_at_1"] for c in ret_cases) / n
        avg_r3 = sum(c["recall_at_3"] for c in ret_cases) / n
        avg_r10 = sum(c["recall_at_k"] for c in ret_cases) / n
        avg_mrr = sum(c["mrr"] for c in ret_cases) / n

        total_mrr += sum(c["mrr"] for c in ret_cases)
        total_r1 += sum(c["recall_at_1"] for c in ret_cases)
        total_r3 += sum(c["recall_at_3"] for c in ret_cases)
        total_r10 += sum(c["recall_at_k"] for c in ret_cases)
        n_retrieval += n

        print(f"  {intent:<25} {len(cases):>5} {avg_r1:>6.3f} {avg_r3:>6.3f} {avg_r10:>6.3f} {avg_mrr:>8.3f}")

    if n_retrieval > 0:
        print(f"  {'-'*58}")
        print(
            f"  {'OVERALL':<25} {n_retrieval:>5} "
            f"{total_r1/n_retrieval:>6.3f} "
            f"{total_r3/n_retrieval:>6.3f} "
            f"{total_r10/n_retrieval:>6.3f} "
            f"{total_mrr/n_retrieval:>8.3f}"
        )


def print_comparison(all_results: Dict[str, List[Dict]]):
    """Print side-by-side MRR comparison across models."""
    # Collect per-intent MRR for each model
    intents = set()
    model_mrrs: Dict[str, Dict[str, float]] = {}

    for model_name, results in all_results.items():
        by_intent: Dict[str, List[Dict]] = {}
        for r in results:
            by_intent.setdefault(r["intent"], []).append(r)

        mrrs = {}
        for intent, cases in by_intent.items():
            ret = [c for c in cases if c["has_retrieval"]]
            if ret:
                mrrs[intent] = sum(c["mrr"] for c in ret) / len(ret)
                intents.add(intent)
        model_mrrs[model_name] = mrrs

    model_names = list(all_results.keys())

    print(f"\n{'='*80}")
    print("  MRR COMPARISON")
    print(f"{'='*80}")

    header = f"  {'Intent':<25}"
    for mn in model_names:
        short = mn.split("/")[-1][:18]
        header += f" {short:>18}"
    print(header)
    print(f"  {'-'*73}")

    for intent in sorted(intents):
        row = f"  {intent:<25}"
        for mn in model_names:
            v = model_mrrs[mn].get(intent)
            if v is not None:
                row += f" {v:>18.3f}"
            else:
                row += f" {'--':>18}"
        print(row)

    # Overall
    print(f"  {'-'*73}")
    row = f"  {'OVERALL':<25}"
    for mn in model_names:
        results = all_results[mn]
        ret = [r for r in results if r["has_retrieval"]]
        if ret:
            overall = sum(r["mrr"] for r in ret) / len(ret)
            row += f" {overall:>18.3f}"
        else:
            row += f" {'--':>18}"
    print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    config = load_config()
    print(f"Loaded {len(config['seed_memories'])} seeds, {len(config['test_cases'])} cases")
    print(f"Reference time: {REFERENCE_TIME}")
    print(f"GPU: {os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip()}")

    all_results = {}

    for model_cfg in MODELS:
        try:
            results = await run_model(model_cfg, config)
            model_name = model_cfg["name"]
            all_results[model_name] = results
            print_report(model_name, results)
        except Exception as e:
            print(f"\n  ERROR with {model_cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    if len(all_results) > 1:
        print_comparison(all_results)

        # Per-case temporal detail
        print(f"\n{'='*80}")
        print("  TEMPORAL RECALL PER-CASE DETAIL")
        print(f"{'='*80}")
        model_names = list(all_results.keys())
        temporal_ids = [c["id"] for c in config["test_cases"]
                        if c.get("expected_intent") == "temporal_recall"
                        and c.get("must_retrieve")]
        for cid in temporal_ids:
            print(f"\n  {cid}:")
            for mn in model_names:
                for r in all_results[mn]:
                    if r["case_id"] == cid:
                        short = mn.split("/")[-1][:18]
                        print(f"    {short:<18} MRR={r['mrr']:.3f}  ranks={r['ranks']}")

    print(f"\n{'='*80}")
    print(f"  Bakeoff complete: {len(all_results)}/{len(MODELS)} models tested")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
