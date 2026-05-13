#!/usr/bin/env python3
"""
Retrieval quality & latency benchmark.

Produces:
  - Recall@1, Recall@3, Recall@10 across 19 test cases spanning 9 intent types
  - MRR (Mean Reciprocal Rank) — aggregate and per-intent
  - Correct memory rank distribution histogram
  - Precision@10
  - Intent classification accuracy (9 types, regex-first)
  - Per-component retrieval latencies (p50, p90, p95)
  - Cold start vs warm cache embedding latency
  - Full prompt build latency (p50, p90, p95)

Usage:
    python scripts/benchmark_retrieval.py
"""

import asyncio
import json
import os
import sys
import time
import statistics
from collections import Counter, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


def ms(secs: float) -> str:
    return f"{secs * 1000:.1f}ms"


def _fmt_pct(data: List[float]) -> str:
    """Format p50/p90/p95 as a compact string."""
    if not data:
        return "N/A"
    return f"p50={ms(pct(data, 50))}  p90={ms(pct(data, 90))}  p95={ms(pct(data, 95))}"


REFERENCE_TIME = datetime(2026, 2, 15, 12, 0, 0)
FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "retrieval_benchmarks.yaml"


# ---------------------------------------------------------------------------
# MockTimeManager (same as conftest.py)
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
# Main
# ---------------------------------------------------------------------------

async def main():
    print("=" * 70)
    print("DAEMON RETRIEVAL BENCHMARK")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---- Load fixtures ----
    print("\n[1/7] Loading test fixtures...")
    with open(FIXTURE_PATH) as f:
        config = yaml.safe_load(f)
    seed_memories = config["seed_memories"]
    test_cases = config["test_cases"]
    print(f"  {len(seed_memories)} seed memories, {len(test_cases)} test cases")

    # ---- Initialize components ----
    print("\n[2/7] Initializing components (loading embedder ~2s)...")
    t0 = time.perf_counter()

    import tempfile
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    from memory.corpus_manager import CorpusManager
    from memory.memory_retriever import MemoryRetriever
    from memory.memory_scorer import MemoryScorer
    from processing.gate_system import MultiStageGateSystem, _EMBED_CACHE
    from core.intent_classifier import IntentClassifier

    tmpdir = tempfile.mkdtemp(prefix="daemon_bench_")
    chroma = MultiCollectionChromaStore(persist_directory=tmpdir)
    corpus_file = os.path.join(tmpdir, "corpus.json")
    corpus = CorpusManager(corpus_file=corpus_file)

    time_mgr = MockTimeManager(REFERENCE_TIME)
    scorer = MemoryScorer(time_manager=time_mgr, conversation_context=deque(maxlen=50))
    intent_classifier = IntentClassifier()

    # Gate system for latency tests (needs model_manager=None, uses cached embedder)
    gate_system = MultiStageGateSystem(model_manager=None)

    retriever = MemoryRetriever(
        corpus_manager=corpus,
        chroma_store=chroma,
        gate_system=None,  # Test pure scoring, no gating filter
        scorer=scorer,
        hybrid_retriever=None,
        time_manager=time_mgr,
    )

    init_time = time.perf_counter() - t0
    print(f"  Done in {ms(init_time)}")

    # ---- Seed benchmark data ----
    print("\n[3/7] Seeding benchmark memories into temp ChromaDB...")
    seeded = 0
    for mem in seed_memories:
        abs_ts = REFERENCE_TIME + timedelta(hours=mem["timestamp_offset_hours"])
        if mem.get("query") and mem.get("response"):
            content = f"User: {mem['query']}\nAssistant: {mem['response']}"
        else:
            content = mem.get("content", "")

        metadata = {
            "timestamp": abs_ts.isoformat(),
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

        chroma.add_to_collection(mem["collection"], content, metadata)
        seeded += 1

        if mem.get("also_in_corpus") and mem.get("query") and mem.get("response"):
            corpus.add_entry(
                query=mem["query"],
                response=mem["response"],
                tags=(mem.get("tags", "").split(",") if mem.get("tags") else []),
                timestamp=abs_ts,
            )

    print(f"  Seeded {seeded} memories")

    # ---- Run recall benchmark with per-case latency ----
    print("\n[4/7] Running recall@K benchmark...")
    from tests.benchmarks.retrieval_benchmark import RetrievalBenchmark

    harness = RetrievalBenchmark(retriever, scorer, intent_classifier, seed_memories)

    results = []
    case_latencies = []
    for case in test_cases:
        t0 = time.perf_counter()
        try:
            result = harness.run_case(case)
            if asyncio.iscoroutine(result):
                result = await result
            elapsed = time.perf_counter() - t0
            results.append(result)
            case_latencies.append(elapsed)
        except Exception as e:
            print(f"  [ERROR] {case['id']}: {e}")

    # ---- Run gate system latency benchmark ----
    print("\n[5/7] Running gate system latency benchmark...")

    gate_queries = [
        "What is my dog's name?",
        "How do I debug a segmentation fault?",
        "I feel really stressed about work",
        "What did we discuss about the API?",
        "Let's brainstorm new features",
        "What happened last week?",
        "Tell me about Python async patterns",
        "Good morning!",
    ]

    # Pull seeded memories for gating
    gate_memories = chroma.query_collection("conversations", "general", n_results=20)
    gate_timings = []
    retrieval_timings = []

    if gate_memories:
        for run in range(5):
            for query in gate_queries:
                # Gate system latency
                t0 = time.perf_counter()
                try:
                    gated = await gate_system.filter_memories(query, gate_memories)
                except Exception:
                    gated = gate_memories  # fallback
                gate_timings.append(time.perf_counter() - t0)

                # ChromaDB retrieval latency
                t0 = time.perf_counter()
                _ = chroma.query_collection("conversations", query, n_results=10)
                retrieval_timings.append(time.perf_counter() - t0)

    # ---- Cold start vs warm cache ----
    print("\n[6/7] Running cold vs warm cache benchmark...")

    cold_timings = []
    warm_timings = []

    # Fresh queries the cache has never seen
    cold_queries = [
        "What are the best hiking trails nearby?",
        "Tell me about quantum computing advances",
        "How do I configure nginx reverse proxy?",
        "What was the Renaissance period known for?",
    ]
    # Queries already seen during the recall benchmark (cached embeddings)
    warm_queries = gate_queries[:4]

    if gate_memories:
        for query in cold_queries:
            # Clear cache to force cold encoding
            _EMBED_CACHE._store.clear()
            t0 = time.perf_counter()
            try:
                await gate_system.filter_memories(query, gate_memories)
            except Exception:
                pass
            cold_timings.append(time.perf_counter() - t0)

        # Warm the cache with a pass
        for query in warm_queries:
            try:
                await gate_system.filter_memories(query, gate_memories)
            except Exception:
                pass
        # Now measure warm
        for query in warm_queries:
            t0 = time.perf_counter()
            try:
                await gate_system.filter_memories(query, gate_memories)
            except Exception:
                pass
            warm_timings.append(time.perf_counter() - t0)

    # ---- Prompt build latency ----
    print("\n[7/7] Running prompt build latency benchmark...")

    prompt_build_timings = []
    try:
        from core.prompt.builder import UnifiedPromptBuilder

        builder = UnifiedPromptBuilder(
            corpus_manager=corpus,
            chroma_store=chroma,
            gate_system=gate_system,
            scorer=scorer,
            model_manager=None,
            time_manager=time_mgr,
        )

        build_queries = [
            "What is my dog's name?",
            "I feel really stressed about work",
            "How do I debug a segmentation fault?",
            "Let's brainstorm new features",
            "Good morning!",
        ]

        for query in build_queries:
            t0 = time.perf_counter()
            try:
                await builder.build_prompt(query, config={})
            except Exception:
                pass  # some components may not be initialized
            prompt_build_timings.append(time.perf_counter() - t0)
    except Exception as e:
        print(f"  [WARN] Prompt build benchmark skipped: {e}")

    # ====================================================================
    # Compute and display results
    # ====================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Split: retrieval cases (have must_retrieve) vs intent-only cases
    retrieval_cases = [r for r in results if r.has_retrieval_requirement]
    intent_only_cases = [r for r in results if not r.has_retrieval_requirement]

    # Quality metrics computed ONLY over retrieval cases (13 of 19)
    recalls_10 = [r.recall_at_k for r in retrieval_cases]
    recalls_1 = [r.recall_at_1 for r in retrieval_cases]
    recalls_3 = [r.recall_at_3 for r in retrieval_cases]
    # MRR: include 0.0 for misses (don't filter them out)
    mrrs = [r.mrr for r in retrieval_cases]
    precisions = [r.precision_at_k for r in retrieval_cases]
    # Intent accuracy over ALL cases (including intent-only)
    intent_correct = sum(1 for r in results if r.intent_expected == r.intent_actual)
    intent_total = sum(1 for r in results if r.intent_expected)
    passed = sum(1 for r in results if r.passed)

    # ---- Quality metrics ----
    n_ret = len(retrieval_cases)
    print(f"\n  RETRIEVAL QUALITY ({n_ret} retrieval cases, {len(intent_only_cases)} intent-only, {len(seed_memories)} seeds)")
    print(f"  {'-'*55}")
    r10 = statistics.mean(recalls_10) if recalls_10 else 0
    r1 = statistics.mean(recalls_1) if recalls_1 else 0
    r3 = statistics.mean(recalls_3) if recalls_3 else 0
    mrr_val = statistics.mean(mrrs) if mrrs else 0
    p10 = statistics.mean(precisions) if precisions else 0
    intent_acc = intent_correct / intent_total if intent_total else 0

    print(f"  Recall@1:            {r1:.3f}  (n={n_ret})")
    print(f"  Recall@3:            {r3:.3f}")
    print(f"  Recall@10:           {r10:.3f}")
    print(f"  Precision@10:        {p10:.3f}")
    print(f"  MRR:                 {mrr_val:.3f}")
    print(f"  Intent accuracy:     {intent_acc:.1%} ({intent_correct}/{intent_total})")
    print(f"  Cases passed:        {passed}/{len(results)}")

    # ---- Rank distribution histogram ----
    all_ranks: List[int] = []
    for r in retrieval_cases:
        all_ranks.extend(p for p in r.rank_positions if p > 0)

    rank_histogram: Dict[str, int] = Counter()
    for rank in all_ranks:
        if rank == 1:
            rank_histogram["rank_1"] += 1
        elif rank <= 3:
            rank_histogram["rank_2_3"] += 1
        elif rank <= 5:
            rank_histogram["rank_4_5"] += 1
        elif rank <= 10:
            rank_histogram["rank_6_10"] += 1
        else:
            rank_histogram["rank_11_plus"] += 1

    if all_ranks:
        total_ranked = len(all_ranks)
        print(f"\n  RANK DISTRIBUTION (n={total_ranked} expected items)")
        print(f"  {'-'*55}")
        for bucket, label in [
            ("rank_1", "Rank 1 (top)"),
            ("rank_2_3", "Rank 2-3"),
            ("rank_4_5", "Rank 4-5"),
            ("rank_6_10", "Rank 6-10"),
            ("rank_11_plus", "Rank 11+"),
        ]:
            count = rank_histogram.get(bucket, 0)
            bar = "#" * int(count / total_ranked * 30)
            print(f"  {label:<18} {count:>3} ({count/total_ranked:>5.0%}) {bar}")

    # Per-intent breakdown — only include retrieval metrics for cases with requirements
    by_intent: Dict[str, Dict] = {}
    for r in results:
        intent = r.intent_expected or "unknown"
        if intent not in by_intent:
            by_intent[intent] = {
                "recalls_1": [], "recalls_3": [], "recalls_10": [],
                "mrrs": [], "passed": 0, "total": 0,
            }
        by_intent[intent]["total"] += 1
        if r.passed:
            by_intent[intent]["passed"] += 1
        if r.has_retrieval_requirement:
            by_intent[intent]["recalls_1"].append(r.recall_at_1)
            by_intent[intent]["recalls_3"].append(r.recall_at_3)
            by_intent[intent]["recalls_10"].append(r.recall_at_k)
            by_intent[intent]["mrrs"].append(r.mrr)

    print(f"\n  {'Intent':<25} {'R@1':>6} {'R@3':>6} {'R@10':>6} {'MRR':>8} {'Pass':>8}")
    print(f"  {'-'*65}")
    for intent in sorted(by_intent.keys()):
        d = by_intent[intent]
        r1v = f"{statistics.mean(d['recalls_1']):.2f}" if d['recalls_1'] else "--"
        r3v = f"{statistics.mean(d['recalls_3']):.2f}" if d['recalls_3'] else "--"
        r10v = f"{statistics.mean(d['recalls_10']):.2f}" if d['recalls_10'] else "--"
        mv = f"{statistics.mean(d['mrrs']):.3f}" if d['mrrs'] else "--"
        pv = f"{d['passed']}/{d['total']}"
        print(f"  {intent:<25} {r1v:>6} {r3v:>6} {r10v:>6} {mv:>8} {pv:>8}")

    # Show failures
    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\n  FAILURES:")
        for r in failures:
            print(f"    {r.case_id}: {'; '.join(r.failure_reasons)}")
            print(f"      Retrieved: {r.retrieved_ids}")

    # ---- Latency metrics ----
    print(f"\n  LATENCY")
    print(f"  {'-'*55}")

    if case_latencies:
        print(f"  Retrieval + scoring per query:")
        print(f"    {_fmt_pct(case_latencies)}")
        print(f"    mean: {ms(statistics.mean(case_latencies))}  n={len(case_latencies)}")

    if gate_timings:
        print(f"\n  Gate system (cosine + blended scoring, {len(gate_memories)} memories):")
        print(f"    {_fmt_pct(gate_timings)}")
        print(f"    mean: {ms(statistics.mean(gate_timings))}  n={len(gate_timings)}")

    if retrieval_timings:
        print(f"\n  ChromaDB query_collection (top-10):")
        print(f"    {_fmt_pct(retrieval_timings)}")
        print(f"    mean: {ms(statistics.mean(retrieval_timings))}  n={len(retrieval_timings)}")

    if prompt_build_timings:
        print(f"\n  Full prompt build (all collections):")
        print(f"    {_fmt_pct(prompt_build_timings)}")
        print(f"    mean: {ms(statistics.mean(prompt_build_timings))}  n={len(prompt_build_timings)}")

    if cold_timings and warm_timings:
        cold_mean = statistics.mean(cold_timings)
        warm_mean = statistics.mean(warm_timings)
        speedup = cold_mean / warm_mean if warm_mean > 0 else float("inf")
        print(f"\n  Cold vs warm cache (gate system, {len(gate_memories)} memories):")
        print(f"    Cold (empty cache):  mean={ms(cold_mean)}")
        print(f"    Warm (cached embeds): mean={ms(warm_mean)}")
        print(f"    Speedup:             {speedup:.1f}x")

    # ---- Summary ----
    gate_p50 = pct(gate_timings, 50) if gate_timings else 0
    retrieval_p50 = pct(case_latencies, 50) if case_latencies else 0

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Recall@1:  {r1:.2f}   Recall@3: {r3:.2f}   Recall@10: {r10:.2f}")
    print(f"  MRR: {mrr_val:.2f}   Intent accuracy: {intent_acc:.0%}")
    print(f"  Retrieval:  {_fmt_pct(case_latencies)}")
    print(f"  ChromaDB:   {_fmt_pct(retrieval_timings)}")
    if prompt_build_timings:
        print(f"  Prompt build: {_fmt_pct(prompt_build_timings)}")
    if cold_timings and warm_timings:
        print(f"  Cache: cold={ms(statistics.mean(cold_timings))} warm={ms(statistics.mean(warm_timings))} ({cold_mean/warm_mean:.1f}x)")
    print(f"{'='*70}")

    # ---- Save JSON ----
    output = {
        "timestamp": datetime.now().isoformat(),
        "quality": {
            "recall_at_1": round(r1, 4),
            "recall_at_3": round(r3, 4),
            "recall_at_10": round(r10, 4),
            "precision_at_10": round(p10, 4),
            "mrr": round(mrr_val, 4),
            "intent_accuracy": round(intent_acc, 4),
            "cases_passed": passed,
            "cases_total": len(results),
            "retrieval_cases": n_ret,
            "intent_only_cases": len(intent_only_cases),
        },
        "rank_distribution": {
            "rank_1": rank_histogram.get("rank_1", 0),
            "rank_2_3": rank_histogram.get("rank_2_3", 0),
            "rank_4_5": rank_histogram.get("rank_4_5", 0),
            "rank_6_10": rank_histogram.get("rank_6_10", 0),
            "rank_11_plus": rank_histogram.get("rank_11_plus", 0),
            "total_expected_items": len(all_ranks),
        },
        "latency": {
            "retrieval_p50_ms": round(pct(case_latencies, 50) * 1000, 1) if case_latencies else None,
            "retrieval_p90_ms": round(pct(case_latencies, 90) * 1000, 1) if case_latencies else None,
            "retrieval_p95_ms": round(pct(case_latencies, 95) * 1000, 1) if case_latencies else None,
            "gate_p50_ms": round(pct(gate_timings, 50) * 1000, 1) if gate_timings else None,
            "gate_p90_ms": round(pct(gate_timings, 90) * 1000, 1) if gate_timings else None,
            "gate_p95_ms": round(pct(gate_timings, 95) * 1000, 1) if gate_timings else None,
            "chroma_p50_ms": round(pct(retrieval_timings, 50) * 1000, 1) if retrieval_timings else None,
            "chroma_p90_ms": round(pct(retrieval_timings, 90) * 1000, 1) if retrieval_timings else None,
            "chroma_p95_ms": round(pct(retrieval_timings, 95) * 1000, 1) if retrieval_timings else None,
            "prompt_build_p50_ms": round(pct(prompt_build_timings, 50) * 1000, 1) if prompt_build_timings else None,
            "prompt_build_p90_ms": round(pct(prompt_build_timings, 90) * 1000, 1) if prompt_build_timings else None,
            "prompt_build_p95_ms": round(pct(prompt_build_timings, 95) * 1000, 1) if prompt_build_timings else None,
            "cache_cold_mean_ms": round(statistics.mean(cold_timings) * 1000, 1) if cold_timings else None,
            "cache_warm_mean_ms": round(statistics.mean(warm_timings) * 1000, 1) if warm_timings else None,
            "cache_speedup": round(statistics.mean(cold_timings) / statistics.mean(warm_timings), 1) if cold_timings and warm_timings and statistics.mean(warm_timings) > 0 else None,
        },
        "by_intent": {
            intent: {
                "has_retrieval_cases": bool(d["recalls_1"]),
                "recall_at_1": round(statistics.mean(d["recalls_1"]), 4) if d["recalls_1"] else None,
                "recall_at_3": round(statistics.mean(d["recalls_3"]), 4) if d["recalls_3"] else None,
                "recall_at_10": round(statistics.mean(d["recalls_10"]), 4) if d["recalls_10"] else None,
                "mrr": round(statistics.mean(d["mrrs"]), 4) if d["mrrs"] else None,
                "pass_rate": round(d["passed"] / d["total"], 4) if d["total"] else None,
            }
            for intent, d in by_intent.items()
        },
        "per_case": [r.to_dict() for r in results],
    }
    out_path = Path("data/benchmark_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to {out_path}")

    # Append summary to history file for tracking over time
    history_path = Path("data/benchmark_history.json")
    history: List[Dict] = []
    if history_path.exists():
        try:
            with open(history_path) as f:
                history = json.load(f)
        except (json.JSONDecodeError, ValueError):
            history = []
    history.append({
        "timestamp": output["timestamp"],
        "quality": output["quality"],
        "rank_distribution": output["rank_distribution"],
        "latency": output["latency"],
    })
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"Summary appended to {history_path} ({len(history)} entries)")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
