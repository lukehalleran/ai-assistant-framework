"""
Profile the deployed memories-retrieval chain (steady-state, offline).

Builds the orchestrator exactly as main.py does, then runs
memory_coordinator.get_memories (THE deployed read path — read-only) several
times: run 1 is the cold pass (already mitigated by the startup warmup), runs
2+ are the steady-state cost the user pays every turn. Key callables are
wrapped with cumulative timers so the per-call breakdown shows where the
3-6s warm `memories` task actually goes (embed vs chroma vs gate vs
cross-encoder rerank).

Read-only: no writes besides normal logs. Run with Daemon SHUT DOWN (shares
ChromaDB) and under the memory cap:
    systemd-run --user --scope -p MemoryMax=9G \
        venv/bin/python scripts/profile_memories_task.py
"""

import asyncio
import functools
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUERIES = [
    "how has my sleep been lately",
    "what did we say about the gym routine",
    "my dad and the family situation",
]

_STATS = defaultdict(lambda: [0, 0.0])  # name -> [calls, seconds] per pass
_PASS_STATS = []


def _wrap(obj, attr, label):
    fn = getattr(obj, attr, None)
    if fn is None or getattr(fn, "_profiled", False):
        return
    if asyncio.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def timed(*a, **k):
            t0 = time.perf_counter()
            try:
                return await fn(*a, **k)
            finally:
                s = _STATS[label]
                s[0] += 1
                s[1] += time.perf_counter() - t0
    else:
        @functools.wraps(fn)
        def timed(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                s = _STATS[label]
                s[0] += 1
                s[1] += time.perf_counter() - t0
    timed._profiled = True
    setattr(obj, attr, timed)


def main() -> int:
    from main import build_orchestrator

    print("Building orchestrator (loads models — ~1 min)...")
    orch = build_orchestrator()
    mem = orch.memory_system
    store = mem.chroma_store

    # Instrument the deployed chain
    _wrap(store, "query_collection", "chroma.query_collection")
    embedder = getattr(store, "embedder", None) or getattr(store, "_embedder", None)
    if embedder is not None:
        _wrap(embedder, "encode", "bge.encode")
    try:
        from sentence_transformers import CrossEncoder
        _wrap(CrossEncoder, "predict", "cross_encoder.predict")
    except ImportError:
        pass
    gate = getattr(orch, "gate_system", None) or getattr(mem, "gate_system", None)
    if gate is not None:
        for name in ("filter_memories", "filter_semantic_memories",
                     "batch_cosine_filter", "filter_summaries"):
            if hasattr(gate, name):
                _wrap(gate, name, f"gate.{name}")
    retr = getattr(mem, "memory_retriever", None) or getattr(mem, "retriever", None)
    if retr is not None and hasattr(retr, "_rerank_with_cross_encoder"):
        _wrap(retr, "_rerank_with_cross_encoder", "retriever.rerank")

    async def one_pass(i, q):
        _STATS.clear()
        t0 = time.perf_counter()
        results = await mem.get_memories(q, limit=20)
        total = time.perf_counter() - t0
        _PASS_STATS.append((i, q, total, dict(_STATS), len(results or [])))

    for i, q in enumerate(QUERIES, 1):
        asyncio.run(one_pass(i, q))

    print(f"\n{'pass':<5}{'total':>8}  breakdown")
    for i, q, total, stats, n in _PASS_STATS:
        print(f"\n#{i} ({'cold' if i == 1 else 'warm'}) "
              f"query={q[:40]!r} → {n} results, {total:.2f}s total")
        for label, (calls, secs) in sorted(stats.items(), key=lambda x: -x[1][1]):
            print(f"    {label:<28} {calls:>4} calls  {secs:>7.2f}s")
    print("\nNote: labels overlap (predict runs inside rerank, encode inside "
          "gate calls) — read each as cumulative wall time in that callable, "
          "not exclusive slices.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
