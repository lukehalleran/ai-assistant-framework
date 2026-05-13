"""
Benchmark test fixtures for retrieval quality testing.

Provides session-scoped fixtures that seed real ChromaDB (with real embeddings)
and CorpusManager, plus a MockTimeManager for deterministic recency scoring.

Usage:
    pytest tests/benchmarks/ -m benchmark -v
"""

import os
import yaml
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# Globals for session-level benchmark result collection
# ---------------------------------------------------------------------------
_benchmark_results: List[Dict[str, Any]] = []


def record_benchmark_result(result: Dict[str, Any]):
    """Called by test functions to record results for the session summary."""
    _benchmark_results.append(result)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def benchmark_config():
    """Load and parse the benchmark YAML configuration."""
    yaml_path = Path(__file__).parent.parent / "fixtures" / "retrieval_benchmarks.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def reference_time():
    """Fixed reference time for deterministic scoring."""
    return datetime(2026, 2, 15, 12, 0, 0)


class MockTimeManager:
    """Deterministic time manager for benchmark scoring."""

    def __init__(self, reference_time: datetime):
        self._ref = reference_time

    def current(self) -> datetime:
        return self._ref

    def calculate_active_day_decay(self, timestamp, decay_rate):
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        age_hours = max(0.0, (self._ref - timestamp).total_seconds() / 3600.0)
        return 1.0 / (1.0 + decay_rate * age_hours)


@pytest.fixture(scope="session")
def seeded_stores(benchmark_config, reference_time, tmp_path_factory):
    """
    Create and seed ChromaDB + CorpusManager with benchmark data.

    Returns dict with 'chroma_store', 'corpus_manager', 'id_mapping',
    and 'seed_memories' (the parsed YAML list).
    """
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    from memory.corpus_manager import CorpusManager

    # Temp directories
    chroma_dir = str(tmp_path_factory.mktemp("chroma_bench"))
    corpus_dir = tmp_path_factory.mktemp("corpus_bench")
    corpus_file = str(corpus_dir / "corpus_bench.json")

    # Initialize stores (loads real SentenceTransformer model — slow first time)
    chroma_store = MultiCollectionChromaStore(persist_directory=chroma_dir)
    corpus_mgr = CorpusManager(corpus_file=corpus_file)

    seed_memories = benchmark_config["seed_memories"]
    id_mapping: Dict[str, str] = {}  # yaml_id -> chroma_doc_id

    for mem in seed_memories:
        abs_ts = reference_time + timedelta(hours=mem["timestamp_offset_hours"])
        ts_iso = abs_ts.isoformat()

        # Build content for ChromaDB
        if mem.get("query") and mem.get("response"):
            chroma_content = f"User: {mem['query']}\nAssistant: {mem['response']}"
        else:
            chroma_content = mem.get("content", "")

        # Build metadata
        metadata = {
            "timestamp": ts_iso,
            "truth_score": float(mem.get("truth_score", 0.6)),
            "importance_score": float(mem.get("importance_score", 0.5)),
            "tags": mem.get("tags", ""),
            "benchmark_id": mem["id"],
            "type": mem.get("memory_type", "episodic"),
        }

        # Add query/response to metadata for conversation memories
        if mem.get("query"):
            metadata["query"] = mem["query"]
        if mem.get("response"):
            metadata["response"] = mem["response"]

        # Seed ChromaDB
        doc_id = chroma_store.add_to_collection(
            mem["collection"],
            chroma_content,
            metadata,
        )
        id_mapping[mem["id"]] = doc_id

        # Seed corpus for conversation memories
        if mem.get("also_in_corpus") and mem.get("query") and mem.get("response"):
            corpus_mgr.add_entry(
                query=mem["query"],
                response=mem["response"],
                tags=(mem.get("tags", "").split(",") if mem.get("tags") else []),
                timestamp=abs_ts,
            )

    return {
        "chroma_store": chroma_store,
        "corpus_manager": corpus_mgr,
        "id_mapping": id_mapping,
        "seed_memories": seed_memories,
    }


@pytest.fixture(scope="session")
def retrieval_env(seeded_stores, reference_time, benchmark_config):
    """
    Fully wired retrieval environment with real ChromaDB embeddings.

    Returns dict with retriever, scorer, intent_classifier, and supporting
    objects. Gate system is None (we test scoring/ranking, not cosine gating).
    Hybrid retriever is None (avoids LLM-based query rewriting).
    """
    from memory.memory_retriever import MemoryRetriever
    from memory.memory_scorer import MemoryScorer
    from core.intent_classifier import IntentClassifier

    chroma_store = seeded_stores["chroma_store"]
    corpus_mgr = seeded_stores["corpus_manager"]

    time_mgr = MockTimeManager(reference_time)

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

    return {
        "retriever": retriever,
        "scorer": scorer,
        "intent_classifier": intent_classifier,
        "chroma_store": chroma_store,
        "corpus_manager": corpus_mgr,
        "id_mapping": seeded_stores["id_mapping"],
        "seed_memories": seeded_stores["seed_memories"],
        "test_cases": benchmark_config["test_cases"],
        "reference_time": reference_time,
    }


# ---------------------------------------------------------------------------
# Session summary hook
# ---------------------------------------------------------------------------

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print benchmark result summary at end of session."""
    if not _benchmark_results:
        return

    terminalreporter.section("Retrieval Benchmark Summary")

    # Group by intent
    by_intent: Dict[str, List[Dict]] = {}
    for r in _benchmark_results:
        intent = r.get("intent_expected", "unknown")
        by_intent.setdefault(intent, []).append(r)

    header = f"{'Intent':<25} {'Cases':>5} {'Pass':>5} {'Fail':>5} {'Avg Recall':>10} {'Avg MRR':>8}"
    terminalreporter.write_line(header)
    terminalreporter.write_line("-" * len(header))

    total_pass = total_fail = 0
    for intent in sorted(by_intent.keys()):
        cases = by_intent[intent]
        passed = sum(1 for c in cases if c.get("passed", False))
        failed = len(cases) - passed
        total_pass += passed
        total_fail += failed
        avg_recall = sum(c.get("recall_at_k", 0) for c in cases) / max(len(cases), 1)
        avg_mrr = sum(c.get("mrr", 0) for c in cases) / max(len(cases), 1)
        terminalreporter.write_line(
            f"{intent:<25} {len(cases):>5} {passed:>5} {failed:>5} {avg_recall:>10.3f} {avg_mrr:>8.3f}"
        )

    terminalreporter.write_line("-" * len(header))
    terminalreporter.write_line(
        f"{'TOTAL':<25} {len(_benchmark_results):>5} {total_pass:>5} {total_fail:>5}"
    )
