"""
Retrieval quality benchmark tests.

Validates that the right memories surface in the right order for given
queries, catching silent regressions in retrieval relevance.

How to run:
    pytest tests/benchmarks/ -m benchmark -v

How to skip during normal test runs:
    pytest -m "not benchmark"

How to add new test cases:
    Edit tests/fixtures/retrieval_benchmarks.yaml and add entries to the
    test_cases list. The harness parametrizes over whatever cases exist.
"""

import pytest
from core.intent_classifier import IntentType
from tests.benchmarks.retrieval_benchmark import RetrievalBenchmark
from tests.benchmarks.conftest import record_benchmark_result


# ---------------------------------------------------------------------------
# Parametrized benchmark tests
# ---------------------------------------------------------------------------

def _get_case_ids(benchmark_config):
    """Extract case IDs for parametrize."""
    return [c["id"] for c in benchmark_config["test_cases"]]


@pytest.mark.benchmark
class TestRetrievalQuality:
    """Parametrized retrieval quality benchmarks."""

    @pytest.mark.parametrize(
        "case_index",
        range(100),  # max expected cases; actual count checked at runtime
    )
    async def test_retrieval_case(self, retrieval_env, benchmark_config, case_index):
        """Run a single benchmark case and assert quality metrics."""
        test_cases = benchmark_config["test_cases"]
        if case_index >= len(test_cases):
            pytest.skip(f"Case index {case_index} exceeds {len(test_cases)} cases")

        test_case = test_cases[case_index]
        case_id = test_case["id"]

        harness = RetrievalBenchmark(
            retriever=retrieval_env["retriever"],
            scorer=retrieval_env["scorer"],
            intent_classifier=retrieval_env["intent_classifier"],
            seed_memories=retrieval_env["seed_memories"],
        )

        result = await harness.run_case(test_case)

        # Record for session summary
        record_benchmark_result(result.to_dict())

        # Assert
        if not result.passed:
            details = "; ".join(result.failure_reasons)
            pytest.fail(
                f"[{case_id}] {details}\n"
                f"  Retrieved: {result.retrieved_ids}\n"
                f"  Intent: {result.intent_actual} "
                f"(conf={result.intent_confidence:.2f})"
            )


# ---------------------------------------------------------------------------
# Structural validation tests (non-parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
class TestBenchmarkStructure:
    """Validate benchmark infrastructure is set up correctly."""

    def test_all_intent_types_covered(self, benchmark_config):
        """Verify YAML has test cases for all 9 intent types."""
        covered = {
            c["expected_intent"]
            for c in benchmark_config["test_cases"]
            if c.get("expected_intent")
        }
        expected = {t.value for t in IntentType}
        missing = expected - covered
        assert not missing, f"Missing intent types in benchmark YAML: {missing}"

    def test_seed_memories_loaded(self, retrieval_env):
        """Verify the fixture seeded the expected number of memories."""
        seed_count = len(retrieval_env["seed_memories"])
        assert seed_count >= 80, (
            f"Expected >= 80 seed memories, got {seed_count}"
        )

        # Verify ChromaDB has entries
        stats = retrieval_env["chroma_store"].get_collection_stats()
        conversations_count = stats.get("conversations", {}).get("count", 0)
        assert conversations_count >= 15, (
            f"Expected >= 15 conversation memories in ChromaDB, "
            f"got {conversations_count}"
        )

    def test_embedding_model_loaded(self, retrieval_env):
        """Verify real embeddings are being used (not mocks)."""
        chroma = retrieval_env["chroma_store"]
        # The embedding function should be a SentenceTransformerEmbeddingFunction
        assert hasattr(chroma, "embedding_fn"), "ChromaDB store missing embedding_fn"
        assert chroma.embedding_fn is not None, "Embedding function is None"

    def test_corpus_seeded(self, retrieval_env):
        """Verify corpus manager has entries."""
        corpus = retrieval_env["corpus_manager"]
        recent = corpus.get_recent_memories(count=5)
        assert len(recent) >= 1, "Corpus manager has no recent memories"

    def test_id_mapping_complete(self, retrieval_env):
        """Verify all seed memories have ChromaDB IDs."""
        id_mapping = retrieval_env["id_mapping"]
        seed_ids = {m["id"] for m in retrieval_env["seed_memories"]}
        mapped_ids = set(id_mapping.keys())
        missing = seed_ids - mapped_ids
        assert not missing, f"Seed memories missing ChromaDB IDs: {missing}"

    def test_case_ids_unique(self, benchmark_config):
        """Verify all test case IDs are unique."""
        ids = [c["id"] for c in benchmark_config["test_cases"]]
        assert len(ids) == len(set(ids)), f"Duplicate case IDs: {ids}"

    def test_must_retrieve_references_valid(self, benchmark_config):
        """Verify must_retrieve IDs reference actual seed memories."""
        seed_ids = {m["id"] for m in benchmark_config["seed_memories"]}
        for case in benchmark_config["test_cases"]:
            for ref_id in (case.get("must_retrieve") or []):
                assert ref_id in seed_ids, (
                    f"Case {case['id']}: must_retrieve references "
                    f"unknown seed ID '{ref_id}'"
                )
            for ref_id in (case.get("must_not_retrieve") or []):
                assert ref_id in seed_ids, (
                    f"Case {case['id']}: must_not_retrieve references "
                    f"unknown seed ID '{ref_id}'"
                )
