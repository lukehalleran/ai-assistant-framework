"""
Retrieval quality benchmark harness.

Orchestrates a single benchmark run: classifies intent, applies weight
overrides, runs retrieval, and computes recall/precision/MRR metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class BenchmarkResult:
    """Result of a single benchmark case."""

    case_id: str
    intent_expected: str
    intent_actual: str
    intent_confidence: float
    recall_at_k: float
    precision_at_k: float
    mrr: float
    false_retrievals: List[str]
    order_violations: List[str]
    retrieved_ids: List[str]
    passed: bool
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for session-level collection."""
        return {
            "case_id": self.case_id,
            "intent_expected": self.intent_expected,
            "intent_actual": self.intent_actual,
            "intent_confidence": self.intent_confidence,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr": self.mrr,
            "false_retrievals": self.false_retrievals,
            "order_violations": self.order_violations,
            "retrieved_ids": self.retrieved_ids,
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
        }


def _identify_memory(memory: Dict, seed_memories: List[Dict]) -> Optional[str]:
    """
    Map a returned memory to its seed memory ID using benchmark_id metadata
    or marker substring matching.
    """
    # Try metadata benchmark_id first (ChromaDB entries)
    bid = (memory.get("metadata") or {}).get("benchmark_id")
    if bid:
        return bid

    # Fall back to marker substring matching (corpus entries)
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


class RetrievalBenchmark:
    """
    Benchmark harness that exercises the real retrieval + scoring pipeline
    with intent-driven weight overrides.
    """

    def __init__(self, retriever, scorer, intent_classifier, seed_memories):
        self.retriever = retriever
        self.scorer = scorer
        self.intent_classifier = intent_classifier
        self.seed_memories = seed_memories

    async def run_case(self, test_case: Dict) -> BenchmarkResult:
        """Execute a single benchmark case and return metrics."""
        query = test_case["query"]
        top_k = test_case.get("top_k", 10)
        must_retrieve = set(test_case.get("must_retrieve") or [])
        must_not_retrieve = set(test_case.get("must_not_retrieve") or [])
        min_recall = test_case.get("min_recall", 0.8)
        expected_order = test_case.get("expected_score_order") or []
        failure_reasons: List[str] = []

        # 1. Classify intent
        intent_result = self.intent_classifier.classify(query)

        # 2. Apply intent weight overrides to scorer (mimics what builder does)
        self.scorer._intent_weight_overrides = intent_result.weight_overrides or None
        try:
            # 3. Run retrieval through the real pipeline
            memories = await self.retriever.get_memories(
                query=query,
                limit=top_k,
            )
        finally:
            # 4. Always clear weight overrides
            self.scorer._intent_weight_overrides = None

        # 5. Identify retrieved memories by seed ID
        retrieved_ids: List[str] = []
        for mem in memories:
            seed_id = _identify_memory(mem, self.seed_memories)
            if seed_id and seed_id not in retrieved_ids:
                retrieved_ids.append(seed_id)

        retrieved_set = set(retrieved_ids)

        # 6. Compute recall@K
        if must_retrieve:
            hits = must_retrieve & retrieved_set
            recall_at_k = len(hits) / len(must_retrieve)
        else:
            recall_at_k = 1.0  # no requirements = trivially satisfied

        # 7. Compute precision@K (fraction of retrieved that were expected)
        if must_retrieve and retrieved_ids:
            precision_at_k = len(must_retrieve & retrieved_set) / len(retrieved_ids)
        else:
            precision_at_k = 1.0 if not must_retrieve else 0.0

        # 8. Compute MRR (mean reciprocal rank of first must_retrieve hit)
        mrr = 0.0
        if must_retrieve:
            for i, rid in enumerate(retrieved_ids):
                if rid in must_retrieve:
                    mrr = 1.0 / (i + 1)
                    break

        # 9. Check negative assertions
        false_retrievals = list(must_not_retrieve & retrieved_set)

        # 10. Check score ordering
        order_violations: List[str] = []
        if expected_order and len(expected_order) >= 2:
            for i in range(len(expected_order) - 1):
                a, b = expected_order[i], expected_order[i + 1]
                if a in retrieved_set and b in retrieved_set:
                    pos_a = retrieved_ids.index(a) if a in retrieved_ids else 999
                    pos_b = retrieved_ids.index(b) if b in retrieved_ids else 999
                    if pos_a > pos_b:
                        order_violations.append(
                            f"{a} (pos {pos_a}) should rank before {b} (pos {pos_b})"
                        )

        # 11. Determine pass/fail
        passed = True

        # Intent check
        if test_case.get("expected_intent"):
            if intent_result.intent.value != test_case["expected_intent"]:
                passed = False
                failure_reasons.append(
                    f"Intent: expected {test_case['expected_intent']}, "
                    f"got {intent_result.intent.value}"
                )

        # Confidence check
        conf_min = test_case.get("expected_confidence_min", 0.0)
        if intent_result.confidence < conf_min:
            passed = False
            failure_reasons.append(
                f"Confidence: {intent_result.confidence:.2f} < {conf_min:.2f}"
            )

        # Recall check
        if recall_at_k < min_recall:
            passed = False
            missing = must_retrieve - retrieved_set
            failure_reasons.append(
                f"Recall: {recall_at_k:.2f} < {min_recall:.2f} "
                f"(missing: {missing})"
            )

        # Negative check
        if false_retrievals:
            passed = False
            failure_reasons.append(
                f"False retrievals: {false_retrievals}"
            )

        # Order check
        if order_violations:
            passed = False
            failure_reasons.append(
                f"Order violations: {order_violations}"
            )

        return BenchmarkResult(
            case_id=test_case["id"],
            intent_expected=test_case.get("expected_intent", ""),
            intent_actual=intent_result.intent.value,
            intent_confidence=intent_result.confidence,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            mrr=mrr,
            false_retrievals=false_retrievals,
            order_violations=order_violations,
            retrieved_ids=retrieved_ids,
            passed=passed,
            failure_reasons=failure_reasons,
        )
