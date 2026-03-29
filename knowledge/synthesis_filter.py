"""
# knowledge/synthesis_filter.py

Module Contract
- Purpose: 8-stage filter pipeline that processes candidates from knowledge graph
  random walks and identifies genuinely novel, coherent cross-domain connections.
  Cheap stages run first, expensive stages (LLM) run last.
- Class: SynthesisFilter(chroma_store, model_manager, synthesis_memory, wiki_collection)
- Key methods:
  - process_candidate(candidate) -> SynthesisResult  [async, full pipeline]
  - process_batch(candidates) -> dict  [async, batch summary with stats]
- Stage ordering (cheap -> expensive):
  - Stage 0: Text Sanity Filter (~0ms, regex/heuristic)
  - Stage 1: Domain Crossing Gate (~1ms, metadata check)
  - Stage 2: Semantic Distance Gate (~5ms, embedding cosine)
  - Stage 3: Novelty Gate -- External (~10ms, wiki vector search)
  - Stage 4: Novelty Gate -- Internal (~10ms, synthesis memory search)
  - Stage 5: Coherence Judge (~500ms-2s, LLM call)
  - Stage 6: Composite Scoring + Gates (computation only)
  - Stage 7: Synthesis Memory Storage (~10ms, ChromaDB write)
- Inputs: SynthesisCandidate from graph walk engine
- Outputs: SynthesisResult with ACCEPTED or REJECTED status
- Side effects: Stores accepted results in synthesis_results ChromaDB collection
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from config.app_config import (
    SYNTHESIS_COHERENCE_MIN_LEVEL,
    SYNTHESIS_COHERENCE_MODEL,
    SYNTHESIS_COMPOSITE_MIN_SCORE,
    SYNTHESIS_DISTANCE_MAX,
    SYNTHESIS_DISTANCE_MIN,
    SYNTHESIS_LOG_ALL_REJECTIONS,
    SYNTHESIS_MAX_REPETITION_RATIO,
    SYNTHESIS_MIN_DOMAINS,
    SYNTHESIS_MIN_TOKEN_LENGTH,
    SYNTHESIS_NOVELTY_ADJACENT_THRESHOLD,
    SYNTHESIS_NOVELTY_KNOWN_THRESHOLD,
    SYNTHESIS_WEIGHT_COHERENCE,
    SYNTHESIS_WEIGHT_DISTANCE,
    SYNTHESIS_WEIGHT_NOVELTY,
    SYNTHESIS_WEIGHT_STRUCTURAL,
)
from knowledge.synthesis_models import (
    CandidateStatus,
    CoherenceLevel,
    StageResult,
    SynthesisCandidate,
    SynthesisResult,
)
from memory.synthesis_memory import SynthesisMemory
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class SynthesisFilter:
    """8-stage synthesis filter pipeline.

    Args:
        chroma_store: MultiCollectionChromaStore for wiki queries and storage
        model_manager: ModelManager for LLM coherence calls
        synthesis_memory: SynthesisMemory instance (or None to create one)
        wiki_collection: name of the wiki knowledge collection for external novelty
    """

    def __init__(
        self,
        chroma_store,
        model_manager,
        synthesis_memory: Optional[SynthesisMemory] = None,
        wiki_collection: str = "wiki_knowledge",
    ):
        self.store = chroma_store
        self.model_manager = model_manager
        self.memory = synthesis_memory or SynthesisMemory(chroma_store)
        self.wiki_collection = wiki_collection

        # Pipeline stage registry -- order matters
        self._stages = [
            ("text_sanity", self._stage_0_text_sanity),
            ("domain_crossing", self._stage_1_domain_crossing),
            ("semantic_distance", self._stage_2_semantic_distance),
            ("novelty_external", self._stage_3_novelty_external),
            ("novelty_internal", self._stage_4_novelty_internal),
            ("coherence_judge", self._stage_5_coherence_judge),
            ("composite_scoring", self._stage_6_composite_scoring),
        ]

    async def process_candidate(self, candidate: SynthesisCandidate) -> SynthesisResult:
        """Run a single candidate through the full pipeline.

        Returns SynthesisResult with status ACCEPTED or REJECTED.
        Accepted results are automatically stored in synthesis memory (Stage 7).
        """
        result = SynthesisResult(candidate=candidate)

        for stage_name, stage_fn in self._stages:
            start = time.perf_counter()
            try:
                stage_result = await stage_fn(result)
            except Exception as e:
                stage_result = StageResult(
                    stage_name=stage_name,
                    passed=False,
                    reason=f"Stage error: {e}",
                )
                logger.error(f"Synthesis stage {stage_name} error: {e}")

            stage_result.elapsed_ms = (time.perf_counter() - start) * 1000
            result.stage_results.append(stage_result)

            if not stage_result.passed:
                result.reject(stage_name, stage_result.reason)
                if SYNTHESIS_LOG_ALL_REJECTIONS:
                    logger.info(
                        f"[SYNTH REJECT] stage={stage_name} | "
                        f"concepts={candidate.concept_a}<->{candidate.concept_b} | "
                        f"reason={stage_result.reason}"
                    )
                return result

        # All stages passed -- store in synthesis memory (Stage 7)
        result.status = CandidateStatus.ACCEPTED
        try:
            doc_id = self.memory.store_result(result)
            logger.info(
                f"[SYNTH ACCEPT] id={doc_id} | "
                f"concepts={candidate.concept_a}<->{candidate.concept_b} | "
                f"composite={result.composite_score:.3f} | "
                f"coherence={result.coherence_level.name if result.coherence_level else 'N/A'}"
            )
        except Exception as e:
            logger.error(f"Failed to store accepted synthesis result: {e}")

        return result

    async def process_batch(self, candidates: List[SynthesisCandidate]) -> Dict[str, Any]:
        """Process a batch of candidates. Returns summary stats."""
        accepted = []
        rejected = []
        rejection_breakdown: Dict[str, int] = {}
        stage_times: Dict[str, List[float]] = {}

        for candidate in candidates:
            result = await self.process_candidate(candidate)

            if result.status == CandidateStatus.ACCEPTED:
                accepted.append(result)
            else:
                rejected.append(result)
                stage = result.rejection_stage or "unknown"
                rejection_breakdown[stage] = rejection_breakdown.get(stage, 0) + 1

            for sr in result.stage_results:
                if sr.stage_name not in stage_times:
                    stage_times[sr.stage_name] = []
                stage_times[sr.stage_name].append(sr.elapsed_ms)

        avg_times = {
            name: sum(times) / len(times) if times else 0.0
            for name, times in stage_times.items()
        }

        summary = {
            "total": len(candidates),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "rejection_breakdown": rejection_breakdown,
            "accepted_results": accepted,
            "avg_stage_times_ms": avg_times,
        }

        logger.info(
            f"[SYNTH BATCH] total={summary['total']} accepted={summary['accepted']} "
            f"rejected={summary['rejected']} breakdown={rejection_breakdown}"
        )
        return summary

    # -- Stage Implementations -------------------------------------------------

    async def _stage_0_text_sanity(self, result: SynthesisResult) -> StageResult:
        """Reject malformed, empty, or repetitive connection claims."""
        claim = result.candidate.connection_claim.strip()

        tokens = claim.split()
        if len(tokens) < SYNTHESIS_MIN_TOKEN_LENGTH:
            return StageResult(
                stage_name="text_sanity",
                passed=False,
                reason=f"Too short: {len(tokens)} tokens < {SYNTHESIS_MIN_TOKEN_LENGTH}",
            )

        # Must contain at least one verb-like word (cheap heuristic)
        verb_patterns = r'\b\w+(s|ed|ing|es|ize|ise|ate|ify)\b'
        has_verb = bool(re.search(verb_patterns, claim, re.IGNORECASE))
        common_verbs = {
            "is", "are", "was", "were", "has", "have", "had", "do", "does",
            "can", "could", "may", "might", "will", "would", "should", "shall",
            "causes", "leads", "suggests", "implies", "connects", "relates",
            "influences", "affects", "enables", "creates", "produces",
        }
        has_common_verb = any(t.lower() in common_verbs for t in tokens)

        if not has_verb and not has_common_verb:
            return StageResult(
                stage_name="text_sanity",
                passed=False,
                reason="No verbs detected -- likely not a coherent claim",
            )

        # Repetition ratio
        unique_tokens = set(t.lower() for t in tokens)
        repetition_ratio = 1.0 - (len(unique_tokens) / len(tokens))
        if repetition_ratio > SYNTHESIS_MAX_REPETITION_RATIO:
            return StageResult(
                stage_name="text_sanity",
                passed=False,
                reason=f"Repetition ratio {repetition_ratio:.2f} > {SYNTHESIS_MAX_REPETITION_RATIO}",
            )

        return StageResult(
            stage_name="text_sanity",
            passed=True,
            score=1.0 - repetition_ratio,
            metadata={"token_count": len(tokens), "repetition_ratio": repetition_ratio},
        )

    async def _stage_1_domain_crossing(self, result: SynthesisResult) -> StageResult:
        """Reject candidates that don't cross domain boundaries."""
        domains = result.candidate.source_domains

        if len(domains) < SYNTHESIS_MIN_DOMAINS:
            return StageResult(
                stage_name="domain_crossing",
                passed=False,
                reason=f"Only {len(domains)} domain(s): {domains}. Need >= {SYNTHESIS_MIN_DOMAINS}",
            )

        return StageResult(
            stage_name="domain_crossing",
            passed=True,
            score=min(len(domains) / 4.0, 1.0),  # normalize: 4+ domains = 1.0
            metadata={"domains": list(domains), "domain_count": len(domains)},
        )

    async def _stage_2_semantic_distance(self, result: SynthesisResult) -> StageResult:
        """Reject candidates with endpoint distance outside acceptable range."""
        dist = result.candidate.endpoint_distance

        if dist < SYNTHESIS_DISTANCE_MIN:
            return StageResult(
                stage_name="semantic_distance",
                passed=False,
                reason=f"Distance {dist:.3f} < {SYNTHESIS_DISTANCE_MIN} (trivially close)",
            )

        if dist > SYNTHESIS_DISTANCE_MAX:
            return StageResult(
                stage_name="semantic_distance",
                passed=False,
                reason=f"Distance {dist:.3f} > {SYNTHESIS_DISTANCE_MAX} (nonsensical)",
            )

        # Score: peak at middle of range, lower toward edges
        range_mid = (SYNTHESIS_DISTANCE_MIN + SYNTHESIS_DISTANCE_MAX) / 2
        range_half = (SYNTHESIS_DISTANCE_MAX - SYNTHESIS_DISTANCE_MIN) / 2
        distance_score = 1.0 - abs(dist - range_mid) / range_half

        return StageResult(
            stage_name="semantic_distance",
            passed=True,
            score=distance_score,
            metadata={"distance": dist, "range": [SYNTHESIS_DISTANCE_MIN, SYNTHESIS_DISTANCE_MAX]},
        )

    async def _stage_3_novelty_external(self, result: SynthesisResult) -> StageResult:
        """Check if connection is already known in wiki/reference corpus."""
        claim = result.candidate.connection_claim

        try:
            search_results = self.store.query_collection(
                collection_name=self.wiki_collection,
                query_text=claim,
                n_results=3,
            )
        except Exception as e:
            logger.warning(f"Wiki novelty check failed: {e}. Passing by default.")
            return StageResult(
                stage_name="novelty_external", passed=True, score=1.0,
                reason="Wiki query failed -- passing by default",
            )

        if not search_results:
            result.novelty_score_external = 1.0
            return StageResult(
                stage_name="novelty_external", passed=True, score=1.0,
                metadata={"nearest_similarity": 0.0},
            )

        # query_collection returns relevance_score = 1/(1+distance)
        top = search_results[0]
        score = top.get("relevance_score")
        if score is None or score <= 0:
            result.novelty_score_external = 1.0
            return StageResult(
                stage_name="novelty_external", passed=True, score=1.0,
                metadata={"nearest_similarity": 0.0},
            )

        distance = (1.0 / score) - 1.0
        nearest_similarity = 1.0 - distance
        nearest_doc = top.get("content", "")

        result.nearest_known_external = nearest_doc[:200]
        result.novelty_score_external = 1.0 - nearest_similarity

        if nearest_similarity > SYNTHESIS_NOVELTY_KNOWN_THRESHOLD:
            return StageResult(
                stage_name="novelty_external",
                passed=False,
                reason=f"Already known (similarity={nearest_similarity:.3f} > {SYNTHESIS_NOVELTY_KNOWN_THRESHOLD})",
                score=result.novelty_score_external,
                metadata={"nearest_similarity": nearest_similarity, "nearest_doc": nearest_doc[:100]},
            )

        label = "novel" if nearest_similarity < SYNTHESIS_NOVELTY_ADJACENT_THRESHOLD else "adjacent"
        return StageResult(
            stage_name="novelty_external",
            passed=True,
            score=result.novelty_score_external,
            metadata={"nearest_similarity": nearest_similarity, "label": label},
        )

    async def _stage_4_novelty_internal(self, result: SynthesisResult) -> StageResult:
        """Check if this insight has already been discovered by the system.

        Key behavior: if the insight exists but was found via a DIFFERENT path,
        don't reject -- update convergence tracking instead.
        """
        similar = self.memory.find_similar(result.candidate.connection_claim)

        if not similar:
            result.novelty_score_internal = 1.0
            return StageResult(
                stage_name="novelty_internal", passed=True, score=1.0,
                metadata={"seen_before": False},
            )

        existing, similarity = similar[0]
        result.nearest_known_internal = existing.candidate.connection_claim[:200]
        result.novelty_score_internal = 1.0 - similarity

        # Check if this is a NEW path to the same insight
        is_new_path = result.candidate.path_hash not in existing.unique_paths

        if is_new_path:
            # Same insight, different path = convergence signal. Don't reject.
            # Convergence update happens at storage time (Stage 7).
            result.unique_paths = existing.unique_paths | {result.candidate.path_hash}
            source_key = f"{result.candidate.concept_a}|{result.candidate.concept_b}"
            result.unique_sources = existing.unique_sources | {source_key}
            result.convergence_strength = len(result.unique_paths) * len(result.unique_sources)

            return StageResult(
                stage_name="novelty_internal",
                passed=True,
                score=0.5,  # partial novelty -- new path, same destination
                metadata={
                    "seen_before": True,
                    "new_path": True,
                    "existing_paths": len(existing.unique_paths),
                    "convergence_strength": result.convergence_strength,
                },
            )

        # Exact same path hash -- true duplicate, reject
        return StageResult(
            stage_name="novelty_internal",
            passed=False,
            reason=f"Duplicate path (similarity={similarity:.3f}, same path_hash)",
            score=result.novelty_score_internal,
            metadata={"seen_before": True, "new_path": False},
        )

    async def _stage_5_coherence_judge(self, result: SynthesisResult) -> StageResult:
        """LLM-based coherence evaluation. Most expensive stage -- runs last before scoring."""
        claim = result.candidate.connection_claim
        concept_a = result.candidate.concept_a
        concept_b = result.candidate.concept_b

        prompt = (
            f"Rate the coherence of this claimed connection between two concepts.\n\n"
            f"Concept A: {concept_a}\n"
            f"Concept B: {concept_b}\n"
            f"Claimed connection: {claim}\n\n"
            f"Choose exactly ONE rating:\n"
            f"INVALID - The connection is nonsensical, factually wrong, or a non-sequitur\n"
            f"WEAK - There is a superficial or very tenuous link\n"
            f"MODERATE - A plausible connection that makes logical sense\n"
            f"STRONG - A compelling, well-grounded connection\n\n"
            f"Reply with ONLY the rating word (INVALID/WEAK/MODERATE/STRONG) "
            f"followed by a one-sentence justification."
        )

        try:
            response = await self.model_manager.generate_once(
                prompt=prompt,
                model_name=SYNTHESIS_COHERENCE_MODEL,
                system_prompt="You are a critical evaluator of cross-domain knowledge connections. Be skeptical but fair.",
                max_tokens=100,
                temperature=0.1,
            )
        except Exception as e:
            logger.warning(f"Coherence judge LLM call failed: {e}. Passing with MODERATE default.")
            result.coherence_level = CoherenceLevel.MODERATE
            result.coherence_justification = "LLM unavailable -- default MODERATE"
            return StageResult(
                stage_name="coherence_judge", passed=True, score=CoherenceLevel.MODERATE.value,
                reason="LLM call failed -- passing with default",
            )

        # Parse response
        response_text = response.strip()
        coherence_level = None
        for level in CoherenceLevel:
            if response_text.upper().startswith(level.name):
                coherence_level = level
                break

        if coherence_level is None:
            # Fallback: look for keywords anywhere in response
            for level in CoherenceLevel:
                if level.name in response_text.upper():
                    coherence_level = level
                    break

        if coherence_level is None:
            coherence_level = CoherenceLevel.WEAK
            logger.warning(f"Could not parse coherence level from: {response_text[:100]}")

        result.coherence_level = coherence_level
        result.coherence_justification = response_text

        # Check minimum level gate
        min_level = CoherenceLevel[SYNTHESIS_COHERENCE_MIN_LEVEL]
        if coherence_level.value < min_level.value:
            return StageResult(
                stage_name="coherence_judge",
                passed=False,
                reason=f"Coherence {coherence_level.name} < minimum {min_level.name}",
                score=coherence_level.value,
                metadata={"level": coherence_level.name, "justification": response_text[:200]},
            )

        return StageResult(
            stage_name="coherence_judge",
            passed=True,
            score=coherence_level.value,
            metadata={"level": coherence_level.name, "justification": response_text[:200]},
        )

    async def _stage_6_composite_scoring(self, result: SynthesisResult) -> StageResult:
        """Compute composite score from all prior stage results and apply minimum threshold."""
        # Gather component scores
        scores = {sr.stage_name: sr.score for sr in result.stage_results if sr.score is not None}

        coherence_score = scores.get("coherence_judge", 0.5)
        # Combine external and internal novelty
        novelty_ext = result.novelty_score_external
        novelty_int = result.novelty_score_internal
        novelty_score = 0.6 * novelty_ext + 0.4 * novelty_int  # weight external more
        distance_score = scores.get("semantic_distance", 0.5)
        structural_score = scores.get("domain_crossing", 0.5)

        composite = (
            SYNTHESIS_WEIGHT_COHERENCE * coherence_score
            + SYNTHESIS_WEIGHT_NOVELTY * novelty_score
            + SYNTHESIS_WEIGHT_DISTANCE * distance_score
            + SYNTHESIS_WEIGHT_STRUCTURAL * structural_score
        )

        result.composite_score = composite

        if composite < SYNTHESIS_COMPOSITE_MIN_SCORE:
            return StageResult(
                stage_name="composite_scoring",
                passed=False,
                reason=f"Composite {composite:.3f} < {SYNTHESIS_COMPOSITE_MIN_SCORE}",
                score=composite,
                metadata={
                    "coherence": coherence_score,
                    "novelty": novelty_score,
                    "distance": distance_score,
                    "structural": structural_score,
                    "weights": {
                        "coherence": SYNTHESIS_WEIGHT_COHERENCE,
                        "novelty": SYNTHESIS_WEIGHT_NOVELTY,
                        "distance": SYNTHESIS_WEIGHT_DISTANCE,
                        "structural": SYNTHESIS_WEIGHT_STRUCTURAL,
                    },
                },
            )

        return StageResult(
            stage_name="composite_scoring",
            passed=True,
            score=composite,
            metadata={
                "coherence": coherence_score,
                "novelty": novelty_score,
                "distance": distance_score,
                "structural": structural_score,
            },
        )
