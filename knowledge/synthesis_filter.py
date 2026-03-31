"""
# knowledge/synthesis_filter.py

Module Contract
- Purpose: 8-stage filter pipeline that processes candidates from knowledge graph
  random walks and identifies genuinely novel, coherent cross-domain connections.
  Cheap stages run first, expensive stages (LLM) run last.
- Class: SynthesisFilter(chroma_store, model_manager, synthesis_memory)
- Key methods:
  - process_candidate(candidate) -> SynthesisResult  [async, full pipeline]
  - process_batch(candidates) -> dict  [async, batch summary with stats]
  - _factual_skeptic_pass(result, claim, concept_a, concept_b) -> bool  [async, Pass 2 of Stage 5]
  - _parse_coherence_level(response_text) -> CoherenceLevel  [static, extracted from inline parsing]
- Stage ordering (cheap -> expensive):
  - Stage 0: Text Sanity Filter (~0ms, regex/heuristic)
  - Stage 1: Domain Crossing Gate (~1ms, metadata check)
  - Stage 2: Semantic Distance Gate (~5ms, embedding cosine)
  - Stage 3: Novelty Gate -- External (~15ms, FAISS wiki vector search, 40M vectors)
    - Sub-check 1: Claim similarity (full claim vs wiki via FAISS)
    - Sub-check 2: Co-occurrence (bare "A B" conjunction vs wiki via FAISS)
    - Sub-check 3: Template specificity (generic bridge pattern detection)
  - Stage 4: Novelty Gate -- Internal (~10ms, synthesis memory search)
  - Stage 5: Coherence Judge (~500ms-2s, two-pass LLM)
    - Pass 1: Structural coherence -- distinguishes structural isomorphisms from
      loose analogies. LLM rates INVALID/WEAK/MODERATE/STRONG. Prompt focuses on
      shared structural patterns (feedback loops, optimization curves, threshold
      dynamics). System prompt: "Focus on whether a shared structural pattern
      genuinely exists". Response format: Mechanism/Against/For/Rating.
      max_tokens=250.
    - Pass 2: Factual skeptic -- fires ONLY on MODERATE results from Pass 1.
      Checks for debunked science, fabricated mechanisms, false neural pathways.
      Binary PASS/FAIL. On FAIL, downgrades coherence to WEAK (rejection).
      Simplification is acceptable; only provably wrong claims fail.
  - Stage 6: Composite Scoring + Gates (multi-signal novelty composite)
  - Stage 7: Synthesis Memory Storage (~10ms, ChromaDB write)
- Multi-signal novelty composite (Stage 6):
  - claim novelty (w=0.25): 1 - claim_sim
  - co-occurrence novelty (w=0.30): 1 - cooccurrence_sim
  - specificity (w=0.25): 1 - template_sim
  - internal novelty (w=0.20): from synthesis memory
- Helpers:
  - _extract_faiss_similarity(results) -> float  [extract top cosine sim from FAISS results]
  - _compute_template_similarity(claim) -> float  [regex-based generic bridge detection]
- Inputs: SynthesisCandidate from graph walk engine
- Outputs: SynthesisResult with ACCEPTED or REJECTED status
- Side effects: Stores accepted results in synthesis_results ChromaDB collection
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from config.app_config import (
    SYNTHESIS_COOCCURRENCE_KNOWN_THRESHOLD,
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
    SYNTHESIS_NOVELTY_W_CLAIM,
    SYNTHESIS_NOVELTY_W_COOCCURRENCE,
    SYNTHESIS_NOVELTY_W_INTERNAL,
    SYNTHESIS_NOVELTY_W_SPECIFICITY,
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

# -- Generic bridge templates (vacuous patterns that pass coherence but say nothing) --
_GENERIC_TEMPLATES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"both (?:involve|require|depend on|use|exhibit|demonstrate|rely on|need)",
        r"share (?:structural|fundamental|deep|underlying|common|similar) (?:similarities|properties|features)",
        r"(?:operates?|functions?|works?) (?:on|via|through) (?:similar|the same|analogous) principles?",
        r"(?:is|are) (?:an?|the) (?:example|instance|case|form|type) of",
        r"(?:mirrors?|echoes?|reflects?) (?:the|a) (?:same|similar|broader)",
        r"both (?:are|represent) (?:forms?|types?|examples?|instances?) of",
        r"(?:just as|much like|similar to) .{5,40},? (?:so too|similarly|likewise)",
        r"(?:at its|at their) core,? (?:is|are) (?:really|essentially|fundamentally)",
        r"can be (?:seen|viewed|understood|thought of) as (?:a form|an instance|a type) of",
    ]
]

_GENERIC_TOKENS = frozenset({
    "systems", "processes", "principles", "dynamics", "mechanisms",
    "fundamental", "inherently", "essentially", "paradigm", "parallels",
    "interconnected", "holistic", "synergy", "synergistic",
})


def _extract_faiss_similarity(results: list) -> float:
    """Extract the top cosine similarity from FAISS search results (0-1 scale)."""
    if not results:
        return 0.0
    score = results[0].get("similarity", 0.0)
    return max(float(score), 0.0)


def _compute_template_similarity(claim: str) -> float:
    """How close is this claim to a generic bridge platitude? 0=specific, 1=pure template."""
    claim_lower = claim.lower()
    matches = sum(1 for pat in _GENERIC_TEMPLATES if pat.search(claim_lower))
    tokens = claim_lower.split()
    generic_count = sum(1 for t in tokens if t in _GENERIC_TOKENS)

    # Normalize: 2+ template matches or 4+ generic tokens → max penalty
    template_score = min(matches / 2.0, 1.0)
    generic_score = min(generic_count / 4.0, 1.0)
    return max(template_score, generic_score * 0.7)


class SynthesisFilter:
    """8-stage synthesis filter pipeline.

    Args:
        chroma_store: MultiCollectionChromaStore for synthesis memory storage
        model_manager: ModelManager for LLM coherence calls
        synthesis_memory: SynthesisMemory instance (or None to create one)
    """

    def __init__(
        self,
        chroma_store,
        model_manager,
        synthesis_memory: Optional[SynthesisMemory] = None,
    ):
        self.store = chroma_store
        self.model_manager = model_manager
        self.memory = synthesis_memory or SynthesisMemory(chroma_store)

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
        """Check if connection is already known in wiki corpus via FAISS (40M vectors).

        Two sub-checks:
        1. Claim similarity — does the full articulated claim match existing wiki text?
        2. Co-occurrence — do concepts A and B already appear together in wiki?
           High co-occurrence + high claim novelty = known connection, novel phrasing.
        """
        from knowledge.semantic_search import semantic_search_with_neighbors

        claim = result.candidate.connection_claim
        concept_a = result.candidate.concept_a
        concept_b = result.candidate.concept_b

        # --- Sub-check 1: Claim similarity via FAISS ---
        try:
            claim_results = semantic_search_with_neighbors(claim, k=3)
        except Exception as e:
            logger.warning(f"FAISS wiki novelty check failed: {e}. Passing by default.")
            return StageResult(
                stage_name="novelty_external", passed=True, score=1.0,
                reason="FAISS wiki query failed -- passing by default",
            )

        claim_sim = _extract_faiss_similarity(claim_results)
        result.novelty_score_external = 1.0 - claim_sim

        if claim_results:
            result.nearest_known_external = claim_results[0].get("content", claim_results[0].get("text", ""))[:200]

        # Hard gate: direct claim rehash
        if claim_sim > SYNTHESIS_NOVELTY_KNOWN_THRESHOLD:
            return StageResult(
                stage_name="novelty_external",
                passed=False,
                reason=f"Claim already known (similarity={claim_sim:.3f} > {SYNTHESIS_NOVELTY_KNOWN_THRESHOLD})",
                score=result.novelty_score_external,
                metadata={"claim_similarity": claim_sim, "cooccurrence_similarity": 0.0},
            )

        # --- Sub-check 2: Co-occurrence via FAISS ---
        bare_query = f"{concept_a} {concept_b}"
        try:
            cooccurrence_results = semantic_search_with_neighbors(bare_query, k=3)
        except Exception as e:
            logger.debug(f"FAISS co-occurrence check failed: {e}. Skipping.")
            cooccurrence_results = []

        cooccurrence_sim = _extract_faiss_similarity(cooccurrence_results)
        result.cooccurrence_similarity = cooccurrence_sim

        # Hard gate: concepts already co-occur heavily in literature
        if cooccurrence_sim > SYNTHESIS_COOCCURRENCE_KNOWN_THRESHOLD:
            return StageResult(
                stage_name="novelty_external",
                passed=False,
                reason=(
                    f"Concepts already co-occur in literature "
                    f"(cooccurrence={cooccurrence_sim:.3f} > {SYNTHESIS_COOCCURRENCE_KNOWN_THRESHOLD})"
                ),
                score=result.novelty_score_external,
                metadata={"claim_similarity": claim_sim, "cooccurrence_similarity": cooccurrence_sim},
            )

        # --- Sub-check 3: Template specificity ---
        result.template_similarity = _compute_template_similarity(claim)

        label = "novel" if claim_sim < SYNTHESIS_NOVELTY_ADJACENT_THRESHOLD else "adjacent"
        return StageResult(
            stage_name="novelty_external",
            passed=True,
            score=result.novelty_score_external,
            metadata={
                "claim_similarity": claim_sim,
                "cooccurrence_similarity": cooccurrence_sim,
                "template_similarity": result.template_similarity,
                "label": label,
            },
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
        """LLM-based coherence evaluation with optional factual skeptic second pass.

        Pass 1: Structural coherence — does a shared pattern genuinely exist?
        Pass 2 (MODERATE only): Factual skeptic — are the domain-specific claims accurate?
        """
        claim = result.candidate.connection_claim
        concept_a = result.candidate.concept_a
        concept_b = result.candidate.concept_b

        # -- Pass 1: Structural coherence --
        prompt_1 = (
            f"Evaluate this claimed connection between two concepts.\n\n"
            f"Concept A: {concept_a}\n"
            f"Concept B: {concept_b}\n"
            f"Claimed connection: {claim}\n\n"
            f"These concepts are from different domains — that is expected and desired. "
            f"Your job is to evaluate the SPECIFIC MECHANISM or SHARED STRUCTURE described, "
            f"not whether cross-domain connections are valid in general.\n\n"
            f"Key distinction: A connection that identifies a shared mathematical pattern, "
            f"feedback loop, optimization curve, threshold dynamic, or structural isomorphism "
            f"across domains is MODERATE or STRONG — even if the two systems operate through "
            f"different physical substrates. Only rate WEAK if the connection is pure metaphor "
            f"with no identifiable shared process.\n\n"
            f"First, identify the specific mechanism or structural pattern claimed "
            f"(e.g., negative feedback, diminishing returns, threshold collapse, competitive selection).\n"
            f"Then, write one sentence on the strongest reason it might be WRONG.\n"
            f"Then, write one sentence on what makes it genuinely insightful.\n\n"
            f"Finally, choose exactly ONE rating:\n"
            f"INVALID - Factually wrong, or pure wordplay with no shared process at any level of abstraction\n"
            f"WEAK - No specific shared mechanism identified; only a surface-level metaphor or vague gesture at similarity\n"
            f"MODERATE - Identifies a real shared structural pattern (feedback loop, optimization curve, phase transition, etc.) that operates in both domains, even if through different substrates\n"
            f"STRONG - A compelling, specific connection where understanding one system generates testable predictions about the other\n\n"
            f"Format your response as:\n"
            f"Mechanism: <the specific shared pattern or process claimed>\n"
            f"Against: <strongest criticism>\n"
            f"For: <what makes it insightful>\n"
            f"Rating: <INVALID|WEAK|MODERATE|STRONG>"
        )

        try:
            response = await self.model_manager.generate_once(
                prompt=prompt_1,
                model_name=SYNTHESIS_COHERENCE_MODEL,
                system_prompt="You evaluate cross-domain knowledge connections. Focus on whether a shared structural pattern genuinely exists, not on whether the domains seem related. Be skeptical of vague claims but generous toward specific structural parallels.",
                max_tokens=250,
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

        response_text = response.strip()
        coherence_level = self._parse_coherence_level(response_text)

        result.coherence_level = coherence_level
        result.coherence_justification = response_text

        # Check minimum level gate
        min_level = CoherenceLevel[SYNTHESIS_COHERENCE_MIN_LEVEL]
        if coherence_level.value < min_level.value:
            return StageResult(
                stage_name="coherence_judge",
                passed=False,
                reason=f"Coherence {coherence_level.name} < minimum {min_level.name}: {response_text[:150]}",
                score=coherence_level.value,
                metadata={"level": coherence_level.name, "pass": 1, "justification": response_text[:400]},
            )

        # -- Pass 2: Factual skeptic (MODERATE only) --
        # STRONG connections are confident enough to skip. MODERATE gets a second look
        # to catch pseudoscience wrapped in structural language.
        if coherence_level == CoherenceLevel.MODERATE:
            downgraded = await self._factual_skeptic_pass(result, claim, concept_a, concept_b)
            if downgraded:
                return StageResult(
                    stage_name="coherence_judge",
                    passed=False,
                    reason=f"Factual skeptic downgraded to {result.coherence_level.name}: {result.coherence_justification[:150]}",
                    score=result.coherence_level.value,
                    metadata={"level": result.coherence_level.name, "pass": 2, "justification": result.coherence_justification[:400]},
                )

        return StageResult(
            stage_name="coherence_judge",
            passed=True,
            score=result.coherence_level.value,
            metadata={"level": result.coherence_level.name, "pass": 1 if coherence_level != CoherenceLevel.MODERATE else 2, "justification": response_text[:200]},
        )

    async def _factual_skeptic_pass(
        self, result: SynthesisResult, claim: str, concept_a: str, concept_b: str
    ) -> bool:
        """Second-pass factual accuracy check for MODERATE-rated candidates.

        Narrowly targets pseudoscience and fabricated mechanisms. Does NOT
        penalize simplification — cross-domain connections are inherently
        simplified. Only downgrades when domain-specific claims are provably
        wrong or based on debunked science.

        Returns True if the candidate should be downgraded (rejected).
        """
        prompt = (
            f"A cross-domain connection claims: {claim}\n\n"
            f"Check ONLY whether the domain-specific facts are wrong or based on debunked science.\n\n"
            f"Examples of what FAILS this check:\n"
            f"- 'Left-brain people are more logical' (debunked lateralization myth)\n"
            f"- 'Mirror neurons cause empathy' (oversimplified; mirror neuron function is disputed)\n"
            f"- ANY claim that mirror neurons cause a specific behavioral response (viewing triggers, anticipatory responses, etc.) — mirror neuron function in humans is not established\n"
            f"- 'Mozart effect improves intelligence' (debunked)\n"
            f"- Claiming a specific neural pathway, enzyme, or mechanism that doesn't exist\n"
            f"- Applying deterministic chaos theory (butterfly effect, sensitivity to initial conditions) to inherently stochastic systems like stock markets or weather — chaos theory describes deterministic systems with sensitive dependence, not systems driven by random external shocks\n\n"
            f"Examples of what PASSES this check:\n"
            f"- Describing feedback loops that genuinely exist in both domains, even if simplified\n"
            f"- Structural parallels where both sides are real phenomena, even if the connection is approximate\n"
            f"- Using well-established concepts (Wolff's law, PID control, diminishing returns) correctly\n"
            f"- Applying mathematical concepts (threshold dynamics, optimization curves) where the math genuinely applies in both domains\n\n"
            f"IMPORTANT: Simplification is expected and acceptable. Only flag claims where a domain "
            f"expert would say 'that specific mechanism is wrong or doesn't exist,' NOT 'that's a "
            f"simplification of how it actually works.'\n\n"
            f"Respond:\n"
            f"PASS - The domain facts are real (even if simplified)\n"
            f"FAIL - A specific claim is factually wrong or based on debunked science (state which one)\n\n"
            f"One sentence of reasoning, then PASS or FAIL on its own line."
        )

        try:
            response = await self.model_manager.generate_once(
                prompt=prompt,
                model_name=SYNTHESIS_COHERENCE_MODEL,
                system_prompt="You are a domain-accuracy fact-checker. Focus only on whether the specific scientific or technical claims are correct, not on whether the cross-domain analogy is interesting.",
                max_tokens=150,
                temperature=0.1,
            )
        except Exception as e:
            logger.debug(f"Factual skeptic pass failed: {e}. Keeping MODERATE.")
            return False

        response_text = response.strip()
        response_upper = response_text.upper()

        # Look for FAIL verdict — downgrade to WEAK
        # Check last line first (most reliable), then anywhere
        last_line = response_text.strip().split("\n")[-1].strip().upper()
        is_fail = last_line == "FAIL" or (
            "FAIL" in response_upper and "PASS" not in response_upper
        )

        if is_fail:
            result.coherence_level = CoherenceLevel.WEAK
            result.coherence_justification += f"\n[FACTUAL SKEPTIC] {response_text}"
            logger.info(f"[SYNTH SKEPTIC] Downgraded to WEAK: {concept_a}<->{concept_b}")
            return True

        # PASS or unparseable — keep MODERATE
        return False

    @staticmethod
    def _parse_coherence_level(response_text: str) -> CoherenceLevel:
        """Parse coherence level from LLM response text."""
        # Primary: find "Rating: LEVEL" line
        for line in response_text.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("RATING:"):
                rating_part = stripped[len("RATING:"):].strip().upper()
                for level in CoherenceLevel:
                    if rating_part.startswith(level.name):
                        return level
                break

        # Fallback: response starts with level name
        for level in CoherenceLevel:
            if response_text.upper().startswith(level.name):
                return level

        # Fallback: keyword anywhere
        for level in CoherenceLevel:
            if level.name in response_text.upper():
                return level

        logger.warning(f"Could not parse coherence level from: {response_text[:100]}")
        return CoherenceLevel.WEAK

    async def _stage_6_composite_scoring(self, result: SynthesisResult) -> StageResult:
        """Compute composite score from all prior stage results and apply minimum threshold.

        Novelty is itself a 4-signal composite:
        - claim novelty (1 - claim_sim): does the articulated claim match wiki?
        - co-occurrence novelty (1 - cooccurrence_sim): do A and B appear together?
        - specificity (1 - template_sim): is the claim vacuous/generic?
        - internal novelty: has the system seen this before?
        """
        # Gather component scores
        scores = {sr.stage_name: sr.score for sr in result.stage_results if sr.score is not None}

        coherence_score = scores.get("coherence_judge", 0.5)
        distance_score = scores.get("semantic_distance", 0.5)
        structural_score = scores.get("domain_crossing", 0.5)

        # Multi-signal novelty composite
        claim_novelty = result.novelty_score_external             # 1 - claim_sim
        cooccurrence_novelty = 1.0 - result.cooccurrence_similarity
        specificity = 1.0 - result.template_similarity
        internal_novelty = result.novelty_score_internal

        novelty_score = (
            SYNTHESIS_NOVELTY_W_CLAIM * claim_novelty
            + SYNTHESIS_NOVELTY_W_COOCCURRENCE * cooccurrence_novelty
            + SYNTHESIS_NOVELTY_W_SPECIFICITY * specificity
            + SYNTHESIS_NOVELTY_W_INTERNAL * internal_novelty
        )

        composite = (
            SYNTHESIS_WEIGHT_COHERENCE * coherence_score
            + SYNTHESIS_WEIGHT_NOVELTY * novelty_score
            + SYNTHESIS_WEIGHT_DISTANCE * distance_score
            + SYNTHESIS_WEIGHT_STRUCTURAL * structural_score
        )

        result.composite_score = composite

        novelty_detail = {
            "claim_novelty": claim_novelty,
            "cooccurrence_novelty": cooccurrence_novelty,
            "specificity": specificity,
            "internal_novelty": internal_novelty,
            "novelty_composite": novelty_score,
        }

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
                    **novelty_detail,
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
                **novelty_detail,
            },
        )
