"""
Calibration test suite for the synthesis filter pipeline.

Loads labeled candidates from tests/fixtures/calibration_candidates.json,
runs them through the filter, and produces a confusion matrix + per-stage
diagnostic report.

Usage:
    pytest tests/test_synthesis_calibration.py -v -s

The main test always passes -- it's a diagnostic tool, not a gate.
The output shows precision, recall, and per-stage score distributions
for manual threshold tuning.

Tiers:
    sanity_fail      - Should die at Stage 0 (text sanity)
    trivial          - Well-known connections, varied distances. Should be rejected.
    noise            - Clearly forced/incoherent. Should be rejected by coherence.
    noise_borderline - Plausible-sounding but wrong. Tests coherence boundary.
    interesting_known - Good connections that are published. Should fail novelty.
    interesting_novel - Non-obvious AND new. Should be accepted.
    boundary         - Hard middle cases. Diagnostic -- expected outcome is a guess.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge.synthesis_filter import SynthesisFilter
from knowledge.synthesis_models import (
    CandidateStatus,
    CoherenceLevel,
    SynthesisCandidate,
    SynthesisResult,
)
from memory.synthesis_memory import SynthesisMemory

FIXTURES_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "calibration_candidates.json"
)

# Tiers expected to be rejected
_REJECT_TIERS = {"sanity_fail", "trivial", "noise", "noise_borderline", "interesting_known"}
# Tiers expected to be accepted
_ACCEPT_TIERS = {"interesting_novel"}
# Diagnostic-only tiers (no expected correctness)
_DIAGNOSTIC_TIERS = {"boundary"}

ALL_VALID_TIERS = _REJECT_TIERS | _ACCEPT_TIERS | _DIAGNOSTIC_TIERS


def _load_calibration_candidates() -> List[Dict[str, Any]]:
    """Load labeled candidates from JSON fixture, skipping comment entries."""
    with open(FIXTURES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [c for c in data["candidates"] if "_comment" not in c]


def _to_synthesis_candidate(entry: Dict[str, Any]) -> SynthesisCandidate:
    """Convert a fixture entry to a SynthesisCandidate."""
    return SynthesisCandidate(
        concept_a=entry["concept_a"],
        concept_b=entry["concept_b"],
        connection_claim=entry["connection_claim"],
        walk_path=[entry["concept_a"].lower(), entry["concept_b"].lower()],
        source_domains=set(entry["source_domains"]),
        endpoint_distance=entry["endpoint_distance"],
        timestamp=datetime.now(),
    )


def _build_mock_store():
    """Build a mock store for synthesis memory (ChromaDB) — wiki queries use FAISS."""
    store = MagicMock()

    def query_collection_side_effect(collection_name, query_text, n_results=3):
        if collection_name == "synthesis_results":
            return []
        return []

    store.query_collection = MagicMock(side_effect=query_collection_side_effect)
    store.add_to_collection = MagicMock(return_value="synth_doc_1")
    store.update_metadata = MagicMock()
    store.get_collection_stats = MagicMock(return_value={
        "synthesis_results": {"count": 0},
    })
    return store


# Topics that should trigger high novelty similarity (trivial)
_TRIVIAL_TOPICS = {
    "physical exercise", "mental health", "sleep", "cognitive", "compound interest",
    "retirement", "musical instrument", "mathematical reasoning",
    "bilingualism", "dementia", "dark chocolate", "cardiovascular",
    "pet ownership", "blood pressure", "nature", "creativity",
    "gut microbiome", "mood", "cold exposure", "immune",
    "video games", "surgical", "standing desk", "back pain",
    "gratitude", "sleep quality",
    # Claim-text keywords (concepts reworded in claims)
    "musical training", "pattern recognition", "spatial-temporal",
    "gut-brain", "serotonin", "vagus nerve", "endorphin",
    "neuroplasticity", "oxytocin", "cortisol",
    "flavanol", "nitric oxide", "endothelial",
    "norepinephrine", "wim hof", "cold water immersion",
    "laparoscopic", "fine motor", "spinal compression",
    "prefrontal cortex", "cognitive arousal",
}

# Topics that should trigger moderate novelty similarity (known-interesting)
_KNOWN_TOPICS = {
    "ant colony", "tcp", "congestion", "immune system", "caching", "cache",
    "mycelial", "content delivery", "mycorrhizal", "predator-prey",
    "autoscaling", "lotka-volterra", "counterpoint", "adversarial",
    "gans", "bach",
    # Claim-text keywords
    "memory b cells", "cdn", "wood wide web",
    "dorigo", "slow-start",
}


def _mock_faiss_wiki_search(query, k=3):
    """Tier-aware FAISS mock for novelty checks in calibration tests.

    Returns FAISS-format results with similarity scores calibrated to match
    the old ChromaDB relevance_score → cosine similarity conversion:
    - Trivial: similarity=0.90 (was relevance_score=0.91)
    - Known: similarity=0.82 (was relevance_score=0.85)
    - Novel: similarity=0.05 (was relevance_score=0.40 → clamped to ~0)
    """
    query_lower = query.lower()
    is_trivial = any(topic in query_lower for topic in _TRIVIAL_TOPICS)
    is_known = any(topic in query_lower for topic in _KNOWN_TOPICS)

    if is_trivial:
        return [{
            "text": query[:200],
            "content": query[:200],
            "title": "Wikipedia: Well-Known Connection",
            "source": "wikipedia",
            "similarity": 0.90,
        }]
    elif is_known:
        return [{
            "text": f"Published analysis of {query[:100]}",
            "content": f"Published analysis of {query[:100]}",
            "title": "Academic Survey",
            "source": "wikipedia",
            "similarity": 0.82,
        }]
    else:
        return [{
            "text": "Unrelated article about ancient pottery techniques.",
            "content": "Unrelated article about ancient pottery techniques.",
            "title": "Pottery",
            "source": "wikipedia",
            "similarity": 0.05,
        }]


def _build_mock_model_manager():
    """Build a mock ModelManager for the coherence judge.

    Keyword heuristics simulate LLM coherence judgments:
    - Clear noise (forced metaphors) -> INVALID/WEAK
    - Borderline noise (plausible pseudoscience) -> WEAK
    - Trivial (well-known, well-articulated) -> STRONG
    - Novel mechanistic claims -> MODERATE/STRONG
    """
    mm = MagicMock()

    # Clear noise markers
    _hard_noise = [
        "living constitution", "nash equilibrium", "temporal singularity",
        "coriolis", "dark matter", "foreign keys", "hegelian",
        "categorical imperative", "just-in-time supply chain",
        "instantaneously",  # false physics
        "living room",  # domestic trivia
    ]

    # Soft noise markers (analogies that might fool naive checks)
    _soft_noise = [
        "just as", "like a", "similar to", "mirrors", "embodies",
        "represent", "analogous to",
    ]

    # Overstatement markers (causal overclaims and lazy analogies)
    _overstatement = [
        "directly explains", "exactly as", "both involve",
    ]

    # Borderline noise markers (pseudoscience wrapped in science language)
    _pseudoscience = [
        "mirror neurons", "left-brain", "right-brain", "mozart effect",
        "sympathetic vibration", "mercury retrograde", "electromagnetic field",
        "neural resonance",
    ]

    # Strong mechanistic markers
    _mechanistic = [
        "feedback mechanism", "stress-recovery", "probability distribution",
        "consensus", "dampening", "multi-armed bandit", "remodel",
        "substrate degradation", "symbiotic", "scalar feedback",
        "energy-minimization", "supercompensation", "diminishing returns",
        "overfitting", "chunking", "voronoi",
    ]

    async def coherence_judge(prompt, model_name=None, system_prompt="",
                              max_tokens=200, temperature=0.1, **kwargs):
        prompt_lower = prompt.lower()

        # Pseudoscience check (borderline noise)
        if any(marker in prompt_lower for marker in _pseudoscience):
            return "Against: The claimed mechanism relies on debunked or oversimplified science.\nFor: Uses scientific vocabulary.\nRating: WEAK"

        # Hard noise check
        hard_count = sum(1 for phrase in _hard_noise if phrase in prompt_lower)
        if hard_count >= 1:
            return "Against: This is a forced analogy with no mechanistic basis.\nFor: None.\nRating: INVALID"

        # Overstatement check (causal overclaims)
        if any(marker in prompt_lower for marker in _overstatement):
            return "Against: The claim overstates a causal relationship without mechanistic support.\nFor: Uses real domain vocabulary.\nRating: WEAK"

        # Soft noise check
        soft_count = sum(1 for phrase in _soft_noise if phrase in prompt_lower)
        if soft_count >= 2:
            return "Against: The connection relies on surface-level metaphor rather than mechanism.\nFor: Identifies a superficial parallel.\nRating: WEAK"

        # Mechanistic check
        mech_count = sum(1 for m in _mechanistic if m in prompt_lower)
        if mech_count >= 2:
            return "Against: Could be coincidental structural similarity.\nFor: Identifies a compelling, mechanistically grounded connection with falsifiable claims.\nRating: STRONG"
        if mech_count == 1:
            return "Against: Only one mechanistic detail provided.\nFor: A plausible connection with some mechanistic basis.\nRating: MODERATE"

        # Default: moderate for anything that doesn't trigger specific heuristics
        return "Against: No obvious flaw but limited mechanistic detail.\nFor: A plausible connection with some logical basis.\nRating: MODERATE"

    mm.generate_once = AsyncMock(side_effect=coherence_judge)
    return mm


@patch("knowledge.semantic_search.semantic_search_with_neighbors", side_effect=_mock_faiss_wiki_search)
class TestSynthesisCalibration:
    """Diagnostic calibration suite."""

    @pytest.mark.asyncio
    async def test_filter_calibration_report(self, mock_faiss):
        """Run all labeled candidates through filter and print diagnostic report.

        This test always passes. The output is the diagnostic.
        """
        candidates_data = _load_calibration_candidates()
        store = _build_mock_store()
        model_manager = _build_mock_model_manager()
        synthesis_memory = SynthesisMemory(store)

        pipeline = SynthesisFilter(
            chroma_store=store,
            model_manager=model_manager,
            synthesis_memory=synthesis_memory,
        )

        results_by_tier: Dict[str, List[Dict]] = defaultdict(list)
        all_results = []

        for entry in candidates_data:
            candidate = _to_synthesis_candidate(entry)
            result = await pipeline.process_candidate(candidate)

            record = {
                "concept_a": entry["concept_a"],
                "concept_b": entry["concept_b"],
                "tier": entry["tier"],
                "expected": entry["expected_outcome"],
                "actual": "accepted" if result.status == CandidateStatus.ACCEPTED else "rejected",
                "rejection_stage": result.rejection_stage,
                "expected_rejection_stage": entry.get("expected_rejection_stage"),
                "composite_score": result.composite_score,
                "coherence": result.coherence_level.name if result.coherence_level else "N/A",
                "novelty_ext": result.novelty_score_external,
                "novelty_int": result.novelty_score_internal,
                "cooccurrence_sim": result.cooccurrence_similarity,
                "template_sim": result.template_similarity,
                "stage_scores": {
                    sr.stage_name: sr.score for sr in result.stage_results if sr.score is not None
                },
                "notes": entry.get("notes", ""),
            }
            results_by_tier[entry["tier"]].append(record)
            all_results.append(record)

        # --- Confusion Matrix (excluding boundary tier) ---
        scored = [r for r in all_results if r["tier"] not in _DIAGNOSTIC_TIERS]
        tp = [r for r in scored if r["expected"] == "accepted" and r["actual"] == "accepted"]
        fp = [r for r in scored if r["expected"] == "rejected" and r["actual"] == "accepted"]
        tn = [r for r in scored if r["expected"] == "rejected" and r["actual"] == "rejected"]
        fn = [r for r in scored if r["expected"] == "accepted" and r["actual"] == "rejected"]

        precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0
        recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print("\n" + "=" * 80)
        print("SYNTHESIS FILTER CALIBRATION REPORT")
        print("=" * 80)

        print(f"\n  Total candidates: {len(all_results)}")
        print(f"  Scored (excl. boundary): {len(scored)}")

        print(f"\n--- Confusion Matrix ---")
        print(f"  True Positives  (novel accepted):    {len(tp)}")
        print(f"  False Positives (should-reject accepted): {len(fp)}")
        print(f"  True Negatives  (should-reject rejected): {len(tn)}")
        print(f"  False Negatives (novel rejected):    {len(fn)}")
        print(f"\n  Precision: {precision:.2%}")
        print(f"  Recall:    {recall:.2%}")
        print(f"  F1 Score:  {f1:.2%}")

        # --- Per-Tier Analysis ---
        tier_order = [
            "sanity_fail", "trivial", "noise", "noise_borderline",
            "interesting_known", "interesting_novel", "boundary",
        ]
        for tier in tier_order:
            tier_results = results_by_tier.get(tier, [])
            if not tier_results:
                continue
            accepted = [r for r in tier_results if r["actual"] == "accepted"]
            rejected = [r for r in tier_results if r["actual"] == "rejected"]

            label = f"{tier.upper()} ({len(tier_results)} candidates)"
            print(f"\n--- {label} ---")
            print(f"  Accepted: {len(accepted)}/{len(tier_results)}")
            print(f"  Rejected: {len(rejected)}/{len(tier_results)}")

            # Rejection stage breakdown
            stage_counts = defaultdict(int)
            for r in rejected:
                stage_counts[r["rejection_stage"] or "none"] += 1
            if stage_counts:
                print(f"  Rejection stages: {dict(stage_counts)}")

            # Stage correctness for sanity_fail tier
            if tier == "sanity_fail":
                correct_stage = sum(
                    1 for r in tier_results
                    if r["actual"] == "rejected"
                    and r["rejection_stage"] == r.get("expected_rejection_stage")
                )
                print(f"  Correct rejection stage: {correct_stage}/{len(tier_results)}")

            # Score distributions
            composites = [r["composite_score"] for r in tier_results if r["composite_score"] > 0]
            if composites:
                print(f"  Composite scores: min={min(composites):.3f}, "
                      f"max={max(composites):.3f}, "
                      f"mean={sum(composites)/len(composites):.3f}")

            # Coherence distribution
            coherence_counts = defaultdict(int)
            for r in tier_results:
                coherence_counts[r["coherence"]] += 1
            print(f"  Coherence levels: {dict(coherence_counts)}")

            # Novelty external scores
            novelties = [r["novelty_ext"] for r in tier_results if r["novelty_ext"] > 0]
            if novelties:
                print(f"  Novelty (external): min={min(novelties):.3f}, "
                      f"max={max(novelties):.3f}, "
                      f"mean={sum(novelties)/len(novelties):.3f}")

            # Co-occurrence similarity scores
            cooccurrences = [r["cooccurrence_sim"] for r in tier_results]
            if any(v > 0 for v in cooccurrences):
                print(f"  Co-occurrence sim:  min={min(cooccurrences):.3f}, "
                      f"max={max(cooccurrences):.3f}, "
                      f"mean={sum(cooccurrences)/len(cooccurrences):.3f}")

            # Template similarity scores
            templates = [r["template_sim"] for r in tier_results]
            if any(v > 0 for v in templates):
                print(f"  Template sim:       min={min(templates):.3f}, "
                      f"max={max(templates):.3f}, "
                      f"mean={sum(templates)/len(templates):.3f}")

        # --- Boundary Cases (composite near threshold) ---
        threshold = 0.40
        boundary = [r for r in all_results if 0.30 <= r["composite_score"] <= 0.50]
        if boundary:
            print(f"\n--- Composite Boundary (0.30-0.50) ---")
            for r in sorted(boundary, key=lambda x: x["composite_score"]):
                status = "PASS" if r["actual"] == "accepted" else "FAIL"
                print(f"  [{status}] {r['concept_a']} <-> {r['concept_b']}: "
                      f"composite={r['composite_score']:.3f}, "
                      f"coherence={r['coherence']}, "
                      f"cooccur={r['cooccurrence_sim']:.2f}, "
                      f"template={r['template_sim']:.2f}, "
                      f"tier={r['tier']}")

        # --- Misclassifications ---
        misclassified = fp + fn
        if misclassified:
            print(f"\n--- Misclassifications ({len(misclassified)}) ---")
            for r in misclassified:
                direction = "FP" if r in fp else "FN"
                print(f"  [{direction}] {r['concept_a']} <-> {r['concept_b']}: "
                      f"tier={r['tier']}, composite={r['composite_score']:.3f}, "
                      f"coherence={r['coherence']}, "
                      f"cooccur={r['cooccurrence_sim']:.2f}, "
                      f"template={r['template_sim']:.2f}, "
                      f"rejection_stage={r['rejection_stage']}")
                print(f"         notes: {r['notes']}")

        # --- Boundary tier analysis (diagnostic only) ---
        boundary_tier = results_by_tier.get("boundary", [])
        if boundary_tier:
            print(f"\n--- BOUNDARY TIER DETAIL (diagnostic, no scoring) ---")
            for r in boundary_tier:
                actual = "ACCEPTED" if r["actual"] == "accepted" else f"REJECTED@{r['rejection_stage']}"
                guess = r["expected"].upper()
                match = "OK" if r["expected"] == r["actual"] else "SURPRISE"
                print(f"  [{match}] {r['concept_a']} <-> {r['concept_b']}: "
                      f"expected={guess}, actual={actual}, "
                      f"composite={r['composite_score']:.3f}, "
                      f"coherence={r['coherence']}, "
                      f"cooccur={r['cooccurrence_sim']:.2f}, "
                      f"template={r['template_sim']:.2f}")

        print("\n" + "=" * 80)
        assert True  # diagnostic — always passes

    @pytest.mark.asyncio
    async def test_fixture_structure_valid(self, mock_faiss):
        """Verify the calibration fixture has correct structure and tier counts."""
        candidates = _load_calibration_candidates()

        assert len(candidates) >= 50, f"Expected >= 50 candidates, got {len(candidates)}"

        tiers = defaultdict(int)
        for c in candidates:
            assert c["tier"] in ALL_VALID_TIERS, f"Unknown tier: {c['tier']}"
            assert c["expected_outcome"] in ("accepted", "rejected")
            assert c["concept_a"]
            assert c["concept_b"]
            assert c["connection_claim"]
            assert len(c["source_domains"]) >= 2
            assert 0.0 <= c["endpoint_distance"] <= 1.0
            tiers[c["tier"]] += 1

        # Minimum counts per tier
        assert tiers["sanity_fail"] >= 3, f"Need >= 3 sanity_fail, got {tiers['sanity_fail']}"
        assert tiers["trivial"] >= 10, f"Need >= 10 trivial, got {tiers['trivial']}"
        assert tiers["noise"] >= 10, f"Need >= 10 noise, got {tiers['noise']}"
        assert tiers["noise_borderline"] >= 3, f"Need >= 3 noise_borderline, got {tiers['noise_borderline']}"
        assert tiers["interesting_known"] >= 3, f"Need >= 3 interesting_known, got {tiers['interesting_known']}"
        assert tiers["interesting_novel"] >= 8, f"Need >= 8 interesting_novel, got {tiers['interesting_novel']}"

        # Distance diversity in trivial tier
        trivial = [c for c in candidates if c["tier"] == "trivial"]
        distances = [c["endpoint_distance"] for c in trivial]
        assert max(distances) >= 0.50, \
            f"Trivial tier needs high-distance candidates (max={max(distances):.2f})"
        assert min(distances) <= 0.40, \
            f"Trivial tier needs low-distance candidates (min={min(distances):.2f})"

    @pytest.mark.asyncio
    async def test_sanity_fail_all_rejected_at_stage_0(self, mock_faiss):
        """Sanity fail candidates should all die at text_sanity stage."""
        candidates_data = _load_calibration_candidates()
        store = _build_mock_store()
        model_manager = _build_mock_model_manager()

        pipeline = SynthesisFilter(
            chroma_store=store,
            model_manager=model_manager,
            synthesis_memory=SynthesisMemory(store),
        )

        sanity_fails = [c for c in candidates_data if c["tier"] == "sanity_fail"]
        for entry in sanity_fails:
            candidate = _to_synthesis_candidate(entry)
            result = await pipeline.process_candidate(candidate)
            assert result.status == CandidateStatus.REJECTED, \
                f"Sanity fail candidate should be rejected: {entry['concept_a']} <-> {entry['concept_b']}"
            assert result.rejection_stage == "text_sanity", \
                f"Should die at text_sanity, died at {result.rejection_stage}: " \
                f"{entry['concept_a']} <-> {entry['concept_b']}"

    @pytest.mark.asyncio
    async def test_trivial_rejection_rate(self, mock_faiss):
        """Diagnostic: report trivial tier rejection rate and pathways."""
        candidates_data = _load_calibration_candidates()
        store = _build_mock_store()
        model_manager = _build_mock_model_manager()

        pipeline = SynthesisFilter(
            chroma_store=store,
            model_manager=model_manager,
            synthesis_memory=SynthesisMemory(store),
        )

        trivial = [c for c in candidates_data if c["tier"] == "trivial"]
        rejected_by_stage = defaultdict(int)
        accepted = 0
        for entry in trivial:
            candidate = _to_synthesis_candidate(entry)
            result = await pipeline.process_candidate(candidate)
            if result.status == CandidateStatus.ACCEPTED:
                accepted += 1
            else:
                rejected_by_stage[result.rejection_stage] += 1

        rejection_rate = 1.0 - (accepted / len(trivial)) if trivial else 0.0
        print(f"\nTrivial rejection rate: {rejection_rate:.0%} "
              f"({len(trivial) - accepted}/{len(trivial)})")
        print(f"  By stage: {dict(rejected_by_stage)}")

    @pytest.mark.asyncio
    async def test_noise_rejection_rate(self, mock_faiss):
        """Diagnostic: report noise tier rejection rate."""
        candidates_data = _load_calibration_candidates()
        store = _build_mock_store()
        model_manager = _build_mock_model_manager()

        pipeline = SynthesisFilter(
            chroma_store=store,
            model_manager=model_manager,
            synthesis_memory=SynthesisMemory(store),
        )

        noise = [c for c in candidates_data if c["tier"] in ("noise", "noise_borderline")]
        rejected_by_stage = defaultdict(int)
        accepted = 0
        for entry in noise:
            candidate = _to_synthesis_candidate(entry)
            result = await pipeline.process_candidate(candidate)
            if result.status == CandidateStatus.ACCEPTED:
                accepted += 1
                print(f"  LEAKED: {entry['concept_a']} <-> {entry['concept_b']} "
                      f"(tier={entry['tier']}, coherence={result.coherence_level})")
            else:
                rejected_by_stage[result.rejection_stage] += 1

        rejection_rate = 1.0 - (accepted / len(noise)) if noise else 0.0
        print(f"\nNoise rejection rate: {rejection_rate:.0%} "
              f"({len(noise) - accepted}/{len(noise)})")
        print(f"  By stage: {dict(rejected_by_stage)}")

    @pytest.mark.asyncio
    async def test_novel_acceptance_rate(self, mock_faiss):
        """Diagnostic: report interesting_novel tier acceptance rate."""
        candidates_data = _load_calibration_candidates()
        store = _build_mock_store()
        model_manager = _build_mock_model_manager()

        pipeline = SynthesisFilter(
            chroma_store=store,
            model_manager=model_manager,
            synthesis_memory=SynthesisMemory(store),
        )

        novel = [c for c in candidates_data if c["tier"] == "interesting_novel"]
        rejected = []
        accepted = 0
        for entry in novel:
            candidate = _to_synthesis_candidate(entry)
            result = await pipeline.process_candidate(candidate)
            if result.status == CandidateStatus.ACCEPTED:
                accepted += 1
            else:
                rejected.append((entry, result))

        acceptance_rate = accepted / len(novel) if novel else 0.0
        print(f"\nNovel acceptance rate: {acceptance_rate:.0%} "
              f"({accepted}/{len(novel)})")
        for entry, result in rejected:
            print(f"  MISSED: {entry['concept_a']} <-> {entry['concept_b']} "
                  f"rejected@{result.rejection_stage}, "
                  f"composite={result.composite_score:.3f}")
