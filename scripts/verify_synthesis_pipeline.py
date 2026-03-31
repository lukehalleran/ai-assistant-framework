#!/usr/bin/env python3
"""
End-to-end verification of the synthesis pipeline against real Wikipedia articles.

Usage:
    python scripts/verify_synthesis_pipeline.py
    python scripts/verify_synthesis_pipeline.py --verbose
    python scripts/verify_synthesis_pipeline.py --filter-only  # skip generation, test filter on pre-written claims

Defines pairs of Wikipedia articles with known true/false connections.
For each pair:
  1. Searches FAISS wiki index (40M vectors) to verify articles are findable
  2. Generates a bridge claim via LLM (or uses pre-written claim in --filter-only mode)
  3. Runs the claim through the 8-stage synthesis filter
  4. Checks the outcome matches expectation

Requires: FAISS index on T9 drive, LLM API key configured.
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_utils import get_logger

logger = get_logger("verify_synthesis")


@dataclass
class WikiPair:
    """A pair of Wikipedia articles with a known connection status."""
    article_a: str              # Wikipedia article title / search query
    article_b: str
    has_connection: bool        # True = genuinely connected, False = unrelated
    connection_type: str        # e.g. "biomimicry", "algorithm_origin", "none"
    pre_written_claim: str = "" # Optional: claim to test in --filter-only mode
    notes: str = ""
    # Expected rejection pathway (for nuanced scoring):
    #   True connections are known, so should be rejected by NOVELTY (not coherence)
    #   False connections are fabricated, so should be rejected by COHERENCE (not novelty)
    expected_rejection_via: str = ""  # "novelty", "coherence", "any", or ""
    # If True, an unexpected pass is acceptable (novel framing of known connection)
    # and gets flagged for review rather than counted as failure
    pass_reviewable: bool = False


# Stages that count as "rejected for coherence reasons"
_COHERENCE_STAGES = {"coherence_judge", "text_sanity"}
# Stages that count as "rejected for novelty reasons"
_NOVELTY_STAGES = {"novelty", "composite"}


# ── Known TRUE connections (Wikipedia explicitly documents the link) ──────────

TRUE_PAIRS = [
    WikiPair(
        article_a="Velcro",
        article_b="Burdock",
        has_connection=True,
        connection_type="biomimicry",
        expected_rejection_via="novelty",
        pre_written_claim=(
            "Velcro was invented after George de Mestral observed burdock burrs "
            "attaching to his dog's fur — the hook-and-loop mechanism is a direct "
            "biomimetic transfer of the burr's natural attachment structure."
        ),
        notes="Textbook biomimicry. Wikipedia Velcro article cites burdock directly.",
    ),
    WikiPair(
        article_a="PageRank",
        article_b="Citation analysis",
        has_connection=True,
        connection_type="algorithm_origin",
        expected_rejection_via="novelty",
        pre_written_claim=(
            "PageRank adapted the academic citation graph structure — a web page "
            "linked by many authoritative pages is important, exactly as a paper "
            "cited by many influential papers has high impact."
        ),
        notes="Brin & Page original paper explicitly cites citation analysis.",
    ),
    WikiPair(
        article_a="Simulated annealing",
        article_b="Annealing (metallurgy)",
        has_connection=True,
        connection_type="algorithm_origin",
        expected_rejection_via="novelty",
        pre_written_claim=(
            "Simulated annealing borrows directly from metallurgical annealing: "
            "slowly cooling metal lets atoms escape local energy minima to find "
            "optimal crystal configurations, just as the algorithm accepts worse "
            "solutions with decreasing probability to escape local optima."
        ),
        notes="Named after the physical process. Kirkpatrick 1983.",
    ),
    WikiPair(
        article_a="Kingfisher",
        article_b="Shinkansen",
        has_connection=True,
        connection_type="biomimicry",
        expected_rejection_via="novelty",
        pre_written_claim=(
            "The Shinkansen 500 series nose cone was redesigned based on the "
            "kingfisher's beak to eliminate sonic booms when exiting tunnels — "
            "both shapes minimize pressure wave formation during high-speed "
            "transition between media of different densities."
        ),
        notes="Famous biomimicry case study, documented in both articles.",
    ),
    WikiPair(
        article_a="Ant colony optimization",
        article_b="Foraging",
        has_connection=True,
        connection_type="algorithm_origin",
        expected_rejection_via="novelty",
        pre_written_claim=(
            "Ant colony optimization algorithms directly simulate how real ant "
            "colonies find shortest paths through pheromone trail reinforcement — "
            "successful foraging routes get stronger pheromone signals, biasing "
            "future ants toward them, exactly as the algorithm reinforces "
            "higher-scoring solution paths."
        ),
        notes="ACO is explicitly bio-inspired. Dorigo 1992.",
    ),
    WikiPair(
        article_a="Information theory",
        article_b="Entropy (thermodynamics)",
        has_connection=True,
        connection_type="cross_discipline_foundation",
        expected_rejection_via="novelty",
        pre_written_claim=(
            "Shannon's information entropy was directly inspired by Boltzmann's "
            "thermodynamic entropy — both measure the number of microstates "
            "consistent with a macrostate, and Shannon adopted the term 'entropy' "
            "on von Neumann's advice because the mathematical formulas are identical."
        ),
        notes="Shannon explicitly acknowledged the thermodynamics connection.",
    ),
    WikiPair(
        article_a="Lotka-Volterra equations",
        article_b="Predation",
        has_connection=True,
        connection_type="mathematical_model",
        expected_rejection_via="novelty",
        pre_written_claim=(
            "The Lotka-Volterra equations mathematically model predator-prey "
            "population dynamics — coupled differential equations where predator "
            "population growth depends on prey availability, producing the "
            "characteristic oscillating cycles observed in real ecosystems."
        ),
        notes="Lotka-Volterra is THE predator-prey model.",
    ),
]

# ── Coverage-asymmetry pairs (tests novelty gate calibration) ────────────────
# These probe whether novelty = "not in Wikipedia" vs "not widely known"

COVERAGE_ASYMMETRY_PAIRS = [
    # Famous connection, likely POORLY documented in Wikipedia as a cross-reference.
    # If novelty gate misses this, it's overfitting on corpus coverage.
    WikiPair(
        article_a="Nervous system",
        article_b="Internet",
        has_connection=True,
        connection_type="well_known_analogy",
        expected_rejection_via="novelty",
        pass_reviewable=True,  # Novel framing possible even though analogy is famous
        pre_written_claim=(
            "The internet's packet-switching architecture parallels the nervous "
            "system's action potential propagation — both use threshold-based "
            "all-or-nothing signaling across distributed networks, with routing "
            "decisions made locally at each node rather than centrally."
        ),
        notes="Famous analogy but Wikipedia may not have a dedicated cross-reference article. Tests whether novelty gate catches well-known ideas that have sparse wiki coverage.",
    ),
    WikiPair(
        article_a="Termite mound",
        article_b="Air conditioning",
        has_connection=True,
        connection_type="biomimicry",
        expected_rejection_via="novelty",
        pass_reviewable=True,
        pre_written_claim=(
            "The Eastgate Centre in Harare uses a ventilation system directly "
            "modeled on termite mound thermoregulation — chimney-effect stacks "
            "and thermal mass replace mechanical air conditioning, maintaining "
            "stable interior temperature through passive convection."
        ),
        notes="Real biomimicry project (Mick Pearce, 1996). Well-known in architecture but Wikipedia coverage of the cross-reference may be thin.",
    ),
    # Obscure but deeply connected — tests whether novelty gate lets through
    # genuinely non-obvious connections between articles that happen to both exist
    WikiPair(
        article_a="Stigmergy",
        article_b="Version control",
        has_connection=True,
        connection_type="structural_isomorphism",
        expected_rejection_via="any",  # Could go either way
        pass_reviewable=True,
        pre_written_claim=(
            "Stigmergy — indirect coordination through environmental modification "
            "(ant pheromone trails, termite mud deposits) — is structurally identical "
            "to how version control systems coordinate developers: each commit modifies "
            "the shared artifact (repository), and subsequent actors respond to the "
            "modified state rather than communicating directly."
        ),
        notes="Stigmergy is an obscure ecology term. The VCS parallel is real and mechanistic but unlikely to be in Wikipedia. Tests whether the filter accepts genuinely novel obscure connections.",
    ),
    WikiPair(
        article_a="Quorum sensing",
        article_b="Consensus algorithm",
        has_connection=True,
        connection_type="structural_isomorphism",
        expected_rejection_via="any",
        pass_reviewable=True,
        pre_written_claim=(
            "Bacterial quorum sensing and distributed consensus algorithms solve the "
            "same problem: a population of independent agents must collectively detect "
            "when a threshold of agreement is reached, without any central coordinator. "
            "Both use broadcast signaling (autoinducers / vote messages) and trigger "
            "state transitions only when local signal concentration exceeds a threshold."
        ),
        notes="Both Wikipedia articles exist. The parallel is real and specific but not commonly articulated. Tests mid-range novelty.",
    ),
]

# ── Known FALSE connections (no real link between these topics) ───────────────

FALSE_PAIRS = [
    WikiPair(
        article_a="Velcro",
        article_b="Thermodynamic entropy",
        has_connection=False,
        connection_type="none",
        expected_rejection_via="coherence",
        pre_written_claim=(
            "Velcro's hook-and-loop mechanism demonstrates entropy reduction "
            "in macroscopic systems, where the ordered attachment of hooks to "
            "loops creates a local decrease in entropy that mirrors crystallization "
            "processes in thermodynamic systems."
        ),
        notes="Forced connection. Velcro has nothing to do with entropy.",
    ),
    WikiPair(
        article_a="PageRank",
        article_b="Burdock",
        has_connection=False,
        connection_type="none",
        expected_rejection_via="coherence",
        pre_written_claim=(
            "Burdock's strategy of dispersing seeds via animal attachment parallels "
            "PageRank's link propagation — both systems amplify reach by hitching "
            "to high-traffic carriers, with the most connected nodes (animals / "
            "web pages) providing disproportionate distribution."
        ),
        notes="Superficially clever but mechanistically wrong. Seed dispersal is random, PageRank is structural.",
    ),
    WikiPair(
        article_a="Sourdough",
        article_b="Quantum mechanics",
        has_connection=False,
        connection_type="none",
        expected_rejection_via="coherence",
        pre_written_claim=(
            "Sourdough fermentation exhibits quantum tunneling effects at the "
            "molecular level, where yeast enzymes overcome activation energy "
            "barriers through quantum-mechanical tunneling, explaining why "
            "fermentation proceeds faster than classical chemistry predicts."
        ),
        notes="Quantum biology is real but this specific claim about sourdough is fabricated.",
    ),
    WikiPair(
        article_a="Great Wall of China",
        article_b="DNA replication",
        has_connection=False,
        connection_type="none",
        expected_rejection_via="coherence",
        pre_written_claim=(
            "The Great Wall's construction strategy of building segments "
            "simultaneously from multiple points mirrors DNA replication's "
            "use of multiple origins of replication — both systems solve the "
            "problem of completing a very long linear structure in limited time "
            "by parallelizing the construction process."
        ),
        notes="Sounds structural but 'build in parallel' is so generic it's meaningless.",
    ),
    WikiPair(
        article_a="Jazz",
        article_b="Plate tectonics",
        has_connection=False,
        connection_type="none",
        expected_rejection_via="coherence",
        pre_written_claim=(
            "Jazz improvisation and tectonic plate movement both exhibit "
            "emergent complexity from simple underlying rules — jazz from "
            "chord progressions, tectonics from convection currents — producing "
            "unpredictable surface phenomena from deterministic deep structure."
        ),
        notes="'Emergent complexity from simple rules' applies to literally everything. Vacuous bridge.",
    ),
    WikiPair(
        article_a="Photosynthesis",
        article_b="Stock market",
        has_connection=False,
        connection_type="none",
        expected_rejection_via="coherence",
        pre_written_claim=(
            "Photosynthesis converts light energy into chemical energy through "
            "a series of electron transfers, just as stock markets convert "
            "information signals into price movements through chains of trades — "
            "both are multi-step energy/information transduction cascades."
        ),
        notes="Extremely forced. 'Multi-step cascade' is too generic to be meaningful.",
    ),
    WikiPair(
        article_a="Knitting",
        article_b="General relativity",
        has_connection=False,
        connection_type="none",
        expected_rejection_via="coherence",
        pre_written_claim=(
            "Knitting creates curved fabric from straight yarn by varying "
            "stitch tension, analogous to how mass curves spacetime in general "
            "relativity — both demonstrate how local constraints on a flexible "
            "medium produce global curvature."
        ),
        notes="Hyperbolic crochet IS used to visualize hyperbolic geometry, but knitting/GR is a stretch. Tests borderline.",
    ),
]

# ── Novelty-resistant FALSE pairs (designed to reach coherence judge) ──────────
# These use obscure topics and novel claim phrasing so FAISS novelty gate
# won't catch them. They MUST reach Stage 5 (coherence) to be rejected.

NOVELTY_RESISTANT_FALSE_PAIRS = [
    WikiPair(
        article_a="Pysanka",  # Ukrainian egg decoration
        article_b="Erlang (programming language)",
        has_connection=False,
        connection_type="none",
        expected_rejection_via="coherence",
        pre_written_claim=(
            "The traditional pysanka egg decoration process, where wax resist is "
            "applied in sequential layers to control dye absorption, mirrors "
            "Erlang's process isolation model — both systems achieve complex "
            "outcomes through strict sequential state transitions where each "
            "stage's output irreversibly constrains subsequent stages."
        ),
        notes="Sounds structural but the mechanism is fabricated. Wax resist is subtractive; Erlang processes are concurrent, not sequential.",
    ),
    WikiPair(
        article_a="Bagpipe",
        article_b="Tidal locking",
        has_connection=False,
        connection_type="none",
        expected_rejection_via="coherence",
        pre_written_claim=(
            "The constant-pressure reservoir of a bagpipe bag, which decouples "
            "the player's intermittent breath from the continuous drone output, "
            "structurally mirrors tidal locking in celestial mechanics — both "
            "systems achieve temporal smoothing by interposing an inertial buffer "
            "between an irregular driving force and a steady-state output."
        ),
        notes="'Inertial buffer between irregular input and steady output' is superficially structural but the physics is completely different. Tidal locking is gravitational torque equilibrium, not buffering.",
    ),
    WikiPair(
        article_a="Campanology",  # bell-ringing
        article_b="Splay tree",
        has_connection=False,
        connection_type="none",
        expected_rejection_via="coherence",
        pre_written_claim=(
            "Change ringing in campanology systematically permutes bell sequences "
            "to visit every possible ordering exactly once, analogous to how splay "
            "trees restructure themselves after each access to move frequently "
            "accessed nodes to the root — both exploit sequential permutation to "
            "optimize access patterns over time."
        ),
        notes="Change ringing IS about permutations, but splay trees do rotations for amortized access, not sequential permutation. The shared 'permutation' label masks completely different mechanisms.",
    ),
    WikiPair(
        article_a="Kintsugi",  # Japanese gold repair
        article_b="Reed-Solomon error correction",
        has_connection=False,
        connection_type="none",
        expected_rejection_via="coherence",
        pre_written_claim=(
            "Kintsugi repairs broken pottery by filling cracks with gold lacquer, "
            "making the repair visible and valued rather than hidden. Reed-Solomon "
            "codes similarly add redundant symbols that make data corruption "
            "detectable and correctable — both systems transform damage into "
            "explicit structural information rather than concealing it."
        ),
        notes="Kintsugi is aesthetic philosophy about embracing imperfection. Reed-Solomon is mathematical redundancy for error correction. 'Making damage visible' is metaphor, not mechanism.",
    ),
]


async def verify_article_exists(store, article: str) -> Optional[str]:
    """Search FAISS wiki index for an article. Returns best match content or None."""
    from knowledge.semantic_search import semantic_search_with_neighbors
    results = semantic_search_with_neighbors(article, k=1)
    if results:
        title = results[0].get("title", "")
        score = results[0].get("similarity", 0)
        content = results[0].get("content", results[0].get("text", ""))[:200]
        return f"{title} (sim={score:.3f}): {content}..."
    return None


async def run_filter_on_claim(
    filter_pipeline, store, concept_a: str, concept_b: str,
    claim: str, source_domains: set, endpoint_distance: float
):
    """Run a single claim through the synthesis filter."""
    from knowledge.synthesis_models import SynthesisCandidate
    from datetime import datetime

    candidate = SynthesisCandidate(
        concept_a=concept_a,
        concept_b=concept_b,
        connection_claim=claim,
        walk_path=[concept_a.lower(), concept_b.lower()],
        source_domains=source_domains,
        endpoint_distance=endpoint_distance,
        timestamp=datetime.now(),
    )

    result = await filter_pipeline.process_candidate(candidate)
    return result


async def run_verification(verbose: bool = False, filter_only: bool = False):
    """Run the full verification suite."""
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    print("=" * 70)
    print("SYNTHESIS PIPELINE VERIFICATION")
    print("=" * 70)

    # Init store (ChromaDB for synthesis_results; FAISS for wiki)
    print("\nInitializing ChromaDB + FAISS...")
    store = MultiCollectionChromaStore()

    # Verify FAISS wiki index is available
    from knowledge.semantic_search import get_index
    faiss_idx = get_index()
    faiss_idx.load()
    if not faiss_idx.loaded:
        print("ERROR: FAISS wiki index not available. Check T9 drive is mounted.")
        sys.exit(1)
    print(f"  FAISS wiki index: {faiss_idx._total_rows:,} vectors")

    # Check article availability
    print("\nChecking article availability...")
    all_pairs = TRUE_PAIRS + FALSE_PAIRS + NOVELTY_RESISTANT_FALSE_PAIRS + COVERAGE_ASYMMETRY_PAIRS
    missing = []
    for pair in all_pairs:
        for article in [pair.article_a, pair.article_b]:
            hit = await verify_article_exists(store, article)
            if hit:
                if verbose:
                    print(f"  FOUND: {article} → {hit[:80]}")
            else:
                missing.append(article)
                print(f"  MISSING: {article}")

    if missing:
        print(f"\nWARNING: {len(missing)} articles not found in wiki_knowledge.")
        print("Results may be less meaningful for pairs with missing articles.")

    # Init filter
    print("\nInitializing synthesis filter...")
    from knowledge.synthesis_filter import SynthesisFilter
    from models.model_manager import ModelManager

    model_manager = ModelManager()
    filter_pipeline = SynthesisFilter(
        chroma_store=store,
        model_manager=model_manager,
    )

    # Init generator (only needed if not filter_only)
    generator = None
    if not filter_only:
        from knowledge.synthesis_generator import SynthesisGenerator
        generator = SynthesisGenerator(
            chroma_store=store,
            model_manager=model_manager,
        )

    # Run verification
    print(f"\n{'='*70}")
    print(f"RUNNING {'FILTER-ONLY' if filter_only else 'FULL PIPELINE'} VERIFICATION")
    print(f"  True connections:        {len(TRUE_PAIRS)}")
    print(f"  False connections:       {len(FALSE_PAIRS)} + {len(NOVELTY_RESISTANT_FALSE_PAIRS)} novelty-resistant")
    print(f"  Coverage asymmetry:      {len(COVERAGE_ASYMMETRY_PAIRS)}")
    print(f"{'='*70}\n")

    results = {
        "rejected_correctly": 0,        # Both outcome and pathway correct
        "rejected_wrong_pathway": 0,    # Rejected (good) but for wrong reason
        "accepted_wrongly": 0,          # Should have been rejected (false pair passed)
        "accepted_review": 0,           # Unexpected pass, flagged for quality review
        "generation_blocked": 0,        # LLM returned NO_CONNECTION
        "errors": 0,
        "details": [],
    }

    for pair in all_pairs:
        label = "TRUE" if pair.has_connection else "FALSE"
        expected_via = pair.expected_rejection_via or "any"
        print(f"[{label:5s}] {pair.article_a} ↔ {pair.article_b} ({pair.connection_type})")
        print(f"        expect: rejected via {expected_via}")

        try:
            # Get or generate claim
            if filter_only or not generator:
                claim = pair.pre_written_claim
                claim_source = "pre-written"
            else:
                # Search wiki for context
                ctx_a = await verify_article_exists(store, pair.article_a) or pair.article_a
                ctx_b = await verify_article_exists(store, pair.article_b) or pair.article_b
                claim = await generator._articulate_bridge(
                    pair.article_a, ctx_a[:300],
                    pair.article_b, ctx_b[:300],
                    "knowledge", "knowledge",
                )
                claim_source = "generated"

                if not claim:
                    print(f"  → LLM returned NO_CONNECTION")
                    if not pair.has_connection:
                        results["rejected_correctly"] += 1
                        print(f"  ✓ Correctly blocked at generation (false pair)")
                    else:
                        results["generation_blocked"] += 1
                        print(f"  ~ True pair but LLM found no bridge")
                    results["details"].append({
                        "pair": f"{pair.article_a} ↔ {pair.article_b}",
                        "has_connection": pair.has_connection,
                        "expected_via": expected_via,
                        "outcome": "no_claim_generated",
                        "claim_source": claim_source,
                        "pathway_correct": not pair.has_connection,
                    })
                    print()
                    continue

            if verbose:
                print(f"  Claim ({claim_source}): {claim[:120]}...")

            # Run through filter
            t0 = time.time()
            # Use connection_type to infer plausible domain pair.
            # These are wiki-vs-wiki so we assign two distinct domains
            # to get past the domain crossing gate (Stage 1).
            _DOMAIN_MAP = {
                "biomimicry": {"biology", "engineering"},
                "algorithm_origin": {"mathematics", "computer_science"},
                "cross_discipline_foundation": {"physics", "information_science"},
                "mathematical_model": {"mathematics", "ecology"},
                "well_known_analogy": {"biology", "technology"},
                "structural_isomorphism": {"biology", "computer_science"},
                "none": {"science", "humanities"},  # false pairs get generic distinct domains
            }
            domains = _DOMAIN_MAP.get(pair.connection_type, {"domain_a", "domain_b"})

            result = await run_filter_on_claim(
                filter_pipeline, store,
                pair.article_a, pair.article_b,
                claim,
                source_domains=domains,
                endpoint_distance=0.55,  # mid-range default
            )
            elapsed = (time.time() - t0) * 1000

            accepted = result.status.value == "accepted"
            stage = result.rejection_stage or "passed"
            score = result.composite_score

            # Determine pathway correctness
            pathway_correct = False
            if accepted:
                pathway_correct = False  # Will be reclassified below
            elif expected_via == "novelty":
                pathway_correct = stage in _NOVELTY_STAGES
            elif expected_via == "coherence":
                pathway_correct = stage in _COHERENCE_STAGES
            elif expected_via == "any":
                pathway_correct = True  # Any rejection is fine

            # Confidence margin: how far the composite score is from the
            # acceptance threshold (default 0.60). Positive = above, negative = below.
            acceptance_threshold = 0.65
            confidence_margin = round(score - acceptance_threshold, 3)

            detail = {
                "pair": f"{pair.article_a} ↔ {pair.article_b}",
                "has_connection": pair.has_connection,
                "expected_via": expected_via,
                "outcome": "accepted" if accepted else f"rejected@{stage}",
                "pathway_correct": pathway_correct,
                "composite_score": round(score, 3),
                "confidence_margin": confidence_margin,
                "coherence": result.coherence_level.name if result.coherence_level else "N/A",
                "novelty_ext": round(result.novelty_score_external, 3),
                "cooccurrence": round(getattr(result, 'cooccurrence_similarity', 0), 3),
                "claim": claim[:200] if claim else "",
                "claim_source": claim_source,
                "elapsed_ms": round(elapsed, 1),
                # Full stage chain for diagnostics
                "stage_chain": [
                    {
                        "stage": sr.stage_name,
                        "passed": sr.passed,
                        "score": round(sr.score, 4) if sr.score is not None else None,
                        "reason": sr.reason[:200] if sr.reason else None,
                        "metadata": sr.metadata,
                    }
                    for sr in result.stage_results
                ],
                "coherence_justification": result.coherence_justification[:400] if result.coherence_justification else None,
                # Quality fields — filled manually after review for accepted candidates
                "quality_originality": None,       # 1-5, manual
                "quality_explanatory_power": None,  # 1-5, manual
                "quality_non_triviality": None,     # 1-5, manual
            }

            # Determine if this is an unexpected outcome worth full diagnostic
            unexpected = False

            if accepted:
                if pair.pass_reviewable:
                    # Unexpected pass on a reviewable pair — flag, don't fail
                    results["accepted_review"] += 1
                    detail["pathway_correct"] = "review"
                    print(f"  ? ACCEPTED (score={score:.3f}, margin=+{confidence_margin:.3f}) — flagged for quality review")
                    print(f"    → Rate this pass: originality? explanatory power? non-triviality? (1-5 each)")
                elif not pair.has_connection:
                    # False pair accepted — real failure
                    results["accepted_wrongly"] += 1
                    print(f"  ✗ ACCEPTED (score={score:.3f}) — FALSE pair should have been rejected")
                    unexpected = True
                else:
                    # True-known pair accepted without pass_reviewable — unexpected
                    results["accepted_review"] += 1
                    detail["pathway_correct"] = "review"
                    print(f"  ? ACCEPTED (score={score:.3f}, margin=+{confidence_margin:.3f}) — unexpected pass, review framing novelty")
                    unexpected = True
            elif pathway_correct:
                results["rejected_correctly"] += 1
                print(f"  ✓ REJECTED@{stage} (score={score:.3f}, margin={confidence_margin:+.3f}, {elapsed:.0f}ms)")
            else:
                results["rejected_wrong_pathway"] += 1
                print(f"  ~ REJECTED@{stage} (score={score:.3f}, margin={confidence_margin:+.3f}) — expected via {expected_via}")
                unexpected = True

            if verbose and result.rejection_reason:
                print(f"  Reason: {result.rejection_reason[:100]}")

            # Full diagnostic dump for unexpected outcomes
            if unexpected:
                print(f"  --- DIAGNOSTIC DUMP ---")
                for sr in result.stage_results:
                    status = "PASS" if sr.passed else "FAIL"
                    score_str = f"{sr.score:.3f}" if sr.score is not None else "N/A"
                    print(f"    {sr.stage_name}: {status} (score={score_str}, {sr.elapsed_ms:.0f}ms)")
                    if sr.metadata:
                        print(f"      {sr.metadata}")
                    if sr.reason:
                        print(f"      reason: {sr.reason[:150]}")
                if result.coherence_justification:
                    print(f"    coherence_justification: {result.coherence_justification[:300]}")
                print(f"  --- END DUMP ---")

            results["details"].append(detail)

        except Exception as e:
            results["errors"] += 1
            print(f"  ERROR: {e}")
            results["details"].append({
                "pair": f"{pair.article_a} ↔ {pair.article_b}",
                "error": str(e),
            })

        print()

    # ── Summary ──────────────────────────────────────────────────────────
    correct = results["rejected_correctly"]
    wrong_path = results["rejected_wrong_pathway"]
    wrong_accept = results["accepted_wrongly"]
    review = results["accepted_review"]
    gen_blocked = results["generation_blocked"]
    total_evaluated = correct + wrong_path + wrong_accept + review + gen_blocked

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Rejected correctly (right pathway): {correct}")
    print(f"  Rejected wrong pathway:             {wrong_path}")
    print(f"  Accepted wrongly (false pair):      {wrong_accept}")
    print(f"  Accepted for review (check quality):{review}")
    print(f"  Generation blocked (NO_CONNECTION):  {gen_blocked}")
    print(f"  Errors:                              {results['errors']}")
    print(f"  Total:                               {total_evaluated}")

    if total_evaluated > 0:
        hard_failures = wrong_accept  # Only false-pair accepts are real failures
        rejections = correct + wrong_path
        rejection_rate = rejections / (rejections + wrong_accept + review) if (rejections + wrong_accept + review) > 0 else 0
        pathway_accuracy = correct / rejections if rejections > 0 else 0
        # Overall: correct rejections + legitimate reviews / total
        overall = (correct + review) / total_evaluated

        print(f"\n  Rejection rate:      {rejection_rate:.1%}  (excluding reviews)")
        print(f"  Pathway accuracy:    {pathway_accuracy:.1%}  (of rejections, % via correct gate)")
        print(f"  Hard failures:       {hard_failures}  (false pairs that passed — this should be 0)")

    # ── Confidence margin analysis ───────────────────────────────────
    margins = [d.get("confidence_margin", 0) for d in results["details"] if "confidence_margin" in d]
    if margins:
        tight = [d for d in results["details"] if "confidence_margin" in d and abs(d["confidence_margin"]) < 0.10]
        print(f"\n  Confidence margins:")
        print(f"    Tightest decisions (|margin| < 0.10): {len(tight)}")
        if tight:
            for d in tight:
                print(f"      {d['pair']}: {d['outcome']} (margin={d['confidence_margin']:+.3f})")

    # ── Breakdown by pair type ───────────────────────────────────────
    true_details = [d for d in results["details"] if d.get("has_connection")]
    false_details = [d for d in results["details"] if not d.get("has_connection")]

    if true_details:
        true_correct = sum(1 for d in true_details if d.get("pathway_correct") is True)
        true_review = sum(1 for d in true_details if d.get("pathway_correct") == "review")
        print(f"\n  True connections ({len(true_details)} pairs):")
        print(f"    Rejected via novelty (correct): {true_correct}")
        print(f"    Passed — flagged for review:    {true_review}")
        print(f"    Rejected via wrong gate:        {len(true_details) - true_correct - true_review}")
        for d in true_details:
            pc = d.get("pathway_correct")
            if pc is True:
                marker = "✓"
            elif pc == "review":
                marker = "?"
            else:
                marker = "✗"
            margin = d.get("confidence_margin", 0)
            print(f"      {marker} {d['pair']}: {d.get('outcome', '?')} (margin={margin:+.3f})")

    if false_details:
        false_correct = sum(1 for d in false_details if d.get("pathway_correct") is True)
        false_passed = sum(1 for d in false_details if "accepted" in d.get("outcome", ""))
        print(f"\n  False connections ({len(false_details)} pairs):")
        print(f"    Rejected via coherence (correct): {false_correct}")
        print(f"    Wrongly accepted (FAILURE):       {false_passed}")
        print(f"    Rejected via wrong gate:          {len(false_details) - false_correct - false_passed}")
        for d in false_details:
            pc = d.get("pathway_correct")
            if pc is True:
                marker = "✓"
            elif "accepted" in d.get("outcome", ""):
                marker = "✗"
            else:
                marker = "~"
            margin = d.get("confidence_margin", 0)
            print(f"      {marker} {d['pair']}: {d.get('outcome', '?')} (margin={margin:+.3f})")

    # ── Review queue (unexpected passes needing quality assessment) ───
    reviews = [d for d in results["details"] if d.get("pathway_correct") == "review"]
    if reviews:
        print(f"\n  {'='*60}")
        print(f"  QUALITY REVIEW QUEUE ({len(reviews)} candidates)")
        print(f"  {'='*60}")
        print(f"  Rate each 1-5 on: originality, explanatory power, non-triviality")
        print(f"  Then update quality_* fields in {Path('data/synthesis_verification_results.json')}")
        for d in reviews:
            print(f"\n    {d['pair']}")
            print(f"    Score: {d.get('composite_score', '?')}, Coherence: {d.get('coherence', '?')}")
            print(f"    Claim: {d.get('claim', '?')[:150]}...")
            print(f"    originality: ___  explanatory_power: ___  non_triviality: ___")

    # Write detailed results
    output_path = Path("data/synthesis_verification_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Detailed results: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify synthesis pipeline against known Wikipedia connections"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show claims and rejection reasons",
    )
    parser.add_argument(
        "--filter-only", action="store_true",
        help="Skip LLM generation, use pre-written claims only (faster, no LLM cost)",
    )
    args = parser.parse_args()

    asyncio.run(run_verification(verbose=args.verbose, filter_only=args.filter_only))


if __name__ == "__main__":
    main()
