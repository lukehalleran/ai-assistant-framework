#!/usr/bin/env python3
"""
Synthesis validation harness — prove the filter without being a domain expert.

Implements the methodology in docs/SYNTHESIS_VALIDATION.md. The core idea: literature
is the ground truth, so we test the coherence JUDGE's ability to separate real
connections from spurious ones using sets we already know the answer to — no human
grading required.

Test A (built here): judge discrimination.
  - POSITIVES: connections known to be real (in literature) → judge should PASS them.
  - NEGATIVES: known-bad surface metaphors + shuffle-mismatched claims → should FAIL.
  - Metric: the GAP between positive and negative pass-rates (signal-detection d').
    Big gap = the judge discriminates and the filter is trustworthy. Small gap =
    the judge accepts everything and the validation is meaningless.

Modes:
  --mode discrimination   Curated positives vs curated/shuffled negatives (default; runs now).
  --mode live             Pull live positives from get_known_connections() (needs dreaming runs).
  --mode all              Both.

Hooks (documented, not yet runnable):
  reindex_delta()  — Test C from the doc: re-check the novel pile against arXiv once it's
                     ingested; the shrinkage = cross-corpus rediscovery. See the function.

Usage:
  python scripts/synthesis_validation.py --mode discrimination
  python scripts/synthesis_validation.py --mode all --model claude-opus-4.8

Makes real LLM calls (the coherence judge). Costs a handful of cents per run.
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_manager import ModelManager
from knowledge.synthesis_filter import SynthesisFilter
from knowledge.synthesis_models import SynthesisCandidate, SynthesisResult, CoherenceLevel
from memory.synthesis_memory import SynthesisMemory
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

# Pass = the filter's accept gate (coherence_level >= MODERATE, per SYNTHESIS_COHERENCE_MIN_LEVEL).
_PASS_LEVELS = {CoherenceLevel.MODERATE, CoherenceLevel.STRONG}

# ── Curated benchmark — stable ground truth that doesn't depend on the generators ──
# POSITIVES: textbook-known structural cross-domain connections. Literature confirms
# these are REAL (not surface analogy). A working judge should PASS (>= MODERATE) them.
POSITIVES = [
    ("natural selection", "gradient descent",
     "Both optimize by iterative local improvement on a fitness/loss surface: each step "
     "moves toward higher fitness / lower loss using only local information, with no global "
     "view of the landscape."),
    ("predator-prey dynamics", "LC oscillator circuit",
     "Both are governed by coupled first-order differential equations producing self-sustaining "
     "limit-cycle oscillations — energy (or population) shuttles between two reservoirs with a "
     "90-degree phase lag (Lotka-Volterra / harmonic oscillator)."),
    ("metallurgical annealing", "simulated annealing",
     "Both escape local minima by injecting controlled randomness that decreases on a cooling "
     "schedule: high 'temperature' permits uphill moves early, freezing into a low-energy "
     "configuration as temperature drops."),
    ("thermodynamic entropy", "Shannon entropy",
     "Both quantify disorder/uncertainty with the identical functional form -sum p log p; "
     "Boltzmann's H-theorem and Shannon's source coding are the same mathematics in different "
     "units."),
    ("epidemic spread (SIR)", "viral information diffusion",
     "Both follow compartmental contagion dynamics: a susceptible population converts to "
     "'infected' at a rate proportional to contact with already-infected members, yielding the "
     "same logistic adoption curve and basic-reproduction-number threshold."),
    ("physical phase transition", "network percolation",
     "Both exhibit critical-point behavior: below a threshold the system is fragmented, at the "
     "critical value a giant connected component / ordered phase emerges suddenly, with "
     "power-law scaling near the transition."),
    ("Bayesian updating", "Kalman filtering",
     "The Kalman filter IS recursive Bayesian estimation for linear-Gaussian systems: the "
     "predict/update cycle is exactly prior -> likelihood -> posterior applied each timestep."),
    ("immune system memory", "cache eviction",
     "Both maintain a bounded store of past encounters and decide what to retain vs discard "
     "based on recency and expected future utility, trading capacity against hit-rate on "
     "re-exposure."),
]

# NEGATIVES: known-BAD claims — surface metaphors and pseudo-connections that SOUND
# structural but aren't. A working judge should FAIL (< MODERATE) these.
NEGATIVES = [
    ("atomic structure", "solar system",
     "Electrons orbit the nucleus just like planets orbit the sun, so atoms and solar systems "
     "share the same gravitational dynamics."),  # different force law; surface analogy
    ("the brain", "a computer",
     "The brain is like a computer because both process information, so neuroscience and "
     "computer architecture share the same memory hierarchy and clock cycle."),
    ("quantum entanglement", "human emotional connection",
     "Entangled particles and emotionally bonded people both stay instantaneously correlated "
     "across any distance, so love is a quantum phenomenon."),
    ("DNA", "computer code",
     "DNA is just code because both are written in a discrete alphabet, so genetics and software "
     "engineering share version control and debugging workflows."),
    ("a flowing river", "electric current",
     "Both are called 'current' and both flow, so hydrology and electrical engineering are "
     "fundamentally the same field."),
    ("fractals", "the stock market",
     "Both look jagged and self-similar, so fractal geometry proves the market is predictable "
     "from its own shape."),
]


def _to_candidate(concept_a, concept_b, claim):
    return SynthesisCandidate(
        concept_a=concept_a, concept_b=concept_b, connection_claim=claim,
        walk_path=[concept_a.lower(), concept_b.lower()],
        source_domains={"validation_a", "validation_b"},
        endpoint_distance=0.5, timestamp=datetime.now(),
    )


def make_shuffle_negatives(positives):
    """Guaranteed-bad control: keep each real claim but relabel concept_b to a DIFFERENT
    positive's concept_b. The claim can no longer fit the mismatched pair, so a working
    judge MUST reject it. A pure floor (vs the realistic floor of curated NEGATIVES)."""
    out = []
    n = len(positives)
    for i, (a, _b, claim) in enumerate(positives):
        wrong_b = positives[(i + 1) % n][1]  # someone else's concept_b
        out.append((a, wrong_b, claim))
    return out


async def judge_one(pipeline, concept_a, concept_b, claim):
    """Run a single candidate through ONLY the coherence judge (bypass the novelty gates),
    so we test the judge's real-vs-spurious discrimination in isolation."""
    result = SynthesisResult(candidate=_to_candidate(concept_a, concept_b, claim))
    await pipeline._stage_5_coherence_judge(result)
    level = result.coherence_level
    return level, (level in _PASS_LEVELS)


async def run_set(pipeline, label, items):
    passed = 0
    rows = []
    for a, b, claim in items:
        level, ok = await judge_one(pipeline, a, b, claim)
        passed += int(ok)
        rows.append((a, b, level.name if level else "N/A", ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {level.name if level else 'N/A':9} {a} <-> {b}")
    rate = passed / max(len(items), 1)
    print(f"  -> {label}: {passed}/{len(items)} passed ({rate:.0%})\n")
    return rate, rows


async def run_discrimination(pipeline):
    print("=" * 78)
    print("TEST A — Judge discrimination (literature ground truth, no human grading)")
    print("=" * 78)
    print("\nPOSITIVES (known-real — should mostly PASS):")
    pos_rate, _ = await run_set(pipeline, "positives", POSITIVES)
    print("NEGATIVES (surface metaphor / pseudo — should mostly FAIL):")
    neg_rate, _ = await run_set(pipeline, "curated negatives", NEGATIVES)
    print("SHUFFLE-NEGATIVES (real claim on mismatched pair — guaranteed FAIL):")
    shuf_rate, _ = await run_set(pipeline, "shuffle negatives", make_shuffle_negatives(POSITIVES))

    worst_neg = max(neg_rate, shuf_rate)
    gap = pos_rate - worst_neg
    print("=" * 78)
    print(f"DISCRIMINATION GAP = {pos_rate:.0%} (positives) - {worst_neg:.0%} (worst negatives) "
          f"= {gap:+.0%}")
    if gap >= 0.50:
        verdict = "STRONG — the judge discriminates; the filter is trustworthy."
    elif gap >= 0.25:
        verdict = "MODERATE — some signal, but tune the judge / thresholds."
    else:
        verdict = "WEAK — the judge is NOT discriminating; do not trust accept rates yet."
    print(f"VERDICT: {verdict}")
    print("=" * 78)
    return gap


async def run_live_positives(pipeline, sm, limit=40):
    """Test A with LIVE positives: the generator's own rediscoveries (rejected_known) —
    connections it proposed that turned out to already be in literature. Populates once
    dreaming runs accumulate them (get_known_connections)."""
    known = sm.get_known_connections(limit=limit)
    if not known:
        print("\n[live] No known-connection rediscoveries captured yet. Run a few dreaming "
              "sessions (generators now persist + don't truncate) and re-run.\n")
        return None
    items = [(r.candidate.concept_a, r.candidate.concept_b, r.candidate.connection_claim)
             for _id, r in known]
    print(f"\nLIVE POSITIVES — {len(items)} generator rediscoveries (should mostly PASS):")
    rate, _ = await run_set(pipeline, "live positives", items)
    return rate


def reindex_delta():
    """TEST C (hook — needs arXiv ingested first). See docs/SYNTHESIS_VALIDATION.md.

    Plan once an arXiv FAISS index exists (same embedder as wiki: all-MiniLM-L6-v2):
      1. Load the current NOVEL pile: synthesis_results with status=accepted that were
         unknown-in-wiki (passed novelty_external).
      2. Re-run the novelty/known check against the arXiv index for each.
      3. Report: how many flip unknown -> known (arXiv-confirmed). That fraction is
         cross-corpus rediscovery — real connections wiki didn't contain. The remainder
         is a cleaner novel pile.
    Not runnable until arXiv is embedded (GOALS Goal 5)."""
    print("[reindex-delta] Not yet runnable — arXiv index not ingested. "
          "See docs/SYNTHESIS_VALIDATION.md Test C.")


async def main():
    ap = argparse.ArgumentParser(description="Synthesis validation harness")
    ap.add_argument("--mode", choices=["discrimination", "live", "all"], default="discrimination")
    ap.add_argument("--model", default=None, help="Override SYNTHESIS_COHERENCE_MODEL")
    args = ap.parse_args()

    from config.app_config import CHROMA_PATH
    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    mm = ModelManager()
    sm = SynthesisMemory(store)
    pipeline = SynthesisFilter(chroma_store=store, model_manager=mm, synthesis_memory=sm)

    if args.model:
        # The judge reads SYNTHESIS_COHERENCE_MODEL bound at synthesis_filter import time, so
        # mutating config.app_config alone is a silent no-op — rebind the filter module's name.
        # (Also set cfg for the articulation path, which late-imports from config.app_config.)
        import config.app_config as cfg
        import knowledge.synthesis_filter as sf
        cfg.SYNTHESIS_COHERENCE_MODEL = args.model
        sf.SYNTHESIS_COHERENCE_MODEL = args.model
        print(f"[config] coherence model overridden -> {args.model}")

    if args.mode in ("discrimination", "all"):
        await run_discrimination(pipeline)
    if args.mode in ("live", "all"):
        await run_live_positives(pipeline, sm)


if __name__ == "__main__":
    asyncio.run(main())
