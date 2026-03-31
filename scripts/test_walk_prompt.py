#!/usr/bin/env python3
"""
Walk articulation prompt validation (Phase 2.5A).

Tests whether narrating a multi-hop knowledge path produces better
coherence scores than the existing random-pair bridge prompt.

For each test path:
  1. Walk prompt: LLM narrates the full path (A → X → Y → B)
  2. Random-pair prompt: LLM bridges just the endpoints (A ↔ B)
  3. Both claims are scored by the coherence judge (Stage 5)

If walk narration consistently scores MODERATE+ while random pairing
scores WEAK, the walk architecture is validated.

Usage:
    python scripts/test_walk_prompt.py
    python scripts/test_walk_prompt.py --model sonnet-4.5
"""

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_utils import get_logger

logger = get_logger("test_walk_prompt")

# Known multi-hop concept paths for testing.
# Each path: personal_concept → intermediate_1 → ... → personal_concept
# Intermediates are general knowledge that Wikidata would provide.
TEST_PATHS = [
    {
        "path": ["brewing", "fermentation", "biochemistry", "serotonin", "exercise"],
        "domain_a": "hobbies",
        "domain_b": "fitness",
        "description": "Brewing → Exercise via fermentation biochemistry",
    },
    {
        "path": ["statistics", "bayesian inference", "decision theory", "game theory", "board games"],
        "domain_a": "education",
        "domain_b": "hobbies",
        "description": "Statistics → Board games via decision theory",
    },
    {
        "path": ["ADHD", "dopamine", "reward circuitry", "variable ratio reinforcement", "video games"],
        "domain_a": "health",
        "domain_b": "hobbies",
        "description": "ADHD → Video games via dopamine reward circuitry",
    },
    {
        "path": ["weight lifting", "progressive overload", "stress adaptation", "hormesis", "vaccination"],
        "domain_a": "fitness",
        "domain_b": "health",
        "description": "Weight lifting → Vaccination via hormesis",
    },
    {
        "path": ["sourdough", "microbiome", "gut-brain axis", "cognitive performance", "studying"],
        "domain_a": "hobbies",
        "domain_b": "education",
        "description": "Sourdough → Studying via gut-brain axis",
    },
    {
        "path": ["rock climbing", "risk assessment", "prospect theory", "loss aversion", "investing"],
        "domain_a": "hobbies",
        "domain_b": "career",
        "description": "Rock climbing → Investing via prospect theory",
    },
    {
        "path": ["running", "VO2 max", "mitochondrial biogenesis", "aging", "telomeres"],
        "domain_a": "fitness",
        "domain_b": "science",
        "description": "Running → Telomeres via mitochondrial biogenesis",
    },
    {
        "path": ["poetry", "meter", "periodicity", "circadian rhythm", "sleep hygiene"],
        "domain_a": "arts",
        "domain_b": "health",
        "description": "Poetry → Sleep via periodicity",
    },
]

# Walk narration prompt — gives the LLM a real path to interpret
WALK_PROMPT = """You are given two personal concepts connected through a chain of general knowledge.

Concept A ({domain_a}): {concept_a}
Concept B ({domain_b}): {concept_b}
Knowledge path: {path_str}

Each arrow represents a real relationship in a knowledge graph.

Describe the specific structural connection this path reveals between Concept A \
and Concept B. Focus on the shared mechanism, feedback loop, or structural \
isomorphism — not surface-level analogy. If the path doesn't reveal a meaningful \
connection, respond with NO_CONNECTION.

Connection:"""

# Existing random-pair prompt (from synthesis_generator.py)
RANDOM_PAIR_PROMPT = """Consider these two concepts from different life domains.

Concept A ({domain_a}): {concept_a}

Concept B ({domain_b}): {concept_b}

If there is a specific, non-obvious connection between them, describe it in one or two sentences. The connection should be:
- Concrete (not "both are important" or "both involve systems")
- Mechanistic (describes HOW they relate, not just THAT they relate)
- Falsifiable (someone could argue against it)

If no meaningful connection exists, respond with exactly: NO_CONNECTION

Connection:"""


async def run_validation(model_override: str | None = None):
    from config.app_config import SYNTHESIS_COHERENCE_MODEL
    from models.model_manager import ModelManager

    model_manager = ModelManager()
    bridge_model = model_override or SYNTHESIS_COHERENCE_MODEL

    print("=" * 70)
    print("WALK PROMPT vs RANDOM-PAIR PROMPT VALIDATION")
    print(f"Bridge model: {bridge_model}")
    print(f"Test paths: {len(TEST_PATHS)}")
    print("=" * 70)

    results = []

    for i, test in enumerate(TEST_PATHS):
        path = test["path"]
        concept_a = path[0]
        concept_b = path[-1]
        intermediates = path[1:-1]
        path_str = " → ".join(path)
        domain_a = test["domain_a"]
        domain_b = test["domain_b"]

        print(f"\n--- [{i+1}/{len(TEST_PATHS)}] {test['description']} ---")
        print(f"    Path: {path_str}")

        # 1. Walk prompt — narrate the full path
        walk_prompt = WALK_PROMPT.format(
            concept_a=concept_a,
            concept_b=concept_b,
            domain_a=domain_a,
            domain_b=domain_b,
            path_str=path_str,
        )

        walk_claim = await model_manager.generate_once(
            prompt=walk_prompt,
            model_name=bridge_model,
            system_prompt="You identify structural connections between concepts.",
            max_tokens=200,
            temperature=0.3,
        )
        walk_claim = walk_claim.strip()

        # 2. Random-pair prompt — just endpoints
        pair_prompt = RANDOM_PAIR_PROMPT.format(
            concept_a=concept_a,
            concept_b=concept_b,
            domain_a=domain_a,
            domain_b=domain_b,
        )

        pair_claim = await model_manager.generate_once(
            prompt=pair_prompt,
            model_name=bridge_model,
            system_prompt="You identify structural connections between concepts.",
            max_tokens=200,
            temperature=0.3,
        )
        pair_claim = pair_claim.strip()

        # 3. Score both with the coherence judge
        walk_level, walk_response = await _score_coherence(
            model_manager, bridge_model, concept_a, concept_b, walk_claim
        )
        pair_level, pair_response = await _score_coherence(
            model_manager, bridge_model, concept_a, concept_b, pair_claim
        )

        print(f"    Walk claim:  {walk_claim[:120]}...")
        print(f"    Walk score:  {walk_level}")
        print(f"    Pair claim:  {pair_claim[:120]}...")
        print(f"    Pair score:  {pair_level}")

        walk_wins = _level_value(walk_level) > _level_value(pair_level)
        tie = _level_value(walk_level) == _level_value(pair_level)
        print(f"    Result:      {'WALK WINS' if walk_wins else 'TIE' if tie else 'PAIR WINS'}")

        results.append({
            "path": path_str,
            "description": test["description"],
            "walk_claim": walk_claim,
            "walk_level": walk_level,
            "pair_claim": pair_claim,
            "pair_level": pair_level,
            "walk_wins": walk_wins,
            "tie": tie,
        })

    # Summary
    walk_wins = sum(1 for r in results if r["walk_wins"])
    ties = sum(1 for r in results if r["tie"])
    pair_wins = len(results) - walk_wins - ties

    walk_moderate_plus = sum(1 for r in results if _level_value(r["walk_level"]) >= 0.66)
    pair_moderate_plus = sum(1 for r in results if _level_value(r["pair_level"]) >= 0.66)

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Walk wins:  {walk_wins}/{len(results)}")
    print(f"  Ties:       {ties}/{len(results)}")
    print(f"  Pair wins:  {pair_wins}/{len(results)}")
    print(f"  Walk MODERATE+: {walk_moderate_plus}/{len(results)}")
    print(f"  Pair MODERATE+: {pair_moderate_plus}/{len(results)}")
    print()
    if walk_moderate_plus > pair_moderate_plus:
        print(f"  VERDICT: Walk narration produces better coherence. GO for graph walk architecture.")
    elif walk_moderate_plus == pair_moderate_plus:
        print(f"  VERDICT: No significant difference. Walk prompt may need tuning before building infrastructure.")
    else:
        print(f"  VERDICT: Random pairing produces better coherence. Walk architecture may not be worth it.")


async def _score_coherence(model_manager, model, concept_a, concept_b, claim):
    """Run the coherence judge (Stage 5 Pass 1) on a claim. Returns (level_str, response)."""
    prompt = (
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

    response = await model_manager.generate_once(
        prompt=prompt,
        model_name=model,
        system_prompt="You evaluate cross-domain knowledge connections. Focus on whether a shared structural pattern genuinely exists, not on whether the domains seem related. Be skeptical of vague claims but generous toward specific structural parallels.",
        max_tokens=250,
        temperature=0.1,
    )
    response = response.strip()

    # Parse rating
    level = "UNKNOWN"
    for line in response.split("\n"):
        if line.strip().startswith("Rating:"):
            for candidate in ["STRONG", "MODERATE", "WEAK", "INVALID"]:
                if candidate in line:
                    level = candidate
                    break
            break

    return level, response


def _level_value(level_str):
    return {"STRONG": 1.0, "MODERATE": 0.66, "WEAK": 0.33, "INVALID": 0.0}.get(level_str, 0.0)


def main():
    parser = argparse.ArgumentParser(description="Walk prompt vs random-pair prompt validation")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model (default: SYNTHESIS_COHERENCE_MODEL)")
    args = parser.parse_args()

    asyncio.run(run_validation(model_override=args.model))


if __name__ == "__main__":
    main()
