#!/usr/bin/env python3
"""
Live LLM calibration for synthesis filter pipeline.

Runs the fixture candidates through the FULL pipeline with real LLM
coherence judging (Stage 5) and mock FAISS wiki search (Stages 3/3b).
Reports per-tier diagnostics including co-occurrence, template specificity,
and composite novelty signals.

Usage:
    python scripts/calibrate_coherence_live.py                      # default model
    python scripts/calibrate_coherence_live.py sonnet-4.5 gpt-4o-mini  # compare two models
"""

import asyncio
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.app_config import SYNTHESIS_COHERENCE_MODEL, SYNTHESIS_COHERENCE_MIN_LEVEL
from knowledge.synthesis_filter import SynthesisFilter
from knowledge.synthesis_models import (
    CandidateStatus,
    CoherenceLevel,
    SynthesisCandidate,
    SynthesisResult,
)
from memory.synthesis_memory import SynthesisMemory
from models.model_manager import ModelManager


FIXTURES_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "calibration_candidates.json"

# Tiers where coherence should reject
_REJECT_TIERS = {"sanity_fail", "trivial", "noise", "noise_borderline", "interesting_known"}
_ACCEPT_TIERS = {"interesting_novel"}
_DIAGNOSTIC_TIERS = {"boundary"}


def _build_mock_store():
    """Mock ChromaDB store for synthesis_results. Wiki queries go through FAISS mock."""
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


# Tier-aware FAISS mock for novelty checks (Stage 3)
_TRIVIAL_TOPICS = {
    "physical exercise", "mental health", "sleep", "cognitive", "compound interest",
    "retirement", "musical instrument", "mathematical reasoning",
    "bilingualism", "dementia", "dark chocolate", "cardiovascular",
    "pet ownership", "blood pressure", "nature", "creativity",
    "gut microbiome", "mood", "cold exposure", "immune",
    "video games", "surgical", "standing desk", "back pain",
    "gratitude", "sleep quality",
    "musical training", "pattern recognition", "spatial-temporal",
    "gut-brain", "serotonin", "vagus nerve", "endorphin",
    "neuroplasticity", "oxytocin", "cortisol",
    "flavanol", "nitric oxide", "endothelial",
    "norepinephrine", "wim hof", "cold water immersion",
    "laparoscopic", "fine motor", "spinal compression",
    "prefrontal cortex", "cognitive arousal",
}

_KNOWN_TOPICS = {
    "ant colony", "tcp", "congestion", "immune system", "caching", "cache",
    "mycelial", "content delivery", "mycorrhizal", "predator-prey",
    "autoscaling", "lotka-volterra", "counterpoint", "adversarial",
    "gans", "bach",
    "memory b cells", "cdn", "wood wide web",
    "dorigo", "slow-start",
}


def _mock_faiss_wiki_search(query, k=3):
    """Tier-aware FAISS mock with calibrated similarity scores."""
    query_lower = query.lower()
    is_trivial = any(topic in query_lower for topic in _TRIVIAL_TOPICS)
    is_known = any(topic in query_lower for topic in _KNOWN_TOPICS)

    if is_trivial:
        return [{"text": query[:200], "content": query[:200],
                 "title": "Wikipedia: Well-Known Connection",
                 "source": "wikipedia", "similarity": 0.90}]
    elif is_known:
        return [{"text": f"Published analysis of {query[:100]}",
                 "content": f"Published analysis of {query[:100]}",
                 "title": "Academic Survey",
                 "source": "wikipedia", "similarity": 0.82}]
    else:
        return [{"text": "Unrelated article about ancient pottery techniques.",
                 "content": "Unrelated article about ancient pottery techniques.",
                 "title": "Pottery",
                 "source": "wikipedia", "similarity": 0.05}]


def _to_candidate(entry):
    return SynthesisCandidate(
        concept_a=entry["concept_a"],
        concept_b=entry["concept_b"],
        connection_claim=entry["connection_claim"],
        walk_path=[entry["concept_a"].lower(), entry["concept_b"].lower()],
        source_domains=set(entry["source_domains"]),
        endpoint_distance=entry["endpoint_distance"],
        timestamp=datetime.now(),
    )


async def run_full_pipeline(model_name, candidates_data, store, mm):
    """Run all candidates through the full synthesis filter with a real LLM."""
    # Monkey-patch model_manager to force specific model for coherence
    original_model = os.environ.get("SYNTHESIS_COHERENCE_MODEL")

    synthesis_memory = SynthesisMemory(store)
    pipeline = SynthesisFilter(
        chroma_store=store,
        model_manager=mm,
        synthesis_memory=synthesis_memory,
    )

    # Override coherence model at config level
    import config.app_config as cfg
    saved = cfg.SYNTHESIS_COHERENCE_MODEL
    cfg.SYNTHESIS_COHERENCE_MODEL = model_name

    results = []
    for i, entry in enumerate(candidates_data):
        candidate = _to_candidate(entry)
        result = await pipeline.process_candidate(candidate)

        record = {
            "tier": entry["tier"],
            "concept_a": entry["concept_a"],
            "concept_b": entry["concept_b"],
            "expected": entry["expected_outcome"],
            "actual": "accepted" if result.status == CandidateStatus.ACCEPTED else "rejected",
            "rejection_stage": result.rejection_stage,
            "composite_score": result.composite_score,
            "coherence": result.coherence_level.name if result.coherence_level else "N/A",
            "coherence_val": result.coherence_level.value if result.coherence_level else 0.0,
            "novelty_ext": result.novelty_score_external,
            "cooccurrence_sim": result.cooccurrence_similarity,
            "template_sim": result.template_similarity,
            "justification": result.coherence_justification[:120] if result.coherence_justification else "",
            "notes": entry.get("notes", ""),
        }
        results.append(record)

        status = "PASS" if record["actual"] == "accepted" else "FAIL"
        coh = record["coherence"]
        print(
            f"  [{i+1:2d}/{len(candidates_data)}] {entry['tier']:18s} | "
            f"{coh:8s} {status:4s} | "
            f"tpl={record['template_sim']:.2f} | "
            f"{entry['concept_a']} <-> {entry['concept_b']}"
        )

    cfg.SYNTHESIS_COHERENCE_MODEL = saved
    return results


def print_report(model_name, results):
    """Print per-tier report with full signal breakdown."""
    tier_order = [
        "sanity_fail", "trivial", "noise", "noise_borderline",
        "interesting_known", "interesting_novel", "boundary",
    ]

    # Confusion matrix (excluding boundary)
    scored = [r for r in results if r["tier"] not in _DIAGNOSTIC_TIERS]
    tp = [r for r in scored if r["expected"] == "accepted" and r["actual"] == "accepted"]
    fp = [r for r in scored if r["expected"] == "rejected" and r["actual"] == "accepted"]
    tn = [r for r in scored if r["expected"] == "rejected" and r["actual"] == "rejected"]
    fn = [r for r in scored if r["expected"] == "accepted" and r["actual"] == "rejected"]

    precision = len(tp) / (len(tp) + len(fp)) * 100 if (len(tp) + len(fp)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) * 100 if (len(tp) + len(fn)) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  CONFUSION: TP={len(tp)} FP={len(fp)} TN={len(tn)} FN={len(fn)}")
    print(f"  Precision={precision:.1f}%  Recall={recall:.1f}%  F1={f1:.1f}%")

    for tier in tier_order:
        tier_results = [r for r in results if r["tier"] == tier]
        if not tier_results:
            continue

        accepted = [r for r in tier_results if r["actual"] == "accepted"]
        rejected = [r for r in tier_results if r["actual"] == "rejected"]

        # Coherence distribution
        coh_counts = defaultdict(int)
        for r in tier_results:
            coh_counts[r["coherence"]] += 1

        # Rejection stage breakdown
        stage_counts = defaultdict(int)
        for r in rejected:
            stage_counts[r["rejection_stage"] or "none"] += 1

        print(f"\n  {tier.upper()} ({len(tier_results)}): "
              f"accepted={len(accepted)}/{len(tier_results)}  "
              f"coherence={dict(coh_counts)}")

        if stage_counts:
            print(f"    Rejection stages: {dict(stage_counts)}")

        # Composite scores
        composites = [r["composite_score"] for r in tier_results if r["composite_score"] > 0]
        if composites:
            print(f"    Composite: min={min(composites):.3f} max={max(composites):.3f} mean={sum(composites)/len(composites):.3f}")

        # Template similarity
        templates = [r["template_sim"] for r in tier_results]
        if any(v > 0 for v in templates):
            print(f"    Template sim: min={min(templates):.3f} max={max(templates):.3f} mean={sum(templates)/len(templates):.3f}")

        # Show leaks (FP) and over-rejections (FN)
        if tier in ("noise", "noise_borderline"):
            for r in accepted:
                print(f"    LEAKED: {r['concept_a']} <-> {r['concept_b']}: "
                      f"{r['coherence']}, composite={r['composite_score']:.3f}, "
                      f"tpl={r['template_sim']:.2f}")
                print(f"      {r['justification']}")
        elif tier == "interesting_novel":
            for r in rejected:
                print(f"    OVER-REJECTED: {r['concept_a']} <-> {r['concept_b']}: "
                      f"{r['coherence']}, rejected@{r['rejection_stage']}, "
                      f"tpl={r['template_sim']:.2f}")
                print(f"      {r['justification']}")

    # Boundary detail
    boundary = [r for r in results if r["tier"] == "boundary"]
    if boundary:
        print(f"\n  BOUNDARY DETAIL:")
        for r in boundary:
            actual = "ACCEPTED" if r["actual"] == "accepted" else f"REJECTED@{r['rejection_stage']}"
            match = "OK" if r["expected"] == r["actual"] else "SURPRISE"
            print(f"    [{match}] {r['concept_a']} <-> {r['concept_b']}: "
                  f"{actual}, {r['coherence']}, composite={r['composite_score']:.3f}, "
                  f"tpl={r['template_sim']:.2f}")

    # Misclassifications
    misclassified = fp + fn
    if misclassified:
        print(f"\n  MISCLASSIFICATIONS ({len(misclassified)}):")
        for r in misclassified:
            direction = "FP" if r in fp else "FN"
            print(f"    [{direction}] {r['concept_a']} <-> {r['concept_b']}: "
                  f"tier={r['tier']}, {r['coherence']}, composite={r['composite_score']:.3f}, "
                  f"rejected@{r['rejection_stage']}, tpl={r['template_sim']:.2f}")
            print(f"      {r['justification']}")

    return {
        "model": model_name,
        "tp": len(tp), "fp": len(fp), "tn": len(tn), "fn": len(fn),
        "precision": precision, "recall": recall, "f1": f1,
    }


async def main():
    models_to_test = sys.argv[1:] if len(sys.argv) > 1 else [SYNTHESIS_COHERENCE_MODEL]

    with open(FIXTURES_PATH) as f:
        data = json.load(f)
    candidates_data = [c for c in data["candidates"] if not c.get("_comment")]

    store = _build_mock_store()
    mm = ModelManager()

    print(f"Candidates: {len(candidates_data)}")
    print(f"Models: {models_to_test}")
    print(f"Pipeline: FULL (mock FAISS wiki + real LLM coherence)")

    all_summaries = []

    for model in models_to_test:
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model}")
        print("=" * 80)

        with patch("knowledge.semantic_search.semantic_search_with_neighbors",
                    side_effect=_mock_faiss_wiki_search):
            results = await run_full_pipeline(model, candidates_data, store, mm)
        summary = print_report(model, results)
        all_summaries.append(summary)

    if len(all_summaries) > 1:
        print(f"\n{'=' * 80}")
        print("COMPARISON TABLE")
        print("=" * 80)
        print(f"  {'Model':<25s} {'Prec':>6s} {'Recall':>7s} {'F1':>6s}  {'TP':>3s} {'FP':>3s} {'TN':>3s} {'FN':>3s}")
        print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*6}  {'-'*3} {'-'*3} {'-'*3} {'-'*3}")
        for s in all_summaries:
            print(f"  {s['model']:<25s} {s['precision']:5.1f}% {s['recall']:6.1f}% {s['f1']:5.1f}%  {s['tp']:3d} {s['fp']:3d} {s['tn']:3d} {s['fn']:3d}")


if __name__ == "__main__":
    asyncio.run(main())
