"""
core/insight — Insight / evidence-assembly mode.

A turn-owning mode (parallel to core/agentic) that deliberately assembles
cross-store evidence on a personal theme instead of relying on the memory
gate's per-doc cosine filter — which structurally cannot pass a
low-pairwise-similarity / high-collective-signal evidence set (the "lucky
string" forensics of 2026-08-23: Daemon's best pattern document worked only
because all evidence happened to sit in one 29-turn recent-conversation
window; the gate passed 0 documents naturally).

Pipeline:  detector → facets (decompose) → sweep (ungated, all stores)
           → provenance (stance labeling) → assessor (adversarial, optional)
           → synthesizer (MI-shaped, denominator caveat, honest verdict).

Doctrine (owner-settled 2026-08-23):
  * evidence-assembly, never verdict-rendering — quote + date, the user draws
    conclusions (except the explicitly-requested assessment path, which gives
    a full honest verdict either way);
  * mandatory denominator caveat — the corpus over-samples distress days;
  * provenance first-class — assistant elaborations are labeled inference,
    never presented as the user's words;
  * never proactive at elevated tone; explicit requests always work.
"""

from core.insight.types import (
    Assessment,
    ClaimAssessment,
    EvidenceItem,
    FacetPlan,
    FacetQuery,
    InsightIntent,
)
from core.insight.detector import detect_insight_request, detect_insight_statement

__all__ = [
    "InsightIntent",
    "FacetQuery",
    "FacetPlan",
    "EvidenceItem",
    "ClaimAssessment",
    "Assessment",
    "detect_insight_request",
    "detect_insight_statement",
]
