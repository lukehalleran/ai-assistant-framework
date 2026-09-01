"""
core/insight/types.py

Module Contract
- Purpose: Pure data models for the insight / evidence-assembly mode. No IO,
  no LLM calls — sibling of core/agentic/types.py.
- Key models:
    InsightIntent    what the user asked for (theme sweep vs. insight
                     assessment) and whether a document should be saved
    FacetQuery/Plan  the decomposed search plan (facets convert the
                     collective-signal problem into N per-doc-tractable
                     searches; assessment plans also carry checkable claims)
    EvidenceItem     one provenance-labeled piece of evidence
    ClaimAssessment/Assessment
                     adversarial verdicts; overall is WORST-OF so the mode
                     fails honest, never fail-agree
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# Ordered worst → best. Assessment.overall is the worst verdict present.
VERDICT_ORDER = ("disagree", "insufficient", "partial", "agree")

# Provenance labels an EvidenceItem can carry (rendered to the user verbatim).
STANCE_LABELS = (
    "user-stated",        # the user's own words (conversation / corpus, user side)
    "users-own-note",     # the user's Obsidian notes
    "assistant-inferred", # Daemon-authored text (responses, summaries,
                          # reflections, threads) — interpretation, not the
                          # user's words
    "extracted-fact",     # structured fact triple from the facts collection
    "graph-edge",         # knowledge-graph relation
    "external-research",  # independently retrieved web/wiki/literature source
    "computed-evidence",  # independently executed computation with inputs retained
)


class InsightIntent(BaseModel):
    """A detected insight-mode request."""

    kind: str = Field(description="theme_sweep | insight_assessment | pattern_temporal")
    theme: str = Field(description="The personal theme / stated insight to work on")
    wants_document: bool = Field(default=False)
    raw_query: str = Field(default="")
    # pattern_temporal only (2026-08-29): parsed time window (0 = config
    # default, -1 = all history) and optional engine-dimension hint.
    window_days: int = Field(default=0)
    dimension: str = Field(default="")


class FacetQuery(BaseModel):
    """One angle of the sweep: a semantic query plus exact keywords/entities."""

    name: str
    query_text: str
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)


class FacetPlan(BaseModel):
    """Decomposed search plan. ``claims`` is populated for assessment runs."""

    facets: list[FacetQuery] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)
    fallback: bool = Field(
        default=False,
        description="True when LLM decomposition failed and a single-facet plan was used",
    )


class EvidenceItem(BaseModel):
    """One provenance-labeled evidence snippet."""

    doc_id: Optional[str] = None
    text: str
    date: Optional[str] = None          # ISO date/datetime string when known
    collection: str = ""                # source store (or 'corpus' / 'graph')
    speaker: str = ""                   # 'user' | 'assistant' | '' (unknown)
    stance_label: str = "user-stated"   # one of STANCE_LABELS
    is_appraisal: bool = False
    facet: str = ""


class ClaimAssessment(BaseModel):
    """Adversarial verdict on a single checkable claim."""

    claim: str
    verdict: str = Field(default="insufficient")  # one of VERDICT_ORDER
    support: list[str] = Field(default_factory=list)  # evidence refs, e.g. "E3"
    refute: list[str] = Field(default_factory=list)
    notes: str = ""
    claim_id: str = ""
    confidence: float = 0.0
    evidence_coverage: str = "unknown"
    directness: str = "unknown"
    dependencies: list[str] = Field(default_factory=list)
    authority: str = "assessment"


class Assessment(BaseModel):
    """Overall assessment: worst-of over claims (fail honest, never fail agree)."""

    overall: str = Field(default="insufficient")
    claims: list[ClaimAssessment] = Field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_claims(cls, claims: list[ClaimAssessment]) -> "Assessment":
        if not claims:
            return cls(overall="insufficient", claims=[])
        worst = min(
            (c.verdict if c.verdict in VERDICT_ORDER else "insufficient" for c in claims),
            key=VERDICT_ORDER.index,
        )
        # Keep legacy worst-of ``overall`` for compatibility, while exposing
        # claim-level results as authoritative and identifying mixed chains.
        kinds = {c.verdict for c in claims}
        summary = "mixed claim statuses; inspect each claim and dependencies" if len(kinds) > 1 else "uniform claim statuses"
        return cls(overall=worst, claims=claims, summary=summary)

    @property
    def allows_document(self) -> bool:
        """Doc auto-save only on agree/partial (explicit override handled upstream)."""
        return self.overall in ("agree", "partial")
