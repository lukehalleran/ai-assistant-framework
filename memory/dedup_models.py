# memory/dedup_models.py
"""
Data models for cross-collection memory deduplication.

Defines the action types, duplicate pair records, contradiction clusters,
and the overall dedup plan used by CrossCollectionDeduplicator.
"""

import time
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DedupAction(str, Enum):
    """Action to take on a detected duplicate or contradiction."""
    KEEP = "keep"            # Retain this document as the canonical version
    DELETE = "delete"        # Remove this document (superseded by another)
    MERGE = "merge"          # Merge content into the kept document
    SUPERSEDE = "supersede"  # Newer version replaces older


class DedupReason(str, Enum):
    """Why a dedup action was recommended."""
    CROSS_DUPLICATE = "cross_duplicate"        # Same content across collections
    WITHIN_DUPLICATE = "within_duplicate"      # Same content within a collection
    FACT_CONTRADICTION = "fact_contradiction"   # Same subject+predicate, different object
    SUBSUMED = "subsumed"                      # Content fully captured in a summary


class DuplicatePair(BaseModel):
    """A pair of documents detected as duplicates or near-duplicates."""

    doc_id_a: str = Field(..., description="ID of the first document")
    doc_id_b: str = Field(..., description="ID of the second document")
    collection_a: str = Field(..., description="Collection of the first document")
    collection_b: str = Field(..., description="Collection of the second document")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    content_preview_a: str = Field("", max_length=200, description="Truncated content of doc A")
    content_preview_b: str = Field("", max_length=200, description="Truncated content of doc B")
    reason: DedupReason = Field(DedupReason.CROSS_DUPLICATE)
    action: DedupAction = Field(DedupAction.DELETE, description="Recommended action for doc B")
    keep_id: str = Field("", description="ID of the document to keep")
    delete_id: str = Field("", description="ID of the document to delete")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in this recommendation")


class ContradictionCluster(BaseModel):
    """A group of facts that contradict each other (same subject+predicate, different object)."""

    subject: str = Field(..., description="Shared subject across contradicting facts")
    predicate: str = Field(..., description="Shared predicate/relation")
    entries: List[dict] = Field(default_factory=list, description="List of {id, content, metadata, collection} dicts")
    keep_id: str = Field("", description="ID of the fact to keep (most recent)")
    delete_ids: List[str] = Field(default_factory=list, description="IDs of facts to remove")


class DedupPlan(BaseModel):
    """Complete deduplication plan with audit trail."""

    duplicate_pairs: List[DuplicatePair] = Field(default_factory=list)
    contradiction_clusters: List[ContradictionCluster] = Field(default_factory=list)
    collections_scanned: List[str] = Field(default_factory=list)
    total_docs_scanned: int = Field(0, ge=0)
    duplicates_found: int = Field(0, ge=0)
    contradictions_found: int = Field(0, ge=0)
    deletions_executed: int = Field(0, ge=0)
    dry_run: bool = Field(True)
    created_at: float = Field(default_factory=time.time)
    errors: List[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the plan as a Markdown report for GUI display."""
        lines = [
            f"**Collections scanned:** {', '.join(self.collections_scanned)}",
            f"**Documents scanned:** {self.total_docs_scanned}",
            f"**Duplicate pairs found:** {self.duplicates_found}",
            f"**Contradiction clusters:** {self.contradictions_found}",
            f"**Deletions {'proposed' if self.dry_run else 'executed'}:** {self.deletions_executed}",
            f"**Mode:** {'DRY RUN' if self.dry_run else 'LIVE'}",
        ]

        if self.duplicate_pairs:
            lines.append("\n#### Duplicate Pairs")
            for p in self.duplicate_pairs[:30]:
                preview_a = p.content_preview_a[:80] + "..." if len(p.content_preview_a) > 80 else p.content_preview_a
                preview_b = p.content_preview_b[:80] + "..." if len(p.content_preview_b) > 80 else p.content_preview_b
                lines.append(
                    f"- **{p.collection_a}** vs **{p.collection_b}** "
                    f"(sim={p.similarity:.3f}, {p.reason.value})\n"
                    f"  - Keep: `{preview_a}`\n"
                    f"  - Delete: `{preview_b}`"
                )

        if self.contradiction_clusters:
            lines.append("\n#### Fact Contradictions")
            for c in self.contradiction_clusters[:20]:
                lines.append(f"- **{c.subject}** / **{c.predicate}** — {len(c.entries)} versions, keeping most recent")
                for entry in c.entries[:5]:
                    marker = "KEEP" if entry.get("id") == c.keep_id else "DELETE"
                    content = str(entry.get("content", ""))[:80]
                    lines.append(f"  - [{marker}] `{content}`")

        if self.errors:
            lines.append("\n#### Errors")
            for err in self.errors:
                lines.append(f"- {err}")

        return "\n".join(lines)
