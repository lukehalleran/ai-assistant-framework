# memory/surfacing_models.py
"""
Pydantic data models for proactive context surfacing.

Supports cross-domain insight generation by classifying knowledge graph
entities into life domains and identifying non-obvious bridges between them.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class DomainEntity(BaseModel):
    """An entity classified into a life domain."""

    entity_id: str = Field(..., description="Canonical entity ID from graph")
    display_name: str = Field(..., description="Human-readable name")
    domain: str = Field(..., description="ProfileCategory value, e.g. 'career', 'health'")
    relation: str = Field(..., description="Relation from user to this entity")
    edge_weight: float = Field(default=1.0, ge=0.0)


class DomainCluster(BaseModel):
    """A group of entities sharing the same life domain."""

    domain: str = Field(..., description="ProfileCategory value")
    entities: List[DomainEntity] = Field(default_factory=list)
    fact_sentences: List[str] = Field(
        default_factory=list,
        description="Natural language facts about entities in this cluster",
    )


class CrossDomainCandidate(BaseModel):
    """A potential bridge between two life domains."""

    active_domain: str = Field(..., description="Domain the current query touches")
    bridged_domain: str = Field(..., description="Domain being connected to")
    active_cluster: DomainCluster
    bridged_cluster: DomainCluster
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ProactiveInsight(BaseModel):
    """A synthesized cross-domain insight ready for prompt injection."""

    insight_text: str = Field(..., description="One-sentence insight for the user")
    active_domain: str = Field(..., description="Domain the query touched")
    bridged_domain: str = Field(..., description="Domain bridged to")
    entity_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    generated_at: Optional[datetime] = Field(default=None)

    def novelty_key(self) -> str:
        """Deterministic key for dedup: sorted domain pair + sorted entity IDs."""
        domains = sorted([self.active_domain, self.bridged_domain])
        entities = sorted(self.entity_ids)
        return "|".join(domains) + "|" + ",".join(entities)
