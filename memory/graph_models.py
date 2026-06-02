# memory/graph_models.py
"""
Pydantic data models for the knowledge graph.

GraphNode represents an entity (person, place, project, concept, etc.).
GraphEdge represents a directed relationship between two entities.

These models are used by GraphMemory for JSON serialization and by
EntityResolver for alias management.

GraphNode.source property returns provenance ('personal', 'wikidata',
or 'wiki_retrieved') from metadata, defaulting to 'personal'.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """An entity in the knowledge graph."""

    entity_id: str = Field(
        ...,
        description="Normalized lowercase key, e.g. 'daemon', 'spain', 'whiskers'",
    )
    display_name: str = Field(
        ...,
        description="Original casing for display: 'Daemon', 'Spain'",
    )
    entity_type: str = Field(
        default="other",
        description="Entity type: person, project, place, concept, organization, pet, other",
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names that resolve to this entity",
    )
    first_seen: Optional[datetime] = Field(default=None)
    last_seen: Optional[datetime] = Field(default=None)
    mention_count: int = Field(default=0, ge=0)
    metadata: dict = Field(default_factory=dict)

    @property
    def source(self) -> str:
        """Node provenance: 'personal', 'wikidata', or 'wiki_retrieved'."""
        return self.metadata.get("source", "personal")

    @property
    def is_wikidata(self) -> bool:
        return self.source == "wikidata"

    @property
    def wikidata_qid(self) -> str | None:
        """Wikidata Q-ID (e.g. 'Q44') if this is a Wikidata-sourced node."""
        return self.metadata.get("wikidata_qid")

    def to_dict(self) -> dict:
        """Serialize for JSON persistence."""
        d = {
            "display_name": self.display_name,
            "entity_type": self.entity_type,
            "aliases": self.aliases,
            "mention_count": self.mention_count,
            "metadata": self.metadata,
        }
        if self.first_seen:
            d["first_seen"] = self.first_seen.isoformat()
        if self.last_seen:
            d["last_seen"] = self.last_seen.isoformat()
        return d

    @classmethod
    def from_dict(cls, entity_id: str, data: dict) -> "GraphNode":
        """Deserialize from JSON persistence."""
        first_seen = None
        if data.get("first_seen"):
            first_seen = datetime.fromisoformat(data["first_seen"])
        last_seen = None
        if data.get("last_seen"):
            last_seen = datetime.fromisoformat(data["last_seen"])
        return cls(
            entity_id=entity_id,
            display_name=data.get("display_name", entity_id),
            entity_type=data.get("entity_type", "other"),
            aliases=data.get("aliases", []),
            first_seen=first_seen,
            last_seen=last_seen,
            mention_count=data.get("mention_count", 0),
            metadata=data.get("metadata", {}),
        )


class GraphEdge(BaseModel):
    """A directed relationship between two entities."""

    source_id: str = Field(
        ...,
        description="Source entity ID (normalized lowercase)",
    )
    relation: str = Field(
        ...,
        description="Normalized relation: 'lives_in', 'works_on', 'created'",
    )
    target_id: str = Field(
        ...,
        description="Target entity ID (normalized lowercase)",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Strengthened by repeated mentions",
    )
    truth_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Evidence-based truth score",
    )
    first_seen: Optional[datetime] = Field(default=None)
    last_seen: Optional[datetime] = Field(default=None)
    source_fact_ids: list[str] = Field(
        default_factory=list,
        description="ChromaDB fact doc IDs for provenance",
    )
    metadata: dict = Field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for JSON persistence."""
        d = {
            "source_id": self.source_id,
            "relation": self.relation,
            "target_id": self.target_id,
            "weight": self.weight,
            "truth_score": self.truth_score,
            "source_fact_ids": self.source_fact_ids,
            "metadata": self.metadata,
        }
        if self.first_seen:
            d["first_seen"] = self.first_seen.isoformat()
        if self.last_seen:
            d["last_seen"] = self.last_seen.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "GraphEdge":
        """Deserialize from JSON persistence."""
        first_seen = None
        if data.get("first_seen"):
            first_seen = datetime.fromisoformat(data["first_seen"])
        last_seen = None
        if data.get("last_seen"):
            last_seen = datetime.fromisoformat(data["last_seen"])
        return cls(
            source_id=data["source_id"],
            relation=data["relation"],
            target_id=data["target_id"],
            weight=data.get("weight", 1.0),
            truth_score=data.get("truth_score", 1.0),
            first_seen=first_seen,
            last_seen=last_seen,
            source_fact_ids=data.get("source_fact_ids", []),
            metadata=data.get("metadata", {}),
        )

    def edge_key(self) -> str:
        """Composite key for deduplication: source|relation|target."""
        return f"{self.source_id}|{self.relation}|{self.target_id}"

    def to_natural_language(self, source_display: str = "", target_display: str = "", with_attribution: bool = False) -> str:
        """Render as a human-readable sentence for prompt injection.

        Args:
            source_display: Display name for source entity
            target_display: Display name for target entity
            with_attribution: If True, append derivation markers to indicate this is from relationship data
        """
        src = source_display or self.source_id
        tgt = target_display or self.target_id
        rel = self.relation.replace("_", " ")
        base_sentence = f"{src} {rel} {tgt}"

        if with_attribution:
            return f"{base_sentence} (from relationship data)"
        return base_sentence
