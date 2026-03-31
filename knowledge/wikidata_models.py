# knowledge/wikidata_models.py
"""
Pydantic data models for Wikidata subgraph import.

WikidataEntity represents a Wikidata item (Q-ID + label + description).
WikidataRelation represents a Wikidata property linking two items.
WikidataImportResult tracks import statistics.
"""

from pydantic import BaseModel, Field


class WikidataEntity(BaseModel):
    """A Wikidata item to be imported into the knowledge graph."""

    qid: str = Field(..., description="Wikidata Q-ID, e.g. 'Q44'")
    label: str = Field(..., description="English label, e.g. 'beer'")
    description: str = Field(default="", description="Short Wikidata description")
    aliases: list[str] = Field(default_factory=list)
    domain_category: str = Field(default="cross_domain_science")


class WikidataRelation(BaseModel):
    """A Wikidata property linking two entities."""

    source_qid: str = Field(..., description="Source entity Q-ID")
    property_id: str = Field(..., description="Wikidata property ID, e.g. 'P31'")
    target_qid: str = Field(..., description="Target entity Q-ID")
    relation_label: str = Field(
        ..., description="Normalized relation name, e.g. 'instance_of'"
    )


class WikidataImportResult(BaseModel):
    """Statistics from a Wikidata import run."""

    entities_imported: int = 0
    relations_imported: int = 0
    personal_mappings: int = 0
    bridge_edges: int = 0
    errors: list[str] = Field(default_factory=list)
