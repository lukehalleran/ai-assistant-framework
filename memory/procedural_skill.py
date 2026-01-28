# memory/procedural_skill.py
"""
Procedural Skill schema for 'How-To' memory.

Represents reusable problem-solving patterns extracted from sessions.
Stored in ChromaDB 'procedural_skills' collection.
Embedding is generated from trigger + tags (situation context);
full action_pattern is stored in metadata for retrieval.
"""

import json
import time
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class SkillCategory(str, Enum):
    """Categories for procedural skills."""
    DEBUGGING = "debugging"
    WORKFLOW = "workflow"
    PROMPT_ENG = "prompt_engineering"
    INTERPERSONAL = "interpersonal"
    ARCHITECTURAL = "architectural"
    OPTIMIZATION = "optimization"
    TESTING = "testing"


class ProceduralSkill(BaseModel):
    """
    A 'How-To' memory -- a specific pattern or workflow that led to
    success in the past.

    Stored in ChromaDB 'procedural_skills' collection.
    Embedding is generated from trigger + tags (situation context).
    Full action_pattern stored in metadata for retrieval.
    """

    trigger: str = Field(
        ...,
        description="The situation that triggers this skill",
        min_length=10,
        max_length=500,
    )
    action_pattern: str = Field(
        ...,
        description="The abstract solution pattern (not specific code)",
        min_length=20,
        max_length=1000,
    )
    category: SkillCategory
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How generalizable this pattern is (0.0-1.0)",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Keywords for retrieval filtering",
        max_length=10,
    )
    source_session_id: str = Field(
        ...,
        description="Session ID where this skill was learned",
    )
    created_at: float = Field(default_factory=time.time)
    times_retrieved: int = Field(
        default=0, description="Usage counter for relevance boosting"
    )
    last_retrieved: Optional[float] = Field(default=None)

    def to_embedding_text(self) -> str:
        """
        Generate text for embedding.  Focuses on TRIGGER (situation) +
        TAGS (context).  Action pattern is stored in metadata, not
        embedded -- we match on problem, not solution.
        """
        tag_str = ", ".join(self.tags) if self.tags else ""
        return (
            f"Situation: {self.trigger} | Context: {tag_str} "
            f"| Category: {self.category.value}"
        )

    def to_metadata(self) -> dict:
        """
        Serialize for ChromaDB metadata storage.
        Tags are JSON-serialized since ChromaDB doesn't support list metadata.
        """
        return {
            "trigger": self.trigger,
            "action_pattern": self.action_pattern,
            "category": self.category.value,
            "confidence": self.confidence,
            "tags_json": json.dumps(self.tags),
            "source_session_id": self.source_session_id,
            "created_at": self.created_at,
            "times_retrieved": self.times_retrieved,
            "last_retrieved": self.last_retrieved or 0.0,
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "ProceduralSkill":
        """Deserialize from ChromaDB metadata."""
        return cls(
            trigger=metadata["trigger"],
            action_pattern=metadata["action_pattern"],
            category=SkillCategory(metadata["category"]),
            confidence=metadata["confidence"],
            tags=json.loads(metadata.get("tags_json", "[]")),
            source_session_id=metadata["source_session_id"],
            created_at=metadata.get("created_at", time.time()),
            times_retrieved=metadata.get("times_retrieved", 0),
            last_retrieved=metadata.get("last_retrieved") or None,
        )

    def generate_id(self) -> str:
        """Generate deterministic ID for deduplication checks."""
        return f"skill_{self.source_session_id}_{int(self.created_at * 1000)}"
