# memory/thread_models.py
"""
Data models for proactive thread surfacing.

Module Contract
- Purpose: Pydantic models for open threads (commitments, deadlines, unfinished
  topics, unanswered questions) extracted from conversations and surfaced at
  session start.
- Inputs: Raw dicts from LLM JSON output (via from_dict / from_metadata)
- Outputs: Serializable thread objects with status lifecycle and priority scoring
- Key behaviors:
  - Pydantic-validated fields with sensible defaults
  - to_metadata() / from_metadata() for ChromaDB storage (flat primitives)
  - to_dict() / from_dict() for full JSON serialization (tests, export)
  - Status lifecycle: OPEN -> RESOLVED | STALE
  - Priority scoring: type_weight * urgency * recency_decay
  - is_stale(stale_days, deadline_grace_hours): checks both time-since-reference
    AND deadline_date + grace period for deadline-aware expiry
- Side effects: None (pure data model)
"""

import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ThreadType(str, Enum):
    """Type of open thread."""
    COMMITMENT = "commitment"
    DEADLINE = "deadline"
    UNFINISHED = "unfinished"
    QUESTION = "question"


class ThreadStatus(str, Enum):
    """Lifecycle status of a thread."""
    OPEN = "open"
    RESOLVED = "resolved"
    STALE = "stale"


# Priority weights by thread type (higher = more urgent to surface)
TYPE_PRIORITY = {
    ThreadType.DEADLINE: 1.0,
    ThreadType.COMMITMENT: 0.8,
    ThreadType.QUESTION: 0.6,
    ThreadType.UNFINISHED: 0.4,
}


class OpenThread(BaseModel):
    """
    An open thread representing an unresolved commitment, deadline,
    unfinished topic, or unanswered question from a conversation.

    Stored in ChromaDB 'threads' collection.
    Embedding text is generated from topic + summary for semantic matching.
    """

    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = Field(..., description="Short topic label (3-100 chars)", min_length=1, max_length=200)
    summary: str = Field(default="", description="Brief description of the open thread")
    status: ThreadStatus = Field(default=ThreadStatus.OPEN)
    thread_type: ThreadType = Field(default=ThreadType.UNFINISHED)
    urgency: float = Field(default=0.5, ge=0.0, le=1.0, description="Urgency score 0.0-1.0")
    mentioned_at: float = Field(default_factory=time.time, description="Epoch when first mentioned")
    last_referenced: float = Field(default_factory=time.time, description="Epoch when last referenced")
    resolution_hint: str = Field(default="", description="Hint for what would resolve this thread")
    source_summary: str = Field(default="", description="ID or excerpt of source conversation summary")
    deadline_date: Optional[str] = Field(default=None, description="ISO date string for deadline threads")

    def to_embedding_text(self) -> str:
        """Generate text for embedding. Focuses on topic + summary for semantic matching."""
        parts = [f"Thread: {self.topic}"]
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        parts.append(f"Type: {self.thread_type.value}")
        if self.deadline_date:
            parts.append(f"Deadline: {self.deadline_date}")
        return " | ".join(parts)

    def to_metadata(self) -> dict:
        """Serialize for ChromaDB metadata storage (flat primitives only)."""
        return {
            "thread_id": self.thread_id,
            "topic": self.topic[:200],
            "summary": self.summary[:1000],
            "status": self.status.value,
            "thread_type": self.thread_type.value,
            "urgency": self.urgency,
            "mentioned_at": self.mentioned_at,
            "last_referenced": self.last_referenced,
            "resolution_hint": self.resolution_hint[:500],
            "source_summary": self.source_summary[:500],
            "deadline_date": self.deadline_date or "",
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "OpenThread":
        """Deserialize from ChromaDB metadata."""
        deadline = metadata.get("deadline_date", "")
        return cls(
            thread_id=metadata.get("thread_id", str(uuid.uuid4())),
            topic=metadata.get("topic", "Unknown"),
            summary=metadata.get("summary", ""),
            status=ThreadStatus(metadata.get("status", "open")),
            thread_type=ThreadType(metadata.get("thread_type", "unfinished")),
            urgency=float(metadata.get("urgency", 0.5)),
            mentioned_at=float(metadata.get("mentioned_at", time.time())),
            last_referenced=float(metadata.get("last_referenced", time.time())),
            resolution_hint=metadata.get("resolution_hint", ""),
            source_summary=metadata.get("source_summary", ""),
            deadline_date=deadline if deadline else None,
        )

    def to_dict(self) -> dict:
        """Full serialization for JSON export / testing."""
        return {
            "thread_id": self.thread_id,
            "topic": self.topic,
            "summary": self.summary,
            "status": self.status.value,
            "thread_type": self.thread_type.value,
            "urgency": self.urgency,
            "mentioned_at": self.mentioned_at,
            "last_referenced": self.last_referenced,
            "resolution_hint": self.resolution_hint,
            "source_summary": self.source_summary,
            "deadline_date": self.deadline_date,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OpenThread":
        """Deserialize from a full dict (inverse of to_dict)."""
        return cls(
            thread_id=data.get("thread_id", str(uuid.uuid4())),
            topic=data.get("topic", "Unknown"),
            summary=data.get("summary", ""),
            status=ThreadStatus(data.get("status", "open")),
            thread_type=ThreadType(data.get("thread_type", "unfinished")),
            urgency=float(data.get("urgency", 0.5)),
            mentioned_at=float(data.get("mentioned_at", time.time())),
            last_referenced=float(data.get("last_referenced", time.time())),
            resolution_hint=data.get("resolution_hint", ""),
            source_summary=data.get("source_summary", ""),
            deadline_date=data.get("deadline_date"),
        )

    def priority_score(self) -> float:
        """
        Calculate priority score for ranking threads.

        Formula: TYPE_PRIORITY[thread_type] * urgency * recency_decay
        where recency_decay = max(0.1, 1.0 - (days_since_mention / 14.0))
        """
        type_weight = TYPE_PRIORITY.get(self.thread_type, 0.4)
        days_since = (time.time() - self.last_referenced) / 86400.0
        recency_decay = max(0.1, 1.0 - (days_since / 14.0))
        return type_weight * self.urgency * recency_decay

    def is_stale(self, stale_days: int = 14, deadline_grace_hours: int = 48) -> bool:
        """Check if thread has gone stale (no reference in stale_days, or deadline passed)."""
        # Deadline-aware: if deadline_date has passed + grace period, it's stale
        if self.deadline_date:
            try:
                from datetime import datetime, timezone
                deadline_dt = datetime.fromisoformat(self.deadline_date)
                if deadline_dt.tzinfo is None:
                    deadline_dt = deadline_dt.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                hours_past = (now - deadline_dt).total_seconds() / 3600.0
                if hours_past >= deadline_grace_hours:
                    return True
            except (ValueError, TypeError):
                pass  # Malformed date, fall through to time-based check
        days_since = (time.time() - self.last_referenced) / 86400.0
        return days_since >= stale_days

    def mark_resolved(self, resolution: str = "") -> None:
        """Transition to RESOLVED status."""
        self.status = ThreadStatus.RESOLVED
        if resolution:
            self.resolution_hint = resolution
        self.last_referenced = time.time()

    def mark_stale(self) -> None:
        """Transition to STALE status."""
        self.status = ThreadStatus.STALE
