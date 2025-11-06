#core/memory_interface
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from utils.logging_utils import get_logger

logger = get_logger("memory_interface")
logger.debug("Memory_Interface.py is alive")


class MemoryInterface(ABC):
    @abstractmethod
    def store(self, content: Dict) -> str:
        """Store a memory object. Returns memory ID."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant memory chunks."""

    @abstractmethod
    def summarize(self) -> str:
        """Return a high-level summary."""


class MemoryType(Enum):
    EPISODIC = "episodic"      # Individual interactions
    SEMANTIC = "semantic"       # Extracted facts and knowledge
    PROCEDURAL = "procedural"   # How-to knowledge, patterns
    SUMMARY = "summary"         # Compressed episodic memories
    META = "meta"              # Memories about memory patterns
    FACT = "fact"

@dataclass
class MemoryNode:
    """Individual memory unit in the hierarchy"""
    id: str
    content: str
    type: MemoryType
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    importance_score: float = 0.5
    decay_rate: float = 0.1
    truth_score: float = 0.5
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    embeddings: Optional[List[float]] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self):
        safe_metadata = {
            k: (
                json.dumps(v) if isinstance(v, (list, dict)) else v
            )
            for k, v in self.metadata.items()
        }

        return {
            "id": self.id,
            "content": self.content,
            "type": self.type.value,  # you might want this too
            "timestamp": self.timestamp.isoformat(),
            "importance_score": self.importance_score,
            "decay_rate": self.decay_rate,
            "truth_score": self.truth_score,    # <-- persists
            "metadata": safe_metadata
        }

    @classmethod
    def from_dict(cls, data: Dict):
        metadata = data.get('metadata', {})
        # Parse tags from metadata
        tags = metadata.get('tags', [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except:
                tags = []

        # Read type from top-level data (matches to_dict output), fall back to metadata for compatibility
        type_str = data.get('type') or metadata.get('type', 'semantic')

        return cls(
            id=data["id"],
            content=data["content"],
            type=MemoryType(type_str),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            importance_score=data.get("importance_score", 0.5),
            decay_rate=data.get("decay_rate", 0.01),
            truth_score=data.get("truth_score", 0.5),   # <-- loads
            tags=tags,
            metadata=metadata
        )
