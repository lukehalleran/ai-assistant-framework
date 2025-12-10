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


# =============================================================================
# New Contracts for Memory Coordinator Refactor
# =============================================================================

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryScorerProtocol(Protocol):
    """Contract for memory scoring and ranking operations"""

    def calculate_truth_score(self, query: str, response: str) -> float:
        """Calculate truth/reliability score for a memory"""
        ...

    def calculate_importance_score(self, content: str) -> float:
        """Calculate importance score based on content analysis"""
        ...

    def rank_memories(self, memories: List[Dict], query: str) -> List[Dict]:
        """Rank memories by composite score for given query"""
        ...

    def update_truth_scores_on_access(self, memories: List[Dict]) -> None:
        """Boost truth scores when memories are accessed"""
        ...


@runtime_checkable
class MemoryStorageProtocol(Protocol):
    """Contract for memory storage operations"""

    async def store_interaction(
        self,
        query: str,
        response: str,
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Store a conversation interaction.

        Returns:
            str: Database ID (UUID) of the stored memory, or None if storage failed
        """
        ...

    async def add_reflection(
        self,
        text: str,
        *,
        tags: Optional[List[str]] = None,
        source: str = "reflection",
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Store a reflection memory"""
        ...

    async def extract_and_store_facts(
        self,
        query: str,
        response: str,
        truth_score: float
    ) -> None:
        """Extract facts from conversation and store them"""
        ...


@runtime_checkable
class MemoryRetrieverProtocol(Protocol):
    """Contract for memory retrieval operations"""

    async def get_memories(
        self,
        query: str,
        limit: int = 20,
        topic_filter: Optional[str] = None
    ) -> List[Dict]:
        """Main retrieval pipeline - get relevant memories for query"""
        ...

    async def get_facts(self, query: str, limit: int = 8) -> List[Dict]:
        """Retrieve semantic facts relevant to query"""
        ...

    async def get_recent_facts(self, limit: int = 5) -> List[Dict]:
        """Retrieve most recent facts"""
        ...

    async def get_reflections(self, limit: int = 2) -> List[Dict]:
        """Retrieve recent reflections"""
        ...

    async def get_reflections_hybrid(
        self,
        query: str,
        limit: int = 3
    ) -> List[Dict]:
        """Retrieve reflections using hybrid recent+semantic search"""
        ...

    def get_summaries(self, limit: int = 3) -> List[Dict]:
        """Retrieve recent summaries"""
        ...

    def get_summaries_hybrid(self, query: str, limit: int = 4) -> List[Dict]:
        """Retrieve summaries using hybrid search"""
        ...

    def get_dreams(self, limit: int = 2) -> List[Dict]:
        """Retrieve dream memories"""
        ...

    async def search_by_type(
        self,
        type_name: str,
        query: str = "",
        limit: int = 5
    ) -> List[Dict]:
        """Search memories by type"""
        ...


@runtime_checkable
class ThreadManagerProtocol(Protocol):
    """Contract for conversation thread management"""

    def get_thread_context(self) -> Optional[Dict]:
        """Get current thread context if active"""
        ...

    def detect_or_create_thread(self, query: str, is_heavy: bool) -> Dict:
        """Detect existing thread or create new one"""
        ...

    def detect_topic_for_query(self, query: str) -> str:
        """Extract topic from query"""
        ...


@runtime_checkable
class MemoryConsolidatorProtocol(Protocol):
    """Contract for memory consolidation operations"""

    async def process_shutdown_memory(
        self,
        session_conversations: Optional[List[Dict]] = None
    ) -> None:
        """Run end-of-session memory consolidation"""
        ...

    async def run_shutdown_reflection(
        self,
        session_conversations: Optional[List[Dict]] = None,
        model_manager: Optional[object] = None
    ) -> Optional[str]:
        """Generate shutdown reflection summary"""
        ...

    async def consolidate_and_store_summary(self) -> None:
        """Consolidate recent conversations into summary"""
        ...
