# memory/coordinator.py
"""
Refactored Memory Coordinator - thin orchestration layer.

This is the new modular version that delegates to specialized components:
- MemoryScorer: scoring and ranking
- MemoryStorage: persistence operations
- MemoryRetriever: retrieval operations
- ThreadManager: conversation thread tracking

The public API remains the same as the original MemoryCoordinator for
backwards compatibility.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
from collections import deque

from utils.logging_utils import get_logger
from utils.topic_manager import TopicManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from memory.fact_extractor import FactExtractor
from memory.memory_consolidator import MemoryConsolidator
from memory.memory_scorer import MemoryScorer
from memory.memory_storage import MemoryStorage
from memory.memory_retriever import MemoryRetriever
from memory.thread_manager import ThreadManager
from memory.hybrid_retriever import HybridRetriever
from processing.gate_system import MultiStageGateSystem

logger = get_logger("memory_coordinator")


class MemoryCoordinatorV2:
    """
    Refactored Memory Coordinator using modular components.

    This is a thin orchestration layer that delegates to specialized components
    while maintaining the same public API as the original MemoryCoordinator.
    """

    def __init__(
        self,
        corpus_manager,
        chroma_store: MultiCollectionChromaStore,
        gate_system: Optional[MultiStageGateSystem] = None,
        topic_manager=None,
        model_manager=None,
        time_manager=None
    ):
        # Store core dependencies
        self.corpus_manager = corpus_manager
        self.chroma_store = chroma_store
        self.gate_system = gate_system
        self.model_manager = model_manager
        self.time_manager = time_manager
        self.topic_manager = topic_manager or TopicManager()

        # State
        self.current_topic = "general"
        self.conversation_context = deque(maxlen=50)
        self.access_history: Dict[str, int] = {}
        self.interactions_since_consolidation = 0
        self.last_consolidation_time = datetime.now()

        # Session tracking
        try:
            self.session_start = self._now()
        except Exception:
            self.session_start = datetime.now()

        # Initialize consolidator
        try:
            from config.app_config import config as _app_cfg
            cfg_n = int(((_app_cfg.get('memory') or {}).get('summary_interval') or 10))
        except Exception:
            cfg_n = 10
        try:
            env_n = int(os.getenv('SUMMARY_EVERY_N', str(cfg_n)))
        except Exception:
            env_n = cfg_n

        self.consolidator = MemoryConsolidator(
            consolidation_threshold=max(1, int(env_n)),
            model_manager=model_manager
        )

        # Initialize fact extractor
        self.fact_extractor = FactExtractor()

        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(chroma_store=chroma_store)

        # Initialize modular components
        self.scorer = MemoryScorer(
            time_manager=time_manager,
            conversation_context=self.conversation_context
        )

        self.thread_manager = ThreadManager(
            corpus_manager=corpus_manager,
            topic_manager=self.topic_manager,
            time_manager=time_manager
        )

        self.storage = MemoryStorage(
            corpus_manager=corpus_manager,
            chroma_store=chroma_store,
            fact_extractor=self.fact_extractor,
            consolidator=self.consolidator,
            topic_manager=self.topic_manager,
            scorer=self.scorer,
            time_manager=time_manager
        )
        # Connect thread detection to storage
        self.storage._thread_detect_fn = self.thread_manager.detect_or_create_thread
        self.storage.conversation_context = self.conversation_context
        self.storage.current_topic = self.current_topic

        self.retriever = MemoryRetriever(
            corpus_manager=corpus_manager,
            chroma_store=chroma_store,
            gate_system=gate_system,
            scorer=self.scorer,
            hybrid_retriever=self.hybrid_retriever,
            time_manager=time_manager
        )
        self.retriever.conversation_context = self.conversation_context
        self.retriever.current_topic = self.current_topic

    # =========================================================================
    # Time helpers
    # =========================================================================

    def _now(self) -> datetime:
        if self.time_manager and hasattr(self.time_manager, "current"):
            return self.time_manager.current()
        return datetime.now()

    def _now_iso(self) -> str:
        if self.time_manager and hasattr(self.time_manager, "current_iso"):
            return self.time_manager.current_iso()
        return self._now().isoformat()

    # =========================================================================
    # Delegated to MemoryScorer
    # =========================================================================

    def _calculate_truth_score(self, query: str, response: str) -> float:
        return self.scorer.calculate_truth_score(query, response)

    def _calculate_importance_score(self, content: str) -> float:
        return self.scorer.calculate_importance_score(content)

    def _update_truth_scores_on_access(self, memories: List[Dict]) -> None:
        self.scorer.update_truth_scores_on_access(memories)

    def _rank_memories(self, memories: List[Dict], current_query: str) -> List[Dict]:
        return self.scorer.rank_memories(memories, current_query)

    # =========================================================================
    # Delegated to ThreadManager
    # =========================================================================

    def get_thread_context(self) -> Optional[Dict]:
        return self.thread_manager.get_thread_context()

    def _detect_topic_for_query(self, query: str) -> str:
        return self.thread_manager.detect_topic_for_query(query)

    def _detect_or_create_thread(self, query: str, is_heavy: bool) -> Dict:
        return self.thread_manager.detect_or_create_thread(query, is_heavy)

    # =========================================================================
    # Delegated to MemoryStorage
    # =========================================================================

    async def store_interaction(
        self,
        query: str,
        response: str,
        tags: Optional[List[str]] = None
    ) -> None:
        # Sync state before delegation
        self.storage.current_topic = self.current_topic
        await self.storage.store_interaction(query, response, tags)
        # Sync state back
        self.conversation_context = self.storage.conversation_context
        self.interactions_since_consolidation = self.storage.interactions_since_consolidation

    async def add_reflection(
        self,
        text: str,
        *,
        tags: Optional[List[str]] = None,
        source: str = "reflection",
        timestamp: Optional[datetime] = None
    ) -> bool:
        return await self.storage.add_reflection(text, tags=tags, source=source, timestamp=timestamp)

    async def _extract_and_store_facts(
        self,
        query: str,
        response: str,
        truth_score: float
    ) -> None:
        await self.storage.extract_and_store_facts(query, response, truth_score)

    async def _consolidate_and_store_summary(self) -> None:
        await self.storage.consolidate_and_store_summary()

    # =========================================================================
    # Delegated to MemoryRetriever
    # =========================================================================

    async def get_memories(
        self,
        query: str,
        limit: int = 20,
        topic_filter: Optional[str] = None
    ) -> List[Dict]:
        # Sync state before delegation
        self.retriever.current_topic = self.current_topic
        self.retriever.conversation_context = list(self.conversation_context)
        return await self.retriever.get_memories(query, limit, topic_filter)

    async def get_facts(self, query: str, limit: int = 8) -> List[Dict]:
        return await self.retriever.get_facts(query, limit)

    async def get_recent_facts(self, limit: int = 5) -> List[Dict]:
        return await self.retriever.get_recent_facts(limit)

    async def get_reflections(self, limit: int = 2) -> List[Dict]:
        return await self.retriever.get_reflections(limit)

    async def get_reflections_hybrid(self, query: str, limit: int = 3) -> List[Dict]:
        return await self.retriever.get_reflections_hybrid(query, limit)

    def get_summaries(self, limit: int = 3) -> List[Dict]:
        return self.retriever.get_summaries(limit)

    def get_summaries_hybrid(self, query: str, limit: int = 4) -> List[Dict]:
        return self.retriever.get_summaries_hybrid(query, limit)

    def get_dreams(self, limit: int = 2) -> List[Dict]:
        return self.retriever.get_dreams(limit)

    async def search_by_type(
        self,
        type_name: str,
        query: str = "",
        limit: int = 5
    ) -> List[Dict]:
        return await self.retriever.search_by_type(type_name, query, limit)

    async def get_semantic_top_memories(self, query: str, limit: int = 10) -> List[Dict]:
        # This method has complex logic - delegate to retriever's get_memories for now
        return await self.retriever.get_memories(query, limit)

    def _get_recent_conversations(self, k: int = 5) -> List[Dict]:
        return self.retriever._get_recent_conversations(k)

    async def _get_semantic_memories(self, query: str, n_results: int = 30) -> List[Dict]:
        return await self.retriever._get_semantic_memories(query, n_results)

    def _get_memory_key(self, memory: Dict) -> str:
        return self.retriever._get_memory_key(memory)

    def _parse_result(self, item: Dict, source: str, default_truth: float = 0.6) -> Dict:
        return self.retriever._parse_result(item, source, default_truth)

    # =========================================================================
    # Shutdown processing (keep in coordinator for now - complex orchestration)
    # =========================================================================

    async def process_shutdown_memory(
        self,
        session_conversations: Optional[List[Dict]] = None
    ) -> None:
        """Run end-of-session memory consolidation."""
        # This method has complex logic that coordinates multiple components
        # For now, keep it delegating to storage's consolidation
        await self.storage.consolidate_and_store_summary()

    async def run_shutdown_reflection(
        self,
        session_conversations: Optional[List[Dict]] = None,
        model_manager: Optional[object] = None
    ) -> Optional[str]:
        """Generate shutdown reflection summary."""
        # Complex orchestration - keep for now
        # TODO: Extract to consolidator module
        pass

    # =========================================================================
    # Debug and utility methods
    # =========================================================================

    async def debug_memory_state(self) -> Dict:
        """Return debug information about memory state."""
        return {
            "current_topic": self.current_topic,
            "conversation_context_size": len(self.conversation_context),
            "interactions_since_consolidation": self.interactions_since_consolidation,
            "session_start": self.session_start.isoformat() if self.session_start else None,
            "last_consolidation": self.last_consolidation_time.isoformat() if self.last_consolidation_time else None,
        }

    def _format_hierarchical_memory(self, memory) -> Dict:
        """Format a hierarchical memory node into standard dict format."""
        if hasattr(memory, 'to_dict'):
            return memory.to_dict()
        if isinstance(memory, dict):
            return memory
        return {"content": str(memory)}

    def _safe_detect_topic(self, text: str) -> str:
        """Safely detect topic with fallback."""
        try:
            return self._detect_topic_for_query(text)
        except Exception:
            return "general"


# Alias for backwards compatibility during transition
ModularMemoryCoordinator = MemoryCoordinatorV2
