"""
# memory/memory_coordinator.py

Module Contract
- Purpose: Central coordinator for conversational memory. Persists each turn, retrieves relevant memories (recent + semantic), runs shutdown reflections, and synthesizes/stores block summaries at shutdown. ENHANCED: Now integrates UserProfile for structured fact storage with categorization.
- Inputs:
  - store_interaction(query, response, tags?, session_id=?, provenance=?): adds a turn to corpus + Chroma (with provenance metadata) [ENHANCED 2026-03-26]
  - session_id (property): stable session identifier derived from session_start timestamp [NEW 2026-03-26]
  - get_memories(query, limit, topic_filter?): unified retrieval/gating/ranking
  - get_unresolved_threads(max_results): returns top-priority open threads for prompt surfacing
  - process_shutdown_memory(): summarize blocks (size N) and extract end‑of‑session facts and procedural skills → UPDATED: also populates UserProfile with categorized facts
  - run_shutdown_reflection(...): generate a short reflection at session end
- Outputs:
  - Lists of normalized memory dicts with scores/metadata; new summary/reflection/fact nodes in storage.
  - UPDATED: UserProfile populated with categorized facts (fitness, identity, career, etc.) saved to data/user_profile.json
- Key collaborators:
  - CorpusManager (JSON short‑term store)
  - MultiCollectionChromaStore (semantic collections)
  - MemoryConsolidator (LLM summarization API, may be bypassed for micro‑summaries)
  - MultiStageGateSystem (cosine/rerank gating)
  - UserProfile (NEW: structured user fact storage with 12 categories)
  - ThreadStore (open thread persistence; initialized internally, passed to ShutdownProcessor)
- Side effects:
  - Writes to corpus JSON and Chroma collections; updates access/score metadata.
  - UPDATED: Writes to data/user_profile.json with categorized user facts at shutdown
- Async behavior:
  - Most public methods are async to coordinate with model calls and gating.
"""
from __future__ import annotations

import uuid
from typing import List, Dict, Optional
from datetime import datetime
from collections import deque

from utils.logging_utils import get_logger
from utils.topic_manager import TopicManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from memory.fact_extractor import FactExtractor
from memory.memory_consolidator import MemoryConsolidator
from memory.memory_scorer import MemoryScorer
from memory.hybrid_retriever import HybridRetriever
from memory.memory_storage import MemoryStorage
from memory.memory_retriever import MemoryRetriever
from memory.thread_manager import ThreadManager
from memory.user_profile import UserProfile
from processing.gate_system import MultiStageGateSystem

logger = get_logger("memory_coordinator")


# ---------------------------
# Memory Coordinator
# ---------------------------

class MemoryCoordinator:
    """
    Single funnel for short/long-term memory fetch/store:
    - Stores interactions to corpus + Chroma with truth/importance/topic metadata
    - Retrieves from corpus (recent) and Chroma collections (semantic)
    - Combines → gates → ranks → returns top-K
    - Handles shutdown consolidation and fact extraction
    """

    def __init__(self,
                 corpus_manager,
                 chroma_store: MultiCollectionChromaStore,
                 gate_system: Optional[MultiStageGateSystem] = None,
                 topic_manager=None,
                 model_manager=None,
                 time_manager=None):
        self.corpus_manager = corpus_manager
        self.chroma_store = chroma_store
        self.gate_system = gate_system
        self.current_topic = "general"
        self.fact_extractor = FactExtractor()
        # Determine summary cadence from config or env (fallback to 10)
        try:
            from config.app_config import config as _app_cfg
            cfg_n = int(((_app_cfg.get('memory') or {}).get('summary_interval') or 10))
        except (ImportError, AttributeError, ValueError, TypeError) as e:
            logger.debug(f"[MemoryCoordinator] Could not load summary_interval from config: {e}")
            cfg_n = 10
        try:
            import os as _os
            env_n = int(_os.getenv('SUMMARY_EVERY_N', str(cfg_n)))
        except (ValueError, TypeError) as e:
            logger.debug(f"[MemoryCoordinator] Could not parse SUMMARY_EVERY_N env var: {e}")
            env_n = cfg_n
        self.consolidator = MemoryConsolidator(
            consolidation_threshold=max(1, int(env_n)),
            model_manager=model_manager
        )
        self.model_manager = model_manager
        self.topic_manager = topic_manager or TopicManager()
        self.conversation_context = deque(maxlen=50)
        # Initialize hybrid retriever for improved semantic search
        self.hybrid_retriever = HybridRetriever(chroma_store=chroma_store)
        self.access_history: Dict[str, int] = {}  # Track memory access for truth updates
        self._ACCESS_HISTORY_MAX_SIZE = 500  # Prevent unbounded growth
        self.interactions_since_consolidation = 0
        self.time_manager = time_manager
        self.last_consolidation_time = datetime.now()
        # Mark the start of this application session; used to scope shutdown reflections/facts
        try:
            self.session_start = self._now()
        except (AttributeError, TypeError) as e:
            logger.debug(f"[MemoryCoordinator] _now() failed during init: {e}")
            from datetime import datetime as _dt
            self.session_start = _dt.now()

        # Initialize modular components for delegation
        self.scorer = MemoryScorer(
            time_manager=time_manager,
            conversation_context=self.conversation_context
        )

        self.thread_manager = ThreadManager(
            corpus_manager=corpus_manager,
            topic_manager=self.topic_manager,
            time_manager=time_manager
        )

        # Initialize knowledge graph (NetworkX-based entity relationship graph)
        self.graph_memory = None
        self.entity_resolver = None
        try:
            from config.app_config import (
                KNOWLEDGE_GRAPH_ENABLED, KNOWLEDGE_GRAPH_PERSIST_PATH,
                KNOWLEDGE_GRAPH_AUTO_SAVE_THRESHOLD, KNOWLEDGE_GRAPH_ALIASES_PATH,
            )
            if KNOWLEDGE_GRAPH_ENABLED:
                from memory.graph_memory import GraphMemory
                from memory.entity_resolver import EntityResolver
                self.graph_memory = GraphMemory(persist_path=KNOWLEDGE_GRAPH_PERSIST_PATH)
                self.graph_memory._auto_save_threshold = KNOWLEDGE_GRAPH_AUTO_SAVE_THRESHOLD
                self.entity_resolver = EntityResolver(
                    graph_memory=self.graph_memory,
                    aliases_path=KNOWLEDGE_GRAPH_ALIASES_PATH,
                )
                logger.info(
                    f"[MemoryCoordinator] Knowledge graph initialized: "
                    f"{self.graph_memory.node_count()} nodes, {self.graph_memory.edge_count()} edges"
                )
        except Exception as e:
            logger.debug(f"[MemoryCoordinator] Knowledge graph init failed (non-fatal): {e}")

        # Initialize fact verification gate (pre-storage conflict checking)
        self.fact_verifier = None
        try:
            from config.app_config import FACT_VERIFICATION_ENABLED
            if FACT_VERIFICATION_ENABLED:
                from memory.fact_verification import FactVerifier
                self.fact_verifier = FactVerifier(
                    chroma_store=chroma_store,
                    model_manager=model_manager,
                )
                logger.debug("[MemoryCoordinator] Fact verification gate initialized")
        except Exception as e:
            logger.debug(f"[MemoryCoordinator] Fact verifier init failed (non-fatal): {e}")

        self._storage = MemoryStorage(
            corpus_manager=corpus_manager,
            chroma_store=chroma_store,
            fact_extractor=self.fact_extractor,
            consolidator=self.consolidator,
            topic_manager=self.topic_manager,
            scorer=self.scorer,
            time_manager=time_manager,
            graph_memory=self.graph_memory,
            entity_resolver=self.entity_resolver,
            fact_verifier=self.fact_verifier,
        )
        # Connect thread detection to storage
        self._storage._thread_detect_fn = self.thread_manager.detect_or_create_thread
        self._storage.conversation_context = self.conversation_context

        self._retriever = MemoryRetriever(
            corpus_manager=corpus_manager,
            chroma_store=chroma_store,
            gate_system=gate_system,
            scorer=self.scorer,
            hybrid_retriever=self.hybrid_retriever,
            time_manager=time_manager
        )
        self._retriever.conversation_context = list(self.conversation_context)

        # Initialize user profile for structured fact storage
        self.user_profile = UserProfile()

        # Initialize thread store for proactive thread surfacing
        self.thread_store = None
        try:
            from config.app_config import THREAD_SURFACING_ENABLED
            if THREAD_SURFACING_ENABLED:
                from memory.thread_store import ThreadStore
                self.thread_store = ThreadStore(chroma_store=chroma_store)
                logger.debug("[MemoryCoordinator] Thread store initialized")
        except Exception as e:
            logger.debug(f"[MemoryCoordinator] Thread store init failed (non-fatal): {e}")

        # Initialize proactive context surfacer
        self.context_surfacer = None
        try:
            from config.app_config import PROACTIVE_SURFACING_ENABLED
            if PROACTIVE_SURFACING_ENABLED and self.graph_memory and self.entity_resolver:
                from memory.context_surfacer import ContextSurfacer
                self.context_surfacer = ContextSurfacer(
                    graph_memory=self.graph_memory,
                    entity_resolver=self.entity_resolver,
                    model_manager=model_manager,
                )
                logger.debug("[MemoryCoordinator] Context surfacer initialized")
        except Exception as e:
            logger.debug(f"[MemoryCoordinator] Context surfacer init failed (non-fatal): {e}")

        # Initialize claim index for memory staleness tracking
        self.claim_index = None
        try:
            from config.app_config import STALENESS_ENABLED, STALENESS_INDEX_PATH
            if STALENESS_ENABLED:
                from memory.claim_tracker import ClaimIndex
                self.claim_index = ClaimIndex(persist_path=STALENESS_INDEX_PATH)
                logger.debug(
                    "[MemoryCoordinator] Claim index initialized: %d claims, %d docs",
                    self.claim_index.total_claims, self.claim_index.total_documents,
                )
        except Exception as e:
            logger.debug(f"[MemoryCoordinator] Claim index init failed (non-fatal): {e}")

        # Initialize shutdown processor for end-of-session consolidation
        from memory.shutdown_processor import ShutdownProcessor
        self._shutdown = ShutdownProcessor(
            corpus_manager=corpus_manager,
            chroma_store=chroma_store,
            consolidator=self.consolidator,
            fact_extractor=self.fact_extractor,
            model_manager=model_manager,
            user_profile=self.user_profile,
            storage=self._storage,
            session_start=self.session_start,
            memory_coordinator=self,
            thread_store=self.thread_store,
            claim_index=self.claim_index,
        )

        logger.debug("[MemoryCoordinator] All components initialized")

    # --------- time helpers (prefer TimeManager) ---------
    def _now(self):
        from utils.time_manager import now_from
        return now_from(self.time_manager)

    def _now_iso(self):
        from utils.time_manager import now_iso_from
        return now_iso_from(self.time_manager)

    @property
    def session_id(self) -> str:
        """Stable session identifier derived from session_start timestamp."""
        if isinstance(self.session_start, datetime):
            return self.session_start.isoformat()
        return str(self.session_start)

    # ---------------------------
    # Scoring helpers (delegated to MemoryScorer)
    # ---------------------------

    def _calculate_truth_score(self, query: str, response: str) -> float:
        """Delegates to MemoryScorer."""
        self.scorer.conversation_context = list(self.conversation_context)
        return self.scorer.calculate_truth_score(query, response)

    def _calculate_importance_score(self, content: str) -> float:
        """Delegates to MemoryScorer."""
        return self.scorer.calculate_importance_score(content)

    def _update_truth_scores_on_access(self, memories: List[Dict]):
        """Delegates core scoring to MemoryScorer, then caps access_history growth."""
        self.scorer.update_truth_scores_on_access(memories)

        # Sync scorer's access counts into coordinator's history for size capping
        for mem in memories:
            mem_id = mem.get('id')
            if mem_id:
                self.access_history[mem_id] = self.access_history.get(mem_id, 0) + 1

        # Prevent unbounded growth of access_history
        if len(self.access_history) > self._ACCESS_HISTORY_MAX_SIZE:
            sorted_entries = sorted(self.access_history.items(), key=lambda x: x[1], reverse=True)
            self.access_history = dict(sorted_entries[:self._ACCESS_HISTORY_MAX_SIZE // 2])


    async def get_recent_facts(self, limit: int = 5) -> List[Dict]:
        """Fetch the most recent facts by timestamp.

        Delegates to MemoryRetriever component.
        """
        return await self._retriever.get_recent_facts(limit)

    async def get_facts(self, query: str, limit: int = 8) -> List[Dict]:
        """Get facts for query.

        Delegates to MemoryRetriever component.
        """
        return await self._retriever.get_facts(query, limit)





    # ---------------------------
    # Public API: store
    # ---------------------------

    # In memory_coordinator.py, update the store_interaction method
    # In memory_coordinator.py, update the store_interaction method with more debugging
    async def store_interaction(
        self,
        query: str,
        response: str,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        provenance: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Persist a turn in both corpus & Chroma with computed metadata.

        Delegates to MemoryStorage component.

        Returns:
            str: Database ID (UUID) of the stored memory, or None if storage failed
        """
        # Sync state before delegation
        self._storage.current_topic = self.current_topic
        self._storage.conversation_context = self.conversation_context

        memory_id = await self._storage.store_interaction(
            query, response, tags,
            session_id=session_id,
            provenance=provenance,
        )

        # Sync state back from storage
        self.conversation_context = self._storage.conversation_context
        self.interactions_since_consolidation = self._storage.interactions_since_consolidation

        # --- Lightweight per-turn thread resolution (pure regex, ~1ms) ---
        if self.thread_store and query:
            try:
                from memory.thread_store import check_quick_resolutions, _COMPLETION_SIGNALS
                # Fast pre-check: skip DB query if no completion signal in message
                if _COMPLETION_SIGNALS.search(query.lower()):
                    open_threads = self.thread_store.list_open_threads()
                    if open_threads:
                        resolved_ids = check_quick_resolutions(query, open_threads)
                        for tid in resolved_ids:
                            self.thread_store.resolve_thread(
                                tid, "auto-resolved: completion signal in user message"
                            )
            except Exception as e:
                logger.debug(f"[MemoryCoordinator] Quick thread resolution failed (non-fatal): {e}")

        return memory_id

    async def _consolidate_and_store_summary(self):
        """Consolidate recent memories and store the summary.

        Delegates to MemoryStorage component.
        """
        await self._storage.consolidate_and_store_summary()
        self.last_consolidation_time = self._storage.last_consolidation_time or datetime.now()


    async def get_reflections(self, limit: int = 2):
        """Fetch recent reflections.

        Delegates to MemoryRetriever component.
        """
        return await self._retriever.get_reflections(limit)

    async def get_reflections_hybrid(self, query: str, limit: int = 3) -> List[Dict]:
        """Hybrid retrieval for reflections.

        Delegates to MemoryRetriever component.
        """
        return await self._retriever.get_reflections_hybrid(query, limit)

    async def run_shutdown_reflection(self, session_conversations: list[dict] | None = None,
                                    session_summaries: list[dict | str] | None = None) -> bool:
        """Generate and store an end-of-session reflection. Delegates to ShutdownProcessor."""
        return await self._shutdown.run_shutdown_reflection(session_conversations, session_summaries)

    async def _extract_and_store_facts(self, query: str, response: str, truth_score: float):
        """Extract and store facts from a turn.

        Delegates to MemoryStorage component.
        """
        await self._storage.extract_and_store_facts(query, response, truth_score)

    # ---------------------------
    # Public API: retrieve
    # ---------------------------

    async def get_memories(self, query: str, limit: int = 20, topic_filter: Optional[str] = None) -> List[Dict]:
        """Unified retrieval pipeline.

        Delegates to MemoryRetriever component.
        """
        # Sync state before delegation
        self._retriever.current_topic = self.current_topic
        self._retriever.conversation_context = list(self.conversation_context)
        return await self._retriever.get_memories(query, limit, topic_filter)

    # ---------------------------
    # Shutdown processing
    # ---------------------------

    async def process_shutdown_memory(self, session_conversations: list[dict] | None = None):
        """Create any due summaries in fixed-size blocks and run fact extraction.

        Delegates to ShutdownProcessor.
        """
        await self._shutdown.process_shutdown_memory(session_conversations)

    # ---------------------------
    # Helper methods
    # ---------------------------

    def _get_memory_key(self, memory: Dict) -> str:
        """Generate unique key for deduplication"""
        # Include ID or timestamp to avoid false duplicates
        mem_id = memory.get('id', '')
        timestamp = memory.get('timestamp', memory.get('metadata', {}).get('timestamp', ''))
        q = (memory.get('query', '') or '')[:40]  # Shorter to leave room for ID
        r = (memory.get('response', '') or '')[:40]

        # Use multiple identifiers to ensure uniqueness
        if mem_id:
            return f"id:{mem_id}__{q[:20]}__{r[:20]}"
        elif timestamp:
            return f"ts:{timestamp}__{q[:20]}__{r[:20]}"
        else:
            return f"hash:{hash(str(memory))}__{q[:30]}__{r[:30]}"

    def _safe_detect_topic(self, text: str) -> str:
        try:
            if hasattr(self.topic_manager, 'detect_topic'):
                return self.topic_manager.detect_topic(text) or 'general'
        except Exception:
            pass
        return 'general'

    def _format_hierarchical_memory(self, memory) -> Dict:
        """Adapt hierarchical memory node to flat dict for ranker."""
        content = getattr(memory, 'content', '')
        parts = content.split('\nAssistant: ')
        q = content[:100]
        r = "[Could not parse response]"
        if len(parts) == 2:
            q = parts[0].replace("User: ", "").strip()
            r = parts[1].strip()

        truth = getattr(memory, 'truth_score', None)
        if truth is None:
            truth = (getattr(memory, 'metadata', {}) or {}).get('truth_score', 0.5)

        ts = getattr(memory, 'timestamp', datetime.now())
        return {
            'id': getattr(memory, 'id', f"hier::{uuid.uuid4().hex[:8]}"),
            'query': q,
            'response': r,
            'content': content or f"User: {q}\nAssistant: {r}",
            'timestamp': ts if isinstance(ts, datetime) else datetime.now(),
            'source': 'hierarchical',
            'collection': 'hierarchical',
            'metadata': getattr(memory, 'metadata', {}) or {},
            'tags': getattr(memory, 'tags', []),
            'truth_score': float(truth or 0.5),
            'relevance_score': float(getattr(memory, 'score', 0.5)),
            'importance_score': float((getattr(memory, 'metadata', {}) or {}).get('importance_score', 0.5)),
        }
    async def debug_memory_state(self):
        """Debug method to check what's in memory and get statistics"""
        stats = {
            'corpus_entries': len(self.corpus_manager.corpus),
            'non_summary_entries': len([e for e in self.corpus_manager.corpus if "@summary" not in e.get("tags", [])]),
            'summaries': len(self.corpus_manager.get_summaries()),
            'chroma_collections': {}
        }

        for name, collection in self.chroma_store.collections.items():
            stats['chroma_collections'][name] = collection.count()

        logger.info(f"[DEBUG] Memory state: {stats}")
        return stats
    # ---------------------------
    # Optional: summaries / dreams passthroughs
    # ---------------------------

    def get_summaries(self, limit: int = 3) -> List[Dict]:
        """Get recent summaries.

        Delegates to MemoryRetriever component.
        """
        return self._retriever.get_summaries(limit)

    def get_summaries_hybrid(self, query: str, limit: int = 4) -> List[Dict]:
        """Hybrid retrieval for summaries.

        Delegates to MemoryRetriever component.
        """
        return self._retriever.get_summaries_hybrid(query, limit)

    def get_dreams(self, limit: int = 2) -> List[Dict]:
        """Get dreams.

        Delegates to MemoryRetriever component.
        """
        return self._retriever.get_dreams(limit)

    async def store_skill(self, skill) -> Optional[str]:
        """Store a procedural skill with semantic deduplication.

        Delegates to MemoryStorage component.
        """
        return await self._storage.store_skill(skill)

    async def get_skills(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve procedural skills relevant to query.

        Delegates to MemoryRetriever component.
        """
        return await self._retriever.get_skills(query, limit)

    def get_unresolved_threads(self, max_results: int = 3) -> List[Dict]:
        """Get top priority unresolved threads for session surfacing.

        Delegates to ThreadStore component.

        Returns:
            List of thread dicts with topic, summary, thread_type, urgency, deadline_date
        """
        if not self.thread_store:
            return []
        try:
            threads = self.thread_store.get_top_threads(max_results=max_results)
            return [t.to_dict() for t in threads]
        except Exception as e:
            logger.debug(f"[MemoryCoordinator] get_unresolved_threads failed: {e}")
            return []

    # ---------------------------
    # Delegation methods for sub-components
    # ---------------------------

    def _parse_result(self, item: Dict, source: str, default_truth: float = 0.6) -> Dict:
        """Delegates to MemoryRetriever."""
        return self._retriever._parse_result(item, source, default_truth)

    def _rank_memories(self, memories: List[Dict], current_query: str,
                       current_topic: Optional[str] = None,
                       is_meta_conversational: bool = False,
                       weight_overrides: Optional[Dict] = None) -> List[Dict]:
        """Delegates to MemoryScorer."""
        return self.scorer.rank_memories(
            memories, current_query, current_topic,
            is_meta_conversational, weight_overrides,
        )

    async def add_reflection(self, text: str, **kwargs) -> bool:
        """Delegates to MemoryStorage."""
        return await self._storage.add_reflection(text, **kwargs)

    def get_thread_context(self) -> Optional[Dict]:
        """Delegates to ThreadManager."""
        return self.thread_manager.get_thread_context()

    def _detect_or_create_thread(self, query: str, is_heavy: bool) -> Dict:
        """Delegates to ThreadManager."""
        return self.thread_manager.detect_or_create_thread(query, is_heavy)

    async def get_semantic_top_memories(self, query: str, limit: int = 10) -> List[Dict]:
        """Delegates to MemoryRetriever."""
        return await self._retriever.get_semantic_top_memories(query, limit)

    async def search_by_type(self, type_name: str, query: str = "", limit: int = 5) -> List[Dict]:
        """Delegates to MemoryRetriever."""
        return await self._retriever.search_by_type(type_name, query, limit)

    def _get_recent_conversations(self, k: int = 5) -> List[Dict]:
        """Delegates to MemoryRetriever."""
        return self._retriever._get_recent_conversations(k)

    async def _get_semantic_memories(self, query: str, n_results: int = 30) -> List[Dict]:
        """Delegates to MemoryRetriever."""
        return await self._retriever._get_semantic_memories(query, n_results)

    async def _get_meta_conversational_memories(self, query: str, limit: int = 20,
                                                 topic_filter: Optional[str] = None) -> List[Dict]:
        """Delegates to MemoryRetriever."""
        return await self._retriever._get_meta_conversational_memories(query, limit, topic_filter)
