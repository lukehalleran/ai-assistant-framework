# memory/memory_storage.py
"""
Memory storage module.

Implements the MemoryStorageProtocol contract for persisting memories.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
from collections import deque

from utils.logging_utils import get_logger

logger = get_logger("memory_storage")

# Environment configuration
FACTS_EXTRACT_EACH_TURN = os.getenv("FACTS_EXTRACT_EACH_TURN", "0").strip().lower() not in ("0", "false", "no", "off")
SUMMARIZE_AT_SHUTDOWN_ONLY = os.getenv("SUMMARIZE_AT_SHUTDOWN_ONLY", "1").strip().lower() not in ("0", "false", "no", "off")


class MemoryStorage:
    """
    Memory storage operations.

    Implements MemoryStorageProtocol contract.
    """

    def __init__(
        self,
        corpus_manager,
        chroma_store,
        fact_extractor,
        consolidator=None,
        topic_manager=None,
        scorer=None,
        time_manager=None,
    ):
        """
        Initialize MemoryStorage.

        Args:
            corpus_manager: CorpusManager for JSON persistence
            chroma_store: MultiCollectionChromaStore for vector storage
            fact_extractor: FactExtractor for extracting facts
            consolidator: Optional MemoryConsolidator for summarization
            topic_manager: Optional TopicManager for topic detection
            scorer: Optional MemoryScorer for calculating scores
            time_manager: Optional TimeManager for timestamps
        """
        self.corpus_manager = corpus_manager
        self.chroma_store = chroma_store
        self.fact_extractor = fact_extractor
        self.consolidator = consolidator
        self.topic_manager = topic_manager
        self.scorer = scorer
        self.time_manager = time_manager

        # State
        self.conversation_context: deque = deque(maxlen=50)
        self.current_topic: str = "general"
        self.interactions_since_consolidation: int = 0
        self.last_consolidation_time: Optional[datetime] = None

        # Thread tracking (will be delegated to ThreadManager later)
        self._thread_detect_fn = None  # Set by coordinator

    def _now(self) -> datetime:
        """Get current time from time_manager or datetime.now()"""
        if self.time_manager and hasattr(self.time_manager, 'current'):
            return self.time_manager.current()
        return datetime.now()

    def _now_iso(self) -> str:
        """Get current time as ISO string"""
        if self.time_manager and hasattr(self.time_manager, 'current_iso'):
            return self.time_manager.current_iso()
        return self._now().isoformat()

    def _calculate_truth_score(self, query: str, response: str) -> float:
        """Calculate truth score using scorer or fallback"""
        if self.scorer and hasattr(self.scorer, 'calculate_truth_score'):
            return self.scorer.calculate_truth_score(query, response)
        # Fallback
        score = 0.5
        if len(response or "") > 200:
            score += 0.1
        if '?' in (query or ""):
            score += 0.1
        return min(score, 1.0)

    def _calculate_importance_score(self, content: str) -> float:
        """Calculate importance score using scorer or fallback"""
        if self.scorer and hasattr(self.scorer, 'calculate_importance_score'):
            return self.scorer.calculate_importance_score(content)
        # Fallback
        score = 0.5
        if len(content or "") > 200:
            score += 0.1
        if '?' in content:
            score += 0.1
        return min(score, 1.0)

    async def store_interaction(
        self,
        query: str,
        response: str,
        tags: Optional[List[str]] = None
    ) -> None:
        """Persist a turn in both corpus & Chroma with computed metadata"""
        try:
            # Detect heavy topic before anything else
            from utils.query_checker import _is_heavy_topic_heuristic
            is_heavy = _is_heavy_topic_heuristic(query)

            # Thread detection (if available)
            thread_info = {}
            if self._thread_detect_fn:
                thread_info = self._thread_detect_fn(query, is_heavy)

            # Add to corpus (JSON) with stable timestamp and thread metadata
            self.corpus_manager.add_entry(
                query, response, tags or [], timestamp=self._now(),
                thread_id=thread_info.get("thread_id"),
                thread_depth=thread_info.get("depth"),
                thread_started=thread_info.get("started"),
                thread_topic=thread_info.get("topic"),
                is_heavy_topic=is_heavy,
                topic=self.current_topic
            )

            # Update conversation context
            self.conversation_context.append({
                "query": query,
                "response": response,
                "timestamp": self._now()
            })

            # Topic detection
            primary_topic = "general"
            if self.topic_manager:
                if hasattr(self.topic_manager, "update_from_user_input"):
                    topics = self.topic_manager.update_from_user_input(query)
                    primary_topic = topics[0] if topics else "general"
                elif hasattr(self.topic_manager, "detect_topic"):
                    primary_topic = self.topic_manager.detect_topic(f"{query} {response}")

            # Ensure tags list exists and includes the topic
            tags = tags or []
            if f"topic:{primary_topic}" not in tags:
                tags.append(f"topic:{primary_topic}")

            # Calculate scores
            truth_score = self._calculate_truth_score(query, response)
            importance_score = self._calculate_importance_score(f"{query} {response}")

            # Create metadata with proper type checking
            raw_metadata = {
                "timestamp": self._now_iso(),
                "query": query if query else "",
                "response": response if response else "",
                "tags": ",".join(tags) if tags else "",
                "topic": primary_topic or "general",
                "truth_score": float(truth_score),
                "importance_score": float(importance_score),
                "access_count": 0,
                "last_accessed": self._now_iso(),
            }

            # Filter out None values and ensure correct types
            clean_metadata = {}
            for k, v in raw_metadata.items():
                if v is not None:
                    if isinstance(v, (str, int, float, bool)):
                        clean_metadata[k] = v
                    else:
                        clean_metadata[k] = str(v)

            # Store in Chroma
            memory_id = self.chroma_store.add_conversation_memory(query, response, clean_metadata)

            # Facts extraction (by default only at shutdown)
            if FACTS_EXTRACT_EACH_TURN:
                if (response or "").strip() and len(response) >= 8:
                    await self.extract_and_store_facts(query, response, truth_score)

            # Consolidation (mid-session unless disabled)
            if not SUMMARIZE_AT_SHUTDOWN_ONLY and self.consolidator:
                self.interactions_since_consolidation += 1
                if self.interactions_since_consolidation >= self.consolidator.consolidation_threshold:
                    await self.consolidate_and_store_summary()
                    self.interactions_since_consolidation = 0

            logger.debug(f"[MemoryStorage] Stored memory {memory_id} (topic={primary_topic}, truth={truth_score:.2f})")

        except Exception as e:
            logger.error(f"Error storing interaction: {e}")
            import traceback
            traceback.print_exc()

    async def add_reflection(
        self,
        text: str,
        *,
        tags: Optional[List[str]] = None,
        source: str = "reflection",
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Store a reflection memory"""
        if not text:
            return False

        ts = timestamp or self._now()
        tags = list(tags or [])
        if "type:reflection" not in tags:
            tags.append("type:reflection")
        if source and f"source:{source}" not in tags:
            tags.append(f"source:{source}")

        # 1) Corpus (reuse add_summary schema for compatibility)
        try:
            if hasattr(self.corpus_manager, "add_summary"):
                self.corpus_manager.add_summary({
                    "content": text,
                    "timestamp": ts,
                    "type": "reflection",
                    "tags": tags
                })
        except Exception as e:
            logger.debug(f"[MemoryStorage] Corpus add_summary failed: {e}")

        # 2) Chroma (semantic)
        try:
            if hasattr(self.chroma_store, "add_to_collection"):
                md = {
                    "timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                    "type": "reflection",
                    "tags": ",".join(tags),
                    "source": source,
                    "importance_score": 0.7,
                }
                # Ensure collection exists
                if (getattr(self.chroma_store, "collections", None) is not None
                    and "reflections" not in self.chroma_store.collections
                    and hasattr(self.chroma_store, "create_collection")):
                    try:
                        self.chroma_store.create_collection("reflections")
                    except Exception:
                        pass
                self.chroma_store.add_to_collection("reflections", text, md)
        except Exception as e:
            logger.debug(f"[MemoryStorage] Chroma add_to_collection failed: {e}")

        return True

    async def extract_and_store_facts(
        self,
        query: str,
        response: str,
        truth_score: float
    ) -> None:
        """Extract and store facts from a turn"""
        try:
            logger.debug(f"[MemoryStorage] Extracting facts from query: {query[:100]}...")
            facts = await self.fact_extractor.extract_facts(query, "") or []
            total = len(facts)
            logger.debug(f"[MemoryStorage] Extracted {total} facts (raw)")

            def _to_dict(item):
                if isinstance(item, dict):
                    return {"content": item.get("content", ""), "metadata": item.get("metadata", {})}
                content = getattr(item, "content", None)
                metadata = getattr(item, "metadata", None)
                if content is not None or metadata is not None:
                    return {"content": content or "", "metadata": metadata or {}}
                return {"content": str(item or ""), "metadata": {}}

            stored = 0
            for idx, raw in enumerate(facts, 1):
                safe = _to_dict(raw)
                md = safe.get("metadata", {}) or {}

                # Prefer canonical triple if present
                subj = (md.get("subject") or md.get("subj") or "").strip()
                rel = (md.get("relation") or md.get("rel") or "").strip()
                obj = (md.get("object") or md.get("obj") or "").strip()

                if subj and rel and obj:
                    fact_text = f"{subj} | {rel} | {obj}"
                else:
                    fact_text = (safe.get("content") or "").strip()
                    if not fact_text:
                        fact_text = (query + " -> " + response[:140]).strip()

                if not fact_text:
                    continue

                conf = float(md.get("confidence", truth_score))
                src = md.get("source", "conversation")

                try:
                    result = self.chroma_store.add_fact(
                        fact=fact_text,
                        source=src,
                        confidence=conf,
                    )
                    if result is not None:
                        stored += 1
                except Exception as inner:
                    logger.warning(f"[MemoryStorage] add_fact failed: {inner}")
                    continue

            logger.debug(f"[MemoryStorage] Total stored this turn: {stored}")

        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}", exc_info=True)

    async def consolidate_and_store_summary(self) -> None:
        """Consolidate recent memories and store the summary"""
        if not self.consolidator:
            logger.debug("[MemoryStorage] No consolidator available")
            return

        try:
            recent_memories = self.corpus_manager.get_recent_memories(
                count=self.consolidator.consolidation_threshold
            )

            if not recent_memories:
                logger.debug("[MemoryStorage] No recent memories to consolidate")
                return

            summary_node = await self.consolidator.consolidate_memories(recent_memories)

            if summary_node:
                # Store summary in corpus
                if hasattr(self.corpus_manager, 'add_summary'):
                    self.corpus_manager.add_summary(
                        content=summary_node.content,
                        tags=summary_node.tags,
                        timestamp=summary_node.timestamp
                    )

                # Store summary in Chroma
                if hasattr(self.chroma_store, 'add_to_collection'):
                    summary_metadata = {
                        "timestamp": summary_node.timestamp.isoformat(),
                        "type": "summary",
                        "importance_score": summary_node.importance_score,
                        "tags": ",".join(summary_node.tags) if summary_node.tags else "",
                        "memory_count": len(recent_memories)
                    }
                    self.chroma_store.add_to_collection(
                        "summaries",
                        summary_node.content,
                        summary_metadata
                    )

                self.last_consolidation_time = datetime.now()
                logger.info(f"[MemoryStorage] Consolidated {len(recent_memories)} memories into summary")

        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")
