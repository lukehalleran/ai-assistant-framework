# memory/memory_storage.py
"""
Memory storage module.

Module Contract
- Purpose: Implements the MemoryStorageProtocol contract for persisting memories. Handles storage of conversations, facts, and summaries to both corpus and ChromaDB.
- Inputs:
  - store_interaction(query, response, metadata) -> bool
  - store_fact(fact_dict) -> bool
  - consolidate_if_needed() -> bool
- Outputs:
  - Stored memories in corpus JSON and ChromaDB collections
  - Consolidation results (summary nodes when threshold reached)
- Key behaviors:
  - Coordinates between corpus_manager and chroma_store
  - Triggers fact extraction per turn (if FACTS_EXTRACT_EACH_TURN enabled)
  - Triggers summary consolidation (if not SUMMARIZE_AT_SHUTDOWN_ONLY)
  - _maybe_regenerate_narrative(): Triggers narrative context refresh after consolidation [NEW 2026-01-17]
  - Entity metadata forwarding: extract_and_store_facts() uses dict-based source to pass
    fact_scope, entity_type, user_connection through to ChromaDB metadata [NEW 2026-03]
- Dependencies:
  - memory.corpus_manager (JSON persistence)
  - memory.storage.multi_collection_chroma_store (vector storage)
  - memory.fact_extractor (fact extraction — user + entity facts)
  - memory.memory_consolidator (summary generation, narrative synthesis)
- Side effects:
  - Writes to corpus JSON file
  - Writes to ChromaDB collections (including entity fact metadata)
  - Triggers narrative context regeneration after summary consolidation [NEW 2026-01-17]
"""

import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Optional
from collections import deque

from utils.logging_utils import get_logger

logger = get_logger("memory_storage")

# Environment configuration
FACTS_EXTRACT_EACH_TURN = os.getenv("FACTS_EXTRACT_EACH_TURN", "0").strip().lower() not in ("0", "false", "no", "off")
SUMMARIZE_AT_SHUTDOWN_ONLY = os.getenv("SUMMARIZE_AT_SHUTDOWN_ONLY", "1").strip().lower() not in ("0", "false", "no", "off")

# Knowledge graph config (imported inside methods to avoid circular imports)
def _get_graph_enabled():
    try:
        from config.app_config import KNOWLEDGE_GRAPH_ENABLED
        return KNOWLEDGE_GRAPH_ENABLED
    except ImportError:
        return False


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
        graph_memory=None,
        entity_resolver=None,
        fact_verifier=None,
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
            graph_memory: Optional GraphMemory for knowledge graph ingestion
            entity_resolver: Optional EntityResolver for entity resolution
            fact_verifier: Optional FactVerifier for pre-storage conflict checking
        """
        self.corpus_manager = corpus_manager
        self.chroma_store = chroma_store
        self.fact_extractor = fact_extractor
        self.consolidator = consolidator
        self.topic_manager = topic_manager
        self.scorer = scorer
        self.time_manager = time_manager
        self.graph_memory = graph_memory
        self.entity_resolver = entity_resolver
        self.fact_verifier = fact_verifier

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

    def _is_file_error_response(self, response: str) -> bool:
        """
        Check if response contains file processing error messages.

        These are ephemeral technical errors that should not be stored in long-term memory,
        as they create false memories of problems that may have been fixed.

        Args:
            response: Assistant's response text

        Returns:
            True if response contains file errors that should not be stored
        """
        if not response:
            return False

        # File error patterns from utils/file_processor.py
        # These patterns are specific enough that their presence indicates a file error
        file_error_patterns = [
            "[security error processing",       # Security validation errors
            "[error reading",                   # General file reading errors
            "[unsupported file type:",          # Unsupported extensions
            "[empty file:",                     # Empty file notices
            "[no text content extracted from",  # DOCX extraction failures
            "[total file size exceeds limit:",  # Size limit violations
        ]

        # Check if response contains any file error patterns (case-insensitive)
        response_lower = response.lower()
        for pattern in file_error_patterns:
            if pattern in response_lower:
                logger.debug(f"[MemoryStorage] Detected file error pattern: {pattern}")
                return True

        return False

    async def store_interaction(
        self,
        query: str,
        response: str,
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Persist a turn in both corpus & Chroma with computed metadata.

        Returns:
            str: Database ID (UUID) of the stored memory, or None if storage failed
        """
        try:
            # SKIP STORAGE: Don't persist file error responses
            # These are ephemeral technical issues that create false memories
            if self._is_file_error_response(response):
                logger.info(f"[MemoryStorage] Skipped storing file error response to prevent false memories")
                return None

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
                if hasattr(self.topic_manager, "get_primary_topic"):
                    # get_primary_topic() updates internal state and returns the topic
                    primary_topic = self.topic_manager.get_primary_topic(query) or "general"
                elif hasattr(self.topic_manager, "detect_topic"):
                    primary_topic = self.topic_manager.detect_topic(f"{query} {response}") or "general"

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
            return memory_id

        except Exception as e:
            logger.error(f"Error storing interaction: {e}")
            import traceback
            traceback.print_exc()
            return None

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
            import traceback
            logger.debug(f"[MemoryStorage] Traceback:\n{traceback.format_exc()}")

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

                # Build source dict to forward entity metadata to ChromaDB
                source_dict = {
                    "source": src,
                    "confidence": conf,
                }
                for key in ("fact_scope", "entity_type", "user_connection"):
                    val = md.get(key)
                    if val:
                        source_dict[key] = val

                try:
                    # Fact verification gate: check for conflicts before storage
                    if self.fact_verifier:
                        try:
                            from memory.fact_verification import FactVerdict
                            vr = await self.fact_verifier.verify(
                                subject=subj, predicate=rel, object_val=obj,
                                fact_text=fact_text, source=src, confidence=conf,
                                fact_scope=md.get("fact_scope", "user"),
                            )
                            if vr.verdict == FactVerdict.REJECT:
                                logger.debug(
                                    f"[MemoryStorage] Fact rejected by verifier: "
                                    f"{fact_text[:80]} (reason={vr.reason})"
                                )
                                continue
                            if vr.verdict == FactVerdict.STORE_AND_FLAG:
                                # Flag conflicting old facts as superseded
                                for cand in vr.conflicting_candidates:
                                    if cand.doc_id:
                                        try:
                                            supersede_md = {"superseded_by": fact_text[:200]}
                                            supersede_md.update(vr.metadata_updates)
                                            self.chroma_store.update_metadata(
                                                "facts", cand.doc_id, supersede_md,
                                            )
                                            logger.debug(
                                                f"[MemoryStorage] Marked fact {cand.doc_id} "
                                                f"as superseded"
                                            )
                                        except Exception as flag_err:
                                            logger.debug(
                                                f"[MemoryStorage] Failed to flag old fact: {flag_err}"
                                            )
                        except Exception as verify_err:
                            logger.debug(f"[MemoryStorage] Verification failed, proceeding: {verify_err}")

                    result = self.chroma_store.add_fact(
                        fact=fact_text,
                        source=source_dict,
                    )
                    if result is not None:
                        stored += 1
                        # Feed knowledge graph if available
                        if self.graph_memory and self.entity_resolver and _get_graph_enabled():
                            self._ingest_fact_to_graph(
                                subj=subj, rel=rel, obj=obj,
                                fact_id=str(result) if result else "",
                                entity_type=md.get("entity_type", ""),
                                confidence=conf,
                            )
                except Exception as inner:
                    logger.warning(f"[MemoryStorage] add_fact failed: {inner}")
                    continue

            logger.debug(f"[MemoryStorage] Total stored this turn: {stored}")

        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}", exc_info=True)

    # Compiled regexes for _is_graph_worthy_object (class-level for performance)
    _TEMPORAL_RE = re.compile(
        r"^\d+(\.\d+)?\s*(years?|months?|weeks?|days?|hours?)", re.IGNORECASE
    )
    _FREQUENCY_RE = re.compile(
        r"^(once|twice|three\s+times)\s+a\s+", re.IGNORECASE
    )
    _MEASUREMENT_RE = re.compile(
        r"""^\d[\d'".,]*(lbs?|iu|mg|mcg|kg|oz|ft|in)?\s*$""", re.IGNORECASE
    )
    _VERB_STEMS = frozenset({
        "stopped", "started", "went", "came", "used", "began",
        "finished", "understands", "feels", "thinks", "strained",
        "relying", "protecting", "finishing", "moved", "working",
        "planning", "trying", "wanting", "becoming", "living",
        "dealing", "struggling", "considering", "taking",
    })

    @staticmethod
    def _is_graph_worthy_object(obj: str) -> bool:
        """Check if a triple's object value is a real entity worth graphing.

        Entities/attributes (1-3 words) are kept as nodes — they're valid
        hop targets (e.g. auggie --breed--> golden retriever).  Phrases
        (4+ words), temporal durations, measurements, and verb phrases
        are stored as subject-node metadata instead.
        """
        o = obj.strip().lower()

        # Too short or too long
        if len(o) < 2 or len(o) > 60:
            return False

        # 4+ words = descriptive phrase, not an entity.
        if len(o.split()) >= 4:
            return False

        # Generic/meaningless words
        generic = {
            "a lot", "some", "none", "true", "false",
            "yes", "no", "many", "few", "lots", "not sure", "unknown",
            "graph", "user",
        }
        if o in generic:
            return False

        # Temporal/duration patterns: "2 years", "6 months ago", "10 days"
        if MemoryStorage._TEMPORAL_RE.match(o):
            return False

        # Frequency patterns: "once a week", "twice a day"
        if MemoryStorage._FREQUENCY_RE.match(o):
            return False

        # Measurement patterns: "5'11\"", "20lbs", "10000iu"
        if MemoryStorage._MEASUREMENT_RE.match(o):
            return False

        # Verb-phrase filter: "stopped being religious", "finishing grad school"
        first_word = o.split()[0]
        if first_word in MemoryStorage._VERB_STEMS:
            return False

        return True

    def _ingest_fact_to_graph(
        self, subj: str, rel: str, obj: str,
        fact_id: str = "", entity_type: str = "", confidence: float = 0.5,
    ) -> None:
        """Push a single S-R-O triple into the knowledge graph.

        Called after a fact is successfully stored in ChromaDB.
        Resolves entities via EntityResolver and normalizes relations.
        Filters out non-entity objects (durations, phrases, generic words).
        """
        if not subj or not rel or not obj:
            return
        try:
            from memory.entity_resolver import normalize_relation
            from memory.graph_models import GraphNode, GraphEdge
            from config.app_config import KNOWLEDGE_GRAPH_MIN_CONFIDENCE

            if confidence < KNOWLEDGE_GRAPH_MIN_CONFIDENCE:
                return

            canon_rel = normalize_relation(rel)

            # Non-entity objects (sentence fragments, generic words) get stored
            # as metadata on the subject node instead of creating junk nodes.
            if not self._is_graph_worthy_object(obj):
                subj_display = subj if subj.lower() != "user" else "User"
                subj_type = "person" if subj.lower() == "user" else (entity_type or "other")
                subj_id = self.entity_resolver.resolve_or_create(subj, entity_type=subj_type, display_name=subj_display)
                # Store as node metadata: {"relation": "value"}
                node = self.graph_memory.get_entity(subj_id)
                if node:
                    self.graph_memory.add_entity(GraphNode(
                        entity_id=subj_id,
                        display_name=node.display_name,
                        entity_type=node.entity_type,
                        metadata={canon_rel: obj},
                    ))
                    logger.debug(f"[MemoryStorage] Graph metadata: {subj_id}.{canon_rel} = '{obj}'")
                return

            # Map "user" subject to a canonical user node
            subj_display = subj if subj.lower() != "user" else "User"
            obj_display = obj

            # Resolve or create entities
            subj_type = "person" if subj.lower() == "user" else (entity_type or "other")
            subj_id = self.entity_resolver.resolve_or_create(subj, entity_type=subj_type, display_name=subj_display)
            obj_id = self.entity_resolver.resolve_or_create(obj, display_name=obj_display)

            # Add relation as graph edge
            self.graph_memory.add_relation(
                GraphEdge(source_id=subj_id, relation=canon_rel, target_id=obj_id),
                fact_id=fact_id,
            )
            logger.debug(f"[MemoryStorage] Graph: {subj_id} --{canon_rel}--> {obj_id}")
        except Exception as e:
            logger.debug(f"[MemoryStorage] Graph ingestion failed: {e}")

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

                # Trigger narrative context regeneration (non-critical)
                await self._maybe_regenerate_narrative()

        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")

    async def store_skill(self, skill) -> Optional[str]:
        """
        Store a procedural skill with semantic deduplication.

        Args:
            skill: ProceduralSkill instance

        Returns:
            str: Document ID if stored, None if duplicate or failed
        """
        try:
            from config.app_config import PROCEDURAL_SKILLS_ENABLED, SKILL_DEDUP_THRESHOLD
            if not PROCEDURAL_SKILLS_ENABLED:
                return None

            collection_name = "procedural_skills"

            # Ensure collection exists
            if collection_name not in self.chroma_store.collections or self.chroma_store.collections[collection_name] is None:
                if hasattr(self.chroma_store, "create_collection"):
                    self.chroma_store.create_collection(collection_name)

            coll = self.chroma_store.collections.get(collection_name)
            if coll is None:
                logger.warning("[MemoryStorage] procedural_skills collection not available")
                return None

            embedding_text = skill.to_embedding_text()

            # Semantic deduplication: query for similar skills
            if coll.count() > 0:
                similar = self.chroma_store.query_collection(
                    collection_name,
                    query_text=embedding_text,
                    n_results=min(3, coll.count()),
                )
                for match in similar:
                    score = match.get("relevance_score", 0.0)
                    if score and score >= SKILL_DEDUP_THRESHOLD:
                        existing_trigger = (match.get("metadata") or {}).get("trigger", "")
                        logger.info(
                            f"[MemoryStorage] Skill deduplicated (score={score:.2f}): "
                            f"'{skill.trigger[:60]}' ~ '{existing_trigger[:60]}'"
                        )
                        return None

            # Store
            metadata = skill.to_metadata()
            doc_id = self.chroma_store.add_to_collection(
                collection_name, embedding_text, metadata
            )
            logger.info(
                f"[MemoryStorage] Stored skill {doc_id}: "
                f"trigger='{skill.trigger[:80]}', category={skill.category.value}"
            )
            return doc_id

        except Exception as e:
            logger.error(f"[MemoryStorage] Failed to store skill: {e}")
            return None

    async def _maybe_regenerate_narrative(self) -> None:
        """
        Regenerate the narrative context after summary creation.

        This is a non-critical operation - failures are logged and swallowed.
        """
        try:
            from config.app_config import NARRATIVE_CONTEXT_ENABLED
            if not NARRATIVE_CONTEXT_ENABLED:
                return

            # Retrieve recent summaries for synthesis
            recent_weeklies = self._get_recent_summaries_by_timespan("weekly", limit=4)
            recent_monthlies = self._get_recent_summaries_by_timespan("monthly", limit=2)

            if not recent_weeklies and not recent_monthlies:
                logger.debug("[MemoryStorage] No summaries available for narrative synthesis")
                return

            if not self.consolidator:
                logger.debug("[MemoryStorage] No consolidator available for narrative synthesis")
                return

            # Generate narrative via consolidator
            narrative = await self.consolidator.generate_narrative_context(
                recent_weeklies=recent_weeklies,
                recent_monthlies=recent_monthlies
            )

            if narrative:
                self.corpus_manager.save_narrative_context(narrative)
                logger.info("[MemoryStorage] Narrative context updated after summary generation")

        except Exception as e:
            # Non-critical: log and continue
            logger.warning(f"[MemoryStorage] Failed to update narrative context: {e}")

    def _get_recent_summaries_by_timespan(self, span_type: str, limit: int = 4) -> List[Dict]:
        """
        Get recent summaries categorized by time span.

        Args:
            span_type: "weekly" (summaries from last 4 weeks) or "monthly" (last 2 months)
            limit: Maximum number of summaries to return

        Returns:
            List of summary dicts sorted by timestamp (most recent first)
        """
        from datetime import timedelta

        try:
            # Get all recent summaries
            all_summaries = self.corpus_manager.get_summaries(limit=50)

            if not all_summaries:
                return []

            now = self._now()

            # Filter by timespan
            if span_type == "weekly":
                # Summaries from the last 4 weeks
                cutoff = now - timedelta(weeks=4)
            elif span_type == "monthly":
                # Summaries from the last 2 months (use 60 days as proxy)
                cutoff = now - timedelta(days=60)
            else:
                return []

            filtered = []
            for s in all_summaries:
                ts = s.get("timestamp")
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except:
                        continue
                if isinstance(ts, datetime):
                    # Handle timezone-aware datetimes
                    if ts.tzinfo is not None:
                        ts = ts.replace(tzinfo=None)
                    if ts >= cutoff:
                        filtered.append(s)

            # Sort by timestamp (most recent first) and limit
            filtered.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
            return filtered[:limit]

        except Exception as e:
            logger.debug(f"[MemoryStorage] Error getting {span_type} summaries: {e}")
            return []
