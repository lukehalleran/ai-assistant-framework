"""
# core/prompt/gatherer_memory.py

Mixin providing memory retrieval methods for ContextGatherer.

Methods:
  - _get_recent_conversations(limit) -> List[Dict]
  - _get_semantic_memories(query, limit) -> List[Dict]
  - _deduplicate_memories(memories) -> List[Dict]
  - _expand_query_with_graph(query, max_terms) -> str
  - _get_summaries_separated(query, limit) -> Dict[str, List]
  - _get_summaries(query, limit) -> List[Dict]
  - _get_summaries_separate(query, recent_limit, semantic_limit) -> Dict[str, List]
  - _get_summaries_hybrid_filtered(summaries, query, limit) -> List[Dict]
  - _synthesize_summaries_from_recent(recent_conversations) -> List[Dict]
  - _get_reflections(query, limit) -> List[Dict]
  - _get_reflections_separated(query, limit) -> Dict[str, List]
  - _get_reflections_separate(query, recent_limit, semantic_limit) -> Dict[str, List]
  - _get_reflections_hybrid(query, limit) -> List[Dict]
  - _get_reflections_hybrid_filtered(reflections, query, limit) -> List[Dict]
  - get_facts(query, limit) -> List[Dict]
  - get_recent_facts(limit) -> List[Dict]
  - get_user_profile_context(query, max_tokens) -> str

Depends on self.memory_coordinator, self.memory_id_map, self.token_manager,
self.gate_system, self.user_profile, self.model_manager (set by ContextGatherer.__init__).
"""

import os
import logging
from typing import Dict, List, Any
from datetime import datetime

from .formatter import _as_summary_dict, _parse_bool

logger = logging.getLogger("prompt_context_gatherer")

# Configuration loading
try:
    from config.app_config import config as _APP_CFG
    _MEM_CFG = (_APP_CFG.get("memory") or {})
except (ImportError, AttributeError):
    _MEM_CFG = {}


def _cfg_int(key: str, default_val: int) -> int:
    try:
        v = _MEM_CFG.get(key, default_val)
        return int(v) if v is not None else int(default_val)
    except (ValueError, TypeError):
        return int(default_val)


def _cfg_bool(key: str, default_val: bool) -> bool:
    try:
        v = _MEM_CFG.get(key, default_val)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ('1', 'true', 'yes', 'on')
        return bool(v) if v is not None else bool(default_val)
    except (ValueError, TypeError, AttributeError):
        return bool(default_val)


# Configuration constants
PROMPT_MAX_RECENT = _cfg_int("prompt_max_recent", 15)
PROMPT_MAX_MEMS = _cfg_int("prompt_max_mems", 15)
PROMPT_MAX_FACTS = _cfg_int("prompt_max_facts", 15)
PROMPT_MAX_RECENT_FACTS = _cfg_int("prompt_max_recent_facts", 15)
PROMPT_MAX_SUMMARIES = _cfg_int("prompt_max_summaries", 10)
PROMPT_MAX_REFLECTIONS = _cfg_int("prompt_max_reflections", 10)
USER_PROFILE_FACTS_PER_CATEGORY = _cfg_int("user_profile_facts_per_category", 10)

PROMPT_MAX_RECENT_SUMMARIES = _cfg_int("prompt_max_recent_summaries", 5)
PROMPT_MAX_SEMANTIC_SUMMARIES = _cfg_int("prompt_max_semantic_summaries", 5)
PROMPT_MAX_RECENT_REFLECTIONS = _cfg_int("prompt_max_recent_reflections", 5)
PROMPT_MAX_SEMANTIC_REFLECTIONS = _cfg_int("prompt_max_semantic_reflections", 5)

SEMANTIC_RETRIEVAL_LIMIT = _cfg_int("semantic_retrieval_limit", 40)
FORCE_MIN_MEMORIES = _cfg_bool("semantic_force_min_memories", True)

# Feature flags
REFLECTIONS_HYBRID_FILTER = _parse_bool(os.getenv("REFLECTIONS_HYBRID_FILTER", "1"))
SUMMARIES_HYBRID_FILTER = _parse_bool(os.getenv("SUMMARIES_HYBRID_FILTER", "1"))


class MemoryRetrievalMixin:
    """Mixin providing memory retrieval methods."""

    async def get_recent_facts(self, limit: int = PROMPT_MAX_RECENT_FACTS) -> List[Dict[str, Any]]:
        """Get recent facts from memory coordinator."""
        try:
            if hasattr(self.memory_coordinator, 'get_recent_facts'):
                facts = await self.memory_coordinator.get_recent_facts(limit)
                return facts or []
            else:
                # Fallback to regular facts
                return await self.get_facts(limit)
        except Exception as e:
            logger.warning(f"Error getting recent facts: {e}")
            return []

    async def get_facts(self, query: str = "", limit: int = PROMPT_MAX_FACTS) -> List[Dict[str, Any]]:
        """Get facts from memory coordinator."""
        try:
            if hasattr(self.memory_coordinator, 'get_facts'):
                facts = await self.memory_coordinator.get_facts(query, limit)
                return facts or []
            else:
                return []
        except Exception as e:
            logger.warning(f"Error getting facts: {e}")
            return []

    async def _get_recent_conversations(self, limit: int = PROMPT_MAX_RECENT) -> List[Dict[str, Any]]:
        """Get recent conversation memories."""
        try:
            # Get recent memories from corpus manager
            memories = []
            if hasattr(self.memory_coordinator, 'corpus_manager'):
                corpus_manager = self.memory_coordinator.corpus_manager
                if hasattr(corpus_manager, 'get_recent_memories'):
                    memories = corpus_manager.get_recent_memories(count=limit)
                    logger.debug(f"Got {len(memories or [])} recent conversations from corpus_manager (requested {limit})")

            # If we didn't get enough, try fallback
            if not memories or len(memories) < limit:
                logger.debug(f"Trying fallback for recent conversations (got {len(memories or [])} of {limit})")
                try:
                    fallback_memories = await self.memory_coordinator.get_memories("", limit=limit, topic_filter=None)
                    if fallback_memories:
                        logger.debug(f"Got {len(fallback_memories)} memories from fallback")
                        # Use fallback if we had no memories, or combine if we need more
                        if not memories:
                            memories = fallback_memories[:limit]
                        else:
                            needed = limit - len(memories)
                            memories.extend(fallback_memories[:needed])
                except Exception as e:
                    logger.warning(f"Fallback memory retrieval failed: {e}")

            # Track memory IDs for citations
            result_memories = memories or []
            logger.warning(f"[DEBUG RECENT] _get_recent_conversations: Returning {len(result_memories)} memories")
            for idx, mem in enumerate(result_memories, start=1):
                mem_id = f"MEM_RECENT_{idx}"
                ts = mem.get('timestamp', 'NO_TS')
                query = mem.get('query', '')[:80]
                # Log first 3 and last 3
                if idx < 3 or idx >= len(result_memories) - 3:
                    logger.warning(f"[DEBUG RECENT] Memory {idx}: ts={ts}, query={query}...")
                _content = str(mem.get('content', '')) or ''
                if not _content.strip():
                    _q = str(mem.get('query', ''))[:200]
                    _a = str(mem.get('response', ''))[:200]
                    _content = f"Q: {_q} A: {_a}" if (_q or _a) else ''
                # db_id: try id > memory_id > metadata.id
                _recent_db_id = mem.get('id') or mem.get('memory_id')
                if not _recent_db_id and isinstance(mem.get('metadata'), dict):
                    _recent_db_id = mem['metadata'].get('id')
                self.memory_id_map[mem_id] = {
                    'type': 'episodic_recent',
                    'timestamp': mem.get('timestamp', ''),
                    'content': _content[:500],
                    'relevance_score': 1.0,  # Recent memories always relevant
                    'db_id': _recent_db_id,
                }

            return result_memories

        except Exception as e:
            logger.warning(f"Error getting recent conversations: {e}")
            return []

    async def _get_summaries_separated(self, query: str = "", limit: int = PROMPT_MAX_SUMMARIES) -> Dict[str, List[Dict[str, Any]]]:
        """Get conversation summaries separated into recent and semantic hits."""
        try:
            if hasattr(self.memory_coordinator, 'get_summaries'):
                # Fetch with buffer to allow hybrid filtering/top-up to reach 'limit'
                summaries = self.memory_coordinator.get_summaries(limit * 3)
            elif hasattr(self.memory_coordinator, 'corpus_manager'):
                corpus_manager = self.memory_coordinator.corpus_manager
                if hasattr(corpus_manager, 'get_summaries'):
                    summaries = corpus_manager.get_summaries(limit * 3)
                else:
                    summaries = []
            else:
                summaries = []

            # Normalize legacy schema (some records may have 'response' or 'text')
            try:
                summaries = [
                    (s if not isinstance(s, dict) else (
                        s if s.get('content') else {**s, 'content': s.get('response', s.get('text', ''))}
                    ))
                    for s in (summaries or [])
                ]
            except (TypeError, KeyError, AttributeError):
                pass  # Keep original summaries if normalization fails

            logger.debug(f"Retrieved {len(summaries)} summaries from memory coordinator")

            result = {"recent": [], "semantic": []}

            if summaries:
                # Split: half recent + half semantic
                recent_limit = limit // 2
                semantic_limit = limit - recent_limit

                # Get recent summaries (most recent first)
                recent_summaries = summaries[:recent_limit] if summaries else []
                remaining_summaries = summaries[recent_limit:] if len(summaries) > recent_limit else []

                result["recent"] = recent_summaries

                # Apply semantic filtering to remaining summaries if we have a query and filtering is enabled
                if SUMMARIES_HYBRID_FILTER and query and remaining_summaries:
                    logger.debug(f"Applying semantic filtering to {len(remaining_summaries)} older summaries with query: {query[:50]}...")
                    gated_summaries = await self._get_summaries_hybrid_filtered(remaining_summaries, query, semantic_limit)
                    result["semantic"] = gated_summaries
                    logger.debug(f"Separated summaries: {len(result['recent'])} recent + {len(result['semantic'])} semantic")
                else:
                    logger.debug(f"No semantic filtering applied, returning {len(result['recent'])} recent summaries only")
                    result["semantic"] = []

                # Top-up recent if we still have fewer than expected using a synthesized
                # summary from recent conversations (cheap fallback)
                total_count = len(result["recent"]) + len(result["semantic"])
                if total_count < limit:
                    try:
                        recent_conversations = await self._get_recent_conversations(limit)
                        synth = await self._synthesize_summaries_from_recent(recent_conversations)
                        if synth:
                            needed = limit - total_count
                            result["recent"].extend(synth[:needed])
                    except Exception as e:
                        logger.debug(f"Summary top-up failed: {e}")

            return result

        except Exception as e:
            logger.warning(f"Error getting summaries: {e}")
            return {"recent": [], "semantic": []}

    async def _get_summaries(self, query: str = "", limit: int = PROMPT_MAX_SUMMARIES) -> List[Dict[str, Any]]:
        """Get conversation summaries (legacy combined method for backward compatibility)."""
        separated = await self._get_summaries_separated(query, limit)
        return separated["recent"] + separated["semantic"]

    async def _get_summaries_separate(self, query: str = "",
                                     recent_limit: int = PROMPT_MAX_RECENT_SUMMARIES,
                                     semantic_limit: int = PROMPT_MAX_SEMANTIC_SUMMARIES) -> Dict[str, List[Dict[str, Any]]]:
        """Get summaries separated into recent and semantic with independent limits."""
        try:
            if hasattr(self.memory_coordinator, 'get_summaries'):
                # Fetch with buffer to allow semantic filtering to reach semantic_limit
                summaries = self.memory_coordinator.get_summaries((recent_limit + semantic_limit) * 3)
            elif hasattr(self.memory_coordinator, 'corpus_manager'):
                corpus_manager = self.memory_coordinator.corpus_manager
                if hasattr(corpus_manager, 'get_summaries'):
                    summaries = corpus_manager.get_summaries((recent_limit + semantic_limit) * 3)
                else:
                    summaries = []
            else:
                summaries = []

            # Normalize legacy schema
            try:
                summaries = [
                    (s if not isinstance(s, dict) else (
                        s if s.get('content') else {**s, 'content': s.get('response', s.get('text', ''))}
                    ))
                    for s in (summaries or [])
                ]
            except (TypeError, KeyError, AttributeError):
                pass  # Keep original summaries if normalization fails

            logger.debug(f"Retrieved {len(summaries)} summaries from memory coordinator")
            result = {"recent": [], "semantic": []}

            if summaries:
                # Get recent summaries (most recent first, no filtering)
                recent_summaries = summaries[:recent_limit] if summaries else []
                result["recent"] = recent_summaries

                # Apply semantic filtering to remaining summaries
                if query and len(summaries) > recent_limit:
                    remaining_summaries = summaries[recent_limit:]
                    logger.debug(f"Applying semantic filtering to {len(remaining_summaries)} older summaries with query: {query[:50]}...")
                    gated_summaries = await self._get_summaries_hybrid_filtered(remaining_summaries, query, semantic_limit)
                    result["semantic"] = gated_summaries
                    logger.debug(f"Separated summaries: {len(result['recent'])} recent + {len(result['semantic'])} semantic")
                else:
                    logger.debug(f"No semantic filtering applied, returning {len(result['recent'])} recent summaries only")
                    result["semantic"] = []

            # Track summary IDs for citations
            for idx, summ in enumerate(result.get("recent", []), start=1):
                sum_id = f"SUM_RECENT_{idx}"
                self.memory_id_map[sum_id] = {
                    'type': 'summary_recent',
                    'timestamp': summ.get('timestamp', ''),
                    'content': str(summ.get('content', ''))[:500],
                    'relevance_score': 1.0,  # Recent summaries always relevant
                    'db_id': summ.get('id', None)  # Track database ID
                }

            for idx, summ in enumerate(result.get("semantic", []), start=1):
                sum_id = f"SUM_SEMANTIC_{idx}"
                self.memory_id_map[sum_id] = {
                    'type': 'summary_semantic',
                    'timestamp': summ.get('timestamp', ''),
                    'content': str(summ.get('content', ''))[:500],
                    'relevance_score': summ.get('relevance_score', summ.get('score', 0.0)),
                    'db_id': summ.get('id', None)  # Track database ID
                }

            return result

        except Exception as e:
            logger.warning(f"Error in _get_summaries_separate: {e}")
            return {"recent": [], "semantic": []}

    def _expand_query_with_graph(self, query: str, max_terms: int = 8) -> str:
        """Expand a query with display names of graph-connected entities.

        Extracts entities from the query via alias resolution, walks 2-hop
        edges (to traverse through hub nodes like "user" in star topologies),
        and appends target entity display names to the query.
        This bridges vocabulary gaps (e.g. "my brother" -> "dillion").

        Args:
            query: Original user query
            max_terms: Maximum expansion terms to append

        Returns:
            Expanded query string (original + appended terms)
        """
        try:
            from config.app_config import (
                KNOWLEDGE_GRAPH_ENABLED,
                GRAPH_QUERY_EXPANSION_ENABLED,
                GRAPH_QUERY_EXPANSION_MAX_TERMS,
            )
            if not KNOWLEDGE_GRAPH_ENABLED or not GRAPH_QUERY_EXPANSION_ENABLED:
                return query

            mc = self.memory_coordinator
            graph = getattr(mc, "graph_memory", None)
            resolver = getattr(mc, "entity_resolver", None)
            if not graph or not resolver or graph.node_count() == 0:
                return query

            from memory.graph_utils import extract_graph_entities, rank_expansion_candidates

            effective_max = min(max_terms, GRAPH_QUERY_EXPANSION_MAX_TERMS)
            query_entities = extract_graph_entities(query, resolver, graph_memory=graph)
            if not query_entities:
                return query

            # Rank by connectivity (non-hub edges), filter junk, cap at max
            expansion_terms = rank_expansion_candidates(
                query_entities, graph, depth=2,
                skip_ids={"user"}, max_terms=effective_max,
            )
            if not expansion_terms:
                return query

            expanded = query + " " + " ".join(expansion_terms)
            logger.debug(
                f"[ContextGatherer] Query expansion: +{len(expansion_terms)} terms "
                f"from entities {query_entities}: {expansion_terms}"
            )
            return expanded

        except Exception as e:
            logger.debug(f"[ContextGatherer] Query expansion failed: {e}")
            return query

    async def _get_semantic_memories(self, query: str = "", limit: int = PROMPT_MAX_MEMS) -> List[Dict[str, Any]]:
        """Get relevant memories using semantic search only."""
        try:
            # FAST MODE: Reduce retrieval pool 50x (2150 -> ~45) for 15x speed boost
            if hasattr(self, '_fast_mode') and self._fast_mode:
                # With retrieval_limit=15: memory_retriever does 15*3=45, hybrid does 45*1=45 total
                logger.warning(f"[FAST MODE] Quick semantic search: {limit} results from ~45 candidates (vs 2150)")
                retrieval_limit = 15  # Tiny candidate pool for mobile
            else:
                retrieval_limit = SEMANTIC_RETRIEVAL_LIMIT  # Full 2150

            logger.debug(f"Semantic memories: retrieving up to {limit} semantic results")

            if not query:
                logger.debug("No query provided, returning empty memories")
                return []

            # Graph-driven query expansion: append related entity names
            expanded_query = self._expand_query_with_graph(query)

            # Get semantic memories with filtering
            semantic_memories = []
            try:
                # Retrieve memories for semantic search (use expanded query)
                search_query = expanded_query

                logger.debug(f"[CONTEXT_GATHERER] About to call memory_coordinator.get_memories with query='{search_query[:50]}...', limit={retrieval_limit}")

                all_memories = await self.memory_coordinator.get_memories(
                    search_query, limit=retrieval_limit, topic_filter=None
                )

                logger.debug(f"[CONTEXT_GATHERER] Retrieved {len(all_memories)} memories from coordinator (requested {retrieval_limit})")
                logger.debug(f"[CONTEXT_GATHERER] Memory coordinator type: {type(self.memory_coordinator)}")

                if len(all_memories) > 0:
                    logger.debug(f"[CONTEXT_GATHERER] Sample memory keys: {list(all_memories[0].keys()) if all_memories[0] else 'empty'}")
                else:
                    logger.warning(f"[CONTEXT_GATHERER] NO MEMORIES RETURNED from coordinator despite successful call!")

                # Skip redundant gating - memory coordinator already filtered memories
                gated_memories = all_memories
                logger.debug(f"[CONTEXT_GATHERER] Skipping redundant gating - using {len(all_memories)} memories from memory coordinator")

                # Force minimum count - if enabled and we don't have enough after gating, use highest-scoring ungated memories
                if FORCE_MIN_MEMORIES and len(gated_memories) < limit:
                    logger.debug(f"Only {len(gated_memories)} memories after gating, need {limit}, using top-scoring fallback...")

                    # Check the gating threshold to see what's being filtered
                    try:
                        # Sort all memories by relevance score (descending)
                        scored_memories = []
                        for mem in all_memories:
                            score = mem.get('relevance_score', 0.0)
                            scored_memories.append((score, mem))

                        scored_memories.sort(key=lambda x: x[0], reverse=True)

                        # Show what scores we're working with
                        if scored_memories:
                            top_score = scored_memories[0][0]
                            cutoff_score = scored_memories[len(gated_memories)][0] if len(gated_memories) < len(scored_memories) else 0.0
                            logger.debug(f"Memory scores: top={top_score:.3f}, cutoff={cutoff_score:.3f}, have={len(gated_memories)}/{limit}")

                        # Use gated memories first, then supplement with highest-scoring ungated memories
                        # Use object ids for lookup (dicts are unhashable)
                        gated_ids = set(id(m) for m in gated_memories)
                        additional_needed = limit - len(gated_memories)

                        for score, mem in scored_memories[len(gated_memories):]:
                            if additional_needed <= 0:
                                break
                            if id(mem) not in gated_ids:
                                gated_memories.append(mem)
                                gated_ids.add(id(mem))
                                additional_needed -= 1
                                logger.debug(f"Added memory with score {score:.3f}")

                        logger.debug(f"Final count: {len(gated_memories)} memories (forced minimum)")
                    except Exception as e:
                        logger.warning(f"Memory scoring fallback failed: {e}")
                elif len(gated_memories) < limit:
                    logger.debug(f"Only {len(gated_memories)} memories after gating (force min disabled), keeping high-quality results only")

                semantic_memories = gated_memories[:limit]  # Ensure we don't exceed limit

                # Limit to requested number
                semantic_memories = semantic_memories[:limit]
                logger.debug(f"After limiting to {limit}: {len(semantic_memories)} memories")

            except Exception as e:
                logger.warning(f"Semantic memory retrieval failed: {e}")

            # Apply enhanced deduplication
            result = self._deduplicate_memories(semantic_memories)
            result = result[:limit]  # Final limit in case deduplication removed items

            # Track memory IDs for citations
            for idx, mem in enumerate(result, start=1):
                mem_id = f"MEM_SEMANTIC_{idx}"
                # Score: prefer final_score (from scorer) > relevance_score > score
                _sem_score = mem.get('final_score', mem.get('relevance_score', mem.get('score', 0.0)))
                # Timestamp: top-level > metadata.timestamp
                _sem_ts = mem.get('timestamp', '')
                if not _sem_ts and isinstance(mem.get('metadata'), dict):
                    _sem_ts = mem['metadata'].get('timestamp', '')
                self.memory_id_map[mem_id] = {
                    'type': 'episodic_semantic',
                    'timestamp': _sem_ts,
                    'content': str(mem.get('content', ''))[:500],  # Truncate for citation display
                    'relevance_score': float(_sem_score) if _sem_score else 0.0,
                    'db_id': mem.get('id', None)  # Track database ID (UUID or generated ID)
                }

            logger.debug(f"[CONTEXT_GATHERER] Final result: {len(result)} semantic memories (limit was {limit})")
            if len(result) > 0:
                logger.debug(f"[CONTEXT_GATHERER] Sample result memory: {list(result[0].keys()) if result[0] else 'empty'}")
            return result

        except Exception as e:
            logger.error(f"[CONTEXT_GATHERER] Semantic memory retrieval failed: {e}")
            import traceback
            logger.debug(f"[CONTEXT_GATHERER] Exception traceback: {traceback.format_exc()}")
            return []

    def _deduplicate_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced deduplication using memory IDs and content."""
        seen_ids = set()
        seen_content = set()
        deduped = []
        duplicates_removed = 0

        for mem in memories:
            # First try memory ID deduplication
            mem_id = mem.get('id') or mem.get('memory_id')
            if mem_id and mem_id in seen_ids:
                duplicates_removed += 1
                continue

            # Fall back to content deduplication
            content_key = str(mem.get('content', '')) + str(mem.get('response', ''))
            if content_key in seen_content:
                duplicates_removed += 1
                continue

            # Add to results and mark as seen
            if mem_id:
                seen_ids.add(mem_id)
            seen_content.add(content_key)
            deduped.append(mem)

        if duplicates_removed > 0:
            logger.debug(f"Deduplication removed {duplicates_removed} duplicate memories")

        return deduped

    async def _get_reflections_separated(self, query: str = "", limit: int = PROMPT_MAX_REFLECTIONS) -> Dict[str, List[Dict[str, Any]]]:
        """Get reflections separated into recent and semantic hits."""
        try:
            # Calculate split: half semantic, half recent
            semantic_limit = limit // 2
            recent_limit = limit - semantic_limit

            logger.debug(f"Separating reflections: {recent_limit} recent + {semantic_limit} semantic")

            # Get all reflections first - try different approaches
            all_reflections = []
            try:
                all_reflections = await self.memory_coordinator.get_reflections(limit * 3)
                logger.debug(f"Got {len(all_reflections)} total reflections")
            except Exception as e:
                logger.warning(f"get_reflections failed: {e}")

            # If no reflections, try alternative method
            if not all_reflections:
                try:
                    # Try getting from corpus manager directly if available
                    if hasattr(self.memory_coordinator, 'corpus_manager'):
                        corpus = self.memory_coordinator.corpus_manager
                        # Get reflections from corpus
                        corpus_data = getattr(corpus, 'corpus', [])
                        reflections = [entry for entry in corpus_data if entry.get('type') == 'reflection']
                        # Sort by timestamp, most recent first
                        all_reflections = sorted(reflections,
                                               key=lambda x: x.get('timestamp', ''),
                                               reverse=True)[:limit * 2]
                        logger.debug(f"Got {len(all_reflections)} reflections from corpus fallback")
                except Exception as e:
                    logger.warning(f"Corpus reflection fallback failed: {e}")

            if not all_reflections:
                return {"recent": [], "semantic": []}

            result = {"recent": [], "semantic": []}

            # Recent reflections (first N, no filtering)
            recent_reflections = all_reflections[:recent_limit]
            result["recent"] = recent_reflections

            # Semantic reflections (filtered by query relevance)
            if query and len(all_reflections) > recent_limit:
                remaining_reflections = all_reflections[recent_limit:]
                try:
                    # Apply gating to semantic reflections
                    semantic_reflections = await self._apply_gating(remaining_reflections, query)
                    semantic_reflections = semantic_reflections[:semantic_limit]
                    result["semantic"] = semantic_reflections
                except Exception as e:
                    logger.warning(f"Semantic reflection gating failed: {e}")
                    # If gating fails, just take first few remaining
                    result["semantic"] = remaining_reflections[:semantic_limit]
            else:
                result["semantic"] = []

            logger.debug(f"Separated reflections: {len(result['recent'])} recent + {len(result['semantic'])} semantic")
            return result

        except Exception as e:
            logger.warning(f"Error in _get_reflections_separated: {e}")
            return {"recent": [], "semantic": []}

    async def _get_reflections_hybrid(self, query: str = "", limit: int = PROMPT_MAX_REFLECTIONS) -> List[Dict[str, Any]]:
        """Get reflections using hybrid approach: half semantic hits, half most recent (legacy combined method)."""
        separated = await self._get_reflections_separated(query, limit)
        return separated["recent"] + separated["semantic"]

    async def _get_summaries_hybrid_filtered(self, summaries: List[Dict[str, Any]],
                                           query: str, limit: int) -> List[Dict[str, Any]]:
        """Apply hybrid filtering to summaries based on query relevance."""
        if not summaries or not query:
            return summaries[:limit]

        try:
            # Use cached gate system for similarity scoring
            gate = self.gate_system

            # Apply standard gating to summaries
            gated_summaries = await self._apply_gating(summaries, query)
            return gated_summaries[:limit]

        except Exception as e:
            logger.warning(f"Hybrid summary filtering failed: {e}")
            return summaries[:limit]

    async def _get_reflections(self, query: str = "", limit: int = PROMPT_MAX_REFLECTIONS) -> List[Dict[str, Any]]:
        """Get reflections from memory."""
        try:
            reflections = []

            if hasattr(self.memory_coordinator, 'get_reflections'):
                result = await self.memory_coordinator.get_reflections(limit)
                # Handle the result
                if isinstance(result, list):
                    reflections = result
                else:
                    reflections = [result] if result else []

            elif hasattr(self.memory_coordinator, 'corpus_manager'):
                corpus_manager = self.memory_coordinator.corpus_manager
                if hasattr(corpus_manager, 'get_reflections'):
                    result = corpus_manager.get_reflections(limit)
                    # Handle async streams/generators
                    if hasattr(result, '__aiter__'):
                        reflections = [item async for item in result]
                    elif hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
                        reflections = list(result)
                    else:
                        reflections = result if isinstance(result, list) else [result] if result else []

            # Normalize legacy schema for reflections
            try:
                reflections = [
                    (r if not isinstance(r, dict) else (
                        r if r.get('content') else {**r, 'content': r.get('response', r.get('text', ''))}
                    ))
                    for r in (reflections or [])
                ]
            except (TypeError, KeyError, AttributeError):
                pass  # Keep original reflections if normalization fails

            # Apply hybrid filtering if enabled
            if REFLECTIONS_HYBRID_FILTER and query and reflections:
                return await self._get_reflections_hybrid_filtered(reflections, query, limit)

            return reflections or []

        except Exception as e:
            logger.warning(f"Error getting reflections: {e}")
            return []

    async def _get_reflections_hybrid_filtered(self, reflections: List[Dict[str, Any]],
                                             query: str, limit: int) -> List[Dict[str, Any]]:
        """Apply hybrid filtering to reflections based on query relevance."""
        if not reflections or not query:
            return reflections[:limit]

        try:
            # Use cached gate system for similarity scoring
            gate = self.gate_system

            # Apply standard gating to reflections
            gated_reflections = await self._apply_gating(reflections, query)
            return gated_reflections[:limit]

        except Exception as e:
            logger.warning(f"Hybrid reflection filtering failed: {e}")
            return reflections[:limit]

    async def _get_reflections_separate(self, query: str = "",
                                       recent_limit: int = PROMPT_MAX_RECENT_REFLECTIONS,
                                       semantic_limit: int = PROMPT_MAX_SEMANTIC_REFLECTIONS) -> Dict[str, List[Dict[str, Any]]]:
        """Get reflections separated into recent and semantic with independent limits."""
        try:
            logger.debug(f"Separating reflections: {recent_limit} recent + {semantic_limit} semantic")

            # Get all reflections first
            all_reflections = []
            try:
                all_reflections = await self.memory_coordinator.get_reflections((recent_limit + semantic_limit) * 3)
                logger.debug(f"Got {len(all_reflections)} total reflections")
            except Exception as e:
                logger.warning(f"get_reflections failed: {e}")

            # Fallback methods
            if not all_reflections:
                try:
                    if hasattr(self.memory_coordinator, 'corpus_manager'):
                        corpus = self.memory_coordinator.corpus_manager
                        corpus_data = getattr(corpus, 'corpus', [])
                        reflections = [entry for entry in corpus_data if entry.get('type') == 'reflection']
                        all_reflections = sorted(reflections,
                                               key=lambda x: x.get('timestamp', ''),
                                               reverse=True)[:(recent_limit + semantic_limit) * 2]
                        logger.debug(f"Got {len(all_reflections)} reflections from corpus fallback")
                except Exception as e:
                    logger.warning(f"Corpus reflection fallback failed: {e}")

            if not all_reflections:
                return {"recent": [], "semantic": []}

            result = {"recent": [], "semantic": []}

            # Recent reflections (first N, no filtering)
            recent_reflections = all_reflections[:recent_limit]
            result["recent"] = recent_reflections

            # Semantic reflections (filtered by query relevance)
            if query and len(all_reflections) > recent_limit:
                remaining_reflections = all_reflections[recent_limit:]
                try:
                    # Apply gating to semantic reflections
                    semantic_reflections = await self._apply_gating(remaining_reflections, query)
                    semantic_reflections = semantic_reflections[:semantic_limit]
                    result["semantic"] = semantic_reflections
                    logger.debug(f"Separated reflections: {len(result['recent'])} recent + {len(result['semantic'])} semantic")
                except Exception as e:
                    logger.warning(f"Semantic reflection filtering failed: {e}")
                    result["semantic"] = []

            # Track reflection IDs for citations
            for idx, refl in enumerate(result.get("recent", []), start=1):
                refl_id = f"REFL_RECENT_{idx}"
                self.memory_id_map[refl_id] = {
                    'type': 'reflection_recent',
                    'timestamp': refl.get('timestamp', ''),
                    'content': str(refl.get('content', ''))[:500],
                    'relevance_score': 1.0,  # Recent reflections always relevant
                    'db_id': refl.get('id', None)  # Track database ID
                }

            for idx, refl in enumerate(result.get("semantic", []), start=1):
                refl_id = f"REFL_SEMANTIC_{idx}"
                self.memory_id_map[refl_id] = {
                    'type': 'reflection_semantic',
                    'timestamp': refl.get('timestamp', ''),
                    'content': str(refl.get('content', ''))[:500],
                    'relevance_score': refl.get('relevance_score', refl.get('score', 0.0)),
                    'db_id': refl.get('id', None)  # Track database ID
                }

            return result

        except Exception as e:
            logger.warning(f"Error in _get_reflections_separate: {e}")
            return {"recent": [], "semantic": []}

    async def _synthesize_summaries_from_recent(self, recent_conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Synthesize summaries from recent conversations if no summaries exist."""
        if not recent_conversations:
            return []

        try:
            # Import summarizer
            from .summarizer import LLMSummarizer

            summarizer = LLMSummarizer(self.model_manager, self.memory_coordinator)

            # Generate summary from recent conversations
            summary_text = await summarizer._llm_summarize_recent(
                recent_conversations,
                max_conversations=10
            )

            if summary_text:
                summary_dict = _as_summary_dict(
                    summary_text,
                    ["synthesized", "recent"],
                    "context_gatherer"
                )
                return [summary_dict]

        except Exception as e:
            logger.warning(f"Summary synthesis failed: {e}")

        return []

    async def get_user_profile_context(self, query: str, max_tokens: int = 500) -> str:
        """
        Get user profile context with hybrid retrieval (2/3 semantic + 1/3 recent).
        Replaces the old semantic_facts and fresh_facts approach.

        Args:
            query: Current user query for semantic relevance
            max_tokens: Token budget for profile context

        Returns:
            Formatted profile string ready for prompt injection
        """
        if not self.user_profile:
            logger.debug("[ContextGatherer] No user profile available")
            return ""

        try:
            profile_context = self.user_profile.get_context_injection(
                max_tokens=max_tokens,
                query=query,
                facts_per_category=USER_PROFILE_FACTS_PER_CATEGORY
            )
            logger.debug(f"[ContextGatherer] Generated profile context ({len(profile_context)} chars)")

            # Track user profile for citations (single entry for the whole profile)
            if profile_context:
                self.memory_id_map["PROFILE_CONTEXT"] = {
                    'type': 'user_profile',
                    'timestamp': datetime.now().isoformat(),
                    'content': profile_context[:500],  # Truncate for citation display
                    'relevance_score': 1.0,  # Profile always relevant when included
                    'db_id': None  # User profile is generated on-the-fly, not stored in DB
                }

            return profile_context
        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get profile context: {e}")
            return ""
