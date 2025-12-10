"""
# core/prompt/context_gatherer.py

Module Contract
- Purpose: Data collection and retrieval for prompt building context assembly. ENHANCED: Replaces flat facts with categorized UserProfile using hybrid retrieval (2/3 semantic + 1/3 recent).
- Inputs:
  - gather_context(query: str, limit_memories: int, limit_facts: int) -> Dict[str, Any]
  - get_recent_conversations(limit: int) -> List[Dict]
  - retrieve_semantic_memories(query: str, limit: int) -> List[Dict]
  - get_wiki_content(query: str, limit: int) -> List[Dict]
  - UPDATED: get_user_profile_context(query: str) -> str [NEW: replaces semantic_facts + fresh_facts]
- Outputs:
  - Comprehensive context dictionary with all gathered data
  - Recent conversation history within specified limits
  - Semantically relevant memories and facts
  - Wikipedia content and semantic chunks
  - UPDATED: 'user_profile' key with categorized facts (replaces 'semantic_facts' and 'fresh_facts')
- Behavior:
  - Retrieves data from multiple memory collections (episodic, semantic, procedural)
  - Applies filtering and relevance scoring to retrieved content
  - Implements caching for expensive operations (wiki lookups, semantic search)
  - Coordinates parallel data fetching with timeout management
  - Handles graceful fallbacks when data sources are unavailable
  - UPDATED: Uses UserProfile hybrid retrieval (2/3 semantic + 1/3 recent per category) instead of flat facts
- Dependencies:
  - memory.memory_coordinator (memory retrieval)
  - memory.user_profile (NEW: categorized fact storage)
  - core.wiki_util (Wikipedia content)
  - utils.time_manager (temporal context)
  - processing.gate_system (relevance filtering)
- Side effects:
  - Memory system queries and retrievals
  - Cache writes for performance optimization
  - Network requests for Wikipedia content
  - Logging of data collection activities
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque
from utils.time_manager import TimeManager
from utils.logging_utils import get_logger
from core.wiki_util import get_wiki_snippet, clean_query
from knowledge.semantic_search import semantic_search_with_neighbors
from .formatter import _as_summary_dict, _dedupe_keep_order, _truncate_list, _parse_bool

logger = get_logger("prompt_context_gatherer")

# Configuration loading
try:
    from config.app_config import config as _APP_CFG
    _MEM_CFG = (_APP_CFG.get("memory") or {})
except Exception:
    _MEM_CFG = {}

def _cfg_int(key: str, default_val: int) -> int:
    try:
        v = _MEM_CFG.get(key, default_val)
        return int(v) if v is not None else int(default_val)
    except Exception:
        return int(default_val)

def _cfg_float(key: str, default_val: float) -> float:
    try:
        v = _MEM_CFG.get(key, default_val)
        return float(v) if v is not None else float(default_val)
    except Exception:
        return float(default_val)

def _cfg_bool(key: str, default_val: bool) -> bool:
    try:
        v = _MEM_CFG.get(key, default_val)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ('1', 'true', 'yes', 'on')
        return bool(v) if v is not None else bool(default_val)
    except Exception:
        return bool(default_val)

# Configuration constants
PROMPT_MAX_RECENT = _cfg_int("prompt_max_recent", 15)  # 15 recent conversations
PROMPT_MAX_MEMS = _cfg_int("prompt_max_mems", 15)     # 15 relevant memories (semantic search results only)
PROMPT_MAX_FACTS = _cfg_int("prompt_max_facts", 15)    # 15 facts
PROMPT_MAX_RECENT_FACTS = _cfg_int("prompt_max_recent_facts", 15)  # 15 recent facts
PROMPT_MAX_SUMMARIES = _cfg_int("prompt_max_summaries", 10)  # Total summaries (recent + semantic)
PROMPT_MAX_REFLECTIONS = _cfg_int("prompt_max_reflections", 10)  # Total reflections (recent + semantic)
USER_PROFILE_FACTS_PER_CATEGORY = _cfg_int("user_profile_facts_per_category", 3)  # Facts per category in user profile

# Separate limits for recent vs semantic
PROMPT_MAX_RECENT_SUMMARIES = _cfg_int("prompt_max_recent_summaries", 5)  # Recent summaries only
PROMPT_MAX_SEMANTIC_SUMMARIES = _cfg_int("prompt_max_semantic_summaries", 5)  # Semantic summaries only
PROMPT_MAX_RECENT_REFLECTIONS = _cfg_int("prompt_max_recent_reflections", 5)  # Recent reflections only
PROMPT_MAX_SEMANTIC_REFLECTIONS = _cfg_int("prompt_max_semantic_reflections", 5)  # Semantic reflections only

# Semantic memory retrieval
SEMANTIC_RETRIEVAL_LIMIT = _cfg_int("semantic_retrieval_limit", 40)  # How many memories to retrieve for semantic search (reduced from 75 for speed)
FORCE_MIN_MEMORIES = _cfg_bool("semantic_force_min_memories", True)  # Force minimum memory count even if gating is strict
PROMPT_MAX_DREAMS = _cfg_int("prompt_max_dreams", 3)
PROMPT_MAX_SEMANTIC = _cfg_int("prompt_max_semantic", 10)
PROMPT_MAX_WIKI = _cfg_int("prompt_max_wiki", 3)

# Feature flags
DREAMS_ENABLED = _parse_bool(os.getenv("DREAMS_ENABLED", "1"))
REFLECTIONS_HYBRID_FILTER = _parse_bool(os.getenv("REFLECTIONS_HYBRID_FILTER", "1"))
SUMMARIES_HYBRID_FILTER = _parse_bool(os.getenv("SUMMARIES_HYBRID_FILTER", "1"))

# Semantic search configuration
SEM_K = int(os.getenv("SEM_K", "50"))
SEM_TIMEOUT_S = int(os.getenv("SEM_TIMEOUT_S", "8"))
SEM_STITCH_MAX_CHARS = int(os.getenv("SEM_STITCH_MAX_CHARS", "4000"))

# Gating configuration
GATE_COSINE_THRESHOLD = float(os.getenv("GATE_COSINE_THRESHOLD", "0.45"))
GATE_XENC_THRESHOLD = float(os.getenv("GATE_XENC_THRESHOLD", "0.55"))

# Caching
_wiki_cache = {}  # Simple in-memory cache for wiki snippets


class ContextGatherer:
    """Handles all data collection and retrieval for prompt building."""

    def __init__(self, memory_coordinator, model_manager, token_manager, gate_system=None, time_manager=None):
        self.memory_coordinator = memory_coordinator
        self.model_manager = model_manager
        self.token_manager = token_manager
        self.time_manager = time_manager or TimeManager()
        # Use provided gate system or create cached one
        self._gate_system = gate_system

        # Memory citation tracking
        self.memory_id_map = {}  # Maps citation IDs to memory metadata

        # Initialize user profile (gets from memory_coordinator if available)
        self.user_profile = None
        if hasattr(memory_coordinator, 'user_profile'):
            self.user_profile = memory_coordinator.user_profile
            logger.debug("[ContextGatherer] Using UserProfile from memory_coordinator")
        else:
            # Fallback: create our own instance
            from memory.user_profile import UserProfile
            self.user_profile = UserProfile()
            logger.debug("[ContextGatherer] Created new UserProfile instance")

    @property
    def gate_system(self):
        """Get cached gate system, creating it only once."""
        if self._gate_system is None:
            from processing.gate_system import CosineSimilarityGateSystem
            self._gate_system = CosineSimilarityGateSystem()
        return self._gate_system

    def _wiki_cache_key(self, query: str) -> str:
        """Generate cache key for wiki queries."""
        return clean_query(query).lower().strip()

    async def _get_wiki_snippet_cached(self, query: str) -> Optional[Dict[str, Any]]:
        """Get wiki snippet with caching."""
        cache_key = self._wiki_cache_key(query)

        # Check cache first
        if cache_key in _wiki_cache:
            cached_entry = _wiki_cache[cache_key]
            # Simple TTL: 1 hour
            if datetime.now() - cached_entry["timestamp"] < timedelta(hours=1):
                return cached_entry["data"]

        # Fetch from wiki
        try:
            snippet = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, get_wiki_snippet, query),
                timeout=5.0
            )
            if snippet:
                # Cache the result
                _wiki_cache[cache_key] = {
                    "data": snippet,
                    "timestamp": datetime.now()
                }
                return snippet
        except asyncio.TimeoutError:
            logger.warning(f"Wiki snippet timeout for: {query}")
        except Exception as e:
            logger.warning(f"Wiki snippet error for {query}: {e}")

        return None

    async def _apply_gating(self, memories: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Apply multi-stage gating to filter memories by relevance.

        Uses the processing/gate_system.py for advanced filtering.
        """
        if not memories:
            return memories

        try:
            # Use cached gate system or fallback to creating one if not available
            if hasattr(self, '_gate_system') and self._gate_system is not None:
                gate_system = self._gate_system
                logger.debug("Using cached gate system for gating")
            else:
                # Fallback: Create gate system with model_manager
                from processing.gate_system import MultiStageGateSystem
                gate_system = MultiStageGateSystem(self.model_manager)
                logger.debug("Creating new gate system (fallback)")

            # Apply gating
            gated_memories = await gate_system.filter_memories(query, memories)

            logger.debug(f"Gating: {len(memories)} -> {len(gated_memories)} memories")
            return gated_memories

        except Exception as e:
            logger.warning(f"Gating failed, returning original memories: {e}")
            return memories

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
            for idx, mem in enumerate(result_memories):
                mem_id = f"MEM_RECENT_{idx}"
                self.memory_id_map[mem_id] = {
                    'type': 'episodic_recent',
                    'timestamp': mem.get('timestamp', ''),
                    'content': str(mem.get('content', ''))[:500],  # Truncate for citation display
                    'relevance_score': 1.0,  # Recent memories always relevant
                    'db_id': mem.get('id', None)  # Track database ID (UUID or generated ID)
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
            except Exception:
                pass

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
            except Exception:
                pass

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
            for idx, summ in enumerate(result.get("recent", [])):
                sum_id = f"SUM_RECENT_{idx}"
                self.memory_id_map[sum_id] = {
                    'type': 'summary_recent',
                    'timestamp': summ.get('timestamp', ''),
                    'content': str(summ.get('content', ''))[:500],
                    'relevance_score': 1.0,  # Recent summaries always relevant
                    'db_id': summ.get('id', None)  # Track database ID
                }

            for idx, summ in enumerate(result.get("semantic", [])):
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

    async def _get_semantic_memories(self, query: str = "", limit: int = PROMPT_MAX_MEMS) -> List[Dict[str, Any]]:
        """Get relevant memories using semantic search only."""
        try:
            logger.debug(f"Semantic memories: retrieving up to {limit} semantic results")

            if not query:
                logger.debug("No query provided, returning empty memories")
                return []

            # Get semantic memories with filtering
            semantic_memories = []
            try:
                # Retrieve memories for semantic search
                retrieval_limit = SEMANTIC_RETRIEVAL_LIMIT

                logger.debug(f"[CONTEXT_GATHERER] About to call memory_coordinator.get_memories with query='{query[:50]}...', limit={retrieval_limit}")

                all_memories = await self.memory_coordinator.get_memories(
                    query, limit=retrieval_limit, topic_filter=None
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
                        gated_set = set(gated_memories)
                        additional_needed = limit - len(gated_memories)

                        for score, mem in scored_memories[len(gated_memories):]:
                            if additional_needed <= 0:
                                break
                            if mem not in gated_set:
                                gated_memories.append(mem)
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
            for idx, mem in enumerate(result):
                mem_id = f"MEM_SEMANTIC_{idx}"
                self.memory_id_map[mem_id] = {
                    'type': 'episodic_semantic',
                    'timestamp': mem.get('timestamp', ''),
                    'content': str(mem.get('content', ''))[:500],  # Truncate for citation display
                    'relevance_score': mem.get('relevance_score', mem.get('score', 0.0)),
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
            except Exception:
                pass

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
            for idx, refl in enumerate(result.get("recent", [])):
                refl_id = f"REFL_RECENT_{idx}"
                self.memory_id_map[refl_id] = {
                    'type': 'reflection_recent',
                    'timestamp': refl.get('timestamp', ''),
                    'content': str(refl.get('content', ''))[:500],
                    'relevance_score': 1.0,  # Recent reflections always relevant
                    'db_id': refl.get('id', None)  # Track database ID
                }

            for idx, refl in enumerate(result.get("semantic", [])):
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

    def _bounded(self, items: List[Any], max_items: int) -> List[Any]:
        """Helper to bound list length."""
        return items[:max_items] if len(items) > max_items else items

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

    async def _get_dreams(self, limit: int = PROMPT_MAX_DREAMS) -> List[Dict[str, Any]]:
        """Get dreams if enabled."""
        if not DREAMS_ENABLED:
            return []

        try:
            if hasattr(self.memory_coordinator, 'get_dreams'):
                dreams = self.memory_coordinator.get_dreams(limit)
                return dreams or []
        except Exception as e:
            logger.warning(f"Error getting dreams: {e}")

        return []

    def _should_skip_wikipedia(self, query: str) -> bool:
        """Determine if query is too simple/conversational for Wikipedia lookup."""
        if not query:
            return True

        query_lower = query.lower().strip()
        words = query_lower.split()

        # Skip very short queries
        if len(words) <= 2:
            return True

        # Skip common conversational patterns
        conversational_patterns = [
            'hello', 'hi', 'hey', 'thanks', 'thank you', 'ok', 'okay',
            'yes', 'no', 'lol', 'haha', 'good', 'great', 'nice',
            'how are you', 'what\'s up', 'see you', 'bye', 'goodbye'
        ]

        if any(pattern in query_lower for pattern in conversational_patterns):
            return True

        # Skip if query is mostly short words (< 4 chars)
        long_words = [w for w in words if len(w) >= 4 and w.isalpha()]
        if len(long_words) < 2:
            return True

        return False

    async def _get_wiki_content(self, query: str, limit: int = PROMPT_MAX_WIKI) -> List[Dict[str, Any]]:
        """Get wiki content for query."""
        if not query:
            return []

        # Smart skip for simple/conversational queries
        if self._should_skip_wikipedia(query):
            return []

        try:
            # Extract key terms for wiki search
            search_terms = []

            # Simple term extraction
            words = query.lower().split()
            for word in words:
                if len(word) > 3 and word.isalpha():
                    search_terms.append(word)

            # Get wiki snippets for top terms
            wiki_results = []
            for term in search_terms[:limit]:
                snippet = await self._get_wiki_snippet_cached(term)
                if snippet:
                    wiki_results.append(snippet)

            return wiki_results

        except Exception as e:
            logger.warning(f"Error getting wiki content: {e}")
            return []

    async def _get_semantic_chunks(self, query: str, k: int = SEM_K,
                                 max_results: int = PROMPT_MAX_SEMANTIC) -> List[Dict[str, Any]]:
        """Get semantic chunks using semantic search."""
        if not query:
            return []

        try:
            # Use semantic search with neighbors
            results = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    semantic_search_with_neighbors,
                    query,
                    k
                ),
                timeout=SEM_TIMEOUT_S
            )

            if not results:
                return []

            # Process and stitch results by title
            chunks_by_title = {}
            for result in results[:max_results * 2]:  # Get more to allow stitching
                title = result.get("title", "")
                content = result.get("content", "")

                if title and content:
                    if title not in chunks_by_title:
                        chunks_by_title[title] = {
                            "title": title,
                            "content": content,
                            "metadata": result.get("metadata", {})
                        }
                    else:
                        # Stitch content together
                        existing = chunks_by_title[title]
                        combined = existing["content"] + "\n\n" + content

                        # Check length limit
                        if len(combined) <= SEM_STITCH_MAX_CHARS:
                            existing["content"] = combined
                        else:
                            # Start new chunk with different title
                            title_alt = f"{title} (continued)"
                            chunks_by_title[title_alt] = {
                                "title": title_alt,
                                "content": content,
                                "metadata": result.get("metadata", {})
                            }

            # Return limited results
            chunks = list(chunks_by_title.values())
            return chunks[:max_results]

        except asyncio.TimeoutError:
            logger.warning(f"Semantic search timeout after {SEM_TIMEOUT_S}s")
        except Exception as e:
            logger.warning(f"Semantic search error: {e}")

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
