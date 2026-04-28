"""
# core/prompt/context_gatherer.py

Module Contract
- Purpose: Parallel async data retrieval for prompt building. Gathers memories, facts,
  summaries, reflections, wiki, web search, personal notes, reference docs, git commits,
  procedural skills, proposals, graph context, threads, proactive insights, codebase changes,
  and user profile — all as separate async methods called in parallel by builder.py.
- Key public methods:
  - get_user_profile_context(query, max_tokens) -> str  [categorized UserProfile, replaces flat facts]
  - get_personal_notes(query, limit) -> List[Dict]  [Obsidian vault: 1/3 keyword + 2/3 semantic]
  - get_reference_docs(query, limit) -> List[Dict]  [uploaded docs: 1/3 keyword + 2/3 semantic]
  - get_user_uploads(query, limit) -> List[Dict]  [user_uploads collection]
  - get_git_commits(query, limit) -> List[Dict]  [procedural collection git commits]
  - get_proposed_features(query, limit) -> List[Dict]  [proposals collection]
  - get_procedural_skills(query, limit) -> List[Dict]  [procedural_skills collection]
  - get_graph_context(query, max_sentences) -> List[str]  [knowledge graph BFS traversal]
  - get_unresolved_threads(max_results) -> List[Dict]  [open threads for proactive surfacing]
  - get_proactive_insights(query, max_insights) -> List[str]  [cross-domain insights from graph]
  - get_codebase_changes(since_datetime) -> Dict  [git diff since last session]
  - get_narrative_context() -> str  [cached temporal grounding from corpus_manager]
  - should_trigger_web_search(query, crisis_level) -> bool  [heuristic + LLM trigger detection]
  - clear_memory_id_map() -> None  [reset citation tracking between turns]
- Internal retrieval methods (called by builder.py via _hygiene_and_caps):
  - _get_recent_conversations(limit) -> List[Dict]
  - _get_semantic_memories(query, limit) -> List[Dict]  [with graph query expansion]
  - _get_summaries_separated(query, limit) -> Dict[str, List]
  - _get_reflections_separated(query, limit) -> Dict[str, List]
  - _get_wiki_content(query, limit) -> List[Dict]  [ChromaDB wiki_knowledge first, API fallback]
  - _get_semantic_chunks(query, k) -> List[Dict]
  - _get_web_search_results(query, crisis_level, ...) -> WebSearchResult/MultiSearchResult
  - _get_dreams(limit) -> List[Dict]
  - get_facts(query, limit) / get_recent_facts(limit) -> List[Dict]
  - _expand_query_with_graph(query, max_terms) -> str  [appends graph neighbor names]
  - _apply_gating(memories, query) -> List[Dict]  [multi-stage gate system]
  - _deduplicate_memories(memories) -> List[Dict]
- Outputs:
  - Individual retrieval results returned to builder for parallel assembly
  - memory_id_map: Dict tracking doc_id → content for citation provenance
  - Narrative context string (synthesized life state) [NEW 2026-01-17]
- Behavior:
  - Retrieves data from multiple memory collections (episodic, semantic, procedural)
  - Applies filtering and relevance scoring to retrieved content
  - Implements caching for expensive operations (wiki lookups, semantic search, web search)
  - Coordinates parallel data fetching with timeout management
  - Handles graceful fallbacks when data sources are unavailable
  - Triggers web search using LLM-first analysis with heuristic fallback
  - ENHANCED: LLM generates optimized search_terms for better results
  - ENHANCED: Complex queries are auto-decomposed into parallel sub-queries (e.g., "Compare Tesla and Rivian" → 2 searches)
  - Suppresses web search during HIGH/MEDIUM crisis levels
  - UPDATED: Uses UserProfile hybrid retrieval (2/3 semantic + 1/3 recent per category) instead of flat facts
  - Retrieves cached narrative context (0ms latency) for temporal grounding [NEW 2026-01-17]
- Dependencies:
  - memory.memory_coordinator (memory retrieval)
  - memory.user_profile (NEW: categorized fact storage)
  - memory.corpus_manager (narrative context retrieval) [NEW 2026-01-17]
  - core.wiki_util (Wikipedia content)
  - knowledge.web_search_manager (ENHANCED: Tavily web search with multi_search + query decomposition)
  - utils.web_search_trigger (search trigger detection)
  - utils.time_manager (temporal context)
  - processing.gate_system (relevance filtering)
- Side effects:
  - Memory system queries and retrievals
  - Cache writes for performance optimization
  - Network requests for Wikipedia content
  - Network requests to Tavily API for web search (may be multiple for decomposed queries)
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
except (ImportError, AttributeError):
    _MEM_CFG = {}

def _cfg_int(key: str, default_val: int) -> int:
    try:
        v = _MEM_CFG.get(key, default_val)
        return int(v) if v is not None else int(default_val)
    except (ValueError, TypeError):
        return int(default_val)

def _cfg_float(key: str, default_val: float) -> float:
    try:
        v = _MEM_CFG.get(key, default_val)
        return float(v) if v is not None else float(default_val)
    except (ValueError, TypeError):
        return float(default_val)

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
PROMPT_MAX_RECENT = _cfg_int("prompt_max_recent", 15)  # 15 recent conversations
PROMPT_MAX_MEMS = _cfg_int("prompt_max_mems", 15)     # 15 relevant memories (semantic search results only)
PROMPT_MAX_FACTS = _cfg_int("prompt_max_facts", 15)    # 15 facts
PROMPT_MAX_RECENT_FACTS = _cfg_int("prompt_max_recent_facts", 15)  # 15 recent facts
PROMPT_MAX_SUMMARIES = _cfg_int("prompt_max_summaries", 10)  # Total summaries (recent + semantic)
PROMPT_MAX_REFLECTIONS = _cfg_int("prompt_max_reflections", 10)  # Total reflections (recent + semantic)
USER_PROFILE_FACTS_PER_CATEGORY = _cfg_int("user_profile_facts_per_category", 10)  # Facts per category in user profile (default 10)

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
try:
    from config.app_config import SEMANTIC_CHUNKS_GATE_THRESHOLD
except ImportError:
    SEMANTIC_CHUNKS_GATE_THRESHOLD = 0.35

# Gating configuration
GATE_COSINE_THRESHOLD = float(os.getenv("GATE_COSINE_THRESHOLD", "0.45"))
GATE_XENC_THRESHOLD = float(os.getenv("GATE_XENC_THRESHOLD", "0.55"))

# Web search configuration (import from app_config if available)
try:
    from config.app_config import (
        WEB_SEARCH_ENABLED,
        WEB_SEARCH_TIMEOUT,
        WEB_SEARCH_MAX_CONTENT_CHARS,
        WEB_SEARCH_API_KEY,
        WEB_SEARCH_DAILY_CREDIT_LIMIT,
    )
except ImportError:
    WEB_SEARCH_ENABLED = _cfg_bool("web_search_enabled", True)
    WEB_SEARCH_TIMEOUT = _cfg_float("web_search_timeout", 30.0)
    WEB_SEARCH_MAX_CONTENT_CHARS = _cfg_int("web_search_max_content_chars", 10000)
    WEB_SEARCH_API_KEY = os.getenv("TAVILY_API_KEY", "")
    WEB_SEARCH_DAILY_CREDIT_LIMIT = 100

# Caching
_wiki_cache = {}  # Simple in-memory cache for wiki snippets
_WIKI_CACHE_MAX_SIZE = 100  # Maximum cache entries to prevent memory leaks


def _cleanup_wiki_cache():
    """Remove expired entries and enforce max size."""
    global _wiki_cache
    now = datetime.now()
    # Remove expired entries (older than 1 hour)
    _wiki_cache = {
        k: v for k, v in _wiki_cache.items()
        if now - v["timestamp"] < timedelta(hours=1)
    }
    # Enforce max size by removing oldest entries
    if len(_wiki_cache) > _WIKI_CACHE_MAX_SIZE:
        sorted_entries = sorted(_wiki_cache.items(), key=lambda x: x[1]["timestamp"])
        _wiki_cache = dict(sorted_entries[-_WIKI_CACHE_MAX_SIZE:])


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

        # Initialize web search manager (lazy - only created when first used)
        self._web_search_manager = None
        self._web_search_trigger = None

    @property
    def gate_system(self):
        """Get cached gate system, creating it only once."""
        if self._gate_system is None:
            from processing.gate_system import CosineSimilarityGateSystem
            self._gate_system = CosineSimilarityGateSystem()
        return self._gate_system

    @property
    def web_search_manager(self):
        """Get cached web search manager, creating it only once."""
        if self._web_search_manager is None:
            try:
                from knowledge.web_search_manager import WebSearchManager, WebSearchRateLimiter

                # Create rate limiter with config values
                rate_limiter = WebSearchRateLimiter(
                    daily_limit=WEB_SEARCH_DAILY_CREDIT_LIMIT
                )

                self._web_search_manager = WebSearchManager(
                    api_key=WEB_SEARCH_API_KEY,  # Pass API key from config
                    rate_limiter=rate_limiter,
                    default_timeout=WEB_SEARCH_TIMEOUT,
                    max_content_chars=WEB_SEARCH_MAX_CONTENT_CHARS
                )

                # Log initialization status
                if self._web_search_manager.is_available():
                    logger.info("[ContextGatherer] WebSearchManager initialized with Tavily API")
                else:
                    logger.warning("[ContextGatherer] WebSearchManager initialized but Tavily API not available (check TAVILY_API_KEY)")

            except ImportError as e:
                logger.warning(f"[ContextGatherer] WebSearchManager not available: {e}")
                return None
            except Exception as e:
                logger.warning(f"[ContextGatherer] Failed to initialize WebSearchManager: {e}")
                return None
        return self._web_search_manager

    @property
    def web_search_trigger(self):
        """Get cached web search trigger (sync heuristic version), creating it only once."""
        if self._web_search_trigger is None:
            try:
                from utils.web_search_trigger import analyze_for_web_search
                self._web_search_trigger = analyze_for_web_search
                logger.debug("[ContextGatherer] Initialized web search trigger (sync)")
            except ImportError as e:
                logger.warning(f"[ContextGatherer] WebSearchTrigger not available: {e}")
                return None
        return self._web_search_trigger

    @property
    def web_search_trigger_llm(self):
        """
        Get the async LLM-first web search trigger function.

        This is the preferred trigger method that uses LLM classification
        first with heuristic fallback. Returns optimized search_terms.
        """
        try:
            from utils.web_search_trigger import analyze_for_web_search_llm
            return analyze_for_web_search_llm
        except ImportError as e:
            logger.warning(f"[ContextGatherer] LLM WebSearchTrigger not available: {e}")
            return None

    @property
    def obsidian_manager(self):
        """
        Get cached Obsidian manager, creating it only once.

        Returns None if Obsidian integration is disabled or unavailable.
        """
        if not hasattr(self, '_obsidian_manager'):
            self._obsidian_manager = None

        if self._obsidian_manager is None:
            try:
                from config.app_config import OBSIDIAN_ENABLED
                if not OBSIDIAN_ENABLED:
                    logger.debug("[ContextGatherer] Obsidian integration disabled")
                    return None

                from knowledge.obsidian_manager import ObsidianManager
                self._obsidian_manager = ObsidianManager()
                logger.info("[ContextGatherer] ObsidianManager initialized")
            except ImportError as e:
                logger.debug(f"[ContextGatherer] ObsidianManager not available: {e}")
                return None
            except Exception as e:
                logger.warning(f"[ContextGatherer] Failed to initialize ObsidianManager: {e}")
                return None
        return self._obsidian_manager

    @property
    def reference_docs_manager(self):
        """
        Get cached Reference Docs manager, creating it only once.

        Returns None if Reference Docs integration is disabled or unavailable.
        """
        if not hasattr(self, '_reference_docs_manager'):
            self._reference_docs_manager = None

        if self._reference_docs_manager is None:
            try:
                from config.app_config import REFERENCE_DOCS_ENABLED
                if not REFERENCE_DOCS_ENABLED:
                    logger.debug("[ContextGatherer] Reference docs integration disabled")
                    return None

                from knowledge.reference_docs_manager import ReferenceDocsManager
                self._reference_docs_manager = ReferenceDocsManager()
                logger.info("[ContextGatherer] ReferenceDocsManager initialized")
            except ImportError as e:
                logger.debug(f"[ContextGatherer] ReferenceDocsManager not available: {e}")
                return None
            except Exception as e:
                logger.warning(f"[ContextGatherer] Failed to initialize ReferenceDocsManager: {e}")
                return None
        return self._reference_docs_manager

    def _wiki_cache_key(self, query: str) -> str:
        """Generate cache key for wiki queries."""
        return clean_query(query).lower().strip()

    def clear_memory_id_map(self):
        """Clear the memory ID map to prevent memory leaks between queries."""
        self.memory_id_map = {}

    async def _get_wiki_snippet_cached(self, query: str) -> Optional[Dict[str, Any]]:
        """Get wiki snippet with caching."""
        # Periodic cleanup to prevent memory leaks
        if len(_wiki_cache) > _WIKI_CACHE_MAX_SIZE // 2:
            _cleanup_wiki_cache()

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

    async def get_personal_notes(
        self,
        query: str,
        limit: int = 10,
        include_images: bool = False,
        max_images_per_note: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get relevant personal notes from Obsidian vault.

        Args:
            query: Search query for semantic retrieval
            limit: Maximum notes to return
            include_images: If True, load actual image data for multimodal models
            max_images_per_note: Maximum images to load per note chunk

        Returns:
            List of note dicts with content, metadata, relevance_score,
            and optionally 'image_data' containing base64-encoded images
        """
        manager = self.obsidian_manager
        if not manager:
            return []

        try:
            # Clean query: remove self-referential phrases that pollute search
            import re
            clean_query = re.sub(
                r'\b(from|in|check|look at|search|find in)?\s*(my|the)?\s*(notes?|vault|obsidian)\b',
                '', query, flags=re.IGNORECASE
            )
            # Clean up extra whitespace
            clean_query = ' '.join(clean_query.split()).strip()
            # Fall back to original if cleaning removed everything
            search_query = clean_query if clean_query else query
            logger.debug(f"[ContextGatherer] Personal notes query: '{query}' -> '{search_query}'")

            # Retrieve notes via semantic search (with optional image loading)
            logger.warning(f"[ContextGatherer] IMAGE DEBUG: Calling get_notes with include_images={include_images}")
            notes = await manager.get_notes(
                search_query,
                limit=limit,
                include_images=include_images,
                max_images_per_note=max_images_per_note
            )
            # Debug: check what we got back
            if notes:
                total_images = sum(len(n.get('image_data', [])) for n in notes)
                logger.warning(f"[ContextGatherer] IMAGE DEBUG: Got {len(notes)} notes with {total_images} total images")

            # Track note IDs for citations
            if notes:
                for idx, note in enumerate(notes[:limit], start=1):
                    note_id = f"NOTE_{idx}"
                    meta = note.get('metadata', {})
                    image_data = note.get('image_data', [])
                    self.memory_id_map[note_id] = {
                        'type': 'personal_note',
                        'timestamp': meta.get('timestamp', ''),
                        'content': note.get('content', '')[:500],
                        'relevance_score': note.get('relevance_score', 0.0),
                        'title': meta.get('title', ''),
                        'file_path': meta.get('file_path', ''),
                        'has_images': len(image_data) > 0,
                        'image_count': len(image_data),
                        'db_id': note.get('id', None),
                    }

                image_count = sum(len(n.get('image_data', [])) for n in notes)
                logger.debug(f"[ContextGatherer] Retrieved {len(notes)} personal notes with {image_count} images")

            return notes or []

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get personal notes: {e}")
            return []

    async def get_reference_docs(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get relevant reference documents from uploaded docs collection.
        Excludes user uploads (type='user_upload') which have their own section.

        Args:
            query: Search query for semantic retrieval
            limit: Maximum document chunks to return

        Returns:
            List of doc dicts with content, metadata, and relevance_score
        """
        manager = self.reference_docs_manager
        if not manager:
            return []

        try:
            # Retrieve documents via hybrid search (fetch extra to allow filtering)
            docs = await manager.get_documents(query, limit=limit * 2)

            # Filter OUT user uploads — they appear in [USER UPLOADED ITEMS] instead
            docs = [d for d in docs if d.get('metadata', {}).get('type') != 'user_upload']
            docs = docs[:limit]

            # Track doc IDs for citations
            if docs:
                for idx, doc in enumerate(docs[:limit], start=1):
                    doc_id = f"REFDOC_{idx}"
                    meta = doc.get('metadata', {})
                    self.memory_id_map[doc_id] = {
                        'type': 'reference_doc',
                        'timestamp': meta.get('timestamp', ''),
                        'content': doc.get('content', '')[:500],
                        'relevance_score': doc.get('relevance_score', 0.0),
                        'title': meta.get('title', ''),
                        'section': meta.get('section', ''),
                        'file_path': meta.get('file_path', ''),
                        'db_id': doc.get('id', None),
                    }

                logger.debug(f"[ContextGatherer] Retrieved {len(docs)} reference docs for query")

            return docs or []

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get reference docs: {e}")
            return []

    async def get_user_uploads(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant user-uploaded items from reference_docs collection,
        filtered to only include entries with type='user_upload'.

        Args:
            query: Search query for semantic retrieval
            limit: Maximum uploads to return

        Returns:
            List of upload dicts with content, metadata, and relevance_score
        """
        manager = self.reference_docs_manager
        if not manager:
            return []

        try:
            # Fetch extra to allow filtering to user_upload type
            docs = await manager.get_documents(query, limit=limit * 2)

            # Filter to only user uploads
            uploads = [d for d in docs if d.get('metadata', {}).get('type') == 'user_upload']
            uploads = uploads[:limit]

            # Track for citations
            if uploads:
                for idx, upload in enumerate(uploads, start=1):
                    upload_id = f"UPLOAD_{idx}"
                    meta = upload.get('metadata', {})
                    self.memory_id_map[upload_id] = {
                        'type': 'user_upload',
                        'timestamp': meta.get('timestamp', ''),
                        'content': upload.get('content', '')[:500],
                        'relevance_score': upload.get('relevance_score', 0.0),
                        'title': meta.get('title', ''),
                        'is_image': meta.get('is_image', False),
                        'image_path': meta.get('image_path', ''),
                        'db_id': upload.get('id', None),
                    }

                logger.debug(f"[ContextGatherer] Retrieved {len(uploads)} user uploads for query")

            return uploads or []

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get user uploads: {e}")
            return []

    async def get_git_commits(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get relevant git commits from PROCEDURAL collection.

        Uses hybrid retrieval: 1/3 most recent commits + 2/3 semantically
        relevant commits, deduplicated by document ID.

        Args:
            query: Search query for semantic retrieval
            limit: Maximum commits to return

        Returns:
            List of commit dicts with content, metadata, and relevance_score
        """
        try:
            chroma = getattr(self.memory_coordinator, 'chroma_store', None)
            if not chroma or 'procedural' not in chroma.collections:
                return []

            from config.app_config import GIT_MEMORY_ENABLED
            if not GIT_MEMORY_ENABLED:
                return []

            # Hybrid split: 1/3 recent, 2/3 semantic
            recent_limit = max(limit // 3, 1)
            semantic_limit = limit - recent_limit

            # 1/3: Most recent commits (chronological)
            recent = chroma.get_recent('procedural', limit=recent_limit)

            # 2/3: Semantically relevant commits
            semantic = chroma.query_collection('procedural', query, n_results=semantic_limit + recent_limit)

            # Deduplicate: recent first, then fill with semantic
            seen_ids = set()
            merged = []

            for commit in recent:
                doc_id = commit.get('id')
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged.append(commit)

            for commit in semantic:
                if len(merged) >= limit:
                    break
                doc_id = commit.get('id')
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged.append(commit)

            # Track for citations
            for idx, commit in enumerate(merged, start=1):
                cid = f"COMMIT_{idx}"
                meta = commit.get('metadata', {})
                self.memory_id_map[cid] = {
                    'type': 'git_commit',
                    'timestamp': meta.get('timestamp', ''),
                    'content': commit.get('content', '')[:500],
                    'relevance_score': commit.get('relevance_score', 0.0),
                    'commit_hash': meta.get('commit_hash', ''),
                    'db_id': commit.get('id', None),
                }

            logger.debug(f"[ContextGatherer] Retrieved {len(merged)} git commits ({len(recent)} recent + {len(semantic)} semantic, deduped to {len(merged)})")
            return merged

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get git commits: {e}")
            return []

    async def get_proposed_features(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get relevant code proposals (proposed features) for prompt injection.

        Uses ProposalFilter for retrieval, dedup, gating, and ranking.
        Only returns results for project-related queries.

        Args:
            query: Search query for relevance filtering
            limit: Maximum proposals to return

        Returns:
            List of proposal dicts with content, metadata, and relevance_score
        """
        try:
            from config.app_config import CODE_PROPOSALS_PROMPT_ENABLED
            if not CODE_PROPOSALS_PROMPT_ENABLED:
                return []

            # Lazy-init ProposalFilter
            if not hasattr(self, '_proposal_filter'):
                self._proposal_filter = None

            if self._proposal_filter is None:
                from .proposal_filter import ProposalFilter
                chroma = getattr(self.memory_coordinator, 'chroma_store', None)
                self._proposal_filter = ProposalFilter(
                    chroma_store=chroma,
                    gate_system=self._gate_system,
                    model_manager=self.model_manager,
                )

            proposals = await self._proposal_filter.get_proposals(query, limit=limit)

            # Track for citations
            for idx, prop in enumerate(proposals[:limit], start=1):
                prop_id = f"PROPOSAL_{idx}"
                meta = prop.get('metadata', {})
                self.memory_id_map[prop_id] = {
                    'type': 'code_proposal',
                    'timestamp': str(meta.get('created_at', '')),
                    'content': prop.get('content', '')[:500],
                    'relevance_score': prop.get('relevance_score', 0.0),
                    'title': meta.get('title', ''),
                    'priority': meta.get('priority', 5),
                    'db_id': meta.get('proposal_id', None),
                }

            logger.info(f"[ContextGatherer] Retrieved {len(proposals)} proposed features")
            return proposals or []

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get proposed features: {e}", exc_info=True)
            return []

    async def get_procedural_skills(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant procedural skills (adaptive workflows) from procedural_skills collection.

        Uses hybrid retrieval via MemoryCoordinator.get_skills().

        Args:
            query: Search query for semantic retrieval
            limit: Maximum skills to return

        Returns:
            List of skill dicts with content, metadata, and relevance_score
        """
        try:
            from config.app_config import PROCEDURAL_SKILLS_ENABLED
            if not PROCEDURAL_SKILLS_ENABLED:
                return []

            if not hasattr(self.memory_coordinator, 'get_skills'):
                return []

            skills = await self.memory_coordinator.get_skills(query, limit=limit)

            # Track for citations
            for idx, skill in enumerate(skills[:limit], start=1):
                skill_id = f"SKILL_{idx}"
                meta = skill.get('metadata', {})
                self.memory_id_map[skill_id] = {
                    'type': 'procedural_skill',
                    'timestamp': meta.get('created_at', ''),
                    'content': meta.get('trigger', '')[:500],
                    'relevance_score': skill.get('relevance_score', 0.0),
                    'category': meta.get('category', ''),
                    'db_id': skill.get('id', None),
                }

            logger.debug(f"[ContextGatherer] Retrieved {len(skills)} procedural skills")
            return skills or []

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get procedural skills: {e}")
            return []

    async def get_graph_context(self, query: str, max_sentences: int = 12) -> List[str]:
        """Retrieve knowledge graph context for entities mentioned in the query.

        Extracts entities from the query, traverses their graph neighborhood,
        and returns natural language sentences describing relationships.

        Args:
            query: User query to extract entities from
            max_sentences: Maximum sentences to return

        Returns:
            List of natural language relationship sentences
        """
        try:
            from config.app_config import KNOWLEDGE_GRAPH_ENABLED, KNOWLEDGE_GRAPH_RETRIEVAL_DEPTH, ENABLE_GRAPH_ATTRIBUTION
            if not KNOWLEDGE_GRAPH_ENABLED:
                return []

            mc = self.memory_coordinator
            graph = getattr(mc, "graph_memory", None)
            resolver = getattr(mc, "entity_resolver", None)
            if not graph or not resolver or graph.node_count() == 0:
                return []

            # Extract entity mentions from query by checking each word/phrase
            # against the alias index
            sentences: list[str] = []
            seen_entities: set[str] = set()

            # Try multi-word then single-word resolution
            words = query.lower().split()
            candidates: list[str] = []
            # Check bigrams and trigrams first
            for n in (3, 2):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i + n])
                    candidates.append(phrase)
            # Then single words (skip stopwords)
            _STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "do", "does",
                          "did", "have", "has", "had", "what", "who", "where", "when",
                          "how", "why", "about", "with", "from", "for", "and", "or",
                          "but", "not", "to", "in", "on", "at", "of", "my", "your",
                          "i", "me", "you", "we", "they", "it", "this", "that", "can",
                          "will", "would", "should", "could", "tell", "know", "think"}
            for w in words:
                if w not in _STOPWORDS and len(w) > 2:
                    candidates.append(w)

            for mention in candidates:
                eid = resolver.resolve(mention)
                if eid and eid not in seen_entities:
                    seen_entities.add(eid)
                    ctx = graph.get_context_sentences(
                        eid, depth=KNOWLEDGE_GRAPH_RETRIEVAL_DEPTH,
                        max_sentences=max_sentences - len(sentences),
                        with_attribution=ENABLE_GRAPH_ATTRIBUTION,
                    )

                    # Track graph sentences in memory_id_map for citation
                    for i, sentence in enumerate(ctx):
                        citation_id = f"GRAPH_REL_{len(self.memory_id_map) + 1}"
                        self.memory_id_map[citation_id] = {
                            "content": sentence,
                            "entity_id": eid,
                            "query_mention": mention,
                            "source_type": "graph_relationship",
                            "metadata": {"entity": eid, "mention": mention}
                        }

                    sentences.extend(ctx)
                    if len(sentences) >= max_sentences:
                        break

            if sentences:
                logger.debug(
                    f"[ContextGatherer] Graph context: {len(sentences)} sentences "
                    f"for entities {seen_entities}"
                )
            return sentences[:max_sentences]

        except Exception as e:
            logger.warning(f"[ContextGatherer] Graph context retrieval failed: {e}")
            return []

    async def get_unresolved_threads(self, max_results: int = 3) -> List[Dict[str, Any]]:
        """Get top priority unresolved threads for session surfacing.

        Delegates to MemoryCoordinator.get_unresolved_threads().

        Args:
            max_results: Maximum threads to return

        Returns:
            List of thread dicts with topic, summary, thread_type, urgency, deadline_date
        """
        try:
            from config.app_config import THREAD_SURFACING_ENABLED
            if not THREAD_SURFACING_ENABLED:
                return []

            if not hasattr(self.memory_coordinator, 'get_unresolved_threads'):
                return []

            threads = self.memory_coordinator.get_unresolved_threads(max_results=max_results)
            logger.debug(f"[ContextGatherer] Retrieved {len(threads)} unresolved threads")
            return threads or []

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get unresolved threads: {e}")
            return []

    async def get_proactive_insights(self, query: str, max_insights: int = 2) -> List[str]:
        """Get cross-domain proactive insights from the knowledge graph.

        Delegates to MemoryCoordinator.context_surfacer.generate_insights().

        Args:
            query: Current user query
            max_insights: Maximum insights to return

        Returns:
            List of insight text strings for prompt injection
        """
        try:
            from config.app_config import PROACTIVE_SURFACING_ENABLED, ENABLE_INSIGHT_ATTRIBUTION
            if not PROACTIVE_SURFACING_ENABLED:
                return []

            surfacer = getattr(self.memory_coordinator, 'context_surfacer', None)
            if not surfacer:
                return []

            raw_insights = await surfacer.generate_insights(query, max_insights=max_insights)

            # Add attribution markers and track in memory_id_map for citation
            attributed_insights = []
            for i, insight_text in enumerate(raw_insights):
                # Add attribution marker if enabled
                if ENABLE_INSIGHT_ATTRIBUTION:
                    attributed_text = f"Analysis suggests: {insight_text}"
                else:
                    attributed_text = insight_text

                # Track in memory_id_map for citation system
                citation_id = f"AI_INSIGHT_{len(self.memory_id_map) + 1}"
                self.memory_id_map[citation_id] = {
                    "content": attributed_text,
                    "original_insight": insight_text,
                    "source_type": "ai_synthesis",
                    "query": query,
                    "metadata": {"insight_index": i, "query": query}
                }

                attributed_insights.append(attributed_text)

            logger.debug(f"[ContextGatherer] Retrieved {len(attributed_insights)} proactive insights")
            return attributed_insights

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get proactive insights: {e}")
            return []

    async def get_codebase_changes(self, since_datetime) -> Dict[str, Any]:
        """Detect codebase file changes since last session via git.

        Runs git log, git diff, and git status to identify committed and
        uncommitted changes, filtering by allowed extensions and excluding
        build artifacts.

        Args:
            since_datetime: datetime object or None. If None, returns empty.

        Returns:
            Dict with committed, uncommitted_modified, uncommitted_new,
            since_label keys. Empty dict on failure or when disabled.
        """
        try:
            from config.app_config import (
                SESSION_DIFF_ENABLED,
                SESSION_DIFF_MAX_COMMITTED,
                SESSION_DIFF_MAX_UNCOMMITTED,
                SESSION_DIFF_EXTENSIONS,
            )
            if not SESSION_DIFF_ENABLED or since_datetime is None:
                return {}

            import subprocess
            from datetime import datetime

            # Resolve the repo root
            repo_root = None
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    repo_root = result.stdout.strip()
            except Exception:
                return {}
            if not repo_root:
                return {}

            # Exclusion patterns for paths
            _EXCLUDE_PATTERNS = ("__pycache__", ".pyc", "venv/", "dist/", "build/", ".egg-info")

            def _ext_ok(path: str) -> bool:
                """Check if file extension is in the allowed list."""
                import os as _os
                _, ext = _os.path.splitext(path)
                return ext.lower() in SESSION_DIFF_EXTENSIONS

            def _path_ok(path: str) -> bool:
                """Check if path is not in exclusion patterns."""
                return not any(excl in path for excl in _EXCLUDE_PATTERNS)

            def _filter(paths: list) -> list:
                return [p for p in paths if _ext_ok(p) and _path_ok(p)]

            # 1) Committed changes since last session
            iso_since = since_datetime.isoformat() if hasattr(since_datetime, 'isoformat') else str(since_datetime)
            committed = []
            try:
                result = subprocess.run(
                    ["git", "log", f"--since={iso_since}", "--oneline", "--no-merges"],
                    capture_output=True, text=True, timeout=10, cwd=repo_root
                )
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split("\n")
                    committed = lines[:SESSION_DIFF_MAX_COMMITTED]
            except Exception as e:
                logger.debug(f"[ContextGatherer] git log failed: {e}")

            # 2) Uncommitted modified files
            uncommitted_modified = []
            try:
                result = subprocess.run(
                    ["git", "diff", "--name-only"],
                    capture_output=True, text=True, timeout=10, cwd=repo_root
                )
                if result.returncode == 0 and result.stdout.strip():
                    files = result.stdout.strip().split("\n")
                    uncommitted_modified = _filter(files)[:SESSION_DIFF_MAX_UNCOMMITTED]
            except Exception as e:
                logger.debug(f"[ContextGatherer] git diff failed: {e}")

            # 3) Untracked new files
            uncommitted_new = []
            try:
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True, text=True, timeout=10, cwd=repo_root
                )
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split("\n")
                    untracked = [line[3:].strip() for line in lines if line.startswith("??")]
                    uncommitted_new = _filter(untracked)[:SESSION_DIFF_MAX_UNCOMMITTED]
            except Exception as e:
                logger.debug(f"[ContextGatherer] git status failed: {e}")

            # Human-readable time delta
            since_label = "last session"
            try:
                if hasattr(since_datetime, 'timestamp'):
                    now = datetime.now()
                    delta = now - since_datetime
                    total_secs = int(delta.total_seconds())
                    if total_secs < 3600:
                        since_label = f"{total_secs // 60}m ago"
                    elif total_secs < 86400:
                        hours = total_secs // 3600
                        mins = (total_secs % 3600) // 60
                        since_label = f"{hours}h {mins}m ago"
                    else:
                        days = total_secs // 86400
                        hours = (total_secs % 86400) // 3600
                        since_label = f"{days}d {hours}h ago"
            except Exception:
                pass

            if not committed and not uncommitted_modified and not uncommitted_new:
                return {}

            return {
                "committed": committed,
                "uncommitted_modified": uncommitted_modified,
                "uncommitted_new": uncommitted_new,
                "since_label": since_label,
            }

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get codebase changes: {e}")
            return {}

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
        This bridges vocabulary gaps (e.g. "my brother" → "dillion").

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
            # FAST MODE: Reduce retrieval pool 50x (2150 → ~45) for 15x speed boost
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
            'yes', 'no', 'lol', 'haha', 'good', 'great', 'nice', 'cool',
            'how are you', 'what\'s up', 'see you', 'bye', 'goodbye',
            'yeah', 'yep', 'nope', 'sure', 'alright', 'sounds good',
            'i think', 'i feel', 'i hope', 'i guess', 'i mean',
            'that\'s', 'it\'s', 'i\'m', 'i am', 'going to', 'gonna',
        ]

        if any(pattern in query_lower for pattern in conversational_patterns):
            return True

        # Skip if query is mostly short words (< 4 chars)
        long_words = [w for w in words if len(w) >= 4 and w.isalpha()]
        if len(long_words) < 2:
            return True

        return False

    async def _get_wiki_content(self, query: str, limit: int = PROMPT_MAX_WIKI) -> List[Dict[str, Any]]:
        """Get wiki content for query.

        Prefers the local wiki_knowledge ChromaDB collection (pre-embedded
        Wikipedia corpus) for fast, relevant semantic retrieval.  Falls back
        to live Wikipedia API if the collection is empty or unavailable.
        """
        if not query:
            return []

        # Smart skip for simple/conversational queries
        if self._should_skip_wikipedia(query):
            return []

        # --- Try local ChromaDB wiki_knowledge first ---
        chroma = getattr(self.memory_coordinator, 'chroma_store', None)
        if chroma:
            try:
                coll = chroma.collections.get('wiki_knowledge')
                if coll and coll.count() > 0:
                    results = chroma.query_collection(
                        'wiki_knowledge', query, n_results=limit
                    )
                    if results:
                        # Track wiki titles for session enrichment
                        from knowledge.wiki_tracker import WikiArticleTracker
                        for r in results:
                            t = r.get('metadata', {}).get('title', '')
                            if t:
                                WikiArticleTracker.get_instance().track(t, r.get('content', '')[:500])
                        return [
                            {
                                'content': r.get('content', ''),
                                'metadata': r.get('metadata', {}),
                                'relevance_score': r.get('relevance_score', 0.0),
                                'source': 'wiki_knowledge',
                            }
                            for r in results
                        ]
            except Exception as e:
                logger.debug(f"[ContextGatherer] wiki_knowledge query failed, falling back to API: {e}")

        # --- Fallback: live Wikipedia API ---
        try:
            search_terms = []
            words = query.lower().split()
            for word in words:
                if len(word) > 3 and word.isalpha():
                    search_terms.append(word)

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

            # Filter by similarity threshold to prevent irrelevant wiki content
            pre_gate = len(results)
            results = [r for r in results
                       if r.get("similarity", 0) >= SEMANTIC_CHUNKS_GATE_THRESHOLD]
            if pre_gate != len(results):
                logger.debug(f"Semantic chunks gate: {pre_gate} -> {len(results)} (threshold={SEMANTIC_CHUNKS_GATE_THRESHOLD})")
            if not results:
                return []

            # Track wiki titles for session enrichment
            from knowledge.wiki_tracker import WikiArticleTracker
            tracker = WikiArticleTracker.get_instance()
            for r in results:
                t = r.get("title", "")
                if t:
                    tracker.track(t, r.get("content", "")[:500])

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

    async def _get_web_search_results(
        self,
        query: str,
        crisis_level: Optional[str] = None,
        intent_type: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Get web search results if the query triggers a search.

        Uses LLM-first trigger analysis to determine if search is needed,
        and uses LLM-optimized search_terms for better results.

        Args:
            query: User query to analyze and potentially search
            crisis_level: Current tone/crisis level (HIGH/MEDIUM suppresses search)
            intent_type: Intent classifier result (e.g. "casual_social") — skips search for non-search intents

        Returns:
            WebSearchResult if search was triggered and successful, None otherwise
        """
        # Check if web search is enabled
        if not WEB_SEARCH_ENABLED:
            logger.debug("[ContextGatherer] Web search disabled in config")
            return None

        # Skip web search for intents that never need it
        _no_search_intents = {"casual_social", "meta_conversational", "emotional_support", "general"}
        if intent_type and str(intent_type) in _no_search_intents:
            logger.debug(f"[ContextGatherer] Web search skipped for intent={intent_type}")
            return None

        # Check crisis suppression (also done in trigger, but early exit saves time)
        if crisis_level and crisis_level.upper() in ("HIGH", "MEDIUM"):
            logger.debug(f"[ContextGatherer] Web search suppressed during {crisis_level} crisis")
            return None

        # Check if web search manager is available
        manager = self.web_search_manager
        if not manager:
            logger.warning("[ContextGatherer] Web search manager failed to initialize")
            return None
        if not manager.is_available():
            logger.debug("[ContextGatherer] Web search not available (API key missing or invalid)")
            return None

        try:
            # Use LLM-first trigger if available, otherwise fall back to heuristics
            trigger_llm = self.web_search_trigger_llm
            if trigger_llm and self.model_manager:
                # Get remaining credits for credit-aware search planning
                remaining_credits = 100.0  # Default
                if hasattr(manager, 'rate_limiter') and manager.rate_limiter:
                    remaining_credits = manager.rate_limiter.get_remaining_credits()

                logger.debug("[WebSearch] Using LLM-first trigger analysis...")
                decision = await trigger_llm(
                    query=query,
                    model_manager=self.model_manager,
                    crisis_level=crisis_level,
                    web_search_enabled=WEB_SEARCH_ENABLED,
                    remaining_credits=remaining_credits
                )
            else:
                # Fallback to sync heuristic trigger
                logger.warning("[WebSearch] LLM trigger not available, using heuristics...")
                trigger = self.web_search_trigger
                if not trigger:
                    logger.warning("[ContextGatherer] Web search trigger not available")
                    return None
                decision = trigger(query)

            if not decision.should_search:
                logger.debug(
                    f"[ContextGatherer] Web search not triggered: {decision.reason} "
                    f"(confidence={decision.confidence:.2f}, source={getattr(decision, 'source', 'unknown')})"
                )
                return None

            logger.info(
                f"[ContextGatherer] Web search triggered: {decision.reason} "
                f"(confidence={decision.confidence:.2f}, depth={decision.depth.value}, "
                f"source={getattr(decision, 'source', 'unknown')})"
            )

            # Import the depth enum from the manager
            from knowledge.web_search_manager import WebSearchDepth as ManagerDepth

            # Map trigger depth to manager depth
            depth_map = {
                "quick": ManagerDepth.QUICK,
                "standard": ManagerDepth.STANDARD,
                "deep": ManagerDepth.DEEP,
            }
            search_depth = depth_map.get(decision.depth.value, ManagerDepth.STANDARD)

            # Use LLM-optimized search_terms if available, otherwise use original query
            search_terms = getattr(decision, 'search_terms', [])
            if search_terms:
                logger.info(f"[ContextGatherer] Using LLM-optimized search terms: {search_terms}")
                # Execute search with optimized terms (bypass auto_decompose since LLM already did this)
                # Use first LLM term as primary query; skip auto_decompose since LLM already optimized
                result = await manager.multi_search(
                    query=search_terms[0],
                    depth=search_depth,
                    crisis_level=crisis_level,
                    timeout=WEB_SEARCH_TIMEOUT,
                    use_cache=True,
                    auto_decompose=False
                )
            else:
                # No LLM search terms, use original query with auto-decompose
                result = await manager.multi_search(
                    query=query,
                    depth=search_depth,
                    crisis_level=crisis_level,
                    timeout=WEB_SEARCH_TIMEOUT,
                    use_cache=True,
                    auto_decompose=True  # Enable automatic query decomposition
                )

            if result.has_results:
                decomp_info = ""
                if hasattr(result, 'decomposition_used') and result.decomposition_used:
                    decomp_info = f", decomposed into {len(result.sub_queries)} sub-queries"
                logger.info(
                    f"[ContextGatherer] Web search returned {len(result.pages)} results "
                    f"(credits={result.total_credits_used}, cached={result.from_cache}{decomp_info})"
                )

                # Track web search results for citations
                self.memory_id_map["WEB_SEARCH"] = {
                    'type': 'web_search',
                    'timestamp': datetime.now().isoformat(),
                    'content': f"Web search for: {query[:100]}",
                    'relevance_score': decision.confidence,
                    'db_id': None,
                    'sources': [p.url for p in result.pages[:5]]
                }

                return result
            else:
                logger.debug(f"[ContextGatherer] Web search returned no results: {result.error}")
                return None

        except Exception as e:
            logger.warning(f"[ContextGatherer] Web search failed: {e}")
            return None

    def should_trigger_web_search(self, query: str, crisis_level: Optional[str] = None) -> bool:
        """
        Quick check to determine if a query should trigger web search.

        Useful for pre-checking before gathering context.

        Args:
            query: User query
            crisis_level: Current crisis level

        Returns:
            True if web search should be triggered
        """
        if not WEB_SEARCH_ENABLED:
            return False

        if crisis_level and crisis_level.upper() in ("HIGH", "MEDIUM"):
            return False

        trigger = self.web_search_trigger
        if not trigger:
            return False

        try:
            decision = trigger(query)
            return decision.should_search
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Web search trigger check failed: {e}")
            return False

    def get_narrative_context(self) -> str:
        """
        Retrieve the cached narrative context (temporal grounding).

        This reads the pre-synthesized narrative from the filesystem.
        The narrative is generated asynchronously during summary creation
        and cached to avoid per-query latency costs.

        Returns:
            The narrative context string, or empty string if not available.
        """
        try:
            from config.app_config import NARRATIVE_CONTEXT_ENABLED
            if not NARRATIVE_CONTEXT_ENABLED:
                return ""

            # Access corpus manager through memory coordinator
            if hasattr(self.memory_coordinator, 'corpus_manager'):
                corpus = self.memory_coordinator.corpus_manager
                if hasattr(corpus, 'get_narrative_context'):
                    narrative = corpus.get_narrative_context()
                    if narrative:
                        logger.debug(f"[ContextGatherer] Retrieved narrative context ({len(narrative)} chars)")
                    return narrative

            logger.debug("[ContextGatherer] Narrative context not available (no corpus manager)")
            return ""

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to retrieve narrative context: {e}")
            return ""
