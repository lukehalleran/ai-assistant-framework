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
    [passes all LLM search terms via sub_queries for parallel search]
  - _get_dreams(limit) -> List[Dict]
  - get_facts(query, limit) / get_recent_facts(limit) -> List[Dict]
  - _expand_query_with_graph(query, max_terms) -> str  [appends graph neighbor names]
  - _apply_gating(memories, query) -> List[Dict]  [multi-stage gate system]
  - _deduplicate_memories(memories) -> List[Dict]
- Outputs:
  - Individual retrieval results returned to builder for parallel assembly
  - memory_id_map: Dict tracking doc_id -> content for citation provenance
  - Narrative context string (synthesized life state) [NEW 2026-01-17]
- Behavior:
  - Retrieves data from multiple memory collections (episodic, semantic, procedural)
  - Applies filtering and relevance scoring to retrieved content
  - Implements caching for expensive operations (wiki lookups, semantic search, web search)
  - Coordinates parallel data fetching with timeout management
  - Handles graceful fallbacks when data sources are unavailable
  - Triggers web search using LLM-first analysis with heuristic fallback
  - ENHANCED: LLM generates optimized search_terms for better results
  - ENHANCED: Complex queries are auto-decomposed into parallel sub-queries (e.g., "Compare Tesla and Rivian" -> 2 searches)
  - Suppresses web search during HIGH/MEDIUM crisis levels
  - UPDATED: Uses UserProfile hybrid retrieval (2/3 semantic + 1/3 recent per category) instead of flat facts
  - Retrieves cached narrative context (0ms latency) for temporal grounding [NEW 2026-01-17]
- Architecture:
  - ContextGatherer inherits from three mixins that provide domain-specific methods:
    - WebSearchMixin (gatherer_web.py): web search trigger + result retrieval
    - MemoryRetrievalMixin (gatherer_memory.py): conversations, semantic, summaries, reflections, facts, profile
    - KnowledgeRetrievalMixin (gatherer_knowledge.py): notes, docs, uploads, wiki, git, graph, threads, insights
  - This file retains __init__, lazy properties, and shared utility methods
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
from typing import Dict, List, Any
from utils.time_manager import TimeManager
from utils.logging_utils import get_logger
from core.wiki_util import clean_query
from .formatter import _parse_bool

# Mixin imports
from .gatherer_web import WebSearchMixin
from .gatherer_memory import MemoryRetrievalMixin
from .gatherer_knowledge import KnowledgeRetrievalMixin

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

class ContextGatherer(WebSearchMixin, MemoryRetrievalMixin, KnowledgeRetrievalMixin):
    """Handles all data collection and retrieval for prompt building.

    Composed from three domain-specific mixins:
    - WebSearchMixin: web search trigger and result retrieval
    - MemoryRetrievalMixin: conversations, semantic memories, summaries, reflections, facts, profile
    - KnowledgeRetrievalMixin: notes, docs, uploads, wiki, git, graph, threads, insights, dreams
    """

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

    def _bounded(self, items: List[Any], max_items: int) -> List[Any]:
        """Helper to bound list length."""
        return items[:max_items] if len(items) > max_items else items
