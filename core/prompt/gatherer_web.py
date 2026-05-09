"""
# core/prompt/gatherer_web.py

Mixin providing web search retrieval methods for ContextGatherer.

Methods:
  - _get_web_search_results(query, crisis_level, intent_type) -> WebSearchResult or None
  - should_trigger_web_search(query, crisis_level) -> bool

Depends on self.web_search_manager, self.web_search_trigger, self.web_search_trigger_llm,
self.model_manager, self.memory_id_map (set by ContextGatherer.__init__).
"""

import os
import logging
from typing import Optional, Any
from datetime import datetime

logger = logging.getLogger("prompt_context_gatherer")

# Web search configuration
try:
    from config.app_config import (
        WEB_SEARCH_ENABLED,
        WEB_SEARCH_TIMEOUT,
        WEB_SEARCH_MAX_CONTENT_CHARS,
        WEB_SEARCH_API_KEY,
        WEB_SEARCH_DAILY_CREDIT_LIMIT,
    )
except ImportError:
    WEB_SEARCH_ENABLED = True
    WEB_SEARCH_TIMEOUT = 30.0
    WEB_SEARCH_MAX_CONTENT_CHARS = 10000
    WEB_SEARCH_API_KEY = os.getenv("TAVILY_API_KEY", "")
    WEB_SEARCH_DAILY_CREDIT_LIMIT = 100


class WebSearchMixin:
    """Mixin providing web search retrieval methods."""

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
                result = await manager.multi_search(
                    query=search_terms[0],
                    depth=search_depth,
                    crisis_level=crisis_level,
                    timeout=WEB_SEARCH_TIMEOUT,
                    use_cache=True,
                    auto_decompose=False,
                    sub_queries=search_terms if len(search_terms) > 1 else None,
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
