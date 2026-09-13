"""
# core/prompt/gatherer_web.py

Mixin providing web search retrieval methods for ContextGatherer.

Methods:
  - _get_web_search_results(query, crisis_level, intent_type, conversation_context)
      -> WebSearchResult or None
      (conversation_context: prior-turn digest so elliptical follow-ups resolve
       against the current topic — mirrors the agentic gate)
  - should_trigger_web_search(query, crisis_level) -> bool

Depends on self.web_search_manager, self.web_search_trigger, self.web_search_trigger_llm,
self.model_manager, self.memory_id_map (set by ContextGatherer.__init__).
"""

import dataclasses
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


def _web_search_enabled() -> bool:
    """Live value of the web-search toggle (2026-09-09, audit F04).

    The Settings page flips ``config.app_config.WEB_SEARCH_ENABLED`` at
    runtime; a module-level ``from`` import froze the value at import time,
    so the already-running gatherer kept searching after the owner disabled
    it. Import-doctrine case 3: read the live attribute at call time.
    """
    try:
        import config.app_config as _cfg  # lazy import: live-config read
        return bool(getattr(_cfg, "WEB_SEARCH_ENABLED", WEB_SEARCH_ENABLED))
    except ImportError:
        return bool(WEB_SEARCH_ENABLED)


def _text_field(obj: Any, name: str) -> str:
    """A string attribute, or "". Trigger doubles in tests are mocks whose
    every attribute is truthy; a receipt must never mistake one for a real
    block reason."""
    value = getattr(obj, name, "")
    return value if isinstance(value, str) else ""


class WebSearchMixin:
    """Mixin providing web search retrieval methods."""

    async def _get_web_search_results(
        self,
        query: str,
        crisis_level: Optional[str] = None,
        intent_type: Optional[str] = None,
        conversation_context: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Get web search results if the query triggers a search.

        Uses LLM-first trigger analysis to determine if search is needed,
        and uses LLM-optimized search_terms for better results.

        Args:
            query: User query to analyze and potentially search
            crisis_level: Current tone/crisis level (HIGH/MEDIUM suppresses search)
            intent_type: Intent classifier result (e.g. "casual_social") — skips search for
                intents that are explicitly non-search; ambiguous fallback intents still
                consult the shared trigger
            conversation_context: Compact digest of prior turns so an elliptical
                follow-up ("they're only giving us 7 days") can be resolved to the
                topic just discussed. Without it a pronoun-only claim scores 0 on
                the standalone heuristic and never reaches the trigger LLM. The
                agentic gate already passes this; enhanced mode now does too.

        Returns:
            WebSearchResult if search was triggered and successful, None otherwise
        """
        self.last_web_decision = {
            "triggered": False,
            "source": None,
            "reason": None,
            "confidence": None,
            "results": None,
            "error": None,
            # Evidence receipt (2026-09-12, review F4) — see the update below.
            "requested": False,
            "blocked": None,
            "budget_remaining": None,
            "from_cache": False,
        }

        # Check if web search is enabled (live value — Settings can flip it)
        if not _web_search_enabled():
            self.last_web_decision["reason"] = "web search disabled"
            logger.debug("[ContextGatherer] Web search disabled in config")
            return None

        # Skip web search for intents that never need it
        # ``general`` is the classifier's low-confidence/unknown bucket.  It is
        # intentionally not a veto: unrecognized current-news questions can
        # land here, so the shared trigger must decide.
        _no_search_intents = {"casual_social", "meta_conversational", "emotional_support"}
        if intent_type and str(intent_type) in _no_search_intents:
            self.last_web_decision["reason"] = f"intent veto: {intent_type}"
            logger.debug(f"[ContextGatherer] Web search skipped for intent={intent_type}")
            return None

        # Check crisis suppression (also done in trigger, but early exit saves time)
        if crisis_level and crisis_level.upper() in ("HIGH", "MEDIUM"):
            self.last_web_decision["reason"] = f"crisis veto: {crisis_level.upper()}"
            logger.debug(f"[ContextGatherer] Web search suppressed during {crisis_level} crisis")
            return None

        # Check if web search manager is available
        manager = self.web_search_manager
        if not manager:
            self.last_web_decision["reason"] = "manager missing"
            logger.warning("[ContextGatherer] Web search manager failed to initialize")
            return None
        if not manager.is_available():
            self.last_web_decision["reason"] = "manager unavailable"
            logger.debug("[ContextGatherer] Web search not available (API key missing or invalid)")
            return None

        try:
            # This manager's live budget, read ONCE for both trigger paths —
            # the synchronous fallback below never saw it before 2026-09-12.
            remaining_credits = 100.0  # no limiter: unknown is not exhausted
            if hasattr(manager, 'rate_limiter') and manager.rate_limiter:
                _live = manager.rate_limiter.get_remaining_credits()
                if isinstance(_live, (int, float)) and not isinstance(_live, bool):
                    remaining_credits = float(_live)

            # Use LLM-first trigger if available, otherwise fall back to heuristics
            trigger_llm = self.web_search_trigger_llm
            if trigger_llm and self.model_manager:
                logger.debug("[WebSearch] Using LLM-first trigger analysis...")
                decision = await trigger_llm(
                    query=query,
                    model_manager=self.model_manager,
                    crisis_level=crisis_level,
                    web_search_enabled=_web_search_enabled(),
                    remaining_credits=remaining_credits,
                    conversation_context=conversation_context,
                )
            else:
                # Fallback to sync heuristic trigger
                logger.warning("[WebSearch] LLM trigger not available, using heuristics...")
                trigger = self.web_search_trigger
                if not trigger:
                    self.last_web_decision["reason"] = "trigger unavailable"
                    logger.warning("[ContextGatherer] Web search trigger not available")
                    return None
                decision = trigger(query)
                # The heuristic path bypassed the shared budget veto
                # (2026-09-12, adversarial review F3): at zero budget an
                # explicit "search" query still went to the provider path.
                if dataclasses.is_dataclass(decision):
                    from utils.web_search_trigger import apply_search_budget
                    decision = apply_search_budget(decision, remaining_credits)

            _blocked = _text_field(decision, "blocked_reason") or None
            _decision_budget = getattr(decision, "budget_remaining", None)
            self.last_web_decision.update({
                "triggered": bool(decision.should_search),
                "source": getattr(decision, "source", None),
                "reason": getattr(decision, "reason", None),
                "confidence": getattr(decision, "confidence", None),
                # Evidence receipt (2026-09-12, review F4): the need survives
                # a budget block, and the budget recorded is the one the
                # decision was made against — not a later limiter reading.
                "requested": bool(decision.should_search)
                or getattr(decision, "evidence_needed", False) is True,
                "blocked": _blocked,
                "budget_remaining": (
                    _decision_budget
                    if isinstance(_decision_budget, (int, float))
                    and not isinstance(_decision_budget, bool)
                    else remaining_credits
                ),
            })

            if not decision.should_search:
                if _blocked == "budget":
                    # A spent budget blocks NEW paid searches, not evidence
                    # already in the local cache — the veto used to sit in
                    # front of both.
                    cached = self._cached_web_evidence(manager, decision, query)
                    if cached is not None:
                        self.last_web_decision["results"] = len(cached.pages)
                        self.last_web_decision["from_cache"] = True
                        logger.info(
                            "[ContextGatherer] Search budget spent — using an "
                            "exact cached result (no provider call)"
                        )
                        return cached
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

            pages = list(getattr(result, "pages", None) or [])
            self.last_web_decision["results"] = len(pages)

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
                if getattr(result, "error", None):
                    self.last_web_decision["error"] = str(result.error)
                logger.debug(f"[ContextGatherer] Web search returned no results: {result.error}")
                return None

        except Exception as e:
            self.last_web_decision["error"] = type(e).__name__
            logger.warning(f"[ContextGatherer] Web search failed: {e}")
            return None

    @staticmethod
    def _cached_web_evidence(manager: Any, decision: Any, query: str) -> Optional[Any]:
        """An exact cached result for a search the budget blocked, or None.

        WebSearchManager.search checks its cache BEFORE the limiter, so a
        spent budget never blocked cached evidence until the 2026-09-12
        trigger veto started returning early. This restores that path without
        ever reaching the provider: the terms the veto withheld (or the query
        itself), localized exactly as search() would key them, tried at every
        depth."""
        cache = getattr(manager, "cache", None)
        if cache is None or not hasattr(cache, "get"):
            return None
        from knowledge.web_search_manager import WebSearchDepth as ManagerDepth
        wanted = getattr(getattr(decision, "depth", None), "value", "")
        depths = sorted(ManagerDepth, key=lambda d: d.value != wanted)
        terms = [t for t in (getattr(decision, "blocked_search_terms", None) or [])
                 if isinstance(t, str) and t.strip()] or [query]
        localize = getattr(manager, "_localize_query", None)
        for term in terms:
            key = localize(term) if callable(localize) else term
            for depth in depths:
                try:
                    hit = cache.get(key, depth)
                except Exception:
                    hit = None
                if hit is not None and getattr(hit, "has_results", False) is True:
                    return hit
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
        if not _web_search_enabled():
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
