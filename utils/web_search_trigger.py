# /utils/web_search_trigger.py
"""
WebSearchTrigger - Detects when a query should trigger web search.

Module Contract:
- Purpose: Classify queries to determine if they need real-time web information
- Inputs:
  - Query text, optional conversation context
- Outputs:
  - WebSearchDecision with should_search flag and recommended depth
- Side effects:
  - None (pure classification)

Detection Strategy:
1. Keyword matching for recency indicators ("latest", "current", "2024", etc.)
2. Pattern matching for news/event queries
3. Entity detection for fast-changing topics (stocks, weather, sports scores)
4. Explicit search requests ("search for", "look up online")
5. LLM-based classification for ambiguous cases (optional)
"""

import asyncio
import os
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Set, Tuple

from utils.logging_utils import get_logger

logger = get_logger("web_search_trigger")


class WebSearchDepth(Enum):
    """Search depth levels - mirrored from web_search_manager for decoupling."""
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


@dataclass
class WebSearchDecision:
    """Result of web search trigger analysis."""
    should_search: bool
    depth: WebSearchDepth
    confidence: float  # 0.0-1.0
    reason: str
    matched_keywords: List[str]
    matched_patterns: List[str]


# ===== Configuration =====

# Minimum confidence to trigger search
SEARCH_CONFIDENCE_THRESHOLD = float(os.getenv("WEB_SEARCH_CONFIDENCE_THRESHOLD", "0.5"))

# LLM classification settings
SEARCH_TRIGGER_MODEL = os.getenv("WEB_SEARCH_TRIGGER_MODEL", "gpt-4o-mini")
SEARCH_TRIGGER_TIMEOUT = float(os.getenv("WEB_SEARCH_TRIGGER_TIMEOUT", "2.0"))
SEARCH_TRIGGER_MAX_TOKENS = int(os.getenv("WEB_SEARCH_TRIGGER_MAX_TOKENS", "20"))


# ===== Keyword Sets =====

# Strong recency indicators - high confidence triggers
RECENCY_KEYWORDS_STRONG: Set[str] = {
    "latest", "newest", "most recent", "current", "today",
    "this week", "this month", "right now", "breaking",
    "just announced", "just released", "just happened",
    "happening now", "live", "real-time", "realtime",
}

# Moderate recency indicators
RECENCY_KEYWORDS_MODERATE: Set[str] = {
    "recent", "new", "updated", "modern", "contemporary",
    "this year", "last week", "yesterday", "recently",
    "nowadays", "these days", "currently",
}

# Year patterns that suggest recency needs
CURRENT_YEAR = datetime.now().year
RECENT_YEARS = {str(CURRENT_YEAR), str(CURRENT_YEAR - 1)}

# Explicit search request phrases
EXPLICIT_SEARCH_PHRASES: Tuple[str, ...] = (
    "search for", "look up", "find online", "search online",
    "google", "search the web", "web search", "internet search",
    "find me information", "find information about",
    "what's happening with", "what's going on with",
    "check online", "look online",
)

# News and event indicators
NEWS_KEYWORDS: Set[str] = {
    "news", "headline", "headlines", "story", "stories",
    "reported", "reports", "announced", "announcement",
    "update", "updates", "press release", "breaking",
    "coverage", "article", "articles",
}

# Fast-changing topics that often need fresh data
FAST_CHANGING_TOPICS: Set[str] = {
    # Financial
    "stock", "stocks", "share price", "stock price", "market",
    "bitcoin", "crypto", "cryptocurrency", "ethereum", "btc", "eth",
    "exchange rate", "forex", "trading",

    # Weather
    "weather", "forecast", "temperature", "rain", "storm",
    "hurricane", "tornado", "snow", "climate",

    # Sports
    "score", "scores", "game", "match", "tournament",
    "championship", "playoffs", "standings", "roster",
    "injury", "trade", "transfer",

    # Tech/Product
    "release date", "launch date", "availability",
    "version", "update", "patch", "beta", "alpha",

    # Events
    "election", "vote", "voting", "poll", "polls",
    "concert", "event", "festival", "conference",
    "sale", "discount", "deal", "price drop",
}

# Topics that typically DON'T need web search (knowledge-based)
STATIC_TOPICS: Set[str] = {
    # Historical/factual
    "history", "historical", "ancient", "medieval",
    "biography", "born", "died", "founded",

    # Scientific concepts
    "theory", "theorem", "formula", "equation",
    "chemistry", "physics", "biology", "mathematics",
    "element", "compound", "molecule", "atom",

    # Definitions/concepts
    "definition", "meaning", "what is", "explain",
    "concept", "principle", "law of",

    # How-to/procedural
    "how to", "how do i", "tutorial", "guide",
    "recipe", "instructions", "steps",

    # Creative/personal
    "write me", "create", "generate", "compose",
    "help me with", "advice on", "opinion",
}

# Suppression patterns - don't search for these
SUPPRESSION_PATTERNS: Tuple[str, ...] = (
    # Personal/conversational
    "how are you", "how do you feel", "what do you think",
    "tell me about yourself", "who are you",

    # Memory/context queries
    "remember when", "do you remember", "recall",
    "we talked about", "you mentioned", "earlier",

    # Emotional/therapeutic
    "feeling", "I feel", "I'm feeling", "i am feeling",
    "stressed", "anxious", "depressed", "sad", "happy",
    "need to talk", "can we talk", "listen to me",
)


def _normalize(text: str) -> str:
    """Normalize text for matching."""
    return (text or "").strip().lower()


def _contains_year(text: str, years: Set[str]) -> bool:
    """Check if text contains any of the specified years."""
    text_lower = _normalize(text)
    for year in years:
        # Match year as whole word to avoid false positives
        if re.search(rf'\b{year}\b', text_lower):
            return True
    return False


def _count_keyword_matches(text: str, keywords: Set[str]) -> Tuple[int, List[str]]:
    """Count keyword matches and return matched keywords."""
    text_lower = _normalize(text)
    matched = []
    for kw in keywords:
        if kw in text_lower:
            matched.append(kw)
    return len(matched), matched


def _matches_phrase(text: str, phrases: Tuple[str, ...]) -> Tuple[bool, List[str]]:
    """Check if text starts with or contains any phrase."""
    text_lower = _normalize(text)
    matched = []
    for phrase in phrases:
        if phrase in text_lower:
            matched.append(phrase)
    return len(matched) > 0, matched


def should_search_heuristic(query: str) -> WebSearchDecision:
    """
    Determine if query needs web search using heuristics only.

    Scoring approach:
    - Explicit search phrases: +0.6 confidence
    - Strong recency keywords: +0.4 per match (max 2)
    - Moderate recency keywords: +0.2 per match (max 2)
    - Current year mention: +0.3
    - News keywords: +0.2 per match (max 2)
    - Fast-changing topics: +0.3 per match (max 2)
    - Static topics: -0.3 per match
    - Suppression patterns: -0.5

    Returns:
        WebSearchDecision with confidence and recommendation
    """
    if not query:
        return WebSearchDecision(
            should_search=False,
            depth=WebSearchDepth.QUICK,
            confidence=0.0,
            reason="Empty query",
            matched_keywords=[],
            matched_patterns=[]
        )

    confidence = 0.0
    all_matched_keywords: List[str] = []
    all_matched_patterns: List[str] = []
    reasons: List[str] = []

    query_lower = _normalize(query)

    # Check suppression patterns first
    suppressed, supp_matches = _matches_phrase(query, SUPPRESSION_PATTERNS)
    if suppressed:
        return WebSearchDecision(
            should_search=False,
            depth=WebSearchDepth.QUICK,
            confidence=0.0,
            reason=f"Suppressed: matches personal/conversational pattern",
            matched_keywords=[],
            matched_patterns=supp_matches
        )

    # Check explicit search phrases (strongest signal)
    has_explicit, explicit_matches = _matches_phrase(query, EXPLICIT_SEARCH_PHRASES)
    if has_explicit:
        confidence += 0.6
        all_matched_patterns.extend(explicit_matches)
        reasons.append("explicit search request")

    # Strong recency keywords
    strong_count, strong_matches = _count_keyword_matches(query, RECENCY_KEYWORDS_STRONG)
    if strong_count > 0:
        confidence += min(strong_count * 0.4, 0.8)
        all_matched_keywords.extend(strong_matches)
        reasons.append(f"{strong_count} strong recency keyword(s)")

    # Moderate recency keywords
    mod_count, mod_matches = _count_keyword_matches(query, RECENCY_KEYWORDS_MODERATE)
    if mod_count > 0:
        confidence += min(mod_count * 0.2, 0.4)
        all_matched_keywords.extend(mod_matches)
        reasons.append(f"{mod_count} moderate recency keyword(s)")

    # Current/recent year mention
    if _contains_year(query, RECENT_YEARS):
        confidence += 0.3
        all_matched_keywords.append(f"year:{list(RECENT_YEARS)}")
        reasons.append("mentions current/recent year")

    # News keywords
    news_count, news_matches = _count_keyword_matches(query, NEWS_KEYWORDS)
    if news_count > 0:
        confidence += min(news_count * 0.2, 0.4)
        all_matched_keywords.extend(news_matches)
        reasons.append(f"{news_count} news keyword(s)")

    # Fast-changing topics
    fast_count, fast_matches = _count_keyword_matches(query, FAST_CHANGING_TOPICS)
    if fast_count > 0:
        confidence += min(fast_count * 0.3, 0.6)
        all_matched_keywords.extend(fast_matches)
        reasons.append(f"{fast_count} fast-changing topic(s)")

    # Static topics (reduce confidence) - but only if no strong positive signals
    # Strong positive signals: strong recency keywords, fast-changing topics, or explicit search
    has_strong_positive_signals = strong_count > 0 or fast_count > 0 or has_explicit
    static_count, static_matches = _count_keyword_matches(query, STATIC_TOPICS)
    if static_count > 0 and not has_strong_positive_signals:
        confidence -= min(static_count * 0.3, 0.6)
        reasons.append(f"{static_count} static topic(s) (-)")
    elif static_count > 0 and has_strong_positive_signals:
        # Reduced penalty when strong positive signals present
        confidence -= min(static_count * 0.1, 0.2)
        reasons.append(f"{static_count} static topic(s) (reduced penalty due to strong signals)")

    # Clamp confidence
    confidence = max(0.0, min(1.0, confidence))

    # Determine search depth based on confidence and query characteristics
    if confidence >= 0.8 or has_explicit:
        depth = WebSearchDepth.STANDARD
    elif confidence >= 0.6 and fast_count >= 2:
        depth = WebSearchDepth.STANDARD
    elif confidence >= 0.5:
        depth = WebSearchDepth.QUICK
    else:
        depth = WebSearchDepth.QUICK

    # Build reason string
    reason = "; ".join(reasons) if reasons else "No strong indicators"

    should_search = confidence >= SEARCH_CONFIDENCE_THRESHOLD

    logger.debug(
        f"[WebSearchTrigger] Query: '{query[:50]}...' | "
        f"confidence={confidence:.2f} | should_search={should_search} | "
        f"depth={depth.value} | reason={reason}"
    )

    return WebSearchDecision(
        should_search=should_search,
        depth=depth,
        confidence=confidence,
        reason=reason,
        matched_keywords=all_matched_keywords,
        matched_patterns=all_matched_patterns
    )


async def should_search_with_llm(
    query: str,
    model_manager=None,
    heuristic_first: bool = True
) -> WebSearchDecision:
    """
    Determine if query needs web search, optionally using LLM for ambiguous cases.

    Args:
        query: User query text
        model_manager: Optional ModelManager for LLM classification
        heuristic_first: If True, only use LLM if heuristics are uncertain

    Returns:
        WebSearchDecision with final recommendation
    """
    # Always run heuristics first
    heuristic_result = should_search_heuristic(query)

    # If heuristics are confident (high or low), use that result
    if heuristic_first:
        if heuristic_result.confidence >= 0.7 or heuristic_result.confidence <= 0.2:
            return heuristic_result

    # For uncertain cases, try LLM if available
    if model_manager is not None:
        try:
            llm_result = await _classify_with_llm(query, model_manager)
            if llm_result is not None:
                # Combine heuristic and LLM signals
                combined_confidence = (heuristic_result.confidence + llm_result) / 2
                should_search = combined_confidence >= SEARCH_CONFIDENCE_THRESHOLD

                # Adjust depth based on combined confidence
                if combined_confidence >= 0.7:
                    depth = WebSearchDepth.STANDARD
                else:
                    depth = WebSearchDepth.QUICK

                return WebSearchDecision(
                    should_search=should_search,
                    depth=depth,
                    confidence=combined_confidence,
                    reason=f"{heuristic_result.reason}; LLM confidence={llm_result:.2f}",
                    matched_keywords=heuristic_result.matched_keywords,
                    matched_patterns=heuristic_result.matched_patterns
                )
        except asyncio.TimeoutError:
            logger.debug("[WebSearchTrigger] LLM classification timed out")
        except Exception as e:
            logger.debug(f"[WebSearchTrigger] LLM classification failed: {e}")

    return heuristic_result


async def _classify_with_llm(query: str, model_manager) -> Optional[float]:
    """
    Use LLM to classify if query needs web search.

    Args:
        query: User query
        model_manager: ModelManager instance

    Returns:
        Confidence score (0.0-1.0) or None if classification failed
    """
    if not model_manager:
        return None

    prompt = f"""Classify if this query requires CURRENT/LIVE web information or can be answered with general knowledge.

Query: "{query[:500]}"

Consider:
- Does it ask about recent events, news, or current data?
- Does it need real-time information (prices, weather, scores)?
- Does it reference specific recent dates or "latest/current" information?
- Or can it be answered with general/historical knowledge?

Respond with ONLY a number from 0-10:
- 0-3: General knowledge is sufficient
- 4-6: Might benefit from web search
- 7-10: Definitely needs current web information

Number:"""

    try:
        response = await asyncio.wait_for(
            model_manager.generate_once(
                prompt,
                max_tokens=SEARCH_TRIGGER_MAX_TOKENS
            ),
            timeout=SEARCH_TRIGGER_TIMEOUT
        )

        if not response:
            return None

        # Parse numeric response
        response = response.strip()
        match = re.search(r'\b(\d+)\b', response)
        if match:
            score = int(match.group(1))
            return min(max(score / 10.0, 0.0), 1.0)

        return None

    except Exception as e:
        logger.debug(f"[WebSearchTrigger] LLM parse error: {e}")
        return None


def analyze_for_web_search(query: str) -> WebSearchDecision:
    """
    Synchronous convenience function for web search trigger analysis.

    Usage:
        decision = analyze_for_web_search("What's the latest news on AI?")
        if decision.should_search:
            # Trigger web search with decision.depth
            pass
    """
    return should_search_heuristic(query)


# ===== Integration Helpers =====

def get_search_decision_for_prompt(
    query: str,
    crisis_level: Optional[str] = None,
    web_search_enabled: bool = True
) -> WebSearchDecision:
    """
    Get search decision with additional context filtering.

    Args:
        query: User query
        crisis_level: Current tone/crisis level
        web_search_enabled: Whether web search is enabled in config

    Returns:
        WebSearchDecision (may have should_search=False due to suppression)
    """
    # Quick exits
    if not web_search_enabled:
        return WebSearchDecision(
            should_search=False,
            depth=WebSearchDepth.QUICK,
            confidence=0.0,
            reason="Web search disabled in config",
            matched_keywords=[],
            matched_patterns=[]
        )

    # Crisis suppression
    if crisis_level and crisis_level.upper() in ("HIGH", "MEDIUM"):
        return WebSearchDecision(
            should_search=False,
            depth=WebSearchDepth.QUICK,
            confidence=0.0,
            reason=f"Suppressed during {crisis_level} crisis level",
            matched_keywords=[],
            matched_patterns=[]
        )

    return should_search_heuristic(query)


if __name__ == "__main__":
    # Quick test
    import logging
    logging.basicConfig(level=logging.DEBUG)

    test_queries = [
        "What's the latest news on AI?",
        "current bitcoin price",
        "weather forecast for tomorrow",
        "who was the first president of the united states",
        "how do I feel today",
        "search for python tutorials",
        "what's happening with the stock market 2024",
        "explain quantum computing",
        "latest iPhone release date",
        "how to make pasta",
    ]

    for q in test_queries:
        result = analyze_for_web_search(q)
        print(f"\nQuery: {q}")
        print(f"  Should search: {result.should_search}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Depth: {result.depth.value}")
        print(f"  Reason: {result.reason}")
