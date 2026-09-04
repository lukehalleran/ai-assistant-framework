# /utils/web_search_trigger.py
"""
WebSearchTrigger - Detects when a query should trigger web search, memory search, or knowledge search.

Module Contract:
- Purpose: Classify queries to determine if they need real-time web information, stored memory recall, or encyclopedic/wiki knowledge
- Inputs:
  - Query text
  - Optional: model_manager (for LLM classification)
  - Optional: crisis_level, remaining_credits, web_search_enabled
  - Optional: conversation_context — compact digest of prior turns so elliptical
    follow-ups ("check the news") resolve to topic-specific search_terms instead
    of generic ones; part of the LLM-trigger cache key [NEW 2026-06-21]
  - User location (resolved internally via utils/location_resolver.py) — injected
    into the trigger prompt so location-dependent queries (weather, local news,
    "near me") carry the user's place in search_terms instead of leaking literal
    "my area" to the search engine [NEW 2026-07-02]. Localization is limited to
    physical-surroundings queries: the prompt forbids attaching the location to
    institution/account/login queries, and parsed search_terms pass through
    location_resolver.strip_unjustified_location() as a deterministic backstop
    (2026-07-08: a school-login query localized to "Springfield IL" retrieved
    a college the user never attended, asserted as "your school")
- Outputs:
  - WebSearchDecision with:
    - should_search: bool
    - depth: WebSearchDepth (QUICK/STANDARD/DEEP)
    - confidence: float (0.0-1.0)
    - search_terms: List[str] (LLM-optimized queries for better results)
    - num_searches: int (parallel search count, 1-4)
    - source: str ("llm" | "heuristic" | "explicit")
    - needs_memory_search: bool (LLM detected memory/recall intent) [NEW 2026-03-15]
    - needs_knowledge_search: bool (LLM detected encyclopedic/wiki intent) [NEW 2026-03-31]
    - needs_document_generation + document_topic/document_type/document_source —
      document_source is "research" (research the topic externally) or
      "conversation" (write up THIS conversation's content; sharing cues like
      "so I can text that to my therapist" mean conversation) [NEW 2026-08-24];
      normalized to ""|"research"|"conversation" in LLMSearchTriggerResponse.parse,
      consumed by the agentic gate's Tier-4 doc_gen_intent and gui/handlers'
      _resolve_doc_source (which also has a deterministic regex backstop)
- Adaptive anchors [NEW 2026-08-03]:
  - _get_search_anchors() merges learned exemplars (utils.adaptive_exemplars,
    domain "web_search") into the semantic anchor sets: "search_worthy" → positive,
    "no_search" → negative; anchor-embedding cache is keyed on the store version,
    and per-text vectors are reused via adaptive_exemplars.encode_texts_cached
    (2026-08-21) so a version bump re-encodes only newly learned texts.
  - Teachers are OUTCOME-based (never this module's own semantic signal): a response
    that actually cited [WEB_ markers teaches search_worthy (hook in
    handlers._write_turn_telemetry; elevated-tone turns never teach), and a
    tone-corroborated agentic-gate veto teaches no_search (gate.apply_intent_veto).
- Side effects:
  - LLM API call when using analyze_for_web_search_llm() (async)
  - None for heuristic-only path
- Routing: At most one of should_search, needs_memory_search, needs_knowledge_search is true (web vs memory vs knowledge are separate paths)

Detection Strategy (LLM-First with Heuristic Fallback):
1. Crisis suppression: Block search during HIGH/MEDIUM crisis levels
2. Quick pre-filter: Skip LLM for obvious non-search queries (conversational, emotional)
3. LLM classification: Single call returns should_search + needs_memory_search + needs_knowledge_search + optimized search_terms + depth
4. Confidence blend: 70% LLM + 30% heuristic for final decision
5. Heuristic fallback: On LLM timeout/error, use keyword-based detection

Legacy Heuristic Detection (fallback):
1. Keyword matching for recency indicators ("latest", "current", "2024", etc.)
2. Pattern matching for news/event queries
3. Entity detection for fast-changing topics (stocks, weather, sports scores)
4. Explicit search requests ("search for", "look up online")
5. Suppression patterns for personal/visual queries ("do you see", "can you see", etc.)
"""

import asyncio
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from utils.logging_utils import get_logger
from utils.query_checker import is_personal_doc_search
from utils.trigger_match import is_negated as _trigger_is_negated
import json

logger = get_logger("web_search_trigger")

# Short-lived cache for LLM trigger results (prevents duplicate calls within same request)
# Structure: {query_hash: (timestamp, WebSearchDecision)}
_llm_trigger_cache: Dict[int, Tuple[float, "WebSearchDecision"]] = {}
_LLM_TRIGGER_CACHE_TTL = 10.0  # 10 seconds - long enough for same request, short enough to not stale


class WebSearchDepth(Enum):
    """Search depth levels - mirrored from web_search_manager for decoupling."""
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


@dataclass
class WebSearchDecision:
    """
    Result of web search trigger analysis.

    Attributes:
        should_search: Whether to trigger web search
        depth: Recommended search depth (QUICK/STANDARD/DEEP)
        confidence: Decision confidence score (0.0-1.0)
        reason: Human-readable explanation for decision
        matched_keywords: Keywords that matched (for heuristic debugging)
        matched_patterns: Patterns that matched (for heuristic debugging)
        search_terms: LLM-optimized search queries (empty if heuristic-only)
        num_searches: Number of parallel searches to execute (1-4)
        source: Decision source ("llm" | "heuristic" | "explicit")
    """
    should_search: bool
    depth: WebSearchDepth
    confidence: float  # 0.0-1.0
    reason: str
    matched_keywords: List[str]
    matched_patterns: List[str]
    # LLM-first fields
    search_terms: List[str] = None  # LLM-optimized search queries
    num_searches: int = 1  # Parallel search count (1-4)
    source: str = "heuristic"  # "llm" | "heuristic" | "explicit"
    needs_memory_search: bool = False  # LLM detected memory/recall intent
    needs_knowledge_search: bool = False  # LLM detected encyclopedic/wiki knowledge intent
    needs_document_generation: bool = False  # LLM detected document generation intent
    needs_pattern_analysis: bool = False  # LLM detected cross-history pattern deliberation
    document_topic: str = ""  # Topic for document generation
    document_type: str = ""  # "report" or "summary"
    document_source: str = ""  # "research" (external lookup) | "conversation" (summarize THIS conversation) | ""

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.search_terms is None:
            self.search_terms = []


@dataclass
class LLMSearchTriggerResponse:
    """
    Parsed response from unified LLM trigger classification.

    This dataclass represents the structured output from the LLM when
    it analyzes a query for web search needs. The LLM returns JSON that
    gets parsed into this structure.

    Attributes:
        should_search: Whether the query needs web search
        confidence: LLM's confidence in the decision (0.0-1.0)
        reason: Brief explanation of the decision
        search_terms: Optimized search queries generated by LLM
        search_depth: Recommended depth ("quick" | "standard" | "deep")
        num_searches: How many parallel searches to execute (1-4)
    """
    should_search: bool
    confidence: float
    reason: str
    search_terms: List[str]
    search_depth: str  # "quick" | "standard" | "deep"
    num_searches: int
    needs_memory_search: bool = False  # Whether query wants stored memories/facts/notes
    needs_knowledge_search: bool = False  # Whether query wants encyclopedic/wiki knowledge
    needs_document_generation: bool = False  # Whether query wants a saved document
    needs_pattern_analysis: bool = False  # Whether query asks for longitudinal/pattern evaluation
    document_topic: str = ""  # Topic for document generation
    document_type: str = ""  # "report" or "summary"
    document_source: str = ""  # "research" | "conversation" | ""

    @classmethod
    def parse(cls, json_str: str) -> Optional['LLMSearchTriggerResponse']:
        """
        Parse LLM JSON response with error handling.

        Handles common issues:
        - Markdown code blocks (```json...```)
        - Missing fields (uses safe defaults)
        - Invalid JSON (returns None for fallback to heuristics)

        Args:
            json_str: Raw JSON string from LLM response

        Returns:
            LLMSearchTriggerResponse if parsing succeeds, None otherwise
        """

        if not json_str:
            return None

        # Strip markdown code blocks if present
        text = json_str.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)

            # Validate and clamp values
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))

            num_searches = int(data.get("num_searches", 1))
            num_searches = max(1, min(4, num_searches))

            search_depth = str(data.get("search_depth", "quick")).lower()
            if search_depth not in ("quick", "standard", "deep"):
                search_depth = "quick"

            document_source = str(data.get("document_source", "")).strip().lower()
            if document_source not in ("research", "conversation"):
                document_source = ""

            return cls(
                should_search=bool(data.get("should_search", False)),
                confidence=confidence,
                reason=str(data.get("reason", "")),
                search_terms=list(data.get("search_terms", [])),
                search_depth=search_depth,
                num_searches=num_searches,
                needs_memory_search=bool(data.get("needs_memory_search", False)),
                needs_knowledge_search=bool(data.get("needs_knowledge_search", False)),
                needs_document_generation=bool(data.get("needs_document_generation", False)),
                needs_pattern_analysis=bool(data.get("needs_pattern_analysis", False)),
                document_topic=str(data.get("document_topic", "")),
                document_type=str(data.get("document_type", "")),
                document_source=document_source,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"[LLMSearchTriggerResponse] JSON parse error: {e}")
            return None


# ===== Configuration =====

# Minimum confidence to trigger search
SEARCH_CONFIDENCE_THRESHOLD = float(os.getenv("WEB_SEARCH_CONFIDENCE_THRESHOLD", "0.5"))

# LLM classification settings
SEARCH_TRIGGER_MODEL = os.getenv("WEB_SEARCH_TRIGGER_MODEL", "gpt-4o-mini")
SEARCH_TRIGGER_TIMEOUT = float(os.getenv("WEB_SEARCH_TRIGGER_TIMEOUT", "5.0"))  # Increased from 2.0
SEARCH_TRIGGER_MAX_TOKENS = int(os.getenv("WEB_SEARCH_TRIGGER_MAX_TOKENS", "150"))

# LLM-first mode settings
LLM_FIRST_ENABLED = os.getenv("WEB_SEARCH_LLM_FIRST_ENABLED", "true").lower() == "true"
LLM_HEURISTIC_BLEND_WEIGHT = float(os.getenv("WEB_SEARCH_LLM_WEIGHT", "0.7"))  # 70% LLM, 30% heuristic


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
    "press release", "breaking",
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
    "version", "beta", "alpha", "release", "features", "changelog",

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

    # Visual/perception queries (personal recall, not web)
    "do you see", "can you see",
    "do you recognize", "what do you see",

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
    """Count keyword matches (word-boundary, not substring) and return them.

    Word-boundary matching prevents short tickers/keywords from firing inside
    unrelated words — e.g. "eth" (Ethereum) matching "som**eth**ing", or "live"
    matching "de**live**ry". That substring bug once scored a casual life-update
    ("...now walking to get some ice cream or something") as a web-search query
    (eth+today = 0.70). Multi-word keywords ("stock price", "this week") and
    hyphenated ones ("real-time") still match as whole units.
    """
    text_lower = _normalize(text)
    matched = []
    for kw in keywords:
        if re.search(rf'\b{re.escape(kw)}\b', text_lower):
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


def _matches_phrase_non_negated(text: str, phrases: Tuple[str, ...]) -> Tuple[bool, List[str]]:
    """Same as _matches_phrase, but drops a phrase hit whose immediate
    preceding context (5-token lookback) negates the request — "no need to
    search for this" / "don't look that up" must not count as an explicit
    search phrase (2026-09-04). Used only for EXPLICIT_SEARCH_PHRASES, whose
    hit is a strong positive signal (+0.6 confidence) — SUPPRESSION_PATTERNS
    stays plain substring (negating a suppression would need to mean
    "allow search", a different and out-of-scope direction)."""
    text_lower = _normalize(text)
    matched = []
    for phrase in phrases:
        idx = text_lower.find(phrase)
        if idx == -1 or _trigger_is_negated(text_lower, idx):
            continue
        matched.append(phrase)
    return len(matched) > 0, matched


# ========================================================================
# Semantic similarity layer for search trigger
# ========================================================================

# Anchor phrases for "needs web search" — embedded once, compared to query
_SEARCH_ANCHOR_PHRASES = [
    "latest news current events headlines today",
    "stock market price bitcoin crypto trading today",
    "weather forecast temperature tomorrow this week",
    "election results polls voting latest count",
    "sports scores game results standings playoffs",
    "product release date launch availability price",
    "breaking news just announced reported confirmed",
    "what happened situation update conflict war",
]

# Anchor phrases for "does NOT need web search" — personal/static/creative
_NO_SEARCH_ANCHOR_PHRASES = [
    "how are you feeling tell me about yourself",
    "explain concept definition theory formula meaning",
    "help me write create generate compose advice",
    "remember when we talked about you mentioned earlier",
    "my pet family friend relationship feelings emotions",
]

_search_anchor_embs = None
_no_search_anchor_embs = None
_search_anchor_version = None
_anchor_text_emb_cache: dict = {}


def _get_search_anchors():
    """Lazily compute and cache search anchor embeddings.

    Anchors = code seeds merged with per-user LEARNED phrases from
    utils.adaptive_exemplars (domain "web_search", 2026-08-02): queries whose
    responses actually cited [WEB_N] results teach "search_worthy"; queries
    the tone-corroborated gate veto stood down teach "no_search" (an
    emotional vent that once fired a search stops semantically boosting
    future vents). Cache is keyed on the adaptive store version.
    """
    global _search_anchor_embs, _no_search_anchor_embs, _search_anchor_version
    try:
        from utils.adaptive_exemplars import get_store
        _version = get_store().version
    except Exception:
        _version = -1
    if _search_anchor_embs is not None and _search_anchor_version == _version:
        return _search_anchor_embs, _no_search_anchor_embs
    try:
        from models.model_manager import ModelManager
        embedder = ModelManager._get_cached_embedder()
        if embedder is None:
            return None, None
        pos = list(_SEARCH_ANCHOR_PHRASES)
        neg = list(_NO_SEARCH_ANCHOR_PHRASES)
        try:
            from utils.adaptive_exemplars import get_store
            pos += get_store().get_learned("web_search", "search_worthy")
            neg += get_store().get_learned("web_search", "no_search")
        except Exception:
            pass
        from utils.adaptive_exemplars import encode_texts_cached
        _search_anchor_embs = encode_texts_cached(
            embedder, pos, _anchor_text_emb_cache, normalize=True
        )
        _no_search_anchor_embs = encode_texts_cached(
            embedder, neg, _anchor_text_emb_cache, normalize=True
        )
        _search_anchor_version = _version
        return _search_anchor_embs, _no_search_anchor_embs
    except Exception:
        return None, None


def _semantic_search_boost(query: str, threshold: float = 0.35) -> float:
    """
    Compute semantic similarity boost for search trigger.

    Compares query embedding to search vs no-search anchor phrases.
    Returns a boost (0.0 to 0.3) if query is semantically close to
    search-worthy topics and far from no-search topics.
    """
    search_embs, no_search_embs = _get_search_anchors()
    if search_embs is None:
        return 0.0
    try:
        from models.model_manager import ModelManager
        embedder = ModelManager._get_cached_embedder()
        if embedder is None:
            return 0.0
        import numpy as np
        q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        # Max similarity to any search anchor
        search_sims = search_embs @ q_emb
        max_search_sim = float(np.max(search_sims))
        # Max similarity to any no-search anchor
        no_search_sims = no_search_embs @ q_emb
        max_no_search_sim = float(np.max(no_search_sims))
        # Only boost if closer to search anchors than no-search anchors
        margin = max_search_sim - max_no_search_sim
        if max_search_sim > threshold and margin > 0.05:
            # Scale boost based on both absolute similarity and margin
            # Higher margin = more confident this is search-worthy
            boost = min(0.3, margin * 0.8 + (max_search_sim - threshold) * 0.5)
            return round(boost, 2)
    except Exception:
        pass
    return 0.0


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

    # Personal-document search (2026-08-29): "search for documents related to
    # the MGT class I am currently enrolled in" is an INTERNAL retrieval
    # request — the search verb made the old heuristic call it an "explicit
    # search request" (conf 0.80) and burn Tavily credits on the user's own
    # corpus (one sub-query was literally "Add dates and deadlines to Google
    # Calendar"). The agentic gate routes these to file/memory tools instead.
    if is_personal_doc_search(query):
        return WebSearchDecision(
            should_search=False,
            depth=WebSearchDepth.QUICK,
            confidence=0.0,
            reason="Suppressed: personal-document search (internal retrieval target)",
            matched_keywords=[],
            matched_patterns=["personal_doc_search"],
        )

    # Check explicit search phrases (strongest signal)
    has_explicit, explicit_matches = _matches_phrase_non_negated(query, EXPLICIT_SEARCH_PHRASES)
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

    # Semantic similarity boost: catch queries near keyword sets but without exact matches
    # Only runs if keyword confidence is ambiguous (0.1-0.45) — clear hits/misses skip this
    if 0.1 <= confidence < SEARCH_CONFIDENCE_THRESHOLD:
        sem_boost = _semantic_search_boost(query)
        if sem_boost > 0:
            confidence += sem_boost
            reasons.append(f"semantic boost +{sem_boost:.2f}")

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


# ===== LLM-First Trigger System =====


_PATTERN_CANDIDATE_FIRST_PERSON_RE = re.compile(
    r"\b(?:i|me|my|mine|we|us|our)\b", re.I,
)
# Audit F7 (2026-08-31): the original list accepted pronoun + ANY comparator
# ("before", "data", "changed", "theory"…) — a wide slice of ordinary turns
# paid an extra LLM round-trip and vents with "before" in them reached the
# classifier. Now an explicit RECORD/LONGITUDINAL operation is required; bare
# comparators fall through to the ordinary Tier-4 path, which can still
# return needs_pattern_analysis for genuine mixed requests.
_PATTERN_CANDIDATE_OPERATION_RE = re.compile(
    # record/longitudinal operations + two-series RELATIONAL vocabulary
    # ("could X and Y be connected?" is a genuine mixed shape); bare
    # comparators (before/after/data/changed/theory) stay out.
    r"\b(?:patterns?|trends?|over\s+time|"
    r"across\s+(?:time|(?:my\s+|the\s+)?(?:months|weeks|years|history))|"
    r"co-?occur\w*|correlat\w*|covar\w*|timeline|"
    r"linked|connected|relationship|associated\s+with|"
    r"my\s+(?:recorded\s+)?(?:history|records?|notes|data|corpus)|"
    r"how\s+often|how\s+many\s+(?:times|days|nights|weeks)|"
    r"before\s+and\s+after|compare\s+my)\b",
    re.I,
)


def _looks_like_pattern_candidate(query: str) -> bool:
    """Cheap recall-oriented signal that only decides whether to ask the LLM.

    It never routes the turn itself, and it must not drag personal-state
    statements (vents) into an LLM round-trip — the 08-05 guard's doctrine.
    """
    text = query or ""
    if not (_PATTERN_CANDIDATE_FIRST_PERSON_RE.search(text)
            and _PATTERN_CANDIDATE_OPERATION_RE.search(text)):
        return False
    return not is_personal_state_statement(text)


def quick_prefilter_should_skip(query: str) -> bool:
    """
    Quick pre-filter to skip LLM for obvious non-search queries.

    This runs in <1ms and catches queries that definitely don't need
    web search, avoiding unnecessary LLM calls.

    Args:
        query: User query text

    Returns:
        True if we should SKIP the LLM and not search, False to continue
    """
    if not query or len(query.strip()) < 5:
        return True  # Too short to be a meaningful search

    query_lower = query.lower().strip()

    # Check suppression patterns (personal/emotional queries)
    for pattern in SUPPRESSION_PATTERNS:
        if pattern in query_lower:
            return True

    # Very short conversational queries
    if len(query_lower) < 20 and any(
        query_lower.startswith(p) for p in
        ("hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "sure", "yes", "no", "yeah", "nah")
    ):
        return True

    # Long pastes (song lyrics, articles, etc.) without explicit search phrases
    # are almost never search requests — skip LLM to save time and avoid Tavily 400s
    if len(query_lower) > 500 and not any(p in query_lower for p in EXPLICIT_SEARCH_PHRASES):
        logger.debug(f"[WebSearchTrigger] Prefilter: skipping long paste ({len(query)} chars)")
        return True

    # Deictic follow-up references - these refer to conversation context, not web content
    # Patterns like "i just watched it", "saw that", "read it", etc.
    deictic_followup_patterns = (
        "watched it", "saw it", "read it", "heard it", "checked it",
        "finished it", "started it", "looked at it",
        "watched that", "saw that", "read that", "heard that",
        "just watched", "just saw", "just read", "just heard",
        "i watched", "i saw", "i read",
    )
    if any(pattern in query_lower for pattern in deictic_followup_patterns):
        logger.debug(f"[WebSearchTrigger] Prefilter: skipping deictic follow-up: {query[:50]}")
        return True

    return False


# Referential/deictic tokens: a query built around these leans on the PRIOR turn
# for its meaning ("they're only giving us 7 days" → 7 days of WHAT?). Such a
# query scores 0 on the standalone heuristic (no topic of its own) but can still
# be search-worthy once resolved against conversation context — so we let the LLM
# (which sees that context) decide instead of short-circuiting to no-search.
_REFERENTIAL_TOKEN_RE = re.compile(
    r"\b(?:it|they|them|their|that|this|those|these|us|we)\b"
)


_FIRST_PERSON_OPENER_RE = re.compile(r"^(?:i|i'm|im|i've|ive|i'd|id|my)\b", re.I)
_INTERROGATIVE_OPENER_RE = re.compile(
    r"^(?:what|what's|how|why|when|where|who|which|can|could|should|would|"
    r"is|are|do|does|did|will|any\b)", re.I)
_LOOKUP_CUE_RE = re.compile(
    r"\b(?:look\s+up|search|google|find\s+out|latest|news|current|what'?s\s+the)\b", re.I)


def is_personal_state_statement(query: str) -> bool:
    """True for a first-person state statement with no info-seeking shape.

    2026-08-05: "I was taking about 900 mg a day... I don't want to start it
    again unless a doctor says I should" ran a 3.5s web search — the bare
    pronoun ("start IT again") made query_depends_on_context() treat it as a
    referential follow-up, which bypasses the decisive conf=0.0 skip and
    consults the LLM trigger; the LLM said search. In a first-person state
    statement the pronoun refers to the user's OWN situation, not an
    unresolved search target — these never consult the LLM. Third-party
    elliptical follow-ups ("they're only giving us 7 days") are unaffected.
    """
    q = (query or "").strip()
    if not q or "?" in q:
        return False
    if _INTERROGATIVE_OPENER_RE.match(q):
        return False
    if _LOOKUP_CUE_RE.search(q):
        return False
    if not _FIRST_PERSON_OPENER_RE.match(q):
        return False
    # An epistemic-stance opener ("I think…", "I mean…", "I don't believe…")
    # frames a claim about the OUTSIDE WORLD, not the user's own state — on
    # 2026-08-18 political commentary like "I think people would be arrested
    # if they were aware…" passed this check and was TAUGHT as a no_search
    # exemplar (poisoning class). Strip the markers (shared doctrine with the
    # gate's vent-shape test; lazy import — both modules import each other
    # only inside functions) and require remaining substantive first person.
    try:
        # lazy import: cycle (gate imports web_search_trigger at call time)
        from core.agentic.gate import strip_epistemic_markers, _FIRST_PERSON_RE
        return bool(_FIRST_PERSON_RE.search(strip_epistemic_markers(q)))
    except Exception:
        return True


def query_depends_on_context(query: str) -> bool:
    """True when the query's searchability hinges on the prior turn (referential).

    Detects short follow-ups whose subject is a pronoun/deictic pointing at
    something established earlier ("it", "they", "that", "us") rather than a
    standalone searchable noun. Used to decide whether a conf=0.0 query is worth
    consulting the LLM about WHEN conversation context is available.

    Word-boundary regex, not padded-substring shapes: the old hand-enumerated
    tuple missed 'that?', 'them?', 'it:' and any token at end-of-query, so the
    exact follow-ups this check exists for still short-circuited at conf=0.0.
    """
    if not query:
        return False
    return bool(_REFERENTIAL_TOKEN_RE.search(query.lower()))


def _build_llm_trigger_prompt(
    query: str,
    current_date: str,
    remaining_credits: float = 100,
    conversation_context: str = None,
    user_location: str = None,
    user_institution: str = None,
) -> str:
    """
    Build the classification prompt for the unified LLM trigger.

    Args:
        query: User query text
        current_date: Current date string (e.g., "2026-01-06")
        remaining_credits: Available search credits (affects num_searches suggestion)
        conversation_context: Optional compact digest of the immediately prior
            turns. When present, lets the LLM resolve elliptical / deictic
            follow-ups ("check the news", "any updates on that") against the
            active topic instead of emitting generic terms.
        user_location: Optional user location string ("Springfield, IL").
            When present, location-dependent queries (weather, local news,
            "near me") get the location baked into search_terms instead of
            leaking literal "my area" into the search engine.
        user_institution: Optional name of the user's school ("Georgia
            Tech"). When present, the user's OWN academic-logistics queries
            (drop/withdrawal deadlines, registrar, tuition) get the school
            named in search_terms instead of generic "college"/"school" —
            generic academic queries return generic pages (2026-08-27: a
            drop-date confirmation searched "college drop date August 2026").

    Returns:
        Formatted prompt string for LLM
    """
    context_block = ""
    if conversation_context and conversation_context.strip():
        _ctx = conversation_context.strip()
        if len(_ctx) > 1200:
            _ctx = _ctx[:1200] + " [...truncated]"
        context_block = (
            "\nRECENT CONVERSATION (the turns immediately before this query — "
            "use ONLY to resolve follow-up/elliptical references in the query):\n"
            f"{_ctx}\n"
        )
    location_line = ""
    location_guideline = ""
    if user_location and user_location.strip():
        location_line = f"User location: {user_location.strip()}\n"
        location_guideline = (
            f"\n- LOCAL QUERIES: For weather, temperature, forecasts, air quality, local news, "
            f"local events, or nearby places, include \"{user_location.strip()}\" explicitly in every "
            f"relevant search term. NEVER emit \"my area\", \"near me\", \"local\", or \"nearby\" as "
            f"literal search text — replace them with the user's location."
            f"\n- LOCATION IS ONLY FOR PHYSICAL SURROUNDINGS: never add the user's location to "
            f"queries about their accounts, logins, school/college/university, employer, bank, "
            f"healthcare, or any website/service they use. Their institutions are NOT determined "
            f"by where they are — a college in their city is not their college unless they named "
            f"it. If the institution is unnamed, search the error/issue generically without any place."
        )
    institution_line = ""
    institution_guideline = ""
    if user_institution and user_institution.strip():
        _inst = user_institution.strip()
        institution_line = f"User's school: {_inst}\n"
        institution_guideline = (
            f"\n- SCHOOL-LOGISTICS QUERIES: the user attends \"{_inst}\". For queries about "
            f"THEIR OWN school logistics (drop/withdrawal deadlines, registration, registrar, "
            f"tuition, academic calendar, transcripts), use \"{_inst}\" in search terms instead "
            f"of generic \"college\"/\"school\"/\"my school\". NEVER apply it when the user names "
            f"a DIFFERENT school, and never use it for general coursework/concept questions. "
            f"NEVER search for THEIR course's assignment/homework/quiz details or due dates — "
            f"the web does not know their course section; those facts live in their own "
            f"syllabus/Canvas/uploaded documents, which the assistant can already retrieve."
        )
    return f"""Analyze if this query needs real-time web search OR stored memory search, and generate optimized search terms.

Query: "{query[:500]}"
Current date: {current_date}
{location_line}{institution_line}Available search budget: {remaining_credits:.0f} credits
{context_block}
WEB SEARCH CRITERIA:
- SEARCH if: current events, recent news, live data (stocks, weather, sports), time-sensitive health info the user is explicitly asking about (recalls, outbreaks, new guidance), or references dates/years needing verification
- DON'T SEARCH if: historical facts, scientific concepts, how-to guides, personal/emotional topics, or can be answered with general knowledge
- NEVER SEARCH for:
  * Casual acknowledgments (nice, thanks, cool, got it, okay)
  * Statements of opinion, feeling, or reaction ("X seems terrible", "I wish X", "that's the worst") — even when the topic touches health, drugs, news, or tech. The user is conversing, not requesting information; searching interrupts the conversation
  * Statements of the user's OWN plan to look something up themselves ("I'll check Reddit", "I'm gonna google it later", "I will see what is said on X") — they are declining information, not requesting it
  * Meta-comments about the conversation or system
  * Greetings or short responses under 5 words that aren't explicit search requests
  * Follow-up references to prior conversation ("watched it", "saw that", "just read it", "i just watched", etc.) - these refer to something already discussed, not web content
  * Comments about media the user consumed ("i just watched", "i finished reading", "just saw") - these are conversation follow-ups, not search requests
  * Questions about the user's OWN schedule or state anchored to a weekday/date ("is this useful info for Monday", "will I be ready by Friday") — "Monday" is THEIR appointment or deadline from the conversation, not a news topic. There is nothing on the web about the user's Monday. Deictic questions ("is this expected/normal?") are about conversation content, not the internet
  * Questions about the user's OWN private arrangements in ANY life domain — their course's assignments/quizzes/due dates ("is the first assignment a quiz or a deliverable", "when is HW 1 due"), their job's meetings/shifts/deadlines ("when is our standup"), their club or hobby group's sessions/practice times, their own appointments. The web does not know the user's course section, workplace, or group; those facts live in their own documents and conversation history, which the assistant can already retrieve. Consider needs_memory_search instead
  * "Are you saying..." / "Do you mean..." / "So you're telling me..." questions about the ASSISTANT'S own previous statement — they resolve from the conversation, not the web
- should_search=true requires an actual information need: the reply would be materially wrong or stale without fresh external facts. When in doubt, false.

MEMORY SEARCH CRITERIA (needs_memory_search):
- TRUE if: the user wants to recall past conversations, stored facts about themselves or people they know, personal notes, uploaded documents, or anything from their history with this assistant
- Examples: "tell me more about Whiskers", "what was that thing we discussed?", "remind me about my dog", "what do my notes say about X", "what have I told you about my job"
- FALSE if: general knowledge question, web search query, casual chat, or creative request

KNOWLEDGE SEARCH CRITERIA (needs_knowledge_search):
- TRUE if: the user asks an in-depth factual, scientific, historical, or encyclopedic QUESTION that would benefit from consulting reference material (Wikipedia, textbooks, documentation)
- Examples: "explain nuclear fission vs fusion", "how does photosynthesis work", "what is the history of the Roman Empire", "compare TCP and UDP", "consult Wikipedia about X", "tell me about quantum entanglement in depth"
- Also TRUE if: user explicitly asks to consult Wikipedia, look something up, or requests a detailed/in-depth explanation of a concept
- FALSE if: casual chat, simple yes/no questions, personal/emotional topics, creative writing, or questions answerable in one sentence without references
- NEVER for COLLABORATIVE / PERSONAL-TASK work — helping with the user's OWN materials: writing, editing, formatting, or restructuring their resume, cover letter, email, essay, code, or documents. "I need my resume in ATS-friendly format", "make the intro punchier", "also give me a healthcare version" are task instructions to ACT ON, not encyclopedic questions. Naming a concept (ATS, STAR method, SQL) while directing work on the user's material is NOT a knowledge request.
- NEVER for FOLLOW-UP CONTINUATIONS of an ongoing task ("yeah and…", "also…", "I'll also need…", "plus…") — these extend the current collaborative work; answer them directly.
- The trigger must be an actual QUESTION or an explicit "explain / what is / how does / tell me about" request. A task statement ("I need X in Y format", "make it Z") is never knowledge search.

PATTERN DELIBERATION CRITERIA (needs_pattern_analysis):
- TRUE when the user asks the system to examine their history/notes/data across
  time or periods, identify or test a recurring pattern, compare before versus
  after an event, evaluate whether two personal variables covary, or assemble
  longitudinal evidence for a decision.
- This includes natural phrasings that do not literally say "pattern": "look
  across everything I've told you", "has this changed since I moved?", "compare
  how I was before and after", "what tends to happen when...", "check whether
  my theory fits the record", and "use my history plus outside research".
- TRUE even when the same request also needs web, Wikipedia, PubMed, computation,
  or another tool. The deliberation coordinator will select and audit those
  channels; do not reduce a mixed request to ordinary web search.
- FALSE for a single current symptom, ordinary recall of one fact/event, casual
  self-reflection without a request to examine the record, or general factual
  questions about patterns in populations.

DOCUMENT GENERATION CRITERIA (needs_document_generation):
- TRUE if: the user wants a document SAVED to disk — a report, summary, or research document written and stored as a file
- Examples: "write a report about climate change", "create a document about AI", "save a summary on quantum computing", "prepare a report on economic trends", "make me a research document about X", "generate a report and save it", "draft a report about Y", "I need a written report on Z"
- Also TRUE if: user references the document writing feature, asks to "try the document feature", or says something like "write that up as a report"
- FALSE if: user just wants information verbally, asks a question, wants a summary in chat, or says "summarize X" without asking to save/write/create a document
- When TRUE, also set document_topic to the core topic and document_type to "report" or "summary"
- When TRUE, also set document_source: "conversation" if the user wants THIS conversation's content written up (e.g. "write up what we just discussed", "summarize these insights so I can send them to my therapist", "put our conversation in a doc") — the source is what was already said, not external research. Use "research" when the user wants a topic researched and written up from external sources ("write a report about climate change"). Sharing cues ("so I can text/send/show that to X") about the current discussion mean "conversation".

OUTPUT (JSON only, no markdown):
{{
  "should_search": true or false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation",
  "search_terms": ["optimized query 1", "optimized query 2"],
  "search_depth": "quick" or "standard" or "deep",
  "num_searches": 1-4,
  "needs_memory_search": true or false,
  "needs_knowledge_search": true or false,
  "needs_pattern_analysis": true or false,
  "needs_document_generation": true or false,
  "document_topic": "topic for document (only if needs_document_generation is true)",
  "document_type": "report or summary (only if needs_document_generation is true)",
  "document_source": "research or conversation (only if needs_document_generation is true)"
}}

GUIDELINES:
- search_terms: Rewrite for better results (add year if relevant, be specific, remove conversational filler){location_guideline}{institution_guideline}
- FOLLOW-UPS: If the query is elliptical or refers back to the conversation ("check the news", "any updates on that", "what's the latest", "look into it", "anything new on it"), resolve the referent using RECENT CONVERSATION above and make search_terms SPECIFIC to that established topic. Never emit generic or world-news terms when the user is clearly asking for an update on something already being discussed. If no RECENT CONVERSATION is provided, treat the query as self-contained.
- num_searches: Use 2-4 only for comparison queries or multi-faceted topics
- search_depth: "quick" for simple facts, "standard" for news/analysis, "deep" for research
- If not searching, return empty search_terms and num_searches: 0
- At most one of should_search, needs_memory_search, needs_knowledge_search, needs_pattern_analysis, needs_document_generation should be true. A mixed personal pattern request owns the turn via needs_pattern_analysis and records its required evidence channels inside the later frozen plan.

JSON:"""


# ---------------------------------------------------------------------------
# Private-sphere generic term guard (2026-09-01): asked whether his first
# assignment "is just a quiz on lectures or is there deliverable", the trigger
# LLM said search at conf 0.8 with terms like "Georgia Tech assignment due
# September 13 2026" — institution + generic category nouns + dates. The
# public web does not know the user's PRIVATE ARRANGEMENTS — their course
# section's assignments, their job's meetings/shifts, their club's practice
# schedule, their own appointments; those facts live in their own documents
# and conversation history (which retrieval already had — 6 Tavily credits
# returned GT football schedules). Sibling of the gate's temporal-generic
# guard: if EVERY proposed term reduces to a known personal-context anchor
# (school/employer name) + private-sphere-generic nouns + time tokens, the
# search cannot succeed. Any other content word rescues: a course code
# ("MGT"), a topic, or public-institutional vocabulary ("drop", "tuition" —
# registrar facts ARE on the web).
#
# GENERALITY DOCTRINE: this is ONE categorized vocabulary spanning life
# domains (school / work / appointments / hobby-groups), not a per-domain
# guard — extend by adding nouns to the right category, never by cloning the
# function (the _REQUEST_FRAMING_STOPWORDS lesson). Only include nouns that
# are overwhelmingly private-sphere when combined with nothing but an
# institution name and dates; polysemous public-event words ("game", "exam",
# "calendar", "club", "campaign") stay OUT so real public queries ("Georgia
# Tech game September", "academic calendar") are always rescued. Under-fires
# by design.
# ---------------------------------------------------------------------------

_PRIVATE_GUARD_TIME_STOP_TOKENS = frozenset({
    # time/date
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "today", "tomorrow", "yesterday", "week", "weekend", "month",
    "semester", "fall", "spring", "summer", "winter",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    # stopwords / filler
    "for", "on", "in", "at", "the", "a", "an", "of", "and", "or", "to",
    "about", "this", "that", "is", "are", "what", "when", "time", "times",
    "my", "our", "their", "there", "details", "detail", "information",
    "info",
})

# Private-sphere category nouns — things everyone HAS whose specifics only
# the user's own context knows. Categorized for maintainability; one set at
# runtime.
_PRIVATE_SPHERE_GENERIC_TOKENS = frozenset({
    # school (course-section facts; registrar-public words excluded)
    "assignment", "assignments", "homework", "homeworks", "hw", "hws",
    "quiz", "quizzes", "deliverable", "deliverables", "course", "courses",
    "class", "classes", "lecture", "lectures", "syllabus", "module",
    "modules", "grade", "grades",
    # work (employer-internal logistics; bare "work"/"job" = THEIR job)
    "work", "job", "meeting", "meetings", "standup", "standups", "shift",
    "shifts", "agenda", "agendas", "sprint", "sprints", "timesheet",
    "timesheets", "paycheck", "payday", "onboarding",
    # appointments / personal scheduling
    "appointment", "appointments", "session", "sessions", "schedule",
    "schedules", "checkup", "followup",
    # hobby / group activities
    "practice", "practices", "rehearsal", "rehearsals", "workout",
    "workouts",
    # shared scheduling vocabulary
    "due", "date", "dates", "deadline", "deadlines", "submission",
    "submissions", "submit", "first", "second", "third", "next", "upcoming",
})


def terms_are_private_sphere_generic(terms, anchors=None) -> bool:
    """True when EVERY proposed search term is a personal-context anchor
    (the user's school/employer/group name) + private-sphere-generic nouns +
    time tokens — a question about the user's own arrangements that the web
    cannot answer. Each term must actually contain a private-sphere noun
    (pure-temporal junk is the gate guard's job), and any other content
    word rescues the term.

    anchors: iterable of the user's named institutions (school today;
    employer/group names when resolvers for them exist). Terms naming an
    UNKNOWN organization are rescued by its name — under-fires by design."""
    if not terms:
        return False
    anchor_tokens = set()
    for anchor in (anchors or []):
        anchor_tokens |= {t for t in re.findall(r"[a-z]+",
                                                str(anchor or "").lower())}
    allowed = (_PRIVATE_GUARD_TIME_STOP_TOKENS
               | _PRIVATE_SPHERE_GENERIC_TOKENS | anchor_tokens)
    for term in terms:
        tokens = re.findall(r"[a-z]+", str(term).lower())
        if not any(t in _PRIVATE_SPHERE_GENERIC_TOKENS for t in tokens):
            return False
        if any(t not in allowed for t in tokens):
            return False
    return True


async def _classify_with_llm_unified(
    query: str,
    model_manager,
    remaining_credits: float = 100,
    timeout: float = None,
    conversation_context: str = None,
) -> Optional[LLMSearchTriggerResponse]:
    """
    Unified LLM call for search trigger classification.

    This is the core LLM-first classification that returns structured
    output including search terms, depth, and number of searches.

    Args:
        query: User query text
        model_manager: ModelManager instance for LLM calls
        remaining_credits: Available credits (affects num_searches)
        timeout: Override default timeout
        conversation_context: Optional digest of prior turns, used to resolve
            elliptical follow-up queries into topic-specific search terms.

    Returns:
        LLMSearchTriggerResponse if successful, None if failed/timeout
    """
    if not model_manager:
        return None

    current_date = datetime.now().strftime("%Y-%m-%d")
    user_location = None
    try:
        from utils.location_resolver import get_user_location
        user_location = get_user_location()
    except Exception as e:
        logger.debug(f"[WebSearchTrigger] Location resolution failed: {e}")
    user_institution = None
    try:
        from utils.institution_resolver import get_user_institution
        user_institution = get_user_institution()
    except Exception as e:
        logger.debug(f"[WebSearchTrigger] Institution resolution failed: {e}")
    prompt = _build_llm_trigger_prompt(
        query, current_date, remaining_credits, conversation_context,
        user_location=user_location,
        user_institution=user_institution,
    )
    effective_timeout = timeout or SEARCH_TRIGGER_TIMEOUT

    logger.debug(
        f"[WebSearchTrigger] Calling LLM: model={SEARCH_TRIGGER_MODEL}, "
        f"timeout={effective_timeout}s, max_tokens={SEARCH_TRIGGER_MAX_TOKENS}"
    )

    try:
        response = await asyncio.wait_for(
            model_manager.generate_once(
                prompt,
                model_name=SEARCH_TRIGGER_MODEL,
                max_tokens=SEARCH_TRIGGER_MAX_TOKENS,
                temperature=0.0  # Deterministic for consistent decisions
            ),
            timeout=effective_timeout
        )

        if not response:
            logger.debug("[WebSearchTrigger] LLM returned empty response")
            return None

        logger.debug(f"[WebSearchTrigger] LLM raw response: {response[:200]}...")
        parsed = LLMSearchTriggerResponse.parse(response)
        if parsed:
            if parsed.search_terms and user_location:
                # Backstop: the prompt forbids localizing institution/account
                # queries, but the LLM sometimes does it anyway (2026-07-08:
                # "college login" + injected city -> wrong college asserted as
                # the user's school). Strip location the query never justified.
                from utils.location_resolver import strip_unjustified_location
                parsed.search_terms = strip_unjustified_location(
                    parsed.search_terms, query, user_location
                )
            if parsed.search_terms and user_institution:
                # Backstop: name the user's school in academic-logistics
                # terms the LLM left generic ("college drop date August
                # 2026" retrieved nothing usable, 2026-08-27). Runs after
                # the location strip so a de-localized term can still gain
                # the institution.
                from utils.institution_resolver import apply_institution
                parsed.search_terms = apply_institution(
                    parsed.search_terms, query, user_institution
                )
            if parsed.should_search and parsed.search_terms:
                try:
                    from utils.institution_resolver import get_user_anchors
                    anchors = get_user_anchors()
                except Exception:
                    anchors = [user_institution] if user_institution else []
                if terms_are_private_sphere_generic(parsed.search_terms, anchors):
                    # Runs AFTER apply_institution so it vets the final term
                    # forms. Suppression only — never teaches no_search.
                    logger.info(
                        "[WebSearchTrigger] Search suppressed — every term is "
                        "anchor+private-sphere-generic; the user's own "
                        f"arrangements are not on the web: {parsed.search_terms}"
                    )
                    parsed.should_search = False
                    parsed.search_terms = []
            logger.debug(
                f"[WebSearchTrigger] LLM parsed: should_search={parsed.should_search}, "
                f"conf={parsed.confidence}, terms={parsed.search_terms}"
            )
        return parsed

    except asyncio.TimeoutError:
        logger.warning(f"[WebSearchTrigger] LLM classification timed out after {effective_timeout}s")
        return None
    except Exception as e:
        logger.warning(f"[WebSearchTrigger] LLM classification error: {e}")
        return None


_SEARCH_INABILITY_MARKERS = (
    "can't search", "cannot search", "couldn't search", "unable to search",
    "no search results came through", "search right now",
    "can't pull anything live", "no lookup tools",
)
_RETRY_CUES = (
    "try again", "retry", "one more time", "try it now", "fixed it",
    "should be able", "should work now", "try that again",
)


def _detect_retry_after_inability(query: str, conversation_context) -> str:
    """Return search terms when the user is retrying a search the assistant
    said it couldn't perform; None otherwise. Terms = text after the last
    colon when present ("Okay: Tactical Taylors"), else the query with retry
    preamble words stripped."""
    try:
        q = (query or "").strip()
        ctx = str(conversation_context or "").lower()
        if not q or not ctx:
            return None
        if not any(m in ctx for m in _SEARCH_INABILITY_MARKERS):
            return None
        ql = q.lower()
        if not any(c in ql for c in _RETRY_CUES):
            return None
        if ":" in q:
            term = q.rsplit(":", 1)[1].strip()
            if len(term.split()) >= 1 and term:
                return term
        # strip retry preamble words; keep the remainder if substantive
        import re as _re
        stripped = _re.sub(
            r"\b(?:ok(?:ay)?|let'?s|try|again|retry|i|fixed|it|now|you|should|"
            r"be|able|to|so|well|one|more|time|that)\b",
            " ", ql,
        )
        stripped = " ".join(stripped.split()).strip(" .,!:")
        return stripped if len(stripped) >= 3 else None
    except Exception:
        return None


async def analyze_for_web_search_llm(
    query: str,
    model_manager=None,
    crisis_level: Optional[str] = None,
    web_search_enabled: bool = True,
    remaining_credits: float = 100,
    timeout: float = None,
    conversation_context: str = None,
) -> WebSearchDecision:
    """
    LLM-first web search trigger analysis with heuristic fallback.

    This is the primary entry point for determining if a query needs web search.
    Uses LLM classification first, with 70/30 confidence blend with heuristics.
    Falls back to pure heuristics on LLM timeout or error.

    Flow:
    1. Crisis level check (HIGH/MEDIUM suppresses search)
    2. Quick pre-filter (skip obvious non-search queries)
    3. LLM classification (returns should_search + search_terms + depth)
    4. Blend: 70% LLM + 30% heuristic confidence
    5. Fallback: Pure heuristics on LLM failure

    Args:
        query: User query text
        model_manager: ModelManager instance for LLM calls (optional)
        crisis_level: Current tone/crisis level from ToneDetector
        web_search_enabled: Whether web search is enabled in config
        remaining_credits: Available credits for search (affects num_searches)
        timeout: Override default LLM timeout
        conversation_context: Optional compact digest of the immediately prior
            turns. Threaded into the trigger prompt so elliptical follow-ups
            ("check the news", "any updates") resolve to topic-specific terms
            instead of generic ones. Part of the cache key.

    Returns:
        WebSearchDecision with should_search, search_terms, depth, num_searches
    """
    global _llm_trigger_cache
    pattern_candidate = _looks_like_pattern_candidate(query)

    # Check cache first (prevents duplicate LLM calls within same request).
    # Hash on (query, conversation_context): the SAME elliptical query ("check the
    # news") must resolve to different search terms under different prior topics, so
    # context is part of the cache identity. crisis_level is intentionally excluded.
    cache_key = hash((query, conversation_context))
    now = time.time()
    if cache_key in _llm_trigger_cache:
        cached_time, cached_result = _llm_trigger_cache[cache_key]
        if now - cached_time < _LLM_TRIGGER_CACHE_TTL:
            logger.debug(f"[WebSearchTrigger] Cache hit for query (age={now - cached_time:.2f}s)")
            return cached_result

    # Clean expired cache entries periodically (every ~100 calls)
    if len(_llm_trigger_cache) > 50:
        _llm_trigger_cache = {
            k: v for k, v in _llm_trigger_cache.items()
            if now - v[0] < _LLM_TRIGGER_CACHE_TTL
        }

    # Quick exits
    if not web_search_enabled:
        return WebSearchDecision(
            should_search=False,
            depth=WebSearchDepth.QUICK,
            confidence=0.0,
            reason="Web search disabled in config",
            matched_keywords=[],
            matched_patterns=[],
            source="heuristic"
        )

    # Crisis suppression
    if crisis_level and crisis_level.upper() in ("HIGH", "MEDIUM"):
        return WebSearchDecision(
            should_search=False,
            depth=WebSearchDepth.QUICK,
            confidence=0.0,
            reason=f"Suppressed during {crisis_level} crisis level",
            matched_keywords=[],
            matched_patterns=[],
            source="heuristic"
        )

    # Quick pre-filter: skip LLM for obvious non-search queries
    if quick_prefilter_should_skip(query) and not pattern_candidate:
        return WebSearchDecision(
            should_search=False,
            depth=WebSearchDepth.QUICK,
            confidence=0.0,
            reason="Pre-filter: query unlikely to need search",
            matched_keywords=[],
            matched_patterns=[],
            source="heuristic"
        )

    # Personal-document search (2026-08-29, deterministic — LLM-independent):
    # the target of the search is the user's own corpus (notes/files/uploads/
    # syllabus + personal anchor); never a web request, never consult the LLM.
    # The agentic gate routes the same shape to file/memory tools.
    if is_personal_doc_search(query):
        return WebSearchDecision(
            should_search=False,
            depth=WebSearchDepth.QUICK,
            confidence=0.0,
            reason="Personal-document search: internal retrieval target",
            matched_keywords=[],
            matched_patterns=["personal_doc_search"],
            source="heuristic",
        )

    # Always compute heuristic result (used for blending and fallback)
    heuristic_result = should_search_heuristic(query)

    # Retry-after-inability (2026-08-22, deterministic — LLM-independent):
    # when the ASSISTANT's recent turn disclosed it couldn't search ("I can't
    # search right now", "no search results came through") and the user comes
    # back with a retry cue ("Let's try again I fixed it. Okay: Tactical
    # Taylors"), the search contract is explicit in the conversation — but
    # only the LLM channel could see it, and on a busy cold turn that call
    # TIMED OUT (5s) and the heuristic fallback said "no indicators", so the
    # explicit retry silently didn't search. The model's own inability
    # disclosure + a retry cue IS the deterministic signal.
    _retry_search = _detect_retry_after_inability(query, conversation_context)
    if _retry_search is not None:
        logger.info(
            f"[WebSearchTrigger] Retry-after-inability — deterministic search: "
            f"terms={_retry_search!r}"
        )
        return WebSearchDecision(
            should_search=True,
            depth=WebSearchDepth.QUICK,
            confidence=0.85,
            reason="user retried after assistant disclosed search inability",
            matched_keywords=[],
            matched_patterns=["retry_after_inability"],
            search_terms=[_retry_search],
            num_searches=1,
            source="explicit",
        )

    # If LLM-first mode is disabled or no model manager, use heuristics only
    if not LLM_FIRST_ENABLED or model_manager is None:
        return heuristic_result

    # Skip expensive LLM call when heuristic is already decisive.
    # confidence=0.0 with no keywords means "nothing search-related at all" —
    # the LLM won't find anything the heuristic missed in these cases.
    # confidence >= 0.7 means strong heuristic yes (explicit search phrases) —
    # the LLM would just confirm.
    if heuristic_result.confidence <= 0.0 and not heuristic_result.matched_keywords:
        # A conf=0.0/no-keyword query is normally "nothing search-related." But an
        # elliptical follow-up ("they're only giving us 7 days") is search-worthy
        # once resolved against the prior turn — the heuristic can't see that, the
        # LLM (with conversation_context) can. Only skip the short-circuit for a
        # referential query when we actually have context to resolve it against.
        _referential_followup = (
            bool(conversation_context and conversation_context.strip())
            and query_depends_on_context(query)
        )
        if _referential_followup and is_personal_state_statement(query):
            # First-person state statement: the pronoun points at the user's
            # own situation, not a search target — never consult the LLM.
            # Teach no_search through this deterministic channel (independent
            # of the semantic anchors, same pattern as the tone-veto teacher)
            # so the phrasing stops semantically boosting future statements.
            _referential_followup = False
            try:
                from utils.adaptive_exemplars import get_store
                get_store().record("web_search", "no_search", query, "personal_state_statement")
            except Exception:
                pass
        if not _referential_followup and not pattern_candidate:
            logger.debug("[WebSearchTrigger] Skipping LLM: heuristic confident no-search (conf=0.0, no keywords)")
            _llm_trigger_cache[cache_key] = (now, heuristic_result)
            return heuristic_result
        logger.debug("[WebSearchTrigger] conf=0.0 referential follow-up with context — consulting LLM")
    if (heuristic_result.confidence >= 0.7 and heuristic_result.should_search
            and not pattern_candidate):
        logger.debug(f"[WebSearchTrigger] Skipping LLM: heuristic confident search (conf={heuristic_result.confidence:.2f})")
        _llm_trigger_cache[cache_key] = (now, heuristic_result)
        return heuristic_result

    # Try LLM classification
    llm_response = await _classify_with_llm_unified(
        query,
        model_manager,
        remaining_credits,
        timeout,
        conversation_context=conversation_context,
    )

    # If LLM failed, fall back to heuristics
    if llm_response is None:
        logger.debug("[WebSearchTrigger] LLM failed, falling back to heuristics")
        return heuristic_result

    # A mixed longitudinal request owns the turn. Do not let the ordinary web
    # confidence blend or a personal-topic search veto erase this independent
    # routing decision; the frozen planner will audit every requested channel.
    if llm_response.needs_pattern_analysis:
        result = WebSearchDecision(
            should_search=False,
            depth=WebSearchDepth.QUICK,
            confidence=llm_response.confidence,
            reason=f"LLM: {llm_response.reason}",
            matched_keywords=heuristic_result.matched_keywords,
            matched_patterns=heuristic_result.matched_patterns,
            search_terms=[],
            num_searches=0,
            source="llm",
            needs_pattern_analysis=True,
        )
        _llm_trigger_cache[cache_key] = (time.time(), result)
        return result

    # Heuristic veto: only override LLM when heuristic has an ACTIVE reason
    # to suppress (matched a suppression/static pattern). A confidence of 0.0
    # with "No strong indicators" means the heuristic has no opinion — not a
    # confident "no" — so it should not block the LLM.
    has_active_suppression = (
        heuristic_result.matched_patterns  # matched a suppression pattern
        or "static topic" in heuristic_result.reason  # matched static topics
    )
    if has_active_suppression and not heuristic_result.should_search:
        logger.debug(
            f"[WebSearchTrigger] Heuristic veto: active suppression detected "
            f"(reason={heuristic_result.reason}), LLM wanted search={llm_response.should_search} but heuristic says no"
        )
        veto_result = WebSearchDecision(
            should_search=False,
            depth=WebSearchDepth.QUICK,
            confidence=heuristic_result.confidence,
            reason=f"Heuristic veto ({heuristic_result.reason}); LLM overridden",
            matched_keywords=heuristic_result.matched_keywords,
            matched_patterns=heuristic_result.matched_patterns,
            search_terms=[],
            num_searches=0,
            source="heuristic"
        )
        _llm_trigger_cache[cache_key] = (time.time(), veto_result)
        return veto_result

    # Convert LLM confidence to directional search score.
    # confidence=0.90 + should_search=False means "90% sure we DON'T need search" → search_score=0.10
    # confidence=0.90 + should_search=True means "90% sure we DO need search" → search_score=0.90
    llm_search_score = llm_response.confidence if llm_response.should_search else (1.0 - llm_response.confidence)

    # Blend directional LLM score with heuristic confidence (70% LLM, 30% heuristic)
    llm_weight = LLM_HEURISTIC_BLEND_WEIGHT
    heuristic_weight = 1.0 - llm_weight
    blended_confidence = (
        llm_weight * llm_search_score +
        heuristic_weight * heuristic_result.confidence
    )

    # Determine final should_search based on blended confidence
    should_search = blended_confidence >= SEARCH_CONFIDENCE_THRESHOLD

    # High-confidence LLM override: if LLM is very confident (>=0.8) in either
    # direction, trust it directly — the heuristic having no opinion (0.0)
    # should not block a definitive LLM decision
    if llm_response.confidence >= 0.8:
        should_search = llm_response.should_search

    # Map LLM depth string to enum
    depth_map = {
        "quick": WebSearchDepth.QUICK,
        "standard": WebSearchDepth.STANDARD,
        "deep": WebSearchDepth.DEEP
    }
    depth = depth_map.get(llm_response.search_depth, WebSearchDepth.QUICK)

    # Build combined reason
    reason = (
        f"LLM: {llm_response.reason} (conf={llm_response.confidence:.2f}); "
        f"Heuristic: {heuristic_result.reason} (conf={heuristic_result.confidence:.2f}); "
        f"Blended={blended_confidence:.2f}"
    )

    logger.debug(
        f"[WebSearchTrigger] LLM-first result: should_search={should_search}, "
        f"terms={llm_response.search_terms}, depth={depth.value}, "
        f"blended_conf={blended_confidence:.2f}"
    )

    result = WebSearchDecision(
        should_search=should_search,
        depth=depth,
        confidence=blended_confidence,
        reason=reason,
        matched_keywords=heuristic_result.matched_keywords,
        matched_patterns=heuristic_result.matched_patterns,
        search_terms=llm_response.search_terms if should_search else [],
        num_searches=llm_response.num_searches if should_search else 0,
        source="llm",
        needs_memory_search=llm_response.needs_memory_search,
        needs_knowledge_search=llm_response.needs_knowledge_search,
        needs_pattern_analysis=llm_response.needs_pattern_analysis,
        needs_document_generation=llm_response.needs_document_generation,
        document_topic=llm_response.document_topic,
        document_type=llm_response.document_type,
        document_source=llm_response.document_source,
    )

    # Cache the result to avoid duplicate LLM calls within same request
    _llm_trigger_cache[cache_key] = (time.time(), result)

    return result


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
