
"""
utils/query_checker.py
Utilities for quick query analysis and gating hints.

This module began as a small deictic checker; it now provides a few
lightweight heuristics that help the orchestrator and gate system make
fast decisions without model calls.

Now includes heavy topic classification for inline fact extraction.
Heavy topics include: political violence, human rights crises, mental health
emergencies (depression, anxiety, suicidal ideation), emotional distress
(grief, trauma, relationship crises), and personal safety concerns.
"""

import os
import asyncio
from dataclasses import dataclass
from typing import List, Optional, Set
from utils.logging_utils import get_logger

logger = get_logger("query_checker")


DEICTIC_HINTS: tuple[str, ...] = (
    "explain", "that", "it", "this", "again", "another way",
    "different way", "more", "elaborate", "clarify", "what about",
    "those", "these", "there", "former", "latter"
)

QUESTION_LEADS: tuple[str, ...] = (
    "what", "who", "when", "where", "why", "how", "which"
)

COMMAND_SIGNS: tuple[str, ...] = (
    "/", "please ", "do ", "tell me to ", "create ", "generate ", "write ", "summarize ",
)

META_CONVERSATIONAL_MARKERS: tuple[str, ...] = (
    "do you recall", "do you remember", "did you", "did we",
    "we discussed", "we talked about", "last time", "the other day",
    "you said", "you mentioned", "you told me",
    "earlier you", "before you", "didn't we", "haven't we",
    "recall the", "remember when", "remember the",
    # Queries about the conversation/responses themselves
    "that response", "that off topic response", "your response", "your answer",
    "that message", "earlier message", "previous response", "last response",
    "what causes that", "why did you", "geared to", "seems like it was"
)

# Temporal markers for detecting time-based memory queries
TEMPORAL_MARKERS = {
    # Yesterday/recent
    "yesterday": 1,
    "last night": 1,
    "this morning": 1,
    "earlier today": 1,

    # Days ago
    "days ago": 3,
    "few days ago": 3,
    "couple days ago": 2,
    "other day": 2,

    # Last week
    "last week": 7,
    "week ago": 7,
    "few weeks ago": 14,
    "couple weeks ago": 14,

    # Longer periods
    "last month": 30,
    "month ago": 30,
    "while back": 14,
    "while ago": 14,
    "long time ago": 30,

    # Specific days
    "monday": 7,
    "tuesday": 7,
    "wednesday": 7,
    "thursday": 7,
    "friday": 7,
    "saturday": 7,
    "sunday": 7,
}


def _normalize(q: str) -> str:
    return (q or "").strip().lower()


def is_deictic(query: str) -> bool:
    """True if the query likely refers to earlier context (anaphora)."""
    if not query:
        return False
    ql = _normalize(query)

    # Short follow-ups with hints are often deictic
    if len(ql.split()) <= 6 and any(h in ql for h in DEICTIC_HINTS):
        return True

    # Pronouns/markers at beginning suggest reference
    if ql.startswith(("that", "this", "it", "they", "those", "these", "so", "and", "then")):
        return True

    return False


def is_deictic_followup(q: str) -> bool:
    """Softer check for follow-up phrasing used by wiki gating."""
    ql = _normalize(q)
    return any(h in ql for h in DEICTIC_HINTS)


def is_question(q: str) -> bool:
    ql = _normalize(q)
    return ql.endswith("?") or ql.startswith(QUESTION_LEADS)


def is_command(q: str) -> bool:
    ql = _normalize(q)
    return ql.startswith(COMMAND_SIGNS)


def is_meta_conversational(q: str) -> bool:
    """True if the query is asking about the conversation history itself."""
    if not q:
        return False
    ql = _normalize(q)
    return any(marker in ql for marker in META_CONVERSATIONAL_MARKERS)


def extract_temporal_window(q: str) -> int:
    """
    Extract the temporal window (in days) from a query based on time markers.

    This analyzes queries for temporal references like "yesterday", "last week",
    "few days ago" and returns an appropriate retrieval window in days.

    Args:
        q: Query text

    Returns:
        Number of days to look back. Returns 0 if no temporal markers found.
        Examples:
            "yesterday" -> 1
            "few days ago" -> 3
            "last week" -> 7
            "last month" -> 30
            "no temporal marker" -> 0
    """
    if not q:
        return 0

    ql = _normalize(q)

    # Check for temporal markers and find the largest window
    max_days = 0
    for marker, days in TEMPORAL_MARKERS.items():
        if marker in ql:
            max_days = max(max_days, days)

    # Also check for explicit date references (e.g., "Nov 1st", "November 1")
    import re

    # Pattern: Month name/abbreviation + day number (with optional suffixes like "st", "nd", "rd", "th")
    date_pattern = r'\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)\s*\d{1,2}(?:st|nd|rd|th)?\b'
    if re.search(date_pattern, ql):
        # If explicit date mentioned, assume up to 30 days back
        max_days = max(max_days, 30)

    # Pattern: "N days ago" where N is a number
    num_days_pattern = r'(\d+)\s+days?\s+ago'
    match = re.search(num_days_pattern, ql)
    if match:
        try:
            num_days = int(match.group(1))
            max_days = max(max_days, num_days)
        except ValueError:
            pass

    return max_days


def keyword_tokens(q: str, min_len: int = 3) -> List[str]:
    ql = _normalize(q)
    return [t for t in ql.split() if len(t) >= min_len]


@dataclass
class QueryAnalysis:
    text: str
    tokens: List[str]
    is_question: bool
    is_command: bool
    is_deictic: bool
    is_followup: bool
    token_count: int
    char_count: int
    intents: Set[str]
    is_heavy_topic: bool = False  # Crisis/sensitive topics requiring inline fact extraction
    is_meta_conversational: bool = False  # Query asking about conversation history itself


def analyze_query(q: str, model_manager=None) -> QueryAnalysis:
    """
    Analyze query for various properties including heavy topic classification.

    Args:
        q: Query text
        model_manager: Optional model manager for LLM-based heavy topic classification.
                      If not provided, only heuristic classification is used.

    Returns:
        QueryAnalysis with all query properties
    """
    tokens = keyword_tokens(q)
    intents: Set[str] = set()
    q_is_question = is_question(q)
    q_is_command = is_command(q)
    q_is_deictic = is_deictic(q)
    q_is_follow = is_deictic_followup(q)
    q_is_meta = is_meta_conversational(q)

    if q_is_question:
        intents.add("question")
    if q_is_command:
        intents.add("command")
    if q_is_meta:
        intents.add("meta_conversational")
    if not intents:
        intents.add("statement")

    # Heavy topic classification (synchronous - uses heuristics only by default)
    q_is_heavy = _is_heavy_topic_heuristic(q)

    return QueryAnalysis(
        text=q or "",
        tokens=tokens,
        is_question=q_is_question,
        is_command=q_is_command,
        is_deictic=q_is_deictic,
        is_followup=q_is_follow,
        token_count=len(tokens),
        char_count=len(q or ""),
        intents=intents,
        is_heavy_topic=q_is_heavy,
        is_meta_conversational=q_is_meta,
    )


async def analyze_query_async(q: str, model_manager=None) -> QueryAnalysis:
    """
    Async version of analyze_query that can use LLM for heavy topic classification.

    Args:
        q: Query text
        model_manager: Optional model manager for LLM classification

    Returns:
        QueryAnalysis with all query properties including LLM-based heavy topic result
    """
    # Run synchronous analysis first
    analysis = analyze_query(q, model_manager=None)

    # If heuristic already says it's heavy, skip LLM
    if analysis.is_heavy_topic:
        return analysis

    # Try LLM classification if available
    if model_manager is not None:
        try:
            is_heavy = await _classify_heavy_topic_llm(q, model_manager)
            # Update the analysis
            return QueryAnalysis(
                text=analysis.text,
                tokens=analysis.tokens,
                is_question=analysis.is_question,
                is_command=analysis.is_command,
                is_deictic=analysis.is_deictic,
                is_followup=analysis.is_followup,
                token_count=analysis.token_count,
                char_count=analysis.char_count,
                intents=analysis.intents,
                is_heavy_topic=is_heavy,
                is_meta_conversational=analysis.is_meta_conversational,
            )
        except Exception as e:
            logger.debug(f"[QueryChecker] LLM heavy topic classification failed: {e}")

    return analysis

# ===== Heavy Topic Classification =====

# Configuration
HEAVY_TOPIC_CHAR_THRESHOLD = int(os.getenv("HEAVY_TOPIC_CHAR_THRESHOLD", "2500"))
HEAVY_TOPIC_MODEL = os.getenv("HEAVY_TOPIC_MODEL", "gpt-4o-mini")
HEAVY_TOPIC_TIMEOUT = float(os.getenv("HEAVY_TOPIC_TIMEOUT", "2.0"))
HEAVY_TOPIC_MAX_TOKENS = int(os.getenv("HEAVY_TOPIC_MAX_TOKENS", "10"))

# Heavy topic keywords (crisis, violence, human rights, emotional/mental health)
HEAVY_KEYWORDS = {
    # Political violence & enforcement
    "raid", "raids", "ice", "deportation", "deported", "arrested", "arrest", "arrests",
    "military", "police", "protest", "protests", "riot", "riots", "violence", "violent",
    "shooting", "shot", "tear gas", "pepper spray", "detention", "detained", "detain",
    "federal agents", "national guard", "troops", "soldiers",
    "undocumented", "illegal", "immigration", "deport",

    # Crisis & trauma
    "crisis", "emergency", "disaster", "tragedy", "trauma", "traumatic",
    "killed", "dead", "death", "deaths", "casualties", "wounded", "injured",

    # Human rights & persecution
    "persecution", "discriminat", "racism", "racist", "hate crime",
    "ethnic cleansing", "genocide", "war crime", "torture", "abuse",
    "refugee", "refugees", "asylum", "sanctuary",

    # Conflict & war
    "war", "warfare", "combat", "attack", "attacks", "bomb", "bombing",
    "terrorist", "terrorism", "insurgent", "militant",

    # Authoritarianism
    "authoritarian", "dictatorship", "oppression", "crackdown",
    "martial law", "curfew", "lockdown",

    # Mental health & emotional distress
    "depressed", "depression", "anxiety", "anxious", "panic", "panic attack",
    "ptsd", "mental health", "mental illness", "bipolar", "schizophrenia",
    "therapy", "therapist", "psychiatrist", "psychologist", "counseling",
    "medication", "antidepressant", "psychiatric", "psych ward",
    "suicidal", "suicide", "kill myself", "end my life", "self-harm", "self harm",
    "cutting", "overdose", "pills",

    # Emotional crisis states
    "breakdown", "nervous breakdown", "meltdown", "losing it",
    "can't take it", "can't cope", "overwhelmed", "hopeless", "helpless",
    "despair", "devastated", "heartbroken", "broken", "shattered",
    "lonely", "isolated", "alone", "abandoned", "worthless", "hate myself",
    "scared", "terrified", "afraid", "frightened", "fear",

    # Relationship distress & life crises
    "breakup", "broke up", "divorce", "divorcing", "separated", "separation",
    "cheating", "cheated", "affair", "betrayed", "betrayal",
    "miscarriage", "stillborn", "pregnancy loss", "lost the baby",
    "funeral", "mourning", "grieving", "grief", "loss",
    "fired", "laid off", "lost my job", "unemployment",
    "evicted", "eviction", "homeless", "foreclosure",

    # Anger & violence (personal)
    "angry", "furious", "rage", "enraged", "hate", "hatred",
    "want to hurt", "want to kill", "violence", "fight", "assault",
    "domestic violence", "abusive", "abuser",
}


def _is_heavy_topic_heuristic(q: str) -> bool:
    """
    Fast heuristic check for heavy topics.

    Strategy:
      1. Length check (>2500 chars = likely article/news)
      2. Keyword matching against crisis/violence/rights/emotional/mental health terms

    Returns:
        True if heuristics suggest heavy topic (political, emotional, or mental health crisis)
    """
    if not q or not isinstance(q, str):
        return False
    
    # Length check
    if len(q) > HEAVY_TOPIC_CHAR_THRESHOLD:
        return True
    
    # Keyword matching (case-insensitive)
    q_lower = q.lower()
    
    # Count keyword hits
    hits = sum(1 for keyword in HEAVY_KEYWORDS if keyword in q_lower)
    
    # If multiple heavy keywords appear, likely a heavy topic
    if hits >= 2:
        return True
    
    # Single keyword but with contextual markers (numbers, locations, quotes)
    if hits == 1:
        # Check for news article markers
        import re
        has_numbers = bool(re.search(r"\b\d{1,3}[,\s]?\d{0,3}\b", q))
        has_quotes = '"' in q or '"' in q or '"' in q
        has_locations = bool(re.search(
            r"\b(Chicago|Illinois|Texas|California|New York|Washington|D\.?C\.?)\b",
            q, re.IGNORECASE
        ))
        
        if (has_numbers and has_quotes) or (has_numbers and has_locations):
            return True
    
    return False


async def _classify_heavy_topic_llm(q: str, model_manager) -> bool:
    """
    Use LLM to classify topic as heavy/normal.
    
    Args:
        q: Query text
        model_manager: ModelManager instance
    
    Returns:
        True if heavy, False if normal
    
    Raises:
        asyncio.TimeoutError if classification times out
        Exception if LLM call fails
    """
    if not model_manager or not hasattr(model_manager, "generate_once"):
        return False
    
    # Build prompt
    prompt = _build_heavy_topic_prompt(q)
    
    # Preserve current model
    prev_model = None
    try:
        if hasattr(model_manager, "get_active_model_name"):
            prev_model = model_manager.get_active_model_name()
        
        # Switch to classifier model if registered
        if hasattr(model_manager, "switch_model"):
            if hasattr(model_manager, "api_models") and HEAVY_TOPIC_MODEL in model_manager.api_models:
                model_manager.switch_model(HEAVY_TOPIC_MODEL)
    except Exception as e:
        logger.debug(f"[QueryChecker] Model switch failed: {e}")
    
    try:
        # Call LLM with timeout
        response = await asyncio.wait_for(
            model_manager.generate_once(
                prompt,
                max_tokens=HEAVY_TOPIC_MAX_TOKENS
            ),
            timeout=HEAVY_TOPIC_TIMEOUT
        )
        
        # Parse response
        result = _parse_heavy_topic_response(response)
        logger.debug(f"[QueryChecker] LLM heavy topic result: {result}")
        return result
        
    finally:
        # Restore previous model
        try:
            if prev_model and hasattr(model_manager, "switch_model"):
                model_manager.switch_model(prev_model)
        except Exception:
            pass


def _build_heavy_topic_prompt(q: str) -> str:
    """Build LLM prompt for heavy topic classification."""
    # Truncate very long input
    truncated = q[:1000]
    if len(q) > 1000:
        truncated += "..."

    return f"""Classify this message as HEAVY or NORMAL.

HEAVY topics: political violence, protests, raids, arrests, deportation, war, terrorism, human rights crises, personal safety threats, trauma, discrimination, persecution, mental health crises (depression, anxiety, suicidal thoughts, self-harm, PTSD), emotional distress (grief, heartbreak, breakup, divorce, job loss), relationship crises (abuse, betrayal, domestic violence), personal emergencies.

NORMAL topics: coding, general knowledge, casual conversation, entertainment, hobbies, academic topics, advice, mild emotions (slightly happy/sad), everyday stress, general questions.

MESSAGE:
{truncated}

Respond with ONLY one word: HEAVY or NORMAL"""


def _parse_heavy_topic_response(response: str) -> bool:
    """
    Parse LLM response for heavy topic classification.
    
    Args:
        response: Raw LLM output
    
    Returns:
        True if HEAVY, False otherwise
    """
    if not response:
        return False
    
    normalized = response.strip().upper()
    
    # Explicit markers
    if "HEAVY" in normalized:
        return True
    if "NORMAL" in normalized:
        return False
    
    # Fuzzy matches
    heavy_indicators = ["CRISIS", "SENSITIVE", "SEVERE", "YES", "TRUE"]
    if any(indicator in normalized for indicator in heavy_indicators):
        return True
    
    return False


# ===== Conversation Thread Detection =====

# Configuration
THREAD_TIME_HARD_CUTOFF = int(os.getenv("THREAD_TIME_HARD_CUTOFF", "7200"))  # 2 hours
THREAD_TIME_CLOSE_BONUS = int(os.getenv("THREAD_TIME_CLOSE_BONUS", "300"))   # 5 min
THREAD_TIME_MEDIUM_BONUS = int(os.getenv("THREAD_TIME_MEDIUM_BONUS", "1800"))  # 30 min
THREAD_CONTINUITY_THRESHOLD = float(os.getenv("THREAD_CONTINUITY_THRESHOLD", "0.5"))

# Thread scoring weights
THREAD_WEIGHT_KEYWORDS = float(os.getenv("THREAD_WEIGHT_KEYWORDS", "0.5"))
THREAD_WEIGHT_TIME = float(os.getenv("THREAD_WEIGHT_TIME", "0.25"))
THREAD_WEIGHT_HEAVY = float(os.getenv("THREAD_WEIGHT_HEAVY", "0.15"))
THREAD_WEIGHT_TOPIC = float(os.getenv("THREAD_WEIGHT_TOPIC", "0.1"))

# Thread-breaking phrases
THREAD_BREAK_MARKERS = {
    "changing topics", "different topic", "switching gears",
    "on another note", "anyway,", "by the way,", "moving on",
    "different subject", "new question", "unrelated"
}


def extract_thread_keywords(text: str) -> Set[str]:
    """
    Extract meaningful keywords for thread continuity detection.

    Args:
        text: Input text

    Returns:
        Set of lowercase keywords (min 3 chars, no stopwords)
    """
    import re

    # Extract alphanumeric words only (strips punctuation)
    text_lower = text.lower()
    words = re.findall(r'\b[a-z]+\b', text_lower)

    # Filter by minimum length
    tokens = [w for w in words if len(w) >= 3]

    # Common stopwords to filter
    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "about",
        "what", "when", "where", "which", "who", "why", "how",
        "can", "could", "would", "should", "will", "are", "were", "was",
        "have", "has", "had", "been", "being", "does", "did", "doing",
        "you", "your", "they", "their", "them", "his", "her", "him"
    }

    return set(t for t in tokens if t not in stopwords)


def has_thread_break_marker(query: str) -> bool:
    """
    Check if query contains explicit thread-breaking phrases.
    
    Args:
        query: User query text
    
    Returns:
        True if query signals a topic switch
    """
    query_lower = query.lower()
    return any(marker in query_lower for marker in THREAD_BREAK_MARKERS)


def calculate_thread_continuity_score(
    current_query: str,
    last_query: str,
    time_diff_seconds: float,
    both_heavy: bool = False,
    same_topic: bool = False,
    current_topic: Optional[str] = None,
    last_was_heavy: bool = False
) -> float:
    """
    Calculate a continuity score (0.0-1.0) indicating if current query continues last conversation.

    Args:
        current_query: Current user query
        last_query: Previous conversation query
        time_diff_seconds: Time elapsed since last conversation
        both_heavy: True if both queries are heavy topics
        same_topic: True if both have same detected topic
        current_topic: The current topic (to filter "general")
        last_was_heavy: True if last conversation was heavy

    Returns:
        Continuity score (0.0-1.0). Score >= 0.5 suggests thread continuity.
    """
    score = 0.0
    
    # Hard cutoff: too much time = auto-break
    if time_diff_seconds > THREAD_TIME_HARD_CUTOFF:
        return 0.0
    
    # Explicit break markers override everything
    if has_thread_break_marker(current_query):
        return 0.0
    
    # Factor 1: Keyword overlap
    last_keywords = extract_thread_keywords(last_query)
    curr_keywords = extract_thread_keywords(current_query)

    if last_keywords and curr_keywords:
        overlap = len(last_keywords & curr_keywords)
        union = len(last_keywords | curr_keywords)
        overlap_ratio = overlap / union if union > 0 else 0.0
        keyword_score = overlap_ratio * THREAD_WEIGHT_KEYWORDS
        score += keyword_score
        logger.debug(
            f"[Thread] Keyword overlap: last={last_keywords}, curr={curr_keywords}, "
            f"overlap={overlap}, union={union}, ratio={overlap_ratio:.3f}, score_contrib={keyword_score:.3f}"
        )
    else:
        logger.debug(f"[Thread] No keywords extracted from one or both queries")
    
    # Factor 2: Time proximity
    if time_diff_seconds < THREAD_TIME_CLOSE_BONUS:
        # Very recent (< 5 min): full time bonus
        score += THREAD_WEIGHT_TIME
    elif time_diff_seconds < THREAD_TIME_MEDIUM_BONUS:
        # Medium recent (< 30 min): decaying bonus
        decay = 1.0 - ((time_diff_seconds - THREAD_TIME_CLOSE_BONUS) / 
                       (THREAD_TIME_MEDIUM_BONUS - THREAD_TIME_CLOSE_BONUS))
        score += THREAD_WEIGHT_TIME * decay
    # else: no time bonus (but not auto-break unless > 2 hours)
    
    # Factor 3: Heavy topic continuity
    # Full bonus if both are heavy
    if both_heavy:
        score += THREAD_WEIGHT_HEAVY
    # Partial bonus if previous was heavy and we're discussing same specific topic
    # (follow-up questions about a crisis are likely to continue the thread)
    elif last_was_heavy and same_topic and current_topic and current_topic.lower() != "general":
        score += THREAD_WEIGHT_HEAVY * 0.5  # Half credit for heavy topic continuity
    
    # Factor 4: Same detected topic (exclude "general")
    # Give larger bonus for specific topic match (helps when keywords don't overlap semantically)
    if same_topic and current_topic and current_topic.lower() != "general":
        # Double the topic weight for non-general topics (0.2 instead of 0.1)
        # This helps when discussing the same specific topic with different vocabulary
        score += THREAD_WEIGHT_TOPIC * 2.0

    return score


def belongs_to_thread(
    current_query: str,
    last_conversation: dict,
    current_topic: Optional[str] = None
) -> bool:
    """
    Determine if current query continues the immediate previous conversation thread.

    Threads are strictly consecutive - any topic switch breaks the thread.

    Args:
        current_query: Current user query
        last_conversation: Dict with keys: query, response, timestamp, is_heavy_topic, topic, thread_depth
        current_topic: Current detected topic

    Returns:
        True if current query continues the thread
    """
    from datetime import datetime

    # Calculate time difference
    last_time = last_conversation.get("timestamp")
    if isinstance(last_time, str):
        try:
            last_time = datetime.fromisoformat(last_time.replace("Z", "+00:00"))
            time_diff = (datetime.now() - last_time).total_seconds()
        except Exception:
            # Can't parse timestamp, use conservative time bonus
            time_diff = 3600.0  # Assume 1 hour
    elif isinstance(last_time, datetime):
        time_diff = (datetime.now() - last_time).total_seconds()
    else:
        time_diff = 3600.0  # Default

    # Check if both are heavy topics
    last_heavy = last_conversation.get("is_heavy_topic", False)
    curr_heavy = _is_heavy_topic_heuristic(current_query)

    # Check if same topic
    last_topic = last_conversation.get("topic", "general")
    same_topic = (last_topic == current_topic) if current_topic else False

    # Build previous conversation text (query + response for better keyword matching)
    last_query = last_conversation.get("query", "")
    last_response = last_conversation.get("response", "")
    last_full_text = f"{last_query} {last_response}"

    # Calculate continuity score
    score = calculate_thread_continuity_score(
        current_query=current_query,
        last_query=last_full_text,  # Use full conversation text
        time_diff_seconds=time_diff,
        both_heavy=(last_heavy and curr_heavy),
        same_topic=same_topic,
        current_topic=current_topic,
        last_was_heavy=last_heavy  # Pass last_heavy for partial credit
    )

    # Thread momentum bonus: if already in a thread (depth >= 2), give a small bonus
    # This helps maintain longer threads even when keywords don't overlap perfectly
    last_depth = last_conversation.get("thread_depth", 1)
    if last_depth >= 2 and same_topic and current_topic and current_topic.lower() != "general":
        # Add 0.1 bonus for thread momentum (helps reach 0.5 threshold)
        momentum_bonus = 0.1
        score += momentum_bonus
        logger.debug(f"[Thread] Thread momentum bonus: +{momentum_bonus:.3f} (depth={last_depth})")

    # Debug logging
    logger.debug(
        f"[Thread] Continuity check: score={score:.3f}, threshold={THREAD_CONTINUITY_THRESHOLD}, "
        f"time_diff={time_diff:.1f}s, last_heavy={last_heavy}, curr_heavy={curr_heavy}, "
        f"same_topic={same_topic} (last={last_topic}, curr={current_topic}), depth={last_depth}"
    )

    return score >= THREAD_CONTINUITY_THRESHOLD
