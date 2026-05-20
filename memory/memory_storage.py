# memory/memory_storage.py
"""
Memory storage module.

Module Contract
- Purpose: Implements the MemoryStorageProtocol contract for persisting memories. Handles storage of conversations, facts, reflections, and summaries to both corpus and ChromaDB.
- Inputs:
  - store_interaction(query, response, tags=?, session_id=?, provenance=?) -> Optional[str]
    - session_id: optional session identifier for audit trail [NEW 2026-03-26]
    - provenance: optional dict with response_mode, thinking_block (truncated to PROVENANCE_THINKING_MAX_CHARS), cited_ids, model_name, prompt_hash, agentic_summary [NEW 2026-03-26]
  - store_fact(fact_dict) -> bool
  - add_reflection(text, tags=?, source=?, timestamp=?) -> bool
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
    fact_scope, entity_type, user_connection, source_excerpt through to ChromaDB metadata [NEW 2026-03]
  - Thread metadata forwarding: store_interaction() propagates thread_id and thread_depth
    from thread_info to ChromaDB conversation metadata [NEW 2026-03]
  - Reflection embedding cleanup: _clean_reflection_for_embedding() strips boilerplate
    headers/prefixes (16+ patterns) before embedding to prevent vector collapse across
    reflections. Original text preserved in metadata for display [NEW 2026-05]
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


def _clean_reflection_for_embedding(text: str) -> str:
    """
    Strip boilerplate headers/prefixes from reflection text for embedding.

    Session reflections typically start with identical markdown structure:
      ### What went well
      - The assistant effectively...
      ### What could improve
      - ...

    This causes embedding collapse (all reflections map to similar vectors).
    Stripping the boilerplate lets the embedding model focus on the
    substantive, differentiating content.

    The original markdown is preserved in metadata for display.
    """
    if not text:
        return text

    lines = text.split("\n")
    cleaned = []

    _SKIP_PREFIXES = (
        "### what went well",
        "### what could improve",
        "### what could be improved",
        "### areas for improvement",
        "### summary",
        "### overall",
    )
    _SKIP_STARTS = (
        "the assistant effectively",
        "the assistant provided",
        "the assistant demonstrated",
        "the assistant acknowledged",
        "the assistant recognized",
        "the assistant successfully",
        "the assistant showed",
        "the assistant maintained",
        "the assistant helped",
        "there was a lack of",
        "there was effective",
        "clear acknowledgment of",
        "overall, the conversation",
        "overall, the assistant",
        "in summary,",
        "the conversation was",
    )

    for line in lines:
        stripped = line.strip().lstrip("•*- ")
        lower = stripped.lower()

        # Skip markdown headers
        if lower.startswith("#"):
            lower_no_hash = lower.lstrip("# ").strip()
            if any(lower_no_hash.startswith(p.lstrip("# ")) for p in _SKIP_PREFIXES):
                continue

        # Skip empty lines
        if len(stripped) < 5:
            continue

        # Skip boilerplate sentence openings
        if any(lower.startswith(s) for s in _SKIP_STARTS):
            continue

        cleaned.append(stripped)

    result = " ".join(cleaned)
    # Fall back to original if cleaning removed everything
    return result if len(result) > 20 else text


# ---------------------------------------------------------------------------
# Reflection retrieval text generation (v2)
# ---------------------------------------------------------------------------

# Boilerplate patterns to strip entirely from retrieval text
_REFLECTION_BOILERPLATE_HEADERS = {
    "what went well", "what could improve", "what could be improved",
    "areas for improvement", "summary", "overall", "what to improve",
    "high-level insights", "key insights",
}

_REFLECTION_BOILERPLATE_STARTS = (
    "the assistant effectively",
    "the assistant provided",
    "the assistant demonstrated",
    "the assistant acknowledged",
    "the assistant recognized",
    "the assistant successfully",
    "the assistant showed",
    "the assistant maintained",
    "the assistant helped",
    "the assistant offered",
    "the assistant asked",
    "the assistant displayed",
    "the assistant was",
    "the assistant initially",
    "the assistant could",
    "the assistant should",
    "the conversation was",
    "the conversation focused",
    "the conversation maintained",
    "the conversation showed",
    "there was a lack",
    "there was effective",
    "there was a missed",
    "clear acknowledgment",
    "clear communication",
    "overall, the conversation",
    "overall, the assistant",
    "overall, the session",
    "in summary,",
    "in future conversations",
    "in future interactions",
    "this session showed",
    "this session demonstrated",
    "it maintained a",
    "it effectively",
    "responses were generally",
    "responses were well",
)

# Generic filler phrases to remove from within lines
_REFLECTION_FILLER = (
    "which helped in building rapport",
    "which helped in maintaining",
    "which helped to maintain",
    "which helped create a deeper connection",
    "demonstrating adaptability to user preferences",
    "contributing to a positive conversational flow",
    "promoting a positive interaction",
    "allowing for an engaging dialogue",
    "maintaining clarity about",
    "which can help in building",
    "indicating a need for",
    "particularly when the user",
)


def _extract_reflection_retrieval_text(text: str) -> str:
    """
    Generate a compact, topic-dense retrieval text from reflection markdown.

    Unlike _clean_reflection_for_embedding (which just strips headers),
    this function aggressively extracts only the distinguishing content:
    specific topics, entities, actions, decisions, and states.

    Returns a structured retrieval document optimized for embedding search.
    """
    if not text:
        return ""

    # Handle single-line format (YAML serialization collapses newlines).
    # Re-split on markdown headers and bullet markers that appear mid-line.
    normalized = text
    # Split "### Header" markers back to newlines
    normalized = re.sub(r'\s*(###\s+)', r'\n\1', normalized)
    # Split header line from its content: "### What went well - content" → header + content
    normalized = re.sub(
        r'(###\s+(?:What\s+(?:went\s+well|to\s+improve|could\s+(?:improve|be\s+improved))'
        r'|Areas?\s+for\s+improvement|Summary|Overall|High-level\s+insights|Key\s+insights))'
        r'\s*[-–—]\s*',
        r'\1\n- ',
        normalized,
        flags=re.IGNORECASE,
    )
    # Split " - " bullet points that follow a period or end of sentence
    normalized = re.sub(r'(?<=[.!?])\s+- ', r'\n- ', normalized)

    lines = normalized.split("\n")
    substantive = []
    entities = set()
    topics = set()

    for line in lines:
        stripped = line.strip().lstrip("•*-# ")
        if len(stripped) < 10:
            continue
        lower = stripped.lower()

        # Skip headers
        if line.strip().startswith("#"):
            header_text = line.strip().lstrip("# ").lower().strip()
            if any(header_text.startswith(h) for h in _REFLECTION_BOILERPLATE_HEADERS):
                continue

        # Skip boilerplate sentence starts
        if any(lower.startswith(s) for s in _REFLECTION_BOILERPLATE_STARTS):
            # But still extract specific nouns/entities from these lines
            _extract_specifics(stripped, entities, topics)
            continue

        # Remove filler phrases from the line
        cleaned_line = stripped
        for filler in _REFLECTION_FILLER:
            cleaned_line = cleaned_line.replace(filler, "")
        cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()

        if len(cleaned_line) > 10:
            substantive.append(cleaned_line)
            _extract_specifics(cleaned_line, entities, topics)

    # Clean up entities: remove fragments and possessives
    entities = _clean_entities(entities)

    # Clean up topics: remove duplicates, substrings, and trim
    topics = _dedupe_topics(topics)

    # Build retrieval text as natural language (not structured format).
    # Embedding models perform better with natural text than key-value pairs.
    # Prepend entities/topics as a brief context line, then substantive content.
    parts = []

    # Brief context prefix with entities and topics (helps disambiguation)
    context_tokens = []
    for e in sorted(entities)[:5]:
        context_tokens.append(e)
    for t in sorted(topics)[:4]:
        # Only add topics that aren't already covered by entities
        if not any(t.lower() in e.lower() or e.lower() in t.lower() for e in context_tokens):
            context_tokens.append(t)
    if context_tokens:
        parts.append(" ".join(context_tokens[:6]))

    if substantive:
        content = " ".join(substantive)
        if len(content) > 500:
            content = content[:500]
        parts.append(content)

    result = " ".join(parts)
    return result if len(result) > 20 else _clean_reflection_for_embedding(text)


def _clean_entities(entities: set) -> set:
    """Remove fragment entities and normalize possessives."""
    result = set()
    for e in entities:
        # Skip fragments starting with 's or possessive leftovers
        if e.startswith("s ") or e.startswith("'s"):
            continue
        # Remove trailing 's from possessives: "Auggie's" → "Auggie"
        if e.endswith("'s"):
            e = e[:-2]
        # Skip very short or common
        if len(e) < 3 or e.lower() in _COMMON_WORDS or e in _HEADER_WORDS:
            continue
        result.add(e)
    return result


def _dedupe_topics(topics: set) -> set:
    """Remove duplicate/substring topics, trim to reasonable length."""
    # Trim each topic
    trimmed = set()
    for t in topics:
        t = t.strip().rstrip(".,;)")
        if len(t) > 50:
            t = t[:50]
        if len(t) > 4:
            trimmed.add(t)

    # Remove substrings (keep the longer version)
    result = set()
    sorted_topics = sorted(trimmed, key=len, reverse=True)
    for t in sorted_topics:
        # Check if this topic is a substring of any already-kept topic
        if not any(t.lower() in kept.lower() for kept in result):
            result.add(t)

    return result


def _extract_specifics(text: str, entities: set, topics: set):
    """
    Extract specific nouns, entities, and topics from a line.
    Looks for capitalized words, quoted terms, technical terms, and
    specific patterns that indicate real content vs generic language.
    """
    # Capitalized proper nouns (3+ chars, not sentence-start, not common)
    words = text.split()
    for i, word in enumerate(words):
        clean = word.strip(".,!?\"'()[]{}:;-")
        if len(clean) < 3:
            continue
        # Proper nouns: capitalized, not at sentence start, not common
        if (i > 0 and clean[0].isupper() and clean.lower() not in _COMMON_WORDS
                and not clean.isupper() and clean not in _HEADER_WORDS):
            entities.add(clean)

    # Multi-word proper nouns: "Auggie's name" → extract "Auggie"
    for m in re.finditer(r"(?<!\w')(?<!\w)\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*)\b", text):
        term = m.group(1).strip()
        if len(term) > 2 and term.lower() not in _COMMON_WORDS and term not in _HEADER_WORDS:
            entities.add(term)

    # Quoted terms
    for m in re.finditer(r'"([^"]{3,40})"', text):
        entities.add(m.group(1))
    for m in re.finditer(r"'([^']{3,30})'", text):
        term = m.group(1)
        if term.lower() not in _COMMON_WORDS:
            entities.add(term)

    # Technical/specific terms (patterns)
    tech_patterns = [
        r'\b(FAISS|ChromaDB|LLM|API|RAG|NLP|ML|AI|GPU|CPU|OMSA|SVM|GAN|BGE|IVF)\b',
        r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b',  # CamelCase
        r'\b\w+\.py\b',  # Python files
        r'\b(?:Python|JavaScript|TypeScript|React|FastAPI|Django|Flask|Gradio)\b',
    ]
    for pattern in tech_patterns:
        for m in re.finditer(pattern, text):
            entities.add(m.group(0))

    # Parenthetical specifics: "e.g., X and Y" or "(e.g., X)"
    for m in re.finditer(r'\(e\.g\.,?\s*([^)]{5,80})\)', text):
        content = m.group(1).strip()
        topics.add(content)
    for m in re.finditer(r'e\.g\.,?\s+([^,.\n)]{5,60})', text):
        content = m.group(1).strip().rstrip(")")
        if len(content) > 4:
            topics.add(content)

    # Topic extraction: "regarding X", "about X", "related to X"
    topic_patterns = [
        r'(?:regarding|about|around|concerning|related to|involving)\s+(?:the\s+)?(?:user\'?s?\s+)?([^,.\n]{5,60})',
        r'(?:topics?\s+(?:like|such as|including))\s+([^,.\n]{5,60})',
        r'(?:discussed|mentioned|explored|addressed)\s+([^,.\n]{5,60})',
        r'(?:such as|like)\s+(?:the\s+)?([^,.\n]{5,50})',
    ]
    for pattern in topic_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            topic = m.group(1).strip().rstrip(".,;)")
            if len(topic) > 4 and topic.lower() not in _COMMON_WORDS:
                topics.add(topic)


# Words that appear in headers/boilerplate but should not be extracted as entities.
# Includes capitalized verbs/adjectives common in reflection bullet points.
_HEADER_WORDS = frozenset({
    "What", "Went", "Well", "Could", "Improve", "Areas", "Improvement",
    "Summary", "Overall", "Insights", "High", "Key", "Level",
    "Technical", "Effective", "Suggestions", "Response", "However",
    "Additionally", "Furthermore", "Moreover", "Particularly",
    "Specifically", "Generally", "Sometimes", "Perhaps",
    # Capitalized verbs/adjectives common at bullet starts
    "Prioritize", "Prioritizing", "Incorporate", "Incorporating",
    "Confirming", "Confirmed", "Regularly", "Summarize", "Summarizing",
    "Structure", "Structured", "Maintaining", "Maintained",
    "Avoiding", "Addressed", "Continued", "Continuing",
    "Providing", "Recognizing", "Acknowledged", "Acknowledging",
    "Encouraging", "Encouraged", "Discussing", "Discussed",
    "Exploring", "Explored", "Clarifying", "Clarified",
    "Balancing", "Balanced", "Tracking", "Tracked",
    "Responding", "Responded", "Supporting", "Supported",
    "Adapting", "Adapted", "Engaging", "Engaged",
    "Offering", "Offered", "Checking", "Checked",
    "Validating", "Validated", "Reviewing", "Reviewed",
    "Proactive", "Contextual", "Personalization", "Personalized",
    "Awareness", "Always", "Never", "Often", "Rarely",
    "Sometimes", "Consider", "Ensure", "Focus",
    # Generic reflection evaluation words
    "Strengths", "Weaknesses", "Opportunities", "Challenges",
    "Progress", "Growth", "Development", "Achievement",
})


def extract_reflection_metadata(text: str) -> dict:
    """
    Extract structured metadata from a reflection for storage and retrieval.

    Returns dict with: primary_topic, secondary_topics, entities, themes,
    emotional_tone, project_area.
    """
    entities = set()
    topics = set()

    for line in text.split("\n"):
        stripped = line.strip().lstrip("•*- ")
        lower = stripped.lower()
        # Skip headers
        if stripped.startswith("#"):
            continue
        # Skip boilerplate
        if any(lower.lstrip("# ").startswith(s) for s in _REFLECTION_BOILERPLATE_STARTS):
            continue
        if len(stripped) > 10:
            _extract_specifics(stripped, entities, topics)

    # Clean entities
    entities = _clean_entities(entities)

    # Detect emotional tone
    emotional_tone = _detect_reflection_tone(text)

    # Detect project area
    project_area = _detect_project_area(text)

    # Clean and dedupe topics
    topics = _dedupe_topics(topics)

    # Primary topic = longest/most specific topic found
    sorted_topics = sorted(topics, key=len, reverse=True)
    primary_topic = sorted_topics[0] if sorted_topics else ""
    secondary_topics = sorted_topics[1:5] if len(sorted_topics) > 1 else []

    # Themes (high-level categories)
    themes = _detect_themes(text)

    return {
        "primary_topic": primary_topic[:100],
        "secondary_topics": ",".join(secondary_topics)[:200],
        "entities": ",".join(sorted(entities)[:10])[:200],
        "themes": ",".join(sorted(themes)[:5])[:100],
        "emotional_tone": emotional_tone,
        "project_area": project_area,
    }


def _detect_reflection_tone(text: str) -> str:
    """Detect the emotional tone of a reflection."""
    lower = text.lower()
    tones = []
    if any(w in lower for w in ("anxiety", "anxious", "worried", "stress", "overwhelm")):
        tones.append("anxious")
    if any(w in lower for w in ("happy", "excited", "proud", "accomplish", "breakthrough")):
        tones.append("positive")
    if any(w in lower for w in ("frustrated", "stuck", "confused", "difficult")):
        tones.append("frustrated")
    if any(w in lower for w in ("sad", "lonely", "grief", "loss", "miss")):
        tones.append("sad")
    if any(w in lower for w in ("calm", "balanced", "stable", "grounded")):
        tones.append("calm")
    if any(w in lower for w in ("productive", "focused", "progress", "momentum")):
        tones.append("productive")
    return ",".join(tones) if tones else "neutral"


def _detect_project_area(text: str) -> str:
    """Detect project/domain area from reflection text."""
    lower = text.lower()
    if any(w in lower for w in ("daemon", "rag", "retrieval", "chromadb", "embedding")):
        return "daemon"
    if any(w in lower for w in ("school", "course", "study", "homework", "lecture", "omsa", "exam")):
        return "academic"
    if any(w in lower for w in ("health", "medication", "doctor", "sleep", "exercise", "gym")):
        return "health"
    if any(w in lower for w in ("work", "job", "career", "interview", "internship")):
        return "career"
    if any(w in lower for w in ("family", "brother", "sister", "mom", "dad", "partner")):
        return "relationships"
    return ""


def _detect_themes(text: str) -> set:
    """Detect high-level themes from reflection content."""
    lower = text.lower()
    themes = set()
    theme_keywords = {
        "technical": ("code", "programming", "api", "system", "architecture", "bug", "test"),
        "emotional": ("feeling", "emotion", "anxiety", "stress", "happy", "sad", "cope"),
        "academic": ("study", "course", "lecture", "exam", "homework", "school", "grade"),
        "health": ("health", "sleep", "exercise", "medication", "diet", "pain", "doctor"),
        "social": ("friend", "family", "relationship", "partner", "conversation", "empathy"),
        "productivity": ("goal", "progress", "deadline", "task", "plan", "schedule", "focus"),
        "creative": ("idea", "brainstorm", "explore", "experiment", "novel", "design"),
    }
    for theme, keywords in theme_keywords.items():
        if any(k in lower for k in keywords):
            themes.add(theme)
    return themes


# Common words to exclude from entity/topic extraction
_COMMON_WORDS = frozenset({
    "the", "this", "that", "these", "those", "they", "them", "their",
    "what", "which", "where", "when", "who", "whom", "whose", "why", "how",
    "about", "above", "after", "again", "against", "also", "although",
    "because", "before", "between", "both", "could", "would", "should",
    "does", "have", "having", "here", "there", "very", "more", "most",
    "some", "such", "than", "then", "from", "into", "with", "without",
    "been", "being", "each", "every", "well", "good", "better", "best",
    "user", "assistant", "conversation", "response", "responses",
    "session", "interaction", "interactions", "future", "overall",
    "effectively", "particularly", "specifically", "generally",
    "however", "moreover", "furthermore", "additionally", "provide",
    "provided", "providing", "demonstrated", "maintained", "showed",
    "helped", "included", "discussed", "addressed", "offered",
    "needs", "need", "important", "helpful", "relevant", "appropriate",
    "clear", "effective", "positive", "negative", "better", "improved",
    "true", "false", "none", "null", "undefined",
})


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
        from utils.time_manager import now_from
        return now_from(self.time_manager)

    def _now_iso(self) -> str:
        from utils.time_manager import now_iso_from
        return now_iso_from(self.time_manager)

    def _calculate_truth_score(self, query: str, response: str) -> float:
        """Calculate truth score using scorer (falls back to 0.5 if no scorer)."""
        if self.scorer and hasattr(self.scorer, 'calculate_truth_score'):
            return self.scorer.calculate_truth_score(query, response)
        return 0.5

    def _calculate_importance_score(self, content: str) -> float:
        """Calculate importance score using scorer (falls back to 0.5 if no scorer)."""
        if self.scorer and hasattr(self.scorer, 'calculate_importance_score'):
            return self.scorer.calculate_importance_score(content)
        return 0.5

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
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        provenance: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Persist a turn in both corpus & Chroma with computed metadata.

        Args:
            query: User input
            response: Assistant response
            tags: Optional list of tags
            session_id: Optional session identifier for provenance tracking
            provenance: Optional dict with audit-trail fields (response_mode,
                        thinking_block, cited_ids, agentic_rounds, model_name, etc.)

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

            # Forward thread metadata to ChromaDB for expand_memory window slicing
            if thread_info.get("thread_id"):
                raw_metadata["thread_id"] = thread_info["thread_id"]
            if thread_info.get("depth") is not None:
                raw_metadata["thread_depth"] = int(thread_info["depth"])

            # Session ID for provenance tracking
            if session_id:
                raw_metadata["session_id"] = session_id

            # Content type detection (lyrics, poems, code, quotes, etc.)
            try:
                from core.content_type_detector import detect_content_type
                ct = detect_content_type(query)
                if ct.content_type:
                    raw_metadata["content_type"] = ct.content_type
                    if ct.title_hint:
                        raw_metadata["content_title"] = ct.title_hint
                    if ct.attribution_hint:
                        raw_metadata["content_attribution"] = ct.attribution_hint
            except Exception:
                pass  # Non-fatal — content type detection is best-effort

            # Provenance metadata (audit trail)
            if provenance and isinstance(provenance, dict):
                try:
                    from config.app_config import PROVENANCE_ENABLED, PROVENANCE_THINKING_MAX_CHARS
                except ImportError:
                    PROVENANCE_ENABLED = True
                    PROVENANCE_THINKING_MAX_CHARS = 4000
                if PROVENANCE_ENABLED:
                    import json as _json
                    if provenance.get("thinking_block"):
                        tb = str(provenance["thinking_block"])
                        raw_metadata["thinking_block"] = tb[:PROVENANCE_THINKING_MAX_CHARS]
                        if len(tb) > PROVENANCE_THINKING_MAX_CHARS:
                            raw_metadata["thinking_block_truncated"] = True
                    if provenance.get("response_mode"):
                        raw_metadata["response_mode"] = provenance["response_mode"]
                    if provenance.get("model_name"):
                        raw_metadata["model_name"] = provenance["model_name"]
                    if provenance.get("cited_ids"):
                        raw_metadata["cited_memory_ids"] = _json.dumps(provenance["cited_ids"])
                    if provenance.get("final_prompt_hash"):
                        raw_metadata["prompt_hash"] = provenance["final_prompt_hash"]
                    if provenance.get("agentic_rounds"):
                        compact = [
                            {"r": r.get("round"), "a": r.get("action", "")[:60]}
                            for r in provenance["agentic_rounds"]
                        ]
                        raw_metadata["agentic_summary"] = _json.dumps(compact)

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
        # Generate topic-dense retrieval text for embedding. Original markdown
        # is preserved in metadata for display.
        try:
            if hasattr(self.chroma_store, "add_to_collection"):
                embedding_text = _extract_reflection_retrieval_text(text)
                # Extract structured metadata for entity/topic overlap scoring
                refl_meta = extract_reflection_metadata(text)
                md = {
                    "timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                    "type": "reflection",
                    "tags": ",".join(tags),
                    "source": source,
                    "importance_score": 0.7,
                    "original_text": text[:2000],  # preserve for display
                    "primary_topic": refl_meta.get("primary_topic", ""),
                    "secondary_topics": refl_meta.get("secondary_topics", ""),
                    "entities": refl_meta.get("entities", ""),
                    "themes": refl_meta.get("themes", ""),
                    "emotional_tone": refl_meta.get("emotional_tone", ""),
                    "project_area": refl_meta.get("project_area", ""),
                }
                # Ensure collection exists
                if (getattr(self.chroma_store, "collections", None) is not None
                    and "reflections" not in self.chroma_store.collections
                    and hasattr(self.chroma_store, "create_collection")):
                    try:
                        self.chroma_store.create_collection("reflections")
                    except Exception:
                        pass
                self.chroma_store.add_to_collection("reflections", embedding_text, md)
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
                for key in ("fact_scope", "entity_type", "user_connection", "source_excerpt"):
                    val = md.get(key)
                    if val:
                        source_dict[key] = val[:200] if key == "source_excerpt" else val

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
            from memory.graph_utils import is_junk_entity
            from config.app_config import KNOWLEDGE_GRAPH_MIN_CONFIDENCE

            if confidence < KNOWLEDGE_GRAPH_MIN_CONFIDENCE:
                return

            # Filter junk subjects (pronouns, stopwords, numbers) — these
            # should never become graph nodes. "user" is exempt.
            if subj.lower() != "user" and is_junk_entity(subj):
                logger.debug(f"[MemoryStorage] Graph skip junk subject: '{subj}'")
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

            # Ensure collection exists (lazy init handles creation)
            try:
                coll = self.chroma_store._get_collection(collection_name)
            except (ValueError, Exception) as e:
                logger.warning(f"[MemoryStorage] procedural_skills collection not available: {e}")
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
                    except Exception:
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
