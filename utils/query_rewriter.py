"""
utils/query_rewriter.py

Query rewriting for improved semantic search.
Expands casual/slang queries into semantic-rich versions for better retrieval.

Module Contract:
- Purpose: Expand user queries with topics, synonyms, and related terms before semantic search
- Inputs: Raw user query string
- Outputs: Expanded query string with additional keywords
- Dependencies: topic_manager, spaCy NER
- Side effects: None (pure transformation)
- Ordering (2026-08-23): the rewritten string feeds an order-sensitive embedding
  model (bge), so keywords keep the user's first-occurrence order and synonym
  expansions append after them in deterministic SYNONYM_GROUPS order — a set()
  round-trip here used to scramble word order and change the query vector
  run-to-run for the same query.
"""

import contextvars
import re
from typing import List, NamedTuple, Optional, Set
from utils.logging_utils import get_logger
import utils.topic_manager as topic_manager
from utils.repository_context import _SOFTWARE_ARTIFACT_MATCHER  # the one software-vocabulary matcher
from utils.test_envelope import inner_text

logger = get_logger(__name__)

# Domain-specific synonym expansion
SYNONYM_GROUPS = {
    'gym': ['gym', 'workout', 'exercise', 'fitness', 'training', 'lift', 'bench', 'squat', 'deadlift'],
    'work': ['work', 'job', 'shift', 'employment'],
    'tired': ['tired', 'exhausted', 'beat', 'fatigued', 'drained', 'worn out'],
    'health': ['health', 'medication', 'med', 'covid', 'sick', 'illness'],
    'body': ['body', 'physique', 'shape', 'fit', 'fitness', 'weight', 'muscle'],
    'code': ['code', 'coding', 'programming', 'debug', 'refactor', 'project'],
    'cat': ['cat', 'kitty', 'kitten', 'pet', 'feline'],
}

# Noise words to remove
NOISE_WORDS = {
    'lmao', 'lol', 'haha', 'ugh', 'hmm', 'um', 'uh', 'like',
    'just', 'really', 'very', 'actually', 'basically', 'literally',
    'kinda', 'sorta', 'gonna', 'wanna'
}

def expand_with_synonyms(keywords: Set[str]) -> Set[str]:
    """Expand keywords with domain-specific synonyms."""
    expanded = set(keywords)

    for keyword in keywords:
        keyword_lower = keyword.lower()
        # Check if keyword matches any synonym group
        for group_key, synonyms in SYNONYM_GROUPS.items():
            if keyword_lower in synonyms or keyword_lower == group_key:
                # Add all synonyms from this group
                expanded.update(synonyms)
                break

    return expanded

def extract_keywords(query: str) -> Set[str]:
    """Extract meaningful keywords from query, removing noise."""
    # Tokenize - keep alphanumeric words 3+ chars
    words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())

    # Remove noise words
    keywords = {w for w in words if w not in NOISE_WORDS}

    return keywords

def _ordered_keywords(text: str) -> List[str]:
    """Keywords in first-occurrence order, noise removed (order matters: the
    rewritten query is embedded by an order-sensitive model)."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [w for w in dict.fromkeys(words) if w not in NOISE_WORDS]

# Sense gate (2026-10-25, class: BC-05): SYNONYM_GROUPS is keyed on surface words,
# but "work"/"job"/"shift"/"training"/"fit"/"shape"/"weight" are homonyms of
# software vocabulary ("...daemon codebase last couple days work" gained
# "job shift employment"). When the query carries a software-domain anchor only
# the 'code' group expands; the user's own words always stay.
_CODE_GROUP = 'code'
_CODE_ANCHORS = frozenset(SYNONYM_GROUPS[_CODE_GROUP]) - {'project'}  # bare "project" is ambiguous


def has_software_anchor(text: str, keywords: Optional[List[str]] = None) -> bool:
    """True when ``text`` (and its keyword list) is about software work."""
    if _SOFTWARE_ARTIFACT_MATCHER(text.lower()):
        return True
    return bool(_CODE_ANCHORS.intersection(keywords if keywords is not None else _ordered_keywords(text)))


def rewrite_query(query: str, use_topic_extraction: bool = True,
                  topics: Optional[List[str]] = None) -> str:
    """
    Rewrite a casual query into a semantic-rich version for better retrieval.

    Args:
        query: Original user query (e.g., "No more body image issue than determination lmao")
        use_topic_extraction: Whether to use topic_manager for entity extraction
        topics: topics already extracted this turn (ContextPipeline Stage 1);
            when given they replace the TopicManager call (an LLM round-trip)

    Returns:
        Expanded query string with additional keywords

    Example:
        >>> rewrite_query("No more body image issue than determination lmao")
        "body image gym workout exercise fitness determination motivation"
    """
    if not query or len(query.strip()) < 3:
        return query

    # A [test]-wrapped probe is judged by what it said: unwrap (the one
    # unwrap, utils.test_envelope.inner_text) so the whole multi-line text is
    # keyworded and the envelope tag never becomes a retrieval term.
    query = inner_text(query)
    logger.debug(f"[QueryRewriter] Original query: '{query}'")

    # 1. Base keywords in the user's own word order
    ordered = _ordered_keywords(query)

    # 2. Try topic extraction if available (topic words append after the
    #    user's words, in topic order)
    if use_topic_extraction:
        try:
            if topics:
                found = list(topics)
            else:
                topic_mgr = topic_manager.TopicManager()
                topic = topic_mgr.get_primary_topic(query)
                found = [topic] if topic else []

            if found:
                for topic in found[:5]:  # Limit to top 5
                    for word in _ordered_keywords(topic):
                        if word not in ordered:
                            ordered.append(word)
                logger.debug(f"[QueryRewriter] Extracted topics: {found[:5]}")
        except Exception as e:  # degrades: topic keywords missing from the rewritten retrieval query
            logger.debug(f"[QueryRewriter] Topic extraction failed: {e}")

    # 3. Synonym expansions, appended in deterministic order: keywords in
    #    query order, each keyword's first matching group in SYNONYM_GROUPS
    #    order, synonyms in group-list order.
    seen = set(ordered)
    additional_keywords = []
    # anchor text = the query itself + any topic words appended after its keywords
    topic_words = ordered[len(_ordered_keywords(query)):]
    software_query = has_software_anchor(' '.join([query] + topic_words), ordered)
    for keyword in ordered:
        for group_key, synonyms in SYNONYM_GROUPS.items():
            if keyword == group_key or keyword in synonyms:
                if software_query and group_key != _CODE_GROUP:
                    break  # homonym of the query's software sense: no foreign-domain terms
                for syn in synonyms:
                    if syn not in seen:
                        additional_keywords.append(syn)
                        seen.add(syn)
                break

    # 4. Combine: original + additional (limit total to avoid bloat)
    all_keywords = ordered + additional_keywords[:10]
    rewritten = ' '.join(all_keywords)

    logger.info(f"[QueryRewriter] Rewritten: '{query}' → '{rewritten}'")
    logger.debug(f"[QueryRewriter] Keyword count: {len(ordered)} original + {len(additional_keywords[:10])} expanded")

    return rewritten

# ---------------------------------------------------------------------------
# Per-turn hand-off from ContextPipeline (2026-10-25, class: BC-31, BC-82)
#
# Two independent pieces, both published once per turn by ContextPipeline:
#   * topics     — the Stage-1 topic extraction. rewrite_query() used to call
#                  TopicManager (a ~1 s LLM round-trip) for the SAME topic;
#                  the hybrid retriever now reuses it. Not a rewrite: published
#                  regardless of the rewrite switch.
#   * rewritten  — the pipeline's LLM rewrite sentence (only when
#                  features.enable_query_rewrite). An ADDITIONAL candidate
#                  query for the hybrid retriever; the local keyword/synonym/
#                  graph expansion and the memory gate (raw query) are
#                  unchanged.
# Keyed to the raw turn text so an unrelated retrieval in the same turn
# (agentic tool query, "" fallbacks) never picks it up; a task created after
# set() inherits it (same seam as context_pipeline._PRECOMPUTED_FILES).
# ---------------------------------------------------------------------------
class TurnHint(NamedTuple):
    rewritten: Optional[str]
    topics: Optional[List[str]]


_TURN_HINT: "contextvars.ContextVar[Optional[tuple]]" = contextvars.ContextVar(
    "daemon_turn_query_hint", default=None
)


def set_turn_hint(raw_query: Optional[str], rewritten: Optional[str] = None,
                  topics: Optional[List[str]] = None) -> None:
    """Publish this turn's hint (or clear the previous turn's with no args)."""
    raw = (raw_query or "").strip()
    new = (rewritten or "").strip() or None
    tops = [t for t in (topics or []) if isinstance(t, str) and t.strip()] or None
    _TURN_HINT.set((raw, TurnHint(new, tops)) if raw and (new or tops) else None)


def get_turn_hint(query: Optional[str]) -> Optional[TurnHint]:
    """The turn's hint when ``query`` is the raw turn text or that text with
    terms appended (graph expansion); None otherwise."""
    hint = _TURN_HINT.get()
    if not hint or not query:
        return None
    raw, value = hint
    return value if query.strip().startswith(raw) else None


def query_rewrite_enabled() -> bool:
    """The ONE live switch (features.enable_query_rewrite). Read at call
    time: Settings flips it mid-process (BC-82)."""
    from config import app_config  # lazy import: live-config read
    return bool(app_config.ENABLE_QUERY_REWRITE)


def extract_query_keywords(query: str) -> List[str]:
    """
    Extract just the important keywords for keyword matching.
    Used separately from rewriting for exact match scoring.

    Returns:
        List of important keywords to look for in memories
    """
    keywords = extract_keywords(query)

    # Also extract from original (before lowercasing) for proper nouns
    proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', query)
    keywords.update([n.lower() for n in proper_nouns if len(n) >= 3])

    return list(keywords)


if __name__ == "__main__":
    # Test cases
    test_queries = [
        "No more body image issue than determination lmao",
        "I'm tired from work today",
        "At the gym working out",
        "Whiskers is being crazy again haha",
        "Fixing the daemon project code",
    ]

    print("Query Rewriting Test Cases:")
    print("=" * 80)
    for q in test_queries:
        rewritten = rewrite_query(q, use_topic_extraction=False)
        print(f"\nOriginal:  {q}")
        print(f"Rewritten: {rewritten}")
        print(f"Keywords:  {extract_query_keywords(q)}")
