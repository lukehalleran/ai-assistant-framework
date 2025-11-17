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
"""

import re
from typing import Set, List
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Domain-specific synonym expansion
SYNONYM_GROUPS = {
    'gym': ['gym', 'workout', 'exercise', 'fitness', 'training', 'lift', 'bench', 'squat', 'deadlift'],
    'work': ['work', 'job', 'shift', 'brewery', 'employment'],
    'tired': ['tired', 'exhausted', 'beat', 'fatigued', 'drained', 'worn out'],
    'health': ['health', 'medication', 'med', 'amantadine', 'covid', 'sick', 'illness'],
    'body': ['body', 'physique', 'shape', 'fit', 'fitness', 'weight', 'muscle'],
    'code': ['code', 'coding', 'programming', 'debug', 'refactor', 'project', 'daemon'],
    'cat': ['cat', 'flapjack', 'poppy', 'pumpernickel', 'pickle', 'pet'],
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

def rewrite_query(query: str, use_topic_extraction: bool = True) -> str:
    """
    Rewrite a casual query into a semantic-rich version for better retrieval.

    Args:
        query: Original user query (e.g., "No more body image issue than determination lmao")
        use_topic_extraction: Whether to use topic_manager for entity extraction

    Returns:
        Expanded query string with additional keywords

    Example:
        >>> rewrite_query("No more body image issue than determination lmao")
        "body image gym workout exercise fitness determination motivation"
    """
    if not query or len(query.strip()) < 3:
        return query

    logger.debug(f"[QueryRewriter] Original query: '{query}'")

    # 1. Extract base keywords
    keywords = extract_keywords(query)

    # 2. Try topic extraction if available
    if use_topic_extraction:
        try:
            from utils.topic_manager import TopicManager
            topic_mgr = TopicManager()
            topics = topic_mgr.extract_topics(query)

            if topics:
                # Add extracted topics
                for topic in topics[:5]:  # Limit to top 5
                    topic_words = extract_keywords(topic)
                    keywords.update(topic_words)
                logger.debug(f"[QueryRewriter] Extracted topics: {topics[:5]}")
        except Exception as e:
            logger.debug(f"[QueryRewriter] Topic extraction failed: {e}")

    # 3. Expand with synonyms
    expanded = expand_with_synonyms(keywords)

    # 4. Build rewritten query
    # Keep original keywords first, then add expansions
    original_keywords = list(keywords)
    additional_keywords = list(expanded - keywords)

    # Combine: original + additional (limit total to avoid bloat)
    all_keywords = original_keywords + additional_keywords[:10]
    rewritten = ' '.join(all_keywords)

    logger.info(f"[QueryRewriter] Rewritten: '{query}' → '{rewritten}'")
    logger.debug(f"[QueryRewriter] Keyword count: {len(original_keywords)} original + {len(additional_keywords[:10])} expanded")

    return rewritten

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
        "Flapjack is being crazy again haha",
        "Fixing the daemon project code",
    ]

    print("Query Rewriting Test Cases:")
    print("=" * 80)
    for q in test_queries:
        rewritten = rewrite_query(q, use_topic_extraction=False)
        print(f"\nOriginal:  {q}")
        print(f"Rewritten: {rewritten}")
        print(f"Keywords:  {extract_query_keywords(q)}")
