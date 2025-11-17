"""
utils/keyword_matcher.py

Keyword-based relevance scoring for hybrid retrieval.
Boosts memories that contain exact keyword matches.

Module Contract:
- Purpose: Add keyword matching scores to complement semantic search
- Inputs: Query keywords, memory content
- Outputs: Keyword match score (0.0-1.0)
- Dependencies: None (pure keyword matching)
- Side effects: None
"""

import re
from typing import List, Dict, Any
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def calculate_keyword_score(keywords: List[str], memory: Dict[str, Any]) -> float:
    """
    Calculate keyword match score for a memory.

    Args:
        keywords: List of important query keywords
        memory: Memory dictionary with content/query/response fields

    Returns:
        Score from 0.0 (no matches) to 1.0 (all keywords match)
    """
    if not keywords:
        return 0.0

    # Extract all text from memory
    content_parts = []

    # Try different content fields
    if memory.get('content'):
        content_parts.append(str(memory['content']))
    if memory.get('query'):
        content_parts.append(str(memory['query']))
    if memory.get('response'):
        content_parts.append(str(memory['response']))

    # Combine and lowercase
    combined_text = ' '.join(content_parts).lower()

    if not combined_text:
        return 0.0

    # Count keyword matches
    matches = 0
    for keyword in keywords:
        # Use word boundary regex for exact word matching
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        if re.search(pattern, combined_text):
            matches += 1

    # Calculate score as percentage of keywords found
    score = matches / len(keywords)

    return score


def apply_keyword_boost(
    memories: List[Dict[str, Any]],
    keywords: List[str],
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Apply keyword boosting to memories with semantic scores.

    Args:
        memories: List of memories with 'relevance_score' or 'final_score'
        keywords: Query keywords for matching
        semantic_weight: Weight for semantic score (default 0.7)
        keyword_weight: Weight for keyword score (default 0.3)

    Returns:
        Memories with updated 'hybrid_score' field
    """
    logger.debug(f"[KeywordMatcher] Applying keyword boost with keywords: {keywords}")

    for mem in memories:
        # Get base semantic score
        semantic_score = mem.get('relevance_score', mem.get('final_score', 0.0))

        # Calculate keyword score
        keyword_score = calculate_keyword_score(keywords, mem)

        # Blend scores
        hybrid_score = semantic_weight * semantic_score + keyword_weight * keyword_score

        # Store all scores
        mem['hybrid_score'] = hybrid_score
        mem['keyword_score'] = keyword_score
        mem['semantic_score'] = semantic_score

        # Log significant keyword boosts
        if keyword_score > 0.3:
            logger.debug(
                f"[KeywordMatcher] Boosted memory: "
                f"semantic={semantic_score:.3f}, keyword={keyword_score:.3f}, "
                f"hybrid={hybrid_score:.3f}"
            )

    # Sort by hybrid score
    memories.sort(key=lambda m: m.get('hybrid_score', 0), reverse=True)

    logger.info(
        f"[KeywordMatcher] Applied keyword boost to {len(memories)} memories, "
        f"avg keyword score: {sum(m.get('keyword_score', 0) for m in memories) / len(memories) if memories else 0:.3f}"
    )

    return memories


if __name__ == "__main__":
    # Test cases
    test_memory_gym = {
        'query': 'I am at the gym working out',
        'response': 'Great job staying consistent with your fitness routine!'
    }

    test_memory_code = {
        'query': 'Fixed the daemon project bug',
        'response': 'Awesome work debugging that issue!'
    }

    keywords_gym = ['gym', 'workout', 'exercise']
    keywords_code = ['code', 'debug', 'project']

    print("Keyword Matching Test Cases:")
    print("=" * 80)

    print(f"\nTest 1: Gym memory with gym keywords")
    score = calculate_keyword_score(keywords_gym, test_memory_gym)
    print(f"Keywords: {keywords_gym}")
    print(f"Score: {score:.3f}")

    print(f"\nTest 2: Gym memory with code keywords (mismatch)")
    score = calculate_keyword_score(keywords_code, test_memory_gym)
    print(f"Keywords: {keywords_code}")
    print(f"Score: {score:.3f}")

    print(f"\nTest 3: Code memory with code keywords")
    score = calculate_keyword_score(keywords_code, test_memory_code)
    print(f"Keywords: {keywords_code}")
    print(f"Score: {score:.3f}")
