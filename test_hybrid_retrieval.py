#!/usr/bin/env python3
"""
Test hybrid retrieval system with casual gym/health queries.
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.hybrid_retriever import HybridRetriever
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from config.app_config import CHROMA_PATH

async def test_hybrid_retrieval():
    """Test hybrid retrieval with the problematic casual query."""

    print("="*80)
    print("TESTING HYBRID RETRIEVAL SYSTEM")
    print("="*80)
    print(f"\nUsing ChromaDB at: {CHROMA_PATH}\n")

    # Initialize hybrid retriever
    chroma = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    hybrid = HybridRetriever(chroma_store=chroma)

    # Test queries - including the problematic casual one
    test_queries = [
        # Original problematic query
        "No more body image issue than determination lmao",

        # More direct gym queries
        "gym workout exercise fitness",
        "I'm at the gym today",
        "working out bench press",
        "amantadine medication health",

        # Casual variations
        "feeling good about gym progress",
        "body positivity at the gym",
        "tired after workout"
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: '{query}'")
        print(f"{'='*80}")

        try:
            # Get hybrid results
            results = await hybrid.retrieve(query, limit=10)

            if results:
                print(f"\nTop 10 hybrid results:")
                gym_relevant = 0
                health_relevant = 0

                for i, result in enumerate(results[:10], 1):
                    content = result.get('content', '')[:100]
                    query_text = result.get('query', '')[:80]
                    response_text = result.get('response', '')[:80]

                    # Extract scoring details
                    scoring = result.get('scoring', {})
                    semantic = scoring.get('semantic', 0)
                    keyword = scoring.get('keyword', 0)
                    hybrid_score = scoring.get('hybrid', 0)

                    # Build display text
                    if query_text and response_text:
                        display = f"Q: {query_text}"
                    elif content:
                        display = content
                    else:
                        display = str(result)[:100]

                    # Check relevance
                    combined_text = f"{query_text} {response_text} {content}".lower()
                    is_gym = any(word in combined_text for word in ['gym', 'workout', 'exercise', 'bench', 'fit'])
                    is_health = any(word in combined_text for word in ['amantadine', 'health', 'tired', 'body'])

                    if is_gym:
                        gym_relevant += 1
                    if is_health:
                        health_relevant += 1

                    relevance_marker = ""
                    if is_gym or is_health:
                        relevance_marker = " ✓"

                    timestamp = result.get('metadata', {}).get('timestamp', 'unknown')
                    collection = result.get('collection', 'unknown')

                    print(f"\n{i}. [Hybrid: {hybrid_score:.3f} | S: {semantic:.3f} | K: {keyword:.3f}] {timestamp} [{collection}]{relevance_marker}")
                    print(f"   {display}...")

                    if relevance_marker:
                        print(f"   ✓ RELEVANT: {'Gym' if is_gym else ''}{'/' if is_gym and is_health else ''}{'Health' if is_health else ''}")

                print(f"\nRelevance Summary:")
                print(f"  Gym-related: {gym_relevant}/10")
                print(f"  Health-related: {health_relevant}/10")
                print(f"  Total relevant: {gym_relevant + health_relevant}/10 ({(gym_relevant + health_relevant)*10}%)")

            else:
                print("No results returned!")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Test query rewriting specifically
    print(f"\n{'='*80}")
    print("QUERY REWRITING DEMONSTRATION")
    print(f"{'='*80}")

    from utils.query_rewriter import rewrite_query, extract_keywords

    casual_query = "No more body image issue than determination lmao"
    expanded = rewrite_query(casual_query)

    print(f"\nOriginal: '{casual_query}'")
    print(f"Expanded: '{expanded}'")
    print(f"\nKeywords extracted:")
    keywords = extract_keywords(casual_query)
    for kw in keywords:
        print(f"  - {kw}")

if __name__ == "__main__":
    asyncio.run(test_hybrid_retrieval())