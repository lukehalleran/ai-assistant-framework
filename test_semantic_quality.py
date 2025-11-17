#!/usr/bin/env python3
"""Test ChromaDB semantic search quality for gym/health queries"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from config.app_config import CHROMA_PATH
import asyncio

async def test_gym_health_search():
    print("="*80)
    print("TESTING SEMANTIC SEARCH QUALITY - GYM/HEALTH QUERIES")
    print("="*80)
    print(f"\nUsing ChromaDB at: {CHROMA_PATH}\n")

    chroma = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)

    # Test queries about gym/health/exercise
    test_queries = [
        "gym workout exercise fitness",
        "I'm at the gym",
        "working out today",
        "amantadine medication health",
        "feeling tired after exercise",
        "body image determination gym"
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: '{query}'")
        print(f"{'='*80}")

        try:
            # Query conversations collection with MORE results
            results = chroma.query_collection(
                "conversations",
                query_text=query,
                n_results=20  # Get more to see if relevant ones are buried
            )

            if results:
                print(f"\nTop 20 results from ChromaDB:")
                for i, result in enumerate(results[:20], 1):
                    # Extract content
                    content = result.get('content', '')
                    query_text = result.get('query', '')
                    response_text = result.get('response', '')

                    # Build display text
                    if query_text and response_text:
                        display = f"Q: {query_text[:80]}"
                    elif content:
                        display = content[:100]
                    else:
                        display = str(result)[:100]

                    metadata = result.get('metadata', {})
                    timestamp = metadata.get('timestamp', 'unknown')
                    distance = result.get('distance', 'N/A')

                    print(f"\n{i}. [Distance: {distance}] {timestamp}")
                    print(f"   {display}...")

                    # Check if this is about gym/health
                    combined_text = f"{query_text} {response_text} {content}".lower()
                    is_relevant = any(word in combined_text for word in ['gym', 'workout', 'exercise', 'fit', 'amantadine', 'tired', 'body'])
                    if is_relevant:
                        print(f"   ✓ RELEVANT to gym/health!")
            else:
                print("No results returned!")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Also check if we can search for specific terms
    print(f"\n{'='*80}")
    print("DIRECT SEARCH FOR 'gym' keyword")
    print(f"{'='*80}")

    try:
        # Get collection
        collection = chroma.client.get_collection(
            name="conversations",
            embedding_function=chroma.embedding_fn
        )

        # Count total documents
        count = collection.count()
        print(f"\nTotal documents in conversations collection: {count}")

        # Try to query with 'where' filter if metadata supports it
        results = chroma.query_collection(
            "conversations",
            query_text="gym exercise workout",
            n_results=50
        )

        print(f"\nSearching through top 50 results for 'gym' keyword...")
        gym_count = 0
        for result in results:
            combined = str(result).lower()
            if 'gym' in combined:
                gym_count += 1
                print(f"\n✓ Found gym-related memory:")
                print(f"  Timestamp: {result.get('metadata', {}).get('timestamp', 'unknown')}")
                print(f"  Content: {result.get('query', '')[:100]}")

        print(f"\nFound {gym_count} gym-related memories in top 50 results")

    except Exception as e:
        print(f"Error in direct search: {e}")

if __name__ == "__main__":
    asyncio.run(test_gym_health_search())
