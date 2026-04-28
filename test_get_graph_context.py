#!/usr/bin/env python3
"""
Test the exact get_graph_context logic to see what's failing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory.memory_coordinator import MemoryCoordinator
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from config import app_config


def test_get_graph_context_logic():
    """Test the exact logic from get_graph_context method."""

    print("=== TESTING GET_GRAPH_CONTEXT LOGIC ===\n")

    # Test the configuration check
    print("Step 1: Configuration check")
    print("-" * 30)
    print(f"KNOWLEDGE_GRAPH_ENABLED: {app_config.KNOWLEDGE_GRAPH_ENABLED}")
    print(f"KNOWLEDGE_GRAPH_RETRIEVAL_DEPTH: {app_config.KNOWLEDGE_GRAPH_RETRIEVAL_DEPTH}")

    if not app_config.KNOWLEDGE_GRAPH_ENABLED:
        print("❌ Knowledge graph is DISABLED in config!")
        return
    else:
        print("✅ Knowledge graph is enabled")

    # Test memory coordinator setup
    print("\nStep 2: MemoryCoordinator setup")
    print("-" * 30)

    try:
        store = MultiCollectionChromaStore()
        mc = MemoryCoordinator(corpus_manager=None, chroma_store=store)

        graph = getattr(mc, "graph_memory", None)
        resolver = getattr(mc, "entity_resolver", None)

        print(f"graph exists: {graph is not None}")
        print(f"resolver exists: {resolver is not None}")

        if graph:
            node_count = graph.node_count()
            print(f"graph.node_count(): {node_count}")

            if node_count == 0:
                print("❌ Graph has no nodes!")
                return
            else:
                print("✅ Graph has nodes")

        if not graph or not resolver or graph.node_count() == 0:
            print("❌ Early return conditions met - this is why get_graph_context returns []")
            return
        else:
            print("✅ All components ready")

    except Exception as e:
        print(f"❌ Error setting up components: {e}")
        return

    # Test the actual query processing
    print("\nStep 3: Query processing")
    print("-" * 30)

    test_query = "how is pickle doing"
    print(f"Test query: '{test_query}'")

    # Extract entity mentions (exact logic from get_graph_context)
    words = test_query.lower().split()
    candidates = []

    # Check bigrams and trigrams first
    for n in (3, 2):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            candidates.append(phrase)

    # Then single words (skip stopwords)
    _STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "do", "does",
                  "did", "have", "has", "had", "what", "who", "where", "when",
                  "how", "why", "about", "with", "from", "for", "and", "or",
                  "but", "not", "to", "in", "on", "at", "of", "my", "your",
                  "i", "me", "you", "we", "they", "it", "this", "that", "can",
                  "will", "would", "should", "could", "tell", "know", "think"}

    for w in words:
        if w not in _STOPWORDS and len(w) > 2:
            candidates.append(w)

    print(f"Candidates: {candidates}")

    # Process each candidate
    sentences = []
    seen_entities = set()
    max_sentences = 12

    for mention in candidates:
        eid = resolver.resolve(mention)
        if eid and eid not in seen_entities:
            seen_entities.add(eid)
            print(f"Found entity: '{mention}' -> '{eid}'")

            # Get context sentences
            ctx = graph.get_context_sentences(
                eid,
                depth=app_config.KNOWLEDGE_GRAPH_RETRIEVAL_DEPTH,
                max_sentences=max_sentences - len(sentences)
            )
            print(f"  Context sentences: {len(ctx)}")

            if ctx:
                print(f"  First few: {ctx[:3]}")
                sentences.extend(ctx)
                if len(sentences) >= max_sentences:
                    break

    print(f"\nFinal result: {len(sentences)} sentences")
    if sentences:
        print("✅ get_graph_context logic would return sentences")
        for i, s in enumerate(sentences[:5]):
            print(f"  {i+1}. {s}")
    else:
        print("❌ get_graph_context logic would return empty list")

    print("\n=== TEST COMPLETE ===")


if __name__ == "__main__":
    test_get_graph_context_logic()