#!/usr/bin/env python3
"""
Detailed test to see exactly what memories are being retrieved.
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.memory_coordinator import MemoryCoordinator
from memory.corpus_manager import CorpusManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

async def test():
    query = "Do you recall what time I woke up on Nov 1st"

    print(f"\n{'='*70}")
    print("DETAILED RETRIEVAL TEST")
    print(f"{'='*70}\n")

    corpus = CorpusManager()
    chroma = MultiCollectionChromaStore()
    coordinator = MemoryCoordinator(corpus_manager=corpus, chroma_store=chroma)

    print(f"Query: {query}\n")
    print(f"Total corpus entries: {len(corpus.corpus)}\n")

    # Call get_semantic_top_memories (the actual path)
    print("Calling get_semantic_top_memories()...\n")
    memories = await coordinator.get_semantic_top_memories(query, limit=10)

    print(f"Retrieved {len(memories)} memories\n")
    print("="*70)

    # Show all retrieved memories
    for i, mem in enumerate(memories):
        query_text = mem.get('query', '')[:100]
        timestamp = mem.get('timestamp', 'N/A')
        print(f"\n[{i}] Timestamp: {timestamp}")
        print(f"    Query: {query_text}...")

        # Check if this is the target
        full_content = str(mem.get('query', '')) + str(mem.get('response', ''))
        if 'woken up at noon' in full_content.lower():
            print(f"    *** TARGET FOUND! ***")

    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    asyncio.run(test())
