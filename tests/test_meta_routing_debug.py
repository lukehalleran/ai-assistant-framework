#!/usr/bin/env python3
"""
Debug script to verify meta-conversational routing in memory_coordinator.
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.memory_coordinator import MemoryCoordinator
from memory.corpus_manager import CorpusManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from utils.query_checker import is_meta_conversational, analyze_query_async

async def test_routing():
    print(f"\n{'='*70}")
    print("Meta-Conversational Routing Debug")
    print(f"{'='*70}\n")

    # Test query
    query = "Do you recall what time I said I woke up yesterday"

    # 1. Test detection
    print(f"1. Testing detection...")
    is_meta = is_meta_conversational(query)
    analysis = await analyze_query_async(query)
    print(f"   is_meta_conversational: {is_meta}")
    print(f"   analysis.is_meta_conversational: {analysis.is_meta_conversational}")

    if not is_meta:
        print("\n❌ FAIL: Meta-conversational detection not working!")
        return

    print("   ✅ Detection working\n")

    # 2. Test memory retrieval
    print(f"2. Initializing memory coordinator...")
    corpus = CorpusManager()
    chroma = MultiCollectionChromaStore()
    coordinator = MemoryCoordinator(corpus_manager=corpus, chroma_store=chroma)

    print(f"   Corpus has {len(corpus.corpus)} entries\n")

    # 3. Retrieve memories with the meta-conversational query
    print(f"3. Retrieving memories for query...")
    memories = await coordinator.get_memories(query=query, limit=10)

    print(f"   Retrieved {len(memories)} memories\n")

    # 4. Check if the "woken up at noon" memory is in the results
    print(f"4. Checking for target memory ('woken up at noon')...")
    found = False
    for i, mem in enumerate(memories):
        content = mem.get('query', '') or mem.get('content', '')
        if 'woken up at noon' in content.lower() or 'noon' in content.lower():
            print(f"\n   ✅ FOUND at position {i}:")
            print(f"      {content[:200]}...")
            found = True
            break

    if not found:
        print(f"\n   ❌ NOT FOUND in top {len(memories)} memories")
        print(f"\n   First 3 memories returned:")
        for i, mem in enumerate(memories[:3]):
            content = mem.get('query', '') or mem.get('content', '')
            print(f"\n   [{i}] {content[:150]}...")

    print(f"\n{'='*70}\n")

    return memories

if __name__ == "__main__":
    asyncio.run(test_routing())
