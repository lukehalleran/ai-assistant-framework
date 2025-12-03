#!/usr/bin/env python3
"""
Detailed diagnostic for meta-conversational query flow.
Shows exactly what's happening at each step.
"""

import os
import asyncio
import sys
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from memory.memory_coordinator import MemoryCoordinator
from memory.corpus_manager import CorpusManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from utils.query_checker import is_meta_conversational

async def test():
    query = "Do you recall what time I said I woke up yesterday"

    print(f"\n{'='*70}")
    print("DETAILED META-CONVERSATIONAL DIAGNOSTIC")
    print(f"{'='*70}\n")
    print(f"Query: {query}\n")

    # Step 1: Detection
    print("STEP 1: Meta-conversational detection")
    is_meta = is_meta_conversational(query)
    print(f"  Result: {is_meta}")

    if not is_meta:
        print("  ❌ FAILED - detection not working!")
        return
    print("  ✅ PASSED\n")

    # Step 2: Initialize coordinator
    print("STEP 2: Initialize memory coordinator")
    corpus = CorpusManager()
    chroma = MultiCollectionChromaStore()
    coordinator = MemoryCoordinator(corpus_manager=corpus, chroma_store=chroma)
    print(f"  Corpus size: {len(corpus.corpus)} entries")
    print(f"  ✅ PASSED\n")

    # Step 3: Check for target memory in corpus
    print("STEP 3: Verify target memory exists in corpus")
    target_found = False
    target_entry = None
    for entry in corpus.corpus:
        if 'woken up at noon' in str(entry.get('query', '')).lower():
            target_found = True
            target_entry = entry
            print(f"  ✅ Target found!")
            print(f"     Timestamp: {entry.get('timestamp')}")
            print(f"     Topic: {entry.get('topic')}")
            print(f"     Query preview: {entry.get('query', '')[:100]}...")
            break

    if not target_found:
        print(f"  ❌ Target memory NOT in corpus!")
        return
    print()

    # Step 4: Call get_memories and watch logs
    print("STEP 4: Call coordinator.get_memories()")
    print("  Watch for '[MemoryCoordinator] Detected meta-conversational query' in logs...\n")

    memories = await coordinator.get_memories(query=query, limit=10)

    print(f"\n  Returned {len(memories)} memories\n")

    # Step 5: Check results
    print("STEP 5: Check if target memory is in results")
    found_in_results = False
    for i, mem in enumerate(memories):
        content = str(mem.get('query', '')) + str(mem.get('content', ''))
        if 'woken up at noon' in content.lower() or 'noon' in content.lower():
            print(f"  ✅ Target FOUND at position {i}!")
            print(f"     Score: {mem.get('final_score', mem.get('relevance_score', 'N/A'))}")
            found_in_results = True
            break

    if not found_in_results:
        print(f"  ❌ Target NOT in top {len(memories)} results")
        print(f"\n  Top 5 results:")
        for i, mem in enumerate(memories[:5]):
            content = mem.get('query', '') or mem.get('content', '')
            score = mem.get('final_score', mem.get('relevance_score', 0))
            ts = mem.get('timestamp', 'N/A')
            print(f"    [{i}] score={score:.3f} ts={ts}")
            print(f"        {content[:120]}...")

    print(f"\n{'='*70}\n")

    return memories

if __name__ == "__main__":
    asyncio.run(test())
