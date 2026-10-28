#!/usr/bin/env python3
"""
Test to verify the meta-conversational fix is working.
This tests the actual code path used by the prompt builder.
"""

import os
import asyncio
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.memory_coordinator import MemoryCoordinator
from memory.corpus_manager import CorpusManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
import pytest

@pytest.fixture
def chroma(tmp_path):
    """Store on a tmp dir, closed afterwards (never the cwd-relative data/chroma_multi)."""
    store = MultiCollectionChromaStore(str(tmp_path / "chroma"))
    yield store
    store.close()


def _run_standalone(coro_fn):
    """`python tests/<file>.py` entry: same tmp-dir store, closed afterwards."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        store = MultiCollectionChromaStore(d)
        try:
            return asyncio.run(coro_fn(store))
        finally:
            store.close()


async def test_fix(chroma):
    query = "Do you recall what time I said I woke up yesterday"

    print(f"\n{'='*70}")
    print("VERIFICATION: Meta-Conversational Fix")
    print(f"{'='*70}\n")
    print(f"Query: {query}\n")

    # Initialize
    print("Step 1: Initialize memory coordinator...")
    corpus = CorpusManager()
    coordinator = MemoryCoordinator(corpus_manager=corpus, chroma_store=chroma)
    print(f"  Corpus size: {len(corpus.corpus)} entries\n")

    # Call the ACTUAL method used by prompt builder
    print("Step 2: Call get_semantic_top_memories() (actual code path)...")
    print("  Watch for: '[MemoryCoordinator][Semantic] Detected meta-conversational query'\n")

    memories = await coordinator.get_semantic_top_memories(query, limit=10)

    print(f"\n  Retrieved {len(memories)} memories\n")

    # Check results
    print("Step 3: Check if target memory is in results...")
    found = False
    for i, mem in enumerate(memories):
        content = str(mem.get('query', '')) + str(mem.get('content', ''))
        if 'woken up at noon' in content.lower() or ('noon' in content.lower() and 'woke' in content.lower()):
            print(f"  ✅ TARGET FOUND at position {i}!")
            print(f"     Content preview: {content[:150]}...")
            found = True
            break

    if not found:
        print(f"  ❌ Target NOT found in results")
        print(f"\n  Top 3 results:")
        for i, mem in enumerate(memories[:3]):
            content = mem.get('query', '') or mem.get('content', '')
            print(f"    [{i}] {content[:120]}...")

    print(f"\n{'='*70}\n")

    if found:
        print("✅ SUCCESS: Fix is working! Meta-conversational routing triggered.")
    else:
        print("❌ FAILURE: Target memory still not retrieved.")

    return found

if __name__ == "__main__":
    success = _run_standalone(test_fix)
    sys.exit(0 if success else 1)
