"""
End-to-end test of temporal window detection with actual memory retrieval.

Tests that different temporal queries retrieve appropriate memory windows.
"""

import os
import sys
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.memory_coordinator import MemoryCoordinator
from memory.corpus_manager import CorpusManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from config.app_config import CORPUS_FILE, CHROMA_PATH

async def test_temporal_retrieval():
    print("=" * 80)
    print("END-TO-END TEMPORAL RETRIEVAL TEST")
    print("=" * 80)
    print()

    # Initialize memory coordinator with required dependencies
    corpus_manager = CorpusManager(corpus_file=CORPUS_FILE)
    chroma_store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    coordinator = MemoryCoordinator(corpus_manager=corpus_manager, chroma_store=chroma_store)
    corpus_size = len(corpus_manager.corpus)
    print(f"Corpus size: {corpus_size} entries")
    print()

    # Test queries with different temporal windows
    test_queries = [
        ("Do you recall what time I woke up yesterday?", "Short window (1-2 days)"),
        ("Do you recall what time I woke up on Nov 1st?", "Explicit date (30 days)"),
        ("What did we discuss last week?", "Medium window (7 days)"),
        ("Do you recall when my last day off was?", "No temporal marker (default)"),
    ]

    for query, description in test_queries:
        print("-" * 80)
        print(f"Query: {query}")
        print(f"Description: {description}")
        print()

        # Call the actual retrieval method
        memories = await coordinator.get_semantic_top_memories(query, limit=10)

        print(f"Retrieved {len(memories)} memories")

        # Show first 5 with timestamps
        for i, mem in enumerate(memories[:5]):
            ts = mem.get('timestamp', 'N/A')
            content = mem.get('query', mem.get('content', ''))[:60]
            print(f"  [{i}] {ts}")
            print(f"      {content}...")

        print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_temporal_retrieval())
