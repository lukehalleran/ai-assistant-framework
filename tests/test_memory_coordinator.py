# tests/test_memory_coordinator.py
import asyncio
import os
import tempfile
from memory.corpus_manager import CorpusManager
from memory.memory_coordinator import MemoryCoordinator
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

async def test_memory_coordinator():
    print("Testing Memory Coordinator...")

    # Use temp directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "test_corpus.json")
        chroma_path = os.path.join(tmpdir, "test_chroma_db")

        # Initialize components
        corpus_manager = CorpusManager(corpus_path)

        # Initialize ChromaDB store
        chroma_store = MultiCollectionChromaStore(persist_directory=chroma_path)

        # Create coordinator
        coordinator = MemoryCoordinator(
            corpus_manager=corpus_manager,
            chroma_store=chroma_store
        )

        # Test storing
        print("\n1. Testing storage...")
        await coordinator.store_interaction(
            "What's the weather?",
            "I don't have access to weather data.",
            ["weather", "query"]
        )

        # Test retrieval
        print("\n2. Testing retrieval...")
        limit = 5

        memories = await coordinator.get_memories(
            "Tell me about the weather",
            limit=limit
        )

        print(f"\nRetrieved {len(memories)} memories:")
        for i, mem in enumerate(memories):
            query = mem.get('query', '')
            response = mem.get('response', '')
            print(f"\n{i+1}. Source: {mem.get('metadata', {}).get('source', 'unknown')}")
            if query:
                print(f"   Q: {query[:50]}...")
            if response:
                print(f"   A: {response[:50]}...")
        print("\n✅ Memory coordinator test complete!")

if __name__ == "__main__":
    asyncio.run(test_memory_coordinator())
