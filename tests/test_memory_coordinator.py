# tests/test_memory_coordinator.py
import asyncio
from chromadb import PersistentClient
from memory.corpus_manager import CorpusManager
from memory.memory_coordinator import MemoryCoordinator

async def test_memory_coordinator():
    print("Testing Memory Coordinator...")

    # Initialize components
    corpus_manager = CorpusManager("test_corpus.json")

    # Initialize ChromaDB
    client = PersistentClient(path="test_chroma_db")
    collection = client.get_or_create_collection("test-memory")

    # Create coordinator
    coordinator = MemoryCoordinator(
        corpus_manager=corpus_manager,
        chroma_collection=collection
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
    config = {
        'recent_count': 3,
        'semantic_count': 10,
        'max_memories': 5
    }

    result = await coordinator.retrieve_relevant_memories(
        "Tell me about the weather",
        config
    )

    print(f"\nRetrieved {len(result['memories'])} memories:")
    for i, mem in enumerate(result['memories']):
        print(f"\n{i+1}. Source: {mem.get('source', 'unknown')}")
        print(f"   Q: {mem['query'][:50]}...")
        print(f"   A: {mem['response'][:50]}...")

    print(f"\nCounts: {result['counts']}")
    print("\n✅ Memory coordinator test complete!")

if __name__ == "__main__":
    asyncio.run(test_memory_coordinator())
