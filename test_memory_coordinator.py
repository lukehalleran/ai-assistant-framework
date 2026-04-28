#!/usr/bin/env python3
"""
Simple test to check if MemoryCoordinator properly initializes graph components.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory.memory_coordinator import MemoryCoordinator
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore


def test_memory_coordinator_graph_setup():
    """Test if MemoryCoordinator properly sets up graph components."""

    print("=== TESTING MEMORY COORDINATOR GRAPH SETUP ===\n")

    try:
        # Create store
        store = MultiCollectionChromaStore()
        print("✅ ChromaDB store created")

        # Create memory coordinator with minimal setup
        # Looking at the constructor, it might need more params
        print("Creating MemoryCoordinator...")

        # Let's check what the constructor actually needs
        import inspect
        sig = inspect.signature(MemoryCoordinator.__init__)
        print(f"MemoryCoordinator.__init__ signature: {sig}")

        # Try to create with None for missing deps
        mc = MemoryCoordinator(corpus_manager=None, chroma_store=store)
        print("✅ MemoryCoordinator created")

        # Check if graph components are properly initialized
        graph_memory = getattr(mc, "graph_memory", None)
        entity_resolver = getattr(mc, "entity_resolver", None)

        print(f"graph_memory exists: {graph_memory is not None}")
        print(f"entity_resolver exists: {entity_resolver is not None}")

        if graph_memory:
            print(f"graph_memory type: {type(graph_memory)}")
            print(f"graph node count: {graph_memory.node_count()}")

        if entity_resolver:
            print(f"entity_resolver type: {type(entity_resolver)}")
            # Test resolution
            pickle_resolved = entity_resolver.resolve("pickle")
            poppy_resolved = entity_resolver.resolve("poppy")
            print(f"'pickle' resolves to: '{pickle_resolved}'")
            print(f"'poppy' resolves to: '{poppy_resolved}'")

        if graph_memory and entity_resolver:
            print("✅ Both graph components are properly initialized")
        else:
            print("❌ Graph components missing - this is the issue!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_memory_coordinator_graph_setup()