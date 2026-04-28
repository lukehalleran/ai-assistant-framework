#!/usr/bin/env python3
"""
Test if our logging fix reveals the actual errors happening in get_graph_context.
"""

import sys
import os
import asyncio
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging to see warnings
logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s')

from memory.memory_coordinator import MemoryCoordinator
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from core.prompt.context_gatherer import ContextGatherer


async def test_graph_context_with_logging_fix():
    """Test get_graph_context with proper and improper initialization to see warnings."""

    print("=== TESTING LOGGING FIX ===\n")

    # Test 1: Proper initialization (should work)
    print("Test 1: Proper setup (should work)")
    print("-" * 40)

    try:
        store = MultiCollectionChromaStore()
        mc = MemoryCoordinator(corpus_manager=None, chroma_store=store)

        # Create a mock ContextGatherer that bypasses the model_manager/token_manager requirements
        # by directly calling the get_graph_context method
        class TestContextGatherer:
            def __init__(self, memory_coordinator):
                self.memory_coordinator = memory_coordinator

        gatherer = TestContextGatherer(mc)

        # Import the actual method and bind it to our test instance
        from core.prompt.context_gatherer import ContextGatherer
        actual_method = ContextGatherer.get_graph_context

        # Call the method directly
        result = await actual_method(gatherer, "how is pickle doing", max_sentences=5)

        print(f"✅ Result: {len(result)} sentences")
        if result:
            print(f"First sentence: {result[0]}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    print()

    # Test 2: Force an exception to see if our logging fix works
    print("Test 2: Simulated failure (should show WARNING)")
    print("-" * 40)

    try:
        # Create a broken gatherer that will cause exceptions
        class BrokenGatherer:
            def __init__(self):
                self.memory_coordinator = None  # This will cause issues

        broken_gatherer = BrokenGatherer()

        # This should trigger the exception handler and show our warning
        result = await actual_method(broken_gatherer, "test query", max_sentences=5)

        print(f"Result from broken gatherer: {result}")

    except Exception as e:
        print(f"Outer exception caught: {e}")

    print("\nIf you see a WARNING message above, our logging fix is working!")
    print("If no WARNING appears, the exception handler might not be getting hit.")


if __name__ == "__main__":
    asyncio.run(test_graph_context_with_logging_fix())