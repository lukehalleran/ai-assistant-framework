#!/usr/bin/env python3
"""
Test if the graph_context async task is throwing exceptions.
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory.memory_coordinator import MemoryCoordinator
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from core.prompt.context_gatherer import ContextGatherer
from models.model_manager import ModelManager
from core.prompt.token_manager import TokenManager
from config import app_config


async def test_graph_context_async_task():
    """Test if the graph_context async task throws exceptions."""

    print("=== TESTING GRAPH CONTEXT ASYNC TASK ===\n")

    try:
        # Set up components like the real system
        store = MultiCollectionChromaStore()
        mc = MemoryCoordinator(corpus_manager=None, chroma_store=store)

        # Try to create the missing components
        print("Setting up ModelManager...")
        try:
            model_manager = ModelManager()
            print("✅ ModelManager created")
        except Exception as e:
            print(f"❌ ModelManager failed: {e}")
            # Create a mock model manager
            print("Creating mock ModelManager...")
            class MockModelManager:
                def get_active_model_name(self):
                    return "gpt-4"
            model_manager = MockModelManager()
            print("✅ Mock ModelManager created")

        print("Setting up TokenManager...")
        try:
            token_manager = TokenManager(model_manager)
            print("✅ TokenManager created")
        except Exception as e:
            print(f"❌ TokenManager failed: {e}")
            return

        print("Setting up ContextGatherer...")
        try:
            gatherer = ContextGatherer(mc, model_manager, token_manager)
            print("✅ ContextGatherer created")
        except Exception as e:
            print(f"❌ ContextGatherer failed: {e}")
            return

        # Test the graph context method directly (async)
        print("\nTesting direct async call...")
        test_query = "how is pickle doing"

        try:
            graph_context = await gatherer.get_graph_context(test_query, max_sentences=12)
            print(f"✅ Direct call succeeded: {len(graph_context)} sentences")

            if graph_context:
                print("Sentences:")
                for i, sentence in enumerate(graph_context[:3]):
                    print(f"  {i+1}. {sentence}")
            else:
                print("❌ No sentences returned")

        except Exception as e:
            print(f"❌ Direct call failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Test as an async task (like in the prompt builder)
        print("\nTesting as async task (like prompt builder)...")

        async def _timed_task(name, coro):
            """Same timed task wrapper as in prompt builder."""
            try:
                result = await coro
                return result
            except Exception as e:
                print(f"_timed_task {name} exception: {e}")
                raise

        try:
            task = asyncio.create_task(
                _timed_task("graph_context", gatherer.get_graph_context(test_query, app_config.PROMPT_MAX_GRAPH_SENTENCES))
            )

            result = await task
            print(f"✅ Async task succeeded: {len(result)} sentences")

            if result:
                print("Task result sentences:")
                for i, sentence in enumerate(result[:3]):
                    print(f"  {i+1}. {sentence}")
            else:
                print("❌ Task returned no sentences")

        except Exception as e:
            print(f"❌ Async task failed: {e}")
            import traceback
            traceback.print_exc()

        # Test with gather (like in prompt builder)
        print("\nTesting with asyncio.gather (like prompt builder)...")

        try:
            tasks = {
                "graph_context": asyncio.create_task(
                    _timed_task("graph_context", gatherer.get_graph_context(test_query, app_config.PROMPT_MAX_GRAPH_SENTENCES))
                )
            }

            results = await asyncio.gather(*tasks.values(), return_exceptions=True)

            # Map results back to names (like prompt builder)
            gathered = {}
            for i, (name, _) in enumerate(tasks.items()):
                result = results[i]
                if isinstance(result, Exception):
                    print(f"❌ Task {name} failed with exception: {result}")
                    gathered[name] = []
                else:
                    gathered[name] = result or []
                    print(f"✅ Task {name} succeeded: {len(gathered[name])} sentences")

            if gathered.get("graph_context"):
                print("Gathered result sentences:")
                for i, sentence in enumerate(gathered["graph_context"][:3]):
                    print(f"  {i+1}. {sentence}")

        except Exception as e:
            print(f"❌ Gather failed: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"❌ Setup error: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== TEST COMPLETE ===")


if __name__ == "__main__":
    asyncio.run(test_graph_context_async_task())