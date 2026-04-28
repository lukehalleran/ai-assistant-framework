#!/usr/bin/env python3
"""
Debug script to trace graph context through the full prompt building pipeline.
This will help identify where Pickle/poppy information gets lost.
"""

import asyncio
import sys
import os

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory.graph_memory import GraphMemory
from memory.entity_resolver import EntityResolver
from core.prompt.context_gatherer import ContextGatherer
from core.prompt.builder import UnifiedPromptBuilder
from memory.memory_coordinator import MemoryCoordinator
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from config import app_config as cfg


async def debug_graph_context_pipeline():
    """Debug the full graph context pipeline to see where info gets lost."""

    print("=== DEBUGGING GRAPH CONTEXT PIPELINE ===\n")

    # Test query
    test_query = "how is pickle doing"
    print(f"Test query: '{test_query}'\n")

    # Step 1: Test direct graph context gathering
    print("Step 1: Direct graph context gathering")
    print("-" * 40)

    try:
        # Initialize components like in real system
        store = MultiCollectionChromaStore()
        mc = MemoryCoordinator(None, store)  # Minimal setup for testing
        gatherer = ContextGatherer(mc)

        # Test graph context directly
        graph_context = await gatherer.get_graph_context(test_query, max_sentences=12)
        print(f"Graph context result: {graph_context}")
        print(f"Number of sentences: {len(graph_context)}")

        if graph_context:
            print("✅ Graph context generation WORKING")
            for i, sentence in enumerate(graph_context):
                print(f"  {i+1}. {sentence}")
        else:
            print("❌ Graph context generation FAILED")
            print("Checking why...")

            # Check individual components
            graph = getattr(mc, "graph_memory", None)
            resolver = getattr(mc, "entity_resolver", None)

            print(f"  - graph_memory exists: {graph is not None}")
            print(f"  - entity_resolver exists: {resolver is not None}")
            print(f"  - KNOWLEDGE_GRAPH_ENABLED: {cfg.KNOWLEDGE_GRAPH_ENABLED}")

            if graph:
                print(f"  - graph node count: {graph.node_count()}")

                # Test entity resolution manually
                if resolver:
                    pickle_resolved = resolver.resolve("pickle")
                    print(f"  - 'pickle' resolves to: '{pickle_resolved}'")

                    if pickle_resolved:
                        ctx_sentences = graph.get_context_sentences(pickle_resolved, depth=2, max_sentences=5)
                        print(f"  - context sentences for {pickle_resolved}: {ctx_sentences}")

        print()

    except Exception as e:
        print(f"❌ Error in graph context gathering: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Step 2: Test full prompt building pipeline
    print("Step 2: Full prompt building pipeline")
    print("-" * 40)

    try:
        # Initialize prompt builder (this is more complex)
        builder = UnifiedPromptBuilder(mc)

        # Build a full prompt like the real system would
        prompt_result = await builder.build_prompt(
            user_input=test_query,
            recent_conversations=[],
            system_prompt="You are a helpful assistant.",
            model_name="gpt-4",
        )

        print(f"Prompt built successfully: {prompt_result is not None}")

        if prompt_result:
            prompt_text = prompt_result.get("prompt", "")
            context = prompt_result.get("context", {})

            # Check if graph context made it into the final context
            final_graph_context = context.get("graph_context", [])
            print(f"Graph context in final result: {len(final_graph_context)} sentences")

            if final_graph_context:
                print("✅ Graph context made it through the pipeline")
                for i, sentence in enumerate(final_graph_context):
                    print(f"  {i+1}. {sentence}")
            else:
                print("❌ Graph context lost in pipeline")

            # Check if it's in the actual prompt text
            if "[KNOWLEDGE GRAPH]" in prompt_text:
                print("✅ Knowledge graph section found in prompt")
                # Extract the knowledge graph section
                lines = prompt_text.split('\n')
                in_kg_section = False
                kg_lines = []
                for line in lines:
                    if line.startswith('[KNOWLEDGE GRAPH]'):
                        in_kg_section = True
                        kg_lines.append(line)
                    elif line.startswith('[') and in_kg_section:
                        break
                    elif in_kg_section:
                        kg_lines.append(line)

                print("Knowledge graph section content:")
                for line in kg_lines[:10]:  # First 10 lines
                    print(f"  {line}")

            else:
                print("❌ No knowledge graph section in final prompt")

            # Check prompt length and budget
            prompt_length = len(prompt_text.split())
            token_budget = getattr(cfg, 'PROMPT_TOKEN_BUDGET_DEFAULT', 15000)
            print(f"Prompt length: ~{prompt_length} words (budget: {token_budget} tokens)")

        print()

    except Exception as e:
        print(f"❌ Error in prompt building: {e}")
        import traceback
        traceback.print_exc()
        print()

    print("=== DEBUG COMPLETE ===")


if __name__ == "__main__":
    asyncio.run(debug_graph_context_pipeline())