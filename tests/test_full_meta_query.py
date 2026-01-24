#!/usr/bin/env python3
"""
Full end-to-end test of meta-conversational query handling.
Tests the complete flow from query → detection → retrieval → response.

NOTE: This is a manual integration test that requires full system initialization.
Run with: python tests/test_full_meta_query.py
"""

import asyncio
import sys
import pytest

# Skip when run via pytest - this is a manual integration script
pytestmark = pytest.mark.skip(reason="Manual integration test - run directly with python")

async def test_full_flow():
    # Import here to avoid import errors during pytest collection
    from core.orchestrator import DaemonOrchestrator
    from models.model_manager import ModelManager
    from core.response_generator import ResponseGenerator
    from utils.file_processor import FileProcessor
    from core.prompt.builder import UnifiedPromptBuilder

    print(f"\n{'='*70}")
    print(f"Full End-to-End Test: Meta-Conversational Query")
    print(f"{'='*70}\n")

    # Initialize components
    print("Initializing orchestrator...")
    model_manager = ModelManager()
    response_generator = ResponseGenerator(model_manager=model_manager)
    file_processor = FileProcessor()

    # Create minimal prompt builder
    from core.prompt.token_manager import TokenManager
    token_manager = TokenManager()
    prompt_builder = UnifiedPromptBuilder(
        memory_coordinator=None,
        model_manager=model_manager,
        token_manager=token_manager
    )

    orchestrator = DaemonOrchestrator(
        model_manager=model_manager,
        response_generator=response_generator,
        file_processor=file_processor,
        prompt_builder=prompt_builder
    )

    # Test query
    query = "Do you recall what time I said I woke up yesterday?"
    print(f"\nTest Query: {query}\n")
    print(f"{'='*70}\n")

    print("Processing query...\n")
    print("Watch for these log entries:")
    print("  1. [TONE] detection (should be 'conversational')")
    print("  2. Meta-conversational routing in memory_coordinator")
    print("  3. Retrieval of recent episodic memories (should be 15+)")
    print("  4. Semantic search being SKIPPED")
    print(f"\n{'='*70}\n")

    # Process the query
    response_parts = []
    async for chunk in orchestrator.process_message(query):
        response_parts.append(chunk)
        # Print chunks as they arrive (streaming)
        print(chunk, end='', flush=True)

    response = ''.join(response_parts)

    print(f"\n\n{'='*70}")
    print("Test Complete")
    print(f"{'='*70}\n")
    print(f"Response length: {len(response)} chars")
    print(f"Response preview: {response[:200]}...")

    return response

if __name__ == "__main__":
    try:
        asyncio.run(test_full_flow())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
