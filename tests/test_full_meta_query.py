#!/usr/bin/env python3
"""
Full end-to-end test of meta-conversational query handling.
Tests the complete flow from query → detection → retrieval → response.
"""

import asyncio
import sys
from core.orchestrator import DaemonOrchestrator

async def test_full_flow():
    print(f"\n{'='*70}")
    print(f"Full End-to-End Test: Meta-Conversational Query")
    print(f"{'='*70}\n")

    # Initialize orchestrator
    print("Initializing orchestrator...")
    orchestrator = DaemonOrchestrator()

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
