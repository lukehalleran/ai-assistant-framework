#!/usr/bin/env python3
"""
Test script to verify memory retrieval fixes.
Tests:
1. Gating allows more memories through
2. Temporal diversity across full date range
3. Semantic relevance improves
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.memory_coordinator import MemoryCoordinator
from memory.corpus_manager import CorpusManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from processing.gate_system import MultiStageGateSystem
from models.model_manager import ModelManager
from utils.time_manager import TimeManager
from utils.logging_utils import get_logger
from config.app_config import CHROMA_PATH, CORPUS_FILE
import logging

# Set debug logging
logging.basicConfig(level=logging.INFO)  # Changed to INFO to reduce noise
logger = get_logger("test_memory_fixes")

print(f"Using CHROMA_PATH: {CHROMA_PATH}")
print(f"Using CORPUS_FILE: {CORPUS_FILE}")

async def test_memory_retrieval():
    """Test memory retrieval with the new fixes."""

    print("=" * 80)
    print("MEMORY RETRIEVAL FIX TEST")
    print("=" * 80)

    # Initialize components
    print("\n[1] Initializing components...")
    corpus_manager = CorpusManager(corpus_file=CORPUS_FILE)
    chroma_store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    gate_system = MultiStageGateSystem(None)  # Will create own embed model
    model_manager = ModelManager()
    time_manager = TimeManager()

    coordinator = MemoryCoordinator(
        corpus_manager=corpus_manager,
        chroma_store=chroma_store,
        gate_system=gate_system,
        model_manager=model_manager,
        time_manager=time_manager
    )

    # Test queries with different characteristics
    test_queries = [
        ("What is Python?", "factual"),
        ("Tell me about our previous conversations", "meta-conversational"),
        ("Do you recall what I said last month?", "historical"),
        ("Explain that again", "deictic")
    ]

    print("\n[2] Testing memory retrieval with various query types...\n")

    for query, query_type in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: '{query}'")
        print(f"Type: {query_type}")
        print(f"{'='*80}")

        try:
            # Retrieve memories
            memories = await coordinator.get_memories(query, limit=25)

            print(f"\n✓ Retrieved {len(memories)} memories")

            if not memories:
                print("⚠ WARNING: No memories retrieved!")
                continue

            # Analyze temporal distribution
            now = datetime.now()
            time_bins = {
                "<24h": 0,
                "1-7d": 0,
                "7-30d": 0,
                "30-90d": 0,
                "90d+": 0
            }

            scores = []
            for mem in memories:
                ts = mem.get('timestamp')
                if isinstance(ts, str):
                    try:
                        from dateutil import parser
                        ts = parser.parse(ts)
                    except:
                        ts = now

                age_days = (now - ts).total_seconds() / 86400

                if age_days < 1:
                    time_bins["<24h"] += 1
                elif age_days < 7:
                    time_bins["1-7d"] += 1
                elif age_days < 30:
                    time_bins["7-30d"] += 1
                elif age_days < 90:
                    time_bins["30-90d"] += 1
                else:
                    time_bins["90d+"] += 1

                score = mem.get('final_score', mem.get('relevance_score', 0))
                scores.append(score)

            # Print temporal distribution
            print(f"\nTemporal Distribution:")
            for bin_name, count in time_bins.items():
                pct = (count / len(memories)) * 100
                bar = "█" * int(pct / 5)
                print(f"  {bin_name:8} : {count:3} ({pct:5.1f}%) {bar}")

            # Print score distribution
            if scores:
                avg_score = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)
                print(f"\nScore Distribution:")
                print(f"  Min: {min_score:.3f}")
                print(f"  Avg: {avg_score:.3f}")
                print(f"  Max: {max_score:.3f}")

            # Show sample memories
            print(f"\nTop 5 Memories:")
            for i, mem in enumerate(memories[:5], 1):
                ts = mem.get('timestamp', 'unknown')
                score = mem.get('final_score', mem.get('relevance_score', 0))
                forced = " [FORCED]" if mem.get('forced_minimum') else ""

                # Get preview content
                content = mem.get('content', '')[:80] or mem.get('query', '')[:80]

                print(f"  {i}. score={score:.3f}{forced} ts={ts}")
                print(f"     {content}...")

        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")

async def test_gating_directly():
    """Test gating system directly to verify minimum enforcement."""
    print("\n" + "="*80)
    print("GATING SYSTEM DIRECT TEST")
    print("="*80)

    gate_system = MultiStageGateSystem(None)

    # Create dummy memories with various similarity levels
    test_memories = []
    for i in range(20):
        test_memories.append({
            "id": f"mem_{i}",
            "content": f"Test memory {i} with some content about Python programming",
            "timestamp": datetime.now() - timedelta(days=i*5),
            "metadata": {
                "truth_score": 0.6,
                "type": "test"
            }
        })

    query = "Python programming language"
    print(f"\nQuery: '{query}'")
    print(f"Test memories: {len(test_memories)}")

    try:
        filtered = await gate_system.filter_memories(query, test_memories)
        print(f"✓ Filtered memories: {len(filtered)}")

        forced_count = sum(1 for m in filtered if m.get('forced_minimum'))
        print(f"  - Forced minimum: {forced_count}")
        print(f"  - Naturally passed: {len(filtered) - forced_count}")

        if filtered:
            print(f"\nTop 3 filtered memories:")
            for i, mem in enumerate(filtered[:3], 1):
                score = mem.get('relevance_score', 0)
                cosine = mem.get('cosine_sim', 0)
                forced = " [FORCED]" if mem.get('forced_minimum') else ""
                print(f"  {i}. {mem['id']}: relevance={score:.3f}, cosine={cosine:.3f}{forced}")

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    print("\nStarting memory retrieval tests...\n")

    # Run tests
    asyncio.run(test_gating_directly())
    asyncio.run(test_memory_retrieval())

    print("\nAll tests complete!")
