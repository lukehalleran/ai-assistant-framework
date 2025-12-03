#!/usr/bin/env python3
"""
Test script for conversation thread tracking.
Run with: python test_thread_tracking.py
"""

import asyncio
import logging
from datetime import datetime
from memory.corpus_manager import CorpusManager
from memory.memory_coordinator import MemoryCoordinator
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from utils.logging_utils import get_logger

# Set up logging to see debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = get_logger("test_thread_tracking")


async def test_thread_tracking():
    """Test conversation thread detection and tracking."""

    logger.info("=" * 80)
    logger.info("STARTING THREAD TRACKING TEST")
    logger.info("=" * 80)

    # Create test corpus manager with temporary file
    import tempfile
    import os
    temp_dir = tempfile.mkdtemp()
    corpus_file = os.path.join(temp_dir, "test_corpus.json")

    corpus_manager = CorpusManager(corpus_file=corpus_file)

    # Create minimal chroma store
    chroma_path = os.path.join(temp_dir, "test_chroma")
    chroma_store = MultiCollectionChromaStore(persist_directory=chroma_path)

    # Create memory coordinator
    memory_coordinator = MemoryCoordinator(
        corpus_manager=corpus_manager,
        chroma_store=chroma_store,
        gate_system=None,
        topic_manager=None,
        model_manager=None,
        time_manager=None
    )

    # Set a topic
    memory_coordinator.current_topic = "ICE raids"

    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: First conversation (should create new thread)")
    logger.info("=" * 80)

    await memory_coordinator.store_interaction(
        query="Did you see the news about ICE raids in Oakland? They arrested 60 people.",
        response="I saw that. It's very concerning. How are you feeling about it?",
        tags=["conversation"]
    )

    # Check thread context
    thread_ctx = memory_coordinator.get_thread_context()
    logger.info(f"\n[TEST] Thread context after message 1: {thread_ctx}")
    assert thread_ctx is not None, "Thread context should exist"
    assert thread_ctx['thread_depth'] == 1, f"Expected depth 1, got {thread_ctx['thread_depth']}"
    thread_id_1 = thread_ctx['thread_id']

    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Continuing same topic (should continue thread)")
    logger.info("=" * 80)

    await asyncio.sleep(0.5)  # Small delay

    await memory_coordinator.store_interaction(
        query="I'm worried about my neighbor. He's undocumented and has kids in school.",
        response="That's a valid concern. Has he talked to a lawyer about his options?",
        tags=["conversation"]
    )

    thread_ctx = memory_coordinator.get_thread_context()
    logger.info(f"\n[TEST] Thread context after message 2: {thread_ctx}")
    assert thread_ctx['thread_id'] == thread_id_1, "Should be same thread"
    assert thread_ctx['thread_depth'] == 2, f"Expected depth 2, got {thread_ctx['thread_depth']}"

    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Continuing thread - depth 3 (should trigger different prompt)")
    logger.info("=" * 80)

    await asyncio.sleep(0.5)

    await memory_coordinator.store_interaction(
        query="Not yet. Do you know any resources for immigration legal help in Oakland?",
        response="Yes, there are several organizations like Centro Legal de la Raza...",
        tags=["conversation"]
    )

    thread_ctx = memory_coordinator.get_thread_context()
    logger.info(f"\n[TEST] Thread context after message 3: {thread_ctx}")
    assert thread_ctx['thread_id'] == thread_id_1, "Should still be same thread"
    assert thread_ctx['thread_depth'] == 3, f"Expected depth 3, got {thread_ctx['thread_depth']}"

    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Topic switch (should create new thread)")
    logger.info("=" * 80)

    memory_coordinator.current_topic = "weather"
    await asyncio.sleep(1.0)

    await memory_coordinator.store_interaction(
        query="Anyway, what's the weather like tomorrow?",
        response="It looks like it will be sunny with a high of 72°F.",
        tags=["conversation"]
    )

    thread_ctx = memory_coordinator.get_thread_context()
    logger.info(f"\n[TEST] Thread context after topic switch: {thread_ctx}")
    assert thread_ctx['thread_id'] != thread_id_1, "Should be NEW thread after topic switch"
    assert thread_ctx['thread_depth'] == 1, f"Expected depth 1 for new thread, got {thread_ctx['thread_depth']}"

    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Explicit thread break marker")
    logger.info("=" * 80)

    await asyncio.sleep(0.5)

    await memory_coordinator.store_interaction(
        query="Actually, changing topics - can you explain quantum computing?",
        response="Quantum computing uses quantum bits or qubits...",
        tags=["conversation"]
    )

    thread_ctx = memory_coordinator.get_thread_context()
    logger.info(f"\n[TEST] Thread context after explicit break: {thread_ctx}")
    assert thread_ctx['thread_depth'] == 1, f"Expected depth 1 after explicit break, got {thread_ctx['thread_depth']}"

    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: Check corpus entries for thread metadata")
    logger.info("=" * 80)

    recent = corpus_manager.get_recent_memories(count=5)
    logger.info(f"\n[TEST] Recent memories count: {len(recent)}")

    for i, entry in enumerate(recent, 1):
        logger.info(f"\nEntry {i}:")
        logger.info(f"  Query: {entry.get('query', '')[:60]}...")
        logger.info(f"  Thread ID: {entry.get('thread_id')}")
        logger.info(f"  Thread Depth: {entry.get('thread_depth')}")
        logger.info(f"  Thread Topic: {entry.get('thread_topic')}")
        logger.info(f"  Is Heavy: {entry.get('is_heavy_topic')}")

    logger.info("\n" + "=" * 80)
    logger.info("ALL TESTS PASSED!")
    logger.info("=" * 80)

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    asyncio.run(test_thread_tracking())
