"""
Lightweight integration tests for MemoryCoordinator meta-conversational and facts paths.

Covers:
- store_interaction and recent retrieval
- meta-conversational path in get_memories (bypasses semantic)
- topic_filter pre-filtering
- get_recent_facts and get_facts fallbacks when empty
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from memory.corpus_manager import CorpusManager
from memory.memory_coordinator import MemoryCoordinator
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore


@pytest.fixture()
def temp_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield {
            "corpus_file": str(Path(tmpdir) / "corpus.json"),
            "chroma_path": str(Path(tmpdir) / "chroma_db"),
        }


@pytest.fixture()
def coordinator(temp_dirs):
    cm = CorpusManager(corpus_file=temp_dirs["corpus_file"])
    store = MultiCollectionChromaStore(persist_directory=temp_dirs["chroma_path"])
    return MemoryCoordinator(corpus_manager=cm, chroma_store=store)


@pytest.mark.asyncio
async def test_meta_conversational_path_returns_recent(coordinator: MemoryCoordinator):
    # Seed a few interactions under different topics
    await coordinator.store_interaction("We discussed vectors", "Yes, 2D vectors", tags=["topic:math"]) 
    await coordinator.store_interaction("We discussed colors", "Favorite is blue", tags=["topic:preferences"]) 

    # Meta-conversational queries should bypass semantic and preserve recency
    result = await coordinator.get_memories("do you recall our discussion?", limit=5)

    assert isinstance(result, list)
    # At least one recent memory surfaces
    assert len(result) >= 1


@pytest.mark.asyncio
async def test_topic_filter_applied_for_normal_queries(coordinator: MemoryCoordinator):
    # Seed interactions with and without topic tag
    await coordinator.store_interaction("Python typing", "Use type hints", tags=["topic:python"]) 
    await coordinator.store_interaction("Gardening", "Tomatoes need sun", tags=["topic:garden"]) 

    # Non-meta query with explicit topic filter should return only matching ones
    res_python = await coordinator.get_memories("Tell me about typing", limit=10, topic_filter="python")
    assert all("topic:python" in (m.get("tags") or []) or "topic:python" in str(m.get("metadata", {}).get("tags", "")) for m in res_python)

    res_garden = await coordinator.get_memories("plants", limit=10, topic_filter="garden")
    assert all("topic:garden" in (m.get("tags") or []) or "topic:garden" in str(m.get("metadata", {}).get("tags", "")) for m in res_garden)


@pytest.mark.asyncio
async def test_facts_fallbacks_when_empty(coordinator: MemoryCoordinator):
    # With no facts stored yet, recent_facts and facts should be empty-safe
    recent = await coordinator.get_recent_facts(limit=3)
    assert isinstance(recent, list)

    facts = await coordinator.get_facts(query="anything", limit=3)
    assert isinstance(facts, list)

