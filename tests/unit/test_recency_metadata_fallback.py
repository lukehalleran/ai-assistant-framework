"""
Regression tests for the dead-recency bug (2026-07-08).

The hybrid/semantic retrieval path (`MemoryRetriever._get_semantic_memories`)
built memory dicts with no top-level 'timestamp'. `MemoryScorer.rank_memories`
read only the top-level key and silently defaulted a missing timestamp to
`now`, so every memory from the main retrieval path scored recency=1.0 —
months-old memories ranked as fresh (observed live: a 3-month-old "Week 11
videos" conversation ranked #1 on a current-coursework query).

Fix is two-sided:
  - scorer: falls back to metadata['timestamp'] before defaulting to now
  - retriever: hybrid path now surfaces metadata timestamp at top level

These tests call the deployed functions directly (no re-derivation).
"""

import logging
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from memory.memory_scorer import MemoryScorer
from memory.memory_retriever import MemoryRetriever

# Debug dict (which exposes per-memory recency) is only populated at DEBUG
logging.getLogger("memory_scorer").setLevel(logging.DEBUG)


def _mem(content: str, **extra) -> dict:
    m = {
        "id": "x",
        "content": content,
        "query": "",
        "response": "",
        "metadata": {},
        "collection": "conversations",
        "relevance_score": 0.5,
    }
    m.update(extra)
    return m


def _iso(days_ago: float) -> str:
    return (datetime.now() - timedelta(days=days_ago)).isoformat()


def test_metadata_timestamp_fallback_gives_old_memory_low_recency():
    """A memory whose timestamp lives only in metadata (the hybrid-path shape)
    must NOT score recency=1.0 when it is months old."""
    scorer = MemoryScorer()
    old = _mem("watch the videos for week 11", metadata={"timestamp": _iso(90)})
    fresh = _mem("call the company about the login", metadata={"timestamp": _iso(0.04)})

    ranked = scorer.rank_memories([old, fresh], current_query="what videos do I have to watch")

    by_content = {m["content"]: m for m in ranked}
    old_rec = by_content["watch the videos for week 11"]["debug"]["recency"]
    fresh_rec = by_content["call the company about the login"]["debug"]["recency"]

    assert old_rec < 0.5, f"90-day-old memory scored recency={old_rec} (dead-recency bug)"
    assert fresh_rec > 0.8
    assert fresh_rec > old_rec


def test_top_level_timestamp_still_takes_precedence():
    """An explicit top-level timestamp wins over a metadata one."""
    scorer = MemoryScorer()
    m = _mem(
        "some memory",
        timestamp=_iso(0.02),
        metadata={"timestamp": _iso(120)},
    )
    ranked = scorer.rank_memories([m, _mem("other")], current_query="anything")
    rec = next(x for x in ranked if x["content"] == "some memory")["debug"]["recency"]
    assert rec > 0.8


def test_missing_timestamp_everywhere_defaults_to_now():
    """No timestamp anywhere → current (documented) behavior: treated as fresh."""
    scorer = MemoryScorer()
    ranked = scorer.rank_memories([_mem("no ts"), _mem("other")], current_query="anything")
    rec = next(x for x in ranked if x["content"] == "no ts")["debug"]["recency"]
    assert rec > 0.8


def test_tz_aware_timestamp_does_not_raise():
    """A tz-aware ISO string must be normalized, not crash the naive-now math."""
    scorer = MemoryScorer()
    m = _mem("tz aware", metadata={"timestamp": "2026-04-05T12:39:00-05:00"})
    ranked = scorer.rank_memories([m], current_query="anything")
    assert ranked[0]["debug"]["recency"] < 0.5


@pytest.mark.asyncio
async def test_hybrid_path_surfaces_metadata_timestamp():
    """_get_semantic_memories (the deployed hybrid path) must emit a top-level
    timestamp taken from the Chroma metadata."""
    ts = _iso(90)
    hybrid = Mock()
    hybrid.retrieve = AsyncMock(return_value=[{
        "id": "abc",
        "content": "watch the videos for week 11",
        "metadata": {"timestamp": ts},
        "collection": "conversations",
        "hybrid_score": 0.9,
        "keyword_score": 0.7,
        "semantic_score": 0.8,
    }])
    retriever = MemoryRetriever(
        corpus_manager=Mock(),
        chroma_store=Mock(),
        hybrid_retriever=hybrid,
    )

    memories = await retriever._get_semantic_memories("what videos do I have to watch")

    assert len(memories) == 1
    assert memories[0]["timestamp"] == ts
