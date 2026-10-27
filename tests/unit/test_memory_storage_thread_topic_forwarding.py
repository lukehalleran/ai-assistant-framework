"""Behavioural sibling for ``test_storage_forwards_current_topic`` in
tests/unit/test_thread_manager_session_start_continuity.py (BC-63: it grepped
``inspect.getsource(memory_storage)`` for ``self._thread_detect_fn(`` and
``current_topic=self.current_topic``).

Incident date: 2026-09-03 (thread topic label stability: thread detection
re-classified the topic instead of reusing the turn's label). Bug classes:
BC-63, BC-62. THE deployed ``MemoryStorage.store_interaction`` is driven with a
recording thread-detect function and a fake corpus: the turn's topic label must
reach the detector, and the detector's answer must reach the stored entry.
"""
from unittest.mock import MagicMock

import pytest

from memory.memory_storage import MemoryStorage


def _storage():
    chroma = MagicMock()
    chroma.add_conversation_memory.return_value = "doc-1"
    corpus = MagicMock()
    ms = MemoryStorage(corpus_manager=corpus, chroma_store=chroma,
                       fact_extractor=MagicMock())
    return ms, corpus


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "Biscuit chased the ball again",
    "    Biscuit chased the\n    ball again\n   ",
])
async def test_turn_topic_label_reaches_thread_detection_and_the_entry(query):
    ms, corpus = _storage()
    seen = []

    def detect(q, is_heavy, current_topic=None):
        seen.append({"query": q, "current_topic": current_topic})
        return {"thread_id": "t-9", "depth": 3, "started": "s", "topic": current_topic}

    ms._thread_detect_fn = detect
    ms.current_topic = "Playing Fetch"
    await ms.store_interaction(query=query, response="That sounds fun.")
    assert [s["current_topic"] for s in seen] == ["Playing Fetch"]
    kwargs = corpus.add_entry.call_args.kwargs
    assert kwargs["thread_id"] == "t-9"
    assert kwargs["thread_depth"] == 3
    assert kwargs["thread_topic"] == "Playing Fetch"


@pytest.mark.asyncio
async def test_topic_label_follows_the_live_attribute_between_turns():
    ms, _ = _storage()
    seen = []
    ms._thread_detect_fn = lambda q, h, current_topic=None: seen.append(current_topic) or {}
    ms.current_topic = "Playing Fetch"
    await ms.store_interaction(query="first message here", response="ok one")
    ms.current_topic = "Vet Visit"
    await ms.store_interaction(query="second message here", response="ok two")
    assert seen == ["Playing Fetch", "Vet Visit"]


@pytest.mark.asyncio
async def test_no_detector_stores_an_unthreaded_entry():
    ms, corpus = _storage()
    ms._thread_detect_fn = None
    await ms.store_interaction(query="a plain message", response="a plain reply")
    kwargs = corpus.add_entry.call_args.kwargs
    assert kwargs["thread_id"] is None
