"""B4 item 4: pastes do not exhaust the N-turn [RECENT CONVERSATION] window (BC-46, BC-75)."""
from types import SimpleNamespace

import pytest

from core.prompt.gatherer_memory import MemoryRetrievalMixin

PASTE = "lorem ipsum " * 400  # > FACT_EXTRACT_PASTE_CHARS (1500)


class _Corpus:
    """Honours `count` like CorpusManager.get_recent_memories (newest first)."""
    def __init__(self, newest_first):
        self._items = newest_first

    def get_recent_memories(self, count=3):
        return list(self._items[:count])


def _gatherer(items):
    g = MemoryRetrievalMixin.__new__(MemoryRetrievalMixin)
    g.memory_coordinator = SimpleNamespace(corpus_manager=_Corpus(items))
    g.memory_id_map = {}
    g.user_profile = None
    return g


def _turn(i, q):
    return {"query": q, "response": f"r{i}", "timestamp": f"2026-10-11T22:{59 - i:02d}:00"}


@pytest.mark.asyncio
async def test_fact_turn_survives_three_pastes():
    fact = "I just started a new job at Northwind as a data analyst"
    # newest first: 8 short, 3 pastes, the fact turn, older chatter
    items = [_turn(i, f"short {i}") for i in range(8)]
    items += [_turn(8 + i, PASTE) for i in range(3)]
    items += [_turn(11, fact)] + [_turn(12 + i, f"old {i}") for i in range(5)]
    got = await _gatherer(items)._get_recent_conversations(limit=10)
    assert any(m["query"] == fact for m in got)
    # 3 pastes did not count, so 10 SHORT turns incl. the fact turn
    assert sum(1 for m in got if m["query"] != PASTE) == 10
    assert len(got) <= 20


@pytest.mark.asyncio
async def test_window_is_still_n_without_pastes():
    items = [_turn(i, f"short {i}") for i in range(25)]
    got = await _gatherer(items)._get_recent_conversations(limit=10)
    assert len(got) == 10 and got[0]["query"] == "short 0"


@pytest.mark.asyncio
async def test_hard_maximum_two_n():
    items = [_turn(i, PASTE) for i in range(30)]
    got = await _gatherer(items)._get_recent_conversations(limit=10)
    assert len(got) == 20
