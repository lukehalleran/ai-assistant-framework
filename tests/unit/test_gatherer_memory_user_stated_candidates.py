"""USR: [THE USER'S OWN STATEMENTS] candidate fixes A/B/C (2026-10-14 [test] re-probe).
Drives the deployed get_user_stated_evidence / builder filter. class: BC-58, BC-46

Original batch label: oct14 (renamed 2026-10-16; old path in tests/RENAMES.tsv)."""
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from core.prompt.builder import _drop_visible_user_stated
from core.prompt.gatherer_memory import (
    MemoryRetrievalMixin, USER_STATED_CANDIDATE_POOL, USER_STATED_MAX_ITEMS,
)

QUERY = "where do I work?"


class _Model:
    def __init__(self, cos):
        self.cos = cos

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        out = []
        for t in texts:
            if t == QUERY:
                out.append([1.0, 0.0])
            else:
                c = self.cos.get(t, 0.6)
                out.append([c, (1 - c * c) ** 0.5])
        return np.array(out, dtype=float)


class _Corpus:
    def __init__(self, items):
        self._items = items

    def get_recent_memories(self, count=3):
        return list(self._items[:count])


# Wall-clock pin (2026-10-15, class: wall-clock-dependent test): the session boost
# needs "same calendar day AND < gap hours" (_detect_session_boundary), so a bare
# datetime.now() - 100min is yesterday for the first 100 minutes after midnight and
# the newer-same-session assertion flipped. Pin the name the deployed gatherer reads
# and build every timestamp from the same instant.
FIXED_NOW = datetime(2026, 10, 15, 12, 0, 0)


class _PinnedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW


@pytest.fixture(autouse=True)
def _pin_gatherer_clock(monkeypatch):
    import core.prompt.gatherer_memory as gm
    monkeypatch.setattr(gm, "datetime", _PinnedDatetime)


def _ts(delta):
    return (FIXED_NOW - delta).isoformat()


def _gatherer(entries, cos):
    g = MemoryRetrievalMixin.__new__(MemoryRetrievalMixin)
    g.memory_coordinator = SimpleNamespace(
        corpus_manager=_Corpus([{"query": q, "response": "r", "timestamp": ts} for q, ts in entries]),
        chroma_store=SimpleNamespace(get_st_model=lambda: _Model(cos)))
    return g


@pytest.mark.asyncio
async def test_a_test_question_excluded_and_test_statement_unwrapped():
    q = "[test]where do I work?[/test]"
    stmt = "[test]I work at Halvorsen Marine as an operations planner.[/test]"
    inner = "I work at Halvorsen Marine as an operations planner."
    entries = [(q, _ts(timedelta(minutes=1))), (stmt, _ts(timedelta(minutes=5)))]
    cos = {"where do I work?": 0.885, inner: 0.595}
    got = await _gatherer(entries, cos).get_user_stated_evidence(QUERY, "factual_recall")
    texts = [r["query"] for r in got]
    assert texts == [inner]
    assert not any("where do I work" in t for t in texts)


@pytest.mark.asyncio
async def test_c_newer_same_session_statement_ranks_first():
    old = "I work at Brightwater Logistics as a data analyst."
    new = "I work at Halvorsen Marine as an operations planner."
    entries = [(new, _ts(timedelta(minutes=1))), (old, _ts(timedelta(minutes=100)))]
    got = await _gatherer(entries, {old: 0.613, new: 0.595}).get_user_stated_evidence(QUERY, "factual_recall")
    assert [r["query"] for r in got] == [new, old]


@pytest.mark.asyncio
async def test_c_old_session_pair_keeps_cosine_order():
    a = "I work at Brightwater Logistics as a data analyst."
    b = "I work at Halvorsen Marine as an operations planner."
    entries = [(b, _ts(timedelta(days=10))), (a, _ts(timedelta(days=11)))]
    got = await _gatherer(entries, {a: 0.613, b: 0.595}).get_user_stated_evidence(QUERY, "factual_recall")
    assert [r["query"] for r in got] == [a, b]


def test_b_dedup_before_k_not_empty():
    ranked = [{"query": f"statement number {i} about my job"} for i in range(5)]
    recent = [{"query": r["query"]} for r in ranked[:3]]
    got = _drop_visible_user_stated(ranked, recent, USER_STATED_MAX_ITEMS)
    assert [u["query"] for u in got] == [ranked[3]["query"], ranked[4]["query"]]


def test_b_envelope_inner_text_compared_and_k_applied():
    ranked = [{"query": f"statement number {i} about my job"} for i in range(6)]
    recent = [{"query": f"[test]{ranked[0]['query']}[/test]"}]
    got = _drop_visible_user_stated(ranked, recent, 3)
    assert [u["query"] for u in got] == [r["query"] for r in ranked[1:4]]


def test_b_pool_wider_than_k():
    assert USER_STATED_CANDIDATE_POOL > USER_STATED_MAX_ITEMS
