"""B4R: [THE USER'S OWN STATEMENTS] ranks by cosine + content overlap + session,
not raw cosine alone (2026-10-13 probe: answer 0.613 vs generic chatter 0.628).
Drives the deployed get_user_stated_evidence. class: BC-46"""
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.prompt.formatter import PromptFormatter
from core.prompt.gatherer_memory import (
    MemoryRetrievalMixin, USER_STATED_MIN_COSINE, _content_stems, _lexical_overlap,
)

QUERY = "where do I work?"
ANSWER = "I work at Brightwater Logistics as a data analyst."


class _Model:
    """Dict-keyed fake bge: query is unit x; each text gets the requested cosine."""
    def __init__(self, cos):
        self.cos = cos

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        out = []
        for t in texts:
            if t == QUERY:
                out.append([1.0, 0.0])
            else:
                c = self.cos[t]
                out.append([c, (1 - c * c) ** 0.5])
        return np.array(out, dtype=float)


class _Corpus:
    def __init__(self, items):
        self._items = items

    def get_recent_memories(self, count=3):
        return list(self._items[:count])


# Wall-clock pin (2026-10-15, class: wall-clock-dependent test): the session boost
# is "same calendar day AND < gap hours" (_detect_session_boundary), so a bare
# datetime.now() - 10min crossed into yesterday between 00:00 and 00:10 and the
# boosted statement lost. The deployed gatherer reads datetime.now() from its own
# module; pin THAT name and build every timestamp from the same instant.
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


OLD = timedelta(days=14)
GENERIC = ["I went to the career fair and talked to recruiters",
           "I think I did pretty well at the booth today",
           "I handed out a few copies of my resume yesterday"]


@pytest.mark.asyncio
async def test_probe_shape_answer_ranks_first():
    entries = [(ANSWER, _ts(timedelta(minutes=5)))] + [(g, _ts(OLD + timedelta(hours=i))) for i, g in enumerate(GENERIC)]
    cos = {ANSWER: 0.613, GENERIC[0]: 0.628, GENERIC[1]: 0.626, GENERIC[2]: 0.62}
    got = await _gatherer(entries, cos).get_user_stated_evidence(QUERY, "factual_recall")
    assert got[0]["query"] == ANSWER
    assert got[0]["similarity"] == 0.613 and "score" in got[0] and got[0]["score"] > 0.613


@pytest.mark.asyncio
async def test_overlap_alone_wins():
    a, b = "I work on weekends at the store downtown", "I walked to the lake with friends yesterday"
    entries = [(b, _ts(OLD)), (a, _ts(OLD + timedelta(hours=1)))]
    got = await _gatherer(entries, {a: 0.60, b: 0.61}).get_user_stated_evidence(QUERY, "factual_recall")
    assert got[0]["query"] == a


@pytest.mark.asyncio
async def test_session_boost_alone_wins():
    a, b = "I walked to the lake with friends yesterday", "I baked bread with my grandmother once"
    entries = [(b, _ts(OLD)), (a, _ts(timedelta(minutes=10)))]
    got = await _gatherer(entries, {a: 0.60, b: 0.62}).get_user_stated_evidence(QUERY, "factual_recall")
    assert got[0]["query"] == a


def test_stop_words_never_count():
    assert _content_stems("where do I work?") == frozenset({"work"})
    assert _lexical_overlap(_content_stems(QUERY), "where do I go") == 0.0


@pytest.mark.asyncio
async def test_empty_query_stems_keeps_cosine_order():
    q = "where do I go"
    assert _content_stems(q) == frozenset()
    a, b = "I walked to the lake with friends yesterday", "I baked bread with my grandmother once"
    entries = [(a, _ts(OLD)), (b, _ts(OLD + timedelta(hours=1)))]

    class M(_Model):
        def encode(self, texts, **k):
            return np.array([[1.0, 0.0] if t == q else [self.cos[t], (1 - self.cos[t] ** 2) ** 0.5]
                             for t in texts])
    g = _gatherer(entries, {a: 0.55, b: 0.65})
    g.memory_coordinator.chroma_store = SimpleNamespace(get_st_model=lambda: M({a: 0.55, b: 0.65}))
    got = await g.get_user_stated_evidence(q, "factual_recall")
    assert [r["query"] for r in got] == [b, a]


@pytest.mark.asyncio
async def test_min_cosine_floor_on_raw_cosine():
    low = "I work at Brightwater Logistics as an analyst"
    entries = [(low, _ts(timedelta(minutes=1)))]
    got = await _gatherer(entries, {low: 0.40}).get_user_stated_evidence(QUERY, "factual_recall")
    assert 0.40 < USER_STATED_MIN_COSINE and got == []


def test_formatter_header_sentence():
    fmt = PromptFormatter(token_manager=MagicMock(count_tokens=MagicMock(return_value=10)), time_manager=None)
    fmt._feature_inventory_cache = None
    prompt = fmt._assemble_prompt({"user_stated_evidence": [{"query": ANSWER, "timestamp": None}]}, QUERY)
    assert "Use only the statements that answer this question; ignore any that do not." in prompt
