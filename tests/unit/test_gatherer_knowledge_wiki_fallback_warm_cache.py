"""Wiki live fallback keeps deadline-missed lookups alive to warm the cache.

2026-10-17, class BC-69 (latency).  Since the 2.5 s deadline landed the live
fallback ran 10 times and kept 1 term once: the deadline CANCELLED the pending
lookups, so a slow lookup never reached ``_wiki_cache`` and the next turn on the
same term timed out again.  Drives the DEPLOYED ``_get_wiki_content`` and the
deployed ``_get_wiki_snippet_cached`` (only the network lookup is a fake).
"""
import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import core.prompt.gatherer_knowledge as mod
from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin

QUERY = "tell me about quantum entanglement and black holes"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(mod, "_wiki_cache", {})
    monkeypatch.setattr(mod, "_WIKI_BACKGROUND_LOOKUPS", set())
    monkeypatch.setattr(mod, "logger", Mock())
    import config.app_config as ac
    monkeypatch.setattr(ac, "WIKI_FALLBACK_DEADLINE_S", 0.1)
    yield


class _G(KnowledgeRetrievalMixin):
    def _wiki_cache_key(self, query):  # the compositor's definition (context_gatherer.py:366)
        return mod.clean_query(query).lower().strip()


def _gatherer(monkeypatch, terms):
    monkeypatch.setitem(sys.modules, "knowledge.WikiManager", SimpleNamespace(
        _keywords_from_query=Mock(return_value=list(terms))))
    g = _G()
    g.memory_coordinator = SimpleNamespace(chroma_store=None)
    g._should_skip_wikipedia = lambda q: False
    return g


def _slow_lookup(monkeypatch, delays, calls):
    def fake(term):
        import time
        calls.append(term)
        time.sleep(delays[term])
        return {"title": term, "content": f"about {term}"}
    monkeypatch.setattr(mod, "get_wiki_snippet", fake)


@pytest.mark.asyncio
async def test_deadline_missed_lookup_warms_cache_for_next_turn(monkeypatch):
    calls = []
    _slow_lookup(monkeypatch, {"fast": 0.0, "slow": 0.4}, calls)
    g = _gatherer(monkeypatch, ["fast", "slow"])
    res = await g._get_wiki_content(QUERY, limit=2)
    assert [r["title"] for r in res] == ["fast"]
    assert len(mod._WIKI_BACKGROUND_LOOKUPS) == 1
    await asyncio.gather(*list(mod._WIKI_BACKGROUND_LOOKUPS))
    assert g._wiki_cache_key("slow") in mod._wiki_cache
    assert not mod._WIKI_BACKGROUND_LOOKUPS  # done-callback discarded it
    # second turn: served from the warmed cache, no new network lookup
    calls.clear()
    res2 = await g._get_wiki_content(QUERY, limit=2)
    assert [r["title"] for r in res2] == ["fast", "slow"]
    assert calls == []


@pytest.mark.asyncio
async def test_background_set_is_bounded(monkeypatch):
    mod._WIKI_BACKGROUND_LOOKUPS.update(
        asyncio.ensure_future(asyncio.sleep(5)) for _ in range(mod._WIKI_BACKGROUND_MAX))
    seeded = set(mod._WIKI_BACKGROUND_LOOKUPS)
    g = _gatherer(monkeypatch, ["a", "b"])
    started = []

    async def never(term):
        started.append(asyncio.current_task())
        await asyncio.sleep(5)

    g._get_wiki_snippet_cached = never
    await g._get_wiki_content(QUERY, limit=2)
    await asyncio.sleep(0)
    assert mod._WIKI_BACKGROUND_LOOKUPS == seeded
    assert all(t.cancelled() or t.cancelling() or t.done() for t in started)
    for t in seeded:
        t.cancel()


@pytest.mark.asyncio
async def test_caller_cancellation_still_cancels_everything(monkeypatch):
    g = _gatherer(monkeypatch, ["a", "b"])
    started = []

    async def never(term):
        started.append(asyncio.current_task())
        await asyncio.sleep(5)

    g._get_wiki_snippet_cached = never
    task = asyncio.ensure_future(g._get_wiki_content(QUERY, limit=2))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert not mod._WIKI_BACKGROUND_LOOKUPS
    assert all(t.cancelled() for t in started)


@pytest.mark.asyncio
async def test_background_exception_is_retrieved(monkeypatch):
    g = _gatherer(monkeypatch, ["bad"])

    async def boom(term):
        await asyncio.sleep(0.3)
        raise RuntimeError("boom")

    g._get_wiki_snippet_cached = boom
    await g._get_wiki_content(QUERY, limit=1)
    pending = list(mod._WIKI_BACKGROUND_LOOKUPS)
    assert len(pending) == 1
    await asyncio.gather(*pending, return_exceptions=True)
    await asyncio.sleep(0)
    assert not mod._WIKI_BACKGROUND_LOOKUPS
