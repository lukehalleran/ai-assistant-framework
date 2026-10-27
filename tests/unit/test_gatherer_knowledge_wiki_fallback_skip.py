"""Wiki live fallback is skipped when its expected yield is nil.

Incident: oct27 wave 2 lane L1 (M-6), 2026-10-27. Class: BC-69 (latency),
BC-70 (disabled state re-paid per turn).  The FAISS semantic index was absent
in 51/51 processes; with the local ``wiki_knowledge`` store also answering
nothing, the live Wikipedia lookup ran every such turn (timed out on 37/192,
best score below the floor 33 times).  Contract (drives the DEPLOYED
``_get_wiki_content``; only the store / network lookup / index probe are fakes):
  * chroma answered EMPTY and the index is absent -> no live lookup;
  * the index IS present -> fallback unchanged (still runs);
  * chroma missing or errored -> fallback unchanged (the only source).
"""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import core.prompt.gatherer_knowledge as mod
import knowledge.semantic_search as semantic_search
from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin

QUERIES = [
    "tell me about quantum entanglement and black holes",
    "   tell me about quantum entanglement\n   and black holes\n",
]


class _G(KnowledgeRetrievalMixin):
    def _wiki_cache_key(self, query):
        return mod.clean_query(query).lower().strip()


class _Chroma:
    def __init__(self, rows=None, raises=False):
        self.collections = {"wiki_knowledge": SimpleNamespace(count=lambda: 5)}
        self._rows, self._raises = rows or [], raises

    def query_collection(self, name, query, n_results=5):
        if self._raises:
            raise RuntimeError("store down")
        return list(self._rows)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(mod, "_wiki_cache", {})
    monkeypatch.setattr(mod, "_WIKI_BACKGROUND_LOOKUPS", set())
    monkeypatch.setattr(mod, "logger", Mock())
    monkeypatch.setitem(sys.modules, "knowledge.WikiManager", SimpleNamespace(
        _keywords_from_query=Mock(return_value=["quantum", "entanglement"])))


def _gatherer(monkeypatch, chroma, index_ok, calls):
    monkeypatch.setattr(semantic_search, "index_available", lambda: index_ok)

    def fake(term):
        calls.append(term)
        return {"title": term, "content": f"about {term}"}

    monkeypatch.setattr(mod, "get_wiki_snippet", fake)
    g = _G()
    g.memory_coordinator = SimpleNamespace(chroma_store=chroma)
    g._should_skip_wikipedia = lambda q: False
    return g


@pytest.mark.asyncio
@pytest.mark.parametrize("query", QUERIES, ids=["clean", "wrapped"])
async def test_empty_store_and_no_index_skips_the_live_lookup(monkeypatch, query):
    calls = []
    g = _gatherer(monkeypatch, _Chroma(), index_ok=False, calls=calls)
    res = await g._get_wiki_content(query, limit=2)
    assert list(res) == []
    assert calls == []            # base: live lookup runs for every term


@pytest.mark.asyncio
@pytest.mark.parametrize("query", QUERIES, ids=["clean", "wrapped"])
async def test_index_present_keeps_the_fallback(monkeypatch, query):
    calls = []
    g = _gatherer(monkeypatch, _Chroma(), index_ok=True, calls=calls)
    res = await g._get_wiki_content(query, limit=2)
    assert sorted(calls) == ["entanglement", "quantum"]
    assert len(res) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("query", QUERIES, ids=["clean", "wrapped"])
async def test_missing_or_failing_store_keeps_the_fallback(monkeypatch, query):
    for chroma in (None, _Chroma(raises=True)):
        calls = []
        g = _gatherer(monkeypatch, chroma, index_ok=False, calls=calls)
        mod._wiki_cache.clear()
        res = await g._get_wiki_content(query, limit=2)
        assert sorted(calls) == ["entanglement", "quantum"]
        assert len(res) == 2
