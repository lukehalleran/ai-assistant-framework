"""Wiki lane (2026-10-15, class BC-69 latency / BC-87 gather / BC-58 categorized rule).

Probe turn 14:48 ("What was I working on in late September, and when is sim HW 4
due?"): wiki task 12.3s because the live-Wikipedia fallback looked terms up
SEQUENTIALLY (up to 5s each), then the budget emptied the section. Drives the
DEPLOYED ``_get_wiki_content`` with a fake lookup (no network), and the deployed
intent profiles.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin

QUERY = "tell me about quantum entanglement and black holes and neutron stars"


@pytest.fixture
def gk(monkeypatch):
    import core.prompt.gatherer_knowledge as mod
    logger = Mock()
    monkeypatch.setattr(mod, "logger", logger)
    monkeypatch.setattr(mod, "_WIKI_CHROMA_INFLIGHT", threading.Semaphore(2))
    return mod, logger


def _gatherer(monkeypatch, terms, delays, raising=()):
    monkeypatch.setitem(sys.modules, "knowledge.WikiManager", SimpleNamespace(
        _keywords_from_query=Mock(return_value=list(terms)),
    ))
    g = KnowledgeRetrievalMixin()
    g.memory_coordinator = SimpleNamespace(chroma_store=None)  # empty -> live fallback
    g._should_skip_wikipedia = lambda q: False

    async def fake(term):
        await asyncio.sleep(delays[term])
        if term in raising:
            raise RuntimeError("boom")
        return {"title": term, "content": f"about {term}"}

    g._get_wiki_snippet_cached = fake
    return g


@pytest.mark.asyncio
async def test_three_terms_run_concurrently_in_order(gk, monkeypatch):
    g = _gatherer(monkeypatch, ["a", "b", "c"], {"a": 1.0, "b": 1.0, "c": 1.0})
    t0 = time.perf_counter()
    res = await g._get_wiki_content(QUERY, limit=3)
    wall = time.perf_counter() - t0
    assert wall < 1.8, wall
    assert [r["title"] for r in res] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_slow_term_cut_by_deadline(gk, monkeypatch):
    import config.app_config as ac
    mod, logger = gk
    monkeypatch.setattr(ac, "WIKI_FALLBACK_DEADLINE_S", 0.5)

    g = _gatherer(monkeypatch, ["a", "slow", "c"], {"a": 0.05, "slow": 10.0, "c": 0.05})
    t0 = time.perf_counter()
    res = await g._get_wiki_content(QUERY, limit=3)
    wall = time.perf_counter() - t0
    assert wall < 1.0, wall
    assert [r["title"] for r in res] == ["a", "c"]
    info = [c for c in logger.info.call_args_list if "live fallback deadline" in c.args[0]]
    assert len(info) == 1 and info[0].args[1:] == (2, 3)
    timing = [c for c in logger.debug.call_args_list
              if c.args and c.args[0] == "[WikiTiming] task=wiki %s"]
    assert timing[0].args[1]["timed_out"] is True


@pytest.mark.asyncio
async def test_raising_lookup_is_skipped(gk, monkeypatch):
    g = _gatherer(monkeypatch, ["a", "bad", "c"],
                  {"a": 0.0, "bad": 0.0, "c": 0.0}, raising=("bad",))
    res = await g._get_wiki_content(QUERY, limit=3)
    assert [r["title"] for r in res] == ["a", "c"]


@pytest.mark.asyncio
async def test_caller_cancellation_propagates_and_cancels_lookups(gk, monkeypatch):
    g = _gatherer(monkeypatch, ["a", "b"], {"a": 10.0, "b": 10.0})
    task = asyncio.ensure_future(g._get_wiki_content(QUERY, limit=2))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("name", ["TEMPORAL_RECALL", "EMOTIONAL_SUPPORT"])
def test_personal_intents_skip_wiki(name):
    from core.intent_classifier import _PROFILES, IntentType
    assert _PROFILES[getattr(IntentType, name)]["retrieval"]["max_wiki"] == 0
