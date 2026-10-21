"""The web-search heuristic (and its semantic boost) runs off the event loop.

Incident 2026-10-20, class BC-41.  LoopStall 836 ms at
``web_search_trigger._semantic_search_boost`` (the cached MiniLM embedder's first
use: device move + encode) reached from the async trigger/gatherer.  Every test
drives THE deployed async caller; only the model-touching boost is replaced by a
recorder of ``threading.get_ident()``.
"""
import threading

import pytest

import utils.web_search_trigger as wst
from core.prompt.gatherer_web import WebSearchMixin

# Confidence lands in the ambiguous band (0.1-0.45), so the heuristic consults
# the semantic boost.
AMBIGUOUS = "how is the weather"
AMBIGUOUS_WRAPPED = "   \n    how is the weather\n\n   "


@pytest.fixture
def boost_thread(monkeypatch):
    seen = []

    def fake_boost(query, threshold=0.35):
        seen.append(threading.get_ident())
        return 0.0

    monkeypatch.setattr(wst, "_semantic_search_boost", fake_boost)
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [AMBIGUOUS, AMBIGUOUS_WRAPPED])
async def test_analyze_for_web_search_llm_boost_off_loop(boost_thread, query):
    await wst.analyze_for_web_search_llm(query, model_manager=None)
    assert boost_thread, "the semantic boost was never reached"
    assert all(t != threading.get_ident() for t in boost_thread)


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [AMBIGUOUS, AMBIGUOUS_WRAPPED])
async def test_should_search_with_llm_boost_off_loop(boost_thread, query):
    await wst.should_search_with_llm(query, model_manager=None)
    assert boost_thread
    assert all(t != threading.get_ident() for t in boost_thread)


class _Manager:
    def is_available(self):
        return True

    def is_enabled(self):
        return True

    rate_limiter = None


class _Host(WebSearchMixin):
    def __init__(self, trigger):
        self.web_search_manager = _Manager()
        self.web_search_trigger_llm = None
        self.web_search_trigger = trigger
        self.model_manager = None
        self.memory_id_map = {}


@pytest.mark.asyncio
async def test_gatherer_sync_heuristic_fallback_off_loop(boost_thread):
    host = _Host(wst.analyze_for_web_search)
    await host._get_web_search_results(AMBIGUOUS)
    assert boost_thread
    assert all(t != threading.get_ident() for t in boost_thread)
