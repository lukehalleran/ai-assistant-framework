"""Startup warm-up loads Tavily off the event loop (2026-10-16, class: BC-41).

Live evidence: the first turn after restart logged a 6965 ms LoopStall at
``_ensure_tavily`` because ``is_available()`` (on the loop) imported ``tavily``.
These tests drive the deployed ``_run_model_warmup`` and ``WebSearchManager``;
the tavily library is a fake -- no network, no real import.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from knowledge.web_search_manager import WebSearchManager


class _InlineThread:
    def __init__(self, target=None, daemon=None):
        self._target = target

    def start(self):
        self._target()


@pytest.fixture(autouse=True)
def _isolate_warmup_models(monkeypatch):
    import core.prompt.gatherer_knowledge as gatherer
    import knowledge.clip_manager as clip
    import knowledge.visual_memory_store as visual_store
    import memory.memory_retriever as retriever
    import utils.need_detector as needs
    import utils.tone_detector as tone
    import utils.web_search_trigger as trigger

    monkeypatch.setattr(retriever.MemoryRetriever, "_cross_encoder", MagicMock())
    monkeypatch.setattr(trigger, "_get_search_anchors", MagicMock())
    monkeypatch.setattr(tone, "_get_exemplar_embeddings", MagicMock())
    monkeypatch.setattr(needs, "_get_need_exemplar_embeddings", MagicMock())
    sem = MagicMock()
    sem.acquire.return_value = False
    monkeypatch.setattr(gatherer, "_WIKI_SEM_INFLIGHT", sem)
    monkeypatch.setattr(clip, "get_clip_manager", MagicMock())
    monkeypatch.setattr(visual_store, "VisualMemoryStore", MagicMock())


@pytest.fixture
def fake_tavily(monkeypatch):
    built = []

    class _Client:
        def __init__(self, api_key):
            built.append(api_key)

    mod = types.ModuleType("tavily")
    mod.TavilyClient = _Client
    monkeypatch.setitem(sys.modules, "tavily", mod)
    return built


def _orchestrator(manager):
    orch = MagicMock()
    orch.prompt_builder.context_gatherer.web_search_manager = manager
    return orch


def _warm(orch):
    from gui.launch import _run_model_warmup
    with patch("threading.Thread", _InlineThread):
        _run_model_warmup(orch)


def test_warmup_builds_the_live_managers_client_once_and_is_available_is_cheap(fake_tavily):
    manager = WebSearchManager(api_key="test-key")
    assert manager._initialized is False
    _warm(_orchestrator(manager))
    assert fake_tavily == ["test-key"]          # built during warm-up
    assert manager.is_available() is True       # hot path: no further build
    assert fake_tavily == ["test-key"]


def test_warmup_honours_disabled_web_search(fake_tavily, monkeypatch):
    monkeypatch.setattr("config.app_config.WEB_SEARCH_ENABLED", False)
    manager = WebSearchManager(api_key="test-key")
    _warm(_orchestrator(manager))
    assert fake_tavily == [] and manager._initialized is False
    monkeypatch.setattr("config.app_config.WEB_SEARCH_ENABLED", True)  # same instance, toggled on
    manager.warm()
    assert fake_tavily == ["test-key"]


def test_warmup_survives_a_manager_that_raises():
    manager = MagicMock()
    manager.warm.side_effect = RuntimeError("boom")
    _warm(_orchestrator(manager))  # must not raise


def test_warm_never_raises_when_ensure_tavily_does(monkeypatch):
    manager = WebSearchManager(api_key="k")
    monkeypatch.setattr(manager, "_ensure_tavily", MagicMock(side_effect=RuntimeError("x")))
    assert manager.warm() is False


def test_warmup_constructs_no_second_manager(fake_tavily, monkeypatch):
    constructed = []
    real_init = WebSearchManager.__init__

    def counting_init(self, *a, **kw):
        constructed.append(1)
        real_init(self, *a, **kw)

    monkeypatch.setattr(WebSearchManager, "__init__", counting_init)
    manager = WebSearchManager(api_key="test-key")
    n_before = len(constructed)
    _warm(_orchestrator(manager))
    assert len(constructed) == n_before == 1
