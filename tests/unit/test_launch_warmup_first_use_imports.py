"""Warmup imports the lazily-imported calendar/email modules (incident 2026-10-19/20, BC-41).

Live: first-turn LoopStalls of 316 ms (get_google_calendar_events) and 487 ms
(EmailService.__init__) were importlib work on the loop. Those imports are deliberate
function-body imports, so warmup performs them once on its background thread.
"""
import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from gui import launch


class _InlineThread:
    def __init__(self, target=None, daemon=None):
        self._target = target

    def start(self):
        self._target()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    import core.prompt.gatherer_knowledge as gatherer
    import knowledge.clip_manager as clip
    import knowledge.visual_memory_store as vs
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
    monkeypatch.setattr(vs, "VisualMemoryStore", MagicMock())


def _warm():
    orch = MagicMock()
    orch.memory_system = None
    orch.model_manager = None
    with patch("threading.Thread", _InlineThread):
        launch._run_model_warmup(orch)


def test_warmup_imports_the_first_use_modules(monkeypatch):
    for mod in launch._FIRST_USE_IMPORT_MODULES:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    _warm()
    for mod in launch._FIRST_USE_IMPORT_MODULES:
        assert mod in sys.modules, mod
    assert "core.actions.google_calendar" in launch._FIRST_USE_IMPORT_MODULES
    assert "core.email.registry" in launch._FIRST_USE_IMPORT_MODULES


def test_failing_import_is_logged_and_later_modules_still_import(monkeypatch, capsys):
    mods = launch._FIRST_USE_IMPORT_MODULES
    real = importlib.import_module

    def _fake(name, *a, **k):
        if name == mods[0]:
            raise ImportError("boom")
        return real(name, *a, **k)

    for mod in mods[1:]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(launch.importlib, "import_module", _fake)
    _warm()
    out = capsys.readouterr().out
    assert f"first-use import {mods[0]} skip: boom" in out
    for mod in mods[1:]:
        assert mod in sys.modules, mod
    assert "Model warmup complete" in out
