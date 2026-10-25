"""
Guard smoke test for gui.launch._run_model_warmup.

Every warmup step must be independently guarded — warmup can never take down
startup, even with a completely broken orchestrator. Extended 2026-08-02 with
tone/need exemplar warm and the end-to-end memory-retrieval warm pass (turn-1
memories task ran 8-17s cold vs 3-5s warm; exemplars were computed inside the
user's first message).
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_warmup_models(monkeypatch):
    """Guard tests must never cold-load models or open the owner's stores."""
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
    semaphore = MagicMock()
    semaphore.acquire.return_value = False
    monkeypatch.setattr(gatherer, "_WIKI_SEM_INFLIGHT", semaphore)
    monkeypatch.setattr(clip, "get_clip_manager", MagicMock())
    monkeypatch.setattr(visual_store, "VisualMemoryStore", MagicMock())


class _InlineThread:
    """Run the warmup body synchronously so the test observes any exception."""

    def __init__(self, target=None, daemon=None):
        self._target = target

    def start(self):
        self._target()


class TestWarmupGuards:
    def test_broken_orchestrator_never_raises(self):
        from gui.launch import _run_model_warmup

        orch = MagicMock()
        # Force every attribute access down the failure path
        orch.memory_system = None
        orch.model_manager = None
        with patch("threading.Thread", _InlineThread):
            _run_model_warmup(orch)  # must not raise

    def test_retrieval_warm_uses_memory_system(self):
        from gui.launch import _run_model_warmup

        async def _fake_get_memories(query, limit=3):
            _fake_get_memories.called = (query, limit)
            return []

        orch = MagicMock()
        orch.memory_system.get_memories = _fake_get_memories
        with patch("threading.Thread", _InlineThread):
            _run_model_warmup(orch)
        assert getattr(_fake_get_memories, "called", None) is not None


class TestWarmupSpacy:
    """2026-10-24: warmup loads spaCy and warns once if an installed model fails."""

    def _run(self, monkeypatch, nlp, installed, caplog):
        import importlib.util
        import logging

        import memory.fact_extractor as fe
        from gui.launch import _run_model_warmup

        monkeypatch.setattr(fe, "_get_nlp", lambda: nlp)
        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util, "find_spec",
            lambda n, *a, **k: object() if n == "en_core_web_sm" and installed
            else (None if n == "en_core_web_sm" else real(n, *a, **k)))
        with caplog.at_level(logging.WARNING, logger="gui.launch"), \
                patch("threading.Thread", _InlineThread):
            _run_model_warmup(MagicMock())
        return [r.getMessage() for r in caplog.records if "spaCy" in r.getMessage()]

    def test_failed_load_with_model_installed_warns(self, monkeypatch, caplog):
        msgs = self._run(monkeypatch, None, True, caplog)
        assert len(msgs) == 1 and "Preflight" in msgs[0]

    def test_healthy_load_is_quiet(self, monkeypatch, caplog):
        assert self._run(monkeypatch, object(), True, caplog) == []

    def test_model_absent_is_quiet(self, monkeypatch, caplog):
        assert self._run(monkeypatch, None, False, caplog) == []
