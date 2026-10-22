"""Startup warm-up pays the web-trigger embedder's first query encode (BC-41).

Incident 2026-10-20: the cached MiniLM embedder's first per-query use (device
move + encode) stalled the loop 836 ms on turn 1.  The warm-up must run THE
deployed ``_semantic_search_boost`` once, and keep warming the Tavily client.
"""
from unittest.mock import MagicMock, patch

import pytest

from tests.unit.test_warmup_guards import _InlineThread, _isolate_warmup_models  # noqa: F401


def test_warmup_runs_the_per_query_boost_and_tavily_warm(monkeypatch):
    import utils.web_search_trigger as trigger
    from gui.launch import _run_model_warmup

    boost = MagicMock(return_value=0.0)
    monkeypatch.setattr(trigger, "_semantic_search_boost", boost)
    orch = MagicMock()
    manager = orch.prompt_builder.context_gatherer.web_search_manager
    with patch("threading.Thread", _InlineThread):
        _run_model_warmup(orch)
    boost.assert_called_once()
    manager.warm.assert_called_once()
