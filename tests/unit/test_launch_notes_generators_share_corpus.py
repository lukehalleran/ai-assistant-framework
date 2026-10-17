"""The startup daily-notes catch-up shares the LIVE corpus manager.

2026-10-17, class BC-81 (a second CorpusManager is a second writer).
gui/launch built `DailyNotesGenerator()` with no corpus manager inside the
live process; main.py's shutdown path was fixed the same way in #80.
Drives THE deployed `_run_daily_notes_catchup` / `start_background_tasks`.
"""
import threading
from types import SimpleNamespace

import gui.launch as launch


class _Thread:
    """Run the target inline so the test is deterministic."""

    def __init__(self, target=None, daemon=None, **kw):
        self._t = target

    def start(self):
        self._t()


def test_catchup_passes_live_corpus_manager(monkeypatch):
    seen = {}

    class FakeGen:
        def __init__(self, *a, **kw):
            seen["kw"] = kw

        async def generate_yesterday_if_missing(self):
            return None

    monkeypatch.setattr(launch.app_config, "DAILY_NOTES_ENABLED", True)
    monkeypatch.setattr(launch.daily_notes_generator, "DailyNotesGenerator", FakeGen)
    monkeypatch.setattr(launch.threading, "Thread", _Thread)
    live = object()
    launch._run_daily_notes_catchup(corpus_manager=live)
    assert seen["kw"]["corpus_manager"] is live


def test_start_background_tasks_hands_over_orchestrator_corpus(monkeypatch):
    live = object()
    got = {}
    monkeypatch.setattr(launch, "_run_daily_notes_catchup", lambda corpus_manager=None: got.setdefault("cm", corpus_manager))
    for name in ("_run_weekly_notes_catchup", "_run_monthly_notes_catchup", "_run_reference_docs_seed", "_run_model_warmup"):
        monkeypatch.setattr(launch, name, lambda *a, **k: None)
    orch = SimpleNamespace(memory_system=SimpleNamespace(corpus_manager=live, chroma_store=object()))
    launch.start_background_tasks(orch)
    assert got["cm"] is live
