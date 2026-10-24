"""Shutdown step that keeps the procedural git index fresh.

Incident 2026-10-24, classes BC-81 (the sync must use the daemon's OWN chroma
store, never construct a second one) / BC-58 / BC-82 (toggle read at call time).
Drives the deployed ``main._sync_git_index`` and the deployed
``main._do_shutdown_async`` ordering seam (sync before backup).
"""

import asyncio
import logging
import sys
import threading
import types
from types import SimpleNamespace

import pytest

import config.app_config as app_config


@pytest.fixture
def main_mod():
    import main
    return main


@pytest.fixture
def seen(monkeypatch):
    """Replace GitMemoryLoader with a recorder; return the recorded calls."""
    import knowledge.git_memory_loader as gml
    rec = SimpleNamespace(stores=[], updates=[], behaviour=lambda cancel: 3)

    class FakeLoader:
        def __init__(self, chroma_store=None, repo_path=None):
            assert chroma_store is not None, "must never build its own store (BC-81)"
            rec.stores.append(chroma_store)

        def incremental_update(self, include_diffs=True, max_commits=None, cancel=None):
            rec.updates.append((include_diffs, max_commits, cancel))
            return rec.behaviour(cancel)

    monkeypatch.setattr(gml, "GitMemoryLoader", FakeLoader)
    monkeypatch.setattr(app_config, "GIT_MEMORY_ENABLED", True)
    monkeypatch.setattr(app_config, "GIT_MEMORY_SYNC_ON_SHUTDOWN", True)
    monkeypatch.setattr(app_config, "GIT_MEMORY_SYNC_TIMEOUT_S", 5.0)
    return rec


def _orch(store):
    return SimpleNamespace(memory_system=SimpleNamespace(chroma_store=store))


def test_runs_against_the_live_store(main_mod, seen):
    live = object()
    asyncio.run(main_mod._sync_git_index(_orch(live)))
    assert seen.stores == [live]
    assert len(seen.updates) == 1


def test_sync_on_shutdown_false_is_not_called_and_reads_live_config(main_mod, seen, monkeypatch):
    monkeypatch.setattr(app_config, "GIT_MEMORY_SYNC_ON_SHUTDOWN", False)
    asyncio.run(main_mod._sync_git_index(_orch(object())))
    assert seen.updates == []
    monkeypatch.setattr(app_config, "GIT_MEMORY_SYNC_ON_SHUTDOWN", True)  # same process, flipped back
    asyncio.run(main_mod._sync_git_index(_orch(object())))
    assert len(seen.updates) == 1


def test_git_memory_disabled_is_not_called(main_mod, seen, monkeypatch):
    monkeypatch.setattr(app_config, "GIT_MEMORY_ENABLED", False)
    asyncio.run(main_mod._sync_git_index(_orch(object())))
    assert seen.updates == []


def test_no_live_store_is_skipped_not_constructed(main_mod, seen):
    asyncio.run(main_mod._sync_git_index(SimpleNamespace(memory_system=SimpleNamespace(chroma_store=None))))
    assert seen.stores == [] and seen.updates == []


def test_timeout_warns_and_cancels_the_late_thread(main_mod, seen, monkeypatch, caplog):
    monkeypatch.setattr(app_config, "GIT_MEMORY_SYNC_TIMEOUT_S", 0.2)
    release = threading.Event()
    seen.behaviour = lambda cancel: (release.wait(2), 0)[1]
    with caplog.at_level(logging.WARNING):
        asyncio.run(main_mod._sync_git_index(_orch(object())))
    cancel = seen.updates[0][2]
    assert cancel.is_set(), "a timed-out run must be told to leave the anchor alone"
    assert any("resumes next shutdown" in r.getMessage() for r in caplog.records)
    release.set()


def test_exception_is_a_warning_not_fatal(main_mod, seen, caplog):
    def boom(cancel):
        raise RuntimeError("store exploded")
    seen.behaviour = boom
    with caplog.at_level(logging.WARNING):
        asyncio.run(main_mod._sync_git_index(_orch(object())))
    assert any("Git index sync failed (non-fatal)" in r.getMessage() for r in caplog.records)


def test_deployed_shutdown_runs_git_sync_before_backup(main_mod, seen, monkeypatch):
    order = []
    seen.behaviour = lambda cancel: (order.append("git"), 0)[1]

    async def _ok(*a, **k):
        return None

    fake_handlers = types.ModuleType("gui.handlers")
    fake_handlers.wait_for_pending_storage = _ok
    monkeypatch.setitem(sys.modules, "gui.handlers", fake_handlers)

    ms = SimpleNamespace(
        chroma_store=object(),
        run_shutdown_reflection=_ok,
        process_shutdown_memory=_ok,
        run_synthesis_dreaming=_ok,
    )
    orch = SimpleNamespace(memory_system=ms)
    monkeypatch.setattr(app_config, "DAILY_NOTES_ENABLED", False)
    monkeypatch.setattr(app_config, "CURATION_ENABLED", False)
    monkeypatch.setattr(app_config, "BACKUP_ENABLED", True)

    def fake_backup():
        order.append("backup")
        return SimpleNamespace(ok=False, path=None, skipped_reason="test", error=None,
                               chroma_included=False)

    monkeypatch.setattr(main_mod.backup_manager, "run_shutdown_backup", fake_backup)
    asyncio.run(main_mod._do_shutdown_async(orch, [], []))
    assert order == ["git", "backup"]
