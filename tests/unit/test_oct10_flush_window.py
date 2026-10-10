"""2026-10-10 (BC-30, BC-58): an exit run after an idle flush processes only the
turns since that flush.

`main._gather_session_state` returns no conversations, so every shutdown pass
slices the corpus by `ShutdownProcessor.session_start` (set ONCE from
`MemoryCoordinator.session_start` at process start). After an idle flush the
exit run therefore re-reflected / re-summarised / re-confirmed every turn since
PROCESS start. `_end_shutdown_run` now advances both session_start copies to the
completed non-exit run's START time.

These tests drive the deployed main-level entries (`_run_shutdown_tasks` for the
idle flush, `run_shutdown_tasks_async` for the exit) into a REAL
MemoryCoordinator -> REAL ShutdownProcessor.run_shutdown_reflection; only the LLM
and the reflection store are faked, and the prompt the LLM receives is the
observable "which turns did this run process".
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import memory.shutdown_processor as sp_mod
from memory.memory_coordinator import MemoryCoordinator
from memory.shutdown_processor import ShutdownProcessor


class _Corpus:
    def __init__(self):
        self.corpus = []


class _FakeModels:
    def __init__(self):
        self.prompts = []
        self.api_models = {}

    async def generate_once(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return "SESSION TOPIC: x"


class _FakeStorage:
    def __init__(self):
        self.reflections = []

    async def add_reflection(self, text, **kwargs):
        self.reflections.append(text)
        return True


def _turn(tag, ts):
    return {"query": f"question-{tag}", "response": f"answer-{tag}", "timestamp": ts, "tags": []}


@pytest.fixture
def env(monkeypatch):
    import main

    monkeypatch.setattr(sp_mod, "REFLECTION_MIN_EXCHANGES", 1)
    monkeypatch.setattr(sp_mod, "REFLECTIONS_ENABLED", True)

    corpus = _Corpus()
    models = _FakeModels()
    storage = _FakeStorage()
    process_start = datetime.now() - timedelta(hours=3)

    proc = ShutdownProcessor.__new__(ShutdownProcessor)
    proc.corpus_manager = corpus
    proc.model_manager = models
    proc._storage = storage
    proc.session_start = process_start

    coord = MemoryCoordinator.__new__(MemoryCoordinator)
    coord.time_manager = None  # _now() falls back to datetime.now()
    coord.session_start = process_start
    coord._shutdown = proc

    orch = SimpleNamespace(memory_system=coord)

    monkeypatch.setattr(main, "_mark_session_end", lambda orchestrator: None)
    monkeypatch.setattr(main, "_shutdown_requested", False)
    monkeypatch.setattr(main, "_process_exiting", False)
    monkeypatch.setattr(main, "_exit_shutdown_handled", False)
    monkeypatch.setattr(main, "_last_flush_done_at", 0.0)
    monkeypatch.setattr(main, "_flush_run_started_at", 0.0)
    monkeypatch.setattr(main, "_last_failed_flush_at", 0.0)
    monkeypatch.setattr(main, "_last_activity_time", time.time())
    monkeypatch.setattr(main, "_orchestrator_ref", orch)
    monkeypatch.setattr(main, "_shutdown_owner_thread", None)

    # The run body is the deployed reflection pass, reached through the
    # coordinator exactly as `_do_shutdown_async` reaches it.
    body = {"during": None, "raise": False}

    async def fake_do_shutdown_async(orchestrator, session_convos, session_summaries):
        if body["during"]:
            body["during"]()
        if body["raise"]:
            raise RuntimeError("flush failed")
        await orchestrator.memory_system.run_shutdown_reflection(
            session_conversations=session_convos, session_summaries=session_summaries)

    monkeypatch.setattr(main, "_do_shutdown_async", fake_do_shutdown_async)

    was_set = main._shutdown_done.is_set()
    try:
        yield SimpleNamespace(main=main, corpus=corpus, models=models, coord=coord,
                              proc=proc, orch=orch, body=body, process_start=process_start)
    finally:
        if was_set:
            main._shutdown_done.set()
        else:
            main._shutdown_done.clear()


def _flush(env):
    env.main._run_shutdown_tasks(env.orch)


def _exit_run(env):
    # New activity after the flush stamp, as a real post-flush turn would set.
    env.main._last_activity_time = max(env.main._last_flush_done_at, time.time()) + 10
    asyncio.run(env.main.run_shutdown_tasks_async(env.orch))


def _pre_flush_turns(env, n=3):
    now = datetime.now()
    for i in range(n):
        env.corpus.corpus.append(_turn(f"pre{i}", now - timedelta(hours=1, minutes=i)))


def _post_flush_turns(env, n=2):
    base = datetime.now() + timedelta(minutes=1)
    for i in range(n):
        env.corpus.corpus.append(_turn(f"post{i}", base + timedelta(seconds=i)))


def test_exit_run_after_flush_processes_only_post_flush_turns(env):
    _pre_flush_turns(env)
    _flush(env)
    assert len(env.models.prompts) == 1 and "question-pre0" in env.models.prompts[0]
    _post_flush_turns(env)

    _exit_run(env)

    assert len(env.models.prompts) == 2
    exit_prompt = env.models.prompts[1]
    assert "question-post0" in exit_prompt and "question-post1" in exit_prompt
    assert "question-pre" not in exit_prompt, "flushed turns were reprocessed"


def test_both_session_start_copies_advance_together(env):
    _pre_flush_turns(env)
    before = time.time()
    _flush(env)
    assert env.coord.session_start > env.process_start
    assert env.proc.session_start == env.coord.session_start
    # the new window opens at the run's START (not process start, not its end)
    delta = (datetime.now() - env.coord.session_start).total_seconds()
    assert 0 <= delta <= time.time() - before + 1


def test_failed_flush_does_not_advance_the_window(env):
    _pre_flush_turns(env)
    env.body["raise"] = True
    _flush(env)
    assert env.coord.session_start == env.process_start
    assert env.proc.session_start == env.process_start

    env.body["raise"] = False
    _post_flush_turns(env)
    _exit_run(env)
    exit_prompt = env.models.prompts[-1]
    assert "question-pre0" in exit_prompt and "question-post0" in exit_prompt


def test_turn_stored_during_the_run_stays_in_the_next_window(env):
    _pre_flush_turns(env)

    def store_mid_run():
        env.corpus.corpus.append(_turn("mid", datetime.now()))

    env.body["during"] = store_mid_run
    _flush(env)
    env.body["during"] = None
    _post_flush_turns(env)

    _exit_run(env)

    exit_prompt = env.models.prompts[-1]
    assert "question-mid" in exit_prompt
    assert "question-post0" in exit_prompt
    assert "question-pre" not in exit_prompt


def test_exit_run_itself_does_not_move_the_window(env):
    """The exit path sets _process_exiting first; the process is going away, so
    nothing is advanced (and a failed exit retry must see the same window)."""
    _pre_flush_turns(env)
    _exit_run(env)
    assert env.coord.session_start == env.process_start
    assert env.proc.session_start == env.process_start


def test_coordinator_window_is_forward_only(env):
    later = env.coord.session_start + timedelta(hours=1)
    env.coord.session_start = later
    env.proc.session_start = later
    env.coord.advance_session_window(time.time() - 5 * 3600)  # 5h ago < later
    assert env.coord.session_start == later
    assert env.proc.session_start == later


def test_advance_failure_never_breaks_slot_release(env, monkeypatch):
    def boom(_started):
        raise RuntimeError("clock exploded")

    monkeypatch.setattr(env.coord, "advance_session_window", boom, raising=False)
    _pre_flush_turns(env)
    _flush(env)
    assert env.main._shutdown_requested is False
    assert env.main._shutdown_done.is_set()
