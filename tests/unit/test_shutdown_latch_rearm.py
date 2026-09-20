"""Shutdown-coordination re-arming (2026-09-18 incident, fixed 2026-09-19).

`_shutdown_requested` used to be a permanent one-shot latch shared by the idle
monitor's session flush and the real exit-time shutdown: one idle flush (a)
ended `_idle_monitor_thread`'s loop for the life of the process and (b) made
`run_shutdown_tasks_async` (FastAPI lifespan exit) return silently — turns
after the first idle hour got no session reflection / shutdown fact pass /
backup (observed 2026-09-18: flush at 16:05, turns at 17:13-17:24, silent
exit next day).

The fix makes `_shutdown_requested` true only WHILE a run is in flight
(`_begin_shutdown_run()` / `_end_shutdown_run()`), adds `_process_exiting`
(ends the idle loop) and `_exit_shutdown_handled` (tells the legacy `finally`
block an exit path already handled the tasks), and gates both the idle
monitor and the exit paths on `_activity_since_last_flush()` so a run only
happens again when a genuinely new turn arrived — logged, never silent, when
it is skipped.

These tests drive the DEPLOYED functions in the `main` module (not a
source-text check — see tests/unit/test_process_signals.py's
TestMainWiring.test_second_shutdown_entrant_waits_for_inflight_run, whose
source-shape assertions are this file's non-behavioural sibling; class: BC-63).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from types import SimpleNamespace

import pytest


@pytest.fixture
def main_mod(monkeypatch):
    """Fresh shutdown-coordination state for one test, driving the real
    `main` module. `_do_shutdown_async`/`_mark_session_end`/
    `_gather_session_state` are stubbed so no real shutdown processing
    (reflection, fact extraction, backups, data/ writes) ever runs."""
    import main

    calls = []

    async def fake_do_shutdown_async(orchestrator, session_convos, session_summaries):
        calls.append((orchestrator, session_convos, session_summaries))

    monkeypatch.setattr(main, "_do_shutdown_async", fake_do_shutdown_async)
    monkeypatch.setattr(main, "_mark_session_end", lambda orchestrator: None)
    monkeypatch.setattr(main, "_gather_session_state", lambda orchestrator: ([], []))

    monkeypatch.setattr(main, "_shutdown_requested", False)
    monkeypatch.setattr(main, "_process_exiting", False)
    monkeypatch.setattr(main, "_exit_shutdown_handled", False)
    monkeypatch.setattr(main, "_last_flush_done_at", 0.0)
    monkeypatch.setattr(main, "_last_activity_time", time.time())
    monkeypatch.setattr(main, "_orchestrator_ref", None)
    monkeypatch.setattr(main, "_shutdown_owner_thread", None)

    was_set = main._shutdown_done.is_set()
    try:
        yield main, calls
    finally:
        # `_shutdown_done` is a threading.Event mutated in place (.set()/.clear()),
        # not reassigned, so monkeypatch's attribute-restore does not cover it.
        if was_set:
            main._shutdown_done.set()
        else:
            main._shutdown_done.clear()


# ---------------------------------------------------------------------------
# T1 — the incident itself: a completed idle flush must not block a later
# exit-time run when a turn arrived after it.
# ---------------------------------------------------------------------------

def test_exit_run_happens_after_completed_flush_with_new_activity(main_mod):
    main, calls = main_mod
    orch = SimpleNamespace()

    main._run_shutdown_tasks(orch)
    assert len(calls) == 1, "the idle flush itself must run"
    assert main._shutdown_requested is False
    assert main._last_flush_done_at > 0

    main._last_activity_time = time.time() + 1  # a turn arrived after the flush

    asyncio.run(main.run_shutdown_tasks_async(orch))
    assert len(calls) == 2, "exit-time shutdown must still run when there was new activity"


# ---------------------------------------------------------------------------
# T2 — no new activity since the last completed run: the exit path skips,
# and says so (never silently).
# ---------------------------------------------------------------------------

def test_exit_run_skips_and_logs_when_no_new_activity(main_mod, caplog):
    main, calls = main_mod
    orch = SimpleNamespace()

    main._run_shutdown_tasks(orch)
    assert len(calls) == 1

    main._last_activity_time = main._last_flush_done_at - 5  # nothing new since the flush

    with caplog.at_level(logging.INFO, logger="main"):
        asyncio.run(main.run_shutdown_tasks_async(orch))

    assert len(calls) == 1, "no new activity: the exit path must not re-run the LLM sequence"
    assert any("nothing new to process" in r.message for r in caplog.records)
    assert main._exit_shutdown_handled is True
    assert main._shutdown_requested is False


# ---------------------------------------------------------------------------
# T3 — an in-flight run (e.g. the idle monitor's own thread) is awaited by a
# concurrent exit-time entrant, never duplicated and never cut off.
# ---------------------------------------------------------------------------

async def test_inflight_run_is_awaited_not_duplicated(main_mod, monkeypatch):
    main, calls = main_mod
    orch = SimpleNamespace()

    entered = threading.Event()
    blocker = threading.Event()

    async def blocking_do_shutdown(orchestrator, session_convos, session_summaries):
        entered.set()
        await asyncio.to_thread(blocker.wait)
        calls.append((orchestrator, session_convos, session_summaries))

    monkeypatch.setattr(main, "_do_shutdown_async", blocking_do_shutdown)
    monkeypatch.setattr(main, "_SHUTDOWN_INFLIGHT_WAIT_S", 3.0)

    worker = threading.Thread(target=main._run_shutdown_tasks, args=(orch,), daemon=True)
    worker.start()
    try:
        assert entered.wait(timeout=2), "background run never started"
        assert main._shutdown_requested is True

        def _release():
            time.sleep(0.2)
            blocker.set()

        timer = threading.Thread(target=_release, daemon=True)
        timer.start()

        start = time.time()
        await main.run_shutdown_tasks_async(orch)
        elapsed = time.time() - start
        timer.join(timeout=2)
    finally:
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(calls) == 1, "the in-flight run must not be duplicated"
    assert elapsed >= 0.1, "the exit entrant must have waited for the in-flight run, not returned early"
    assert main._shutdown_requested is False


# ---------------------------------------------------------------------------
# T4 — the idle monitor re-arms: it must keep looping after its own flush
# (only the PROCESS exiting ends the loop) and fire again once there is
# genuinely new activity newer than the last completed flush.
# ---------------------------------------------------------------------------

def test_idle_monitor_rearms_after_its_own_flush(main_mod, monkeypatch):
    import gui.handlers as handlers

    main, _ = main_mod
    monkeypatch.setattr(main, "_idle_timeout_minutes", 0)
    monkeypatch.setattr(main, "_orchestrator_ref", object())
    monkeypatch.setattr(handlers, "has_inflight_turns", lambda max_age_s=None: False)

    run_count = []

    def fake_run_shutdown_tasks(orchestrator):
        assert main._begin_shutdown_run() is True
        run_count.append(len(run_count) + 1)
        main._end_shutdown_run()

    monkeypatch.setattr(main, "_run_shutdown_tasks", fake_run_shutdown_tasks)

    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(main.time, "time", lambda: clock["t"])
    monkeypatch.setattr(main, "_last_activity_time", clock["t"] - 3600)  # far in the past

    sleeps = []

    def tick(*_args):
        sleeps.append(1)
        n = len(sleeps)
        if n == 3:
            # New activity, newer than the flush tick 1 produced, and the
            # clock has moved on: this must fire again.
            clock["t"] = main._last_flush_done_at + 200
            main._last_activity_time = main._last_flush_done_at + 50
        elif n == 4:
            # Only the PROCESS exiting ends the loop — never the flush itself.
            main._process_exiting = True

    monkeypatch.setattr(main.time, "sleep", tick)

    main._idle_monitor_thread()

    assert run_count == [1, 2], "tick 1 must fire, tick 2 (no new activity) must not, tick 3 must re-fire"
    assert len(sleeps) == 4


# ---------------------------------------------------------------------------
# T5 — _signal_handler shares the same activity-gated skip.
# ---------------------------------------------------------------------------

def _raise_system_exit(code=0):
    raise SystemExit(code)


def test_signal_handler_runs_shutdown_with_new_activity(main_mod, monkeypatch):
    main, calls = main_mod
    orch = SimpleNamespace()
    monkeypatch.setattr(main, "_orchestrator_ref", orch)
    monkeypatch.setattr(main.sys, "exit", _raise_system_exit)

    ran = []
    monkeypatch.setattr(main, "_run_shutdown_tasks", lambda orchestrator: ran.append(orchestrator))
    main._last_activity_time = time.time()  # newer than _last_flush_done_at (0.0 -> always "new")

    with pytest.raises(SystemExit):
        main._signal_handler(15, None)

    assert ran == [orch]
    assert main._process_exiting is True
    assert main._exit_shutdown_handled is True


def test_signal_handler_skips_without_new_activity(main_mod, monkeypatch, caplog):
    main, calls = main_mod
    orch = SimpleNamespace()
    monkeypatch.setattr(main, "_orchestrator_ref", orch)
    monkeypatch.setattr(main.sys, "exit", _raise_system_exit)

    ran = []
    monkeypatch.setattr(main, "_run_shutdown_tasks", lambda orchestrator: ran.append(orchestrator))
    main._last_flush_done_at = time.time()
    main._last_activity_time = main._last_flush_done_at - 10  # nothing new since the flush

    with caplog.at_level(logging.INFO, logger="main"):
        with pytest.raises(SystemExit):
            main._signal_handler(15, None)

    assert ran == []
    assert any("nothing new to process" in r.message for r in caplog.records)
    assert main._process_exiting is True
    assert main._exit_shutdown_handled is True


# ---------------------------------------------------------------------------
# T6 — the state helpers themselves.
# ---------------------------------------------------------------------------

def test_begin_end_shutdown_run_state_helpers(main_mod):
    main, _ = main_mod

    assert main._begin_shutdown_run() is True
    assert main._shutdown_requested is True
    assert main._shutdown_owner_thread == threading.get_ident()
    assert main._shutdown_done.is_set() is False
    assert main._begin_shutdown_run() is False, "a second claim while one is in flight must fail"

    main._end_shutdown_run()
    assert main._shutdown_requested is False
    assert main._shutdown_owner_thread is None
    assert main._shutdown_done.is_set() is True
    assert main._last_flush_done_at > 0
