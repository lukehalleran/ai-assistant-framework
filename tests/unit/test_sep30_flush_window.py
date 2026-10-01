"""2026-09-30 (BC-47, BC-30): the flush stamp is the run's START and only moves
on a successful run — a failed run or a turn sent DURING a run is never "flushed".
"""
import pytest

import main


@pytest.fixture
def clock(monkeypatch):
    c = {"t": 100.0}
    monkeypatch.setattr(main.time, "time", lambda: c["t"])
    monkeypatch.setattr(main, "_shutdown_requested", False)
    monkeypatch.setattr(main, "_shutdown_owner_thread", None)
    monkeypatch.setattr(main, "_last_flush_done_at", 50.0)  # an earlier completed flush
    monkeypatch.setattr(main, "_flush_run_started_at", 0.0, raising=False)
    monkeypatch.setattr(main, "_last_activity_time", 60.0)  # activity after that flush
    monkeypatch.setattr(main, "_last_failed_flush_at", 0.0, raising=False)
    yield c
    main._shutdown_requested = False
    main._shutdown_done.set()


def test_failed_run_does_not_mark_flushed(clock):
    assert main._begin_shutdown_run() is True
    clock["t"] = 400.0
    main._end_shutdown_run(success=False)
    assert main._activity_since_last_flush() is True


def test_turn_during_run_is_not_covered(clock):
    main._last_activity_time = 10.0
    assert main._begin_shutdown_run() is True  # t=100
    main._last_activity_time = 150.0           # turn arrives mid-run
    clock["t"] = 400.0
    main._end_shutdown_run(success=True)
    assert main._activity_since_last_flush() is True


def test_quiet_successful_run_marks_flushed(clock):
    main._last_activity_time = 90.0
    assert main._begin_shutdown_run() is True  # t=100
    clock["t"] = 400.0
    main._end_shutdown_run(success=True)
    assert main._activity_since_last_flush() is False


def test_second_begin_does_not_move_start(clock):
    assert main._begin_shutdown_run() is True
    started = main._flush_run_started_at
    clock["t"] = 300.0
    assert main._begin_shutdown_run() is False
    assert main._flush_run_started_at == started
    main._end_shutdown_run(success=False)


def test_fail_flag_path_used_by_callers(clock):
    main._last_activity_time = 90.0
    assert main._begin_shutdown_run() is True
    main._fail_shutdown_run()
    main._end_shutdown_run()
    assert main._last_flush_done_at == 50.0


def test_idle_monitor_does_not_retry_a_failed_run_until_new_activity(clock, monkeypatch):
    monkeypatch.setattr(main, "_idle_timeout_minutes", 60)
    main._last_activity_time = 90.0
    assert main._idle_flush_due(120) is True
    assert main._begin_shutdown_run() is True  # t=100
    clock["t"] = 400.0
    main._end_shutdown_run(success=False)
    assert main._idle_flush_due(120) is False   # no retry storm while idle
    main._last_activity_time = 500.0            # user comes back
    assert main._idle_flush_due(120) is True
