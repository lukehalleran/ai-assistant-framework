"""Idle Chroma keep-warm loop (incident 2026-10-28, class: BC-41, BC-82).

The loop touches only when nothing is in flight and the last activity is older than
the live interval; interval 0 disables; stop ends the thread. Injected clock/inputs,
no real sleeps beyond tens of ms.
"""
import threading
import time

from memory.storage.chroma_keepwarm import ChromaKeepWarm


class _Env:
    def __init__(self):
        self.now = 1000.0
        self.interval = 600.0
        self.inflight = False
        self.activity = 0.0
        self.touched = []

    def make(self, **kw):
        return ChromaKeepWarm(
            lambda: "store",
            interval_fn=lambda: self.interval,
            inflight_fn=lambda: self.inflight,
            last_activity_fn=lambda: self.activity,
            touch_fn=lambda s: self.touched.append(s),
            clock=lambda: self.now, **kw)


def test_idle_and_due_touches_then_waits_for_the_next_interval():
    e = _Env(); k = e.make()
    assert k.tick() == "not_due"
    e.now += 601
    assert k.tick() == "touched" and e.touched == ["store"]
    assert k.tick() == "not_due"
    e.now += 601
    assert k.tick() == "touched" and len(e.touched) == 2


def test_skips_while_a_turn_is_in_flight_then_resumes():
    e = _Env(); k = e.make(); e.now += 601
    e.inflight = True
    assert k.tick() == "busy" and not e.touched
    e.inflight = False
    assert k.tick() == "touched"


def test_skips_when_the_last_activity_is_recent():
    e = _Env(); k = e.make(); e.now += 601
    e.activity = e.now - 100
    assert k.tick() == "recent_activity" and not e.touched
    e.activity = e.now - 700
    assert k.tick() == "touched"


def test_interval_zero_disables_and_a_live_change_applies_to_the_same_instance():
    e = _Env(); k = e.make(); e.now += 601
    e.interval = 0
    assert k.tick() == "disabled" and not e.touched
    e.interval = 300  # live change, same instance
    assert k.tick() == "touched"


def test_a_failing_touch_does_not_kill_the_loop():
    e = _Env(); k = e.make(); e.now += 601
    k.touch_fn = lambda s: (_ for _ in ()).throw(RuntimeError("x"))
    assert k.tick() == "error"
    e.now += 601
    k.touch_fn = lambda s: e.touched.append(s)
    assert k.tick() == "touched"


def test_thread_runs_and_stops_on_shutdown():
    e = _Env(); e.now = 1000.0
    hit = threading.Event()
    k = ChromaKeepWarm(lambda: "s", interval_fn=lambda: 0.01, inflight_fn=lambda: False,
                       last_activity_fn=lambda: 0.0, touch_fn=lambda s: hit.set(),
                       clock=time.time, poll_s=0.01)
    k.start()
    assert hit.wait(0.2)
    k.stop()
    assert not k._thread.is_alive()
    n = k.passes
    time.sleep(0.05)
    assert k.passes == n


def test_lifecycle_stop_stops_the_singleton(monkeypatch):
    from api.lifecycle import stop_chroma_keepwarm
    from memory.storage import chroma_keepwarm as m
    inst = m.start_keepwarm(lambda: None)
    assert inst is not None and inst._thread.is_alive()
    stop_chroma_keepwarm()
    assert not inst._thread.is_alive() and m._instance is None
