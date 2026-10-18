"""Long loop stalls name what the other threads were running.

Incident 2026-10-18, class BC-41 / BC-69 (diagnostics): live stalls of 3858 ms and
2211 ms showed the loop thread parked in threading.wait with no repo frame -- a GIL
holder on ANOTHER thread, invisible to a detector that samples only the loop thread.
"""
import asyncio
import logging
import threading
import time

import pytest

import utils.loop_stall as loop_stall
from utils.loop_stall import LoopStallDetector


def _helper_parked_here(evt):
    evt.wait(10)


@pytest.fixture
def parked_thread():
    evt = threading.Event()
    t = threading.Thread(target=_helper_parked_here, args=(evt,), name="rowan-helper", daemon=True)
    t.start()
    time.sleep(0.05)
    yield t
    evt.set()
    t.join(2)


async def _stall(det_kwargs, secs=0.5):
    async with LoopStallDetector(**det_kwargs) as det:
        time.sleep(secs)
        await asyncio.sleep(0.15)
    return det


@pytest.mark.asyncio
async def test_long_stall_names_other_thread_function(monkeypatch, parked_thread, caplog):
    monkeypatch.setattr(loop_stall, "OTHER_THREADS_MIN_MS", 200)
    with caplog.at_level(logging.WARNING):
        det = await _stall(dict(threshold_ms=100, enabled=True))
    recs = det.records()
    assert len(recs) == 1
    other = recs[0]["other_threads"]
    assert "rowan-helper" in other and "_helper_parked_here" in other
    assert "loop-stall-watch" not in other
    assert det.turn_fields()[0]["other_threads"] == other
    assert any("| other threads:" in r.getMessage() and "_helper_parked_here" in r.getMessage()
               for r in caplog.records)


@pytest.mark.asyncio
async def test_short_stall_has_no_other_threads_field(monkeypatch, parked_thread):
    monkeypatch.setattr(loop_stall, "OTHER_THREADS_MIN_MS", 5000)
    det = await _stall(dict(threshold_ms=100, enabled=True))
    recs = det.records()
    assert len(recs) == 1
    assert "other_threads" not in recs[0]
    assert "other_threads" not in det.turn_fields()[0]


def test_telemetry_sanitizer_keeps_other_threads():
    from utils.turn_telemetry import _sanitize_loop_stalls
    out = _sanitize_loop_stalls([{"ms": 1500, "top": "a.py:1 f", "other_threads": "t: x.py:2 g"}])
    assert out[0]["other_threads"] == "t: x.py:2 g"
    assert "other_threads" not in _sanitize_loop_stalls([{"ms": 1, "top": "a"}])[0]
