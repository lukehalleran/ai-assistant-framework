"""Event-loop stall detector (2026-10-15, class: BC-41 / BC-69).

Live evidence: a ``wait_for(…, 3.0)`` "took" 8.4 s without timing out -- the
loop itself was blocked by inline synchronous work. ``utils/loop_stall.py``
names the blocking frame; these tests drive the deployed detector, and the
deployed ``build_prompt`` gather with a fake store whose query sleeps
synchronously. Fixtures are synthetic.
"""

import asyncio
import json
import threading
import time

import pytest

from utils.loop_stall import LoopStallDetector
from tests.unit.test_independent_prompt_audit import full_builder, retrieval_limits


def _blocking_work():
    time.sleep(0.4)


async def _blocker():
    _blocking_work()


@pytest.mark.asyncio
async def test_inline_sleep_is_one_stall_naming_the_frame():
    async with LoopStallDetector(threshold_ms=250, enabled=True) as det:
        await _blocker()
        await asyncio.sleep(0.15)
    recs = det.records()
    assert len(recs) == 1, recs
    assert recs[0]["ms"] >= 300
    assert "test_loop_stall_detector.py" in recs[0]["top"]
    assert "_blocking_work" in recs[0]["top"]
    assert any("_blocker" in f for f in recs[0]["frames"])


@pytest.mark.asyncio
async def test_cooperative_sleep_is_not_a_stall():
    async with LoopStallDetector(threshold_ms=250, enabled=True) as det:
        await asyncio.sleep(0.4)
    assert det.records() == []


@pytest.mark.asyncio
async def test_disabled_starts_no_thread_and_records_nothing():
    before = {t.name for t in threading.enumerate()}
    async with LoopStallDetector(enabled=False) as det:
        assert "loop-stall-watch" not in {t.name for t in threading.enumerate()} - before
        await _blocker()
    assert det.records() == [] and det.turn_fields() == []


@pytest.mark.asyncio
async def test_disabled_by_live_config(monkeypatch):
    monkeypatch.setattr("config.app_config.LOOP_STALL_DETECTOR_ENABLED", False)
    async with LoopStallDetector() as det:
        await _blocker()
    assert det.records() == []
    monkeypatch.setattr("config.app_config.LOOP_STALL_DETECTOR_ENABLED", True)
    monkeypatch.setattr("config.app_config.LOOP_STALL_THRESHOLD_MS", 250)
    async with LoopStallDetector() as det2:
        await _blocker()
        await asyncio.sleep(0.1)
    assert len(det2.records()) == 1


@pytest.mark.asyncio
async def test_thread_stops_even_when_body_raises():
    det = LoopStallDetector(threshold_ms=250, enabled=True)
    with pytest.raises(RuntimeError):
        async with det:
            assert any(t.name == "loop-stall-watch" for t in threading.enumerate())
            raise RuntimeError("boom")
    await asyncio.sleep(0.05)
    assert not any(t.name == "loop-stall-watch" and t.is_alive() for t in threading.enumerate())


@pytest.mark.asyncio
async def test_records_capped_at_five_sorted_and_content_free():
    secret = "SYNTHETIC-SECRET-MESSAGE-TEXT"

    async def block(seconds, text):
        local_text = text  # noqa: F841 -- a local that must never be captured
        time.sleep(seconds)

    async with LoopStallDetector(threshold_ms=100, enabled=True) as det:
        for i in range(7):
            await block(0.15 + 0.03 * i, secret)
            await asyncio.sleep(0.12)
    recs = det.turn_fields()
    assert len(recs) == 5
    assert [r["ms"] for r in recs] == sorted((r["ms"] for r in recs), reverse=True)
    for r in recs:
        assert set(r) == {"ms", "top", "entry"}
    assert secret not in json.dumps(det.records())


def test_turn_record_carries_only_numbers_and_locations(tmp_path, monkeypatch):
    from utils import turn_telemetry
    path = tmp_path / "turns.jsonl"
    monkeypatch.setattr("config.app_config.TURN_TELEMETRY_PATH", str(path))
    monkeypatch.setattr("config.app_config.TURN_TELEMETRY_ENABLED", True)
    stalls = [{"ms": 100 + i, "top": f"core/x.py:{i} f", "message": "LEAK-TEXT", "frames": ["LEAK"]}
              for i in range(8)]
    assert turn_telemetry.record_turn({"query": "q", "loop_stalls": stalls})
    row = json.loads(path.read_text().splitlines()[-1])
    assert len(row["loop_stalls"]) == 5
    assert [s["ms"] for s in row["loop_stalls"]] == [107, 106, 105, 104, 103]
    assert all(set(s) == {"ms", "top"} for s in row["loop_stalls"])
    with_entry = [{"ms": 9, "top": "core/x.py:1 f", "entry": "core/y.py:2 g", "frames": ["LEAK"]}]
    assert turn_telemetry.record_turn({"query": "q", "loop_stalls": with_entry})
    row2 = json.loads(path.read_text().splitlines()[-1])
    assert row2["loop_stalls"] == [{"ms": 9, "top": "core/x.py:1 f", "entry": "core/y.py:2 g"}]
    assert "LEAK" not in json.dumps(row)


@pytest.mark.asyncio
async def test_deployed_build_prompt_gather_names_the_blocking_store(monkeypatch):
    """The real build_prompt gather, with a fake store whose query sleeps synchronously."""
    builder = full_builder(monkeypatch, [])

    class _SleepyStore:
        def query(self):
            time.sleep(0.45)
            return []

    store = _SleepyStore()

    async def gather_task(*args, **kwargs):
        return store.query()  # synchronous store call inline on the loop

    builder.context_gatherer._get_recent_conversations = gather_task
    result = await asyncio.wait_for(
        builder.build_prompt("Synthetic question", retrieval_overrides=retrieval_limits()), 10)
    assert "_build_time" in result
    stalls = result["_loop_stalls"]
    assert stalls, "gather blocked the loop for 450 ms; the detector must have named it"
    assert stalls[0]["ms"] >= 350
    assert "test_loop_stall_detector.py" in stalls[0]["top"]
    assert set(stalls[0]) == {"ms", "top", "entry"}


@pytest.mark.asyncio
async def test_top_is_the_repo_frame_when_the_block_is_inside_a_library(tmp_path):
    """Referee 2026-10-15: a block deep inside library code (torch import, chroma)
    must still name the innermost REPO frame as `top`, not the library frame."""
    import importlib.util
    lib = tmp_path / "fake_heavy_lib.py"
    lib.write_text(
        "import time\n"
        "def deep(n):\n"
        "    return deep(n - 1) if n else _spin()\n"
        "def _spin():\n"
        "    end = time.monotonic() + 0.4\n"
        "    while time.monotonic() < end:\n"
        "        pass\n"
    )
    spec = importlib.util.spec_from_file_location("fake_heavy_lib", lib)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def _repo_caller():
        mod.deep(20)  # 20 library frames sit between this frame and the block

    async with LoopStallDetector(threshold_ms=150, enabled=True) as det:
        _repo_caller()
        await asyncio.sleep(0.1)
    recs = det.records()
    assert recs, "stall not recorded"
    assert "test_loop_stall_detector.py" in recs[0]["top"]
    assert "_repo_caller" in recs[0]["top"]
    assert "fake_heavy_lib.py" in recs[0]["frames"][0]


@pytest.mark.asyncio
async def test_record_names_the_outermost_repo_frame_as_entry():
    """2026-10-16, BC-41: a library blocks under repo frame B called from repo frame A.
    top = B (innermost repo frame), entry = A (outermost repo frame)."""
    import utils.loop_stall as ls

    class _Code:
        def __init__(self, path, name):
            self.co_filename, self.co_name = path, name

    class _Frame:
        def __init__(self, path, line, name, back=None):
            self.f_code, self.f_lineno, self.f_back = _Code(path, name), line, back

    root = ls._ROOT
    outer = _Frame(root + "/gui/launch.py", 10, "frame_a")           # repo, outermost
    mid = _Frame(root + "/core/orchestrator.py", 20, "frame_b", outer)  # repo
    lib = _Frame("/x/site-packages/httpx/_client.py", 30, "send", mid)  # library, innermost

    det = LoopStallDetector(threshold_ms=100, enabled=True)
    async with det:
        real = ls.sys._current_frames
        ls.sys._current_frames = lambda: {det._loop_tid: lib}
        try:
            time.sleep(0.3)  # genuine stall so the deployed watcher samples
            await asyncio.sleep(0.15)
        finally:
            ls.sys._current_frames = real
    rec = det.records()[0]
    assert rec["top"].split(" ")[0] == "core/orchestrator.py" + chr(58) + "20"
    assert rec["entry"].split(" ")[0] == "gui/launch.py" + chr(58) + "10"
    assert det.turn_fields()[0]["entry"] == rec["entry"]
