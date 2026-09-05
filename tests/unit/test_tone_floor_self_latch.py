"""
Distress-sticky-floor self-latch (2026-08-22).

The floor's own CONCERN output fed _last_tone_level AND data/tone_state.json,
so one latch chained indefinitely: every short technical message all afternoon
got LIGHT SUPPORT ("pull up the veto logic" → "let them vent"). Telemetry
showed trigger=distress_sticky_floor with constant conf=0.43 from 10:41
through 14:36. Two feedback edges are now cut:

  1. Floor-produced levels never SEED across a restart (tone_state.json
     carries the trigger; _load_persisted_tone skips floor-triggered levels).
  2. In-process, the floor may chain at most TONE_FLOOR_CHAIN_MAX (3)
     consecutive turns; after that previous_tone is withheld and fresh
     organic evidence (semantic/keyword/arbiter, or heavy recent history)
     is required to re-elevate. The 07-21 anti-flatline behavior (short-turn
     distress spirals) is preserved within the chain budget.

Plus: _recent_distress_from_history no longer slices [-window:] of a
NEWEST-first list (it was examining the OLDEST rows of the fetch), and
tests/conftest.py sandboxes tone_state.json + turn_records.jsonl (test runs
had written light_support rows into PROD telemetry and seeded the live
daemon's floor across the owner's restart).
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from core.context_pipeline import ContextPipeline


def _bare_pipeline(tmp_path, state=None):
    pipe = object.__new__(ContextPipeline)
    p = tmp_path / "tone_state.json"
    if state is not None:
        p.write_text(json.dumps(state))
    # per-instance override (conftest also patches the class attr)
    pipe._TONE_STATE_PATH = str(p)
    return pipe


class TestSeedTriggerAwareness:
    def test_floor_triggered_level_never_seeds(self, tmp_path):
        pipe = _bare_pipeline(tmp_path, {
            "level": "light_support",
            "trigger": "distress_sticky_floor",
            "ts": datetime.now().isoformat(),
        })
        assert pipe._load_persisted_tone() is None

    def test_organic_elevation_still_seeds(self, tmp_path):
        pipe = _bare_pipeline(tmp_path, {
            "level": "light_support",
            "trigger": "semantic",
            "ts": datetime.now().isoformat(),
        })
        assert pipe._load_persisted_tone() == "light_support"

    def test_legacy_state_without_trigger_still_seeds(self, tmp_path):
        # pre-fix files have no trigger field — treat as organic (safe side)
        pipe = _bare_pipeline(tmp_path, {
            "level": "elevated_support",
            "ts": datetime.now().isoformat(),
        })
        assert pipe._load_persisted_tone() == "elevated_support"

    def test_stale_state_never_seeds(self, tmp_path):
        pipe = _bare_pipeline(tmp_path, {
            "level": "light_support",
            "trigger": "semantic",
            "ts": (datetime.now() - timedelta(hours=3)).isoformat(),
        })
        assert pipe._load_persisted_tone() is None

    def test_persist_writes_trigger(self, tmp_path):
        pipe = _bare_pipeline(tmp_path)
        pipe._persist_tone("light_support", trigger="distress_sticky_floor")
        state = json.loads(Path(pipe._TONE_STATE_PATH).read_text())
        assert state["trigger"] == "distress_sticky_floor"
        assert state["level"] == "light_support"


class TestFloorChainBudget:
    def test_chain_counter_source_wiring(self):
        # the pipeline must count consecutive floor triggers and withhold
        # previous_tone once the budget is exhausted
        import inspect
        src = inspect.getsource(ContextPipeline._detect_tone)
        assert "TONE_FLOOR_CHAIN_MAX" in src
        assert "_floor_chain" in src
        assert "distress_sticky_floor" in src


class TestHistoryScanOrderAgnostic:
    def test_newest_first_heavy_row_detected(self):
        from utils.tone_detector import _recent_distress_from_history
        now = datetime.now()
        newest_first = [
            {"query": "my short msg", "timestamp": now.isoformat(), "is_heavy_topic": True},
            {"query": "older", "timestamp": (now - timedelta(minutes=5)).isoformat()},
            {"query": "older2", "timestamp": (now - timedelta(minutes=8)).isoformat()},
            {"query": "older3", "timestamp": (now - timedelta(minutes=12)).isoformat()},
            {"query": "older4", "timestamp": (now - timedelta(minutes=15)).isoformat()},
        ]
        # pre-fix, [-3:] examined only the oldest 3 rows and MISSED the newest
        # heavy row on a newest-first list
        assert _recent_distress_from_history(newest_first) is True

    def test_stale_heavy_rows_still_excluded(self):
        from utils.tone_detector import _recent_distress_from_history
        old = (datetime.now() - timedelta(hours=4)).isoformat()
        rows = [{"query": "political rant", "timestamp": old, "is_heavy_topic": True}]
        assert _recent_distress_from_history(rows) is False


class TestStateSandbox:
    def test_conftest_sandboxes_tone_state_and_telemetry(self):
        src = Path("tests/conftest.py").read_text()
        assert "_TONE_STATE_PATH" in src
        assert "TURN_TELEMETRY_PATH" in src

    def test_tone_state_path_is_sandboxed_here(self):
        # the autouse fixture must have repointed the class attr off data/
        assert not str(ContextPipeline._TONE_STATE_PATH).startswith("data/")
