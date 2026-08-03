"""
Regression tests for cross-restart tone carryover (2026-08-02).

`ContextPipeline._last_tone_level` made distress sticky within a process, but
Luke's sessions are 1-3 turns with restarts minutes apart — CONCERN detected
at 12:13 was gone by the 12:33 restart, so a flat-semantic continuation turn
had no floor. The pipeline now persists each turn's tone level to
data/tone_state.json (atomic, best-effort) and seeds `_last_tone_level` on
init when the saved level is ELEVATED and within
TONE_STICKINESS_MAX_GAP_MINUTES. Derived, self-healing state: corrupt or
missing files mean a cold start, never an error.
"""

import json
from datetime import datetime, timedelta

import pytest

from core.context_pipeline import ContextPipeline


@pytest.fixture
def state_path(tmp_path, monkeypatch):
    p = tmp_path / "tone_state.json"
    monkeypatch.setattr(ContextPipeline, "_TONE_STATE_PATH", str(p))
    return p


def _pipeline():
    return ContextPipeline(model_manager=None, topic_manager=None)


def _write_state(p, level, age_minutes=0):
    ts = (datetime.now() - timedelta(minutes=age_minutes)).isoformat()
    p.write_text(json.dumps({"level": level, "ts": ts}))


class TestToneCarryover:
    def test_fresh_elevated_tone_carries_across_restart(self, state_path):
        _write_state(state_path, "light_support", age_minutes=5)
        assert _pipeline()._last_tone_level == "light_support"

    def test_crisis_level_enum_encoding_carries(self, state_path):
        _write_state(state_path, "CrisisLevel.CONCERN", age_minutes=5)
        assert _pipeline()._last_tone_level == "CrisisLevel.CONCERN"

    def test_stale_tone_not_carried(self, state_path):
        _write_state(state_path, "light_support", age_minutes=120)
        assert _pipeline()._last_tone_level is None

    def test_conversational_not_carried(self, state_path):
        _write_state(state_path, "conversational", age_minutes=5)
        assert _pipeline()._last_tone_level is None

    def test_missing_file_cold_start(self, state_path):
        assert _pipeline()._last_tone_level is None

    def test_corrupt_file_cold_start_no_raise(self, state_path):
        state_path.write_text("{not json")
        assert _pipeline()._last_tone_level is None

    def test_persist_roundtrip(self, state_path):
        pipe = _pipeline()
        pipe._persist_tone("elevated_support")
        saved = json.loads(state_path.read_text())
        assert saved["level"] == "elevated_support"
        assert _pipeline()._last_tone_level == "elevated_support"
