"""
tests/unit/test_tone_stickiness_reset.py

Distress tone is sticky across terse turns WITHIN a session, but must not carry
across a long gap into a fresh session (observed: a CONCERN from a distressed
session hours earlier floored a calm/technical message). The pipeline drops the
carried tone once the gap since the last turn exceeds the threshold.
"""

from datetime import datetime, timedelta

from core.context_pipeline import ContextPipeline
from utils.tone_detector import CrisisLevel, _recent_distress_from_history


def _pipeline_with_carried_tone(tone=CrisisLevel.CONCERN):
    p = ContextPipeline.__new__(ContextPipeline)
    p._last_tone_level = tone
    return p


def _recent(ts):
    return [{"query": "prev", "response": "...", "timestamp": ts}]


def test_long_gap_resets_stickiness():
    p = _pipeline_with_carried_tone()
    two_hours_ago = datetime.now() - timedelta(hours=2)
    assert p._should_reset_tone_stickiness(_recent(two_hours_ago)) is True


def test_within_session_keeps_stickiness():
    p = _pipeline_with_carried_tone()
    two_min_ago = datetime.now() - timedelta(minutes=2)
    assert p._should_reset_tone_stickiness(_recent(two_min_ago)) is False


def test_no_carried_tone_never_resets():
    p = _pipeline_with_carried_tone(tone=None)
    assert p._should_reset_tone_stickiness(_recent(datetime.now() - timedelta(hours=5))) is False


def test_no_recent_memories_treated_as_fresh_session():
    p = _pipeline_with_carried_tone()
    assert p._should_reset_tone_stickiness([]) is True


def test_iso_string_timestamp_parsed():
    p = _pipeline_with_carried_tone()
    old_iso = (datetime.now() - timedelta(hours=3)).isoformat()
    assert p._should_reset_tone_stickiness(_recent(old_iso)) is True
    fresh_iso = (datetime.now() - timedelta(minutes=1)).isoformat()
    assert p._should_reset_tone_stickiness(_recent(fresh_iso)) is False


def test_unparseable_timestamp_fails_closed_keeps_stickiness():
    # On uncertainty, do NOT drop distress mid-session.
    p = _pipeline_with_carried_tone()
    assert p._should_reset_tone_stickiness(_recent("not-a-timestamp")) is False
    assert p._should_reset_tone_stickiness(_recent(None)) is False


def test_threshold_boundary_configurable(monkeypatch):
    import config.app_config as cfg
    monkeypatch.setattr(cfg, "TONE_STICKINESS_MAX_GAP_MINUTES", 30, raising=False)
    p = _pipeline_with_carried_tone()
    just_under = datetime.now() - timedelta(minutes=25)
    just_over = datetime.now() - timedelta(minutes=35)
    assert p._should_reset_tone_stickiness(_recent(just_under)) is False
    assert p._should_reset_tone_stickiness(_recent(just_over)) is True


def test_history_distress_ignores_stale_heavy_turn(monkeypatch):
    import config.app_config as cfg
    monkeypatch.setattr(cfg, "TONE_STICKINESS_MAX_GAP_MINUTES", 30, raising=False)

    stale = {
        "query": "old heavy turn",
        "response": "...",
        "is_heavy_topic": True,
        "timestamp": datetime.now() - timedelta(hours=12),
    }
    fresh = dict(stale, timestamp=datetime.now() - timedelta(minutes=2))

    assert _recent_distress_from_history([stale]) is False
    assert _recent_distress_from_history([fresh]) is True


def test_history_distress_missing_timestamp_fails_closed():
    assert _recent_distress_from_history([{"is_heavy_topic": True}]) is True
