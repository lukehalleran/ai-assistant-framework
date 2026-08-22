"""STM gate: recent-history override for the in-process depth counter.

Regression for 2026-08-05: `_conversation_depth` resets on every restart, so
the first `stm_min_depth` messages after a mid-day restart ran WITHOUT the
[SHORT-TERM CONTEXT SUMMARY] — no temporal_facts, no restate warning —
exactly when the model reconstructs "where were we" across a gap. A
first-message agentic answer asserted "three-plus days of documented pain"
for a ~2-day episode. With a recent corpus conversation available, STM now
runs regardless of process depth.
"""
from datetime import datetime, timedelta

from core.context_pipeline import ContextPipeline


def _pipeline(depth=0, min_depth=3, use_stm=True, analyzer=object()):
    p = ContextPipeline.__new__(ContextPipeline)
    p._use_stm = use_stm
    p.stm_analyzer = analyzer
    p._conversation_depth = depth
    p._stm_min_depth = min_depth
    return p


def _history(hours_ago):
    ts = (datetime.now() - timedelta(hours=hours_ago)).isoformat()
    return [{"query": "q", "response": "r", "timestamp": ts}]


class TestShouldRunStm:
    def test_depth_met_runs(self):
        assert _pipeline(depth=5)._should_run_stm(None)

    def test_fresh_process_no_history_skips(self):
        assert not _pipeline(depth=1)._should_run_stm(None)
        assert not _pipeline(depth=1)._should_run_stm([])

    def test_fresh_process_recent_history_runs(self):
        # The live miss: restart at 18:46, message at 18:48, last conversation
        # 4.6h earlier — same-day continuity, STM must run.
        assert _pipeline(depth=1)._should_run_stm(_history(hours_ago=4.6))

    def test_fresh_process_stale_history_skips(self):
        assert not _pipeline(depth=1)._should_run_stm(_history(hours_ago=12))

    def test_disabled_never_runs(self):
        assert not _pipeline(depth=9, use_stm=False)._should_run_stm(_history(1))
        assert not _pipeline(depth=9, analyzer=None)._should_run_stm(_history(1))


class TestHasRecentHistory:
    def test_garbage_timestamps_are_stale(self):
        assert not ContextPipeline._has_recent_history(
            [{"timestamp": "not-a-date"}, {"timestamp": None}, {}]
        )

    def test_newest_entry_wins(self):
        entries = _history(hours_ago=48) + _history(hours_ago=2)
        assert ContextPipeline._has_recent_history(entries)

    def test_datetime_objects_accepted(self):
        entries = [{"timestamp": datetime.now() - timedelta(hours=1)}]
        assert ContextPipeline._has_recent_history(entries)

    def test_boundary_respects_max_gap(self):
        assert not ContextPipeline._has_recent_history(
            _history(hours_ago=7), max_gap_hours=6.0
        )
        assert ContextPipeline._has_recent_history(
            _history(hours_ago=5), max_gap_hours=6.0
        )
