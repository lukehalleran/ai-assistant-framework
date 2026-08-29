"""
Regression tests for the 2026-07-25 pacing-metrics incident.

mark_query_time() ran inside ResponseGenerator.generate_response — AFTER the
prompt (with its [TIME CONTEXT] block) was assembled, and never on agentic
turns (which bypass ResponseGenerator). Result: "Time since last message"
lagged one turn and froze across agentic responses — the live convo showed
"53 m" for a 2-minute gap, twice. The mark now happens once, at handle_submit
ingress, before prompt build, for every mode.
"""

import inspect
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from utils.time_manager import TimeManager


def _bare_tm():
    tm = TimeManager.__new__(TimeManager)
    tm.last_query_time = None
    tm.last_session_end_time = None
    tm.previous_query_time = None
    tm.current_message_time = None
    tm._save_last_query_time = MagicMock()
    tm._register_active_day = MagicMock()
    return tm


class TestMarkSequence:
    def test_first_message_reports_na(self):
        tm = _bare_tm()
        tm.mark_query_time()
        assert tm.time_since_previous_message() == "N/A (first message in session)"

    def test_gap_reflects_current_turn_when_marked_at_ingress(self):
        tm = _bare_tm()
        tm.mark_query_time()
        # Simulate the prior turn having happened 2 minutes ago.
        tm.last_query_time = datetime.now() - timedelta(minutes=2)
        tm.mark_query_time()
        # Marked at ingress, the reported gap is THIS message vs the previous
        # one — minutes, not a stale value.
        assert tm.time_since_previous_message() == "2 m"

    def test_new_session_resets_previous(self):
        tm = _bare_tm()
        tm.last_query_time = datetime.now() - timedelta(hours=5)
        tm.last_session_end_time = datetime.now() - timedelta(hours=4)
        tm.mark_query_time()
        assert tm.time_since_previous_message() == "N/A (first message in session)"


class TestWiring:
    def test_handle_submit_marks_at_ingress(self):
        from gui import handlers
        # handle_submit is now a thin ingress wrapper (2026-08-28 dedupe
        # guard); the dispatcher body lives in _handle_submit_inner.
        src = inspect.getsource(handlers._handle_submit_inner)
        assert "mark_query_time()" in src, (
            "handle_submit must mark pacing at ingress — the [TIME CONTEXT] "
            "block is assembled from this state during prompt build"
        )
        # The mark must come before the prompt-context preparation.
        assert src.index("mark_query_time()") < src.index("_prepare_submit_context")

    def test_response_generator_no_longer_marks(self):
        from core.response_generator import ResponseGenerator
        src = inspect.getsource(ResponseGenerator)
        assert "self.time_manager.mark_query_time()" not in src, (
            "marking in ResponseGenerator is the regression: it runs after "
            "prompt assembly and is skipped entirely by agentic turns"
        )
