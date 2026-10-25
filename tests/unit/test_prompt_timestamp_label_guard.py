"""
Prompt timestamp labels carry the item's AGE (supply side of BC-103).

Incident 2026-10-25 16:07 (kimi-3): the previous message was sent at 16:06; the
next reply said "how wrecked you were feeling an hour ago". [RECENT CONVERSATION]
rendered the item as ``16:06 (today)`` and [TIME CONTEXT] said ``Time since last
message: 1 m`` in a separate, unbound line, so the model did the arithmetic
itself and got it wrong.

class: BC-103 (a quantity the code already knows is left for the model to
compute), BC-46.

(a) Fixture tests drive the DEPLOYED renderers (PromptFormatter, AgenticFormatter,
    format_relative_timestamp) with a frozen clock and assert the age label for
    items 1 min / 50 min / 2 h 10 min / 5 h / 1 day / 40 days old, in clean AND
    indented/wrapped fixture form.
(b) Source ratchet (same shape as test_env_read_guard.py): every ``strftime(``
    call in core/prompt/*.py and core/agentic/*.py must be on the allowlist below
    with a reason; the count may not grow, and the ceiling may not be left slack.
"""
from __future__ import annotations

import ast
from datetime import datetime as _real_datetime
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

NOW = _real_datetime(2026, 10, 25, 18, 0, 0)


class _Meta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, _real_datetime)


class _FrozenDT(_real_datetime, metaclass=_Meta):
    @classmethod
    def now(cls, tz=None):
        return NOW


@pytest.fixture
def frozen_clock(monkeypatch):
    import utils.time_manager as tm
    monkeypatch.setattr(tm, "datetime", _FrozenDT)
    return NOW


CASES = [
    (timedelta(minutes=1), "(today, 1 min ago)"),
    (timedelta(minutes=50), "(today, 50 min ago)"),
    (timedelta(hours=2, minutes=10), "(today, 2 h 10 min ago)"),
    (timedelta(hours=5), "(today, 5 h ago)"),
    (timedelta(days=1), "(yesterday)"),
    (timedelta(days=40), "(1 month ago)"),
]


# ---------------------------------------------------------------------------
# (a) fixtures
# ---------------------------------------------------------------------------

YNOW = _real_datetime(2026, 10, 25, 0, 10, 0)


class TestPreviousDayAges:
    @pytest.mark.parametrize("ts,label", [
        (_real_datetime(2026, 10, 24, 23, 30), "(yesterday, 40 min ago)"),
        (_real_datetime(2026, 10, 24, 7, 40), "(yesterday, 16 h ago)"),
        (_real_datetime(2026, 10, 24, 0, 10), "(yesterday)"),
        (_real_datetime(2026, 10, 24, 0, 11), "(yesterday, 23 h ago)"),
    ])
    def test_yesterday(self, ts, label):
        from utils.time_manager import format_relative_timestamp
        assert format_relative_timestamp(ts, YNOW).endswith(label)

    def test_tomorrow(self):
        from utils.time_manager import format_relative_timestamp
        now = _real_datetime(2026, 10, 25, 20, 0)
        assert format_relative_timestamp(_real_datetime(2026, 10, 26, 5, 0), now).endswith("(tomorrow, in 9 h)")
        assert format_relative_timestamp(_real_datetime(2026, 10, 26, 20, 0), now).endswith("(tomorrow)")

    @pytest.mark.parametrize("text", ["felt rough", "  felt rough\n    wrapped"])
    def test_renderers_clean_and_indented(self, text):
        from core.agentic.formatters import AgenticFormatter
        ts = "2026-10-24T23:30:00"
        out = AgenticFormatter().format_recent_conversations([{"timestamp": ts, "query": text, "response": "ok"}], now=YNOW)
        assert "(yesterday, 40 min ago)" in out

    def test_session_header_collapses_to_day_label(self, monkeypatch):
        import utils.time_manager as tm
        monkeypatch.setattr(tm, "datetime", type("F", (_real_datetime,), {"now": classmethod(lambda c, tz=None: YNOW)}))
        from core.prompt.formatter import _format_session_header
        out = _format_session_header(_real_datetime(2026, 10, 24, 23, 30))
        assert "Yesterday" in out and "min ago" not in out


class TestHelperLabels:
    @pytest.mark.parametrize("delta,label", CASES)
    def test_label_carries_age(self, delta, label):
        from utils.time_manager import format_relative_timestamp
        ts = NOW - delta
        assert format_relative_timestamp(ts, NOW).endswith(label)

    @pytest.mark.parametrize("seconds,label", [(0, "just now"), (59, "just now"), (3 * 3600 + 59 * 60, "3 h ago")])
    def test_edges(self, seconds, label):
        from utils.time_manager import format_relative_timestamp
        out = format_relative_timestamp(NOW - timedelta(seconds=seconds), NOW)
        assert f"(today, {label}" in out

    def test_future_timestamps(self):
        from utils.time_manager import format_relative_timestamp
        assert format_relative_timestamp(NOW + timedelta(minutes=45), NOW).endswith("(today, in 45 min)")
        assert format_relative_timestamp(NOW + timedelta(hours=2), NOW).endswith("(today, in 2 h)")
        assert format_relative_timestamp(NOW + timedelta(days=1), NOW).endswith("(tomorrow)")
        assert format_relative_timestamp(NOW + timedelta(days=3), NOW).endswith("(in 3 days)")
        assert "-" not in format_relative_timestamp(NOW + timedelta(days=3), NOW).split("(")[1]

    def test_date_only_value_keeps_day_granularity(self):
        from utils.time_manager import render_timestamp_label
        assert render_timestamp_label("2026-10-25", NOW) == "2026-10-25 (today)"
        assert render_timestamp_label("2026-10-10", NOW) == "2026-10-10 (2w ago)"
        assert render_timestamp_label("not a date", NOW) == "not a date"
        assert render_timestamp_label("", NOW) == ""

    def test_aware_datetime_is_converted_not_stripped(self):
        from datetime import timezone
        from utils.time_manager import format_relative_timestamp
        aware = (NOW - timedelta(minutes=5)).astimezone(timezone.utc)
        assert format_relative_timestamp(aware, NOW).endswith("(today, 5 min ago)")


def _formatter():
    from core.prompt.formatter import PromptFormatter
    tm = MagicMock()
    tm.count_tokens = MagicMock(return_value=10)
    return PromptFormatter(token_manager=tm, time_manager=None)


class TestDeployedRenderers:
    @pytest.mark.parametrize("delta,label", CASES)
    def test_prompt_formatter_memory(self, frozen_clock, delta, label):
        ts = (NOW - delta).isoformat()
        for text in ("felt rough", "  felt rough\n    still rough"):  # clean + indented
            out = _formatter()._format_memory({"query": text, "response": "ok", "timestamp": ts})
            assert label in out, out

    @pytest.mark.parametrize("delta,label", CASES)
    def test_agentic_renderers(self, frozen_clock, delta, label):
        from core.agentic.formatters import AgenticFormatter
        f = AgenticFormatter()
        ts = (NOW - delta).isoformat()
        for text in ("felt rough", "   felt rough\n      wrapped"):
            conv = [{"timestamp": ts, "query": text, "response": "ok"}]
            assert label in f.format_recent_conversations(conv, now=NOW)
            assert label in f.format_memories([{"timestamp": ts, "content": text}], now=NOW)
            assert label in f.format_summaries([{"timestamp": ts, "content": text}], now=NOW)
            assert label in f.format_reflections([{"timestamp": ts, "content": text}], now=NOW)

    def test_agentic_unparseable_timestamp_passes_through(self):
        from core.agentic.formatters import AgenticFormatter
        out = AgenticFormatter().format_dreams([{"timestamp": "sometime", "content": "x"}])
        assert "[sometime]" in out

    def test_session_header_still_says_today_for_subday_label(self, frozen_clock):
        from core.prompt.formatter import _format_session_header
        assert "Today" in _format_session_header(NOW - timedelta(minutes=5))


class TestPacingLineBound:
    def _ctx(self, prev):
        fmt = _formatter()
        tm = MagicMock()
        tm.time_since_previous_message.return_value = "1 m"
        tm.elapsed_since_last_session.return_value = "2 h 0 m"
        tm.previous_query_time = prev
        fmt.time_manager = tm
        return fmt._get_time_context()

    def test_gap_names_the_message_it_measures(self):
        prev = datetime_today(16, 6)
        assert "Time since last message: 1 m (your message at 16:06)" in self._ctx(prev)

    def test_no_previous_message_line_unchanged(self):
        out = self._ctx(None)
        assert "Time since last message: 1 m\n" in out + "\n"
        assert "your message at" not in out


def datetime_today(h, m):
    return _real_datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# (b) source ratchet
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("core/prompt", "core/agentic")

# (file, enclosing function) -> why this strftime is NOT a bare item timestamp.
ALLOWED = {
    ("core/prompt/formatter.py", "_session_time_span"): "session span HH:MM under a header that already carries the relative day",
    ("core/prompt/formatter.py", "_format_session_header"): "header day name / session-began clock; the day label comes from format_relative_timestamp",
    ("core/prompt/formatter.py", "_get_time_context"): "wall clock now, session-began clock (the pacing line is bound to its message)",
    ("core/prompt/formatter.py", "_assemble_prompt"): "calendar event times: future, weekday-bearing, not recalled-item timestamps",
    ("core/prompt/gatherer_memory.py", "get_upcoming_schedule"): "event weekday/date for schedule facts (future, weekday-bearing)",
    ("core/prompt/gatherer_knowledge.py", "_fetch_upload_roster"): "upload roster date (date-only, no clock)",
    ("core/prompt/builder.py", "_today_timeline_from_probe"): "HH:MM list headed as today's queries (out of the FIGA fence)",
    ("core/agentic/controller.py", "_build_final_prompt"): "current date/time statement, not an item timestamp",
    ("core/agentic/controller.py", "_build_iteration_prompt"): "current date/time statement, not an item timestamp",
    ("core/agentic/controller.py", "_todays_calendar_line"): "today's date for the calendar line",
}
CEILING = 24  # strftime call count after FIGA's A2 routing; lower it, never raise it


def _scan():
    found = []
    for d in SCAN_DIRS:
        for path in sorted((REPO_ROOT / d).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            parents = {}
            for node in ast.walk(tree):
                for ch in ast.iter_child_nodes(node):
                    parents[ch] = node
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "strftime"):
                    fn = node
                    while fn in parents and not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fn = parents[fn]
                    name = fn.name if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) else "<module>"
                    found.append((str(path.relative_to(REPO_ROOT)), name))
    return found


def test_every_prompt_strftime_site_is_allowlisted():
    unlisted = sorted({s for s in _scan() if s not in ALLOWED})
    assert not unlisted, (
        f"strftime in a prompt module outside the allowlist: {unlisted}. An item timestamp must go through "
        "utils.time_manager.format_relative_timestamp / render_timestamp_label (BC-103); if this is not an "
        "item timestamp, add it to ALLOWED with a reason."
    )


def test_allowlist_has_no_stale_entries():
    stale = sorted(set(ALLOWED) - set(_scan()))
    assert not stale, f"allowlist entries with no strftime call left (remove them): {stale}"


def test_strftime_count_does_not_grow_or_sit_slack():
    n = len(_scan())
    assert n <= CEILING, f"strftime sites grew {CEILING} -> {n}; route item timestamps through the helper"
    assert n == CEILING, f"ceiling is slack ({CEILING} vs {n}); lower CEILING to {n}"
