"""
tests/unit/test_oct08c_pattern_engine_utc.py

2026-10-08 batch 4 (class: BC-58, sibling of BC-21): `_events_email` in
memory/pattern_engine.py parsed an aware ISO timestamp and then stripped it
with a bare `.replace(tzinfo=None)`, so a UTC "Z" stamp kept its UTC wall
clock and landed on the wrong local day / outside the right window.  The fix
routes it through the shared `utils.date_coerce.to_naive_local`
(convert-THEN-strip).

These tests drive THE deployed public entry point (`run_pattern_query`,
dimension="email") with the process zone pinned to UTC-5 via TZ/tzset.
"""

import os
import time
from datetime import datetime

import pytest

from memory.pattern_engine import PatternQuery, run_pattern_query


@pytest.fixture
def utc_minus_5():
    old = os.environ.get("TZ")
    os.environ["TZ"] = "Etc/GMT+5"  # POSIX sign: Etc/GMT+5 == UTC-5
    time.tzset()
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


def _row(date):
    return {"provider": "gmail", "message_id": "utc1",
            "sender": "A <a@example.org>", "subject": "late evening",
            "date": date, "unread": False}


def test_z_stamp_lands_on_previous_local_day(utc_minus_5):
    res = run_pattern_query(
        PatternQuery(dimension="email", window_days=3,
                     now=datetime(2026, 10, 8, 12, 0)),
        email_rows=[_row("2026-10-08T03:30:00Z")])
    exemplars = [e for b in res.buckets for e in b.exemplars]
    assert [e.date for e in exemplars] == ["2026-10-07"]


def test_z_stamp_inside_window_by_local_clock(utc_minus_5):
    # 03:30Z on 10-08 is 22:30 local on 10-07: inside a window ending 23:00
    # local on 10-07.  A bare tzinfo strip reads it as 10-08 03:30 (> until).
    res = run_pattern_query(
        PatternQuery(dimension="email", window_days=1,
                     now=datetime(2026, 10, 7, 23, 0)),
        email_rows=[_row("2026-10-08T03:30:00Z")])
    assert sum(b.count for b in res.buckets) == 1


def test_naive_stamp_unchanged(utc_minus_5):
    res = run_pattern_query(
        PatternQuery(dimension="email", window_days=3,
                     now=datetime(2026, 10, 8, 12, 0)),
        email_rows=[_row("2026-10-08T03:30:00")])
    exemplars = [e for b in res.buckets for e in b.exemplars]
    assert [e.date for e in exemplars] == ["2026-10-08"]
