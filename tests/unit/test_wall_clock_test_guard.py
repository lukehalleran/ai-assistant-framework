"""Repo-wide guard (2026-10-15): tests must not build "now minus a few minutes"
timestamps without saying why a midnight crossing cannot change the outcome.

Incident: ``test_read_time_markers_formatter_strip`` went red on master CI at
00:00 UTC because ``datetime.now() - timedelta(minutes=5)`` is *yesterday* for
the first five minutes of every day, and the formatter treats a date change as
a session boundary. The same shape in ``test_gatherer_memory_user_stated_ranking`` and
``test_gatherer_memory_user_stated_candidates`` flipped a same-session ranking assertion. The
audit (docs: lane CLOCK handoff) found every other hit compared an elapsed
duration only.

Rule: a line of the shape ``datetime.now() - timedelta(minutes|seconds=...)``
(or ``utcnow()``) in a test must carry ``# clock-safe: <reason>`` on that line
-- the reason states why a date/session boundary cannot change the assertion.
If the code under test DOES treat a date change differently, do not mark it:
pin the clock the deployed code reads (monkeypatch the module's ``datetime``
name, or pass ``now=``) and build the timestamps from the same instant.
Ceiling 0. class: wall-clock-dependent test (candidate; see lane CLOCK handoff).
Violations report path and line only.

Extended 2026-10-23 (lane TIME, BC-94 residue): the same mark is required on
``datetime.now() - timedelta(... hours=N ...)`` (or ``utcnow()``) and on
``time.time() - <number>``. Audit: every such fixture was run with the clock
forced to seven instants (00:00:30, 03:00, 06:00, 12:00, 18:00, 23:30, 23:59:50;
the run starting at 23:59:50 crossed midnight); one file was at risk (the STM
probe-window test, fixed by pinning the clock the deployed session helper reads)
and the 75 marked lines compare elapsed time only. A bare ``now``/``NOW`` variable is NOT matched: it
may be a fixed constant, and a wall-clock binding there is audited by the
``now = datetime.now()`` line itself, which this guard does not cover.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()

_NOW_MINUS_RE = re.compile(
    r"datetime\.(?:utc)?now\(\)\s*-\s*timedelta\([^)]*\b(?:minutes|seconds)\s*="
)
_NOW_MINUS_HOURS_RE = re.compile(
    r"datetime\.(?:utc)?now\(\)\s*-\s*timedelta\([^)]*\bhours\s*="
)
_TIME_MINUS_RE = re.compile(r"\btime\.time\(\)\s*-\s*[0-9]")
_PATTERNS = (_NOW_MINUS_RE, _NOW_MINUS_HOURS_RE, _TIME_MINUS_RE)
_MARK = "# clock-safe:"
MAX_UNMARKED = 0


def _violations(text: str) -> list[int]:
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if any(r.search(line) for r in _PATTERNS) and _MARK not in line:
            out.append(n)
    return out


def test_now_minus_delta_needs_clock_safe_reason():
    found = []
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        if path.resolve() == THIS_FILE:
            continue
        for n in _violations(path.read_text(encoding="utf-8", errors="replace")):
            found.append(f"{path.relative_to(TESTS_ROOT)}:{n}")
    assert len(found) <= MAX_UNMARKED, (
        "unmarked `datetime.now() - timedelta(minutes|seconds|hours=...)` or "
        "`time.time() - <number>` in tests "
        "(add `# clock-safe: <reason>` or pin the clock): " + ", ".join(found)
    )


def test_detector_flags_unmarked_and_accepts_marked():
    bad = "ts = (datetime.now() - timedelta(minutes=5)).isoformat()"
    good = bad + "  # clock-safe: elapsed gap only"
    assert _violations(bad) == [1]
    assert _violations(good) == []
    assert _violations("x = datetime.utcnow() - timedelta(seconds=9)") == [1]
    assert _violations("x = datetime.now() - timedelta(days=3)") == []


def test_detector_covers_hours_and_time_time_fixtures():
    assert _violations("old = datetime.now() - timedelta(hours=5)") == [1]
    assert _violations("old = datetime.utcnow() - timedelta(days=2, hours=5)") == [1]
    assert _violations("ts = (datetime.now() - timedelta(hours=(E + H) / 2))") == [1]
    assert _violations("old = time.time() - 100") == [1]
    assert _violations("   old = time.time() - 14 * 86400") == [1]
    assert _violations("old = datetime.now() - timedelta(hours=5)  # clock-safe: TTL") == []
    assert _violations("old = time.time() - 100  # clock-safe: elapsed only") == []
    # not wall-clock now-minus: a fixed anchor, a bare variable, or a later-than-now offset
    assert _violations("x = T0 + timedelta(hours=3)") == []
    assert _violations("x = NOW - timedelta(hours=3)") == []
    assert _violations("x = time.time() + 3600") == []
    assert _violations("x = time.time()") == []
