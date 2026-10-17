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
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()

_NOW_MINUS_RE = re.compile(
    r"datetime\.(?:utc)?now\(\)\s*-\s*timedelta\([^)]*\b(?:minutes|seconds)\s*="
)
_MARK = "# clock-safe:"
MAX_UNMARKED = 0


def _violations(text: str) -> list[int]:
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if _NOW_MINUS_RE.search(line) and _MARK not in line:
            out.append(n)
    return out


def test_now_minus_minutes_needs_clock_safe_reason():
    found = []
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        if path.resolve() == THIS_FILE:
            continue
        for n in _violations(path.read_text(encoding="utf-8", errors="replace")):
            found.append(f"{path.relative_to(TESTS_ROOT)}:{n}")
    assert len(found) <= MAX_UNMARKED, (
        "unmarked `datetime.now() - timedelta(minutes|seconds=...)` in tests "
        "(add `# clock-safe: <reason>` or pin the clock): " + ", ".join(found)
    )


def test_detector_flags_unmarked_and_accepts_marked():
    bad = "ts = (datetime.now() - timedelta(minutes=5)).isoformat()"
    good = bad + "  # clock-safe: elapsed gap only"
    assert _violations(bad) == [1]
    assert _violations(good) == []
    assert _violations("x = datetime.utcnow() - timedelta(seconds=9)") == [1]
    assert _violations("x = datetime.now() - timedelta(days=3)") == []
