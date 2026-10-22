"""Repo-wide guard (2026-10-22): a test must not assert an ABSOLUTE upper bound
on elapsed time without saying why it is safe on shared hardware.

Incident: ``test_shadow_wall_time_is_the_max_of_the_two_not_the_sum`` ran two
0.3 s fakes concurrently and asserted ``elapsed < 0.5``; the GitHub runner took
0.552 s and the PR (README-only) went red. The test meant "the two calls
overlap", which is an ORDER property, not a duration. The audit found the same
shape in decision-model, wiki-deadline and shutdown tests.

Rule: an ``assert`` that bounds a monotonic / perf_counter / time.time /
``.total_seconds()`` difference (inline, or through a variable assigned from
one) with ``<`` / ``<=`` (or ``X > delta``) must carry ``# timing-safe: <reason>``
on that line. Prefer a structural assertion (per-call (start, end) stamps:
overlap / ordering); keep a bound only for a genuine timeout test with headroom
>= 2x the expected time (or >= 1 s). Lower bounds (``elapsed >= 0.1``) are safe
and never flagged. Multi-line asserts are not scanned (keep them on one line).
Ceiling 0. class: absolute wall-time bound in a test (candidate BC-101, see
the oct22 FLAKE bc_draft). Violations report path and line only.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()

_CLOCK = r"(?:\w+\.)?(?:monotonic|perf_counter|time)\(\)"
_CLOCK_DIFF = rf"(?:{_CLOCK}\s*-|\.total_seconds\(\))"
# a variable that IS an elapsed value: `x = clock() - t0` or `x = (...).total_seconds()`
_ASSIGN_RE = re.compile(
    rf"^\s*(\w+)\s*=\s*(?:\(?{_CLOCK}\s*-\s*[\w.]+\)?|.*\.total_seconds\(\))\s*$"
)
_MARK = "# timing-safe:"
MAX_UNMARKED = 0
_LT = r"(?<![<\-])<=?(?!<)"
_GT = r"(?<![>\-])>=?(?!>)"


def _violations(text: str) -> list[int]:
    deltas: set[str] = set()
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        m = _ASSIGN_RE.match(line)
        if m and "assert" not in line.split("=", 1)[0]:
            deltas.add(m.group(1))
        if not line.lstrip().startswith("assert ") or _MARK in line:
            continue
        expr = rf"(?:{_CLOCK}\s*-\s*\w+|\([^()]*{_CLOCK_DIFF}[^()]*\)\.total_seconds\(\)|[^\s]*\.total_seconds\(\)"
        if deltas:
            expr += r"|\b(?:" + "|".join(sorted(map(re.escape, deltas))) + r")\b"
        expr += ")"
        if "total_seconds" in line.split("<", 1)[-1] and line.count("total_seconds") > 1:
            continue  # relative comparison of two deltas, not an absolute bound
        if re.search(expr + r"[^<>]*?\s*" + _LT, line) or re.search(_GT + r"\s*" + expr, line):
            out.append(n)
    return out


def test_elapsed_upper_bounds_in_tests_need_a_timing_safe_reason():
    found = []
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        if path.resolve() == THIS_FILE:
            continue
        for n in _violations(path.read_text(encoding="utf-8", errors="replace")):
            found.append(f"{path.relative_to(TESTS_ROOT)}:{n}")
    assert len(found) <= MAX_UNMARKED, (
        "unmarked absolute elapsed-time upper bound in tests (assert overlap/order "
        "on per-call stamps, or add `# timing-safe: <reason>`): " + ", ".join(found)
    )


def test_detector_flags_the_arbiter_shape_and_accepts_the_alternatives():
    # the 2026-10-22 incident line, verbatim shape
    bad = 'assert time.monotonic() - t0 < 0.5 and got.arbiter_receipt["status"] == "ok"'
    assert _violations(bad) == [1]
    assert _violations(bad + "  # timing-safe: genuine timeout, 2x headroom") == []
    # through a variable, plain and chained (indented / wrapped form)
    var = "    t0 = time.monotonic()\n    elapsed = time.monotonic() - t0\n    assert 1.4 < elapsed < 2.0\n"
    assert _violations(var) == [3]
    assert _violations("    wall = time.perf_counter() - t0\n    assert wall < 1.0, wall\n") == [2]
    assert _violations("    d = (datetime.now() - x).total_seconds()\n    assert 0 <= d <= 1\n") == [2]
    assert _violations("assert (datetime.now() - ts).total_seconds() < 1") == [1]
    assert _violations("assert 0.5 > time.monotonic() - t0") == [1]
    # lower bounds, ordering and overlap are the safe forms
    assert _violations("    elapsed = time.time() - start\n    assert elapsed >= 0.1\n") == []
    assert _violations("assert time.monotonic() - t0 > 1.4") == []
    assert _violations("assert l_start < j_end and j_start < l_end") == []
    assert _violations('assert done_at["thread"] < done_at["tracking"]') == []
