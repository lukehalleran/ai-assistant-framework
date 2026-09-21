"""
tests/unit/test_naive_aware_datetime_sites.py

Lane T (2026-09-20, class BC-58): naive-vs-aware datetime arithmetic at three
sites. Project convention is naive LOCAL time; a value that arrives aware
(a "Z"/offset-suffixed ISO string, or an aware datetime object) must be
CONVERTED to local and stripped before it is subtracted from a naive
`datetime.now()` — never a bare `.replace(tzinfo=None)`, which silently
misreads a UTC value by the UTC offset.

Pre-fix symptoms this guards against (confirmed against the pristine
~/daemon_exec/base_0920 clone before this batch):
  - utils.query_checker.belongs_to_thread: the `elif isinstance(last_time,
    datetime)` branch had NO guard at all and raised TypeError uncaught for
    an aware datetime object; the str branch's "Z"-suffixed path parsed an
    AWARE datetime and then subtracted it from naive `datetime.now()`,
    always raising and being silently swallowed into the hard-coded 3600s
    fallback.
  - core.context_pipeline.ContextPipeline._should_reset_tone_stickiness:
    the same subtraction raised TypeError, but the method's own
    `except Exception` swallowed it and fail-closed to False (no reset) —
    so a 10-hour-old AWARE timestamp silently kept a stale distress tone
    stuck, when the intended behavior (matching the naive-timestamp path)
    is to reset stickiness past TONE_STICKINESS_MAX_GAP_MINUTES.

All timestamps in this file are built relative to `datetime.now()` /
`datetime.now(timezone.utc)` at test time — never a hard-coded wall-clock
value — so the tests are timezone- and date-robust.
"""

from datetime import datetime, timedelta, timezone

from core.context_pipeline import ContextPipeline
from utils.date_coerce import to_naive_local
from utils.query_checker import belongs_to_thread


# ---------------------------------------------------------------------------
# utils.date_coerce.to_naive_local
# ---------------------------------------------------------------------------

def test_to_naive_local_aware_converts_to_local_naive_equivalent():
    aware_utc = datetime.now(timezone.utc)
    converted = to_naive_local(aware_utc)
    assert converted.tzinfo is None
    # Must match the local-clock reading of the same instant, not a bare
    # tzinfo strip (which would misread by the UTC offset whenever local
    # time is not UTC).
    expected = datetime.fromtimestamp(aware_utc.timestamp())
    assert converted == expected


def test_to_naive_local_naive_input_unchanged():
    naive = datetime.now().replace(microsecond=0)
    assert to_naive_local(naive) == naive
    assert to_naive_local(naive).tzinfo is None


# ---------------------------------------------------------------------------
# utils.query_checker.belongs_to_thread — naive string / aware "Z" string /
# aware datetime object must agree, and must not fall into the 3600s fallback.
# ---------------------------------------------------------------------------

def _make_last_conv(timestamp):
    return {
        "query": "tell me about neural networks",
        "response": "Neural networks are computational models",
        "timestamp": timestamp,
        "is_heavy_topic": False,
        "topic": "machine_learning",
    }


def _verdict_for_gap(delta, timestamp_kind):
    naive_ts = datetime.now() - delta
    aware_ts = datetime.now(timezone.utc) - delta
    if timestamp_kind == "naive_str":
        ts = naive_ts.isoformat()
    elif timestamp_kind == "aware_z_str":
        ts = aware_ts.isoformat().replace("+00:00", "Z")
    elif timestamp_kind == "aware_obj":
        ts = aware_ts
    else:
        raise ValueError(timestamp_kind)
    return belongs_to_thread(
        current_query="how do neural networks learn?",
        last_conversation=_make_last_conv(ts),
        current_topic="machine_learning",
    )


def test_belongs_to_thread_naive_string_two_minutes_true_sixty_minutes_false():
    # Baseline: the naive-string path (already correct pre-fix) establishes
    # that this query/topic pair's verdict DIFFERS between a 2-minute and a
    # 60-minute gap, so the aware-input assertions below are meaningful.
    assert _verdict_for_gap(timedelta(minutes=2), "naive_str") is True
    assert _verdict_for_gap(timedelta(minutes=60), "naive_str") is False


def test_belongs_to_thread_aware_z_string_matches_naive_verdict():
    assert _verdict_for_gap(timedelta(minutes=2), "aware_z_str") is True
    assert _verdict_for_gap(timedelta(minutes=60), "aware_z_str") is False


def test_belongs_to_thread_aware_datetime_object_matches_naive_verdict():
    # Pre-fix this branch had NO try/except at all and raised TypeError.
    assert _verdict_for_gap(timedelta(minutes=2), "aware_obj") is True
    assert _verdict_for_gap(timedelta(minutes=60), "aware_obj") is False


def test_belongs_to_thread_aware_inputs_all_three_kinds_agree():
    for delta in (timedelta(minutes=2), timedelta(minutes=60)):
        naive_verdict = _verdict_for_gap(delta, "naive_str")
        aware_z_verdict = _verdict_for_gap(delta, "aware_z_str")
        aware_obj_verdict = _verdict_for_gap(delta, "aware_obj")
        assert naive_verdict == aware_z_verdict == aware_obj_verdict


def test_belongs_to_thread_no_exception_propagates_for_any_timestamp_shape():
    # Regression guard: none of the three timestamp shapes may raise, and
    # none may silently fall into the 3600s "can't parse" fallback path in
    # a way that changes the verdict from the naive-string ground truth.
    for kind in ("naive_str", "aware_z_str", "aware_obj"):
        for delta in (timedelta(minutes=2), timedelta(minutes=60)):
            _verdict_for_gap(delta, kind)  # must not raise


# ---------------------------------------------------------------------------
# core.context_pipeline.ContextPipeline._should_reset_tone_stickiness
# ---------------------------------------------------------------------------

def _pipeline_with_carried_tone():
    p = ContextPipeline.__new__(ContextPipeline)
    p._last_tone_level = "CONCERN"
    return p


def test_should_reset_tone_stickiness_aware_five_minutes_no_reset():
    p = _pipeline_with_carried_tone()
    five_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    assert p._should_reset_tone_stickiness([{"timestamp": five_min_ago}]) is False


def test_should_reset_tone_stickiness_aware_ten_hours_resets():
    p = _pipeline_with_carried_tone()
    ten_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
    assert p._should_reset_tone_stickiness([{"timestamp": ten_hours_ago}]) is True
