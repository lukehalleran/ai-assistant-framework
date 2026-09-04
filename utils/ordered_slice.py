"""
# utils/ordered_slice.py

Module Contract
- Purpose: single source of truth for "sort by timestamp before slicing" —
  the newest-first-then-truncate bug class. A bare ``items[-n:]``,
  ``items[:n]`` or ``items[::-1]`` silently ASSUMES the list already carries
  the ordering convention the caller expects; when it doesn't (a different
  upstream sort order, a merge of two differently-ordered sources, a cache
  built incrementally), the slice picks the WRONG n items with no error —
  five prior incidents share this exact class: the agentic recent-
  conversation digest (2026-08-02, `recent[-4:]` on a newest-first list —
  the coordinator got the 4 OLDEST turns labeled "most recent"), the LLM
  fact extractor's budget walk (2026-08-05, oldest-first break truncated the
  NEWEST turns), `utils.tone_detector._recent_distress_from_history`
  (2026-08-22), the summary/reflection floor top-ups (2026-08-27,
  `[::-1]` applied to an already-newest-first list restored the OLDEST
  entries instead of the newest), and insight evidence rendering
  (2026-08-31 / 2026-09-04, a fixed render cap collapsed to only the newest
  ~3 days of a multi-week sweep).
- Inputs: a list of arbitrary items + a `ts_key` callable extracting each
  item's timestamp (a `datetime`, or an ISO-ish date/datetime string).
  `ts_key` may return None / raise for an item with no usable timestamp —
  every function here treats that as "sorts as oldest", never crashes.
- Outputs:
  - newest_first(items, ts_key): stably sorted, most recent first.
  - oldest_first(items, ts_key): stably sorted, least recent first.
  - head(items, n, *, ts_key, newest=True): SORTS then slices — the
    fix for a bare ``items[-n:]``/``items[:n]`` on a maybe-unsorted list.
  - window_fair_sample(items, ts_key, n=None): round-robin across ISO-week
    buckets, newest bucket first (each bucket keeps its own newest-first
    order); undated items last. n=None reorders the WHOLE list (a pure
    permutation, nothing dropped — the window-fair RENDERING case); n=<int>
    caps the output while still round-robining, so a hard cap doesn't
    starve older buckets (the fixed-size top-up/floor case). This is the
    single implementation behind core.insight.sweep's window-fair
    evidence reordering (interleave_evidence_for_coverage is a thin
    ts_key=item.date wrapper around this, so any future duplication finds
    ONE function to fix).
- Side effects: none — pure functions, never mutate the input list.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")
TsKey = Callable[[T], Any]

__all__ = [
    "newest_first", "oldest_first", "head", "window_fair_sample",
    "week_bucket_key", "round_robin_merge",
]


def round_robin_merge(queues: Sequence[Sequence[T]], n: Optional[int] = None) -> List[T]:
    """Interleave several already-grouped queues round-robin (queue order
    as given — callers order queues newest-first/highest-priority-first
    before calling). Each queue keeps its own internal order.

    This is the shared primitive behind window_fair_sample's week-bucket
    interleaving; also used directly where the grouping axis isn't time at
    all (e.g. gui.handlers._interleave_phase_events round-robins per-PHASE
    event lists, not per-week ones — same "don't let one group's items
    starve the others under a cap" fairness problem, different grouping
    key, so it reuses this merge loop rather than window_fair_sample's
    week-bucketing).
    """
    live = [list(q) for q in queues if q]
    result: List[T] = []
    while any(live):
        for q in live:
            if q:
                result.append(q.pop(0))
                if n is not None and len(result) >= n:
                    return result
    return result


def _safe_call(ts_key: TsKey, item: T) -> Any:
    try:
        return ts_key(item)
    except Exception:
        return None


def _normalize_ts(value: Any) -> datetime:
    """Coerce a timestamp-ish value to a naive ``datetime`` for sorting.

    Unparseable / missing values normalize to ``datetime.min`` — they sort
    as the OLDEST possible item rather than raising, so a single bad
    timestamp in a large list never breaks the whole sort/slice.
    """
    if value is None:
        return datetime.min
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return datetime.min
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return datetime.min
    return datetime.min


def newest_first(items: Sequence[T], ts_key: TsKey) -> List[T]:
    """Stable sort by ``ts_key(item)``, most recent first.

    Applying this to a list that is ALREADY newest-first reproduces the
    same order (stable sort + monotonic key), so adopting it at a call site
    that was already correctly sorted is behavior-preserving.
    """
    return sorted(items, key=lambda it: _normalize_ts(_safe_call(ts_key, it)), reverse=True)


def oldest_first(items: Sequence[T], ts_key: TsKey) -> List[T]:
    """Stable sort by ``ts_key(item)``, least recent first."""
    return sorted(items, key=lambda it: _normalize_ts(_safe_call(ts_key, it)))


def head(items: Sequence[T], n: int, *, ts_key: TsKey, newest: bool = True) -> List[T]:
    """Sort by ``ts_key`` THEN take the first ``n`` — never slice-then-hope.

    ``newest=True`` (default) returns the n most recent items; ``newest=
    False`` returns the n oldest. Replaces patterns like
    ``sorted_maybe[-n:]`` / ``sorted_maybe[:n]`` that assume an ordering
    convention instead of enforcing one.
    """
    if n is None or n < 0:
        n = len(items)
    ordered = newest_first(items, ts_key) if newest else oldest_first(items, ts_key)
    return ordered[:n]


def week_bucket_key(value: Any) -> Optional[Tuple[int, int]]:
    """Return a sortable ``(iso_year, iso_week)`` key for a timestamp-ish
    value, or None when it's missing/unparseable."""
    if value is None:
        return None
    ts = _normalize_ts(value)
    if ts == datetime.min:
        return None
    iso_year, iso_week, _ = ts.isocalendar()
    return (iso_year, iso_week)


def window_fair_sample(
    items: Sequence[T], ts_key: TsKey, n: Optional[int] = None
) -> List[T]:
    """Round-robin across ISO-week buckets, newest bucket first.

    Each bucket internally keeps its own newest-first order (derived from
    each item's ``ts_key``, not from input order — this does NOT assume the
    input is pre-sorted). Undated items (unparseable/missing ``ts_key``)
    are appended last, in their original relative order.

    ``n=None``: return every item, reordered — a pure permutation, nothing
    dropped (the window-fair RENDERING use case: a downstream character/
    count cap should span the whole date range instead of collapsing to
    only the most recent days).
    ``n=<int>``: cap the output at n items while still round-robining
    newest-bucket-first, so a fixed-size cap doesn't starve older buckets
    (the summary/reflection floor top-up use case).
    """
    buckets: dict = {}
    order: list = []
    undated: List[T] = []
    for item in items:
        key = week_bucket_key(_safe_call(ts_key, item))
        if key is None:
            undated.append(item)
            continue
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(item)
    # Each bucket newest-first internally.
    for key in order:
        buckets[key] = newest_first(buckets[key], ts_key)
    order.sort(reverse=True)  # newest ISO week first
    queues = [buckets[k] for k in order]

    result = round_robin_merge(queues, n=n)
    if n is not None and len(result) >= n:
        return result[:n]

    if n is None:
        result.extend(undated)
        return result
    for item in undated:
        if len(result) >= n:
            break
        result.append(item)
    return result
