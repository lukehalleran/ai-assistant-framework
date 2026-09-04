"""
Unit tests for utils/ordered_slice.py — the single source of truth for
"sort by timestamp before slicing" (the newest-first-then-truncate bug
class: agentic digest 08-02, LLM extractor 08-05, tone-detector recent-
distress 08-22, summary floors 08-27, insight evidence 08-31/09-04).
"""
from datetime import datetime, timedelta

import pytest

from utils.ordered_slice import (
    head,
    newest_first,
    oldest_first,
    week_bucket_key,
    window_fair_sample,
)


def _item(i, ts):
    return {"id": i, "ts": ts}


def _k(item):
    return item["ts"]


class TestNewestOldestFirst:
    def test_newest_first_sorts_descending(self):
        items = [_item(1, "2026-08-01"), _item(2, "2026-09-01"), _item(3, "2026-08-15")]
        assert [i["id"] for i in newest_first(items, _k)] == [2, 3, 1]

    def test_oldest_first_sorts_ascending(self):
        items = [_item(1, "2026-08-01"), _item(2, "2026-09-01"), _item(3, "2026-08-15")]
        assert [i["id"] for i in oldest_first(items, _k)] == [1, 3, 2]

    def test_already_sorted_newest_first_unchanged(self):
        """Behavior-preserving: an already-correctly-ordered list must come
        back byte-identical (stable sort + monotonic key)."""
        items = [_item(3, "2026-09-03"), _item(2, "2026-09-02"), _item(1, "2026-09-01")]
        assert newest_first(items, _k) == items

    def test_already_sorted_oldest_first_unchanged(self):
        items = [_item(1, "2026-09-01"), _item(2, "2026-09-02"), _item(3, "2026-09-03")]
        assert oldest_first(items, _k) == items

    def test_accepts_datetime_objects(self):
        base = datetime(2026, 9, 1)
        items = [_item(1, base), _item(2, base + timedelta(days=2)), _item(3, base - timedelta(days=1))]
        assert [i["id"] for i in newest_first(items, _k)] == [2, 1, 3]

    def test_none_timestamp_sorts_as_oldest_never_raises(self):
        items = [_item(1, "2026-09-01"), _item(2, None), _item(3, "2026-08-01")]
        result = newest_first(items, _k)
        assert [i["id"] for i in result] == [1, 3, 2]

    def test_unparseable_string_never_raises(self):
        items = [_item(1, "2026-09-01"), _item(2, "not-a-date")]
        result = newest_first(items, _k)
        assert [i["id"] for i in result] == [1, 2]

    def test_ts_key_raising_never_raises(self):
        def boom(item):
            raise ValueError("boom")
        items = [_item(1, "x")]
        assert newest_first(items, boom) == items

    def test_empty_list(self):
        assert newest_first([], _k) == []
        assert oldest_first([], _k) == []


class TestHead:
    def test_head_sorts_before_slicing(self):
        """The exact bug class: an UNSORTED list where a bare items[-n:]
        would return the wrong items."""
        # Deliberately out of order.
        items = [_item(2, "2026-08-20"), _item(1, "2026-09-01"), _item(3, "2026-07-01")]
        assert [i["id"] for i in head(items, 2, ts_key=_k)] == [1, 2]

    def test_head_oldest(self):
        items = [_item(2, "2026-08-20"), _item(1, "2026-09-01"), _item(3, "2026-07-01")]
        assert [i["id"] for i in head(items, 2, ts_key=_k, newest=False)] == [3, 2]

    def test_head_n_larger_than_list(self):
        items = [_item(1, "2026-09-01")]
        assert len(head(items, 10, ts_key=_k)) == 1

    def test_head_n_zero(self):
        items = [_item(1, "2026-09-01")]
        assert head(items, 0, ts_key=_k) == []


class TestWeekBucketKey:
    def test_none_is_none(self):
        assert week_bucket_key(None) is None

    def test_unparseable_is_none(self):
        assert week_bucket_key("not-a-date") is None

    def test_same_week_same_key(self):
        # 2026-09-01 (Tue) and 2026-09-03 (Thu) are the same ISO week.
        assert week_bucket_key("2026-09-01") == week_bucket_key("2026-09-03")

    def test_different_week_different_key(self):
        assert week_bucket_key("2026-09-01") != week_bucket_key("2026-08-01")


class TestWindowFairSample:
    def _week_item(self, i, week_offset, day_offset=0):
        base = datetime(2026, 9, 4) - timedelta(weeks=week_offset, days=day_offset)
        return _item(f"{week_offset}-{i}-{day_offset}", base.strftime("%Y-%m-%d"))

    def test_pure_permutation_when_n_is_none(self):
        items = [self._week_item(i, w) for w in range(3) for i in range(2)]
        result = window_fair_sample(items, _k)
        assert sorted(r["id"] for r in result) == sorted(r["id"] for r in items)
        assert len(result) == len(items)

    def test_round_robins_newest_week_first(self):
        # 3 weeks, 2 items each -> first two picks should be one from each
        # of the two most recent weeks, not both from week 0.
        items = [self._week_item(0, 0), self._week_item(1, 0),
                  self._week_item(0, 1), self._week_item(1, 1),
                  self._week_item(0, 2), self._week_item(1, 2)]
        result = window_fair_sample(items, _k, n=2)
        weeks_seen = {week_bucket_key(_k(r)) for r in result}
        assert len(weeks_seen) == 2, "a hard cap must not starve older weeks"

    def test_cap_does_not_collapse_to_newest_days_only(self):
        """The live 09-04 incident: newest-first + hard cap collapsed a
        multi-week sweep to only the newest ~3 days. window_fair_sample
        with a cap must span multiple weeks instead."""
        # 5 items in the newest week, 1 item each in 4 older weeks.
        items = [self._week_item(i, 0, day_offset=i) for i in range(5)]
        items += [self._week_item(0, w) for w in range(1, 5)]
        result = window_fair_sample(items, _k, n=5)
        weeks_seen = {week_bucket_key(_k(r)) for r in result}
        assert len(weeks_seen) >= 3, weeks_seen

    def test_undated_items_last(self):
        items = [_item(1, None), self._week_item(0, 0), _item(2, "not-a-date")]
        result = window_fair_sample(items, _k)
        assert result[-1]["id"] in (1, 2)
        assert result[-2]["id"] in (1, 2)

    def test_empty_list(self):
        assert window_fair_sample([], _k) == []
