"""Post-budget/pre-budget floor top-ups must restore the NEWEST items
(2026-08-27).

Live failure: the summaries/reflections floors iterated `stored[::-1]`
assuming an oldest-first list, but get_summaries/get_reflections return
newest-first — so when the token budget zeroed recent_summaries, the floor
restored the OLDEST summaries in the fetch buffer: [RECENT SUMMARIES]
rendered July 26/28 entries while the Aug 22-26 summaries sat unread in the
store. Same newest-first-treated-as-oldest-first class as the agentic digest
inversion (2026-08-02) and the history-stickiness slice (2026-08-22).

select_floor_topup sorts by timestamp explicitly — robust to either input
order — and is the single selection path for all four top-up sites (pre- and
post-budget, summaries and reflections).
"""

from datetime import datetime

from core.prompt.builder import select_floor_topup


def _s(ts, content):
    return {"content": content, "timestamp": ts, "type": "summary"}


NEWEST_FIRST = [
    _s("2026-08-26T13:34:33", "august 26 summary"),
    _s("2026-08-25T16:51:27", "august 25 summary"),
    _s("2026-08-24T21:39:41", "august 24 summary"),
    _s("2026-07-28T16:06:59", "july 28 summary"),
    _s("2026-07-26T13:59:00", "july 26 summary"),
]


class TestSelectFloorTopup:

    def test_newest_first_input_picks_newest(self):
        """The live-turn reproduction: newest-first fetch, floor of 2 —
        must restore Aug 26/25, NOT July 26/28."""
        add = select_floor_topup(NEWEST_FIRST, set(), 2)
        assert [a["content"] for a in add] == [
            "august 26 summary", "august 25 summary",
        ]

    def test_oldest_first_input_also_picks_newest(self):
        add = select_floor_topup(list(reversed(NEWEST_FIRST)), set(), 2)
        assert [a["content"] for a in add] == [
            "august 26 summary", "august 25 summary",
        ]

    def test_shuffled_input_also_picks_newest(self):
        shuffled = [NEWEST_FIRST[2], NEWEST_FIRST[4], NEWEST_FIRST[0],
                    NEWEST_FIRST[3], NEWEST_FIRST[1]]
        add = select_floor_topup(shuffled, set(), 3)
        assert [a["content"] for a in add] == [
            "august 26 summary", "august 25 summary", "august 24 summary",
        ]

    def test_already_present_content_skipped(self):
        add = select_floor_topup(NEWEST_FIRST, {"august 26 summary"}, 2)
        assert [a["content"] for a in add] == [
            "august 25 summary", "august 24 summary",
        ]

    def test_datetime_timestamps_supported(self):
        items = [
            _s(datetime(2026, 7, 1), "old"),
            _s(datetime(2026, 8, 26), "new"),
        ]
        add = select_floor_topup(items, set(), 1)
        assert add[0]["content"] == "new"

    def test_missing_timestamps_sort_last(self):
        items = [
            {"content": "undated", "type": "summary"},
            _s("2026-08-26T13:34:33", "dated"),
        ]
        add = select_floor_topup(items, set(), 1)
        assert add[0]["content"] == "dated"

    def test_empty_and_junk_safe(self):
        assert select_floor_topup([], set(), 2) == []
        assert select_floor_topup(None, set(), 2) == []
        assert select_floor_topup(["not a dict", {"content": ""}], set(), 2) == []

    def test_needed_zero(self):
        assert select_floor_topup(NEWEST_FIRST, set(), 0) == []
