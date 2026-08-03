"""
Regression tests for the 2026-08-02 agentic digest order inversion.

The context gatherer returns `recent_conversations` NEWEST-FIRST, but
`_compute_recent_conversation_digest` assumed oldest-first and took
`recent[-N:]` — the decision rounds got the N OLDEST turns (yesterday's
conversation) labeled "most recent last", while the turns the user's
"that"/"earlier" actually referenced were dropped. Combined with
decision-answer reuse (the decision-round text IS the final response), the
assistant asked the user to re-explain things said 20 minutes earlier twice
in one day.

The digest now orders by timestamp when parseable (robust to either input
ordering) and falls back to treating the input as newest-first.
"""

from core.agentic.controller import AgenticSearchController


def _digest(recent):
    ctrl = AgenticSearchController.__new__(AgenticSearchController)
    return ctrl._compute_recent_conversation_digest(
        {"recent_conversations": recent}
    )


def _conv(ts, query, response="reply"):
    return {"timestamp": ts, "query": query, "response": response}


NEWEST_FIRST = [
    _conv("2026-08-02 12:34:44.991014", "newest turn about the car"),
    _conv("2026-08-02 12:33:48.000000", "second newest turn"),
    _conv("2026-08-02 12:12:30.007079", "car scream turn"),
    _conv("2026-08-02 12:11:00.000000", "first car turn"),
    _conv("2026-08-01 19:42:35.691698", "old turn one"),
    _conv("2026-08-01 16:49:52.818400", "old turn two"),
    _conv("2026-08-01 16:32:04.349228", "old turn three"),
    _conv("2026-08-01 16:31:01.788675", "oldest turn"),
]


class TestDigestOrdering:
    def test_newest_turns_kept_from_newest_first_input(self):
        digest = _digest(NEWEST_FIRST)
        assert "newest turn about the car" in digest
        assert "car scream turn" in digest
        # The old turns (what the pre-fix tail selected) must be gone.
        assert "old turn" not in digest
        assert "oldest turn" not in digest

    def test_newest_turns_kept_from_oldest_first_input(self):
        digest = _digest(list(reversed(NEWEST_FIRST)))
        assert "newest turn about the car" in digest
        assert "oldest turn" not in digest

    def test_rendered_most_recent_last(self):
        digest = _digest(NEWEST_FIRST)
        assert digest.index("first car turn") < digest.index("car scream turn")
        assert digest.index("car scream turn") < digest.index("newest turn about the car")

    def test_unparseable_timestamps_assume_newest_first(self):
        recent = [
            _conv("garbage", "newest unstamped"),
            _conv("", "middle unstamped"),
            _conv(None, "oldest unstamped"),
        ]
        digest = _digest(recent)
        # All three fit within the turn cap; ordering flips to oldest-first.
        assert digest.index("oldest unstamped") < digest.index("newest unstamped")

    def test_empty_input_returns_empty(self):
        assert _digest([]) == ""
        ctrl = AgenticSearchController.__new__(AgenticSearchController)
        assert ctrl._compute_recent_conversation_digest(None) == ""

    def test_header_warns_against_reexplain(self):
        digest = _digest(NEWEST_FIRST)
        assert "re-explain" in digest
