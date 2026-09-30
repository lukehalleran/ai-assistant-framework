"""Regression tests for the 2026-09-27 session-audit fix batch E4 —
"session truth + recall-shaped intent" (docs plan
PLAN_20260927_session_audit_fixes.md, section E4; evidence items 6 and 7).

Live incident (evidence 6): 09-26 21:05 "what time did i sit down" answered
"8:42 PM" — the true answer (18:23) sat outside the 10-turn [RECENT
CONVERSATION] window, and the window's session header presented its own
oldest SHOWN entry (20:42) as if it were the session start. The query also
never classified TEMPORAL_RECALL (no pattern covered "what time did i").

Live incident (evidence 7): "agi last year super low. Definitely under 20k I
would say?" classified temporal_recall@0.85 — the old adverb pattern let a
'?' ending an unrelated LATER sentence satisfy a lookahead paired with an
adverb in an EARLIER sentence.

Covers:
  - formatter._format_session_header's `truncation=` honesty branch
  - formatter._get_time_context's "Current session began" line
  - formatter's [EARLIER TODAY — your messages not shown above] section
  - builder._fetch_session_probe / _session_truth_from_probe /
    _today_timeline_from_probe (pure helpers, no live corpus)
  - intent_classifier's new wh-time recall shape + sentence-scoped adverb fix
  - token_manager.PRIORITY_ORDER carries rows for the new context keys
    (also exercised generally by test_budget_meters_rendered_sections.py)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_formatter():
    from core.prompt.formatter import PromptFormatter
    token_mgr = MagicMock()
    token_mgr.count_tokens = MagicMock(return_value=10)
    fmt = PromptFormatter(token_manager=token_mgr, time_manager=None)
    return fmt


def _base_context(**overrides):
    ctx = {
        "recent_conversations": [],
        "memories": [],
        "user_profile": "",
        "narrative_state": "",
        "summaries": [],
        "reflections": [],
        "dreams": [],
        "semantic_chunks": [],
        "wiki": [],
        "personal_notes": [],
        "reference_docs": [],
        "user_uploads": [],
        "git_commits": [],
        "procedural_skills": [],
        "proposed_features": [],
        "graph_context": [],
        "unresolved_threads": [],
        "upcoming_schedule": [],
        "google_calendar": [],
        "proactive_insights": [],
        "web_search_results": None,
    }
    ctx.update(overrides)
    return ctx


def _shown_window(n=10, step_minutes=3, ref_hour=20, ref_minute=42):
    """`n` conversation entries, newest-first, `step_minutes` apart, ending
    at (ref_hour, ref_minute) + (n-1)*step_minutes — a single continuous
    session (every adjacent gap well under the 2h boundary)."""
    base = datetime.now().replace(hour=ref_hour, minute=ref_minute, second=0, microsecond=0)
    end = base + timedelta(minutes=step_minutes * (n - 1))
    return [
        {
            "query": f"q{i}", "response": f"r{i}",
            "timestamp": (end - timedelta(minutes=step_minutes * i)).isoformat(),
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. formatter._format_session_header — truncation honesty
# ---------------------------------------------------------------------------

class TestSessionHeaderTruncationHonesty:
    def test_truncated_header_names_shown_vs_total_and_start(self):
        """This is the evidence-6 acceptance shape: a 10-turn window, all one
        session, 70 turns total, session actually began at 18:23 — the
        header must say so instead of claiming the window's own edge (20:42)
        is the session start."""
        fmt = _make_formatter()
        window = _shown_window(n=10)
        started_at = datetime.now().replace(hour=18, minute=23, second=0, microsecond=0)
        ctx = _base_context(
            recent_conversations=window,
            recent_window_truncated=True,
            session_turns_total=70,
            session_started_at=started_at,
        )
        result = fmt._assemble_prompt(ctx, "what time did i sit down")
        assert "showing the last 10 of 70 turns" in result
        assert "session began 18:23" in result
        header_lines = [line for line in result.splitlines() if line.startswith("--- Session:")]
        assert len(header_lines) == 1
        # The old, dishonest span-suffix style ("... ), HH:MM–HH:MM ---") must
        # not also appear on this header — it is REPLACED by the honesty text.
        assert "–" not in header_lines[0]

    def test_untruncated_window_keeps_ordinary_header(self):
        fmt = _make_formatter()
        window = _shown_window(n=5)
        ctx = _base_context(
            recent_conversations=window,
            recent_window_truncated=False,
            session_turns_total=None,
            session_started_at=None,
        )
        result = fmt._assemble_prompt(ctx, "hello")
        assert "showing the last" not in result
        assert "session began" not in result

    def test_missing_truncation_keys_behave_like_before(self):
        """Omitting the new keys entirely (an older caller/context) must not
        change [RECENT CONVERSATION] rendering at all."""
        fmt = _make_formatter()
        window = _shown_window(n=4)
        ctx = _base_context(recent_conversations=window)
        result = fmt._assemble_prompt(ctx, "hello")
        assert "showing the last" not in result
        assert "--- Session:" in result


# ---------------------------------------------------------------------------
# 2. formatter._get_time_context — "Current session began" line
# ---------------------------------------------------------------------------

class TestTimeContextSessionBegan:
    def test_line_present_with_both_values(self):
        fmt = _make_formatter()
        started_at = datetime.now().replace(hour=18, minute=23, second=0, microsecond=0)
        ctx = _base_context(
            recent_conversations=_shown_window(n=3),
            session_started_at=started_at,
            session_turns_total=70,
        )
        result = fmt._assemble_prompt(ctx, "hello")
        assert "Current session began:" in result
        assert "18:23" in result
        assert "(70 turns so far)" in result

    def test_line_absent_without_session_data(self):
        fmt = _make_formatter()
        ctx = _base_context(recent_conversations=_shown_window(n=3))
        result = fmt._assemble_prompt(ctx, "hello")
        assert "Current session began:" not in result

    def test_direct_method_zero_total_omits_line(self):
        # A falsy total (0/None) must not render a nonsensical "0 turns so far".
        fmt = _make_formatter()
        started_at = datetime(2026, 5, 17, 18, 23)
        assert "Current session began" not in fmt._get_time_context(started_at, 0)
        assert "Current session began" not in fmt._get_time_context(None, 5)
        assert "Current session began: Sun 18:23 (5 turns so far)" in fmt._get_time_context(started_at, 5)


# ---------------------------------------------------------------------------
# 3. formatter [EARLIER TODAY] section
# ---------------------------------------------------------------------------

class TestEarlierTodaySection:
    def test_renders_when_timeline_present(self):
        fmt = _make_formatter()
        ctx = _base_context(
            recent_conversations=_shown_window(n=3),
            session_timeline=["18:23 sat down at the desk", "19:05 grabbed a snack"],
        )
        result = fmt._assemble_prompt(ctx, "what time did i sit down")
        assert "[EARLIER TODAY — your messages not shown above] n=2" in result
        assert "18:23 sat down at the desk" in result
        assert "19:05 grabbed a snack" in result

    def test_absent_when_no_timeline(self):
        fmt = _make_formatter()
        ctx = _base_context(recent_conversations=_shown_window(n=3))
        result = fmt._assemble_prompt(ctx, "hello")
        assert "[EARLIER TODAY" not in result

    def test_absent_when_timeline_empty_list(self):
        fmt = _make_formatter()
        ctx = _base_context(recent_conversations=_shown_window(n=3), session_timeline=[])
        result = fmt._assemble_prompt(ctx, "hello")
        assert "[EARLIER TODAY" not in result


# ---------------------------------------------------------------------------
# 4. builder.py pure helpers (no live corpus/memory system)
# ---------------------------------------------------------------------------

class TestFetchSessionProbe:
    def test_reads_via_corpus_manager(self):
        from core.prompt.builder import _fetch_session_probe

        rows = [{"query": "a"}, {"query": "b"}]
        corpus_manager = MagicMock()
        corpus_manager.get_recent_memories = MagicMock(return_value=rows)
        memory_coordinator = MagicMock(corpus_manager=corpus_manager)

        out = _fetch_session_probe(memory_coordinator, 200)
        assert out == rows
        corpus_manager.get_recent_memories.assert_called_once_with(count=200)

    def test_no_corpus_manager_returns_empty(self):
        from core.prompt.builder import _fetch_session_probe

        memory_coordinator = MagicMock(spec=[])  # no corpus_manager attribute
        assert _fetch_session_probe(memory_coordinator, 200) == []

    def test_raising_corpus_manager_degrades_to_empty(self):
        from core.prompt.builder import _fetch_session_probe

        corpus_manager = MagicMock()
        corpus_manager.get_recent_memories = MagicMock(side_effect=KeyError("boom"))
        memory_coordinator = MagicMock(corpus_manager=corpus_manager)

        assert _fetch_session_probe(memory_coordinator, 200) == []


class TestSessionTruthFromProbe:
    def _probe(self, ts_list):
        return [{"query": f"q{i}", "timestamp": ts.isoformat()} for i, ts in enumerate(ts_list)]

    def test_single_open_session_is_truncated(self):
        """10 shown turns, but the probe (going further back) finds the
        session actually has more turns before hitting the 2h+ gap — this is
        the evidence-6 shape."""
        from core.prompt.builder import _session_truth_from_probe

        now = datetime.now().replace(hour=21, minute=4, second=0, microsecond=0)
        # 15 entries 3 minutes apart, newest-first (no boundary at all within
        # this slice) — simulates "the session actually has more turns".
        ts_list = [now - timedelta(minutes=3 * i) for i in range(15)]
        probe = self._probe(ts_list)

        truncated, total, started_at = _session_truth_from_probe(probe, shown_count=10, now=ts_list[0])
        assert truncated is True
        assert total == 15
        assert started_at == ts_list[-1]

    def test_boundary_within_probe_stops_the_count(self):
        from core.prompt.builder import _session_truth_from_probe

        now = datetime.now().replace(hour=21, minute=4, second=0, microsecond=0)
        # 5 turns this session, then a 3h gap into an older session.
        this_session = [now - timedelta(minutes=3 * i) for i in range(5)]
        older_session = [now - timedelta(hours=5, minutes=3 * i) for i in range(5)]
        probe = self._probe(this_session + older_session)

        truncated, total, started_at = _session_truth_from_probe(probe, shown_count=10, now=this_session[0])
        assert total == 5
        assert truncated is False  # the whole (small) session fits the window
        assert started_at == this_session[-1]

    def test_shown_count_equal_to_total_is_not_truncated(self):
        from core.prompt.builder import _session_truth_from_probe

        # Pinned to midday: a real now() within 15 min after midnight put the
        # 6 turns across a calendar day, which is itself a session boundary
        # (CI failure 2026-09-30 00:10 UTC, PR #42).
        now = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        ts_list = [now - timedelta(minutes=3 * i) for i in range(6)]
        probe = self._probe(ts_list)

        truncated, total, _ = _session_truth_from_probe(probe, shown_count=6, now=ts_list[0])
        assert total == 6
        assert truncated is False

    def test_empty_probe(self):
        from core.prompt.builder import _session_truth_from_probe

        assert _session_truth_from_probe([], shown_count=5) == (False, None, None)

    def test_stale_newest_turn_means_a_new_session(self):
        """Newest stored turn 14h ago: this turn opens a NEW session — the
        previous session's start must not be reported as current."""
        from core.prompt.builder import _session_truth_from_probe

        now = datetime(2026, 9, 27, 15, 9)
        ts_list = [datetime(2026, 9, 27, 1, 16) - timedelta(minutes=3 * i) for i in range(15)]
        probe = self._probe(ts_list)
        assert _session_truth_from_probe(probe, shown_count=3, now=now) == (False, None, None)


class TestTodayTimelineFromProbe:
    def test_excludes_shown_and_other_days_oldest_first(self):
        from core.prompt.builder import _today_timeline_from_probe
        from core.prompt.hygiene import _canonical_turn_key

        today = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)

        shown_entry = {"query": "shown one", "response": "", "timestamp": today.replace(hour=20).isoformat()}
        earlier_1 = {"query": "sat down at the desk", "response": "", "timestamp": today.replace(hour=18, minute=23).isoformat()}
        earlier_2 = {"query": "grabbed a snack", "response": "", "timestamp": today.replace(hour=19, minute=5).isoformat()}
        old_day = {"query": "yesterday's turn", "response": "", "timestamp": yesterday.isoformat()}

        probe = [shown_entry, earlier_2, earlier_1, old_day]
        shown_keys = {_canonical_turn_key(shown_entry)}

        lines = _today_timeline_from_probe(probe, shown_keys, cap=40)
        assert lines == ["18:23 sat down at the desk", "19:05 grabbed a snack"]

    def test_cap_keeps_the_entries_closest_to_now(self):
        from core.prompt.builder import _today_timeline_from_probe

        today = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        # 45 today-candidates, one minute apart, none shown.
        probe = [
            {"query": f"turn {i}", "response": "", "timestamp": (today + timedelta(minutes=i)).isoformat()}
            for i in range(45)
        ]
        lines = _today_timeline_from_probe(probe, shown_keys=set(), cap=40)
        assert len(lines) == 40
        # The kept set must be the 40 NEWEST (turns 5..44), not the 40 oldest
        # (BC-18: a bare items[:40] on a newest-first probe keeps the wrong end).
        assert "turn 4 " not in "\n".join(lines)
        assert any("turn 44" in line for line in lines)
        # Rendered oldest -> newest.
        first_turn_num = int(lines[0].split("turn ")[1])
        last_turn_num = int(lines[-1].split("turn ")[1])
        assert first_turn_num < last_turn_num

    def test_no_candidates_returns_empty(self):
        from core.prompt.builder import _today_timeline_from_probe

        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        probe = [{"query": "old", "response": "", "timestamp": yesterday}]
        assert _today_timeline_from_probe(probe, shown_keys=set()) == []


# ---------------------------------------------------------------------------
# 5. intent_classifier — wh-time recall shape + sentence-scoped adverb fix
# ---------------------------------------------------------------------------

class TestWhTimeRecallShape:
    @pytest.fixture(scope="class")
    def clf(self):
        from core.intent_classifier import IntentClassifier
        return IntentClassifier()

    def test_what_time_did_i(self, clf):
        from core.intent_classifier import IntentType
        r = clf.classify("what time did i sit down")
        assert r.intent == IntentType.TEMPORAL_RECALL
        assert r.confidence >= 0.85

    def test_when_did_we(self, clf):
        from core.intent_classifier import IntentType
        r = clf.classify("when did we leave the house")
        assert r.intent == IntentType.TEMPORAL_RECALL
        assert r.confidence >= 0.85

    def test_what_time_mid_message(self, clf):
        from core.intent_classifier import IntentType
        r = clf.classify("hey quick one, what time did i sit down at my desk")
        assert r.intent == IntentType.TEMPORAL_RECALL


class TestTemporalAdverbSentenceScoping:
    @pytest.fixture(scope="class")
    def clf(self):
        from core.intent_classifier import IntentClassifier
        return IntentClassifier()

    def test_cross_sentence_question_mark_does_not_leak(self, clf):
        """The exact evidence-7 live message: the adverb sits in sentence 1,
        the '?' ends an unrelated sentence 2 — must NOT classify recall."""
        from core.intent_classifier import IntentType
        r = clf.classify("agi last year super low. Definitely under 20k I would say?")
        assert r.intent != IntentType.TEMPORAL_RECALL

    def test_same_sentence_still_fires(self, clf):
        from core.intent_classifier import IntentType
        r = clf.classify("What did we talk about last week?")
        assert r.intent == IntentType.TEMPORAL_RECALL
        assert r.confidence >= 0.85

    def test_two_sentence_message_with_cue_and_adverb_together_still_fires(self, clf):
        """A second, unrelated sentence must not SUPPRESS a genuine same-
        sentence match either — the fix is scoping, not blanket rejection."""
        from core.intent_classifier import IntentType
        r = clf.classify("Quick one. What did we talk about last week?")
        assert r.intent == IntentType.TEMPORAL_RECALL
        assert r.confidence >= 0.85

    def test_narration_without_cue_still_excluded(self, clf):
        from core.intent_classifier import IntentType
        r = clf.classify("Biscuit turned 2 last week, can't believe it.")
        assert r.intent != IntentType.TEMPORAL_RECALL

    def test_pure_function_matches_direct(self):
        from core.intent_classifier import _temporal_adverb_recall_shaped

        assert _temporal_adverb_recall_shaped("What did we talk about last week?") is True
        assert _temporal_adverb_recall_shaped(
            "agi last year super low. Definitely under 20k I would say?"
        ) is False
        assert _temporal_adverb_recall_shaped("Biscuit turned 2 last week, can't believe it.") is False


# ---------------------------------------------------------------------------
# 6. token_manager.PRIORITY_ORDER carries rows for the new context keys
# ---------------------------------------------------------------------------

class TestPriorityOrderCoversSessionTruthKeys:
    def test_rows_present(self):
        from core.prompt.token_manager import PRIORITY_ORDER

        names = {name for name, _ in PRIORITY_ORDER}
        for key in (
            "session_timeline", "recent_window_truncated",
            "session_turns_total", "session_started_at",
        ):
            assert key in names, f"{key} missing from PRIORITY_ORDER"
