"""Deterministic pattern engine (memory/pattern_engine.py, 2026-08-29).

Drives the DEPLOYED engine against real components on tmp paths (deployed-fn
doctrine): a real CorpusManager loaded from a synthetic corpus file, a real
UserProfile, and a synthetic telemetry JSONL. `now` is injected everywhere —
no wall-clock dependence.

The synthetic corpus reproduces the live motivating case: song-lyric pastes
on two consecutive days (2026-08-28 + 08-29) that the in-context synthesis
noticed but no store-wide system could have.
"""

import json
from datetime import datetime, timedelta

import pytest

from memory.corpus_manager import CorpusManager
from memory.pattern_engine import (
    PatternQuery,
    _auto_bucket,
    _bucket_starts,
    _streak_and_gap,
    _trend,
    PatternBucket,
    run_pattern_query,
)

NOW = datetime(2026, 8, 29, 15, 0, 0)

LYRICS_PASTE = (
    "Somethings happening idk. I am listen to this. "
    + "Draw me a person, draw me a leader\n" * 12
    + "Chorus lyrics verse bridge outro\n" * 6
)


def _mk_corpus(tmp_path, entries):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(entries))
    return CorpusManager(corpus_file=str(path))


def _entry(ts, query, response="ok", **extra):
    return {"query": query, "response": response,
            "timestamp": ts.isoformat(), **extra}


@pytest.fixture
def corpus(tmp_path):
    entries = []
    # background chatter across 3 weeks
    for d in range(20, 0, -1):
        entries.append(_entry(NOW - timedelta(days=d, hours=2),
                              f"regular message day minus {d}"))
    # "itch" mentions: 3 in the last week (user side), 1 assistant echo
    for d in (6, 3, 1):
        entries.append(_entry(NOW - timedelta(days=d),
                              "I was very itchy again last night",
                              response="noted"))
    entries.append(_entry(NOW - timedelta(days=2), "hello",
                          response="is the itch better today?"))
    # song pastes on two consecutive days (the live motivating case)
    entries.append(_entry(NOW - timedelta(days=1), LYRICS_PASTE))
    entries.append(_entry(NOW - timedelta(hours=2), LYRICS_PASTE))
    return _mk_corpus(tmp_path, entries)


class TestHelpers:
    def test_auto_bucket(self):
        assert _auto_bucket(7) == "day"
        assert _auto_bucket(90) == "week"
        assert _auto_bucket(365) == "month"

    def test_bucket_starts_cover_window_with_empty_buckets(self):
        starts = _bucket_starts(NOW - timedelta(days=13), NOW, "day")
        assert len(starts) == 14

    def test_streak_and_gap(self):
        days = {datetime(2026, 8, d).date() for d in (1, 2, 3, 10)}
        streak, gap = _streak_and_gap(days)
        assert streak == 3
        assert gap == 6

    def test_trend_shapes(self):
        def bks(counts):
            return [PatternBucket(start="x", count=c) for c in counts]
        assert _trend(bks([0, 0, 3, 5])) == "increasing"
        assert _trend(bks([5, 3, 0, 0])) == "decreasing"
        assert _trend(bks([2, 2, 2, 2])) == "stable"
        assert _trend(bks([1, 0, 1, 0])) == "insufficient"  # total < 4


class TestTopicKeyword:
    def test_counts_user_side_only(self, corpus):
        res = run_pattern_query(
            PatternQuery(dimension="topic_keyword", terms=["itch", "itchy"],
                         window_days=14, now=NOW),
            corpus_manager=corpus)
        assert res.total == 3          # assistant echo excluded
        assert res.active_days == 3
        assert any("assistant echoes excluded" in n for n in res.notes)

    def test_both_speakers_when_requested(self, corpus):
        res = run_pattern_query(
            PatternQuery(dimension="topic_keyword", terms=["itch", "itchy"],
                         window_days=14, speaker="both", now=NOW),
            corpus_manager=corpus)
        assert res.total == 4

    def test_denominator_present(self, corpus):
        res = run_pattern_query(
            PatternQuery(dimension="topic_keyword", terms=["itchy"],
                         window_days=14, now=NOW),
            corpus_manager=corpus)
        assert res.denominator_total > res.total  # more turns than mentions
        assert any(b.denominator for b in res.buckets)

    def test_exemplars_are_real_dated_excerpts(self, corpus):
        res = run_pattern_query(
            PatternQuery(dimension="topic_keyword", terms=["itchy"],
                         window_days=14, now=NOW),
            corpus_manager=corpus)
        exemplars = [e for b in res.buckets for e in b.exemplars]
        assert exemplars
        for ex in exemplars:
            assert "itchy" in ex.text.lower()
            assert ex.date  # real store timestamp

    def test_window_excludes_older_hits(self, corpus):
        res = run_pattern_query(
            PatternQuery(dimension="topic_keyword", terms=["itchy"],
                         window_days=2, now=NOW),
            corpus_manager=corpus)
        assert res.total == 1  # only the day-1 mention


class TestContentType:
    def test_live_two_song_days_reproduced(self, corpus):
        res = run_pattern_query(
            PatternQuery(dimension="content_type", terms=["lyrics"],
                         window_days=7, now=NOW),
            corpus_manager=corpus)
        assert res.total == 2
        assert res.active_days == 2
        assert res.longest_streak_days == 2   # two CONSECUTIVE days
        assert res.first_seen == (NOW - timedelta(days=1)).date().isoformat()

    def test_regular_messages_not_counted(self, corpus):
        res = run_pattern_query(
            PatternQuery(dimension="content_type", terms=["code"],
                         window_days=7, now=NOW),
            corpus_manager=corpus)
        assert res.total == 0


class TestTone:
    def _telemetry(self, tmp_path, rows):
        path = tmp_path / "turns.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return str(path)

    def test_counts_elevated_and_skips_test_env(self, tmp_path):
        rows = [
            {"ts": (NOW - timedelta(days=5)).isoformat(), "tone_level": "CONCERN"},
            {"ts": (NOW - timedelta(days=4)).isoformat(), "tone_level": "CrisisLevel.MEDIUM"},
            {"ts": (NOW - timedelta(days=3)).isoformat(), "tone_level": "CONVERSATIONAL"},
            {"ts": (NOW - timedelta(days=2)).isoformat(), "tone_level": "HIGH",
             "test_env": True},
        ]
        res = run_pattern_query(
            PatternQuery(dimension="tone", window_days=7, now=NOW),
            telemetry_path=self._telemetry(tmp_path, rows))
        assert res.total == 2          # CONCERN + MEDIUM; test_env excluded
        assert {v for b in res.buckets for v in b.values} == {"concern", "medium"}

    def test_coverage_note_when_telemetry_starts_late(self, tmp_path):
        rows = [{"ts": (NOW - timedelta(days=2)).isoformat(),
                 "tone_level": "CONCERN"}]
        res = run_pattern_query(
            PatternQuery(dimension="tone", window_days=90, now=NOW),
            telemetry_path=self._telemetry(tmp_path, rows))
        assert any("coverage starts" in n for n in res.notes)

    def test_missing_telemetry_degrades_with_note(self, tmp_path):
        res = run_pattern_query(
            PatternQuery(dimension="tone", window_days=7, now=NOW),
            telemetry_path=str(tmp_path / "missing.jsonl"))
        assert res.total == 0
        assert any("no telemetry" in n for n in res.notes)


class TestRelation:
    def test_value_timeline_with_appraisal_label(self, tmp_path):
        from memory.user_profile import UserProfile
        profile = UserProfile(profile_path=str(tmp_path / "profile.json"))
        profile.add_fact("sleep_quality", "bad sleep",
                         timestamp=(NOW - timedelta(days=10)).isoformat())
        profile.add_fact("sleep_quality", "slept well",
                         timestamp=(NOW - timedelta(days=2)).isoformat())
        res = run_pattern_query(
            PatternQuery(dimension="relation", relation="sleep_quality",
                         window_days=30, now=NOW),
            user_profile=profile)
        assert res.total == 2
        values = [v for b in res.buckets for v in b.values]
        assert "bad sleep" in values and "slept well" in values


class TestSessionRhythm:
    def test_message_counts_and_session_values(self, corpus):
        res = run_pattern_query(
            PatternQuery(dimension="session_rhythm", window_days=14, now=NOW),
            corpus_manager=corpus)
        assert res.total > 0                       # message volume
        assert any("sessions:" in v for b in res.buckets for v in b.values)


class TestRenderTable:
    def test_table_carries_counts_dates_denominator(self, corpus):
        res = run_pattern_query(
            PatternQuery(dimension="topic_keyword", terms=["itchy"],
                         window_days=14, now=NOW),
            corpus_manager=corpus)
        table = res.render_table()
        assert "total=3" in table
        assert "user turns in window" in table
        assert res.since in table


class TestToneFloorExclusion:
    def test_floored_turns_excluded_with_note(self, tmp_path):
        rows = [
            {"ts": (NOW - timedelta(days=3)).isoformat(),
             "tone_level": "CONCERN", "tone_trigger": "semantic"},
            {"ts": (NOW - timedelta(days=2)).isoformat(),
             "tone_level": "CONCERN", "tone_trigger": "distress_sticky_floor"},
            {"ts": (NOW - timedelta(days=1)).isoformat(),
             "tone_level": "CONCERN", "tone_trigger": "distress_sticky_floor"},
        ]
        path = tmp_path / "turns.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        res = run_pattern_query(
            PatternQuery(dimension="tone", window_days=7, now=NOW),
            telemetry_path=str(path))
        assert res.total == 1          # only the organic CONCERN counts
        assert any("floored" in n and "excluded" in n for n in res.notes)


class TestDetectorLiveShapes:
    """The two live song-paste shapes the structural heuristic missed
    (2026-08-29 smoke against real data found 0 lyric hits in a week that
    contained two song shares)."""

    def test_genius_page_with_section_headers(self):
        from core.content_type_detector import detect_content_type
        text = (
            "Somethings happening idk. I am listen to this. ITS NOT RIGHT\n"
            "Draw Me A Person Lyrics\n[Verse 1]\nDraw me a person, draw me a "
            "leader\n[Chorus]\nWhat do you do for work? Do you have kids?\n"
            "Q&A\nWho produced the song?\nWhen was it released?\n"
            + "page chrome Buy now $229\n" * 20
        )
        ct = detect_content_type(text)
        assert ct.content_type == "lyrics"
        assert ct.confidence >= 0.85

    def test_runon_paste_with_song_frame(self):
        from core.content_type_detector import detect_content_type
        text = (
            "Lately, I think I was over Time am I just beaten so ? "
            + "the heavy heart I carried went over your head and over mine "
            * 30
            + " this song is making me think of when I moved to Atlanta"
        )
        assert len(text) >= 1200
        ct = detect_content_type(text)
        assert ct.content_type == "lyrics"

    def test_short_song_remark_not_lyrics(self):
        from core.content_type_detector import detect_content_type
        ct = detect_content_type("this song slaps, listen to this one later")
        assert ct.content_type != "lyrics" or ct.confidence < 0.7


class TestDailyNotes:
    """Daily-note frontmatter is an INDEPENDENT per-day series (no
    double-counting) — summaries/reflections stay deliberately uncounted
    because they compress the same turns the corpus dimensions count."""

    def _vault(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        for d, intensity, emo in (
            (2, 8, "Heavy day — dad text and therapist email."),
            (1, 4, "Flat but steadier."),
        ):
            day = (NOW - timedelta(days=d)).date()
            name = f"{day.month} {day.day} {day.strftime('%y')} Daily Note.md"
            (vault / name).write_text(
                f"---\ndate: {day.isoformat()}\nusage_intensity: {intensity}\n"
                f"active_hours: 1.5\n---\n\n# Daily Note\n\n"
                f"## Summary\nstuff\n\n## Emotional State\n{emo}\n")
        return str(vault)

    def test_notes_counted_with_intensity_and_emotion(self, tmp_path):
        res = run_pattern_query(
            PatternQuery(dimension="daily_notes", window_days=7, now=NOW,
                         vault_path=self._vault(tmp_path)),
            corpus_manager=None)
        assert res.total == 2
        values = [v for b in res.buckets for v in b.values]
        assert "intensity 8" in values and "intensity 4" in values
        exemplars = [e for b in res.buckets for e in b.exemplars]
        assert any("Heavy day" in e.text for e in exemplars)
        assert any("no daily note" in n for n in res.notes)  # missing days noted

    def test_missing_vault_degrades_with_note(self, tmp_path):
        res = run_pattern_query(
            PatternQuery(dimension="daily_notes", window_days=7, now=NOW,
                         vault_path=str(tmp_path / "nope")))
        assert res.total == 0
        assert any("no daily note" in n or "vault" in n for n in res.notes)


class TestEmailDimension:
    """2026-09-01 email dimension (docs/EMAIL_INTEGRATION_DESIGN.md): the
    engine is sync and pure — live-fetched headers are INJECTED as rows by
    the async caller; None (not fetched) and [] (nothing in window) are
    reported honestly and never conflated."""

    def _rows(self, now):
        from core.email.provider import EmailMessage
        mk = lambda days_ago, sender, subject, unread=False: EmailMessage(
            provider="gmail", message_id=f"m{days_ago}",
            sender=sender, subject=subject, unread=unread,
            date=(now - timedelta(days=days_ago)).isoformat(),
        )
        return [
            mk(1, "Dr. Smith <smith@clinic.org>", "Appointment follow-up", True),
            mk(2, "Advisor <Morgan@gatech.edu>", "Registration"),
            mk(3, "News <digest@clinic.org>", "Weekly digest"),
            mk(40, "Old <old@old.com>", "Outside window"),
            {"provider": "outlook", "message_id": "d1",
             "sender": "HR <hr@work.com>", "subject": "Benefits",
             "date": (now - timedelta(days=4)).isoformat(), "unread": False},
        ]

    def test_counts_domains_and_window(self):
        from memory.pattern_engine import PatternQuery, run_pattern_query
        now = datetime(2026, 9, 1, 12, 0)
        res = run_pattern_query(
            PatternQuery(dimension="email", window_days=14, now=now),
            email_rows=self._rows(now))
        assert sum(b.count for b in res.buckets) == 4  # 40d-old row excluded
        notes = " ".join(res.notes)
        assert "clinic.org (2)" in notes
        assert "4 emails in window; 1 unread" in notes
        # exemplars are real refs with message ids
        exemplars = [e for b in res.buckets for e in b.exemplars]
        assert any(e.source == "email" and e.doc_id for e in exemplars)

    def test_none_rows_honest_unavailable(self):
        from memory.pattern_engine import PatternQuery, run_pattern_query
        res = run_pattern_query(
            PatternQuery(dimension="email", window_days=7,
                         now=datetime(2026, 9, 1)), email_rows=None)
        assert sum(b.count for b in res.buckets) == 0
        assert any("not available" in n for n in res.notes)

    def test_empty_rows_honest_zero(self):
        from memory.pattern_engine import PatternQuery, run_pattern_query
        res = run_pattern_query(
            PatternQuery(dimension="email", window_days=7,
                         now=datetime(2026, 9, 1)), email_rows=[])
        assert any("no emails found" in n for n in res.notes)

    def test_dimension_registered(self):
        from memory.pattern_engine import DIMENSIONS
        assert "email" in DIMENSIONS

    def test_temporal_stage_auto_arm(self):
        from core.insight.temporal import run_pattern_stage
        from core.insight.types import InsightIntent
        now = datetime(2026, 9, 1, 12, 0)
        intent = InsightIntent(kind="pattern_temporal",
                               theme="my email volume lately",
                               raw_query="how many emails am I getting lately",
                               window_days=14)
        results, evidence = run_pattern_stage(
            intent, corpus_manager=None, user_profile=None, now=now,
            email_rows=self._rows(now))
        email_res = [r for r in results if r.dimension == "email"]
        assert len(email_res) == 1
        assert sum(b.count for b in email_res[0].buckets) == 4
