"""Tests for CorpusManager.search_keyword — the insight sweep's raw-text
channel (defeats the fact extractor's triple-shape bias)."""

from datetime import datetime, timedelta

import pytest

from memory.corpus_manager import CorpusManager


@pytest.fixture
def corpus(tmp_path):
    cm = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
    now = datetime(2026, 8, 20, 12, 0, 0)
    cm.add_entry(
        "She was the only partner I've had who wasn't abusive.",
        "That sounds like an important distinction to you.",
        timestamp=now - timedelta(days=2),
    )
    cm.add_entry(
        "Slept badly again.",
        "Poor sleep can compound stress — abusive self-talk often follows.",
        timestamp=now - timedelta(days=1),
    )
    cm.add_entry(
        "What's the weather like?",
        "Sunny and mild today.",
        timestamp=now,
    )
    # summary node must never surface in keyword search (episodic-only)
    cm.add_summary("User discussed abusive relationship patterns.", timestamp=now)
    return cm


class TestSearchKeyword:
    def test_speaker_attribution(self, corpus):
        hits = corpus.search_keyword("abusive")
        speakers = {(h["speaker"], h["matched_term"]) for h in hits}
        assert ("user", "abusive") in speakers
        assert ("assistant", "abusive") in speakers
        # each hit's excerpt comes from the right side
        for h in hits:
            if h["speaker"] == "user":
                assert "wasn't abusive" in h["excerpt"]

    def test_newest_first(self, corpus):
        hits = corpus.search_keyword("abusive")
        timestamps = [h["timestamp"] for h in hits]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_time_window(self, corpus):
        now = datetime(2026, 8, 20, 12, 0, 0)
        hits = corpus.search_keyword("abusive", start=now - timedelta(hours=36))
        # only the day-old assistant mention is inside the window
        assert len(hits) == 1
        assert hits[0]["speaker"] == "assistant"

        hits = corpus.search_keyword("abusive", end=now - timedelta(hours=36))
        assert len(hits) == 1
        assert hits[0]["speaker"] == "user"

    def test_max_results_cap(self, corpus):
        for i in range(10):
            corpus.add_entry(f"sleep note {i}", "ok", timestamp=datetime(2026, 8, 21, i))
        hits = corpus.search_keyword("sleep", max_results=3)
        assert len(hits) == 3

    def test_word_boundary(self, corpus):
        corpus.add_entry("I visited the sleepy hollow museum.", "Nice.",
                         timestamp=datetime(2026, 8, 21))
        hits = corpus.search_keyword("sleep")
        assert all("sleepy" not in h["excerpt"] or "sleep " in h["excerpt"].lower()
                   for h in hits)
        # 'sleepy' alone must not match the term 'sleep'
        only_sleepy = corpus.search_keyword("slee")
        assert only_sleepy == []

    def test_summaries_excluded(self, corpus):
        hits = corpus.search_keyword("patterns")
        assert hits == []  # only the summary node contains 'patterns'

    def test_multiple_terms(self, corpus):
        hits = corpus.search_keyword(["weather", "sunny"])
        assert {h["speaker"] for h in hits} == {"user", "assistant"}

    def test_junk_turns_skipped(self, corpus):
        corpus.add_entry("test", "test", timestamp=datetime(2026, 8, 21))
        assert corpus.search_keyword("test") == []

    def test_empty_terms(self, corpus):
        assert corpus.search_keyword([]) == []
        assert corpus.search_keyword("") == []

    def test_excerpt_clipping(self, corpus):
        long_text = ("filler " * 100) + "abusive" + (" filler" * 100)
        corpus.add_entry(long_text, "ok", timestamp=datetime(2026, 8, 21))
        hits = corpus.search_keyword("abusive", context_chars=100)
        hit = [h for h in hits if h["timestamp"] == datetime(2026, 8, 21)][0]
        assert len(hit["excerpt"]) <= 120  # window + ellipses slack
        assert "abusive" in hit["excerpt"]
        assert hit["excerpt"].startswith("…") and hit["excerpt"].endswith("…")

    def test_read_only(self, corpus):
        before = len(corpus.corpus)
        corpus.search_keyword("abusive")
        assert len(corpus.corpus) == before
