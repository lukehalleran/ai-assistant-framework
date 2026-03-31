# tests/unit/test_wiki_tracker.py
"""Unit tests for WikiArticleTracker."""

import threading

import pytest

from knowledge.wiki_tracker import WikiArticleTracker


@pytest.fixture(autouse=True)
def reset_tracker():
    """Reset the singleton tracker before each test."""
    WikiArticleTracker._instance = None
    yield
    if WikiArticleTracker._instance:
        WikiArticleTracker._instance.clear()
    WikiArticleTracker._instance = None


class TestWikiArticleTracker:
    def test_singleton(self):
        t1 = WikiArticleTracker.get_instance()
        t2 = WikiArticleTracker.get_instance()
        assert t1 is t2

    def test_track_basic(self):
        t = WikiArticleTracker.get_instance()
        t.track("Serotonin", "Serotonin is a neurotransmitter")
        assert t.count == 1
        tracked = t.get_tracked()
        assert "Serotonin" in tracked
        assert tracked["Serotonin"] == "Serotonin is a neurotransmitter"

    def test_track_dedup(self):
        t = WikiArticleTracker.get_instance()
        t.track("Serotonin", "First snippet")
        t.track("Serotonin", "Second snippet should be ignored")
        assert t.count == 1
        assert t.get_tracked()["Serotonin"] == "First snippet"

    def test_track_ignores_short_titles(self):
        t = WikiArticleTracker.get_instance()
        t.track("ab")
        t.track("")
        t.track("A")
        assert t.count == 0

    def test_track_truncates_snippet(self):
        t = WikiArticleTracker.get_instance()
        long_text = "x" * 1000
        t.track("Long Article", long_text)
        assert len(t.get_tracked()["Long Article"]) == 500

    def test_track_empty_snippet(self):
        t = WikiArticleTracker.get_instance()
        t.track("No Snippet")
        assert t.get_tracked()["No Snippet"] == ""

    def test_clear(self):
        t = WikiArticleTracker.get_instance()
        t.track("Article A", "text")
        t.track("Article B", "text")
        assert t.count == 2
        t.clear()
        assert t.count == 0
        assert t.get_tracked() == {}

    def test_get_tracked_returns_copy(self):
        t = WikiArticleTracker.get_instance()
        t.track("Article", "text")
        copy1 = t.get_tracked()
        copy1["Article"] = "modified"
        assert t.get_tracked()["Article"] == "text"

    def test_multiple_articles(self):
        t = WikiArticleTracker.get_instance()
        for i in range(20):
            t.track(f"Article {i}", f"Content {i}")
        assert t.count == 20

    def test_thread_safety(self):
        t = WikiArticleTracker.get_instance()
        errors = []

        def track_batch(start):
            try:
                for i in range(100):
                    t.track(f"Thread-{start}-Article-{i}", f"text-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=track_batch, args=(j,)) for j in range(5)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert len(errors) == 0
        assert t.count == 500  # 5 threads * 100 articles
