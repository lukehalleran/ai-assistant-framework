# knowledge/wiki_tracker.py
"""
Session-level tracker for Wikipedia articles accessed during a session.

Collects (title, text_snippet) pairs with zero processing overhead at
query time.  At shutdown, the tracked titles are used by WikiGraphEnricher
to add encountered articles to the knowledge graph.

Thread-safe singleton — a single tracker instance per process.
"""

import threading
from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger("wiki_tracker")


class WikiArticleTracker:
    """Lightweight session-level Wikipedia article tracker."""

    _instance: Optional["WikiArticleTracker"] = None
    _init_lock = threading.Lock()

    def __init__(self):
        self._articles: dict[str, str] = {}  # title -> first text snippet
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "WikiArticleTracker":
        """Get or create the singleton tracker instance."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def track(self, title: str, text_snippet: str = "") -> None:
        """Record a Wikipedia article title.  O(1), thread-safe.

        Only stores the first text snippet seen for each title.
        """
        if not title or len(title) < 3:
            return
        with self._lock:
            if title not in self._articles:
                self._articles[title] = text_snippet[:500] if text_snippet else ""

    def get_tracked(self) -> dict[str, str]:
        """Return all tracked articles as {title: text_snippet}.

        Called once at shutdown by WikiGraphEnricher.
        """
        with self._lock:
            return dict(self._articles)

    def clear(self) -> None:
        """Reset for a new session."""
        with self._lock:
            self._articles.clear()

    @property
    def count(self) -> int:
        return len(self._articles)
