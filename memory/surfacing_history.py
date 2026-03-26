# memory/surfacing_history.py
"""
JSON-backed novelty tracking for proactive context surfacing.

Records which cross-domain insights have been shown to the user and when,
so the same connection isn't surfaced repeatedly.  Persisted as a simple
JSON dict keyed by novelty_key.
"""

import json
import os
from datetime import datetime, timedelta

from utils.logging_utils import get_logger

logger = get_logger("surfacing_history")


class SurfacingHistory:
    """Track which insights have been surfaced and when."""

    def __init__(self, persist_path: str = "data/surfacing_history.json"):
        self.persist_path = persist_path
        self._entries: dict[str, dict] = {}
        self.load()

    def was_recently_shown(self, novelty_key: str, cooldown_hours: int = 72) -> bool:
        """Return True if this insight was surfaced within the cooldown window."""
        entry = self._entries.get(novelty_key)
        if not entry:
            return False
        try:
            last = datetime.fromisoformat(entry["last_surfaced"])
            return datetime.now() - last < timedelta(hours=cooldown_hours)
        except (KeyError, ValueError):
            return False

    def record_surfaced(self, novelty_key: str) -> None:
        """Record that an insight was surfaced now."""
        entry = self._entries.get(novelty_key, {"count": 0})
        entry["last_surfaced"] = datetime.now().isoformat()
        entry["count"] = entry.get("count", 0) + 1
        self._entries[novelty_key] = entry
        self.save()

    def cleanup_old(self, max_age_days: int = 30) -> int:
        """Remove entries older than max_age_days. Returns count removed."""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        to_remove = []
        for key, entry in self._entries.items():
            try:
                last = datetime.fromisoformat(entry.get("last_surfaced", ""))
                if last < cutoff:
                    to_remove.append(key)
            except (ValueError, TypeError):
                to_remove.append(key)

        for key in to_remove:
            del self._entries[key]

        if to_remove:
            self.save()
            logger.debug(f"[SurfacingHistory] Cleaned up {len(to_remove)} stale entries")
        return len(to_remove)

    def load(self) -> None:
        """Load history from disk."""
        if not os.path.exists(self.persist_path):
            self._entries = {}
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                self._entries = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[SurfacingHistory] Failed to load {self.persist_path}: {e}")
            self._entries = {}

    def save(self) -> None:
        """Persist history to disk."""
        try:
            os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.warning(f"[SurfacingHistory] Failed to save {self.persist_path}: {e}")
