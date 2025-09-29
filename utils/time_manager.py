"""
# utils/time_manager.py

Module Contract
- Purpose: Track last query/response timing and provide formatted timestamps used across the app.
- Inputs/Outputs:
  - current(), current_iso(), mark_query_time(), measure_response(start, end)
- Side effects:
  - Maintains in‑memory timestamps for latency reporting.
"""
import json, os, logging
from datetime import datetime

logger = logging.getLogger(__name__)

class TimeManager:
    def __init__(self, time_file="data/last_query_time.json"):
        self.time_file = time_file
        self.last_query_time = self._load_last_query_time()
        self.last_response_time = None

    # ---------- persistence ----------
    def _load_last_query_time(self):
        if os.path.exists(self.time_file):
            try:
                with open(self.time_file, "r") as f:
                    return datetime.fromisoformat(json.load(f)["last_query_time"])
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        return None

    def _save_last_query_time(self):
        if self.last_query_time:
            with open(self.time_file, "w") as f:
                json.dump({"last_query_time": self.last_query_time.isoformat()}, f)

    # ---------- public helpers ----------
    def current(self) -> datetime:
        return datetime.now()

    def current_iso(self) -> str:
        return self.current().isoformat(sep=" ", timespec="seconds")

    def elapsed_since_last(self) -> str:
        if not self.last_query_time:
            return "N/A (first query)"
        delta = self.current() - self.last_query_time
        if delta.days:
            return f"{delta.days} d {delta.seconds//3600} h"
        if delta.seconds >= 3600:
            return f"{delta.seconds//3600} h {(delta.seconds%3600)//60} m"
        if delta.seconds >= 60:
            return f"{delta.seconds//60} m"
        return f"{delta.seconds} s"

    def mark_query_time(self) -> datetime:
        """Call at the *start* of request handling."""
        self.last_query_time = self.current()
        self._save_last_query_time()
        return self.last_query_time

    def measure_response(self, start_time, end_time):
        elapsed = end_time - start_time
        self.last_response_time = elapsed
        return f"{elapsed.total_seconds():.2f} s"

    def last_response(self) -> str:
        return f"{self.last_response_time.total_seconds():.2f} s" if self.last_response_time else "N/A"

