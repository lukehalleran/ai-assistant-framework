"""
# utils/time_manager.py

Module Contract
- Purpose: Track last query/response timing, active days for memory decay, and provide formatted timestamps used across the app.
- Inputs/Outputs:
  - current(), current_iso(), mark_query_time(), measure_response(start, end)
  - get_active_days_since(timestamp), calculate_active_day_decay(timestamp, decay_rate)
- Side effects:
  - Maintains in‑memory timestamps for latency reporting.
  - Tracks active days in persistent storage for decay calculation.
"""
import json, os, logging
from datetime import datetime, date, timedelta

logger = logging.getLogger(__name__)

class TimeManager:
    def __init__(self, time_file="data/last_query_time.json"):
        self.time_file = time_file
        self.last_query_time = self._load_last_query_time()
        self.last_response_time = None
        # Active days tracking for memory decay
        self.active_days_file = "data/active_days.json"
        self.active_days = self._load_active_days()

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

    # ---------- active days tracking ----------
    def _load_active_days(self) -> set:
        """Load set of active days from persistent storage"""
        if os.path.exists(self.active_days_file):
            try:
                with open(self.active_days_file, "r") as f:
                    data = json.load(f)
                    return set(data.get("active_days", []))
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to load active days: {e}")
        return set()

    def _save_active_days(self):
        """Save set of active days to persistent storage"""
        try:
            os.makedirs(os.path.dirname(self.active_days_file), exist_ok=True)
            with open(self.active_days_file, "w") as f:
                json.dump({"active_days": sorted(list(self.active_days))}, f)
        except Exception as e:
            logger.warning(f"Failed to save active days: {e}")

    def _register_active_day(self, query_time: datetime = None):
        """Register the current day (or given day) as an active day"""
        if query_time is None:
            query_time = self.last_query_time or self.current()

        day_str = query_time.date().isoformat()
        self.active_days.add(day_str)
        self._save_active_days()

    def get_active_days_since(self, timestamp) -> int:
        """
        Count active days since given timestamp.

        Args:
            timestamp: datetime to count from

        Returns:
            Number of active days (days with at least one message)
        """
        if not timestamp:
            return 0

        start_date = timestamp.date()
        current_date = self.current().date()

        # Count active days in the range including today if today is active
        # but exclude the creation day itself (since memory was created during an active day)
        active_count = 0
        check_date = start_date + timedelta(days=1)

        while check_date <= current_date:
            if check_date.isoformat() in self.active_days:
                active_count += 1
            # Move to next day
            check_date += timedelta(days=1)

        return active_count

    def calculate_active_day_decay(self, timestamp, decay_rate: float = 0.05) -> float:
        """
        Calculate decay factor based on active days rather than calendar days.

        Args:
            timestamp: When the memory was created
            decay_rate: Rate of decay per active day (default matches RECENCY_DECAY_RATE)

        Returns:
            Decay factor between 0.0 and 1.0
        """
        if not timestamp:
            return 1.0

        active_days = self.get_active_days_since(timestamp)

        # Use the same decay formula but with active_days instead of hours/days
        # Formula: 1.0 / (1.0 + decay_rate * active_days)
        return 1.0 / (1.0 + decay_rate * active_days)

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
        # Register this as an active day for decay calculation
        self._register_active_day(self.last_query_time)
        return self.last_query_time

    def measure_response(self, start_time, end_time):
        elapsed = end_time - start_time
        self.last_response_time = elapsed
        return f"{elapsed.total_seconds():.2f} s"

    def last_response(self) -> str:
        return f"{self.last_response_time.total_seconds():.2f} s" if self.last_response_time else "N/A"

