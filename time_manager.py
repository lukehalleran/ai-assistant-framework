"""
TimeManager - Tracks and reports timing information.

Supports:
- Current timestamp
- Elapsed time since last query
- Response time measurement
- Persistent time tracking across sessions
"""
import datetime
import os
import json

class TimeManager:
    def __init__(self, time_file="data/last_query_time.json"):
        self.time_file = time_file
        self.last_query_time = self._load_last_query_time()

    def _load_last_query_time(self):
        """Load last query time from file if it exists"""
        if os.path.exists(self.time_file):
            try:
                with open(self.time_file, 'r') as f:
                    data = json.load(f)
                    return datetime.datetime.fromisoformat(data['last_query_time'])
            except (json.JSONDecodeError, KeyError, ValueError):
                return None
        return None

    def _save_last_query_time(self):
        """Save last query time to file"""
        if self.last_query_time:
            with open(self.time_file, 'w') as f:
                json.dump({'last_query_time': self.last_query_time.isoformat()}, f)
    def get_last_response_time(self):
        """Get the last measured response time"""
        if hasattr(self, 'last_response_time'):
            return self.last_response_time
        return "N/A"

    def save_response_time(self, response_time):
        """Save the last response time"""
        self.last_response_time = response_time
    def get_current_datetime(self):
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")

    def get_elapsed_since_last(self):
        now = datetime.datetime.now()
        if self.last_query_time:
            elapsed = now - self.last_query_time

            # Format nicely
            if elapsed.days > 0:
                return f"{elapsed.days} days, {elapsed.seconds//3600} hours ago"
            elif elapsed.seconds > 3600:
                return f"{elapsed.seconds//3600} hours, {(elapsed.seconds%3600)//60} minutes ago"
            elif elapsed.seconds > 60:
                return f"{elapsed.seconds//60} minutes ago"
            else:
                return f"{elapsed.seconds} seconds ago"
        else:
            return "N/A (first query)"

    def mark_query_time(self):
        self.last_query_time = datetime.datetime.now()
        self._save_last_query_time()

    def measure_response_time(self, start_time, end_time):
        elapsed = end_time - start_time
        return f"{elapsed.total_seconds():.2f} seconds"
