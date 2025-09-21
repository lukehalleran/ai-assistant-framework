
# /utils/conversation_logger.py
import json
from datetime import datetime
from pathlib import Path
import threading
from typing import Optional, Dict, Any

class ConversationLogger:
    """
    A dedicated logger for human-readable conversation transcripts.
    Separate from the debug/system logging to maintain clarity.
    """

    def __init__(self,
                 log_dir: str = "conversation_logs",
                 log_format: str = "text",  # "text" or "json"
                 max_file_size_mb: int = 10):
        """
        Initialize the conversation logger.

        Args:
            log_dir: Directory to store conversation logs
            log_format: Format for logs ("text" for readable, "json" for structured)
            max_file_size_mb: Max size before rotating to new file
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_format = log_format
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

        # Thread safety
        self.lock = threading.Lock()

        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_count = 0

        # Initialize log file
        self.current_log_file = self._get_log_filename()
        self._write_session_header()

    def _get_log_filename(self, index: int = 0) -> Path:
        """Generate a filename for the current session."""
        if self.log_format == "json":
            extension = "jsonl"
        else:
            extension = "txt"

        if index == 0:
            filename = f"conversation_{self.session_id}.{extension}"
        else:
            filename = f"conversation_{self.session_id}_part{index}.{extension}"

        return self.log_dir / filename

    def _check_rotation(self):
        """Check if log file needs rotation due to size."""
        if self.current_log_file.exists():
            size = self.current_log_file.stat().st_size
            if size > self.max_file_size_bytes:
                # Find next available index
                index = 1
                while (self.log_dir / f"conversation_{self.session_id}_part{index}.txt").exists():
                    index += 1
                self.current_log_file = self._get_log_filename(index)
                self._write_session_header()

    def _write_session_header(self):
        """Write a header when starting a new log file."""
        with self.lock:
            if self.log_format == "text":
                with open(self.current_log_file, 'a', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write(f"DAEMON CONVERSATION LOG\n")
                    f.write(f"Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Session ID: {self.session_id}\n")
                    f.write("=" * 80 + "\n\n")
            # JSON format doesn't need headers

    def log_interaction(self,
                        user_input: str,
                        assistant_response: str,
                        metadata: Optional[Dict[str, Any]] = None):
        """
        Log a single interaction between user and assistant.

        Args:
            user_input: The user's query/message
            assistant_response: The assistant's response
            metadata: Optional metadata (files, personality, topic, etc.)
        """
        with self.lock:
            self._check_rotation()
            self.conversation_count += 1

            timestamp = datetime.now()

            if self.log_format == "json":
                self._log_json(timestamp, user_input, assistant_response, metadata)
            else:
                self._log_text(timestamp, user_input, assistant_response, metadata)

    def _log_text(self,
                  timestamp: datetime,
                  user_input: str,
                  assistant_response: str,
                  metadata: Optional[Dict] = None):
        """Write human-readable text format."""
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            # Write timestamp and conversation number
            f.write(f"--- Conversation #{self.conversation_count} ---\n")
            f.write(f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")

            # Write metadata if present
            if metadata:
                if metadata.get('topic'):
                    f.write(f"Topic: {metadata['topic']}\n")
                if metadata.get('personality'):
                    f.write(f"Personality: {metadata['personality']}\n")
                if metadata.get('files'):
                    f.write(f"Files: {', '.join(metadata['files'])}\n")
                if metadata.get('mode'):
                    f.write(f"Mode: {metadata['mode']}\n")

            # Write the conversation
            f.write("\nUSER:\n")
            f.write(f"{user_input}\n")
            f.write("\nASSISTANT:\n")
            f.write(f"{assistant_response}\n")
            f.write("\n" + "-" * 40 + "\n\n")
            f.flush()

    def _log_json(self,
                  timestamp: datetime,
                  user_input: str,
                  assistant_response: str,
                  metadata: Optional[Dict] = None):
        """Write structured JSON format (one JSON object per line)."""
        entry = {
            "conversation_id": self.conversation_count,
            "timestamp": timestamp.isoformat(),
            "user_input": user_input,
            "assistant_response": assistant_response,
            "metadata": metadata or {}
        }

        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            f.flush()

    def log_system_event(self, event: str, details: Optional[str] = None):
        """Log system events like topic switches, summaries, etc."""
        with self.lock:
            timestamp = datetime.now()

            if self.log_format == "text":
                with open(self.current_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[SYSTEM EVENT] {timestamp.strftime('%H:%M:%S')} - {event}\n")
                    if details:
                        f.write(f"  Details: {details}\n")
                    f.write("\n")
                    f.flush()
            else:
                entry = {
                    "type": "system_event",
                    "timestamp": timestamp.isoformat(),
                    "event": event,
                    "details": details
                }
                with open(self.current_log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry) + '\n')
                    f.flush()

    def get_current_log_path(self) -> Path:
        """Return the path to the current log file."""
        return self.current_log_file

    def get_session_stats(self) -> Dict:
        """Return statistics about the current session."""
        return {
            "session_id": self.session_id,
            "conversation_count": self.conversation_count,
            "current_log_file": str(self.current_log_file),
            "log_size_bytes": self.current_log_file.stat().st_size if self.current_log_file.exists() else 0
        }

# Singleton instance for global access
_conversation_logger = None

def get_conversation_logger(log_dir: str = "conversation_logs",
                           log_format: str = "text") -> ConversationLogger:
    """Get or create the singleton conversation logger."""
    global _conversation_logger
    if _conversation_logger is None:
        _conversation_logger = ConversationLogger(log_dir=log_dir, log_format=log_format)
    return _conversation_logger
