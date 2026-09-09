"""Append-only curation audit journal (logs/curation_audit.jsonl).

Same doctrine as logs/actions_audit.jsonl: every scan, proposal, apply,
dismiss, and undo is journaled; the journal is never rewritten, only
appended. The Activity view in the Curation Center reads the tail.
"""

import json
import os
import threading
from collections import deque
from datetime import datetime
from typing import Any, Dict, List

from utils.logging_utils import get_logger

logger = get_logger("curation_journal")

_PROD_JOURNAL_PATH = os.path.join("logs", "curation_audit.jsonl")
_TEST_JOURNAL_PATH = os.path.join("logs", "test_curation_audit.jsonl")
_DEFAULT_JOURNAL_PATH = _PROD_JOURNAL_PATH


def resolve_journal_path(path: str = "") -> str:
    """Explicit path wins; otherwise the (sandbox-repointable) module default.

    A test process (DAEMON_TEST_MODE) that still resolves to the PROD journal
    is redirected to a test-only file, so no test can append to the owner's
    curation audit even when the conftest sandbox is bypassed (subprocess,
    script, ad-hoc run) — 2026-09-02 defence in depth.
    """
    resolved = path or _DEFAULT_JOURNAL_PATH
    if not path and os.getenv("DAEMON_TEST_MODE") and resolved == _PROD_JOURNAL_PATH:
        return _TEST_JOURNAL_PATH
    return resolved


class CurationJournal:
    def __init__(self, path: str = ""):
        # Late-bound default so the test sandbox can repoint the module attr.
        self.path = resolve_journal_path(path)
        self._lock = threading.RLock()

    def record(self, event: str, **detail: Any) -> None:
        """Append and sync a recovery record. Failure MUST reach the engine.

        The leading newline separates a prior torn append from this record;
        readers ignore blank/torn records. A target mutation is never started
        unless its prepare record and queue pre-image both synced successfully.
        """
        line = {"ts": datetime.now().isoformat(), "event": event, **detail}
        encoded = json.dumps(line, ensure_ascii=False, default=str)
        with self._lock:
            directory = os.path.dirname(self.path) or "."
            os.makedirs(directory, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write("\n" + encoded + "\n")
                f.flush()
                os.fsync(f.fileno())
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def records(self):
        """Stream complete records without loading the whole journal."""
        with self._lock:
            try:
                f = open(self.path, "r", encoding="utf-8")
            except FileNotFoundError:
                return
            with f:
                for raw in f:
                    if not raw.strip():
                        continue
                    try:
                        row = json.loads(raw)
                    except ValueError:
                        logger.warning("[CurationJournal] Ignoring torn record")
                        continue
                    if isinstance(row, dict):
                        yield row

    def tail(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Most-recent-first tail for the Activity view."""
        if limit <= 0:
            return []
        try:
            return list(reversed(deque(self.records(), maxlen=limit)))
        except OSError as e:
            logger.warning(f"[CurationJournal] read failed: {e}")
            return []
