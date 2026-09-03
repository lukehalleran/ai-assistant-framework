"""Append-only curation audit journal (logs/curation_audit.jsonl).

Same doctrine as logs/actions_audit.jsonl: every scan, proposal, apply,
dismiss, and undo is journaled; the journal is never rewritten, only
appended. The Activity view in the Curation Center reads the tail.
"""

import json
import os
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

    def record(self, event: str, **detail: Any) -> None:
        """Append one event line. Best-effort: journaling must never break
        the engine (a failed audit line is logged, not raised)."""
        line = {"ts": datetime.now().isoformat(), "event": event, **detail}
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning(f"[CurationJournal] append failed (non-fatal): {e}")

    def tail(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Most-recent-first tail for the Activity view."""
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as e:
            logger.warning(f"[CurationJournal] read failed: {e}")
            return []
        out: List[Dict[str, Any]] = []
        for raw in reversed(lines[-limit * 2:]):
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except ValueError:
                continue  # lenient: a torn tail line never breaks the view
            if len(out) >= limit:
                break
        return out
