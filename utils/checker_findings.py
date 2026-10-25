"""Module Contract -- utils/checker_findings.py

Purpose:
    ONE writer for the findings of the log-only checkers (post-answer review
    gate, grounding check, personal-claim check), so each finding can be
    precision-labelled later (2026-10-25, MEAS2; class: BC-58 unmeasured guard).
    Before this, the turn record kept only counts/booleans and the claim span,
    issues and evidence were persisted nowhere.

Outputs: one JSON line per finding in ``checker_findings.jsonl``, a SIBLING of
    the turn-records file (same directory resolution as ``utils/turn_telemetry``;
    under pytest the conftest sandbox moves that directory, so tests never touch
    prod logs).
Privacy: LOCAL-ONLY. Lines hold conversation excerpts. This file must never be
    read by a share surface (/api/debug, prompt export, diagnostic rollup,
    privacy_redaction paths) -- ``tests/unit/test_checker_findings_recording.py``
    pins that. Rotated by ``utils/log_rotation``.
Failure: best-effort. A write failure never reaches the turn.
Exclusion: probe turns (``test_probe`` or a ``[test]`` span in any text passed
    via ``probe_texts``) are never recorded.
Config: YAML ``turn_telemetry.checker_findings_enabled`` (read live).
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Iterable

from utils.logging_utils import get_logger

logger = get_logger("checker_findings")

FILENAME = "checker_findings.jsonl"
SPAN_MAX = 300
EXCERPT_MAX = 600
SHORT_MAX = 300
_TEST_SPAN_RE = re.compile(r"\[test\]", re.IGNORECASE)


def _bound(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def findings_path() -> str:
    """Sibling of the turn-records file (live read of the configured path)."""
    try:
        from config.app_config import TURN_TELEMETRY_PATH  # lazy import: live-config
    except ImportError:
        TURN_TELEMETRY_PATH = "logs/turn_records.jsonl"
    return os.path.join(os.path.dirname(TURN_TELEMETRY_PATH), FILENAME)


def reply_excerpt_around(reply: str, span: str) -> str:
    """Up to EXCERPT_MAX chars of ``reply`` centred on ``span`` (head if absent)."""
    reply = reply or ""
    if len(reply) <= EXCERPT_MAX:
        return reply
    pos = reply.find(span) if span else -1
    if pos < 0:
        return _bound(reply, EXCERPT_MAX)
    start = max(0, pos + len(span) // 2 - EXCERPT_MAX // 2)
    return reply[start:start + EXCERPT_MAX]


def record_finding(checker: str, *, test_probe: bool = False,
                   probe_texts: Iterable[Any] = (), **fields: Any) -> bool:
    """Append one finding line. Never raises. True if written."""
    try:
        try:
            from config.app_config import TURN_TELEMETRY_CFG  # lazy import: live-config
            enabled = bool((TURN_TELEMETRY_CFG or {}).get("checker_findings_enabled", True))
        except ImportError:
            enabled = True
        if not enabled or test_probe:
            return False
        if any(_TEST_SPAN_RE.search(str(t or "")) for t in probe_texts):
            return False
        row: dict[str, Any] = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "checker": str(checker)[:40],
        }
        if os.getenv("DAEMON_TEST_MODE"):
            row["test_env"] = True
        limits = {"claim_span": SPAN_MAX, "evidence_excerpt": EXCERPT_MAX,
                  "reply_excerpt": EXCERPT_MAX, "reason": SHORT_MAX}
        for key, value in fields.items():
            if isinstance(value, (bool, int, float)) or value is None:
                row[key] = value
            elif isinstance(value, (list, tuple)):
                row[key] = [_bound(v, SHORT_MAX) for v in list(value)[:10]]
            else:
                row[key] = _bound(value, limits.get(key, SHORT_MAX))
        path = findings_path()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True
    except Exception as exc:  # degrades: this finding is lost; the turn is unaffected
        logger.debug(f"[CheckerFindings] write skipped: {exc}")
        return False
