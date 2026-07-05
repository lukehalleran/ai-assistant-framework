"""
Per-turn telemetry — one JSONL line per completed chat turn.

Module Contract:
  Purpose: Record the routing story of each turn (intent classification →
           agentic gate → response path → post-answer checks) so
           classification/routing accuracy can be measured offline instead
           of guessed. This is the data source for building an intent
           confusion matrix and for mining uncertainty/review events as
           weak labels for misrouted turns.
  Inputs:  record_turn(record) — flat dict of JSON-serializable fields.
           The caller (gui/handlers.py) assembles the record from
           SubmitContext.telemetry + orchestrator._last_turn_signals.
  Outputs: Appends one JSON object per line to TURN_TELEMETRY_PATH
           (default logs/turn_records.jsonl). Adds a "ts" ISO timestamp.
  Side effects: File append (creates parent dir on first write).
           NEVER raises — telemetry must not break a turn. Returns bool.
  Config:  TURN_TELEMETRY_ENABLED, TURN_TELEMETRY_PATH (YAML section
           `turn_telemetry`; env override TURN_TELEMETRY_ENABLED).

Typical record fields (all optional — record what the turn produced):
  query, intent, intent_confidence, intent_source, tone_level,
  is_small_talk, plan_points, plan_tone, gate_triggered, gate_modes,
  gate_reason, mode (enhanced|agentic-search|best-of-duel|...),
  uncertainty_fired, uncertainty_accepted, review_fired, review_passed,
  review_retry_accepted, response_len, model, session_id,
  prepare_elapsed_s.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict

from utils.logging_utils import get_logger

logger = get_logger("turn_telemetry")

# Cap free-text fields so the JSONL stays greppable and small.
_MAX_STR_LEN = 300


def _sanitize_value(value: Any, _depth: int = 0) -> Any:
    """Coerce a field to something json.dumps can handle, truncating strings.

    Bounded: recursion happens only through containers with a depth cap, and
    the enum fallback never recurses — following `.value` chains blindly
    walks forever on mock-like objects whose .value is another object
    (found the hard way: MagicMocks in tests allocated gigabytes here).
    """
    if _depth > 6:
        return str(value)[:_MAX_STR_LEN]
    if isinstance(value, str):
        return value[:_MAX_STR_LEN]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitize_value(v, _depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k)[:_MAX_STR_LEN]: _sanitize_value(v, _depth + 1)
                for k, v in value.items()}
    # Enum-like (IntentType, ToneLevel): use .value only if it's a scalar;
    # anything else stringifies — never recurse on arbitrary attributes.
    enum_val = getattr(value, "value", None)
    if isinstance(enum_val, (str, int, float, bool)):
        return enum_val if not isinstance(enum_val, str) else enum_val[:_MAX_STR_LEN]
    return str(value)[:_MAX_STR_LEN]


def record_turn(record: Dict[str, Any]) -> bool:
    """Append one turn record as a JSON line. Never raises.

    Returns True if the line was written, False if telemetry is disabled
    or the write failed (failure is logged at DEBUG — non-fatal by design).
    """
    try:
        try:
            from config.app_config import (
                TURN_TELEMETRY_ENABLED,
                TURN_TELEMETRY_PATH,
            )
        except ImportError:
            TURN_TELEMETRY_ENABLED = True
            TURN_TELEMETRY_PATH = "logs/turn_records.jsonl"

        if not TURN_TELEMETRY_ENABLED:
            return False

        payload = {"ts": datetime.now().astimezone().isoformat(timespec="seconds")}
        payload.update({str(k): _sanitize_value(v) for k, v in (record or {}).items()})

        path = TURN_TELEMETRY_PATH
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True
    except Exception as e:  # noqa: BLE001 — telemetry must never break a turn
        logger.debug(f"[TurnTelemetry] write skipped: {e}")
        return False
