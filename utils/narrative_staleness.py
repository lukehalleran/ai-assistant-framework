"""
# utils/narrative_staleness.py

Module Contract
- Purpose: Derived-state flag marking the cached narrative context
  (data/narrative_context.txt) as potentially stale after a user correction.
  2026-08-23: "6 weeks off vryalr not 1." corrected a duration, but the
  narrative regenerates only at shutdown — the wrong "day 8" framing kept
  re-entering every prompt's [TEMPORAL GROUNDING] section for the rest of
  the session. The flag doesn't rewrite the narrative (an LLM job); it makes
  the prompt HONEST about it until the next regeneration.
- API:
  - mark_stale(reason: str) -> bool — record a correction event (atomic
    write; keeps the EARLIEST mark so a later correction can't hide an
    earlier one from an intervening regeneration check).
  - clear() -> None — remove the flag (called after a successful narrative
    save; best-effort).
  - is_stale(narrative_mtime: float) -> bool — True iff a flag exists AND
    it was marked AFTER the narrative file was generated.
- Flag path: NARRATIVE_STALE_FLAG_PATH env (default data/narrative_stale.json),
  resolved at CALL time so tests can sandbox it.
- Side effects: writes/removes the flag file only.
- Failure policy: lenient derived state — corrupt/missing flag = not stale;
  no function here ever raises (same doctrine as tone_state.json).
"""

import json
import os
import time

from utils.logging_utils import get_logger

logger = get_logger("narrative_staleness")

_DEFAULT_FLAG_PATH = os.path.join("data", "narrative_stale.json")


def _flag_path() -> str:
    return os.getenv("NARRATIVE_STALE_FLAG_PATH", _DEFAULT_FLAG_PATH)


def _load() -> dict:
    """Read the flag file; corrupt or missing → {} (never raises)."""
    try:
        path = _flag_path()
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def mark_stale(reason: str) -> bool:
    """Record that a user correction may have invalidated the narrative.

    Keeps the EARLIEST marked_at across repeated corrections — staleness is
    measured against the narrative generation time, and the first correction
    after generation is the one that makes it stale.
    """
    try:
        existing = _load()
        marked_at = existing.get("marked_at")
        if not isinstance(marked_at, (int, float)):
            marked_at = time.time()
        payload = {
            "marked_at": marked_at,
            "last_marked_at": time.time(),
            "reason": (reason or "")[:200],
        }
        path = _flag_path()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
        logger.info(f"[NarrativeStaleness] Marked stale: {payload['reason'][:80]!r}")
        return True
    except Exception as e:
        logger.debug(f"[NarrativeStaleness] mark_stale failed (non-fatal): {e}")
        return False


def clear() -> None:
    """Remove the flag (after a fresh narrative save). Best-effort."""
    try:
        path = _flag_path()
        if os.path.exists(path):
            os.remove(path)
            logger.debug("[NarrativeStaleness] Flag cleared")
    except Exception as e:
        logger.debug(f"[NarrativeStaleness] clear failed (non-fatal): {e}")


def is_stale(narrative_mtime: float) -> bool:
    """True iff a correction was marked AFTER the narrative was generated."""
    try:
        marked_at = _load().get("marked_at")
        if not isinstance(marked_at, (int, float)):
            return False
        return float(marked_at) > float(narrative_mtime)
    except Exception:
        return False
