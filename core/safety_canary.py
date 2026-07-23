"""
core/safety_canary.py

Module Contract
- Purpose: a lightweight, LOG-ONLY session monitor that defends against the NEXT
  unknown miswire in the tone/safety path. If N consecutive user messages score
  negative on the valence lexicon while the classified tone stays CONVERSATIONAL,
  it emits a WARNING `SAFETY_CANARY_TONE_FLATLINE` with the session id and the
  turn indices of the streak. It NEVER changes behavior — it only tells us the
  invariant "sustained negative-affect input should not read as conversational"
  has been violated, so a future flatline surfaces in seconds, not weeks.
- Reuses memory/valence.negative_affect_score (no new scorer).
- Key class: SafetyCanary(threshold, negative_threshold, enabled, session_id)
  - observe(user_message, tone) -> Optional[dict]  (returns the fired event, or None)
  - reset()
- Side effects: WARNING log only.
"""

from __future__ import annotations

from typing import List, Optional

from utils.logging_utils import get_logger

logger = get_logger("safety_canary")


class SafetyCanary:
    """Detects a sustained negative-affect-but-conversational streak (log only)."""

    def __init__(
        self,
        threshold: int = 4,
        negative_threshold: float = 0.30,
        enabled: bool = True,
        session_id: str = "session",
    ):
        self.threshold = max(1, int(threshold))
        self.negative_threshold = float(negative_threshold)
        self.enabled = bool(enabled)
        self.session_id = session_id or "session"
        self._turn = 0
        self._streak: List[int] = []  # turn indices in the current streak

    @staticmethod
    def _is_conversational(tone) -> bool:
        # Accept any tone encoding: CrisisLevel ("conversational"),
        # ToneLevel ("CONVERSATIONAL"), or a plain string.
        return "conversational" in str(tone).lower()

    def observe(self, user_message: str, tone) -> Optional[dict]:
        """
        Record one turn. Returns the fired event dict when the streak is at/over
        threshold, else None. Never raises — a canary must not break a turn.
        """
        self._turn += 1
        if not self.enabled:
            return None
        try:
            from memory.valence import negative_affect_score
            is_negative = negative_affect_score(user_message or "") >= self.negative_threshold
            is_conversational = self._is_conversational(tone)

            if is_negative and is_conversational:
                self._streak.append(self._turn)
            else:
                self._streak = []
                return None

            if len(self._streak) >= self.threshold:
                event = {
                    "event": "SAFETY_CANARY_TONE_FLATLINE",
                    "session": self.session_id,
                    "turns": list(self._streak),
                    "consecutive": len(self._streak),
                }
                logger.warning(
                    "SAFETY_CANARY_TONE_FLATLINE session=%s turns=%s — %d consecutive "
                    "negative-affect user messages classified CONVERSATIONAL "
                    "(tone/safety path may be miswired; log-only, no behavior change)",
                    self.session_id, self._streak, len(self._streak),
                )
                return event
            return None
        except Exception as e:  # pragma: no cover - a canary must never break a turn
            logger.debug(f"[SafetyCanary] observe failed (ignored): {e}")
            return None

    def reset(self) -> None:
        self._turn = 0
        self._streak = []
