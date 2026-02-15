"""
# core/correction_detector.py

Module Contract
- Purpose: Detect when a user corrects or confirms previously stored facts.
  Feeds correction/confirmation events to the TruthScorer so that fact
  truth_scores evolve based on real evidence rather than access counts.
- Inputs:
  - user_message: str (the latest user utterance)
  - recent_facts: list[dict] (facts from user_profile or ChromaDB)
- Outputs:
  - list[CorrectionEvent] — each event identifies a fact and whether it
    was corrected or confirmed, with a confidence score.
- Key behaviors:
  - Pattern-based detection (no LLM call — fast, deterministic)
  - Correction patterns: "actually it's...", "no, I meant...", "I moved to..."
  - Confirmation patterns: "yeah I still...", "still working at..."
  - Minimum confidence threshold of 0.6 to reduce false positives
- Side effects:
  - None (pure detection; callers decide what to do with events)
"""

import re
from typing import List, Optional

from pydantic import BaseModel, Field

from utils.logging_utils import get_logger

logger = get_logger("correction_detector")


class CorrectionEvent(BaseModel):
    """An event indicating a user correction or confirmation of a stored fact."""

    fact_id: str = Field(..., description="ID of the affected fact")
    relation: str = Field(..., description="Relation/predicate of the fact")
    old_value: str = Field(..., description="Previous value of the fact")
    new_value: str = Field("", description="New value (empty for confirmations)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    event_type: str = Field(..., description="'correction' or 'confirmation'")


# ------------------------------------------------------------------
# Correction patterns
# ------------------------------------------------------------------

# Each tuple: (compiled regex, confidence modifier)
# The regex should match common user correction phrases.
_CORRECTION_PATTERNS = [
    (re.compile(r"\bactually\b.*\b(it'?s|i'?m|my|i)\b", re.I), 0.80),
    (re.compile(r"\bno[,.]?\s*(i\s+meant|that'?s\s+wrong|it'?s\s+not)\b", re.I), 0.85),
    (re.compile(r"\bcorrection\s*:", re.I), 0.90),
    (re.compile(r"\bthat'?s?\s+(wrong|incorrect|not\s+right|not\s+true)\b", re.I), 0.85),
    (re.compile(r"\bi\s+(moved|changed|switched|transferred|quit|left|started)\s+(to|from|at)\b", re.I), 0.70),
    (re.compile(r"\bi\s+no\s+longer\b", re.I), 0.75),
    (re.compile(r"\bnot\s+anymore\b", re.I), 0.70),
    (re.compile(r"\bi\s+don'?t\s+(live|work|go|do|have)\b", re.I), 0.65),
]

# ------------------------------------------------------------------
# Confirmation patterns
# ------------------------------------------------------------------

_CONFIRMATION_PATTERNS = [
    (re.compile(r"\b(yeah|yes|yep)\b.*\bstill\b", re.I), 0.75),
    (re.compile(r"\bstill\s+(live|living|work|working|at|in|do|doing)\b", re.I), 0.80),
    (re.compile(r"\bthat'?s?\s+(right|correct|true|accurate)\b", re.I), 0.80),
    (re.compile(r"\byou('?re|\s+are)\s+(right|correct)\b", re.I), 0.75),
]

# Minimum confidence to emit an event
_MIN_CONFIDENCE = 0.6


class CorrectionDetector:
    """Detects user corrections and confirmations of stored facts."""

    def detect_corrections(
        self, user_message: str, recent_facts: List[dict]
    ) -> List[CorrectionEvent]:
        """Detect if the user is correcting any recently stored facts.

        Args:
            user_message: The latest user utterance.
            recent_facts: List of fact dicts, each with at least
                ``fact_id``, ``relation``, ``value``.

        Returns:
            List of CorrectionEvent for facts that appear corrected.
        """
        if not user_message or not recent_facts:
            return []

        events: List[CorrectionEvent] = []
        msg_lower = user_message.lower()

        # First check if the message matches any correction pattern at all
        best_pattern_conf = 0.0
        for pattern, conf in _CORRECTION_PATTERNS:
            if pattern.search(user_message):
                best_pattern_conf = max(best_pattern_conf, conf)

        if best_pattern_conf < _MIN_CONFIDENCE:
            return []

        # Now check which facts are referenced
        for fact in recent_facts:
            relation = fact.get("relation", "")
            value = fact.get("value", "")
            fact_id = fact.get("fact_id", "")

            if not relation or not value or not fact_id:
                continue

            # Check if the fact's relation or value is mentioned in the message
            relation_words = set(relation.lower().replace("_", " ").split())
            value_lower = value.lower()

            # Relation word overlap with message
            msg_words = set(msg_lower.split())
            overlap = relation_words & msg_words

            # Also check if the old value appears in the message context
            value_mentioned = value_lower in msg_lower

            if overlap or value_mentioned:
                conf = best_pattern_conf
                # Boost confidence if the old value is explicitly mentioned
                if value_mentioned:
                    conf = min(1.0, conf + 0.05)

                if conf >= _MIN_CONFIDENCE:
                    events.append(CorrectionEvent(
                        fact_id=fact_id,
                        relation=relation,
                        old_value=value,
                        new_value="",  # Caller extracts the new value from profile update
                        confidence=conf,
                        event_type="correction",
                    ))

        if events:
            logger.info(
                "[CorrectionDetector] Detected %d correction(s) in: %s",
                len(events), user_message[:80],
            )

        return events

    def detect_confirmations(
        self, user_message: str, recent_facts: List[dict]
    ) -> List[CorrectionEvent]:
        """Detect if the user is confirming/reaffirming stored facts.

        Args:
            user_message: The latest user utterance.
            recent_facts: List of fact dicts.

        Returns:
            List of CorrectionEvent with event_type="confirmation".
        """
        if not user_message or not recent_facts:
            return []

        events: List[CorrectionEvent] = []
        msg_lower = user_message.lower()

        # Check if the message matches any confirmation pattern
        best_pattern_conf = 0.0
        for pattern, conf in _CONFIRMATION_PATTERNS:
            if pattern.search(user_message):
                best_pattern_conf = max(best_pattern_conf, conf)

        if best_pattern_conf < _MIN_CONFIDENCE:
            return []

        # Check which facts are being confirmed
        for fact in recent_facts:
            relation = fact.get("relation", "")
            value = fact.get("value", "")
            fact_id = fact.get("fact_id", "")

            if not relation or not value or not fact_id:
                continue

            value_lower = value.lower()
            relation_words = set(relation.lower().replace("_", " ").split())
            msg_words = set(msg_lower.split())

            # Check for value or relation reference in message
            value_mentioned = value_lower in msg_lower
            relation_overlap = bool(relation_words & msg_words)

            if value_mentioned or relation_overlap:
                conf = best_pattern_conf
                if value_mentioned:
                    conf = min(1.0, conf + 0.05)

                if conf >= _MIN_CONFIDENCE:
                    events.append(CorrectionEvent(
                        fact_id=fact_id,
                        relation=relation,
                        old_value=value,
                        new_value=value,  # Same value = confirmation
                        confidence=conf,
                        event_type="confirmation",
                    ))

        if events:
            logger.info(
                "[CorrectionDetector] Detected %d confirmation(s) in: %s",
                len(events), user_message[:80],
            )

        return events
