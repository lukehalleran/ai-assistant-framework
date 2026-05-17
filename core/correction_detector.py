"""
# core/correction_detector.py

Module Contract
- Purpose: Detect when a user corrects or confirms previously stored facts,
  including entity-level corrections (e.g., "Flapjack did not die").
  Feeds correction/confirmation events to the TruthScorer so that fact
  truth_scores evolve based on real evidence rather than access counts.
  Entity corrections trigger resolution annotations on crisis-era summaries.
- Inputs:
  - user_message: str (the latest user utterance)
  - recent_facts: list[dict] (facts from user_profile or ChromaDB)
- Outputs:
  - list[CorrectionEvent] — each event identifies a fact and whether it
    was corrected or confirmed, with a confidence score.
  - list[EntityCorrectionEvent] — entity-level corrections (alive/survived/etc.)
- Key behaviors:
  - Pattern-based detection (no LLM call — fast, deterministic)
  - Correction patterns: "actually it's...", "no, I meant...", "I moved to..."
  - Confirmation patterns: "yeah I still...", "still working at..."
  - Entity correction patterns: "X did not die", "X is still alive", "X survived"
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


class EntityCorrectionEvent(BaseModel):
    """An event indicating an entity-level correction (e.g., 'Flapjack is alive').

    Unlike CorrectionEvent, this does not reference a specific stored fact —
    it detects when the user corrects an assumption about a named entity
    (pet, person, etc.) so the system can annotate crisis-era summaries
    with resolution metadata.
    """

    entity_name: str = Field(..., description="Name of the entity being corrected")
    correction_type: str = Field(..., description="Type: alive, survived, not_dead")
    correction_text: str = Field(..., description="Raw user text (truncated)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")


class AttributionEvent(BaseModel):
    """An event indicating the user is attributing previously shared content.

    Example: "It's by The Narcissist Cookbook" → attributes the most recent
    unattributed shared content (lyrics, poem, etc.) to this artist.
    """

    attribution_type: str = Field(..., description="'artist', 'author', 'title', or 'source'")
    value: str = Field(..., description="The attributed name/title")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    raw_text: str = Field("", description="Truncated user message")


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

# ------------------------------------------------------------------
# Entity correction patterns (for non-user entities: pets, people, etc.)
# ------------------------------------------------------------------

# Each tuple: (compiled regex, confidence, correction_type)
# Named group (?P<entity>...) captures the entity name.
# re.I makes [A-Z] match lowercase too — intentional so "flapjack did not die" works.
_ENTITY_CORRECTION_PATTERNS = [
    # "Flapjack did not die" / "Flapjack didn't die"
    (re.compile(r"\b(?P<entity>[A-Z]\w+)\s+did\s*n[o']t\s+die\b", re.I), 0.90, "not_dead"),
    # "Flapjack is not dead / still alive / still here / alive"
    (re.compile(r"\b(?P<entity>[A-Z]\w+)\s+is\s+(?:not\s+dead|still\s+alive|still\s+here|alive)\b", re.I), 0.90, "alive"),
    # "Flapjack survived / made it / is fine / is okay / pulled through"
    (re.compile(r"\b(?P<entity>[A-Z]\w+)\s+(?:survived|made\s+it|is\s+fine|is\s+okay|is\s+ok|pulled\s+through)\b", re.I), 0.85, "survived"),
    # "my cat Flapjack is alive" / "my dog Rex survived"
    (re.compile(r"\bmy\s+\w+\s+(?P<entity>[A-Z]\w+)\s+(?:is\s+(?:not\s+dead|still\s+alive|alive|fine|okay|ok)|survived|made\s+it|pulled\s+through)\b", re.I), 0.90, "alive"),
    # "Flapjack is still with us / still around / still kicking"
    (re.compile(r"\b(?P<entity>[A-Z]\w+)\s+is\s+still\s+(?:with\s+us|around|kicking)\b", re.I), 0.85, "alive"),
    # "no, Flapjack didn't die" / "no Flapjack is alive"
    (re.compile(r"\bno[,.]?\s*(?P<entity>[A-Z]\w+)\s+(?:did\s*n[o']t\s+die|is\s+(?:alive|fine|okay|ok|not\s+dead))\b", re.I), 0.90, "not_dead"),
]

# Words that should never be captured as entity names
_ENTITY_STOPWORDS = frozenset({
    "i", "it", "he", "she", "they", "we", "the", "that", "this",
    "who", "what", "my", "your", "our", "but", "and", "not", "no",
})

# ------------------------------------------------------------------
# Attribution patterns (for shared content provenance)
# ------------------------------------------------------------------

# Each tuple: (compiled regex, confidence, attribution_type)
_ATTRIBUTION_PATTERNS = [
    # "it's by X" / "that's by X" / "that is by X"
    (re.compile(r"\b(?:it'?s|that'?s|that\s+(?:was|is))\s+by\s+(.{3,60}?)(?:\.|,|;|\n|$)", re.I), 0.90, "artist"),
    # "the song/poem is called X" / "it's called X"
    (re.compile(r"\b(?:it'?s|that'?s|the\s+(?:song|poem|piece|track))\s+(?:is\s+)?called\s+['\"]?(.{3,60}?)['\"]?(?:\.|,|;|\n|$)", re.I), 0.85, "title"),
    # "that was X's song/poem/piece"
    (re.compile(r"\bthat\s+(?:was|is)\s+(.{3,40}?)['']s\s+(?:song|poem|piece|work|track)", re.I), 0.85, "artist"),
    # "by an artist who performs as X" / "by an artist called X"
    (re.compile(r"\bartist\s+(?:who\s+)?(?:performs|goes)\s+(?:as|by)\s+(.{3,60}?)(?:\.|,|;|\n|$)", re.I), 0.90, "artist"),
    # "the artist is X" / "the poet is X"
    (re.compile(r"\bthe\s+(?:artist|poet|author|band|singer)\s+is\s+(.{3,60}?)(?:\.|,|;|\n|$)", re.I), 0.85, "artist"),
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

    def detect_entity_corrections(
        self, user_message: str
    ) -> List["EntityCorrectionEvent"]:
        """Detect entity-level corrections (e.g., 'Flapjack did not die').

        Unlike detect_corrections(), this does NOT require a fact list —
        it's pure pattern matching on the user's message text.

        Args:
            user_message: The latest user utterance.

        Returns:
            List of EntityCorrectionEvent for detected entity corrections.
        """
        if not user_message:
            return []

        events: List[EntityCorrectionEvent] = []
        seen_entities: set = set()

        for pattern, confidence, correction_type in _ENTITY_CORRECTION_PATTERNS:
            for match in pattern.finditer(user_message):
                entity = match.group("entity").strip()
                entity_lower = entity.lower()

                if entity_lower in _ENTITY_STOPWORDS:
                    continue

                if entity_lower not in seen_entities:
                    seen_entities.add(entity_lower)
                    events.append(EntityCorrectionEvent(
                        entity_name=entity,
                        correction_type=correction_type,
                        correction_text=user_message[:200],
                        confidence=confidence,
                    ))

        if events:
            logger.info(
                "[CorrectionDetector] Detected %d entity correction(s): %s",
                len(events), ", ".join(e.entity_name for e in events),
            )

        return events

    def detect_attributions(
        self, user_message: str
    ) -> List["AttributionEvent"]:
        """Detect when user attributes previously shared content to an artist/author.

        Examples:
            "It's by The Narcissist Cookbook" → artist attribution
            "the song is called Ananke" → title attribution
            "by an artist who performs as the nassaricts cookbook" → artist attribution

        Returns:
            List of AttributionEvent for detected attributions.
        """
        if not user_message:
            return []

        events: List[AttributionEvent] = []

        for pattern, confidence, attr_type in _ATTRIBUTION_PATTERNS:
            m = pattern.search(user_message)
            if m and confidence >= _MIN_CONFIDENCE:
                value = m.group(1).strip().rstrip(".,;")
                if len(value) > 2:
                    events.append(AttributionEvent(
                        attribution_type=attr_type,
                        value=value,
                        confidence=confidence,
                        raw_text=user_message[:200],
                    ))
                    break  # Take the first (highest priority) match

        if events:
            logger.info(
                "[CorrectionDetector] Detected attribution: %s=%s",
                events[0].attribution_type, events[0].value,
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
