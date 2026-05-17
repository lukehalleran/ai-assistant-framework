# core/ambiguity_detector.py
"""
Cross-session ambiguity detection for conversation context.

Detects when a short user message references a phrase that appears in
multiple conversation entries from different sessions. Generates a
disambiguation note for prompt injection so the LLM doesn't conflate
content from separate sessions.

No LLM calls. Regex + substring matching only. Target: <10ms.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from utils.logging_utils import get_logger

logger = get_logger("ambiguity_detector")

# Maximum word count for a user message to trigger ambiguity detection.
# Long messages provide enough context that the LLM can usually disambiguate.
_MAX_WORDS = 50


@dataclass
class AmbiguityMatch:
    """A conversation entry that matches the ambiguous phrase."""
    entry_index: int
    session_label: str
    content_preview: str
    content_type: str
    timestamp: Optional[datetime] = None


@dataclass
class AmbiguityResult:
    """Result of ambiguity detection."""
    is_ambiguous: bool
    confidence: float
    ambiguous_phrase: str
    matching_entries: List[AmbiguityMatch] = field(default_factory=list)
    disambiguation_note: str = ""


class AmbiguityDetector:
    """Detects cross-session ambiguity in user references to context."""

    @staticmethod
    def detect(
        user_message: str,
        recent_conversations: List[Dict],
        gap_hours: float = 2.0,
    ) -> AmbiguityResult:
        """
        Check if user message references a phrase that appears in multiple
        sessions within the conversation context.

        Args:
            user_message: Current user input.
            recent_conversations: List of conversation dicts from context.
            gap_hours: Session boundary threshold in hours.

        Returns:
            AmbiguityResult with disambiguation note if ambiguity detected.
        """
        empty = AmbiguityResult(False, 0.0, "")

        if not user_message or not recent_conversations:
            return empty

        # Only trigger on short messages (long messages self-disambiguate)
        if len(user_message.split()) > _MAX_WORDS:
            return empty

        # Extract referential phrases from user message
        phrases = _extract_referential_phrases(user_message)
        if not phrases:
            return empty

        # Parse timestamps and assign sessions
        entries_with_sessions = _assign_sessions(recent_conversations, gap_hours)

        # Check each phrase for cross-session matches
        for phrase in phrases:
            phrase_lower = phrase.lower()
            matches = []

            for entry in entries_with_sessions:
                # Search in query and response text
                text = _get_entry_text(entry["mem"]).lower()
                if phrase_lower in text:
                    md = entry["mem"].get("metadata", {}) or {}
                    matches.append(AmbiguityMatch(
                        entry_index=entry["index"],
                        session_label=entry["session_label"],
                        content_preview=text[:80],
                        content_type=md.get("content_type", ""),
                        timestamp=entry["timestamp"],
                    ))

            # Check if matches span multiple sessions
            if len(matches) >= 2:
                sessions = set(m.session_label for m in matches)
                if len(sessions) >= 2:
                    note = _build_disambiguation_note(phrase, matches)
                    return AmbiguityResult(
                        is_ambiguous=True,
                        confidence=0.80,
                        ambiguous_phrase=phrase,
                        matching_entries=matches,
                        disambiguation_note=note,
                    )

        return empty


def _extract_referential_phrases(message: str) -> List[str]:
    """
    Extract phrases from user message that might reference context content.

    Looks for:
    - Quoted phrases: "Not entirely alone"
    - Title-case sequences: Not Entirely Alone
    - "X is my favorite" / "I love X" patterns
    - Short distinctive phrases (3+ non-stopword consecutive words)
    """
    phrases = []

    # Quoted phrases (highest confidence)
    for m in re.finditer(r'"([^"]{3,60})"', message):
        phrases.append(m.group(1))
    for m in re.finditer(r"'([^']{3,60})'", message):
        phrases.append(m.group(1))

    # "X is my favorite" / "my favorite is X"
    m = re.search(r'(.{3,40}?)\s+is\s+my\s+(?:favorite|favourite)', message, re.I)
    if m:
        phrases.append(m.group(1).strip())
    m = re.search(r'my\s+(?:favorite|favourite)\s+is\s+(.{3,40})', message, re.I)
    if m:
        phrases.append(m.group(1).strip().rstrip(".,;"))

    # Title-case sequences (2+ capitalized words)
    for m in re.finditer(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', message):
        term = m.group(1)
        if len(term) > 5:
            phrases.append(term)

    # Deduplicate
    seen = set()
    unique = []
    for p in phrases:
        p_lower = p.lower()
        if p_lower not in seen:
            seen.add(p_lower)
            unique.append(p)

    return unique


def _assign_sessions(
    conversations: List[Dict], gap_hours: float
) -> List[Dict]:
    """Assign session labels to conversation entries based on timestamps."""
    from core.prompt.formatter import _detect_session_boundary

    result = []
    session_num = 0
    prev_ts = None

    for i, mem in enumerate(conversations):
        ts = _parse_ts(mem)
        if _detect_session_boundary(prev_ts, ts, gap_hours):
            session_num += 1
        if ts:
            prev_ts = ts

        if ts:
            label = f"Session {session_num} ({ts.strftime('%b %-d')})"
        else:
            label = f"Session {session_num}"

        result.append({
            "mem": mem,
            "index": i,
            "timestamp": ts,
            "session_num": session_num,
            "session_label": label,
        })

    return result


def _parse_ts(mem: dict) -> Optional[datetime]:
    """Extract datetime from a conversation entry."""
    ts = mem.get("timestamp", "")
    if not ts:
        ts = (mem.get("metadata") or {}).get("timestamp", "")
    if isinstance(ts, datetime):
        return ts
    if ts:
        try:
            return datetime.fromisoformat(str(ts))
        except (ValueError, TypeError):
            pass
    return None


def _get_entry_text(mem: dict) -> str:
    """Get searchable text from a conversation entry."""
    parts = []
    for key in ("query", "response", "content"):
        val = mem.get(key, "")
        if val:
            parts.append(str(val))
    return " ".join(parts)


def _build_disambiguation_note(phrase: str, matches: List[AmbiguityMatch]) -> str:
    """Build a disambiguation note for prompt injection."""
    session_labels = sorted(set(m.session_label for m in matches))
    content_types = [m.content_type for m in matches if m.content_type]

    note = f'[DISAMBIGUATION NOTE: The user mentioned "{phrase}"'

    if content_types:
        note += f' — this phrase appears in shared {content_types[0]} content'
        if len(session_labels) == 2:
            note += f' from {session_labels[0]} and also in {session_labels[1]}'
        else:
            note += f' across {len(session_labels)} different sessions'
    else:
        if len(session_labels) == 2:
            note += f' — this appears in conversations from both {session_labels[0]} and {session_labels[1]}'
        else:
            note += f' — this appears across {len(session_labels)} different sessions'

    note += '. Clarify which the user means before making attribution or identity claims.]'
    return note
