# core/content_type_detector.py
"""
Lightweight regex-based content type detection for user messages.

Detects when users share discrete content objects (lyrics, poems, code,
quotes, messages, dreams) rather than normal conversation text. Used to
tag stored interactions with content_type metadata for provenance tracking.

No LLM calls. All patterns are pre-compiled at module level.
Target: <5ms per detection.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ContentTypeResult:
    """Result of content type detection."""
    content_type: str       # "lyrics", "poem", "code", "quote", "message", "dream", ""
    confidence: float       # 0.0-1.0
    title_hint: str         # extracted title if detectable, else ""
    attribution_hint: str   # extracted artist/author if detectable, else ""


# ---------------------------------------------------------------------------
# Detection patterns (pre-compiled, checked in priority order)
# ---------------------------------------------------------------------------

# [Spoken] prefix = spoken word / poem
_SPOKEN_PREFIX = re.compile(r'^\s*\[Spoken\]', re.IGNORECASE)

# Code fences
_CODE_FENCE = re.compile(r'```[\w]*\n', re.MULTILINE)

# File extension references suggesting code sharing
_CODE_FILES = re.compile(r'\b\w+\.(py|js|ts|jsx|tsx|go|rs|java|c|cpp|h|rb|sh|yaml|yml|json|toml|sql)\b')

# Dream narratives
_DREAM_PATTERNS = [
    re.compile(r'\bI\s+(?:had\s+a\s+)?dream(?:t|ed)?\b', re.IGNORECASE),
    re.compile(r'\bin\s+my\s+dream\b', re.IGNORECASE),
    re.compile(r'\bI\s+was\s+dreaming\b', re.IGNORECASE),
]

# Shared message from another person
_MESSAGE_PATTERNS = [
    re.compile(r'\b(?:my\s+(?:mom|dad|brother|sister|friend|partner|boss|coworker))\s+(?:said|texted|wrote|sent|told\s+me)\b', re.IGNORECASE),
    re.compile(r'\b(?:he|she|they)\s+(?:said|texted|wrote|sent)\s*[:\-]', re.IGNORECASE),
]

# Content-sharing preambles
_SHARE_PREAMBLES = [
    re.compile(r'^(?:check\s+(?:this|it)\s+out|look\s+at\s+this|listen\s+to\s+this)', re.IGNORECASE),
    re.compile(r'^(?:I\s+(?:heard|found|read|saw)\s+this)', re.IGNORECASE),
    re.compile(r'^(?:this\s+(?:song|poem|article|piece))', re.IGNORECASE),
]

# Title extraction from preambles: "this song is called X" / "it's called X"
_TITLE_PATTERNS = [
    re.compile(r'(?:called|titled|named)\s+["\']?(.{3,60}?)["\']?\s*(?:\.|,|;|$)', re.IGNORECASE),
    re.compile(r'(?:the\s+(?:song|poem|piece|track))\s+["\'](.{3,60}?)["\']', re.IGNORECASE),
]

# Attribution extraction: "by X" / "from X"
_ATTRIBUTION_PATTERNS = [
    re.compile(r'\bby\s+([A-Z][\w\s]{2,40}?)(?:\.|,|;|\n|\s*$)', re.IGNORECASE),
    re.compile(r'\bfrom\s+(?:the\s+)?(?:artist|band|author|poet)\s+(.{3,40})', re.IGNORECASE),
]


def detect_content_type(text: str) -> ContentTypeResult:
    """
    Detect whether user input contains a shared content object.

    Returns ContentTypeResult with content_type="" for normal conversation.
    Detection is intentionally conservative — false negatives are fine
    (treated as regular conversation), false positives are worse.
    """
    if not text:
        return ContentTypeResult("", 0.0, "", "")

    # 1. [Spoken] prefix → poem/spoken word (highest confidence)
    if _SPOKEN_PREFIX.search(text):
        title, attr = _extract_title_attribution(text)
        return ContentTypeResult("poem", 0.95, title, attr)

    # 2. Code fences → code
    if _CODE_FENCE.search(text):
        return ContentTypeResult("code", 0.90, "", "")

    # 3. Dream narratives
    for pattern in _DREAM_PATTERNS:
        if pattern.search(text):
            return ContentTypeResult("dream", 0.80, "", "")

    # 4. Shared message from another person
    for pattern in _MESSAGE_PATTERNS:
        if pattern.search(text):
            return ContentTypeResult("message", 0.70, "", "")

    # 5. Content-sharing preamble + substantial text
    for pattern in _SHARE_PREAMBLES:
        if pattern.search(text):
            title, attr = _extract_title_attribution(text)
            # Determine if it's lyrics/poem vs general content
            if _looks_like_lyrics(text):
                return ContentTypeResult("lyrics", 0.75, title, attr)
            return ContentTypeResult("quote", 0.70, title, attr)

    # 6. Multi-line poetic/lyrical structure (no explicit marker)
    if _looks_like_lyrics(text):
        title, attr = _extract_title_attribution(text)
        return ContentTypeResult("lyrics", 0.65, title, attr)

    # 7. No content type detected
    return ContentTypeResult("", 0.0, "", "")


def _looks_like_lyrics(text: str) -> bool:
    """
    Heuristic: text looks like lyrics or poetry if it has many short lines,
    no question marks, and consistent line structure.
    """
    lines = text.strip().split("\n")
    if len(lines) < 4:
        return False

    # Count non-empty lines
    non_empty = [l.strip() for l in lines if l.strip()]
    if len(non_empty) < 4:
        return False

    # No questions = not conversation
    if "?" in text:
        # Allow one question mark in poetry (rhetorical)
        if text.count("?") > 1:
            return False

    # Average line length should be short (lyrics are typically < 60 chars/line)
    avg_len = sum(len(l) for l in non_empty) / len(non_empty)
    if avg_len > 80:
        return False

    # Most lines should be short-to-medium
    short_lines = sum(1 for l in non_empty if len(l) < 60)
    if short_lines / len(non_empty) < 0.6:
        return False

    return True


def _extract_title_attribution(text: str) -> tuple:
    """Extract title and attribution hints from text."""
    title = ""
    attribution = ""

    for pattern in _TITLE_PATTERNS:
        m = pattern.search(text[:200])  # Only check beginning
        if m:
            title = m.group(1).strip()
            break

    for pattern in _ATTRIBUTION_PATTERNS:
        m = pattern.search(text[:200])
        if m:
            attribution = m.group(1).strip()
            break

    return title, attribution
