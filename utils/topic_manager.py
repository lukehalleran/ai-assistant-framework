# /knowledge/topic_manager.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class TopicSpan:
    text: str
    start: int
    end: int
    score: float = 1.0

class TopicManager:
    """
    Lightweight topic extractor that keeps track of the last primary topic.
    Designed to feed a single canonical topic string into set_topic_resolver(...).
    """

    def __init__(self):
        self.last_topic: Optional[str] = None

    # --- Public API expected by the rest of your app ---

    def update_from_user_input(self, text: str) -> None:
        """
        Update internal state based on a new user utterance.
        """
        topic = self._extract_primary_from_text(text)
        if topic:
            self.last_topic = topic

    def get_primary_topic(self, text: Optional[str] = None) -> Optional[str]:
        """
        Return a single best topic string. If `text` is provided, derive from it;
        else return the last seen topic (may be None).
        """
        if text:
            topic = self._extract_primary_from_text(text)
            if topic:
                self.last_topic = topic
        return self.last_topic

    # Some call sites used `resolve_topic`; keep it as an alias returning str|None.
    def resolve_topic(self, text: Optional[str] = None) -> Optional[str]:
        return self.get_primary_topic(text)

    # --- Heuristics (cheap, fast, no external deps) ---

    def _extract_primary_from_text(self, text: str) -> Optional[str]:
        """
        Very small heuristic:
        - Strip leading prompty phrases ("tell me about", "what is", etc.)
        - Remove trailing fluff/punct
        - Prefer a capitalized noun-ish span (Title Case fallback)
        - Singularize naive plural (cats -> cat) when that looks right
        """
        if not text:
            return None
        q = text.strip()

        # Remove leading intent phrases
        leaders = [
            r"^(tell me about|what is|what are|who is|who are|explain|briefly explain)\s+",
        ]
        for pat in leaders:
            q = re.sub(pat, "", q, flags=re.IGNORECASE).strip()

        # Trim trailing filler
        q = re.sub(r"\s*(please|thanks|thank you)\s*$", "", q, flags=re.IGNORECASE).strip()
        q = q.strip(" ?!.,")

        if not q:
            return None

        # If the span is long, keep the head nouny tail: e.g., "the president of the United States" -> "United States"
        # Super cheap: take the last capitalized chunk if present.
        caps = re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", q)
        if caps:
            candidate = caps[-1]
        else:
            candidate = q

        # Remove leading articles in candidate
        candidate = re.sub(r"^(the|a|an)\s+", "", candidate, flags=re.IGNORECASE).strip()

        # Naive singularization (dogs -> dog) when it looks like a simple plural
        if len(candidate) > 3 and candidate.lower().endswith("s"):
            candidate = candidate[:-1]

        # Title-case for wiki titles
        candidate = " ".join(w.capitalize() if w.isalpha() else w for w in candidate.split())

        return candidate or None
