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
        Heuristic topic extractor:
        - Strip leading prompts/questions ("can you", "who is", "tell me", etc.)
        - Trim trailing filler/time hints ("now", "today") and punctuation
        - Prefer capitalized spans, else whole cleaned text
        - Collapse modifiers ("current", temporal tails)
        - Prefer "<role> of the <entity>" if present
        - Title-case for wiki titles
        """
        if not text:
            return None
        q = text.strip()

        leaders = [
            r"^(can you please|could you please|would you please)\s+",
            r"^(can you|can u|could you|would you)\s+",
            r"^(please)\s+",
            r"^(tell me|tell me about)\s+",
            r"^(what is|what are|what's|whats)\s+",
            r"^(who is|who are|who's|whos)\s+",
            r"^(do you know|find me|show me|give me)\s+",
        ]
        for pat in leaders:
            q = re.sub(pat, "", q, flags=re.IGNORECASE).strip()

        # Trim trailing filler/time hints and punctuation
        q = re.sub(r"\s*(right now|now|today)\s*$", "", q, flags=re.IGNORECASE).strip()
        q = re.sub(r"\s*(please|thanks|thank you)\s*$", "", q, flags=re.IGNORECASE).strip()
        q = q.strip(" ?!.,")

        if not q:
            return None

        # Prefer capitalized spans; else keep full cleaned q
        caps = re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", q)
        candidate = caps[-1] if caps else q

        # Remove leading articles
        candidate = re.sub(r"^(the|a|an)\s+", "", candidate, flags=re.IGNORECASE).strip()

        # Naive singularization
        if len(candidate) > 3 and candidate.lower().endswith("s"):
            candidate = candidate[:-1]

        # Simplify common modifiers/temporal tails
        candidate = self._simplify_topic(candidate)

        # Prefer "<role> of the <entity>" spans
        m = re.search(r"\b(president|prime minister|king|queen|chancellor|governor|mayor|head of state|head of government)\s+of\s+the\s+([a-zA-Z][a-zA-Z\s]+)\b", candidate, flags=re.IGNORECASE)
        if m:
            role = m.group(1).strip()
            entity = m.group(2).strip()
            candidate = f"{role} of the {entity}"

        # Title-case for wiki titles
        candidate = " ".join(w.capitalize() if w.isalpha() else w for w in candidate.split())

        # Guardrail: if the "topic" is essentially the whole utterance, prefer 'general'
        def _norm(s: str) -> str:
            import re as _re
            return _re.sub(r"[^a-z0-9]+"," ", s.lower()).strip()

        norm_q = _norm(text)
        norm_c = _norm(candidate)
        if norm_c == norm_q and len(norm_q.split()) >= 4:
            return "general"

        return candidate or None

    def _simplify_topic(self, s: str) -> str:
        """Remove common modifiers that hurt title resolution (current/temporal)."""
        t = s.strip()
        # Drop leading adjectives like 'current/new/latest/recent/modern'
        t = re.sub(r"^(current|new|latest|recent|modern)\s+", "", t, flags=re.IGNORECASE).strip()
        # Drop trailing temporal phrases
        t = re.sub(r"\s+in\s+the\s+\d{1,2}(st|nd|rd|th)\s+century\b", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s+in\s+\d{3,4}s\b", "", t, flags=re.IGNORECASE)  # in 1800s / 1990s
        t = re.sub(r"\s+in\s+\d{4}\b", "", t, flags=re.IGNORECASE)     # in 1871
        t = re.sub(r"\s+during\s+the\s+\w+\b", "", t, flags=re.IGNORECASE)
        # Collapse repeated spaces
        t = re.sub(r"\s+", " ", t).strip()
        return t
