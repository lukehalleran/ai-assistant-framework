# /knowledge/topic_manager.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, List, Set

from utils.logging_utils import get_logger

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

    def __init__(
        self,
        *,
        model_manager=None,
        llm_model: str = "gpt-4o-mini",
        enable_llm_fallback: bool = True,
    ):
        """
        Hybrid topic manager:
        - Heuristics first (fast, local)
        - Optional LLM fallback on ambiguous cases (e.g., pronouns like "that")

        Args:
            model_manager: Optional ModelManager instance. If not provided, we
                attempt to resolve from core.dependencies.deps (best effort).
            llm_model: API model name registered in ModelManager.api_models.
            enable_llm_fallback: Whether to attempt LLM on ambiguous inputs.
        """
        self.last_topic: Optional[str] = None
        self.enable_llm_fallback = enable_llm_fallback
        self.llm_model = llm_model
        self.logger = get_logger("topic_manager")

        # Optional dependency injection: resolve ModelManager if not provided
        self.model_manager = model_manager or self._resolve_model_manager()

        # Small stoplist to block deictic pronouns, vague fillers, and connectors
        self._stop_topics: Set[str] = {
            # deictic / vague
            "that", "this", "it", "there", "here",
            "today", "now", "stuff", "things", "misc", "none",
            # common connectors / interjections that are never a real topic
            "and", "but", "so", "ok", "okay", "yeah", "yep", "no", "yes",
        }

    # --- Public API expected by the rest of your app ---

    def update_from_user_input(self, text: str) -> None:
        """
        Update internal state based on a new user utterance.
        Never commit clearly ambiguous candidates (e.g., "it", "this").
        If LLM fallback is disabled/unavailable, keep the previous topic.
        """
        candidate = self._extract_primary_from_text(text)

        # If ambiguous, try LLM; if that fails, do not overwrite last_topic
        if self._is_ambiguous(candidate, text):
            resolved = self._llm_fallback(text)
            if resolved:
                self.last_topic = resolved
            # else: keep prior last_topic (no update)
            return

        if candidate:
            self.last_topic = candidate

    def get_primary_topic(self, text: Optional[str] = None) -> Optional[str]:
        """
        Return a single best topic string. If `text` is provided, derive from it;
        else return the last seen topic (may be None).
        Ambiguous candidates are ignored unless the LLM fallback resolves them.
        """
        if text:
            candidate = self._extract_primary_from_text(text)

            if self._is_ambiguous(candidate, text):
                resolved = self._llm_fallback(text)
                if resolved:
                    self.last_topic = resolved
                # If unresolved, do not change last_topic
            else:
                if candidate:
                    self.last_topic = candidate
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
        # Filter out obvious sentence connectors or single-letter fillers
        cap_exclude = {"i", "and", "but", "so", "ok", "okay", "yes", "no"}
        caps = [c for c in caps if c.strip() and c.strip().lower() not in cap_exclude]
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

    # --- Ambiguity detection and LLM fallback ---

    def _is_ambiguous(self, candidate: Optional[str], source_text: str) -> bool:
        """Return True if the heuristic topic is vague or low quality."""
        if not candidate:
            return True

        c = candidate.strip().strip("'\"")
        if not c:
            return True

        # Deictic or vague words
        if c.lower() in self._stop_topics:
            return True

        # Very short or mostly stopwords
        tokens = [t for t in re.split(r"\s+", c) if t]
        if len(tokens) <= 2 and all(t.lower() in self._stop_topics or len(t) <= 2 for t in tokens):
            return True

        # If heuristics returned 'general', try to get something better
        if c.lower() == "general":
            # Only treat as ambiguous if source is short or pronoun-heavy
            if len(re.findall(r"\b(that|this|it|here|there)\b", source_text.lower())) > 0:
                return True
            # Also if the source has no nouns-ish pattern (very rough)
            if not re.search(r"[a-zA-Z]{3,}", source_text):
                return True

        return False

    def _resolve_model_manager(self):
        """Best-effort resolution of ModelManager from dependency container."""
        try:
            from core.dependencies import deps  # type: ignore
            return deps.get_model_manager()
        except Exception:
            return None

    def _llm_fallback(self, text: str) -> Optional[str]:
        """
        Invoke a small LLM to extract a concrete topic when heuristics are
        ambiguous. Returns a sanitized topic or None.
        """
        if not self.enable_llm_fallback or self.model_manager is None:
            return None

        system = (
            "You extract one short, concrete topic (2–4 words, noun phrase). "
            "No pronouns or vague words (that/this/it/stuff/none). "
            "No punctuation. If impossible, return CONTINUE."
        )
        prompt = text.strip()
        try:
            raw = self.model_manager.generate(
                prompt=prompt,
                model_name=self.llm_model,
                system_prompt=system,
                max_tokens=16,
                temperature=0.0,
                top_p=0.0,
            )
        except Exception as e:
            # Fail open on any model errors
            self.logger.debug(f"[TopicManager] LLM fallback skipped: {e}")
            return None

        if not isinstance(raw, str):
            return None
        resp = raw.strip().strip('"').strip()
        if not resp or resp.upper() == "CONTINUE":
            return None

        # Sanitize output: disallow stoplist, tiny tokens, or punctuation-only
        if resp.startswith("["):
            # Likely a stub or error string from model manager
            return None
        if resp.lower() in self._stop_topics:
            return None
        # Remove trailing punctuation
        resp = resp.strip("?!.,;:")
        if not resp:
            return None

        # Title-case similar to heuristic for consistency
        topic = " ".join(w.capitalize() if w.isalpha() else w for w in resp.split())
        return topic or None
