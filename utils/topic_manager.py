# /knowledge/topic_manager.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, List, Set

from utils.logging_utils import get_logger

# Optional spaCy NER for stage 2 entity extraction
_spacy_nlp = None
def _load_spacy():
    """Lazy load spaCy model (only when needed)."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            # If spaCy unavailable, silently skip (fallback to LLM)
            _spacy_nlp = False
    return _spacy_nlp if _spacy_nlp is not False else None

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

        # Expanded stoplist to block garbage topics
        # Based on actual garbage observed in logs: "Idk", "How", "Well", "Thi", "Hub", etc.
        self._stop_topics: Set[str] = {
            # deictic / vague
            "that", "this", "it", "there", "here", "what", "which", "who",
            "today", "now", "stuff", "things", "misc", "none", "something",
            # common connectors / interjections that are never a real topic
            "and", "but", "so", "ok", "okay", "yeah", "yep", "no", "yes", "nope",
            "oh", "ah", "um", "uh", "hmm", "hm", "well", "like", "anyway",
            # articles and common words that shouldn't be topics
            "the", "a", "an", "i", "just", "only", "first", "second", "maybe",
            # garbage from actual logs - short meaningless words
            "idk", "how", "thi", "hub", "hit", "not", "got", "get", "let",
            "assuming", "other", "those", "loop", "card", "omg", "wow", "man",
            # more vague/short words
            "one", "two", "some", "any", "all", "most", "few", "many",
            "good", "bad", "new", "old", "big", "small", "last", "next",
            "really", "very", "much", "more", "less", "also", "even", "still",
            # web/UI action words
            "click", "tap", "press", "select", "choose", "enter", "submit",
            "our", "your", "my", "their", "his", "her", "its",
            # common newsletter/subscription words
            "newsletter", "update", "updates", "news",
        }

        # Web UI patterns - common text from pasted web content that shouldn't be topics
        self._web_ui_patterns: Set[str] = {
            # Navigation/UI elements
            "homepage", "home page", "sign in", "sign up", "log in", "log out",
            "subscribe", "unsubscribe", "read more", "learn more", "click here",
            "see more", "view more", "show more", "load more", "next page",
            "previous", "back", "forward", "menu", "search", "settings",
            # Common web boilerplate
            "cookie", "cookies", "privacy policy", "terms of service", "terms and conditions",
            "contact us", "about us", "help center", "faq", "support",
            "share", "tweet", "post", "comment", "reply", "like", "follow",
            # News site patterns
            "breaking news", "latest news", "trending", "popular", "featured",
            "advertisement", "sponsored", "promoted", "ad", "ads",
            "suggested topics", "related articles", "you may also like",
        }

    # --- Public API expected by the rest of your app ---

    def update_from_user_input(self, text: str) -> None:
        """
        Update internal state based on a new user utterance.
        Never commit clearly ambiguous candidates (e.g., "it", "this").

        LLM-first pipeline (2026-01-12 refactor):
        1. LLM extraction (most reliable for conversational/pasted content)
        2. spaCy NER fallback (if LLM unavailable/fails)
        3. Heuristics fallback (last resort)

        This ordering ensures quality topics even for emotional messages,
        pasted web content, and casual conversation.
        """
        # Stage 1: Try LLM first (most reliable)
        if self.enable_llm_fallback and self.model_manager:
            llm_topic = self._llm_fallback(text)
            if llm_topic and not self._is_ambiguous(llm_topic, text):
                self.last_topic = llm_topic
                self.logger.debug(f"[TopicManager] LLM extracted topic: {llm_topic}")
                return

        # Stage 2: spaCy NER fallback
        spacy_topic = self._spacy_ner_extraction(text)
        if spacy_topic and not self._is_ambiguous(spacy_topic, text):
            self.last_topic = spacy_topic
            self.logger.debug(f"[TopicManager] spaCy extracted topic: {spacy_topic}")
            return

        # Stage 3: Heuristics fallback (last resort)
        candidate = self._extract_primary_from_text(text)
        if candidate and not self._is_ambiguous(candidate, text):
            self.last_topic = candidate
            self.logger.debug(f"[TopicManager] Heuristic extracted topic: {candidate}")
            return

        # All methods failed or produced garbage - keep prior topic
        self.logger.debug(f"[TopicManager] No quality topic extracted, keeping: {self.last_topic}")

    def get_primary_topic(self, text: Optional[str] = None) -> Optional[str]:
        """
        Return a single best topic string. If `text` is provided, derive from it;
        else return the last seen topic (may be None).

        LLM-first pipeline (2026-01-12 refactor):
        1. LLM extraction (most reliable for conversational/pasted content)
        2. spaCy NER fallback (if LLM unavailable/fails)
        3. Heuristics fallback (last resort)
        """
        if text:
            # Stage 1: Try LLM first (most reliable)
            if self.enable_llm_fallback and self.model_manager:
                llm_topic = self._llm_fallback(text)
                if llm_topic and not self._is_ambiguous(llm_topic, text):
                    self.last_topic = llm_topic
                    return self.last_topic

            # Stage 2: spaCy NER fallback
            spacy_topic = self._spacy_ner_extraction(text)
            if spacy_topic and not self._is_ambiguous(spacy_topic, text):
                self.last_topic = spacy_topic
                return self.last_topic

            # Stage 3: Heuristics fallback (last resort)
            candidate = self._extract_primary_from_text(text)
            if candidate and not self._is_ambiguous(candidate, text):
                self.last_topic = candidate
                return self.last_topic

            # All methods failed - don't update last_topic
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

        # Stricter guardrail: return 'general' if candidate is too similar to source
        q_words = set(norm_q.split())
        c_words = set(norm_c.split())

        # If candidate contains >70% of source words, or is very long, it's the whole utterance
        if len(c_words) > 0:
            overlap = len(q_words & c_words) / len(q_words)
            if overlap > 0.7 or len(c_words) > 6:
                return "general"

        # Original check: exact match with 4+ words
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

        c_lower = c.lower()
        tokens = [t.lower() for t in re.split(r"\s+", c) if t]

        # Single-word stopwords - reject completely
        if len(tokens) == 1 and c_lower in self._stop_topics:
            return True

        # Multi-word: reject if ALL words are stopwords
        if len(tokens) > 1 and all(t in self._stop_topics for t in tokens):
            self.logger.debug(f"[TopicManager] Rejecting all-stopword phrase: {c}")
            return True

        # Web UI patterns - reject pasted web content (exact match)
        if c_lower in self._web_ui_patterns:
            self.logger.debug(f"[TopicManager] Rejecting web UI pattern: {c}")
            return True

        # Very short candidates (< 3 chars) are almost always garbage
        if len(c) < 3:
            return True

        # Very short or mostly stopwords (tokens already defined above)
        if len(tokens) <= 2 and all(t in self._stop_topics or len(t) <= 2 for t in tokens):
            return True

        # If heuristics returned 'general', ALWAYS use LLM to extract something better
        # This handles conversational/emotional messages that don't have clear entities
        if c_lower == "general":
            return True

        # Also check if candidate is too long (>6 words) - likely the whole message
        if len(tokens) > 6:
            return True

        # Check for partial/truncated words (like "Thi" from "This")
        # Single token under 4 chars that's not a known acronym/word
        if len(tokens) == 1 and len(tokens[0]) < 4 and tokens[0].lower() not in {"ice", "fbi", "cia", "fed", "gop", "dem", "nyc", "usa"}:
            return True

        return False

    def _resolve_model_manager(self):
        """Best-effort resolution of ModelManager from dependency container."""
        try:
            from core.dependencies import deps  # type: ignore
            return deps.get_model_manager()
        except Exception:
            return None

    def _spacy_ner_extraction(self, text: str) -> Optional[str]:
        """
        Extract topic using spaCy noun chunks + NER (stage 2).

        Strategy:
        1. First try noun_chunks for meaningful compound phrases (e.g., "first degree murder")
        2. Fall back to NER entities if no good noun chunks
        3. Filter out low-quality NER types (ORDINAL, CARDINAL, DATE, TIME) when alone

        Priority order for NER: PERSON > ORG > GPE > PRODUCT > EVENT > other entities
        """
        nlp = _load_spacy()
        if nlp is None:
            return None

        try:
            doc = nlp(text)

            # Stage 2a: Try noun chunks first (captures compound terms like "first degree murder")
            noun_chunk_candidate = self._extract_best_noun_chunk(doc)
            if noun_chunk_candidate:
                self.logger.debug(f"[TopicManager] spaCy noun_chunk found: {noun_chunk_candidate}")
                return noun_chunk_candidate

            # Stage 2b: Fall back to NER entities
            # Filter entities by priority (exclude low-quality types for single-word entities)
            priority_order = ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "NORP"]
            low_quality_types = {"ORDINAL", "CARDINAL", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY"}

            # Group entities by type
            entities_by_type = {}
            for ent in doc.ents:
                # Skip very short entities (single letters, numbers)
                if len(ent.text.strip()) <= 1:
                    continue
                # Skip stopwords/vague terms
                if ent.text.lower().strip() in self._stop_topics:
                    continue
                # Skip single-word low-quality types (ORDINAL "first", CARDINAL "two", etc.)
                if ent.label_ in low_quality_types and len(ent.text.split()) == 1:
                    self.logger.debug(f"[TopicManager] Skipping low-quality single-word {ent.label_}: {ent.text}")
                    continue

                if ent.label_ not in entities_by_type:
                    entities_by_type[ent.label_] = []
                entities_by_type[ent.label_].append(ent.text.strip())

            # Select best entity based on priority
            for entity_type in priority_order:
                if entity_type in entities_by_type:
                    # Return the first (usually most relevant) entity of this type
                    candidate = entities_by_type[entity_type][0]
                    self.logger.debug(f"[TopicManager] spaCy NER found {entity_type}: {candidate}")
                    return candidate

            # If no priority entities, return any entity found (excluding low-quality)
            for etype, ents in entities_by_type.items():
                if etype not in low_quality_types:
                    candidate = ents[0]
                    self.logger.debug(f"[TopicManager] spaCy NER found {etype}: {candidate}")
                    return candidate

            return None

        except Exception as e:
            self.logger.debug(f"[TopicManager] spaCy NER failed: {e}")
            return None

    def _extract_best_noun_chunk(self, doc) -> Optional[str]:
        """
        Extract the most meaningful noun chunk from spaCy doc.

        Prefers multi-word chunks that contain substantive content.
        Filters out chunks that are just pronouns, determiners, or stop words.
        """
        # Minimum words for a "good" noun chunk (captures "first degree murder" but not "it")
        MIN_CHUNK_WORDS = 2

        # Words that make a chunk low-quality when they're the only content
        chunk_stopwords = {
            "it", "this", "that", "these", "those", "the", "a", "an",
            "i", "you", "he", "she", "they", "we", "me", "him", "her",
            "something", "anything", "nothing", "everything", "stuff", "things"
        }

        best_chunk = None
        best_score = 0

        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            chunk_lower = chunk_text.lower()
            words = chunk_lower.split()

            # Skip single-word chunks
            if len(words) < MIN_CHUNK_WORDS:
                continue

            # Skip if all words are stopwords
            content_words = [w for w in words if w not in chunk_stopwords and len(w) > 2]
            if not content_words:
                continue

            # Score: prefer longer chunks with more content words
            score = len(content_words) + (len(words) * 0.5)

            # Bonus for chunks containing specific domain terms
            domain_terms = {"murder", "case", "trial", "shooting", "incident", "crisis", "policy"}
            if any(term in chunk_lower for term in domain_terms):
                score += 2

            if score > best_score:
                best_score = score
                best_chunk = chunk_text

        # Clean up: remove leading determiners
        if best_chunk:
            best_chunk = re.sub(r"^(the|a|an)\s+", "", best_chunk, flags=re.IGNORECASE).strip()
            # Title case
            best_chunk = " ".join(w.capitalize() if w.isalpha() else w for w in best_chunk.split())

        return best_chunk if best_chunk and len(best_chunk) > 2 else None

    def _llm_fallback(self, text: str) -> Optional[str]:
        """
        Invoke a small LLM to extract a concrete topic when heuristics are
        ambiguous. Returns a sanitized topic or None.
        """
        if not self.enable_llm_fallback or self.model_manager is None:
            return None

        system = (
            "You extract one short, concrete topic (2–5 words, noun phrase). "
            "For emotional/conversational messages, identify the main subject or theme. "
            "Examples: 'I am lonely' → 'Loneliness', 'School starts soon' → 'School starting', "
            "'Having trouble with dating' → 'Dating struggles'. "
            "No pronouns or vague words (that/this/it/stuff). "
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
