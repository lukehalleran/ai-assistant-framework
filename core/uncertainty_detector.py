"""
# core/uncertainty_detector.py

Module Contract
- Purpose: Detect when an LLM response indicates uncertainty or inability to answer,
  enabling automatic fallback to agentic search for deeper memory retrieval.
- Inputs:
  - response: str (the LLM response text to check)
  - embedder: optional SentenceTransformer for semantic layer
  - semantic_threshold: float (cosine similarity threshold for semantic match)
  - max_length: int (responses longer than this skip fallback)
- Outputs:
  - UncertaintyResult (is_uncertain, confidence, trigger_type, matched_pattern)
- Key methods:
  - UncertaintyDetector.detect() — static method, runs keyword + semantic layers
- Side effects: None (pure detection, stateless). Anchor embeddings cached at module level.
- Detection layers:
  1. Length guard: strip hedge prefixes, skip if substantive content > max_length
  2. Keyword layer: ~18 compiled regex patterns with confidence scores
  3. Semantic layer: cosine similarity against 8 pre-embedded anchor sentences
"""
import re
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from utils.logging_utils import get_logger

logger = get_logger("uncertainty_detector")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class UncertaintyResult(BaseModel):
    """Result of uncertainty detection on an LLM response."""
    is_uncertain: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    trigger_type: str = ""      # "keyword" | "semantic" | ""
    matched_pattern: str = ""   # description of what triggered


# ---------------------------------------------------------------------------
# Keyword patterns: (compiled_regex, confidence_score)
# ---------------------------------------------------------------------------

_UNCERTAINTY_PATTERNS = [
    # Direct "don't know / recall" phrasings
    (re.compile(
        r"\bi\s+don'?t\s+(?:recall|remember|have\s+(?:any\s+)?(?:information|record|memory|context))",
        re.I), 0.90, "don't recall/remember/have info"),
    (re.compile(
        r"\bi\s+(?:can'?t|cannot|couldn'?t)\s+(?:find|recall|remember|locate|retrieve)",
        re.I), 0.85, "can't find/recall"),
    (re.compile(
        r"\bi'?m\s+not\s+(?:sure|certain|aware)\s+(?:what|about|if|whether)",
        re.I), 0.80, "not sure/certain/aware"),
    (re.compile(
        r"\bi\s+don'?t\s+(?:seem\s+to\s+)?have\s+(?:enough|any|sufficient)",
        re.I), 0.85, "don't have enough/any"),

    # "No record" phrasings
    (re.compile(
        r"\bno\s+(?:record|information|data|context|memory|conversation)s?\s+(?:about|regarding|on|of|for)",
        re.I), 0.80, "no record/information about"),
    (re.compile(
        r"\bcouldn'?t\s+find\s+(?:any|a)\s+(?:record|mention|reference)",
        re.I), 0.85, "couldn't find any record"),

    # Apology-hedged uncertainty
    (re.compile(
        r"\b(?:unfortunately|I'?m\s+sorry),?\s+i\s+(?:don'?t|can'?t|cannot)",
        re.I), 0.85, "unfortunately I don't/can't"),
    (re.compile(
        r"\bapologize.*(?:don'?t|can'?t|unable)",
        re.I), 0.75, "apologize...unable"),

    # "Based on context" disclaimers
    (re.compile(
        r"\bbased\s+on\s+(?:the|my|available)\s+(?:context|information|records).*(?:don'?t|no\s+|can'?t)",
        re.I), 0.80, "based on context...don't"),
    (re.compile(
        r"\bfrom\s+(?:what|the\s+information)\s+(?:i|available).*(?:don'?t|unable|can'?t)",
        re.I), 0.75, "from available info...can't"),

    # Temporal recall failures
    (re.compile(
        r"\bi\s+don'?t\s+(?:have\s+)?(?:specific|detailed)\s+(?:recall|memory|record)\s+of",
        re.I), 0.85, "don't have specific recall of"),
    (re.compile(
        r"\bi\s+(?:can'?t|don'?t)\s+(?:recall|remember)\s+(?:what\s+we|our|any)\s+(?:discuss|talk|convers)",
        re.I), 0.90, "can't recall what we discussed"),

    # "Not in my" patterns
    (re.compile(
        r"\bnot\s+(?:in|within|part\s+of)\s+(?:my|the)\s+(?:memory|records|context|knowledge)",
        re.I), 0.80, "not in my memory/records"),

    # Unable patterns
    (re.compile(
        r"\bunable\s+to\s+(?:find|locate|recall|retrieve|determine)",
        re.I), 0.85, "unable to find/recall"),

    # "Haven't discussed" / "no conversation"
    (re.compile(
        r"\b(?:haven'?t|have\s+not)\s+(?:discussed|talked\s+about|covered)",
        re.I), 0.80, "haven't discussed"),
    (re.compile(
        r"\bno\s+(?:previous\s+)?(?:conversation|discussion|record)s?\s+(?:about|regarding|on\s+that)",
        re.I), 0.80, "no conversation/discussion about"),

    # Explicit disclaimers
    (re.compile(
        r"\bmy\s+(?:memory|records?|context)\s+(?:doesn'?t|don'?t|does\s+not)\s+(?:contain|include|show)",
        re.I), 0.85, "my memory doesn't contain"),
    (re.compile(
        r"\bthere(?:'s|\s+is)\s+(?:no|nothing)\s+in\s+(?:my|the)\s+(?:context|memory|records)",
        re.I), 0.85, "there's nothing in my memory"),
]

_MIN_CONFIDENCE = 0.70


# ---------------------------------------------------------------------------
# Hedge-prefix stripping (for length guard)
# ---------------------------------------------------------------------------

_HEDGE_PREFIXES = [
    re.compile(r"^(?:I'?m\s+not\s+(?:sure|certain),?\s*(?:but|however)\s*)", re.I),
    re.compile(r"^(?:I\s+don'?t\s+have\s+specific.*?,?\s*(?:but|however)\s*)", re.I),
    re.compile(r"^(?:(?:Unfortunately|I'?m\s+sorry),?\s*)", re.I),
    re.compile(r"^(?:I\s+(?:can'?t|couldn'?t)\s+find.*?,?\s*(?:but|however)\s*)", re.I),
]


def _strip_hedge_prefix(text: str) -> str:
    """Strip known hedge prefixes to measure substantive content length."""
    stripped = text
    for pattern in _HEDGE_PREFIXES:
        stripped = pattern.sub("", stripped, count=1)
    return stripped.strip()


# ---------------------------------------------------------------------------
# Semantic anchor embeddings (cached at module level)
# ---------------------------------------------------------------------------

_UNCERTAINTY_ANCHORS = [
    "I don't have any information about that in my memory.",
    "I'm not sure what we discussed about that topic.",
    "I don't recall any conversations about that.",
    "I couldn't find any records of that in our history.",
    "I don't have enough context to answer that question.",
    "I'm unable to recall the specifics of what we talked about.",
    "I don't seem to have any relevant memories about this.",
    "Unfortunately, I can't find anything in my records about that.",
]

_anchor_embeddings_cache: Optional[np.ndarray] = None


def _get_anchor_embeddings(embedder) -> Optional[np.ndarray]:
    """Get or compute cached anchor embeddings."""
    global _anchor_embeddings_cache
    if _anchor_embeddings_cache is not None:
        return _anchor_embeddings_cache
    try:
        _anchor_embeddings_cache = embedder.encode(
            _UNCERTAINTY_ANCHORS,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        logger.debug(
            f"[UncertaintyDetector] Computed anchor embeddings: "
            f"shape={_anchor_embeddings_cache.shape}"
        )
        return _anchor_embeddings_cache
    except Exception as e:
        logger.warning(f"[UncertaintyDetector] Failed to compute anchor embeddings: {e}")
        return None


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class UncertaintyDetector:
    """Detects when an LLM response indicates uncertainty or inability to answer."""

    SEMANTIC_CHAR_LIMIT = 300  # Only embed first N chars of response

    @staticmethod
    def detect(
        response: str,
        embedder=None,
        semantic_threshold: float = 0.70,
        max_length: int = 400,
    ) -> UncertaintyResult:
        """Check if response indicates uncertainty.

        Args:
            response: The LLM response text to check.
            embedder: Optional SentenceTransformer for semantic layer.
            semantic_threshold: Cosine similarity threshold for semantic match.
            max_length: Responses longer than this (after hedge-stripping) skip fallback.

        Returns:
            UncertaintyResult with detection details.
        """
        text = (response or "").strip()
        if not text or len(text) < 10:
            return UncertaintyResult(is_uncertain=False)

        # --- Length guard: long responses are probably answering, even if hedged ---
        substantive = _strip_hedge_prefix(text)
        if len(substantive) > max_length:
            return UncertaintyResult(is_uncertain=False)

        # --- Keyword layer ---
        best_conf = 0.0
        best_desc = ""
        for pattern, confidence, description in _UNCERTAINTY_PATTERNS:
            if pattern.search(text):
                if confidence > best_conf:
                    best_conf = confidence
                    best_desc = description

        if best_conf >= _MIN_CONFIDENCE:
            return UncertaintyResult(
                is_uncertain=True,
                confidence=best_conf,
                trigger_type="keyword",
                matched_pattern=best_desc,
            )

        # --- Semantic layer (only if embedder provided) ---
        if embedder is not None:
            try:
                anchor_embs = _get_anchor_embeddings(embedder)
                if anchor_embs is not None:
                    from sklearn.metrics.pairwise import cosine_similarity

                    prefix = text[:UncertaintyDetector.SEMANTIC_CHAR_LIMIT]
                    response_emb = embedder.encode(
                        [prefix],
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    )
                    similarities = cosine_similarity(response_emb, anchor_embs)[0]
                    max_sim = float(np.max(similarities))
                    max_idx = int(np.argmax(similarities))

                    if max_sim >= semantic_threshold:
                        return UncertaintyResult(
                            is_uncertain=True,
                            confidence=min(max_sim, 1.0),
                            trigger_type="semantic",
                            matched_pattern=_UNCERTAINTY_ANCHORS[max_idx],
                        )
            except Exception as e:
                logger.debug(f"[UncertaintyDetector] Semantic layer failed (non-fatal): {e}")

        # --- Neither layer triggered ---
        return UncertaintyResult(is_uncertain=False)
