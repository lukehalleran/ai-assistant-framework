"""
utils/need_detector.py

Need-type detection system: presence vs perspective seeking.

Complements tone_detector.py (severity) with a second axis:
- PRESENCE: User needs warmth, acknowledgment, "I'm here with you"
- PERSPECTIVE: User wants engagement, questions, reframes, problem-solving
- NEUTRAL: Mixed or unclear signals

Uses hybrid detection (keyword + semantic) following tone_detector.py patterns.
"""

import os
import re
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from utils.logging_utils import get_logger

logger = get_logger("need_detector")


# ===== Configuration =====

NEED_CONFIG = {
    # Semantic similarity thresholds
    "threshold_presence": float(os.getenv("NEED_THRESHOLD_PRESENCE", "0.60")),
    "threshold_perspective": float(os.getenv("NEED_THRESHOLD_PERSPECTIVE", "0.60")),

    # Score combination weights
    "keyword_weight": float(os.getenv("NEED_KEYWORD_WEIGHT", "0.4")),
    "semantic_weight": float(os.getenv("NEED_SEMANTIC_WEIGHT", "0.6")),

    # Fast-path threshold (skip semantic if keyword confidence high)
    "high_confidence_threshold": float(os.getenv("NEED_HIGH_CONF", "0.8")),

    # Message length thresholds
    "short_message_threshold": int(os.getenv("NEED_SHORT_MSG", "40")),
    "long_message_threshold": int(os.getenv("NEED_LONG_MSG", "80")),
}


class NeedType(Enum):
    """What type of response the user needs."""
    PRESENCE = "presence"      # Warmth, acknowledgment, "I'm here"
    PERSPECTIVE = "perspective" # Engagement, questions, reframes
    NEUTRAL = "neutral"        # Mixed or unclear signals


@dataclass
class NeedAnalysis:
    """Result of need-type detection."""
    need_type: NeedType
    confidence: float
    trigger: str  # "keyword", "semantic", "hybrid_agreement", etc.
    raw_scores: Dict  # Detailed scores for debugging
    explanation: str


# ===== Pattern Constants =====

PRESENCE_PATTERNS = {
    # Direct emotional state declarations
    "i am lonely",
    "i'm lonely",
    "i feel alone",
    "i'm sad",
    "i'm tired",
    "this sucks",
    "it hurts",
    "i'm struggling",
    "i feel lost",
    "i'm scared",
    "i don't know anymore",

    # Resignation/exhaustion markers
    "ugh",
    "i just...",
    "it's just...",
    "i don't know",
    "whatever",

    # Short emotional exhales
    "sigh",
    "man",
    "god",
}

PERSPECTIVE_PATTERNS = {
    # Reasoning/analysis language
    "i think",
    "i feel like",  # Note: different from "i feel X"
    "the reason is",
    "i'm not sure if",
    "maybe",
    "probably",
    "should i",
    "what if",
    "do you think",

    # Problem-framing
    "the problem is",
    "the issue is",
    "i'm trying to",
    "i need to figure out",
    "how do i",

    # Causal/contrastive connectors
    "because",
    "but",
    "although",
    "however",
    "on the other hand",
}

NEED_EXEMPLARS = {
    "presence": [
        # Pure emotional state declarations
        "I am lonely",
        "I'm so sad right now",
        "This hurts",
        "I feel empty inside",
        "I'm exhausted",
        "I don't know what to do with myself",
        "Everything feels heavy",
        "I'm struggling today",
        "I feel lost",
        "It's just hard",
        "I'm tired of this",
        "I feel so alone",
        "Nothing feels right",
        "I'm not okay",
        "This sucks",
        "I can't shake this feeling",
        "I'm worn out",
        "I feel invisible",
        "It's overwhelming",
        "I just need a moment",
    ],
    "perspective": [
        # Problem-framing and analysis-seeking
        "I think the issue is my approach",
        "Should I try something different?",
        "The reason this keeps happening is probably because",
        "I'm not sure if I should take the job or wait",
        "Do you think I'm overreacting to this?",
        "Maybe I need to change my strategy",
        "What would you do in my situation?",
        "I feel like the problem is how I'm framing it",
        "I'm trying to figure out the best approach",
        "On one hand I could do X, but on the other",
        "I'm weighing my options here",
        "What's your take on this?",
        "I think I need to reconsider my plan",
        "The thing I'm struggling to understand is",
        "I'm not sure how to interpret this",
        "Would it make sense to try a different angle?",
        "I've been thinking about whether I should",
        "Help me think through this",
        "I'm trying to decide between these options",
        "What factors should I consider here?",
    ],
}


# ===== Embedder Cache (mirrors tone_detector.py) =====

_embedder_cache: Optional[object] = None
_need_exemplar_embeddings_cache: Optional[Dict[str, np.ndarray]] = None


def _get_embedder(model_manager=None):
    """Get or create the sentence embedder (shared with tone_detector if possible)."""
    global _embedder_cache

    if _embedder_cache is not None:
        return _embedder_cache

    # Try tone_detector's cache first (avoid duplicate embedders)
    try:
        from utils.tone_detector import _get_embedder as tone_get_embedder
        _embedder_cache = tone_get_embedder(model_manager)
        if _embedder_cache:
            logger.info("[NeedDetector] Sharing embedder with tone_detector")
            return _embedder_cache
    except ImportError:
        pass

    # Fallback: create our own
    if model_manager and hasattr(model_manager, "get_embedder"):
        try:
            _embedder_cache = model_manager.get_embedder()
            logger.info("[NeedDetector] Using embedder from model_manager")
            return _embedder_cache
        except Exception as e:
            logger.warning(f"[NeedDetector] Failed to get embedder from model_manager: {e}")

    try:
        from sentence_transformers import SentenceTransformer
        _embedder_cache = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("[NeedDetector] Created fallback embedder")
        return _embedder_cache
    except Exception as e:
        logger.error(f"[NeedDetector] Failed to create embedder: {e}")
        return None


def _get_need_exemplar_embeddings(model_manager=None) -> Dict[str, np.ndarray]:
    """Compute and cache exemplar embeddings for need types."""
    global _need_exemplar_embeddings_cache

    if _need_exemplar_embeddings_cache is not None:
        return _need_exemplar_embeddings_cache

    embedder = _get_embedder(model_manager)
    if embedder is None:
        return {}

    logger.info("[NeedDetector] Computing need exemplar embeddings (one-time setup)...")

    _need_exemplar_embeddings_cache = {}

    for need_type, examples in NEED_EXEMPLARS.items():
        try:
            embeddings = embedder.encode(examples, convert_to_numpy=True)
            mean_embedding = np.mean(embeddings, axis=0)
            _need_exemplar_embeddings_cache[need_type] = mean_embedding
            logger.debug(f"[NeedDetector] Computed {need_type} exemplar from {len(examples)} examples")
        except Exception as e:
            logger.error(f"[NeedDetector] Failed to compute {need_type} exemplar: {e}")
            _need_exemplar_embeddings_cache[need_type] = np.zeros(384)

    logger.info("[NeedDetector] Need exemplar embeddings ready")
    return _need_exemplar_embeddings_cache


# ===== Detection Functions =====

def _keyword_need_detection(message: str) -> NeedAnalysis:
    """Fast keyword + structural detection."""

    message_lower = message.lower()
    presence_score = 0.0
    perspective_score = 0.0

    # Keyword matching
    for pattern in PRESENCE_PATTERNS:
        if pattern in message_lower:
            presence_score += 1.0

    for pattern in PERSPECTIVE_PATTERNS:
        if pattern in message_lower:
            perspective_score += 1.0

    # Structural signals
    msg_len = len(message)

    # Short messages favor presence
    if msg_len < NEED_CONFIG["short_message_threshold"]:
        presence_score += 0.5
    elif msg_len > NEED_CONFIG["long_message_threshold"]:
        perspective_score += 0.5

    # Question marks strongly favor perspective
    if "?" in message:
        perspective_score += 1.0

    # Causal connectors favor perspective
    causal = ["because", "since", "so that", "in order to"]
    if any(c in message_lower for c in causal):
        perspective_score += 0.5

    # Contrastive connectors favor perspective
    contrastive = ["but", "although", "however", "on the other hand"]
    if any(c in message_lower for c in contrastive):
        perspective_score += 0.3

    # Single-word emotional exhales favor presence
    if message_lower.strip() in {"ugh", "sigh", "man", "god", "fuck"}:
        presence_score += 1.5

    # Compute classification
    diff = presence_score - perspective_score

    if abs(diff) < 0.5:
        return NeedAnalysis(
            need_type=NeedType.NEUTRAL,
            confidence=0.3,
            trigger="keyword",
            raw_scores={"presence": presence_score, "perspective": perspective_score},
            explanation="Mixed or insufficient signals"
        )
    elif diff > 0:
        confidence = min(1.0, diff / 3.0)
        return NeedAnalysis(
            need_type=NeedType.PRESENCE,
            confidence=confidence,
            trigger="keyword",
            raw_scores={"presence": presence_score, "perspective": perspective_score},
            explanation=f"Keyword presence signals: {presence_score:.1f}"
        )
    else:
        confidence = min(1.0, abs(diff) / 3.0)
        return NeedAnalysis(
            need_type=NeedType.PERSPECTIVE,
            confidence=confidence,
            trigger="keyword",
            raw_scores={"presence": presence_score, "perspective": perspective_score},
            explanation=f"Keyword perspective signals: {perspective_score:.1f}"
        )


def _semantic_need_detection(
    message: str,
    model_manager=None
) -> NeedAnalysis:
    """Semantic similarity-based need detection."""

    embedder = _get_embedder(model_manager)
    if embedder is None:
        return NeedAnalysis(
            need_type=NeedType.NEUTRAL,
            confidence=0.0,
            trigger="semantic_unavailable",
            raw_scores={},
            explanation="Embedder unavailable"
        )

    exemplar_embeddings = _get_need_exemplar_embeddings(model_manager)
    if not exemplar_embeddings:
        return NeedAnalysis(
            need_type=NeedType.NEUTRAL,
            confidence=0.0,
            trigger="semantic_unavailable",
            raw_scores={},
            explanation="Exemplar embeddings unavailable"
        )

    # Encode message
    try:
        message_embedding = embedder.encode(message, convert_to_numpy=True)
    except Exception as e:
        logger.error(f"[NeedDetector] Failed to encode message: {e}")
        return NeedAnalysis(
            need_type=NeedType.NEUTRAL,
            confidence=0.0,
            trigger="encoding_error",
            raw_scores={},
            explanation=f"Encoding error: {e}"
        )

    # Compute similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    presence_sim = cosine_similarity(message_embedding, exemplar_embeddings["presence"])
    perspective_sim = cosine_similarity(message_embedding, exemplar_embeddings["perspective"])

    raw_scores = {
        "presence": float(presence_sim),
        "perspective": float(perspective_sim)
    }

    # Determine classification
    diff = presence_sim - perspective_sim

    if abs(diff) < 0.05:  # Very close - neutral
        return NeedAnalysis(
            need_type=NeedType.NEUTRAL,
            confidence=0.4,
            trigger="semantic",
            raw_scores=raw_scores,
            explanation=f"Semantic scores too close: P={presence_sim:.2f}, R={perspective_sim:.2f}"
        )
    elif diff > 0 and presence_sim > NEED_CONFIG["threshold_presence"]:
        return NeedAnalysis(
            need_type=NeedType.PRESENCE,
            confidence=min(1.0, presence_sim),
            trigger="semantic",
            raw_scores=raw_scores,
            explanation=f"Semantic presence: {presence_sim:.2f}"
        )
    elif diff < 0 and perspective_sim > NEED_CONFIG["threshold_perspective"]:
        return NeedAnalysis(
            need_type=NeedType.PERSPECTIVE,
            confidence=min(1.0, perspective_sim),
            trigger="semantic",
            raw_scores=raw_scores,
            explanation=f"Semantic perspective: {perspective_sim:.2f}"
        )
    else:
        return NeedAnalysis(
            need_type=NeedType.NEUTRAL,
            confidence=0.3,
            trigger="semantic",
            raw_scores=raw_scores,
            explanation=f"Below thresholds: P={presence_sim:.2f}, R={perspective_sim:.2f}"
        )


def _combine_scores(
    keyword_result: NeedAnalysis,
    semantic_result: NeedAnalysis
) -> NeedAnalysis:
    """Combine keyword and semantic results with weighted average."""

    kw = NEED_CONFIG["keyword_weight"]
    sw = NEED_CONFIG["semantic_weight"]

    # If both agree, boost confidence
    if keyword_result.need_type == semantic_result.need_type:
        combined_confidence = min(1.0,
            kw * keyword_result.confidence + sw * semantic_result.confidence + 0.1)
        return NeedAnalysis(
            need_type=keyword_result.need_type,
            confidence=combined_confidence,
            trigger="hybrid_agreement",
            raw_scores={
                "keyword": keyword_result.raw_scores,
                "semantic": semantic_result.raw_scores
            },
            explanation=f"Keyword + semantic agree: {keyword_result.need_type.value}"
        )

    # If they disagree, use higher confidence result (semantic usually wins)
    if semantic_result.confidence > keyword_result.confidence:
        return NeedAnalysis(
            need_type=semantic_result.need_type,
            confidence=semantic_result.confidence * 0.8,  # Reduce due to disagreement
            trigger="hybrid_semantic_wins",
            raw_scores={
                "keyword": keyword_result.raw_scores,
                "semantic": semantic_result.raw_scores
            },
            explanation=f"Semantic override: {semantic_result.explanation}"
        )
    else:
        return NeedAnalysis(
            need_type=keyword_result.need_type,
            confidence=keyword_result.confidence * 0.8,
            trigger="hybrid_keyword_wins",
            raw_scores={
                "keyword": keyword_result.raw_scores,
                "semantic": semantic_result.raw_scores
            },
            explanation=f"Keyword override: {keyword_result.explanation}"
        )


def detect_need_type(message: str, model_manager=None) -> NeedAnalysis:
    """
    Main entry point: hybrid need-type detection.

    Args:
        message: User message to analyze
        model_manager: Optional model manager for embedder access

    Returns:
        NeedAnalysis with detected need type and metadata
    """
    # Stage 1: Keyword detection (fast path)
    keyword_result = _keyword_need_detection(message)

    # Fast path: high-confidence keyword match
    if keyword_result.confidence >= NEED_CONFIG["high_confidence_threshold"]:
        logger.debug(f"[NeedDetector] Fast path: {keyword_result.need_type.value}")
        return keyword_result

    # Stage 2: Semantic detection
    semantic_result = _semantic_need_detection(message, model_manager)

    # Combine if semantic available
    if semantic_result.confidence > 0:
        combined = _combine_scores(keyword_result, semantic_result)
        logger.debug(f"[NeedDetector] Hybrid: {combined.need_type.value} ({combined.trigger})")
        return combined

    # Fallback to keyword-only
    return keyword_result


def format_need_log(analysis: NeedAnalysis, message: str) -> str:
    """Format need analysis for backend logging."""
    msg_preview = message[:50] + "..." if len(message) > 50 else message
    msg_preview = msg_preview.replace("\n", " ")

    return (
        f"NEED: {analysis.need_type.value} "
        f"(confidence: {analysis.confidence:.2f}, trigger: {analysis.trigger}) "
        f"| Message: \"{msg_preview}\""
    )
