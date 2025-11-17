"""
utils/tone_detector.py

Crisis vs. casual tone detection system.

Provides hybrid detection (keyword + semantic) to determine appropriate response tone
based on user message content. Distinguishes genuine crisis/distress from casual conversation,
world event observations, and routine updates.

Crisis levels:
- HIGH: Suicidal ideation, severe mental health crisis, immediate danger
- MEDIUM: Panic attacks, breakdown, severe emotional distress
- CONCERN: Significant anxiety, worry, stress (light support)
- CONVERSATIONAL: Default casual/friend mode (most interactions)
"""

import os
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from utils.logging_utils import get_logger

logger = get_logger("tone_detector")


# ===== Configuration =====

TONE_CONFIG = {
    # Semantic similarity thresholds for crisis detection
    # Tuned for all-MiniLM-L6-v2 model (lighter model has lower similarity scores)
    # Values based on empirical testing: HIGH ~0.40-0.60, MEDIUM ~0.45-0.55, mild phrases ~0.35-0.50
    "threshold_high": float(os.getenv("TONE_THRESHOLD_HIGH", "0.58")),
    "threshold_medium": float(os.getenv("TONE_THRESHOLD_MEDIUM", "0.50")),
    "threshold_concern": float(os.getenv("TONE_THRESHOLD_CONCERN", "0.43")),

    # Context window for conversation history
    "context_window": int(os.getenv("TONE_CONTEXT_WINDOW", "3")),

    # Escalation boost when prior context shows distress
    "escalation_boost": float(os.getenv("TONE_ESCALATION_BOOST", "1.2")),
}


class CrisisLevel(Enum):
    """Crisis severity levels mapped to response modes."""
    CONVERSATIONAL = "conversational"  # Default friend mode (85% of conversations)
    CONCERN = "light_support"          # Brief validation, acknowledge (10%)
    MEDIUM = "elevated_support"        # Supportive but measured (4%)
    HIGH = "crisis_support"            # Full therapeutic mode (1%)


@dataclass
class ToneAnalysis:
    """Result of tone detection analysis."""
    level: CrisisLevel
    confidence: float
    trigger: str  # "keyword_match", "semantic", "default"
    raw_scores: Dict[str, float]  # Similarity scores for each crisis level
    explanation: str  # Brief reason for the classification


# ===== Crisis Keywords =====

# Explicit high-severity crisis indicators (immediate keyword match → HIGH)
HIGH_CRISIS_KEYWORDS = {
    "want to die", "end it all", "kill myself", "no point living",
    "can't go on", "hurt myself", "suicide", "suicidal",
    "end my life", "better off dead", "want to disappear forever",
    "don't want to be here", "don't want to exist", "want to disappear",
    "better off dead", "no reason to live", "no point in living",
}

# Medium severity keywords
MEDIUM_CRISIS_KEYWORDS = {
    "panic attack", "can't breathe", "spiraling", "losing control",
    "breaking down", "lose it", "can't handle this",
    "having a breakdown", "nervous breakdown", "complete breakdown",
    "mental breakdown", "falling apart",
}

# Concern-level keywords (distress but not crisis)
# Note: "overwhelmed" is handled separately due to context ambiguity
CONCERN_KEYWORDS = {
    "really anxious", "freaking out", "scared", "terrified",
    "worried sick", "can't sleep", "helpless",
}

# World event/observational phrases (should NOT trigger crisis mode)
# These indicate the user is discussing external events, not personal distress
OBSERVATIONAL_MARKERS = {
    "people are", "people can't", "people suffering", "people are suffering", "millions", "thousands",
    "they are", "they can't", "the government", "society", "the world",
    "according to", "reports say", "news", "article",
    "study shows", "data shows", "statistics", "the reports",
    "civilians", "residents", "citizens",
    "cracking down", "protests", "protesters", "authorities", "officials",
    "struggling with", "policies", "regulations", "legislation",
}


# ===== Crisis Exemplars (for semantic detection) =====

# Pre-defined examples for each crisis level
# These will be embedded once and used for similarity comparison
CRISIS_EXEMPLARS = {
    "high": [
        "I want to hurt myself",
        "I can't go on anymore",
        "There's no point in continuing",
        "I'm thinking about ending it",
        "I want to die",
        "I don't want to be alive anymore",
        "I'm planning to kill myself",
        "Everything would be better if I was gone",
        "Everything would be better if I just wasn't here",
        "I can't take this pain anymore",
        "I'm going to end my life",
        "I want to disappear and never come back",
        "There's no way out for me",
        "I'm better off dead",
        "I have a plan to hurt myself",
        "I can't survive this",
        "I'm losing the will to live",
        "Nothing matters anymore, I give up",
        "Everything feels pointless and empty",
        "Everything feels pointless and I don't see a way forward",
        "I want to stop existing",
        "I'm ready to end it all",
        "Life isn't worth living",
    ],
    "medium": [
        "I'm having a panic attack",
        "I feel like I'm losing control",
        "Everything is falling apart",
        "I'm spiraling and can't stop",
        "I can't breathe, I'm freaking out",
        "I'm having a complete breakdown",
        "I feel like I'm going crazy",
        "I'm breaking down right now",
        "I can't handle this anymore",
        "I'm losing my mind",
        "I'm in a really dark place",
        "I feel like I'm drowning",
        "I'm completely falling apart",
        "I can't cope with this",
        "I'm having a mental breakdown",
        "I feel like everything is collapsing",
        "I'm paralyzed by fear",
        "I'm shaking and can't calm down",
        "I feel like I'm suffocating",
        "I'm on the edge of losing it",
    ],
    "concern": [
        "I'm really anxious about this",
        "I can't stop worrying",
        "This is overwhelming me",
        "I'm scared about what's happening",
        "I'm freaking out a little",
        "I'm really worried and stressed",
        "I can't sleep because I'm so anxious",
        "I'm feeling really overwhelmed",
        "I'm pretty worried about this situation",
        "I'm stressed out and need help",
        "I'm having trouble coping",
        "I'm feeling really uneasy",
        "I'm worried sick about this",
        "I'm nervous and don't know what to do",
        "I'm struggling with this",
        "I'm feeling really tense",
        "I'm anxious and can't relax",
        "I'm feeling helpless about this",
        "I'm really upset and don't know what to do",
        "I'm scared this won't work out",
    ],
    "conversational": [
        # Mild fatigue/tiredness (should NOT trigger crisis)
        "I'm a bit tired today",
        "I'm feeling a little tired",
        "I'm sleepy",
        "I need some rest",
        "I'm exhausted from work",
        "I could use a nap",
        "I'm worn out",
        # General conversation
        "How are you doing?",
        "What's up?",
        "I'm doing okay",
        "Thanks for checking in",
        "I'm fine",
        "Just a regular day",
        "Nothing much",
        "I'm alright",
        "Pretty good",
        "I'm okay",
    ],
}


# ===== Global Embedder Cache =====

_embedder_cache: Optional[object] = None
_exemplar_embeddings_cache: Optional[Dict[str, np.ndarray]] = None


def _get_embedder(model_manager=None):
    """Get or create the sentence embedder."""
    global _embedder_cache

    if _embedder_cache is not None:
        return _embedder_cache

    # Try to get embedder from model_manager if provided
    if model_manager and hasattr(model_manager, "get_embedder"):
        try:
            _embedder_cache = model_manager.get_embedder()
            logger.info("[ToneDetector] Using embedder from model_manager")
            return _embedder_cache
        except Exception as e:
            logger.warning(f"[ToneDetector] Failed to get embedder from model_manager: {e}")

    # Fallback: use cached embedder to avoid re-loading
    try:
        from models.model_manager import ModelManager
        _embedder_cache = ModelManager._get_cached_embedder()
        logger.info("[ToneDetector] Using cached embedder from ModelManager")
        return _embedder_cache
    except Exception as e:
        logger.error(f"[ToneDetector] Failed to get cached embedder: {e}")
        return None


def _get_exemplar_embeddings(model_manager=None) -> Dict[str, np.ndarray]:
    """Get or compute cached exemplar embeddings."""
    global _exemplar_embeddings_cache

    if _exemplar_embeddings_cache is not None:
        return _exemplar_embeddings_cache

    embedder = _get_embedder(model_manager)
    if embedder is None:
        logger.error("[ToneDetector] Cannot compute exemplar embeddings without embedder")
        return {}

    logger.info("[ToneDetector] Computing exemplar embeddings (one-time setup)...")

    _exemplar_embeddings_cache = {}

    for level, examples in CRISIS_EXEMPLARS.items():
        try:
            # Encode all examples for this level
            embeddings = embedder.encode(examples, convert_to_numpy=True)
            # Compute mean embedding as the prototype
            mean_embedding = np.mean(embeddings, axis=0)
            _exemplar_embeddings_cache[level] = mean_embedding
            logger.debug(f"[ToneDetector] Computed {level} exemplar from {len(examples)} examples")
        except Exception as e:
            logger.error(f"[ToneDetector] Failed to compute {level} exemplar: {e}")
            # Use zero vector as fallback
            _exemplar_embeddings_cache[level] = np.zeros(384)  # MiniLM embedding size

    logger.info("[ToneDetector] Exemplar embeddings ready")
    return _exemplar_embeddings_cache


# ===== Detection Functions =====

def _check_observational_language(message: str) -> bool:
    """
    Check if message is discussing world events/other people rather than personal crisis.

    Returns True if message appears to be about external events (not personal distress).
    """
    message_lower = message.lower()

    # Count observational markers
    marker_count = sum(1 for marker in OBSERVATIONAL_MARKERS if marker in message_lower)

    # If multiple observational markers, likely discussing external events
    if marker_count >= 2:
        return True

    # Check for third-person pronouns indicating discussion of others
    third_person = ["they", "them", "their", "people", "everyone", "someone"]
    first_person = ["i ", "i'm", "my ", "me ", "myself"]

    third_count = sum(1 for p in third_person if f" {p} " in f" {message_lower} ")
    first_count = sum(1 for p in first_person if p in message_lower)

    # If mostly third-person with observational language, not personal crisis
    if third_count >= 2 and first_count == 0:
        return True

    return False


def _check_keyword_crisis(message: str) -> Optional[Tuple[CrisisLevel, str]]:
    """
    Fast keyword-based crisis detection.

    Returns (CrisisLevel, trigger_phrase) if explicit keywords found, else None.
    """
    message_lower = message.lower()

    # Check HIGH crisis keywords first (most urgent)
    for keyword in HIGH_CRISIS_KEYWORDS:
        if keyword in message_lower:
            logger.info(f"[ToneDetector] HIGH crisis keyword detected: '{keyword}'")
            return (CrisisLevel.HIGH, f"keyword: {keyword}")

    # Check MEDIUM crisis keywords
    for keyword in MEDIUM_CRISIS_KEYWORDS:
        if keyword in message_lower:
            logger.info(f"[ToneDetector] MEDIUM crisis keyword detected: '{keyword}'")
            return (CrisisLevel.MEDIUM, f"keyword: {keyword}")

    # Check CONCERN keywords (with context awareness for ambiguous ones)
    for keyword in CONCERN_KEYWORDS:
        if keyword in message_lower:
            logger.debug(f"[ToneDetector] CONCERN keyword detected: '{keyword}'")
            return (CrisisLevel.CONCERN, f"keyword: {keyword}")

    # Special handling for "overwhelmed" - check if it's in a positive/neutral context
    if "overwhelmed" in message_lower:
        # Positive indicators that suggest this is NOT distress
        positive_markers = ["gift", "birthday", "excited", "happy", "amazing", "wonderful", "options", "choices"]
        if any(marker in message_lower for marker in positive_markers):
            logger.debug(f"[ToneDetector] 'overwhelmed' detected but in positive context, skipping")
            return None
        # Otherwise treat as concern
        logger.debug(f"[ToneDetector] CONCERN keyword detected: 'overwhelmed'")
        return (CrisisLevel.CONCERN, f"keyword: overwhelmed")

    return None


def _semantic_crisis_detection(
    message: str,
    conversation_history: Optional[List[dict]] = None,
    model_manager=None
) -> Tuple[CrisisLevel, float, Dict[str, float]]:
    """
    Semantic similarity-based crisis detection.

    Args:
        message: User message to analyze
        conversation_history: Recent conversation turns (optional, for context)
        model_manager: Optional model manager (for embedder access)

    Returns:
        Tuple of (CrisisLevel, confidence_score, raw_scores_dict)
    """
    embedder = _get_embedder(model_manager)
    if embedder is None:
        logger.warning("[ToneDetector] No embedder available, cannot do semantic detection")
        return (CrisisLevel.CONVERSATIONAL, 0.0, {})

    exemplar_embeddings = _get_exemplar_embeddings(model_manager)
    if not exemplar_embeddings:
        logger.warning("[ToneDetector] No exemplar embeddings available")
        return (CrisisLevel.CONVERSATIONAL, 0.0, {})

    # Encode the message
    try:
        message_embedding = embedder.encode(message, convert_to_numpy=True)
    except Exception as e:
        logger.error(f"[ToneDetector] Failed to encode message: {e}")
        return (CrisisLevel.CONVERSATIONAL, 0.0, {})

    # Compute cosine similarity with each crisis level exemplar
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    similarity_scores = {}
    for level in ["high", "medium", "concern", "conversational"]:
        if level in exemplar_embeddings:
            sim = cosine_similarity(message_embedding, exemplar_embeddings[level])
            similarity_scores[level] = float(sim)
        else:
            similarity_scores[level] = 0.0

    logger.debug(f"[ToneDetector] Semantic scores: {similarity_scores}")

    # Check for recent distress in conversation history
    recent_distress = 0.0
    if conversation_history:
        try:
            recent_turns = conversation_history[-TONE_CONFIG["context_window"]:]
            for turn in recent_turns:
                # Check if prior turn was flagged as crisis/concern
                if turn.get("is_heavy_topic", False):
                    recent_distress = 0.5
                    break
        except Exception as e:
            logger.debug(f"[ToneDetector] Failed to check conversation history: {e}")

    # Apply escalation boost if recent distress detected (but not to conversational)
    if recent_distress > 0.5:
        boosted_scores = {k: (v * TONE_CONFIG["escalation_boost"] if k != "conversational" else v) for k, v in similarity_scores.items()}
        logger.debug(f"[ToneDetector] Applied escalation boost: {boosted_scores}")
        similarity_scores = boosted_scores

    # Determine crisis level based on thresholds AND conversational similarity
    # If conversational similarity is significantly higher, prefer conversational
    conversational_score = similarity_scores.get("conversational", 0.0)

    # Check each level in order of severity, but only return if it's clearly higher than conversational
    if similarity_scores["high"] > TONE_CONFIG["threshold_high"]:
        if similarity_scores["high"] > conversational_score + 0.05:
            return (CrisisLevel.HIGH, similarity_scores["high"], similarity_scores)

    if similarity_scores["medium"] > TONE_CONFIG["threshold_medium"]:
        if similarity_scores["medium"] > conversational_score + 0.05:
            return (CrisisLevel.MEDIUM, similarity_scores["medium"], similarity_scores)

    if similarity_scores["concern"] > TONE_CONFIG["threshold_concern"]:
        if similarity_scores["concern"] > conversational_score + 0.05:
            return (CrisisLevel.CONCERN, similarity_scores["concern"], similarity_scores)

    # Even if below threshold, use the highest non-conversational score if significantly above conversational
    # This catches borderline cases like "Everything feels pointless" (high=0.509, conversational=0.339)
    # BUT only if the score meets a minimum absolute threshold to avoid false positives
    max_crisis_level = None
    max_crisis_score = 0.0
    MIN_ABSOLUTE_SCORE = 0.40  # Minimum score to trigger any crisis level
    MIN_MARGIN = 0.08  # Minimum margin above conversational (lowered from 0.10 to catch more borderline cases)

    for level_name, score in [("high", similarity_scores["high"]),
                               ("medium", similarity_scores["medium"]),
                               ("concern", similarity_scores["concern"])]:
        if score > max_crisis_score and score > conversational_score + MIN_MARGIN and score >= MIN_ABSOLUTE_SCORE:
            max_crisis_score = score
            if level_name == "high":
                max_crisis_level = CrisisLevel.HIGH
            elif level_name == "medium":
                max_crisis_level = CrisisLevel.MEDIUM
            else:
                max_crisis_level = CrisisLevel.CONCERN

    if max_crisis_level:
        return (max_crisis_level, max_crisis_score, similarity_scores)

    # Default to conversational
    return (CrisisLevel.CONVERSATIONAL, conversational_score, similarity_scores)


async def _llm_crisis_fallback(message: str, model_manager=None) -> Optional[Tuple[CrisisLevel, float]]:
    """
    LLM-based fallback for edge cases where semantic similarity is uncertain.
    Uses a small, fast model to classify crisis level.

    Returns (CrisisLevel, confidence) or None if LLM unavailable.
    """
    if not model_manager:
        return None

    try:
        # Construct a simple classification prompt
        prompt = f"""Analyze this message and classify the crisis level. Respond with ONLY one word: HIGH, MEDIUM, CONCERN, or CONVERSATIONAL.

HIGH: Suicidal ideation, self-harm, severe crisis requiring immediate intervention
MEDIUM: Panic attack, breakdown, severe emotional distress
CONCERN: Anxiety, worry, stress needing light support
CONVERSATIONAL: Casual conversation, no crisis

Message: "{message}"

Classification:"""

        # Use async generation with small model preference
        response = await model_manager.generate_async(
            prompt,
            max_tokens=10,
            temperature=0.1
        )

        if not response:
            return None

        # Parse response
        response_clean = response.strip().upper()

        if "HIGH" in response_clean:
            return (CrisisLevel.HIGH, 0.8)
        elif "MEDIUM" in response_clean:
            return (CrisisLevel.MEDIUM, 0.75)
        elif "CONCERN" in response_clean:
            return (CrisisLevel.CONCERN, 0.7)
        else:
            return (CrisisLevel.CONVERSATIONAL, 0.6)

    except Exception as e:
        logger.debug(f"[ToneDetector] LLM fallback failed: {e}")
        return None


async def detect_crisis_level(
    message: str,
    conversation_history: Optional[List[dict]] = None,
    model_manager=None
) -> ToneAnalysis:
    """
    Hybrid crisis detection combining keyword, semantic, and LLM approaches.

    Args:
        message: User message to analyze
        conversation_history: Recent conversation turns (optional)
        model_manager: Optional model manager for embedder and LLM access

    Returns:
        ToneAnalysis with detected crisis level and metadata
    """
    # Stage 0: Check if discussing world events (not personal crisis)
    if _check_observational_language(message):
        logger.debug("[ToneDetector] Detected observational/world event language - defaulting to conversational")
        return ToneAnalysis(
            level=CrisisLevel.CONVERSATIONAL,
            confidence=1.0,
            trigger="observational_language",
            raw_scores={},
            explanation="Discussing external events, not personal distress"
        )

    # Stage 1: Fast keyword check (high confidence)
    keyword_result = _check_keyword_crisis(message)
    if keyword_result:
        level, trigger = keyword_result
        return ToneAnalysis(
            level=level,
            confidence=1.0,
            trigger=trigger,
            raw_scores={},
            explanation=f"Explicit crisis language detected: {trigger}"
        )

    # Stage 2: Semantic detection for nuanced cases
    level, confidence, raw_scores = _semantic_crisis_detection(
        message, conversation_history, model_manager
    )

    # Stage 3: LLM fallback for borderline cases
    # Use LLM when semantic scores are close to thresholds (within 0.10)
    use_llm_fallback = False
    if level != CrisisLevel.CONVERSATIONAL:
        # Check if score is borderline for its detected level
        threshold_map = {
            CrisisLevel.HIGH: TONE_CONFIG["threshold_high"],
            CrisisLevel.MEDIUM: TONE_CONFIG["threshold_medium"],
            CrisisLevel.CONCERN: TONE_CONFIG["threshold_concern"],
        }
        threshold = threshold_map.get(level, 0)
        if confidence - threshold < 0.10:  # Within 0.10 of threshold
            use_llm_fallback = True
            logger.debug(f"[ToneDetector] Borderline {level.value}: confidence={confidence:.2f}, threshold={threshold:.2f}")
    elif level == CrisisLevel.CONVERSATIONAL:
        # Use LLM if highest score is close to CONCERN threshold
        highest_score = max(raw_scores.values())
        concern_threshold = TONE_CONFIG["threshold_concern"]
        if highest_score > concern_threshold - 0.15:
            use_llm_fallback = True
            logger.debug(f"[ToneDetector] Borderline CONVERSATIONAL: highest={highest_score:.2f}, threshold-0.15={concern_threshold-0.15:.2f}")

    if use_llm_fallback and model_manager:
        logger.debug(f"[ToneDetector] Attempting LLM fallback for: {message[:50]}...")
        llm_result = await _llm_crisis_fallback(message, model_manager)
        if llm_result:
            llm_level, llm_confidence = llm_result
            logger.info(f"[ToneDetector] LLM fallback: {llm_level.value} (confidence: {llm_confidence:.2f})")
            return ToneAnalysis(
                level=llm_level,
                confidence=llm_confidence,
                trigger="llm_fallback",
                raw_scores=raw_scores,
                explanation=f"LLM classification: {llm_level.value}"
            )

    # Build explanation
    if level == CrisisLevel.CONVERSATIONAL:
        explanation = "No crisis indicators detected"
    else:
        explanation = f"Semantic similarity to {level.value}: {confidence:.2f}"

    return ToneAnalysis(
        level=level,
        confidence=confidence,
        trigger="semantic",
        raw_scores=raw_scores,
        explanation=explanation
    )


def format_tone_log(analysis: ToneAnalysis, message: str) -> str:
    """
    Format tone analysis for backend logging.

    Args:
        analysis: ToneAnalysis result
        message: Original user message (truncated for privacy)

    Returns:
        Formatted log string
    """
    # Truncate message for privacy/brevity
    msg_preview = message[:60] + "..." if len(message) > 60 else message
    msg_preview = msg_preview.replace("\n", " ")

    log_line = (
        f"TONE: {analysis.level.value} "
        f"(confidence: {analysis.confidence:.2f}, "
        f"trigger: {analysis.trigger}) "
        f"| Message: \"{msg_preview}\""
    )

    if analysis.raw_scores:
        scores_str = ", ".join(f"{k}={v:.2f}" for k, v in analysis.raw_scores.items())
        log_line += f" | Scores: {scores_str}"

    return log_line


# ===== Tone Mode Transitions =====

def should_log_tone_shift(
    previous_level: Optional[CrisisLevel],
    current_level: CrisisLevel
) -> bool:
    """
    Determine if a tone shift should be logged.

    Only log when transitioning between different levels (not same→same).
    """
    if previous_level is None:
        return False  # First message, no shift

    return previous_level != current_level


def format_tone_shift_log(
    previous_level: CrisisLevel,
    current_level: CrisisLevel,
    trigger: str
) -> str:
    """
    Format a tone shift event for logging.

    Args:
        previous_level: Previous crisis level
        current_level: New crisis level
        trigger: What triggered the shift

    Returns:
        Formatted shift log string
    """
    return (
        f"TONE_SHIFT: {previous_level.value} → {current_level.value} "
        f"({trigger})"
    )
