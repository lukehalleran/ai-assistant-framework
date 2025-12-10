"""
utils/tone_detector.py

Crisis vs. casual tone detection system with composite harm scoring.

Provides 3-stage detection (harm scoring → semantic → LLM fallback) to determine
appropriate response tone based on user message content. Distinguishes genuine crisis/distress
from casual conversation, world event observations, and routine updates.

Detection Pipeline:
1. Observational check - Filter world events vs personal crisis
2. Harm scoring - Composite keyword system (NEW 2025-12-09):
   - Scans entire message for ALL crisis indicators (250+ keywords)
   - Accumulates weighted points: HIGH (10pts), MEDIUM (5pts), CONCERN (2pts)
   - Applies pattern multipliers for dangerous combinations (1.2x-1.4x)
   - Routes: ≥20 HIGH, ≥10 MEDIUM, ≥4 CONCERN
3. Semantic similarity - Embedding comparison to crisis exemplars (fallback)
4. LLM fallback - For borderline cases near thresholds

Crisis levels:
- HIGH: Suicidal ideation, severe mental health crisis, immediate danger
- MEDIUM: Panic attacks, breakdown, severe emotional distress
- CONCERN: Significant anxiety, worry, stress (light support)
- CONVERSATIONAL: Default casual/friend mode (most interactions)

Key improvements (2025-12-09):
- Replaced "first keyword wins" with composite harm scoring
- Catches messages with multiple distress signals
- Pattern multipliers escalate severity for dangerous combinations
  (e.g., self-harm + crying, abuse + distress, hopelessness + suicidal)
- Comprehensive keyword coverage (50+ HIGH, 80+ MEDIUM, 100+ CONCERN)
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
# Each keyword contributes 10 points to harm score
HIGH_CRISIS_KEYWORDS = {
    # Direct suicidal ideation
    "want to die", "end it all", "kill myself", "no point living",
    "can't go on", "hurt myself", "suicide", "suicidal",
    "end my life", "better off dead", "want to disappear forever",
    "don't want to be here", "don't want to exist", "want to disappear",
    "no reason to live", "no point in living",
    "nothing to live for", "life isn't worth", "not worth living",
    "ready to die", "wish i was dead", "wish i were dead",
    "everyone would be better", "better without me",

    # Self-harm imagery and expressions
    "peel off my skin", "peel off all my skin", "split off my limbs",
    "cut myself", "cutting myself", "harm myself", "harming myself",
    "hurt myself", "hurting myself", "burn myself",
    "want to cut", "need to cut", "going to cut",
    "blade", "razors", "cutting again",

    # Self-harm ideation with body
    "hate my body", "destroy myself", "rip myself apart",
    "tear myself apart", "mutilate", "disfigure myself",

    # Severe emotional crisis with crying
    "sobbing", "crying uncontrollably", "can't stop crying",
    "cried so much", "crying so hard", "crying all night",
    "can't stop sobbing", "sobbing uncontrollably",
    "crying myself to sleep", "cry myself to sleep",

    # Hopelessness and giving up
    "no hope left", "lost all hope", "given up completely",
    "nothing matters anymore", "don't care anymore",
    "what's the point", "why bother living", "can't do this anymore",
    "too much to bear", "unbearable", "can't take it",

    # Crisis states
    "in crisis", "mental health crisis", "breaking point",
    "edge of breaking", "about to break", "going to break",
}

# Medium severity keywords
# Each keyword contributes 5 points to harm score
MEDIUM_CRISIS_KEYWORDS = {
    # Acute mental distress
    "panic attack", "can't breathe", "spiraling", "losing control",
    "breaking down", "lose it", "can't handle this",
    "having a breakdown", "nervous breakdown", "complete breakdown",
    "mental breakdown", "falling apart", "losing my mind",
    "going insane", "going crazy", "can't cope",
    "drowning", "suffocating", "choking on",

    # Dissociation and detachment
    "dissociating", "dissociated", "out of my body",
    "not real", "nothing feels real", "floating away",
    "watching myself", "not in my body",

    # Flashbacks and trauma responses
    "flashback", "flashbacks", "triggered", "having a trigger",
    "reliving it", "back there again", "ptsd episode",
    "trauma response", "in a trauma loop",

    # Abuse and trauma
    "abusive", "the abusive", "abuser", "gaslighting", "gaslighted",
    "manipulative", "manipulated me", "toxic relationship",
    "medical abuse", "forced me off", "denied my illness",
    "denied I was", "made me feel", "blamed me for",
    "emotionally abusive", "verbally abusive", "physically abusive",
    "narcissist", "narcissistic abuse", "controlling",

    # Substance abuse crisis
    "relapsed", "using again", "drinking again",
    "can't stay sober", "want to drink", "need a drink",
    "want to use", "need to use", "craving badly",

    # Severe anxiety states
    "heart racing", "chest tight", "hyperventilating",
    "shaking uncontrollably", "trembling", "can't calm down",
    "panic mode", "full panic", "anxiety attack",

    # Depressive episodes
    "depressive episode", "major depression", "in a dark place",
    "deep depression", "severely depressed", "can't get out of bed",
    "no energy", "complete exhaustion", "drained completely",

    # Sleep crisis
    "haven't slept in days", "can't sleep at all",
    "insomnia", "sleep deprived", "no sleep",

    # Relationship crisis
    "left me", "abandoned me", "walked out",
    "breakup", "divorce", "ended it",
}

# Concern-level keywords (distress but not crisis)
# Each keyword contributes 2 points to harm score
# Note: "overwhelmed" is handled separately due to context ambiguity
CONCERN_KEYWORDS = {
    # Anxiety and worry
    "really anxious", "freaking out", "scared", "terrified",
    "worried sick", "can't sleep", "anxious about",
    "nervous about", "stressed out", "stressing",
    "so anxious", "very anxious", "super anxious",
    "worried about", "worrying about", "worry about",

    # Emotional state expressions
    "lonely", "i am lonely", "i'm lonely", "i feel lonely",
    "vulnerable", "i feel vulnerable", "feeling vulnerable",
    "i'm scared", "i feel scared", "feeling scared",
    "isolated", "i feel isolated", "feeling isolated",
    "empty", "i feel empty", "feeling empty",
    "hopeless", "i feel hopeless", "feeling hopeless",
    "lost", "i feel lost", "feeling lost",
    "alone", "i feel alone", "feeling alone",
    "abandoned", "i feel abandoned", "feeling abandoned",
    "worthless", "i feel worthless", "feeling worthless",
    "numb", "i feel numb", "feeling numb",
    "sad", "i feel sad", "feeling sad", "so sad",
    "depressed", "i feel depressed", "feeling depressed",
    "down", "feeling down", "really down",

    # Helplessness and struggle
    "helpless", "feel helpless", "feeling helpless",
    "stuck", "feel stuck", "feeling stuck",
    "trapped", "feel trapped", "feeling trapped",
    "powerless", "feel powerless",
    "struggling", "really struggling", "struggling with",

    # Physical anxiety symptoms
    "heart pounding", "sweating", "nauseous",
    "stomach in knots", "tense", "on edge",

    # Sleep issues
    "can't sleep", "trouble sleeping", "bad dreams",
    "nightmares", "woke up anxious", "restless",

    # Social anxiety
    "don't want to go", "can't face", "avoiding",
    "hiding", "withdrawing", "shutting down",

    # Self-doubt and negative self-talk
    "hate myself", "disgusted with myself",
    "disappointed in myself", "failing",
    "not good enough", "can't do anything right",

    # Grief and loss
    "grieving", "mourning", "miss them", "miss her", "miss him",
    "can't believe they're gone", "still hurts",

    # Financial stress
    "broke", "can't afford", "financial stress",
    "money problems", "debt", "bills",

    # Work/school stress
    "work stress", "job stress", "school stress",
    "deadline", "pressure", "performance anxiety",
    "burnout", "burned out", "exhausted",

    # Health anxiety
    "health anxiety", "worried about my health",
    "scared of being sick", "medical anxiety",
    "afraid of dying", "death anxiety",
}

# Event distress keywords (strong reactions to upsetting world events)
# These indicate emotional distress about external events that deserve validation/support
# Triggers CONCERN mode (2-3 paragraphs) rather than CONVERSATIONAL (3 sentences)
EVENT_DISTRESS_KEYWORDS = {
    # Strong expletive reactions
    "what the fuck", "what the actual fuck", "wtf is", "wtf this",
    "fuck this", "this is fucked", "so fucked", "absolutely fucked",
    "completely fucked", "totally fucked",

    # Emotional reactions to events
    "can't believe this", "can't believe they", "can't believe we",
    "horrified by", "horrifying", "appalled", "disgusted by",
    "sickened by", "outraged by", "enraged by",

    # Despair/helplessness about world
    "this is why i stopped", "this is why i don't", "this is why i'm",
    "giving up on", "lost faith in", "no hope for",
    "heading for collapse", "heading for disaster", "moral reckoning",

    # Institutional betrayals
    "betrayal of", "normalizing", "rebranding", "whitewashing",
    "erasure of", "rewriting history", "sanitizing",

    # Expressions of despair about society
    "this country", "this administration", "this government",
    "where we're headed", "where this is going",
    "can't live in a country", "leaving this country", "leaving the country",
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


def _calculate_harm_score(message: str) -> Tuple[float, List[str], Dict[str, int]]:
    """
    Calculate composite harm score by scanning entire message for crisis indicators.

    Scoring system:
    - HIGH keywords: 10 points each
    - MEDIUM keywords: 5 points each
    - CONCERN keywords: 2 points each
    - EVENT_DISTRESS: 2 points each

    Pattern multipliers:
    - Multiple HIGH indicators (2+): 1.2x
    - Abuse + distress combination: 1.2x
    - Self-harm imagery + crying: 1.3x

    Returns:
        (total_score, matched_keywords, category_counts)
    """
    message_lower = message.lower()
    score = 0.0
    matched = []
    category_counts = {"high": 0, "medium": 0, "concern": 0, "event": 0}

    # Scan for HIGH keywords (10 points each)
    for keyword in HIGH_CRISIS_KEYWORDS:
        if keyword in message_lower:
            score += 10
            matched.append(f"HIGH: {keyword}")
            category_counts["high"] += 1
            logger.debug(f"[HarmScore] HIGH keyword: '{keyword}' (+10)")

    # Scan for MEDIUM keywords (5 points each)
    for keyword in MEDIUM_CRISIS_KEYWORDS:
        if keyword in message_lower:
            score += 5
            matched.append(f"MEDIUM: {keyword}")
            category_counts["medium"] += 1
            logger.debug(f"[HarmScore] MEDIUM keyword: '{keyword}' (+5)")

    # Scan for CONCERN keywords (2 points each)
    for keyword in CONCERN_KEYWORDS:
        if keyword in message_lower:
            score += 2
            matched.append(f"CONCERN: {keyword}")
            category_counts["concern"] += 1
            logger.debug(f"[HarmScore] CONCERN keyword: '{keyword}' (+2)")

    # Scan for EVENT_DISTRESS (2 points each)
    for keyword in EVENT_DISTRESS_KEYWORDS:
        if keyword in message_lower:
            score += 2
            matched.append(f"EVENT: {keyword}")
            category_counts["event"] += 1
            logger.debug(f"[HarmScore] EVENT_DISTRESS: '{keyword}' (+2)")

    # Handle "overwhelmed" specially
    if "overwhelmed" in message_lower:
        positive_markers = ["gift", "birthday", "excited", "happy", "amazing", "wonderful", "options", "choices"]
        if not any(marker in message_lower for marker in positive_markers):
            score += 2
            matched.append(f"CONCERN: overwhelmed")
            category_counts["concern"] += 1
            logger.debug(f"[HarmScore] CONCERN keyword: 'overwhelmed' (+2)")

    base_score = score
    multiplier = 1.0

    # Apply pattern multipliers for dangerous combinations
    # Multiple HIGH indicators (compounding crisis)
    if category_counts["high"] >= 2:
        multiplier *= 1.2
        logger.info(f"[HarmScore] Multiple HIGH indicators detected (x1.2)")

    # Abuse + distress combination
    abuse_keywords = ["abusive", "abuser", "gaslighting", "toxic relationship", "medical abuse",
                      "narcissist", "manipulative", "controlling"]
    distress_keywords = ["sobbing", "crying", "can't stop crying", "breaking down",
                        "falling apart", "losing control"]
    has_abuse = any(kw in message_lower for kw in abuse_keywords)
    has_distress = any(kw in message_lower for kw in distress_keywords)
    if has_abuse and has_distress:
        multiplier *= 1.2
        logger.info(f"[HarmScore] Abuse + distress combination detected (x1.2)")

    # Self-harm imagery + crying (severe distress)
    self_harm_keywords = ["peel off", "split off", "cut myself", "hurt myself", "harm myself",
                          "blade", "cutting", "burn myself"]
    crying_keywords = ["sobbing", "crying", "cried so much", "can't stop crying"]
    has_self_harm = any(kw in message_lower for kw in self_harm_keywords)
    has_crying = any(kw in message_lower for kw in crying_keywords)
    if has_self_harm and has_crying:
        multiplier *= 1.3
        logger.info(f"[HarmScore] Self-harm + crying pattern detected (x1.3)")

    # Substance relapse + crisis (dual crisis)
    substance_keywords = ["relapsed", "using again", "drinking again", "can't stay sober"]
    crisis_keywords = ["can't go on", "want to die", "breaking down", "in crisis"]
    has_substance = any(kw in message_lower for kw in substance_keywords)
    has_crisis = any(kw in message_lower for kw in crisis_keywords)
    if has_substance and has_crisis:
        multiplier *= 1.3
        logger.info(f"[HarmScore] Substance relapse + crisis detected (x1.3)")

    # Sleep deprivation + mental distress (compounding vulnerability)
    sleep_keywords = ["haven't slept in days", "can't sleep at all", "no sleep", "sleep deprived"]
    mental_keywords = ["losing my mind", "going crazy", "can't cope", "breaking point"]
    has_sleep = any(kw in message_lower for kw in sleep_keywords)
    has_mental = any(kw in message_lower for kw in mental_keywords)
    if has_sleep and has_mental:
        multiplier *= 1.2
        logger.info(f"[HarmScore] Sleep deprivation + mental distress detected (x1.2)")

    # Dissociation + trauma (severe trauma response)
    dissoc_keywords = ["dissociating", "out of my body", "not real", "floating away"]
    trauma_keywords = ["flashback", "triggered", "reliving it", "ptsd", "trauma"]
    has_dissoc = any(kw in message_lower for kw in dissoc_keywords)
    has_trauma = any(kw in message_lower for kw in trauma_keywords)
    if has_dissoc and has_trauma:
        multiplier *= 1.3
        logger.info(f"[HarmScore] Dissociation + trauma response detected (x1.3)")

    # Hopelessness + suicidal ideation (extreme danger)
    hopeless_keywords = ["no hope", "given up", "nothing matters", "no point"]
    suicidal_keywords = ["want to die", "kill myself", "end it all", "better off dead"]
    has_hopeless = any(kw in message_lower for kw in hopeless_keywords)
    has_suicidal = any(kw in message_lower for kw in suicidal_keywords)
    if has_hopeless and has_suicidal:
        multiplier *= 1.4
        logger.info(f"[HarmScore] Hopelessness + suicidal ideation detected (x1.4)")

    final_score = base_score * multiplier

    if multiplier > 1.0:
        logger.info(f"[HarmScore] Base score: {base_score:.1f}, Multiplier: {multiplier:.1f}x, Final: {final_score:.1f}")
    else:
        logger.debug(f"[HarmScore] Final score: {final_score:.1f}")

    return final_score, matched, category_counts


def _check_keyword_crisis(message: str) -> Optional[Tuple[CrisisLevel, str]]:
    """
    Harm score-based crisis detection.

    Routes based on composite harm score:
    - Score >= 20: HIGH crisis (multiple HIGH keywords or severe combinations)
    - Score >= 10: MEDIUM crisis (multiple MEDIUM keywords or serious distress)
    - Score >= 4: CONCERN (mild distress, several concern indicators)
    - Score < 4: None (use semantic detection)

    Returns (CrisisLevel, trigger_phrase) if score-based routing applies, else None.
    """
    score, matched, category_counts = _calculate_harm_score(message)

    # No keywords found
    if score == 0:
        return None

    # Route based on harm score thresholds
    if score >= 20:
        logger.info(f"[ToneDetector] HIGH crisis via harm score: {score:.1f} (matched: {len(matched)})")
        trigger = f"harm_score: {score:.1f} ({category_counts['high']}H, {category_counts['medium']}M, {category_counts['concern']}C)"
        return (CrisisLevel.HIGH, trigger)

    elif score >= 10:
        logger.info(f"[ToneDetector] MEDIUM crisis via harm score: {score:.1f} (matched: {len(matched)})")
        trigger = f"harm_score: {score:.1f} ({category_counts['high']}H, {category_counts['medium']}M, {category_counts['concern']}C)"
        return (CrisisLevel.MEDIUM, trigger)

    elif score >= 4:
        logger.debug(f"[ToneDetector] CONCERN via harm score: {score:.1f} (matched: {len(matched)})")
        trigger = f"harm_score: {score:.1f} ({category_counts['high']}H, {category_counts['medium']}M, {category_counts['concern']}C)"
        return (CrisisLevel.CONCERN, trigger)

    # Score between 1-3: low signal, defer to semantic detection
    else:
        logger.debug(f"[ToneDetector] Low harm score ({score:.1f}), deferring to semantic detection")
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
