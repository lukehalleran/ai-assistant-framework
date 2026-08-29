"""
# utils/tone_detector.py

Module Contract
- Purpose: Crisis vs. casual tone detection with composite harm scoring. Determines
  appropriate response tone (CrisisLevel) based on user message content.
- Data classes:
  - CrisisLevel(Enum): CONVERSATIONAL, CONCERN, MEDIUM, HIGH
  - ToneAnalysis: level, confidence, trigger, raw_scores, explanation
- Key public functions:
  - detect_crisis_level(message, conversation_history, model_manager, previous_tone) -> ToneAnalysis  [async]
    Main entry point — runs full pipeline. `previous_tone` makes distress STICKY
    across short turns: the <8-word fast path only exits early for recognizably
    casual messages (greeting/ack/logistics), and during an established distress
    session a non-casual terse reply is floored at CONCERN rather than resetting
    to CONVERSATIONAL. This closed the regression where a spiral built from terse
    messages flatlined at CONVERSATIONAL, starving the escalation tracker.
    Positive-recovery exception (2026-08-14): a bounded first-person improvement
    report (_is_positive_state_report — improvement lexicon, negation/hedge veto)
    relaxes like an explicit casual marker AND downgrades a semantic CONCERN
    verdict to CONVERSATIONAL (MEDIUM/HIGH never overridden; "I'm fine"-style
    masking keeps the floor).
  - format_tone_log(analysis, message) -> str  [formatted log string]
  - should_log_tone_shift(prev_level, new_level) -> bool
  - format_tone_shift_log(prev_level, new_level, query) -> str
- Detection pipeline (4 stages):
  1. Observational check — _check_observational_language(): filters world events vs personal crisis
  2. Composite harm scoring — _calculate_harm_score(): 250+ weighted keywords (HIGH=10pts, MEDIUM=5pts,
     CONCERN=2pts), pattern multipliers (1.2x-1.4x), routes: >=20 HIGH, >=10 MEDIUM, >=4 CONCERN
  3. Semantic similarity — _semantic_crisis_detection(): embedding comparison to crisis exemplars
     using configurable thresholds (all-MiniLM-L6-v2). Prototypes = CODE seeds
     (CRISIS_EXEMPLARS) + per-user LEARNED exemplars (utils.adaptive_exemplars,
     2026-08-02): a classification confirmed by the keyword stage or the LLM
     arbiter teaches the store (_learn_tone_exemplar), so future paraphrases
     match semantically without hand-editing lists. The heuristic backstop and
     conversational verdicts NEVER teach (self-reinforcement / poisoning guard).
     Toggle: TONE_EXEMPLAR_LEARNING env (default on).
  4. LLM fallback — _llm_crisis_fallback(): for borderline cases near thresholds.
     Uses model_manager.generate_once (non-streaming; generate_async returns a
     stream object and crashed every borderline call until 2026-07-25) under
     llm_fallback_timeout_s, with disable_reasoning=True (2026-08-02: reasoning
     models burned the tiny max_tokens budget in the reasoning channel and
     returned empty content). An unrecognized arbiter reply is failure (None),
     never a CONVERSATIONAL verdict. The rubric maps sadness/unhappiness/shame/
     low self-worth to CONCERN and violent thoughts to MEDIUM. When the arbiter
     is unavailable/fails OR returns CONVERSATIONAL against distress-dominant
     semantics, a deterministic borderline backstop floors a distress-shaped
     borderline to CONCERN (top distress score > conversational + 0.08 AND >=
     backstop_min_score) instead of failing to CONVERSATIONAL; the arbiter can
     raise the level but its CONVERSATIONAL cannot override that margin.
- Config: TONE_CONFIG dict with threshold_high/medium/concern, context_window,
  escalation_boost, llm_fallback_timeout_s, backstop_min_score.
  All overridable via TONE_THRESHOLD_* / TONE_LLM_FALLBACK_TIMEOUT_S /
  TONE_BACKSTOP_MIN_SCORE env vars.
- Dependencies:
  - numpy (similarity computation)
  - models.model_manager (embedding + LLM for stages 3-4) [optional]
- Side effects:
  - Embedding computation (cached per session)
  - LLM API call for borderline cases only (stage 4)
"""

import asyncio
import os
import re
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

    # Upper bound on the borderline LLM-arbiter call (it sits in the hot
    # context-pipeline path; on timeout the deterministic borderline backstop
    # takes over)
    "llm_fallback_timeout_s": float(os.getenv("TONE_LLM_FALLBACK_TIMEOUT_S", "4.0")),

    # Absolute-score floor for the arbiter-unavailable borderline backstop.
    # Sits below MIN_ABSOLUTE_SCORE (0.40, full class assignment) so the
    # 2026-07-25 live miss (medium=0.390) is caught, but above situational
    # weak matches ("That movie made me sad", concern=0.34) so margin alone
    # can't promote them.
    "backstop_min_score": float(os.getenv("TONE_BACKSTOP_MIN_SCORE", "0.37")),
    # Grow per-user semantic exemplars from confirmed classifications
    # (keyword stage + LLM arbiter) via utils.adaptive_exemplars.
    "exemplar_learning": os.getenv("TONE_EXEMPLAR_LEARNING", "1") not in ("0", "false", "False"),
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
    "denied i was", "made me feel", "blamed me for",
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

    # Violent ideation toward others (2026-08-02: "I would hurt them terribly
    # if legal" scored harm 0.0 and semantic high=0.298 — a live miss).
    # Negation-scoped deliberately: bare "deserve to be alive" must NOT match,
    # or the user's own repair "I deserve to be alive" would floor to distress.
    "i would hurt them", "i would hurt him", "i would hurt her",
    "want to hurt them", "want to hurt him", "want to hurt her",
    "want to hurt someone", "want to hurt people", "could hurt someone",
    "not everyone deserves to be alive", "doesn't deserve to be alive",
    "don't deserve to be alive", "doesn't deserve to live",
    "don't deserve to live",
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
        # Overwhelm (bare "I'm overwhelmed" must stay MEDIUM — the 2026-08-02
        # exemplar additions diluted the prototype below its threshold)
        "I'm overwhelmed",
        "I'm completely overwhelmed",
        # Other-directed violent ideation (2026-08-02: "I would hurt them
        # terribly if legal" scored high=0.298 — below every threshold —
        # because no exemplar covered anger-at-others distress)
        "I want to hurt the people who wronged me",
        "I would hurt them badly if I could get away with it",
        "Some people don't deserve to be alive",
        "I'm so angry I could hurt someone right now",
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
        # Plain sadness / shame / low self-worth (2026-08-02: "I am so
        # unhappy" scored concern=0.349 — under the backstop floor — because
        # every concern exemplar was anxiety-flavored)
        "I am so unhappy",
        "I feel so sad and I don't know why",
        "I'm ashamed of myself",
        "I feel worthless lately",
        "I feel so lonely",
        "I hate myself and I wanna cry",
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
        # Media/situational sadness — reacting to content is not distress
        # (probe 2026-08-02: "this song always makes me cry" scored
        # medium=0.413, ABOVE every real live miss)
        "That movie made me sad",
        "That song always makes me cry a little",
        "The ending of that show was heartbreaking",
        "That show was so depressing to watch",
        # Strategy-game violence talk is play, not ideation (owner plays HOI4)
        "I crushed everyone in my strategy game last night",
        "I destroyed the enemy army in my game, total victory",
        # NOTE (2026-08-14): do NOT add recovery-report exemplars ("I feel
        # much better today") here without a probe run — these are CENTROID
        # prototypes and the margin rule is knife-edge: adding three such
        # lines moved conversational 0.404→0.428 and flipped the HIGH
        # paraphrase "Everything would be better if I just wasn't here" to
        # CONVERSATIONAL. Recovery reports are handled deterministically by
        # _is_positive_state_report instead (semantic-CONCERN override +
        # sticky-floor exemption).
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


def _get_learned_exemplars(level: str) -> list:
    """Learned per-user exemplars for a level (adaptive store; [] on failure)."""
    try:
        from utils.adaptive_exemplars import get_store
        return get_store().get_learned("tone", level)
    except Exception:
        return []


def _adaptive_store_version() -> int:
    try:
        from utils.adaptive_exemplars import get_store
        return get_store().version
    except Exception:
        return -1


# Per-text embedding cache: a store-version bump re-encodes only NEW texts
# (see adaptive_exemplars.encode_texts_cached). Never shared across modules.
_exemplar_text_emb_cache: Dict[str, np.ndarray] = {}


def _get_exemplar_embeddings(model_manager=None) -> Dict[str, np.ndarray]:
    """Get or compute cached exemplar embeddings.

    Prototypes are computed from the CODE seeds (CRISIS_EXEMPLARS) merged with
    the per-user LEARNED exemplars from utils.adaptive_exemplars — confirmed
    live classifications grow the semantic channel so future paraphrases match
    without hand-editing lists. The cache is keyed on the adaptive store's
    version and recomputes when new exemplars were learned.
    """
    global _exemplar_embeddings_cache

    _version = _adaptive_store_version()
    if (
        isinstance(_exemplar_embeddings_cache, tuple)
        and _exemplar_embeddings_cache[0] == _version
    ):
        return _exemplar_embeddings_cache[1]

    embedder = _get_embedder(model_manager)
    if embedder is None:
        logger.error("[ToneDetector] Cannot compute exemplar embeddings without embedder")
        return {}

    logger.info("[ToneDetector] Computing exemplar embeddings (one-time setup)...")

    prototypes: Dict[str, np.ndarray] = {}
    for level, examples in CRISIS_EXEMPLARS.items():
        try:
            merged = list(examples) + _get_learned_exemplars(level)
            from utils.adaptive_exemplars import encode_texts_cached
            embeddings = encode_texts_cached(
                embedder, merged, _exemplar_text_emb_cache
            )
            # Compute mean embedding as the prototype
            prototypes[level] = np.mean(embeddings, axis=0)
            logger.debug(
                f"[ToneDetector] Computed {level} exemplar from {len(examples)} seed "
                f"+ {len(merged) - len(examples)} learned examples"
            )
        except Exception as e:
            logger.error(f"[ToneDetector] Failed to compute {level} exemplar: {e}")
            # Use zero vector as fallback
            prototypes[level] = np.zeros(384)  # MiniLM embedding size

    _exemplar_embeddings_cache = (_version, prototypes)
    logger.info("[ToneDetector] Exemplar embeddings ready")
    return prototypes


def _learn_tone_exemplar(message: str, level_key: str, source: str,
                         model_manager=None) -> None:
    """Teach the adaptive store a CONFIRMED distress classification.

    Poisoning guards: only distress levels are learned (a missed detection
    labeled conversational must never TEACH the miss), and callers only invoke
    this from channels independent of the semantic prototypes themselves —
    the deterministic keyword stage and the LLM arbiter. The heuristic
    borderline backstop never teaches (it would self-reinforce).
    """
    if not TONE_CONFIG.get("exemplar_learning", True):
        return
    if level_key not in ("concern", "medium", "high"):
        return
    try:
        from utils.adaptive_exemplars import get_store
        get_store().record(
            "tone", level_key, message, source,
            embedder=_get_embedder(model_manager),
            seed_texts=CRISIS_EXEMPLARS.get(level_key, []),
        )
    except Exception as e:
        logger.debug(f"[ToneDetector] exemplar learning skipped: {e}")


# ===== Detection Functions =====

# Recognizably casual / acknowledgment / logistics short messages.
# The <8-word fast path exists ONLY to skip embedding for these (greetings,
# thanks, "ok", "what time is it"). It must NOT skip ambiguous or emotional
# short messages — that regression (commit 33246a2) let an entire distress
# session made of short turns flatline at CONVERSATIONAL.
_CASUAL_SHORT_MARKERS = {
    "hi", "hey", "hello", "yo", "sup", "heya", "hiya",
    "thanks", "thank you", "ty", "thx", "cheers", "much appreciated",
    "ok", "okay", "k", "kk", "cool", "nice", "great", "awesome", "perfect",
    "got it", "gotcha", "sure", "yep", "yeah", "yup", "yes", "no", "nope", "nah",
    "lol", "haha", "lmao", "np", "no problem", "sounds good", "will do", "on it",
    "brb", "gtg", "ttyl", "later", "bye", "goodnight", "gn", "morning", "good morning",
    "same", "true", "right", "makes sense", "fair", "word", "for sure", "indeed",
    "what time is it", "what's up", "whats up", "how are you", "how's it going",
    "hows it going", "good", "fine", "alright", "not much", "nothing much",
}

# First-person emotional hint words: if a very short message contains one of
# these, it is NOT treated as low-signal casual (it goes to semantic detection).
_EMOTION_HINT_WORDS = {
    "lonely", "scared", "sad", "hopeless", "anxious", "depressed", "empty",
    "numb", "alone", "worthless", "done", "hurts", "hurt", "crying", "cry",
    "panic", "help", "afraid", "terrified", "overwhelmed", "exhausted",
    "worthless", "hate", "tired", "lost", "stuck", "trapped", "broken",
}


def _normalize_short(message: str) -> str:
    """Lowercase + strip surrounding punctuation/whitespace for marker matching."""
    return message.strip().lower().strip(" .!?,;:~-—)(\"'")


def _is_explicit_casual(message: str) -> bool:
    """
    True only for an EXPLICIT casual/acknowledgment marker (greeting, "ok",
    "thanks", "what time is it"). Used to decide whether a short message is
    allowed to relax tone back to CONVERSATIONAL during a distress session.
    """
    return _normalize_short(message) in _CASUAL_SHORT_MARKERS


# Positive recovery report (2026-08-14): first-person improvement statements.
# During a distress session, "I feel significantly more normal I think" was
# sticky-floored to CONCERN — the reply then carried LIGHT SUPPORT ("let them
# vent, don't offer advice") against a message reporting recovery. A message
# that (a) semantics already scored CONVERSATIONAL, (b) is a first-person
# state report, and (c) names improvement WITHOUT negation/subjunctive hedges
# may relax the floor exactly like an explicit casual marker. Bare masking
# shapes ("I'm fine", "I'm ok") are deliberately NOT in the lexicon — they
# keep the floor, per the anti-amplification design.
_POSITIVE_STATE_OPENERS = re.compile(
    r"\b(?:i(?:'m|m| am| feel| felt| was feeling| have been feeling|ve been feeling)|feeling|today (?:i(?:'m|m)?|is|was|has been))\b",
    re.IGNORECASE,
)
_POSITIVE_STATE_WORDS = re.compile(
    r"\b(?:better|more normal|normal(?: again)?|improved|improving|great|really good|pretty good|much better|clearer|lighter|calmer|recovered|myself again)\b",
    re.IGNORECASE,
)
# Negation/subjunctive hedges anywhere in the message veto the positive read:
# "I don't feel better", "I wish I felt normal", "less normal", "not great".
_POSITIVE_STATE_VETO = re.compile(
    r"\b(?:not|no|never|n't|dont|cant|wont|isnt|aint|wish|hope|hoping|want|wanted|need|needed|trying|if only|less|worse|barely|hardly|except|but)\b",
    re.IGNORECASE,
)


def _is_positive_state_report(message: str) -> bool:
    """True for a bounded first-person improvement report ("I feel a lot more
    normal today"). Only consulted to RELAX the distress-sticky floor when the
    semantic tier already scored the message CONVERSATIONAL — it never
    downgrades a message that scored distress on its own."""
    text = (message or "").strip()
    if not text or len(text.split()) > 20:
        return False
    if _POSITIVE_STATE_VETO.search(text):
        return False
    return bool(
        _POSITIVE_STATE_OPENERS.search(text) and _POSITIVE_STATE_WORDS.search(text)
    )


def _is_casual_short_message(message: str) -> bool:
    """
    Broader casual check for the fast-path-skip decision (only consulted when
    the session is NOT already in distress). True for explicit markers OR very
    short (<=2 word) low-signal fragments with no emotional hint word
    (e.g. "the store", "2nd", "on monday"). Ambiguous emotional short messages
    ("i dont feel like an adult", "i guess im sad") return False so they still
    reach semantic detection.
    """
    norm = _normalize_short(message)
    if not norm:
        return True  # empty / punctuation-only
    if norm in _CASUAL_SHORT_MARKERS:
        return True
    words = norm.split()
    if len(words) <= 2 and not any(w in _EMOTION_HINT_WORDS for w in words):
        return True
    return False


def _recent_distress_from_history(conversation_history: Optional[List[dict]]) -> bool:
    """True if any recent in-session turn was flagged as heavy/crisis.

    The history path must obey the same gap boundary as previous_tone
    stickiness. Otherwise a stale heavy turn from yesterday can re-latch the
    floor after ContextPipeline correctly cleared _last_tone_level.
    """
    if not conversation_history:
        return False
    try:
        from datetime import datetime, timedelta
        from config.app_config import TONE_STICKINESS_MAX_GAP_MINUTES

        def _fresh(turn: dict) -> bool:
            ts = turn.get("timestamp") if isinstance(turn, dict) else None
            if ts is None:
                return True  # fail closed for legacy rows without timestamps
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except ValueError:
                    return True
            if not isinstance(ts, datetime):
                return True
            now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
            return (now - ts) <= timedelta(minutes=TONE_STICKINESS_MAX_GAP_MINUTES)

        # Order-agnostic (2026-08-22): callers pass corpus recents NEWEST-
        # first, so the old [-window:] slice examined the OLDEST rows of the
        # fetch (the digest-order bug class). The freshness bound above does
        # the real limiting; scan every row passed (callers fetch <=5).
        for turn in conversation_history:
            if isinstance(turn, dict) and turn.get("is_heavy_topic", False) and _fresh(turn):
                return True
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"[ToneDetector] history distress check failed: {e}")
    return False


def _session_in_distress(
    previous_tone: Optional[object],
    conversation_history: Optional[List[dict]],
) -> bool:
    """
    Whether the session is already carrying distress, so a new short message
    should maintain rather than reset tone. Accepts previous_tone in any
    encoding: a CrisisLevel, or a string like "CrisisLevel.CONCERN",
    "concern", "light_support", "MEDIUM", "elevated_support", "crisis_support".
    """
    if previous_tone is not None:
        # Prefer the enum's .value ("light_support") over its repr
        # ("CrisisLevel.CONCERN"), then strip any remaining class prefix.
        # The test must be EXACT membership: a substring match on "crisis"
        # latched session_distress True for every level incl. CONVERSATIONAL
        # ("crisis" ⊂ "crisislevel.…") from 2026-07-21 to 2026-08-27 — every
        # session floored to CONCERN from turn 2.
        pt = str(getattr(previous_tone, "value", previous_tone)).strip().lower()
        if "." in pt:
            pt = pt.rsplit(".", 1)[-1]
        distress_levels = {
            "high", "medium", "concern", "crisis", "elevated",
            "light_support", "elevated_support", "crisis_support",
        }
        if pt in distress_levels:
            return True
    return _recent_distress_from_history(conversation_history)


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
    model_manager=None,
    force_escalation: bool = False,
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

    # Check for recent distress in conversation history OR a forced-escalation
    # signal from the caller (sustained-distress session). NOTE: this previously
    # set recent_distress = 0.5 and gated on `> 0.5`, so the boost never fired.
    recent_distress = bool(force_escalation) or _recent_distress_from_history(conversation_history)

    # Apply escalation boost if recent distress detected (but not to conversational)
    if recent_distress:
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
        # Construct a simple classification prompt. The rubric must cover plain
        # sadness/shame/self-worth statements, not just anxiety — a literal
        # "anxiety, worry, stress" rubric made the arbiter label "I am so
        # unhappy" CONVERSATIONAL (2026-08-02).
        prompt = f"""Analyze this message and classify the crisis level. Respond with ONLY one word: HIGH, MEDIUM, CONCERN, or CONVERSATIONAL.

HIGH: Suicidal ideation, self-harm, severe crisis requiring immediate intervention
MEDIUM: Panic attack, breakdown, severe emotional distress, violent thoughts about self or others
CONCERN: Anxiety, worry, stress, sadness, unhappiness, shame, loneliness, low self-worth, or emotional pain needing support
CONVERSATIONAL: Casual conversation, no distress. Mild or momentary reactions to media/events ("that movie made me sad") are CONVERSATIONAL.

Message: "{message}"

Classification:"""

        # generate_once returns a complete string; generate_async returns a
        # stream object for API models (calling .strip() on it crashed every
        # borderline classification until 2026-07-25, silently defaulting the
        # exact ambiguous-distress cases this fallback exists for to
        # CONVERSATIONAL). Bounded so a hung call can't stall the pipeline.
        # disable_reasoning: on reasoning models (kimi-3) the reasoning channel
        # ate the tiny max_tokens budget and returned empty content, silently
        # dropping the arbiter verdict (2026-08-02).
        response = await asyncio.wait_for(
            model_manager.generate_once(
                prompt,
                system_prompt="You are a strict single-word classifier.",
                max_tokens=16,
                temperature=0.1,
                disable_reasoning=True,
            ),
            timeout=TONE_CONFIG.get("llm_fallback_timeout_s", 4.0),
        )

        if not response or not isinstance(response, str):
            return None

        # Parse response. An unrecognized reply is arbiter failure → None so
        # the deterministic backstop applies — NOT a CONVERSATIONAL verdict.
        response_clean = response.strip().upper()

        if "HIGH" in response_clean:
            return (CrisisLevel.HIGH, 0.8)
        elif "MEDIUM" in response_clean:
            return (CrisisLevel.MEDIUM, 0.75)
        elif "CONCERN" in response_clean:
            return (CrisisLevel.CONCERN, 0.7)
        elif "CONVERSATIONAL" in response_clean:
            return (CrisisLevel.CONVERSATIONAL, 0.6)
        else:
            logger.debug(
                f"[ToneDetector] LLM fallback returned unrecognized verdict: {response_clean[:60]!r}"
            )
            return None

    except Exception as e:
        logger.debug(f"[ToneDetector] LLM fallback failed: {e}")
        return None


async def detect_crisis_level(
    message: str,
    conversation_history: Optional[List[dict]] = None,
    model_manager=None,
    previous_tone: Optional[object] = None,
    allow_sticky_floor: bool = True,
) -> ToneAnalysis:
    """
    Hybrid crisis detection combining keyword, semantic, and LLM approaches.

    Args:
        message: User message to analyze
        conversation_history: Recent conversation turns (optional)
        model_manager: Optional model manager for embedder and LLM access
        previous_tone: The prior turn's detected tone (CrisisLevel or string).
            Makes distress sticky across short messages so a spiral built from
            terse turns does not flatline at CONVERSATIONAL.
        allow_sticky_floor: When False the distress-sticky floor stage is
            skipped entirely. The caller (ContextPipeline) sets this from the
            TONE_FLOOR_CHAIN_MAX budget — withholding previous_tone alone did
            NOT enforce the budget, because _session_in_distress falls through
            to the recent-heavy-history path and the floor re-fired anyway
            (2026-08-28: four consecutive floor turns on jokey messages).
            Organic signals (keyword/semantic/arbiter/backstop) are unaffected.

    Returns:
        ToneAnalysis with detected crisis level and metadata
    """
    session_distress = _session_in_distress(previous_tone, conversation_history)
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
        # Deterministic confirmation → teach the semantic channel, so future
        # PARAPHRASES of this user's distress phrasing match without keywords.
        _LEVEL_KEYS = {
            CrisisLevel.HIGH: "high",
            CrisisLevel.MEDIUM: "medium",
            CrisisLevel.CONCERN: "concern",
        }
        if level in _LEVEL_KEYS:
            _learn_tone_exemplar(message, _LEVEL_KEYS[level], "keyword", model_manager)
        return ToneAnalysis(
            level=level,
            confidence=1.0,
            trigger=trigger,
            raw_scores={},
            explanation=f"Explicit crisis language detected: {trigger}"
        )

    # Stage 1.5: Fast-path exit for RECOGNIZABLY CASUAL short messages only.
    # The latency optimization (skip embedding, ~200-500ms) is preserved for
    # greetings/acks/logistics ("hey", "thanks", "what time is it"), but it must
    # NOT swallow ambiguous or emotional short messages, and must NOT reset tone
    # mid-distress. A short message is fast-pathed to CONVERSATIONAL only when:
    #   (a) it carries no keyword harm signal, AND
    #   (b) the session is not already in distress, AND
    #   (c) it is recognizably casual (not an ambiguous emotional fragment).
    word_count = len(message.split())
    if word_count < 8:
        base_score, _, _ = _calculate_harm_score(message)
        if base_score == 0 and not session_distress and _is_casual_short_message(message):
            logger.debug(f"[ToneDetector] Casual short message ({word_count} words) — fast CONVERSATIONAL")
            return ToneAnalysis(
                level=CrisisLevel.CONVERSATIONAL,
                confidence=0.95,
                trigger="short_casual",
                raw_scores={},
                explanation="Recognizably casual short message"
            )
        logger.debug(
            f"[ToneDetector] Short message ({word_count} words) not fast-pathed "
            f"(harm={base_score:.0f}, session_distress={session_distress}) — running semantic"
        )

    # Stage 2: Semantic detection for nuanced cases.
    # force_escalation carries the sustained-distress signal so the escalation
    # boost applies even when the current terse message lacks its own signal.
    level, confidence, raw_scores = _semantic_crisis_detection(
        message, conversation_history, model_manager, force_escalation=session_distress
    )

    # Positive-recovery override (2026-08-14): first-person improvement
    # statements sit semantically close to the "I feel…" distress prototypes
    # ("I feel significantly more normal I think" scored concern=0.48 and got
    # a LIGHT SUPPORT reply against a message reporting recovery). A bounded,
    # negation-free positive state report downgrades a semantic CONCERN to
    # CONVERSATIONAL. MEDIUM/HIGH are never overridden — any real harm signal
    # outranks the phrasing.
    if level == CrisisLevel.CONCERN and _is_positive_state_report(message):
        logger.debug("[ToneDetector] Positive recovery report — semantic CONCERN → CONVERSATIONAL")
        return ToneAnalysis(
            level=CrisisLevel.CONVERSATIONAL,
            confidence=confidence,
            trigger="positive_state_report",
            raw_scores=raw_scores,
            explanation="First-person improvement report; concern-level semantic match overridden",
        )

    # Sticky floor: during an established distress session, do not let a
    # non-casual terse reply collapse tone all the way back to CONVERSATIONAL
    # (that reset is what starved the escalation tracker across this class of
    # session). Explicit casual markers ("ok", "lol") are still allowed to
    # relax, so genuine disengagement de-escalates normally.
    if (
        allow_sticky_floor
        and session_distress
        and level == CrisisLevel.CONVERSATIONAL
        and not _is_explicit_casual(message)
        and not _is_positive_state_report(message)
    ):
        logger.debug("[ToneDetector] Distress-sticky floor: CONVERSATIONAL → CONCERN")
        return ToneAnalysis(
            level=CrisisLevel.CONCERN,
            confidence=max(confidence, TONE_CONFIG["threshold_concern"]),
            trigger="distress_sticky_floor",
            raw_scores=raw_scores,
            explanation="Maintained CONCERN — session already in distress",
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

    _arbiter_said_conversational = False
    if use_llm_fallback and model_manager:
        logger.debug(f"[ToneDetector] Attempting LLM fallback for: {message[:50]}...")
        llm_result = await _llm_crisis_fallback(message, model_manager)
        if llm_result:
            llm_level, llm_confidence = llm_result
            logger.info(f"[ToneDetector] LLM fallback: {llm_level.value} (confidence: {llm_confidence:.2f})")
            # An arbiter CONVERSATIONAL verdict on a distress-dominant borderline
            # falls through to the deterministic backstop below instead of
            # returning — the arbiter can raise the level, but its low-confidence
            # "no crisis" cannot override semantics that clearly favor distress.
            if llm_level == CrisisLevel.CONVERSATIONAL:
                _arbiter_said_conversational = True
            else:
                # Independent-channel confirmation → teach the semantic channel
                # (this borderline shape then scores ABOVE borderline next time,
                # no arbiter round-trip needed).
                _ARB_KEYS = {
                    CrisisLevel.HIGH: "high",
                    CrisisLevel.MEDIUM: "medium",
                    CrisisLevel.CONCERN: "concern",
                }
                if llm_level in _ARB_KEYS:
                    _learn_tone_exemplar(
                        message, _ARB_KEYS[llm_level], "arbiter", model_manager
                    )
                return ToneAnalysis(
                    level=llm_level,
                    confidence=llm_confidence,
                    trigger="llm_fallback",
                    raw_scores=raw_scores,
                    explanation=f"LLM classification: {llm_level.value}"
                )

    # Deterministic borderline backstop: the arbiter is unavailable (no model
    # manager, timeout, or error) or returned a contradicted CONVERSATIONAL,
    # so a distress-shaped borderline must not default to CONVERSATIONAL. If
    # the strongest distress class outranks conversational by the semantic
    # margin, floor to CONCERN — never higher, because the full class
    # assignment still requires the absolute-score bar or a working arbiter.
    # (Fail-conversational was the live failure mode while the arbiter was
    # broken: "I keep thinking I am a stupid piece of shit ... I wanna cry"
    # scored medium=0.39 > conversational=0.30 and was labeled CONVERSATIONAL.)
    if use_llm_fallback and level == CrisisLevel.CONVERSATIONAL and raw_scores:
        _conv = raw_scores.get("conversational", 0.0)
        _top_distress = max(
            (raw_scores.get(k, 0.0) for k in ("high", "medium", "concern")),
            default=0.0,
        )
        if (
            _top_distress > _conv + 0.08
            and _top_distress >= TONE_CONFIG.get("backstop_min_score", 0.37)
        ):
            _why = (
                "arbiter said conversational but was overridden"
                if _arbiter_said_conversational else "arbiter unavailable"
            )
            logger.info(
                f"[ToneDetector] Borderline backstop: distress={_top_distress:.2f} > "
                f"conversational={_conv:.2f} with {_why} → CONCERN"
            )
            return ToneAnalysis(
                level=CrisisLevel.CONCERN,
                confidence=_top_distress,
                trigger="borderline_backstop",
                raw_scores=raw_scores,
                explanation=f"Distress outranked conversational on a borderline case; {_why}",
            )

    if _arbiter_said_conversational:
        return ToneAnalysis(
            level=CrisisLevel.CONVERSATIONAL,
            confidence=0.6,
            trigger="llm_fallback",
            raw_scores=raw_scores,
            explanation="LLM classification: conversational"
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
