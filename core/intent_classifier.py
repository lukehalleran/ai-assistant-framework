"""
Query Intent Classifier — fast, regex-first classification of user intent.

Module Contract:
  Purpose: Classify user queries into categorical intents that drive downstream
           retrieval counts, scoring weights, and gating thresholds.
  Inputs:  classify(query, tone_level=None) -> IntentResult
  Outputs: IntentResult with intent type, confidence, weight/retrieval/gate overrides.
  Side effects: None. Pure computation, no LLM calls.

Integration:
  - Runs as Stage 4.5 in ContextPipeline (after heavy-topic check, before query rewrite)
  - STM refinement: if classifier confidence < STM_REFINEMENT_THRESHOLD and STM
    produced an intent string, refine_with_stm() maps the free-text intent to a
    categorical IntentType, upgrading confidence.
  - ContextResult carries the IntentResult in its .intent field.
  - PromptBuilder reads intent.retrieval_overrides to adjust max_* counts.
  - MemoryScorer reads intent.weight_overrides via rank_memories(weight_overrides=...).

Dependencies: config.app_config (INTENT_* constants), enum, re, dataclasses
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from utils.logging_utils import get_logger

logger = get_logger("intent_classifier")

# ---------------------------------------------------------------------------
# Try to load config; fall back to defaults if unavailable
# ---------------------------------------------------------------------------
try:
    from config.app_config import (
        INTENT_ENABLED,
        INTENT_STM_REFINEMENT_THRESHOLD,
    )
except ImportError:
    INTENT_ENABLED = True
    INTENT_STM_REFINEMENT_THRESHOLD = 0.50


# ═══════════════════════════════════════════════════════════════════════════
# Intent Types
# ═══════════════════════════════════════════════════════════════════════════

class IntentType(str, Enum):
    """Categorical intent buckets that drive retrieval strategy."""
    FACTUAL_RECALL = "factual_recall"
    TEMPORAL_RECALL = "temporal_recall"
    EMOTIONAL_SUPPORT = "emotional_support"
    CASUAL_SOCIAL = "casual_social"
    TECHNICAL_HELP = "technical_help"
    CREATIVE_EXPLORATION = "creative_exploration"
    META_CONVERSATIONAL = "meta_conversational"
    PROJECT_WORK = "project_work"
    GENERAL = "general"


# ═══════════════════════════════════════════════════════════════════════════
# Intent Result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class IntentResult:
    """Output of intent classification — carried on ContextResult."""
    intent: IntentType
    confidence: float  # 0.0 – 1.0
    source: str = "regex"  # "regex" | "stm_refined"

    # Downstream overrides (populated from _PROFILES)
    weight_overrides: Dict[str, float] = field(default_factory=dict)
    retrieval_overrides: Dict[str, int] = field(default_factory=dict)
    gate_threshold_override: Optional[float] = None

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.70


# ═══════════════════════════════════════════════════════════════════════════
# Per-Intent Profiles  (weights, retrieval counts, gate thresholds)
# ═══════════════════════════════════════════════════════════════════════════
# Weight keys match SCORE_WEIGHTS in app_config: relevance, recency, truth,
# importance, continuity, structure.  Values must sum to ~1.0.
#
# Retrieval keys match the PROMPT_MAX_* constants in builder.py / gatherer.

# Per-intent profiles: scoring weights, retrieval count overrides, gate thresholds.
# Retrieval counts updated 2026-05-08 based on Phase 5+6 eval findings
# (50 snapshots, 1799 pairs, two judges: gpt-4o-mini + haiku-4.5).
_PROFILES: Dict[IntentType, dict] = {
    IntentType.FACTUAL_RECALL: {
        "weights": {
            "relevance": 0.40, "recency": 0.10, "truth": 0.30,
            "importance": 0.05, "continuity": 0.05, "structure": 0.10,
        },
        "retrieval": {
            "max_mems": 20, "max_recent": 5, "max_summaries": 5,
            "max_dreams": 0, "max_wiki": 5,
            "max_user_uploads": 0,  # eval: DROP? 0%/0%
        },
        "gate": 0.45,
    },
    IntentType.TEMPORAL_RECALL: {
        "weights": {
            "relevance": 0.20, "recency": 0.40, "truth": 0.10,
            "importance": 0.05, "continuity": 0.20, "structure": 0.05,
        },
        "retrieval": {
            "max_recent": 20, "max_summaries": 15, "max_mems": 10,
            "max_dreams": 0,
        },
        "gate": 0.30,
    },
    IntentType.EMOTIONAL_SUPPORT: {
        "weights": {
            "relevance": 0.25, "recency": 0.30, "truth": 0.10,
            "importance": 0.05, "continuity": 0.25, "structure": 0.05,
        },
        "retrieval": {
            "max_recent": 20, "max_mems": 10, "max_facts": 5,
            "max_dreams": 3,
            "max_reflections": 8,   # eval: KEEP for emotional (50%/100%)
            "max_narrative": 5,     # eval: KEEP (100%/50%)
        },
        "gate": 0.35,
    },
    IntentType.CASUAL_SOCIAL: {
        "weights": {
            "relevance": 0.20, "recency": 0.35, "truth": 0.10,
            "importance": 0.05, "continuity": 0.25, "structure": 0.05,
        },
        "retrieval": {
            "max_recent": 5, "max_mems": 3, "max_facts": 3,
            "max_summaries": 1,         # eval: tightened from 2 (26%/38% BL)
            "max_reflections": 0,       # eval: confirms DROP (21%/37%)
            "max_dreams": 0, "max_wiki": 0, "max_skills": 0,
            "max_proposals": 0, "max_git_commits": 0,
            "max_reference_docs": 0,    # eval: GPT DROP (36%), saves ~2K tokens
            "max_narrative": 0,         # eval: DROP (27%/33%)
            "max_user_uploads": 0,      # eval: DROP (25%/40%)
            "max_proactive": 0,         # eval: low-impact (29%/14%)
        },
        "gate": 0.65,
    },
    IntentType.TECHNICAL_HELP: {
        "weights": {
            "relevance": 0.45, "recency": 0.15, "truth": 0.20,
            "importance": 0.05, "continuity": 0.05, "structure": 0.10,
        },
        "retrieval": {
            "max_skills": 8, "max_mems": 10, "max_wiki": 5,
            "max_reflections": 0,       # eval: DROP (17%/33%)
            "max_reference_docs": 15,   # eval: Haiku 67% KEEP
        },
        "gate": 0.40,
    },
    IntentType.CREATIVE_EXPLORATION: {
        "weights": {
            "relevance": 0.30, "recency": 0.15, "truth": 0.10,
            "importance": 0.15, "continuity": 0.15, "structure": 0.15,
        },
        "retrieval": {
            "max_mems": 15, "max_reflections": 8,
        },
        "gate": 0.35,
    },
    IntentType.META_CONVERSATIONAL: {
        "weights": {
            "relevance": 0.30, "recency": 0.20, "truth": 0.15,
            "importance": 0.15, "continuity": 0.10, "structure": 0.10,
        },
        "retrieval": {
            "max_mems": 5,              # eval: reduced from 20 (DROP? 33%/33%)
            "max_summaries": 5,         # reduced from 10
            "max_reflections": 0,       # eval: DROP (0%/0% for meta)
            "max_reference_docs": 15,   # eval: KEEP? (100%/33%)
        },
        "gate": 0.30,
    },
    IntentType.PROJECT_WORK: {
        "weights": {
            "relevance": 0.40, "recency": 0.15, "truth": 0.25,
            "importance": 0.05, "continuity": 0.05, "structure": 0.10,
        },
        "retrieval": {
            "max_skills": 8, "max_git_commits": 15,
            "max_proposals": 5, "max_mems": 15,
            "max_recent": 5,            # eval: reduced from 15 (DROP 25%/25%)
            "max_reflections": 4,       # eval: SPLIT (50%/25%), keep some
            "max_reference_docs": 15,   # eval: KEEP (75%/50%)
        },
        "gate": 0.40,
    },
    IntentType.GENERAL: {
        "weights": {},  # use global SCORE_WEIGHTS unchanged
        "retrieval": {},  # use default PROMPT_MAX_* unchanged
        "gate": None,  # use default GATE_COSINE_THRESHOLD
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Regex Pattern Bank
# ═══════════════════════════════════════════════════════════════════════════
# Each entry: (compiled_regex, IntentType, confidence_boost)
# Patterns are checked in order; first match wins.

def _compile_patterns() -> List[Tuple[re.Pattern, IntentType, float]]:
    """Build the ordered pattern list (compiled once at import time)."""

    patterns = []

    def _add(regex: str, intent: IntentType, conf: float,
             flags: int = re.IGNORECASE):
        patterns.append((re.compile(regex, flags), intent, conf))

    # --- CASUAL_SOCIAL (high-priority, catches greetings early) -----------
    _add(
        r"^(hey|hi|hello|sup|yo|what'?s up|howdy|good\s+(morning|evening|night|afternoon)"
        r"|thanks|thank you|bye|goodbye|see you|later|cool|nice|awesome"
        r"|ok(ay)?|sure|ye(ah|p)|nah|nope|lol|haha|hmm|mhm)\b[.!?…]*$",
        IntentType.CASUAL_SOCIAL, 0.95,
    )
    # Very short messages with no question mark
    _add(r"^[^?]{1,15}$", IntentType.CASUAL_SOCIAL, 0.40)

    # --- EMOTIONAL_SUPPORT ------------------------------------------------
    _add(
        r"\bi('?m| am| feel| feeling)\s+"
        r"(so\s+|really\s+|very\s+|feeling\s+)?(sad|depressed|anxious|stressed|overwhelmed|lonely|scared"
        r"|angry|frustrated|hurt|tired|exhausted|hopeless|empty|lost"
        r"|broken|miserable|numb|terrified|panick|suicid)",
        IntentType.EMOTIONAL_SUPPORT, 0.90,
    )
    _add(
        r"\bi\s+(can'?t|cannot)\s+(cope|handle|deal|take|do this|go on|sleep|stop|breathe)",
        IntentType.EMOTIONAL_SUPPORT, 0.90,
    )
    _add(
        r"\b(i'?m struggling|having a (hard|tough|bad) time|don'?t know what to do"
        r"|need (to talk|someone|help)|feel(ing)? (like|so) (crap|shit|garbage|nothing))",
        IntentType.EMOTIONAL_SUPPORT, 0.85,
    )
    _add(
        r"\b(i need you|please help me|i'?m (scared|afraid|worried|nervous|anxious))\b",
        IntentType.EMOTIONAL_SUPPORT, 0.80,
    )

    # --- FACTUAL_RECALL ---------------------------------------------------
    _add(
        r"\b(what('?s| is| was) my|do you (know|remember) my"
        r"|what did i (tell|say|mention)|i told you (that|about|my)"
        r"|you (said|told|mentioned) (that|my)|remind me (of|about|what))\b",
        IntentType.FACTUAL_RECALL, 0.90,
    )
    _add(
        r"\b(what('?s| is) (his|her|their) (name|age|job|birthday))\b",
        IntentType.FACTUAL_RECALL, 0.85,
    )
    _add(
        r"\b(who (is|was)|where (do|did) i|when (did|do|is) (i|my))\b",
        IntentType.FACTUAL_RECALL, 0.80,
    )

    # --- TEMPORAL_RECALL --------------------------------------------------
    _add(
        r"\b(last (week|month|time|session|night|year)|yesterday"
        r"|a few (days|weeks|months) ago|earlier today|the other day"
        r"|remember when|what (did|were) we (talk|discuss|chat)"
        r"|what have (i|we) been|history|over time|progression"
        r"|how (long|much) (have|has)|used to)\b",
        IntentType.TEMPORAL_RECALL, 0.85,
    )

    # --- META_CONVERSATIONAL ----------------------------------------------
    _add(
        r"\b(what do you (know|remember|think) about"
        r"|tell me about yourself|how do you work"
        r"|your (memory|memories|capabilities|knowledge)"
        r"|what (have you|do you have) (learned|stored|saved)"
        r"|show me (my|your) (profile|facts|data))\b",
        IntentType.META_CONVERSATIONAL, 0.90,
    )

    # --- TECHNICAL_HELP ---------------------------------------------------
    _add(
        r"\b(how (do|can|should|would) (i|you|we|one)"
        r"|fix(ing)?|debug(ging)?|error|bug|issue|exception|traceback"
        r"|doesn'?t work|isn'?t working|not working|broken|crash(es|ing|ed)?"
        r"|help me (with|fix|debug|understand|figure))\b",
        IntentType.TECHNICAL_HELP, 0.75,
    )
    _add(
        r"\b(implement|refactor|optimize|deploy|install|configure|setup|set up)\b",
        IntentType.TECHNICAL_HELP, 0.65,
    )

    # --- PROJECT_WORK -----------------------------------------------------
    _add(
        r"\b(let'?s (work on|build|create|add|implement|code)"
        r"|add (a |the )?(feature|endpoint|component|module|test|class)"
        r"|write (a |the )?(function|method|test|script|module)"
        r"|update (the |my )?(code|codebase|project|repo)"
        r"|PR|pull request|merge|commit|branch|deploy)\b",
        IntentType.PROJECT_WORK, 0.80,
    )
    # File references
    _add(r"\b\w+\.(py|js|ts|jsx|tsx|go|rs|java|c|cpp|h|yaml|yml|json|toml)\b",
         IntentType.PROJECT_WORK, 0.60)

    # --- CREATIVE_EXPLORATION ---------------------------------------------
    _add(
        r"\b(brainstorm|idea(s)?|what if|imagine|possibilities"
        r"|help me think|let'?s think|explore|hypothetical"
        r"|creative|invent|design|dream up|could we|what about)\b",
        IntentType.CREATIVE_EXPLORATION, 0.75,
    )

    return patterns


_PATTERNS = _compile_patterns()


# ═══════════════════════════════════════════════════════════════════════════
# STM Intent → IntentType Mapping (for refinement)
# ═══════════════════════════════════════════════════════════════════════════
# STM returns free-text intent like "Get practical solution to immediate
# technical problem".  We keyword-match into IntentType.

_STM_KEYWORDS: Dict[str, IntentType] = {
    # FACTUAL_RECALL
    "recall": IntentType.FACTUAL_RECALL,
    "remember": IntentType.FACTUAL_RECALL,
    "verify": IntentType.FACTUAL_RECALL,
    "confirm": IntentType.FACTUAL_RECALL,
    "lookup": IntentType.FACTUAL_RECALL,
    "look up": IntentType.FACTUAL_RECALL,
    "fact": IntentType.FACTUAL_RECALL,

    # TEMPORAL_RECALL
    "history": IntentType.TEMPORAL_RECALL,
    "timeline": IntentType.TEMPORAL_RECALL,
    "past": IntentType.TEMPORAL_RECALL,
    "previous": IntentType.TEMPORAL_RECALL,
    "progression": IntentType.TEMPORAL_RECALL,
    "over time": IntentType.TEMPORAL_RECALL,

    # EMOTIONAL_SUPPORT
    "emotional": IntentType.EMOTIONAL_SUPPORT,
    "support": IntentType.EMOTIONAL_SUPPORT,
    "vent": IntentType.EMOTIONAL_SUPPORT,
    "comfort": IntentType.EMOTIONAL_SUPPORT,
    "reassurance": IntentType.EMOTIONAL_SUPPORT,
    "cope": IntentType.EMOTIONAL_SUPPORT,
    "feel better": IntentType.EMOTIONAL_SUPPORT,

    # TECHNICAL_HELP
    "solution": IntentType.TECHNICAL_HELP,
    "fix": IntentType.TECHNICAL_HELP,
    "debug": IntentType.TECHNICAL_HELP,
    "troubleshoot": IntentType.TECHNICAL_HELP,
    "technical": IntentType.TECHNICAL_HELP,
    "resolve": IntentType.TECHNICAL_HELP,
    "implement": IntentType.TECHNICAL_HELP,

    # META_CONVERSATIONAL
    "about yourself": IntentType.META_CONVERSATIONAL,
    "capabilities": IntentType.META_CONVERSATIONAL,
    "your memory": IntentType.META_CONVERSATIONAL,
    "what you know": IntentType.META_CONVERSATIONAL,

    # CREATIVE_EXPLORATION
    "brainstorm": IntentType.CREATIVE_EXPLORATION,
    "explore": IntentType.CREATIVE_EXPLORATION,
    "creative": IntentType.CREATIVE_EXPLORATION,
    "imagine": IntentType.CREATIVE_EXPLORATION,
    "idea": IntentType.CREATIVE_EXPLORATION,

    # PROJECT_WORK
    "build": IntentType.PROJECT_WORK,
    "develop": IntentType.PROJECT_WORK,
    "project": IntentType.PROJECT_WORK,
    "code": IntentType.PROJECT_WORK,
    "deploy": IntentType.PROJECT_WORK,
}


# ═══════════════════════════════════════════════════════════════════════════
# Classifier
# ═══════════════════════════════════════════════════════════════════════════

class IntentClassifier:
    """
    Fast, regex-first query intent classifier.

    Usage::

        classifier = IntentClassifier()
        result = classifier.classify("What's my sister's name?")
        # result.intent == IntentType.FACTUAL_RECALL
        # result.weight_overrides == {"relevance": 0.40, ...}
    """

    def classify(
        self,
        query: str,
        tone_level: Optional[str] = None,
    ) -> IntentResult:
        """
        Classify a user query into an IntentType.

        Args:
            query: The raw user query.
            tone_level: Optional tone level string (HIGH/MEDIUM/CONCERN/CONVERSATIONAL).
                        HIGH/MEDIUM tone biases toward EMOTIONAL_SUPPORT.

        Returns:
            IntentResult with intent, confidence, and downstream overrides.
        """
        if not INTENT_ENABLED:
            return self._build_result(IntentType.GENERAL, 0.0)

        query_stripped = query.strip()
        if not query_stripped:
            return self._build_result(IntentType.CASUAL_SOCIAL, 0.95)

        # --- Tone-level bias: crisis/elevated → emotional support --------
        if tone_level in ("HIGH", "MEDIUM"):
            # Don't override, but boost emotional if regex also agrees
            emotional_bias = True
        else:
            emotional_bias = False

        # --- Regex pattern matching (first match wins) -------------------
        best_intent = IntentType.GENERAL
        best_conf = 0.0

        for pattern, intent, conf in _PATTERNS:
            if pattern.search(query_stripped):
                if conf > best_conf:
                    best_intent = intent
                    best_conf = conf
                    # High confidence → stop early
                    if best_conf >= 0.90:
                        break

        # --- Emotional bias from tone ------------------------------------
        if emotional_bias and best_intent == IntentType.GENERAL:
            best_intent = IntentType.EMOTIONAL_SUPPORT
            best_conf = max(best_conf, 0.60)
        elif emotional_bias and best_intent == IntentType.EMOTIONAL_SUPPORT:
            best_conf = min(best_conf + 0.10, 1.0)

        result = self._build_result(best_intent, best_conf)

        # Thread temporal anchor for TEMPORAL_RECALL so the scorer can
        # reshape the recency decay curve around the referenced time window.
        if best_intent == IntentType.TEMPORAL_RECALL:
            from utils.query_checker import extract_temporal_window
            days = extract_temporal_window(query_stripped)
            if days > 0:
                result.weight_overrides["_temporal_anchor_hours"] = days * 24

        logger.debug(
            f"Classified '{query_stripped[:40]}...' → {result.intent.value} "
            f"(conf={result.confidence:.2f}, source={result.source})"
        )
        return result

    def refine_with_stm(
        self,
        result: IntentResult,
        stm_intent: Optional[str],
    ) -> IntentResult:
        """
        Optionally refine a low-confidence classification using STM's
        free-text intent string (no extra LLM call — STM already ran).

        Only applied when:
        - result.confidence < INTENT_STM_REFINEMENT_THRESHOLD
        - stm_intent is a non-empty string

        Args:
            result: The regex-based IntentResult.
            stm_intent: Free-text intent from STMAnalyzer (e.g. "Get practical solution").

        Returns:
            Possibly upgraded IntentResult (source="stm_refined").
        """
        if not stm_intent or not isinstance(stm_intent, str):
            return result

        if result.confidence >= INTENT_STM_REFINEMENT_THRESHOLD:
            return result  # Already confident

        stm_lower = stm_intent.lower()

        # Find the best matching IntentType from STM keywords
        for keyword, intent_type in _STM_KEYWORDS.items():
            if keyword in stm_lower:
                # STM refinement bumps confidence to threshold
                refined_conf = max(result.confidence, INTENT_STM_REFINEMENT_THRESHOLD)
                refined = self._build_result(intent_type, refined_conf)
                refined.source = "stm_refined"
                logger.debug(
                    f"STM refined {result.intent.value}→{intent_type.value} "
                    f"(stm_intent='{stm_intent}', conf={refined_conf:.2f})"
                )
                return refined

        return result  # No keyword match → keep original

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    # Keys added in Phase 8 (eval-driven gating). When section gating is
    # disabled, these are stripped so only the original pre-Phase-8 retrieval
    # overrides take effect.
    _PHASE8_GATING_KEYS = frozenset({
        "max_reference_docs", "max_narrative", "max_user_uploads",
        "max_proactive", "max_personal_notes",
    })

    def _build_result(self, intent: IntentType, confidence: float) -> IntentResult:
        """Build an IntentResult with profile overrides populated."""
        from config.app_config import PROMPT_SECTION_GATING_ENABLED

        profile = _PROFILES.get(intent, _PROFILES[IntentType.GENERAL])
        retrieval = dict(profile.get("retrieval", {}))

        # Strip Phase 8 gating keys when section gating is disabled
        if not PROMPT_SECTION_GATING_ENABLED:
            retrieval = {
                k: v for k, v in retrieval.items()
                if k not in self._PHASE8_GATING_KEYS
            }

        return IntentResult(
            intent=intent,
            confidence=round(confidence, 3),
            weight_overrides=dict(profile.get("weights", {})),
            retrieval_overrides=retrieval,
            gate_threshold_override=profile.get("gate"),
        )
