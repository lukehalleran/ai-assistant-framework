"""
Context Pipeline - Builder pattern for prompt preparation.

Purpose: Transform raw user input into fully processed context ready for prompt building.
Inputs: User query, optional files, configuration flags
Outputs: ContextResult with all context components (incl. last_exchange — the
newest prior turn, for downstream anaphora resolution by the ResponsePlanner)
Side effects: May call LLM for tone detection, query rewriting, STM analysis.
Topic stage: anaphoric continuations / referent corrections ("It was...",
"No I mean...") INHERIT the previous turn's topic instead of being
fresh-classified (query_checker.is_anaphoric_continuation, 2026-07-28 —
surface-keyword classification of an unresolvable fragment mislabeled an
illness-frequency message as "Exercise Routine" and derailed the turn).
STM gate (2026-08-05): _should_run_stm() honors the depth counter OR recent
corpus history (≤6h) — depth is in-process state, so a mid-day restart used
to drop the [SHORT-TERM CONTEXT SUMMARY] from the first messages exactly when
the model reconstructs a timeline across a gap (a first-message agentic
answer asserted "3+ days of pain" for a ~2-day episode).
Publishes live stage-progress lines to utils.turn_progress (no-op outside a turn)
so the streaming UI shows pipeline activity instead of a single "Thinking..." stall.

This module extracts the prepare_prompt workflow from the orchestrator,
making it testable, maintainable, and independently evolvable.

SCOPE: Query Analysis ONLY (pre-retrieval).
This pipeline does NOT handle:
- Memory retrieval → That's MemoryCoordinator's job
- Prompt assembly → That's PromptBuilder's job
- LLM generation → That's ResponseGenerator's job

Clean Data Flow:
    ContextPipeline.build()     →  ContextResult
                                        ↓
    MemoryCoordinator.get_memories(context.processed_query, context.topics)  →  memories
                                        ↓
    PromptBuilder.build_prompt(context, memories)  →  final prompt
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol, Union, TYPE_CHECKING
from enum import Enum
import asyncio
import logging
import os
from datetime import datetime, timedelta

from config.app_config import (
    USE_STM_PASS,
    STM_MIN_CONVERSATION_DEPTH,
    STM_MAX_RECENT_MESSAGES,
    REWRITE_TIMEOUT_S,
    INTENT_ENABLED,
)
from core.intent_classifier import IntentClassifier, IntentResult, IntentType
from utils.turn_progress import emit as _progress_emit

if TYPE_CHECKING:
    from utils.topic_manager import TopicManager
    from utils.file_processor import FileProcessor
    from core.stm_analyzer import STMAnalyzer
    from memory.user_profile import UserProfile

logger = logging.getLogger(__name__)


class ToneLevel(Enum):
    """Detected emotional tone levels matching CrisisLevel from tone_detector."""
    CRISIS = "HIGH"
    ELEVATED = "MEDIUM"
    CONCERN = "CONCERN"
    CONVERSATIONAL = "CONVERSATIONAL"

    @classmethod
    def from_string(cls, level: str) -> "ToneLevel":
        """
        Convert a crisis-level string to ToneLevel. Accepts BOTH encodings so
        callers passing either CrisisLevel.name ("HIGH"/"MEDIUM"/"CONCERN") or
        CrisisLevel.value ("crisis_support"/"elevated_support"/"light_support")
        map correctly.

        WARNING (2026-07-21): the pipeline passes `crisis_level.value`, whose
        forms are "*_support"/"conversational" — NOT the name-scale keys. Before
        this table accepted them, every level defaulted to CONVERSATIONAL, so the
        EscalationTracker was fed CONVERSATIONAL every turn and GROUNDING/QUIET
        could never fire in production. Unit tests missed it by calling
        tracker.update(ToneLevel.CRISIS) directly, bypassing this conversion;
        the golden-transcript replay (through the real path) caught it. Keep both
        encodings mapped. Regression: tests/unit/test_tonelevel_from_string.py.
        """
        if level is None:
            return cls.CONVERSATIONAL
        level_map = {
            # CrisisLevel.name scale
            "HIGH": cls.CRISIS,
            "MEDIUM": cls.ELEVATED,
            "CONCERN": cls.CONCERN,
            "CONVERSATIONAL": cls.CONVERSATIONAL,
            # CrisisLevel.value encodings (what the pipeline actually passes)
            "crisis_support": cls.CRISIS,
            "elevated_support": cls.ELEVATED,
            "light_support": cls.CONCERN,
            "conversational": cls.CONVERSATIONAL,
        }
        if level in level_map:
            return level_map[level]
        # Case-insensitive fallback across both scales.
        low = str(level).strip().lower()
        for key, val in level_map.items():
            if key.lower() == low:
                return val
        return cls.CONVERSATIONAL


def _upload_basename(file: Any) -> str:
    """Basename of an attached file as the persisted `upload:<name>` title
    spells it: the client's original name when the API shim carries one
    (`orig_name`), else the basename of `.name` (2026-09-04 — the same-turn
    upload dedupe compared server temp names against real names and never
    matched)."""
    orig = getattr(file, 'orig_name', None)
    if isinstance(orig, (str, os.PathLike)) and str(orig).strip():
        return os.path.basename(str(orig))
    return os.path.basename(str(getattr(file, 'name', '') or ''))


@dataclass
class ContextResult:
    """
    Immutable result from the context pipeline.

    Contains all processed context components needed for:
    1. Memory retrieval (processed_query, topics)
    2. Prompt building (tone_instructions, identity_block, thread_context, etc.)
    3. Response generation (tone_level for appropriate styling)
    """
    # Query information
    processed_query: str
    original_query: str

    # Tone/emotional context
    tone_level: ToneLevel
    tone_instructions: str
    emotional_context: Optional[Any] = None  # Full EmotionalContext object

    # Topic information
    topics: List[str] = field(default_factory=list)
    primary_topic: Optional[str] = None

    # File context
    file_context: Optional[str] = None

    # Basenames of files attached THIS turn (2026-09-04, homework-attachment
    # turn audit item 3) — lets the uploads gatherer avoid re-surfacing a
    # just-persisted chunk of a file whose full content is already present
    # verbatim in [CURRENT QUERY] this same turn.
    uploaded_filenames: List[str] = field(default_factory=list)

    # Thread context
    thread_context: Optional[Dict[str, Any]] = None

    # STM analysis
    stm_summary: Optional[Dict[str, Any]] = None

    # Identity/personality
    identity_block: str = ""
    user_name: Optional[str] = None

    # Heavy topic handling
    is_heavy_topic: bool = False
    extracted_facts: List[Dict[str, Any]] = field(default_factory=list)

    # Query analysis
    query_analysis: Optional[Any] = None  # QueryAnalysis dataclass

    # Intent classification
    intent: Optional["IntentResult"] = None

    # Small talk flag (set when CASUAL_SOCIAL intent with high confidence)
    is_small_talk: bool = False

    # Newest prior turn ({query, response, ...}) from conversation_history.
    # Lets downstream consumers that see ONLY the raw query (ResponsePlanner)
    # resolve pronoun-anchored fragments ("It was maybe 3 years of...")
    # against the exchange they actually refer to.
    last_exchange: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_files(self) -> bool:
        """Check if file context is present."""
        return self.file_context is not None and len(self.file_context) > 0

    @property
    def has_thread(self) -> bool:
        """Check if thread context is present."""
        return self.thread_context is not None and bool(self.thread_context.get("thread_id"))

    @property
    def has_stm(self) -> bool:
        """Check if STM summary is present."""
        return self.stm_summary is not None and len(self.stm_summary) > 0

    @property
    def crisis_level_str(self) -> str:
        """Get crisis level as string for backwards compatibility."""
        return self.tone_level.value



def stm_skip_shape(user_input: str, is_small_talk: bool = False, max_words: int = 6) -> bool:
    """True when the message has nothing for the STM analyzer to summarize: a
    small-talk turn (light path) or a short greeting / casual acknowledgement
    (2026-09-03 — "Hey" after a 2 h gap ran STM, which summarized the PREVIOUS
    turn as a restatement and injected the recall warning). Failure → False."""
    if is_small_talk:
        return True
    try:
        from utils.query_checker import is_greeting_opener, is_casual_acknowledgment
        text = user_input or ""
        if len(text.split()) > max_words:
            return False
        return bool(is_greeting_opener(text) or is_casual_acknowledgment(text))
    except Exception:
        return False


class ContextPipelineProtocol(Protocol):
    """Protocol for context pipeline implementations."""

    async def build(
        self,
        user_input: str,
        files: Optional[List[Any]] = None,
        use_raw_mode: bool = False,
        personality: Optional[str] = None
    ) -> ContextResult:
        """Build context from user input."""
        ...


class ContextPipeline:
    """
    Builder that transforms raw user input into processed context.

    SCOPE: Query Analysis ONLY (pre-retrieval).
    This pipeline does NOT handle memory retrieval—that's ContextGatherer/MemoryCoordinator's job.

    Pipeline stages:
    1. Topic Extraction - Extract topics (delegates to TopicManager)
    2. Tone Detection - Detect emotional state (delegates to analyze_emotional_context)
    3. File Processing - Extract text from PDF/DOCX/CSV (delegates to FileProcessor)
    4. Heavy Topic Check - Check for sensitive content (delegates to QueryChecker)
    5. Query Rewriting - Optionally rewrite for better retrieval
    6. STM Analysis - Analyze recent conversation context (delegates to STMAnalyzer)
    7. Identity Injection - Add user identity context (delegates to UserProfile)
    8. Thread Context - Get active thread (delegates to memory_system)

    Output: ContextResult → feeds into MemoryCoordinator.get_memories() → then PromptBuilder
    """

    def __init__(
        self,
        model_manager: Any,
        topic_manager: "TopicManager",
        file_processor: Optional["FileProcessor"] = None,
        stm_analyzer: Optional["STMAnalyzer"] = None,
        user_profile: Optional["UserProfile"] = None,
        memory_system: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the context pipeline.

        Args:
            model_manager: LLM provider abstraction for embeddings and generation
            topic_manager: Topic extraction utility
            file_processor: File upload processing utility
            stm_analyzer: Short-term memory analyzer
            user_profile: User identity/profile information
            memory_system: Memory coordinator for thread context and facts
            config: Additional configuration options
        """
        self.model_manager = model_manager
        self.topic_manager = topic_manager
        self.file_processor = file_processor
        self.stm_analyzer = stm_analyzer
        self.user_profile = user_profile
        self.memory_system = memory_system
        self.config = config or {}

        # Session tone memory: the prior turn's detected crisis level, fed back
        # into tone detection so distress stays sticky across short/terse turns
        # (prevents a spiral built from brief messages from flatlining at
        # CONVERSATIONAL). Dropped when the gap since the last stored turn
        # exceeds TONE_STICKINESS_MAX_GAP_MINUTES (_should_reset_tone_stickiness).
        # Seeded from data/tone_state.json so a RESTART doesn't cold-start the
        # signal (2026-08-02: sessions minutes apart — CONCERN at 12:13, restart,
        # flat semantic at 12:33 had no floor because the carry was process-local).
        self._last_tone_level: Optional[object] = self._load_persisted_tone()
        # Consecutive distress-sticky-floor turns (2026-08-22): the floor's own
        # CONCERN output was feeding _last_tone_level AND tone_state.json, so
        # one latch chained indefinitely — every short technical message all
        # afternoon got LIGHT SUPPORT. The floor may hold a genuine short-turn
        # spiral (the 07-21 anti-flatline case) for up to TONE_FLOOR_CHAIN_MAX
        # consecutive turns; beyond that, fresh organic evidence is required.
        self._floor_chain: int = 0

        # Configuration with defaults
        self._use_stm = self.config.get("USE_STM_PASS", USE_STM_PASS)
        self._stm_min_depth = self.config.get("STM_MIN_CONVERSATION_DEPTH", STM_MIN_CONVERSATION_DEPTH)
        self._stm_max_recent = self.config.get("STM_MAX_RECENT_MESSAGES", STM_MAX_RECENT_MESSAGES)
        self._rewrite_timeout = self.config.get("REWRITE_TIMEOUT_S", REWRITE_TIMEOUT_S)
        self._enable_query_rewrite = self.config.get("enable_query_rewrite", True)

        # Intent classifier (regex-first, no LLM calls)
        self._intent_enabled = self.config.get("INTENT_ENABLED", INTENT_ENABLED)
        self._intent_classifier = IntentClassifier() if self._intent_enabled else None

        # Track conversation depth for STM decisions
        self._conversation_depth = 0

        # Tone instruction templates
        self._tone_instructions = {
            ToneLevel.CRISIS: self._load_crisis_instructions(),
            ToneLevel.ELEVATED: self._load_elevated_instructions(),
            ToneLevel.CONCERN: self._load_concern_instructions(),
            ToneLevel.CONVERSATIONAL: self._load_conversational_instructions(),
        }

    async def build(
        self,
        user_input: str,
        files: Optional[List[Any]] = None,
        use_raw_mode: bool = False,
        personality: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> ContextResult:
        """
        Main entry point - builds context through the full pipeline.

        Args:
            user_input: Raw user query
            files: Optional list of uploaded files
            use_raw_mode: Skip enrichment (direct passthrough)
            personality: Optional personality override
            conversation_history: Recent conversation for context

        Returns:
            ContextResult with all processed context components
        """
        logger.debug(f"Building context for query: {user_input[:50]}...")

        # Track conversation depth
        self._conversation_depth += 1

        # Initialize result components
        processed_query = user_input
        file_context = None
        tone_level = ToneLevel.CONVERSATIONAL
        emotional_context = None
        topics: List[str] = []
        primary_topic: Optional[str] = None
        is_heavy_topic = False
        extracted_facts: List[Dict] = []
        query_analysis = None
        stm_summary = None
        thread_context = None
        identity_block = ""
        user_name = None
        intent_result: Optional[IntentResult] = None

        # Newest prior turn — carried on the result so consumers that only see
        # the raw query (ResponsePlanner) can resolve anaphoric fragments.
        last_exchange: Optional[Dict] = None
        if conversation_history:
            first = conversation_history[0]
            if isinstance(first, dict):
                last_exchange = first

        # Stages 1+2: Topic Extraction + Tone Detection (parallelized — independent)
        if not use_raw_mode:
            _progress_emit("🧠 Analyzing query — topics + tone…")
            (primary_topic, topics), (tone_level, emotional_context) = await asyncio.gather(
                self._extract_topics(user_input, last_exchange=last_exchange),
                self._detect_tone(user_input, conversation_history),
            )
        else:
            primary_topic, topics = await self._extract_topics(
                user_input, last_exchange=last_exchange
            )
        logger.debug(f"Stage 1 (Topics): primary={primary_topic}, all={topics}")
        if not use_raw_mode:
            logger.debug(f"Stage 2 (Tone): level={tone_level.value}")

        # Stage 3: File Processing
        uploaded_filenames: List[str] = []
        if files and not use_raw_mode:
            file_context = await self._process_files(user_input, files)
            if file_context and file_context != user_input:
                # Files were processed, update processed_query
                processed_query = file_context
                logger.debug(f"Stage 3 (Files): processed {len(files)} files")
            # This turn's attached basenames (2026-09-04, homework-attachment
            # turn audit item 3) — lets the uploads gatherer skip retrieving
            # a just-persisted chunk of a file that's already fully present
            # verbatim in [CURRENT QUERY] this same turn.
            for _f in files:
                try:
                    _bn = _upload_basename(_f)
                    if _bn:
                        uploaded_filenames.append(_bn)
                except Exception:
                    pass

        # Stage 4a: Intent Classification (regex-first, no LLM, <1ms)
        # Moved BEFORE heavy topic check so we can skip expensive LLM calls
        # for casual/simple intents.
        is_small_talk = False
        if not use_raw_mode and self._intent_classifier:
            intent_result = self._intent_classifier.classify(
                user_input,
                tone_level=tone_level.value,
            )
            logger.debug(
                f"Stage 4a (Intent): {intent_result.intent.value} "
                f"(conf={intent_result.confidence:.2f})"
            )
            _progress_emit(
                f"🧭 Intent: {intent_result.intent.value} · tone: {tone_level.value}"
            )
            if intent_result.intent == IntentType.CASUAL_SOCIAL and intent_result.confidence >= 0.70:
                is_small_talk = True
                logger.debug("Stage 4a: is_small_talk=True (CASUAL_SOCIAL, high confidence)")

        # Stage 4b: Heavy Topic Check + Inline Fact Extraction
        # SKIP for casual/social intents and non-crisis short queries —
        # the intent classifier already identified these as lightweight.
        skip_heavy = use_raw_mode or is_small_talk
        if not skip_heavy and intent_result is not None:
            _SKIP_HEAVY_INTENTS = {
                IntentType.CASUAL_SOCIAL,
                IntentType.META_CONVERSATIONAL,
            }
            if (intent_result.intent in _SKIP_HEAVY_INTENTS
                    and intent_result.confidence >= 0.60):
                skip_heavy = True
            elif (tone_level == ToneLevel.CONVERSATIONAL
                    and len(user_input.split()) < 12
                    and intent_result.intent != IntentType.EMOTIONAL_SUPPORT):
                skip_heavy = True
        if not skip_heavy:
            _progress_emit("📋 Topic analysis + fact extraction…")
            is_heavy_topic, extracted_facts, query_analysis = await self._check_heavy_topics(
                user_input,
                topics
            )
            if is_heavy_topic:
                logger.debug(f"Stage 4b (Heavy Topic): detected, {len(extracted_facts)} facts extracted")
        elif skip_heavy:
            logger.debug(f"Stage 4b (Heavy Topic): SKIPPED (intent={intent_result.intent.value if intent_result else '?'}, words={len(user_input.split())})")

        # Stages 5+6: Query Rewriting + STM Analysis (parallelized — independent LLM calls)
        # Skip rewriting for short queries and casual/meta intents — the original
        # phrasing is already good enough and rewriting adds 500-1500ms.
        _word_count = len(user_input.split())
        _skip_rewrite_intents = {IntentType.CASUAL_SOCIAL, IntentType.META_CONVERSATIONAL}
        run_rewrite = (
            not use_raw_mode
            and self._enable_query_rewrite
            and _word_count >= 10
            and not is_small_talk
            and (intent_result is None or intent_result.intent not in _skip_rewrite_intents)
        )
        # 2026-09-03: a bare greeting/ack after a gap ran STM, which summarized
        # the PREVIOUS turn as if the user had restated it and injected the
        # "restates an event" warning onto "Hey". Nothing to analyze there.
        _stm_skip_shape = stm_skip_shape(user_input, is_small_talk=is_small_talk)
        run_stm = (
            not use_raw_mode
            and not _stm_skip_shape
            and self._should_run_stm(conversation_history)
        )

        if run_rewrite and run_stm:
            # Both needed — run in parallel for ~1-2s savings
            _progress_emit("✍️ Refining query + reading short-term context…")

            async def _do_rewrite():
                return await self._rewrite_query(user_input, query_analysis)

            async def _do_stm():
                try:
                    async with asyncio.timeout(10.0):
                        return await self._analyze_stm(user_input, conversation_history)
                except asyncio.TimeoutError:
                    logger.warning("Stage 6 (STM): analysis timed out")
                    return None

            rewritten, stm_summary = await asyncio.gather(
                _do_rewrite(), _do_stm()
            )
            if rewritten and rewritten != user_input:
                processed_query = rewritten
                logger.debug("Stage 5 (Rewrite): query rewritten")
            if stm_summary:
                logger.debug("Stage 6 (STM): analysis complete")

        elif run_rewrite:
            _progress_emit("✍️ Refining query for retrieval…")
            rewritten = await self._rewrite_query(user_input, query_analysis)
            if rewritten and rewritten != user_input:
                processed_query = rewritten
                logger.debug("Stage 5 (Rewrite): query rewritten")

        elif run_stm:
            _progress_emit("🧵 Reading short-term context…")
            try:
                async with asyncio.timeout(10.0):
                    stm_summary = await self._analyze_stm(user_input, conversation_history)
                if stm_summary:
                    logger.debug("Stage 6 (STM): analysis complete")
            except asyncio.TimeoutError:
                logger.warning("Stage 6 (STM): analysis timed out")
                stm_summary = None

        # Stage 6b: Refine intent with STM (no LLM, just keyword matching)
        if stm_summary and intent_result and self._intent_classifier:
            stm_intent_str = stm_summary.get("intent") if isinstance(stm_summary, dict) else None
            intent_result = self._intent_classifier.refine_with_stm(
                intent_result, stm_intent_str, query=user_input,
                tone_level=tone_level.value,
            )

        # Stage 7: Identity Injection
        identity_block, user_name = self._get_identity_context()
        if identity_block:
            logger.debug(f"Stage 7 (Identity): user={user_name}")

        # Stage 8: Thread Context
        if not use_raw_mode:
            thread_context = await self._get_thread_context()
            if thread_context:
                logger.debug(f"Stage 8 (Thread): depth={thread_context.get('thread_depth', 0)}")

        # Get tone instructions based on detected level
        tone_instructions = self._get_tone_instructions(tone_level)

        # Build the result
        return ContextResult(
            processed_query=processed_query,
            original_query=user_input,
            tone_level=tone_level,
            tone_instructions=tone_instructions,
            emotional_context=emotional_context,
            topics=topics,
            primary_topic=primary_topic,
            file_context=file_context,
            uploaded_filenames=uploaded_filenames,
            thread_context=thread_context,
            stm_summary=stm_summary,
            identity_block=identity_block,
            user_name=user_name,
            is_heavy_topic=is_heavy_topic,
            extracted_facts=extracted_facts,
            query_analysis=query_analysis,
            last_exchange=last_exchange,
            intent=intent_result,
            is_small_talk=is_small_talk,
            metadata={
                "use_raw_mode": use_raw_mode,
                "has_files": file_context is not None,
                "topic_count": len(topics),
                "conversation_depth": self._conversation_depth,
                "stm_enabled": self._use_stm,
                "intent": intent_result.intent.value if intent_result else None,
                "intent_confidence": intent_result.confidence if intent_result else None,
            }
        )

    # --- Stage Implementations ---

    async def _extract_topics(
        self,
        query: str,
        last_exchange: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[str], List[str]]:
        """
        Stage 1: Extract topics via TopicManager.

        Returns:
            Tuple of (primary_topic, list_of_topics)
        """
        if not self.topic_manager:
            return None, []

        try:
            # Anaphoric continuations INHERIT the previous turn's topic instead
            # of being fresh-classified. The classifier sees only the message
            # text, so a pronoun-anchored fragment gets labeled by surface
            # keywords while its actual subject sits in the prior exchange
            # (2026-07-28: "It was maybe 3 years of twice a week..." — long
            # covid frequency — became topic "Exercise Routine", which then
            # drove a false [THREAD CONTEXT] shift assertion and a wrong
            # response plan). Inheriting keeps topic, thread continuity, and
            # the planner's Topics: signal aligned with the real referent.
            from utils.query_checker import (
                is_anaphoric_continuation,
                is_continuation_answer,
                is_fragment_continuation,
                topics_related,
            )
            prev_topic = getattr(self.topic_manager, "last_topic", None)
            last_response = ""
            if isinstance(last_exchange, dict):
                last_response = (
                    last_exchange.get("response")
                    or last_exchange.get("assistant")
                    or (
                        last_exchange.get("content", "")
                        if last_exchange.get("role") == "assistant" else ""
                    )
                    or ""
                )
            answers_last_question = is_continuation_answer(query, str(last_response))
            if (
                isinstance(prev_topic, str)
                and prev_topic.strip()
                and prev_topic.strip().lower() != "general"
                # 2026-08-22: bare noun-phrase fragments ("Tactical Taylors")
                # inherit like pronoun fragments — fresh classification of a
                # 2-word riff mid-thread produced topic "Tactical Gear" and a
                # gear-brand reply to a joke.
                and (
                    is_anaphoric_continuation(query)
                    or is_fragment_continuation(query)
                    or answers_last_question
                )
            ):
                logger.debug(
                    f"[ContextPipeline] Anaphoric continuation — inheriting "
                    f"previous topic '{prev_topic}' instead of fresh-classifying"
                )
                return prev_topic, [prev_topic]

            # Get primary topic (also updates internal state + has LLM cache)
            primary = self.topic_manager.get_primary_topic(query)

            # Label stabilization (2026-09-03): when the fresh label is merely
            # a RELABEL of the previous turn's topic (loose match via the same
            # topics_related predicate thread detection uses), keep the
            # previous label — classifier granularity noise ("Playing Fetch"
            # → "Playing Games") had been resetting thread depth and
            # asserting shifts on one continuous conversation. The classifier
            # still runs (it maintains its own state); only the returned label
            # is anchored. Real shifts (unrelated labels) pass through.
            if (
                isinstance(prev_topic, str)
                and isinstance(primary, str)
                and prev_topic.strip()
                and primary.strip()
                and prev_topic.strip().lower() != "general"
                and primary.strip().lower() != "general"
                and prev_topic.strip().lower() != primary.strip().lower()
                and topics_related(prev_topic, primary)
            ):
                logger.debug(
                    f"[ContextPipeline] Topic label '{primary}' is a relabel of "
                    f"'{prev_topic}' — keeping the previous label for continuity"
                )
                primary = prev_topic
                try:
                    self.topic_manager.last_topic = prev_topic
                except Exception:
                    pass

            # Get all topics (primary + any extracted entities)
            topics = []
            if primary:
                topics.append(primary)

            # If topic manager has entity extraction, include those
            if hasattr(self.topic_manager, 'get_entities'):
                entities = self.topic_manager.get_entities(query)
                topics.extend([e for e in entities if e not in topics])

            return primary, topics

        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
            return None, []

    async def _detect_tone(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> tuple[ToneLevel, Optional[Any]]:
        """
        Stage 2: Detect emotional tone level.

        Delegates to analyze_emotional_context from utils/emotional_context.py

        Returns:
            Tuple of (ToneLevel, EmotionalContext)
        """
        try:
            # lazy import: patch point (tone tests monkeypatch
            # utils.emotional_context.analyze_emotional_context; the call-time
            # import is what makes the patch visible here)
            from utils.emotional_context import analyze_emotional_context

            # Get recent memories for context if memory_system available
            recent_memories = []
            if self.memory_system and hasattr(self.memory_system, 'corpus_manager'):
                try:
                    recent_memories = self.memory_system.corpus_manager.get_recent_memories(3)
                except Exception as e:
                    logger.warning(f"[ContextPipeline] Could not retrieve recent memories for emotional analysis: {e}")

            # Also use provided conversation history
            if conversation_history:
                recent_memories = conversation_history[:3]

            # Over-stickiness guard: distress tone is sticky across terse turns
            # WITHIN a session, but a long gap means a new session — a calm
            # message hours after a distressed one must be classified on its own
            # merits, not floored to the earlier tone. Drop the carried tone when
            # the gap since the last turn exceeds the threshold.
            if self._should_reset_tone_stickiness(recent_memories):
                logger.info(
                    f"[ContextPipeline] Session gap exceeded — clearing carried tone "
                    f"stickiness (was {self._last_tone_level})"
                )
                self._last_tone_level = None

            # Analyze emotional context. previous_tone carries the prior turn's
            # crisis level so distress is sticky across short/terse messages.
            import os as _os
            _floor_chain_max = int(_os.getenv("TONE_FLOOR_CHAIN_MAX", "3"))
            _prev_tone = self._last_tone_level
            _floor_budget_left = self._floor_chain < _floor_chain_max
            if not _floor_budget_left:
                # The carried tone is floor-produced N turns deep — stop
                # feeding it back as evidence; only organic signals (semantic/
                # keyword/arbiter/backstop, or heavy history) may re-elevate.
                # Withholding previous_tone alone is NOT enough: the detector's
                # session-distress test falls through to recent-heavy-history
                # and the floor re-fired past the budget (2026-08-28, 4 chained
                # floor turns) — the floor stage itself is disabled too.
                _prev_tone = None
            emotional_ctx = await analyze_emotional_context(
                message=query,
                conversation_history=recent_memories,
                model_manager=self.model_manager,
                previous_tone=_prev_tone,
                allow_sticky_floor=_floor_budget_left,
            )

            # Convert crisis level to ToneLevel
            if emotional_ctx and hasattr(emotional_ctx, 'crisis_level'):
                level_str = emotional_ctx.crisis_level.value if hasattr(emotional_ctx.crisis_level, 'value') else str(emotional_ctx.crisis_level)
                tone_level = ToneLevel.from_string(level_str)
                # Remember this turn's crisis level for the next call's stickiness.
                self._last_tone_level = emotional_ctx.crisis_level
                # EmotionalContext's field is `tone_trigger` (ToneAnalysis uses
                # `trigger`) — reading the wrong name left the chain counter at 0
                # and persisted trigger="" every turn (dead TONE_FLOOR_CHAIN_MAX
                # guard + floor levels seedable across restart, 2026-08-27).
                _trigger = str(
                    getattr(emotional_ctx, 'tone_trigger', None)
                    or getattr(emotional_ctx, 'trigger', '')
                    or ''
                )
                if _trigger == "distress_sticky_floor":
                    self._floor_chain += 1
                else:
                    self._floor_chain = 0
                self._persist_tone(level_str, trigger=_trigger)
            else:
                tone_level = ToneLevel.CONVERSATIONAL

            return tone_level, emotional_ctx

        except ImportError:
            logger.warning("emotional_context module not available, using CONVERSATIONAL")
            return ToneLevel.CONVERSATIONAL, None
        except Exception as e:
            logger.warning(f"Tone detection failed: {e}")
            return ToneLevel.CONVERSATIONAL, None

    # Tone carryover persistence. This is DERIVED, self-healing state (next
    # turn rewrites it), so unlike real stores it loads leniently — a missing
    # or corrupt file just means a cold start, never a startup abort.
    _TONE_STATE_PATH = "data/tone_state.json"

    _ELEVATED_TONE_MARKERS = (
        "concern", "medium", "high", "light_support", "elevated_support",
        "crisis_support", "crisis",
    )

    def _load_persisted_tone(self):
        """Seed `_last_tone_level` from the previous process's last turn.

        Only an ELEVATED level within TONE_STICKINESS_MAX_GAP_MINUTES is
        carried — conversational carries no floor, and stale distress must
        not resurface hours later (same window the in-process gap check uses).
        Returns the level's string encoding (tone detection accepts CrisisLevel
        or string for previous_tone).
        """
        try:
            import json as _json
            from datetime import datetime, timedelta
            from pathlib import Path
            from config.app_config import TONE_STICKINESS_MAX_GAP_MINUTES
            p = Path(self._TONE_STATE_PATH)
            if not p.exists():
                return None
            state = _json.loads(p.read_text())
            level = str(state.get("level", "") or "")
            ts = datetime.fromisoformat(str(state.get("ts", "")))
            if datetime.now() - ts > timedelta(minutes=TONE_STICKINESS_MAX_GAP_MINUTES):
                return None
            low = level.lower()
            if not any(m in low for m in self._ELEVATED_TONE_MARKERS):
                return None
            if str(state.get("trigger", "") or "") == "distress_sticky_floor":
                # Floor-produced tone is the floor's OWN output, not evidence —
                # seeding it re-latches the self-perpetuating chain across
                # restarts (2026-08-22: light_support carried all afternoon).
                return None
            logger.info(
                f"[ContextPipeline] Carried tone across restart: {level} "
                f"(from {ts.isoformat(timespec='minutes')})"
            )
            return level
        except Exception as e:
            logger.debug(f"[ContextPipeline] tone-state load skipped: {e}")
            return None

    def _persist_tone(self, level_str: str, trigger: str = "") -> None:
        """Write this turn's tone level + trigger + timestamp (atomic, best-effort)."""
        try:
            from datetime import datetime
            from utils.safe_json import atomic_write_json
            atomic_write_json(
                self._TONE_STATE_PATH,
                {"level": str(level_str), "trigger": str(trigger or ""),
                 "ts": datetime.now().isoformat()},
            )
        except Exception as e:
            logger.debug(f"[ContextPipeline] tone-state persist skipped: {e}")

    def _should_reset_tone_stickiness(self, recent_memories) -> bool:
        """
        True when the carried tone (`_last_tone_level`) should be dropped because
        the gap since the last stored turn exceeds TONE_STICKINESS_MAX_GAP_MINUTES
        (a new session). Within-session (turns minutes apart) keeps stickiness.

        Fail-CLOSED toward preserving within-session stickiness: on any parse
        uncertainty it returns False (keeps the carried tone) rather than risk
        dropping distress mid-session. `recent_memories[0]` is the newest turn
        (get_recent_memories / conversation_history are newest-first).
        """
        if self._last_tone_level is None:
            return False  # nothing carried
        if not recent_memories:
            return True   # no prior turn on record → treat as a fresh session
        try:
            from datetime import datetime, timedelta
            from config.app_config import TONE_STICKINESS_MAX_GAP_MINUTES
            newest = recent_memories[0]
            ts = newest.get("timestamp") if isinstance(newest, dict) else None
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if not isinstance(ts, datetime):
                return False
            return (datetime.now() - ts) > timedelta(minutes=TONE_STICKINESS_MAX_GAP_MINUTES)
        except Exception as e:
            logger.debug(f"[ContextPipeline] tone-stickiness gap check failed: {e}")
            return False

    async def _process_files(
        self,
        user_input: str,
        files: List[Any]
    ) -> Optional[str]:
        """
        Stage 3: Process uploaded files and merge with user input.

        Delegates to FileProcessor.process_files()

        Returns:
            Combined text with file contents, or original input if processing fails
        """
        if not files:
            return None

        if not self.file_processor:
            logger.warning("FileProcessor not available, skipping file processing")
            return None

        try:
            combined = await self.file_processor.process_files(user_input, files)
            return combined
        except Exception as e:
            logger.warning(f"File processing failed: {e}")
            return None

    async def _check_heavy_topics(
        self,
        query: str,
        topics: List[str]
    ) -> tuple[bool, List[Dict], Optional[Any]]:
        """
        Stage 4: Check for heavy topics and extract inline facts.

        Delegates to QueryChecker for heavy topic detection.
        Uses memory_system for fact extraction if heavy topic detected.

        Returns:
            Tuple of (is_heavy_topic, extracted_facts, query_analysis)
        """
        try:
            from utils.query_checker import analyze_query_async, analyze_query

            # First, get basic query analysis (synchronous, heuristic only)
            query_analysis = analyze_query(query, self.model_manager)

            # Skip LLM heavy topic check for short casual messages —
            # keyword heuristic (250+ weighted terms) is sufficient
            if len(query.split()) < 8 and not query_analysis.is_heavy_topic:
                return False, [], query_analysis

            # Then check for heavy topics (async, may use LLM)
            async_analysis = await analyze_query_async(query, self.model_manager)

            is_heavy = async_analysis.is_heavy_topic if async_analysis else False

            # If heavy topic and memory_system available, extract facts
            extracted_facts = []
            if is_heavy and self.memory_system:
                try:
                    # Trigger inline fact extraction with timeout
                    async with asyncio.timeout(5.0):
                        if hasattr(self.memory_system, '_extract_and_store_facts'):
                            await self.memory_system._extract_and_store_facts(
                                query,
                                response="",  # No response yet
                                truth_score=0.8
                            )

                        # Retrieve extracted facts
                        if hasattr(self.memory_system, 'get_facts'):
                            extracted_facts = await self.memory_system.get_facts(
                                query=query,
                                limit=10
                            )
                except asyncio.TimeoutError:
                    logger.warning("Inline fact extraction timed out")
                except Exception as e:
                    logger.warning(f"Fact extraction failed: {e}")

            return is_heavy, extracted_facts, query_analysis

        except ImportError:
            logger.warning("query_checker module not available")
            return False, [], None
        except Exception as e:
            logger.warning(f"Heavy topic check failed: {e}")
            return False, [], None

    async def _rewrite_query(
        self,
        query: str,
        query_analysis: Optional[Any] = None
    ) -> Optional[str]:
        """
        Stage 5: Rewrite query for better semantic retrieval.

        Uses LLM to expand casual queries into semantic-rich versions.
        Only rewrites if query is a question or command with sufficient tokens.

        DISABLED by default: config.yaml ships `features.rewrite_timeout_s: 0`,
        which app_config resolves to REWRITE_TIMEOUT_S=0.0 — the `_rewrite_timeout
        == 0` guard below returns None immediately in the shipped config (see
        DaemonOrchestrator._build_context_pipeline_config for the wiring that
        used to silently override this with a nonzero default). Enabling it is
        not a free win: it changes the retrieval vector fed to the memory gate
        (the bge cosine thresholds — gate_rel_threshold_retrieval etc. — were
        calibrated against RAW queries, not LLM-rewritten ones), and it can
        overwrite an already-processed_query set by a file upload.

        Returns:
            Rewritten query or None if no rewrite needed
        """
        if not self.model_manager or self._rewrite_timeout == 0:
            return None

        # Check if query should be rewritten
        should_rewrite = False
        if query_analysis:
            should_rewrite = (
                query_analysis.is_question or query_analysis.is_command
            ) and query_analysis.token_count >= 8
        else:
            # Fallback heuristic
            should_rewrite = len(query.split()) >= 5 and (
                query.strip().endswith('?') or
                query.lower().startswith(('how', 'what', 'why', 'when', 'where', 'who'))
            )

        if not should_rewrite:
            return None

        try:
            # Build rewrite prompt
            rewrite_prompt = f"""Rewrite this user query for semantic search retrieval.
Convert casual language to formal statements. Expand abbreviations.
Keep the core meaning but make it more searchable.

Original query: {query}

Rewritten query (just the rewritten text, no explanation):"""

            async with asyncio.timeout(self._rewrite_timeout):
                result = await self.model_manager.generate_once(
                    prompt=rewrite_prompt,
                    model_name="gpt-4o-mini",
                    temperature=0.3,
                    max_tokens=150
                )

                if result and result.strip():
                    return result.strip()

        except asyncio.TimeoutError:
            logger.debug("Query rewrite timed out")
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")

        return None

    def _should_run_stm(self, conversation_history: Optional[List[Dict]] = None) -> bool:
        """Check if STM analysis should run.

        Depth gate PLUS recent-history override (2026-08-05): the depth
        counter is IN-PROCESS state — a mid-day restart zeroes it, so the
        first `stm_min_depth` messages after every restart ran WITHOUT the
        [SHORT-TERM CONTEXT SUMMARY] (no temporal_facts, no restate-warning)
        exactly when the model has to reconstruct "where were we" across a
        gap. Live miss: a first-message-of-session agentic answer asserted
        "three-plus days of documented pain" for a ~2-day episode, blending
        the current episode with July pain memories. When the corpus shows a
        recent conversation (same-day continuity), there IS short-term
        context worth reading — run STM regardless of process depth.
        """
        if not self._use_stm:
            return False
        if not self.stm_analyzer:
            return False
        if self._conversation_depth >= self._stm_min_depth:
            return True
        return self._has_recent_history(conversation_history)

    @staticmethod
    def _has_recent_history(
        history: Optional[List[Dict]], max_gap_hours: float = 6.0
    ) -> bool:
        """True when the newest history entry is within max_gap_hours."""
        if not history:
            return False
        newest = None
        for e in history:
            ts = e.get("timestamp") if isinstance(e, dict) else None
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except ValueError:
                    continue
            if isinstance(ts, datetime) and ts.tzinfo is None:
                if newest is None or ts > newest:
                    newest = ts
        if newest is None:
            return False
        try:
            return (datetime.now() - newest) <= timedelta(hours=max_gap_hours)
        except TypeError:
            return False

    async def _analyze_stm(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Stage 6: Analyze short-term memory context.

        Delegates to STMAnalyzer.analyze()

        Returns:
            Dict with topic, user_question, intent, tone, open_threads, constraints
        """
        if not self.stm_analyzer:
            return None

        try:
            # Get recent memories — prefer the time-windowed method (24h slice
            # capped at STM_MAX_RECENT_MESSAGES) when available so STM gets the
            # full session-day rather than the last N messages. Falls back to
            # the legacy fixed-N pull on older corpus_managers / mocks.
            recent_memories = []
            if self.memory_system and hasattr(self.memory_system, 'corpus_manager'):
                cm = self.memory_system.corpus_manager
                # Class-level hasattr (not instance) so Mock objects with auto-created
                # attributes correctly fall through to the legacy get_recent_memories path.
                if hasattr(type(cm), 'get_recent_within_hours'):
                    try:
                        from config.app_config import STM_RECENT_HOURS
                    except ImportError:
                        STM_RECENT_HOURS = 24
                    recent_memories = cm.get_recent_within_hours(
                        hours=STM_RECENT_HOURS,
                        max_count=self._stm_max_recent,
                    )
                else:
                    recent_memories = cm.get_recent_memories(self._stm_max_recent)
            elif conversation_history:
                recent_memories = conversation_history[:self._stm_max_recent]

            # Get the immediately preceding assistant response. Corpus entries
            # are normally {query, response} pairs (not role/content messages),
            # so the old role-only loop almost always passed None and STM had
            # no way to understand answers like "Day of please".
            last_response = None
            if conversation_history:
                first = conversation_history[0]
                if isinstance(first, dict):
                    last_response = (
                        first.get("response")
                        or first.get("assistant")
                        or (first.get("content") if first.get("role") == "assistant" else None)
                    )
            if not last_response and recent_memories:
                for mem in recent_memories:
                    if not isinstance(mem, dict):
                        continue
                    candidate = (
                        mem.get("response")
                        or mem.get("assistant")
                        or (mem.get("content") if mem.get("role") == "assistant" else None)
                    )
                    if candidate:
                        last_response = candidate
                        break

            result = await self.stm_analyzer.analyze(
                recent_memories=recent_memories,
                user_query=query,
                last_assistant_response=last_response,
                graph_memory=getattr(self.memory_system, "graph_memory", None),
            )

            return result

        except Exception as e:
            logger.warning(f"STM analysis failed: {e}")
            return None

    def _get_identity_context(self) -> tuple[str, Optional[str]]:
        """
        Stage 7: Get identity/personality context from user profile.

        Returns:
            Tuple of (identity_block, user_name)
        """
        if not self.user_profile:
            return "", None

        try:
            identity = getattr(self.user_profile, 'identity', None)
            if not identity:
                return "", None

            user_name = getattr(identity, 'name', None)
            pronouns = getattr(identity, 'pronouns', 'they/them')

            # Build identity block for system prompt
            identity_parts = []
            if user_name:
                identity_parts.append(f"The user's name is {user_name}.")
            if pronouns:
                identity_parts.append(f"Their pronouns are {pronouns}.")

            identity_block = " ".join(identity_parts)
            return identity_block, user_name

        except Exception as e:
            logger.warning(f"Identity context retrieval failed: {e}")
            return "", None

    async def _get_thread_context(self) -> Optional[Dict[str, Any]]:
        """
        Stage 8: Get active thread context.

        Delegates to memory_system.get_thread_context()

        Returns:
            Dict with thread_id, thread_depth, thread_started, thread_topic, is_heavy_topic
        """
        if not self.memory_system:
            return None

        try:
            if hasattr(self.memory_system, 'get_thread_context'):
                return self.memory_system.get_thread_context()
            return None
        except Exception as e:
            logger.warning(f"Thread context retrieval failed: {e}")
            return None

    def _get_tone_instructions(self, tone_level: ToneLevel) -> str:
        """Get response instructions for the detected tone level."""
        return self._tone_instructions.get(tone_level, "")

    # --- Tone Instruction Loaders ---

    def _load_crisis_instructions(self) -> str:
        return """[CRISIS SUPPORT MODE]
You are now in full therapeutic response mode. The user may be experiencing significant emotional distress.

Response Guidelines:
- Provide multi-paragraph empathetic validation
- Acknowledge their pain directly and specifically
- Use warm, supportive language throughout
- Offer relevant crisis resources if appropriate
- Prioritize safety and emotional connection
- Do NOT rush to solutions - focus on being present
- Mirror their emotional intensity appropriately

Remember: Your role is to be a supportive presence, not to fix everything immediately."""

    def _load_elevated_instructions(self) -> str:
        return """[ELEVATED SUPPORT MODE]
The user appears to be experiencing moderate emotional distress.

Response Guidelines:
- Provide 2-3 paragraphs of empathetic response
- Acknowledge and validate their feelings explicitly
- Balance emotional support with gentle guidance
- Ask clarifying questions if helpful
- Offer practical suggestions only after validation

Remember: Lead with empathy before offering solutions."""

    def _load_concern_instructions(self) -> str:
        return """[LIGHT SUPPORT MODE]
The user may have some emotional undertones in their message.

Response Guidelines:
- Keep responses to 2-4 sentences
- Brief acknowledgment of any feelings present
- Maintain a warm but practical focus
- Provide helpful information directly
- Don't over-emphasize emotional aspects

Remember: Be helpful and warm without being excessive."""

    def _load_conversational_instructions(self) -> str:
        return """[CONVERSATIONAL MODE]
Standard conversational interaction.

Response Guidelines:
- Keep responses concise (max 3 sentences for simple queries)
- Be direct and helpful
- Match the user's energy and tone
- No unnecessary emotional validation
- Focus on providing clear, useful information

Remember: Be efficient and natural in your responses."""

    # --- Utility Methods ---

    def reset_conversation_depth(self) -> None:
        """Reset conversation depth counter (call when starting new conversation)."""
        self._conversation_depth = 0

    def get_conversation_depth(self) -> int:
        """Get current conversation depth."""
        return self._conversation_depth
