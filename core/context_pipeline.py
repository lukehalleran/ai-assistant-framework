"""
Context Pipeline - Builder pattern for prompt preparation.

Purpose: Transform raw user input into fully processed context ready for prompt building.
Inputs: User query, optional files, configuration flags
Outputs: ContextResult with all context components
Side effects: May call LLM for tone detection, query rewriting, STM analysis

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

from config.app_config import (
    USE_STM_PASS,
    STM_MIN_CONVERSATION_DEPTH,
    STM_MAX_RECENT_MESSAGES,
    REWRITE_TIMEOUT_S,
    INTENT_ENABLED,
)
from core.intent_classifier import IntentClassifier, IntentResult, IntentType

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
        """Convert string crisis level to ToneLevel enum."""
        level_map = {
            "HIGH": cls.CRISIS,
            "MEDIUM": cls.ELEVATED,
            "CONCERN": cls.CONCERN,
            "CONVERSATIONAL": cls.CONVERSATIONAL,
        }
        return level_map.get(level, cls.CONVERSATIONAL)


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

        # Stage 1: Topic Extraction
        primary_topic, topics = await self._extract_topics(user_input)
        logger.debug(f"Stage 1 (Topics): primary={primary_topic}, all={topics}")

        # Stage 2: Tone Detection (with conversation history for context)
        if not use_raw_mode:
            tone_level, emotional_context = await self._detect_tone(
                user_input,
                conversation_history
            )
            logger.debug(f"Stage 2 (Tone): level={tone_level.value}")

        # Stage 3: File Processing
        if files and not use_raw_mode:
            file_context = await self._process_files(user_input, files)
            if file_context and file_context != user_input:
                # Files were processed, update processed_query
                processed_query = file_context
                logger.debug(f"Stage 3 (Files): processed {len(files)} files")

        # Stage 4: Heavy Topic Check + Inline Fact Extraction
        if not use_raw_mode:
            is_heavy_topic, extracted_facts, query_analysis = await self._check_heavy_topics(
                user_input,
                topics
            )
            if is_heavy_topic:
                logger.debug(f"Stage 4 (Heavy Topic): detected, {len(extracted_facts)} facts extracted")

        # Stage 4.5: Intent Classification (regex-first, no LLM)
        is_small_talk = False
        if not use_raw_mode and self._intent_classifier:
            intent_result = self._intent_classifier.classify(
                user_input,
                tone_level=tone_level.value,
            )
            logger.debug(
                f"Stage 4.5 (Intent): {intent_result.intent.value} "
                f"(conf={intent_result.confidence:.2f})"
            )
            # Flag casual social messages to skip expensive downstream calls
            from core.intent_classifier import IntentType
            if intent_result.intent == IntentType.CASUAL_SOCIAL and intent_result.confidence >= 0.70:
                is_small_talk = True
                logger.debug("Stage 4.5: is_small_talk=True (CASUAL_SOCIAL, high confidence)")

        # Stages 5+6: Query Rewriting + STM Analysis (parallelized — independent LLM calls)
        run_rewrite = not use_raw_mode and self._enable_query_rewrite
        run_stm = not use_raw_mode and self._should_run_stm()

        if run_rewrite and run_stm:
            # Both needed — run in parallel for ~1-2s savings
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
            rewritten = await self._rewrite_query(user_input, query_analysis)
            if rewritten and rewritten != user_input:
                processed_query = rewritten
                logger.debug("Stage 5 (Rewrite): query rewritten")

        elif run_stm:
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
                intent_result, stm_intent_str
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
            thread_context=thread_context,
            stm_summary=stm_summary,
            identity_block=identity_block,
            user_name=user_name,
            is_heavy_topic=is_heavy_topic,
            extracted_facts=extracted_facts,
            query_analysis=query_analysis,
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

    async def _extract_topics(self, query: str) -> tuple[Optional[str], List[str]]:
        """
        Stage 1: Extract topics via TopicManager.

        Returns:
            Tuple of (primary_topic, list_of_topics)
        """
        if not self.topic_manager:
            return None, []

        try:
            # Get primary topic (also updates internal state + has LLM cache)
            primary = self.topic_manager.get_primary_topic(query)

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

            # Analyze emotional context
            emotional_ctx = await analyze_emotional_context(
                message=query,
                conversation_history=recent_memories,
                model_manager=self.model_manager
            )

            # Convert crisis level to ToneLevel
            if emotional_ctx and hasattr(emotional_ctx, 'crisis_level'):
                level_str = emotional_ctx.crisis_level.value if hasattr(emotional_ctx.crisis_level, 'value') else str(emotional_ctx.crisis_level)
                tone_level = ToneLevel.from_string(level_str)
            else:
                tone_level = ToneLevel.CONVERSATIONAL

            return tone_level, emotional_ctx

        except ImportError:
            logger.warning("emotional_context module not available, using CONVERSATIONAL")
            return ToneLevel.CONVERSATIONAL, None
        except Exception as e:
            logger.warning(f"Tone detection failed: {e}")
            return ToneLevel.CONVERSATIONAL, None

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
                    model="gpt-4o-mini",
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

    def _should_run_stm(self) -> bool:
        """Check if STM analysis should run based on config and conversation depth."""
        if not self._use_stm:
            return False
        if not self.stm_analyzer:
            return False
        return self._conversation_depth >= self._stm_min_depth

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

            # Get last assistant response
            last_response = None
            if recent_memories:
                for mem in recent_memories:
                    if mem.get('role') == 'assistant':
                        last_response = mem.get('content', mem.get('response'))
                        break

            result = await self.stm_analyzer.analyze(
                recent_memories=recent_memories,
                user_query=query,
                last_assistant_response=last_response
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
