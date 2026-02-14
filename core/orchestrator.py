"""
# core/orchestrator.py

Module Contract
- Purpose: High‑level request orchestrator. Prepares prompts (topic detection, optional file processing, query rewrite), invokes the model, and persists the interaction to memory.
- Inputs:
  - user_input: str; optional files; mode flags (raw/enhanced)
  - Wired collaborators: model_manager, response_generator, file_processor, prompt_builder, memory_system, personality/topic/wiki/tokenizer managers
- Outputs:
  - process_user_query() → (assistant_text: str, debug_info: dict)
  - prepare_prompt() → (prompt_text: str, system_prompt: Optional[str])
- Key methods:
  - handle_commands(): simple topic switching commands
  - prepare_prompt(): topic update, file processing, optional query rewrite, resolve system prompt, build prompt via prompt_builder
  - process_user_query(): personality hook, deictic check, generate streamed response, store memory, schedule consolidation (now at shutdown)
  - _check_narrative_freshness(): Startup check for stale narrative context (>24h) [NEW 2026-01-17]
- System prompt flow:
  - Resolved via config/app_config.load_system_prompt and/or path override; forwarded to response generation as system role message.
- Side effects:
  - Writes conversation+metadata to corpus DB/Chroma via memory_system; logs events.
  - Logs warning if narrative context is stale on startup [NEW 2026-01-17]
- Async behavior:
  - Uses async streaming from response_generator; overall methods are async.

Additional Contract (Agentic Search):
  - process_user_query() accepts use_agentic_search parameter
  - When agentic=True, routes through AgenticSearchController for multi-round search
  - Emits ProgressEvent for UI status updates during agentic flow
  - Falls back to standard flow on agentic failures
  - Agentic mode uses LLM-first trigger's search_terms for initial search
  - Initializes SandboxManager for E2B code execution support [NEW 2026-01-22]
"""
import asyncio
import os
import re
import processing.gate_system as gate_system
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union
from core.response_parser import ResponseParser
from utils.logging_utils import get_logger
from integrations.wikipedia_api import WikipediaAPI
from utils.tone_detector import (
    detect_crisis_level,
    CrisisLevel,
    should_log_tone_shift,
    format_tone_shift_log,
    format_tone_log,
)
from utils.emotional_context import (
    EmotionalContext,
    analyze_emotional_context,
    format_emotional_context_log,
)
from utils.need_detector import NeedType
from core.context_pipeline import ContextPipeline, ContextResult, ToneLevel
from core.best_of_handler import BestOfHandler, BestOfResult
from core.escalation_tracker import EscalationTracker, ResponseStrategy

SYSTEM_PROMPT = "..."  # safe fallback (replace with your real default)
wiki_api = WikipediaAPI()
gate_system.wikipedia_api = wiki_api  # This sets it globally
# If you have a real helper, import that instead:
from utils.query_checker import is_deictic, analyze_query


class _SimplePromptBuilder:
    """Fallback prompt builder used when the unified builder is unavailable."""

    async def build_prompt(self, user_input: str, **_: Any) -> str:
        return user_input or ""


class _InMemoryCorpus:
    """Minimal corpus manager used by fallback memory coordinator."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []

    def add_entry(self, query: str, response: str, tags: Optional[List[str]] = None) -> None:
        self._entries.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now(),
            "tags": tags or [],
        })

    def get_recent_memories(self, limit: int) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        return self._entries[-limit:]

    def get_summaries(self, _limit: int) -> List[Dict[str, Any]]:
        return []


class _FallbackMemoryCoordinator:
    """Extremely small in-memory memory system for offline testing."""

    def __init__(self) -> None:
        self.corpus_manager = _InMemoryCorpus()
        self.gate_system = None

    async def store_interaction(self, query: str, response: str, tags: Optional[List[str]] = None) -> None:
        self.corpus_manager.add_entry(query, response, tags)

    async def get_memories(self, _query: str, limit: int = 10) -> List[Dict[str, Any]]:
        recent = list(reversed(self.corpus_manager.get_recent_memories(limit)))
        return [
            {
                "query": item.get("query", ""),
                "response": item.get("response", ""),
                "metadata": {"source": "recent", "final_score": 1.0},
            }
            for item in recent
        ]

    async def retrieve_relevant_memories(self, _query: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        limit = (config or {}).get("recent_count", 5)
        recent = list(reversed(self.corpus_manager.get_recent_memories(limit)))
        memories = [
            {
                "query": item.get("query", ""),
                "response": item.get("response", ""),
                "source": "recent",
                "final_score": 1.0,
            }
            for item in recent
        ]
        counts = {
            "recent": len(memories),
            "semantic": 0,
            "hierarchical": 0,
        }
        return {"memories": memories, "counts": counts}

class DaemonOrchestrator:
    """
    Single orchestrator (prepare + generate split).
    - prepare_prompt: topic update, file processing, optional rewrite, prompt build
    - process_user_query: optional personality switch, commands, deictic check, generate, store
    """

    def __init__(
        self,
        *,
        model_manager,
        response_generator,
        file_processor,
        prompt_builder,
        memory_system=None,
        personality_manager=None,
        topic_manager=None,
        wiki_manager=None,
        tokenizer_manager=None,
        conversation_logger=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.logger = get_logger("orchestrator")
        self.conversation_logger = conversation_logger  # kept for compatibility if referenced elsewhere
        self.model_manager = model_manager
        self.response_generator = response_generator
        self.file_processor = file_processor
        self.prompt_builder = prompt_builder
        self.memory_system = memory_system
        self.personality_manager = personality_manager
        self.topic_manager = topic_manager

        # Extract time_manager from response_generator (shared instance)
        self.time_manager = getattr(response_generator, 'time_manager', None)
        # inside __init__ after self.topic_manager = topic_manager
        try:
            if self.topic_manager and hasattr(gate_system, "set_topic_resolver"):
                # gate_system will now call TopicManager to turn “tell me about …” into a clean title
                gate_system.set_topic_resolver(self.topic_manager.get_primary_topic)
                if self.logger:
                    self.logger.debug("[orchestrator] Topic resolver registered with gate_system")
        except Exception as e:
            if self.logger:
                self.logger.debug(f"[orchestrator] Could not register topic resolver: {e}")

        self.wiki_manager = wiki_manager
        self.tokenizer_manager = tokenizer_manager

        # Use a single logger field throughout (can be stdlib logger or your own)


        self.config = config or {}
        self.current_topic = "general"
        self.topic_confidence_threshold = float(self.config.get("topic_confidence_threshold", 0.7))
        self.system_prompt_path = self.config.get("system_prompt_path")

        # Load user profile for identity and preferences
        try:
            from memory.user_profile import UserProfile
            self.user_profile = UserProfile()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[Orchestrator] Failed to load user profile: {e}")
            self.user_profile = None

        # Emotional context tracking (tone + need type)
        self.current_tone_level: Optional[CrisisLevel] = None  # Keep for compatibility
        self.current_emotional_context: Optional[EmotionalContext] = None

        # Agentic search controller (lazy initialization)
        self._agentic_controller = None
        self._agentic_config = self.config.get("agentic_search", {}) if self.config else {}

        # STM (Short-Term Memory) analyzer for multi-pass context summarization
        self.stm_analyzer = None
        self.last_stm_topic = None  # Track last topic for change detection
        try:
            from config.app_config import USE_STM_PASS, STM_MODEL_NAME, STM_MIN_CONVERSATION_DEPTH
            if USE_STM_PASS and model_manager:
                from core.stm_analyzer import STMAnalyzer
                self.stm_analyzer = STMAnalyzer(
                    model_manager=model_manager,
                    model_name=STM_MODEL_NAME
                )
                self.stm_min_depth = STM_MIN_CONVERSATION_DEPTH
                self.logger.info(f"[Orchestrator] STM analyzer enabled (model={STM_MODEL_NAME}, min_depth={STM_MIN_CONVERSATION_DEPTH})")
            else:
                self.logger.debug("[Orchestrator] STM analyzer disabled via config")
        except Exception as e:
            self.logger.warning(f"[Orchestrator] Failed to initialize STM analyzer: {e}")

        # Context Pipeline - Builder pattern for query analysis (pre-retrieval)
        # Handles: tone detection, topic extraction, file processing, heavy topic check,
        # query rewriting, STM analysis, identity injection, thread context
        self.context_pipeline = None
        try:
            self.context_pipeline = ContextPipeline(
                model_manager=model_manager,
                topic_manager=topic_manager,
                file_processor=file_processor,
                stm_analyzer=self.stm_analyzer,
                user_profile=self.user_profile,
                memory_system=memory_system,
                config={
                    "USE_STM_PASS": self.config.get("features", {}).get("use_stm_pass", True) if self.config else True,
                    "STM_MIN_CONVERSATION_DEPTH": getattr(self, 'stm_min_depth', 3),
                    "enable_query_rewrite": self.config.get("features", {}).get("enable_query_rewrite", True) if self.config else True,
                    "REWRITE_TIMEOUT_S": self.config.get("REWRITE_TIMEOUT_S", 2.0) if self.config else 2.0,
                }
            )
            self.logger.info("[Orchestrator] ContextPipeline initialized")
        except Exception as e:
            self.logger.warning(f"[Orchestrator] Failed to initialize ContextPipeline: {e}")

        # Best-of Handler - Orchestrates best-of-N, ensemble, and duel mode generation
        self.best_of_handler = BestOfHandler(
            response_generator=response_generator,
            config=self.config
        )
        self.logger.debug("[Orchestrator] BestOfHandler initialized")

        # Escalation Tracker - Session-level emotional momentum tracking
        self.escalation_tracker = None
        try:
            from config.app_config import (
                ESCALATION_ENABLED,
                ESCALATION_THRESHOLD,
                ESCALATION_DEESCALATION_WINDOW,
                ESCALATION_MAX_HISTORY,
            )
            if ESCALATION_ENABLED:
                self.escalation_tracker = EscalationTracker(
                    escalation_threshold=ESCALATION_THRESHOLD,
                    deescalation_window=ESCALATION_DEESCALATION_WINDOW,
                    max_history=ESCALATION_MAX_HISTORY,
                )
                self.logger.info(
                    f"[Orchestrator] EscalationTracker enabled "
                    f"(threshold={ESCALATION_THRESHOLD}, "
                    f"deesc_window={ESCALATION_DEESCALATION_WINDOW})"
                )
            else:
                self.logger.debug("[Orchestrator] EscalationTracker disabled via config")
        except Exception as e:
            self.logger.warning(f"[Orchestrator] Failed to initialize EscalationTracker: {e}")

        # Memory citation system
        self.enable_citations = False  # Will be set from GUI checkbox
        # Pattern matches citation formats: MEM_RECENT_3, MEM_SEMANTIC_4-7, SUM_RECENT_1, REFL_SEMANTIC_2, PROFILE_CONTEXT
        self.citation_pattern = re.compile(
            r'\[('
            r'MEM_\w+_\d+(?:-\d+)?|'      # MEM_RECENT_3, MEM_SEMANTIC_4-7
            r'SUM_\w+_\d+(?:-\d+)?|'      # SUM_RECENT_1, SUM_SEMANTIC_2-5
            r'REFL_\w+_\d+(?:-\d+)?|'     # REFL_RECENT_1, REFL_SEMANTIC_3
            r'FACT_\d+(?:-\d+)?|'         # FACT_3 (legacy)
            r'PROFILE_\w+'                # PROFILE_CONTEXT
            r')\]'
        )

        # Check narrative context freshness on startup (non-blocking)
        self._check_narrative_freshness()

    def _check_narrative_freshness(self) -> None:
        """
        Check if narrative context exists and is reasonably fresh.

        If stale (>24 hours) or missing, logs info about it.
        Actual refresh happens when daily notes are generated.
        """
        try:
            from config.app_config import NARRATIVE_CONTEXT_ENABLED, NARRATIVE_CONTEXT_PATH
            from pathlib import Path
            from datetime import datetime, timedelta

            if not NARRATIVE_CONTEXT_ENABLED:
                return

            narrative_path = Path(NARRATIVE_CONTEXT_PATH)

            if not narrative_path.exists():
                self.logger.debug("[Orchestrator] No narrative context file found (will be created when daily notes exist)")
                return

            # Check freshness (stale if older than 24 hours)
            mtime = datetime.fromtimestamp(narrative_path.stat().st_mtime)
            age = datetime.now() - mtime
            stale_threshold = timedelta(hours=24)

            if age > stale_threshold:
                age_str = f"{age.days}d {age.seconds // 3600}h" if age.days > 0 else f"{age.seconds // 3600}h"
                self.logger.info(
                    f"[Orchestrator] Narrative context is {age_str} old. "
                    f"Will refresh when next daily note is generated, or run 'python main.py refresh-narrative'."
                )
            else:
                age_str = f"{age.seconds // 3600}h {(age.seconds % 3600) // 60}m"
                self.logger.debug(f"[Orchestrator] Narrative context is fresh ({age_str} old)")

        except Exception as e:
            self.logger.debug(f"[Orchestrator] Narrative freshness check skipped: {e}")

    # ---------- STM Helper Methods ----------
    def _compute_topic_similarity(self, topic1: str, topic2: str) -> float:
        """
        Compute semantic similarity between two topic strings using embeddings.

        Returns a float between 0.0 (unrelated) and 1.0 (identical).
        Falls back to simple string matching if embeddings unavailable.
        """
        if not topic1 or not topic2:
            return 0.0

        # Normalize
        t1 = topic1.lower().strip()
        t2 = topic2.lower().strip()

        # Exact match shortcut
        if t1 == t2:
            return 1.0

        # Try to get embed_model from prompt_builder's gate_system
        try:
            embed_model = None
            if hasattr(self, 'prompt_builder') and self.prompt_builder:
                gate_sys = getattr(self.prompt_builder, 'gate_system', None)
                if gate_sys:
                    embed_model = getattr(gate_sys, 'embed_model', None)

            if embed_model is not None:
                import numpy as np
                from sklearn.metrics.pairwise import cosine_similarity

                # Encode both topics
                emb1 = embed_model.encode([t1], convert_to_numpy=True, normalize_embeddings=True)
                emb2 = embed_model.encode([t2], convert_to_numpy=True, normalize_embeddings=True)

                # Compute cosine similarity
                sim = cosine_similarity(emb1, emb2)[0][0]
                return float(sim)
        except Exception as e:
            self.logger.debug(f"[STM] Embedding similarity failed, using fallback: {e}")

        # Fallback: simple word overlap (Jaccard-like)
        words1 = set(t1.split())
        words2 = set(t2.split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _should_use_stm(self, conversation_history: Optional[List[Dict]], user_input: str) -> bool:
        """
        Determine if STM pass should be used for this query.

        Skips STM for:
        - Meta-conversational queries (asking about the conversation itself)
        - Topic changes (detected via topic_manager)
        - Very short/trivial queries
        - Insufficient conversation depth

        Args:
            conversation_history: Recent conversation turns
            user_input: Current user query

        Returns:
            True if STM should be used, False otherwise
        """
        if not self.stm_analyzer:
            return False

        # Skip for very short, trivial queries
        if len(user_input.strip()) < 10:
            return False

        # Require minimum conversation depth
        min_depth = getattr(self, 'stm_min_depth', 3)
        if not conversation_history or len(conversation_history) < min_depth:
            return False

        # Skip for meta-conversational queries (asking about the conversation itself)
        try:
            from utils.query_checker import is_meta_conversational
            if is_meta_conversational(user_input):
                self.logger.debug("[STM] Skipping STM for meta-conversational query")
                return False
        except Exception as e:
            self.logger.warning(f"[STM] Failed to check meta-conversational status: {e}")

        # Detect topic changes using semantic similarity (not string matching)
        # This prevents garbage topic extraction from causing constant "topic changes"
        try:
            from config.app_config import STM_TOPIC_SIMILARITY_THRESHOLD

            if self.topic_manager:
                # Get current topic from topic manager
                self.topic_manager.update_from_user_input(user_input)
                current_topic = self.topic_manager.get_primary_topic()

                # If we have both topics, check semantic similarity
                if current_topic and self.last_stm_topic and current_topic.lower() != "general":
                    similarity = self._compute_topic_similarity(current_topic, self.last_stm_topic)

                    # Only skip STM if topics are truly unrelated (below threshold)
                    if similarity < STM_TOPIC_SIMILARITY_THRESHOLD:
                        self.logger.info(
                            f"[STM] Major topic shift detected: '{self.last_stm_topic}' -> '{current_topic}' "
                            f"(similarity={similarity:.2f} < {STM_TOPIC_SIMILARITY_THRESHOLD}). "
                            "Skipping STM to prevent contamination."
                        )
                        self.last_stm_topic = current_topic
                        return False  # Skip STM on major topic transitions
                    else:
                        self.logger.debug(
                            f"[STM] Topics similar enough: '{self.last_stm_topic}' -> '{current_topic}' "
                            f"(similarity={similarity:.2f}). STM will run."
                        )

                # Update topic tracking
                if current_topic:
                    self.last_stm_topic = current_topic

        except Exception as e:
            self.logger.warning(f"[STM] Failed to check topic change: {e}")

        return True

    # ---------- 1) Tone Mode Instructions ----------
    def _get_tone_instructions(self, tone_level: CrisisLevel) -> str:
        """
        Return mode-specific response instructions based on detected crisis level.

        Args:
            tone_level: Detected crisis level from tone_detector

        Returns:
            String containing tone-specific instructions to append to system prompt
        """
        # Inject style modifier BEFORE tone instructions (unless in HIGH crisis mode)
        style_modifier = ""
        if tone_level != CrisisLevel.HIGH:
            try:
                profile = getattr(self, 'user_profile', None)
                if profile:
                    style_modifier = profile.get_style_modifier()
            except (AttributeError, TypeError) as e:
                logger.debug(f"[Orchestrator] Could not get style modifier: {e}")
                style_modifier = ""

        if tone_level == CrisisLevel.HIGH:
            # CRISIS_SUPPORT: Full therapeutic mode for genuine crisis
            return (
                "\n\n## RESPONSE MODE: CRISIS SUPPORT\n"
                "The user is experiencing a severe crisis or mental health emergency. "
                "Respond with full therapeutic presence:\n"
                "- Acknowledge the severity of their feelings with empathy and care\n"
                "- Validate their experience without minimizing or rushing to solutions\n"
                "- Multiple paragraphs are appropriate to show you're truly engaged\n"
                "- Offer concrete support resources when relevant (crisis lines, professional help)\n"
                "- Avoid platitudes like \"you've got this\" - focus on genuine connection\n"
                "- Stay present with their pain rather than trying to immediately fix it\n"
                "- Encourage professional help or crisis intervention if appropriate"
            )
        elif tone_level == CrisisLevel.MEDIUM:
            # ELEVATED_SUPPORT: Supportive but measured for acute distress
            base_instructions = (
                "\n\n## RESPONSE MODE: ELEVATED SUPPORT\n"
                "The user is experiencing acute distress or emotional difficulty. "
                "Respond with supportive care:\n"
                "- 2-3 paragraphs maximum - be supportive but not overwhelming\n"
                "- Validate their feelings and acknowledge the difficulty\n"
                "- Offer perspective or gentle suggestions if appropriate\n"
                "- Be warm and empathetic, but don't over-therapize\n"
                "- Focus on their specific situation, not generic coping advice"
            )
            return style_modifier + base_instructions if style_modifier else base_instructions
        elif tone_level == CrisisLevel.CONCERN:
            # LIGHT_SUPPORT: Brief validation for moderate concern
            base_instructions = (
                "\n\n## RESPONSE MODE: LIGHT SUPPORT\n"
                "The user is expressing concern, anxiety, or stress about something. "
                "Respond with brief, grounded validation:\n"
                "- 2-4 sentences - acknowledge without expanding unnecessarily\n"
                "- \"That sucks\" + brief validation is often sufficient\n"
                "- Don't offer unsolicited advice or try to solve their problem\n"
                "- Match their energy - if they're venting, let them vent\n"
                "- Only expand if they explicitly ask for more"
            )
            return style_modifier + base_instructions if style_modifier else base_instructions
        else:  # CrisisLevel.CONVERSATIONAL
            # CONVERSATIONAL: Natural friend voice - most interactions
            base_instructions = (
                "\n\n## RESPONSE MODE: CONVERSATIONAL (Natural Friend Voice)\n"
                "**Goal: Be a confident, grounded friend - not a service agent, hype-man, or therapist**\n\n"
                "**LENGTH: 2-5 sentences typically - brief but substantive. Don't mirror the user's brevity; give complete thoughts.**\n\n"
                "Voice Calibration - The Sweet Spot:\n"
                "❌ TOO HYPE: \"Hell yeah you're crushing it king! 💪 Die mad haters!\"\n"
                "❌ TOO SERVICE-Y: \"Thanks so much, that's awesome to hear! I'm here anytime!\"\n"
                "❌ TOO TERSE: \"Cool.\" (1 word when 2-3 sentences would be more natural)\n"
                "✓ JUST RIGHT: \"Ha, glad that's better. The refactor sounds solid — should make the codebase way easier to navigate. Hope work goes smooth.\"\n\n"
                "Core Principles:\n"
                "1. **Match his energy** - casual when he's casual, focused when he's technical\n"
                "   - ✓ \"Nice work on that.\" / \"Solid progress.\"\n"
                "   - ✗ \"Thanks so much, that's awesome to hear! Always striving to provide value!\"\n\n"
                "2. **Substantive but natural** - even brief topics deserve complete thoughts\n"
                "   - ✓ \"Glad that worked. The edge case handling looks solid now.\"\n"
                "   - ✓ \"Makes sense. That approach should avoid the race condition you were seeing.\"\n"
                "   - ✗ \"Cool.\" (too terse)\n"
                "   - ✗ \"Thank you for your feedback! I'm excited to keep improving!\"\n\n"
                "3. **Confident friend** - comfortable in the relationship, not anxious to please\n"
                "   - ✓ \"That's frustrating, but you handled it well.\"\n"
                "   - ✗ \"Please let me know if there's anything else I can help with!\"\n\n"
                "4. **Helpful without being formal** - offer perspective naturally\n"
                "   - ✓ \"Makes sense you're annoyed. Moving on sounds right.\"\n"
                "   - ✗ \"I appreciate your patience as I strive to deliver relevant information.\"\n\n"
                "FORBIDDEN:\n"
                "- Customer service language (\"Thanks so much!\", \"I'm here anytime!\", \"Please let me know\")\n"
                "- Emojis and excessive exclamation points\n"
                "- Hype language (crushing it, you got this king, Hell yeah)\n"
                "- Corporate speak (\"striving to provide\", \"relevant and valuable info\")\n"
                "- Excessive gratitude or validation-seeking\n"
                "- Being overly terse (1-2 words when a full sentence would be natural)\n"
                "- Multi-paragraph responses for simple acknowledgments\n\n"
                "Think: You're the friend who knows the relationship is solid, so you don't need "
                "to constantly prove your value or seek approval. Be supportive, honest, and chill."
            )
            return style_modifier + base_instructions if style_modifier else base_instructions

    def _get_response_instructions(self, ctx: EmotionalContext) -> str:
        """
        Generate response instructions based on combined emotional context.

        Matrix:
        - HIGH + any need → Full crisis support (existing)
        - MEDIUM + PRESENCE → Warmth first, then measured support
        - MEDIUM + PERSPECTIVE → Supportive engagement
        - CONCERN + PRESENCE → Brief acknowledgment, warmth, stay present
        - CONCERN + PERSPECTIVE → Light engagement, offer reframe
        - CONVERSATIONAL + any → Default casual mode

        Args:
            ctx: EmotionalContext with crisis level and need type

        Returns:
            String containing response instructions to append to system prompt
        """

        # Crisis levels override need-type (safety first)
        if ctx.crisis_level == CrisisLevel.HIGH:
            return self._get_tone_instructions(CrisisLevel.HIGH)

        # Combined instructions for non-crisis
        base = self._get_tone_instructions(ctx.crisis_level)

        if ctx.need_type == NeedType.PRESENCE:
            presence_addon = """

## PRESENCE MODE: User needs warmth and acknowledgment
The user is expressing emotional state, not seeking analysis.
- Lead with warmth and acknowledgment
- Stay with them before offering perspective
- Short, warm responses preferred
- Avoid immediate problem-solving or reframes
- "I hear you" before "here's what I think"
"""
            return base + presence_addon

        elif ctx.need_type == NeedType.PERSPECTIVE:
            perspective_addon = """

## PERSPECTIVE MODE: User is open to engagement
The user is processing/analyzing, open to engagement.
- Engage with their framing
- Questions and reframes welcome
- Can offer alternative viewpoints
- Problem-solving appropriate if relevant
"""
            return base + perspective_addon

        return base  # NEUTRAL - use base tone instructions only

    # ---------- 1b) Session Headers Instructions ----------
    def _get_session_headers_instructions(self) -> str:
        """
        Return concise instructions about temporal reasoning with prompt headers.

        Returns:
            String containing session header guidance to append to system prompt
        """
        return (
            "\n\n## TEMPORAL REASONING\n"
            "**[TIME CONTEXT]**: Contains current datetime AND conversation pacing metrics.\n"
            "- **Current time**: Use to calculate elapsed time from memory timestamps\n"
            "- **Time since last message**: Gap between consecutive messages in this session\n"
            "  - Quick replies (seconds) = active engagement, possible urgency\n"
            "  - Long pauses (minutes/hours) = user returned after break, may need context refresh\n"
            "  - 'N/A (first message in session)' = session just started\n"
            "- **Time since last session**: Gap between previous session end and now\n"
            "  - Hours/days = acknowledge the gap (\"welcome back\", \"it's been a while\")\n"
            "  - Seconds/minutes = continuation, no greeting needed\n"
            "  - 'N/A (first session)' = first time ever using the system\n"
            "\n"
            "**Use pacing metrics for appropriate responses:**\n"
            "- Rapid messages (few seconds apart) → Keep responses concise, match their pace\n"
            "- Long gaps (hours/days since last session) → Warmer greeting, summarize where we left off\n"
            "- First message in session after break → Natural to acknowledge return\n"
            "- Mid-session messages → No need for greetings, continue conversation naturally\n"
            "\n"
            "**All sections below contain timestamps - use them for temporal reasoning:**\n"
            "- **[RECENT CONVERSATION]**: Last exchanges with timestamps, newest first\n"
            "- **[RELEVANT MEMORIES]**: Semantically relevant past conversations with timestamps\n"
            "- **[USER PROFILE]**: User facts (hybrid: semantic + recent) with timestamps in [ISO format] brackets after each fact\n"
            "- **[SUMMARIES]**: Conversation summaries with timestamps\n"
            "- **[RECENT REFLECTIONS]**: Meta-reflections with timestamps\n"
            "\n"
            "**Calculate recency from timestamps, not relative terms:**\n"
            "- Memory from 2025-09-15, Current time 2025-11-17 = \"2 months ago\" (not \"recently\")\n"
            "- Use item's own timestamp when available, not just Current time\n"
            "- \"2 months ago\" (explicit) > \"recently\" (vague)\n"
            "- For sleep/schedule: compare timestamps across memories\n"
            "\n"
            "**[CURRENT USER QUERY]**: The only query to respond to. All other sections are context only."
        )

    # ---------- 2) Agentic Search Controller ----------
    @property
    def agentic_controller(self):
        """
        Lazy-initialize the agentic search controller.

        Returns:
            AgenticSearchController instance or None if not available
        """
        if self._agentic_controller is not None:
            return self._agentic_controller

        # Check if agentic search is enabled
        if not self._agentic_config.get("enabled", False):
            return None

        try:
            from core.agentic import AgenticSearchController

            # Get web search manager from prompt builder
            web_search_manager = None
            if hasattr(self.prompt_builder, 'context_gatherer'):
                web_search_manager = getattr(
                    self.prompt_builder.context_gatherer,
                    'web_search_manager',
                    None
                )

            if not web_search_manager:
                if self.logger:
                    self.logger.warning("[Orchestrator] Agentic search disabled: no web_search_manager")
                return None

            # Get token manager if available
            token_manager = None
            if hasattr(self.prompt_builder, 'token_manager'):
                token_manager = self.prompt_builder.token_manager

            # Initialize Wolfram Alpha manager if configured
            wolfram_manager = None
            try:
                from config.app_config import WOLFRAM_ENABLED, WOLFRAM_APP_ID
                if WOLFRAM_ENABLED and WOLFRAM_APP_ID:
                    from knowledge.wolfram_manager import WolframManager
                    wolfram_manager = WolframManager()
                    if self.logger:
                        self.logger.info("[Orchestrator] Wolfram Alpha manager initialized")
            except ImportError as e:
                if self.logger:
                    self.logger.debug(f"[Orchestrator] Wolfram Alpha not available: {e}")

            # Initialize sandbox manager for code execution
            sandbox_manager = None
            try:
                from knowledge.sandbox_manager import get_sandbox_manager
                sandbox_manager = get_sandbox_manager()
                if sandbox_manager.is_available():
                    if self.logger:
                        self.logger.info("[Orchestrator] Sandbox manager initialized")
                else:
                    sandbox_manager = None
                    if self.logger:
                        self.logger.debug("[Orchestrator] Sandbox not available (E2B_API_KEY not configured)")
            except ImportError as e:
                if self.logger:
                    self.logger.debug(f"[Orchestrator] Sandbox manager not available: {e}")

            self._agentic_controller = AgenticSearchController(
                model_manager=self.model_manager,
                web_search_manager=web_search_manager,
                wolfram_manager=wolfram_manager,
                sandbox_manager=sandbox_manager,
                token_manager=token_manager,
                max_rounds=self._agentic_config.get("max_rounds", 5),
                context_budget_tokens=self._agentic_config.get("context_budget_tokens", 8000),
                compression_model=self._agentic_config.get("compression_model", "gpt-4o-mini"),
            )

            if self.logger:
                self.logger.info("[Orchestrator] Agentic search controller initialized")

            return self._agentic_controller

        except Exception as e:
            if self.logger:
                self.logger.error(f"[Orchestrator] Failed to initialize agentic controller: {e}")
            return None

    def _should_use_agentic_search(
        self,
        user_input: str,
        web_search_decision: Any = None,
        use_agentic_search: bool = False
    ) -> bool:
        """
        Determine if agentic search should be used for this query.

        Args:
            user_input: The user's query
            web_search_decision: WebSearchDecision from LLM-first trigger
            use_agentic_search: Explicit flag from caller

        Returns:
            True if agentic search should be used
        """
        # Must be explicitly enabled
        if not use_agentic_search:
            return False

        # Must have agentic controller available
        if self.agentic_controller is None:
            return False

        # Must have a web search decision that says to search
        if web_search_decision is None:
            return False

        if hasattr(web_search_decision, 'should_search') and not web_search_decision.should_search:
            return False

        return True

    # ---------- 3) Commands & Topic ----------
    def handle_commands(self, user_input: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        if user_input.startswith("/topic "):
            new_topic = user_input.replace("/topic ", "").strip()
            self.current_topic = new_topic
            if getattr(self.memory_system, "current_topic", None) is not None:
                self.memory_system.current_topic = new_topic
            return (f"Switched to topic: {new_topic}", {"command": "topic_switch"})

        if user_input == "/clear_topic":
            self.current_topic = "general"
            if getattr(self.memory_system, "current_topic", None) is not None:
                self.memory_system.current_topic = "general"
            return ("Cleared topic context, starting fresh conversation", {"command": "topic_clear"})

        return None  # no command handled

    # ---------- helpers ----------
    def _should_switch_topic(self, topics) -> bool:
        # Replace with your real threshold logic; conservative default:
        if not topics:
            return False
        if self.current_topic == "general":
            return True
        return (topics[0] or "").lower() != (self.current_topic or "").lower()

    # ---------- 2) Context Building (via Pipeline) ----------

    async def build_context(
        self,
        user_input: str,
        files: Optional[List[Any]] = None,
        use_raw_mode: bool = False,
        personality: Optional[str] = None,
    ) -> ContextResult:
        """
        Build context using the ContextPipeline.

        This is the new preferred method for context preparation. It uses the
        ContextPipeline to transform raw user input into processed context
        ready for memory retrieval and prompt building.

        Pipeline stages:
        1. Topic Extraction - Extract topics (TopicManager)
        2. Tone Detection - Detect emotional state (analyze_emotional_context)
        3. File Processing - Extract text from uploaded files (FileProcessor)
        4. Heavy Topic Check - Check for sensitive content (QueryChecker)
        5. Query Rewriting - Rewrite for better retrieval
        6. STM Analysis - Analyze recent conversation (STMAnalyzer)
        7. Identity Injection - Add user identity context (UserProfile)
        8. Thread Context - Get active thread (memory_system)

        Args:
            user_input: Raw user query
            files: Optional list of uploaded files
            use_raw_mode: Skip enrichment (direct passthrough)
            personality: Optional personality override

        Returns:
            ContextResult with all processed context components

        Example usage:
            context = await self.build_context(user_input, files)
            # Use context.processed_query for memory retrieval
            # Use context.tone_instructions for system prompt
            # Use context.topics for relevance filtering
        """
        if not self.context_pipeline:
            self.logger.warning("[Orchestrator] ContextPipeline not available, using defaults")
            return ContextResult(
                processed_query=user_input,
                original_query=user_input,
                tone_level=ToneLevel.CONVERSATIONAL,
                tone_instructions="",
            )

        # Get recent conversation history for context
        conversation_history = None
        if self.memory_system and hasattr(self.memory_system, 'corpus_manager'):
            try:
                conversation_history = self.memory_system.corpus_manager.get_recent_memories(5)
            except Exception as e:
                logger.warning(f"[Orchestrator] Could not retrieve conversation history: {e}, proceeding without context")

        context = await self.context_pipeline.build(
            user_input=user_input,
            files=files,
            use_raw_mode=use_raw_mode,
            personality=personality,
            conversation_history=conversation_history,
        )

        # Sync state with orchestrator
        if context.primary_topic:
            self.current_topic = context.primary_topic
            if hasattr(self.memory_system, 'current_topic'):
                self.memory_system.current_topic = context.primary_topic

        if context.emotional_context:
            self.current_emotional_context = context.emotional_context
            if hasattr(context.emotional_context, 'crisis_level'):
                self.current_tone_level = context.emotional_context.crisis_level

        return context

    async def build_full_prompt(
        self,
        context: ContextResult,
        use_raw_mode: bool = False,
        return_raw_context: bool = False,
    ) -> Union[Tuple[str, str], Tuple[str, str, Dict[str, Any]]]:
        """
        Build the full prompt and system prompt from a ContextResult.

        This method:
        1. Builds the system prompt with all injections (identity, tone, thread, etc.)
        2. Calls prompt_builder.build_prompt_from_context() for context gathering
        3. Assembles the final prompt

        Args:
            context: ContextResult from build_context()
            use_raw_mode: If True, skip enrichment
            return_raw_context: If True, also return the raw context dict (for agentic search)

        Returns:
            If return_raw_context=False: (prompt_string, system_prompt_string)
            If return_raw_context=True: (prompt_string, system_prompt_string, raw_context_dict)
        """
        import time
        _t_start = time.perf_counter()

        # --- 1) Build System Prompt ---
        SYSTEM_PROMPT_FALLBACK = (
            "You are Daemon, a helpful assistant with memory and RAG. "
            "Be direct, truthful, concise."
        )
        system_prompt: str = SYSTEM_PROMPT_FALLBACK

        # Load from config
        try:
            persona_cfg = self.personality_manager.get_current_config() if self.personality_manager else {}
        except (AttributeError, TypeError) as e:
            logger.debug(f"[Orchestrator] Could not get personality config: {e}")
            persona_cfg = {}
        base_cfg = getattr(self, "config", {}) or {}
        merged_cfg = {**base_cfg, **(persona_cfg or {})}

        try:
            from config.app_config import load_system_prompt
            loaded = load_system_prompt(merged_cfg)
            if isinstance(loaded, str) and loaded.strip():
                system_prompt = loaded
        except (ImportError, AttributeError) as e:
            logger.debug(f"[Orchestrator] Could not load system prompt from config: {e}")

        # Override path from persona or orchestrator
        override_path = (persona_cfg or {}).get("system_prompt_file")
        if isinstance(override_path, dict):
            override_path = override_path.get("system_prompt_file")
        if not override_path:
            override_path = getattr(self, "system_prompt_path", None)

        if override_path and isinstance(override_path, str):
            try:
                if os.path.exists(override_path):
                    with open(override_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    if text.strip():
                        system_prompt = text
            except (IOError, OSError) as e:
                logger.warning(f"[Orchestrator] Could not read system prompt override file '{override_path}': {e}")

        # --- Identity placeholder substitution ---
        if isinstance(system_prompt, str) and system_prompt.strip():
            try:
                profile = getattr(self, 'user_profile', None)
                if profile:
                    identity = profile.identity
                    name = identity.name if identity.name else "the user"
                    pronouns = identity.pronouns if identity.pronouns else "they/them"

                    PRONOUN_MAP = {
                        "he/him": ("he", "him", "his"),
                        "she/her": ("she", "her", "her"),
                        "they/them": ("they", "them", "their"),
                    }
                    subj, obj, poss = PRONOUN_MAP.get(pronouns.lower(), ("they", "them", "their"))

                    system_prompt = system_prompt.replace("{USER_NAME}", name)
                    system_prompt = system_prompt.replace("{USER_PRONOUNS}", pronouns)
                    system_prompt = system_prompt.replace("{PRONOUN_SUBJ}", subj)
                    system_prompt = system_prompt.replace("{PRONOUN_OBJ}", obj)
                    system_prompt = system_prompt.replace("{PRONOUN_POSS}", poss)
            except (AttributeError, TypeError) as e:
                logger.debug(f"[Orchestrator] Identity placeholder substitution failed: {e}")

        # --- Citation instructions ---
        if self.enable_citations:
            citation_instruction = (
                "\n\n"
                "═══════════════════════════════════════════════════════════════\n"
                "MANDATORY MEMORY CITATION PROTOCOL (REQUIRED)\n"
                "═══════════════════════════════════════════════════════════════\n"
                "\n"
                "CRITICAL REQUIREMENT: You MUST cite every memory you reference in your response.\n"
                "Citation Format: [MEM_RECENT_{n}], [MEM_SEMANTIC_{n}], [SUM_RECENT_{n}], etc.\n"
                "═══════════════════════════════════════════════════════════════\n"
            )
            system_prompt = system_prompt.rstrip() + citation_instruction

        # --- Topic hint ---
        topic_str = context.primary_topic or "general"
        system_prompt = system_prompt.rstrip() + f"\n\nQuery topic: {topic_str}"

        # --- Thread context injection ---
        if not use_raw_mode and context.has_thread:
            thread_ctx = context.thread_context
            thread_depth = thread_ctx.get("thread_depth", 1)
            is_heavy = thread_ctx.get("is_heavy_topic", False)
            thread_topic = thread_ctx.get("thread_topic", "")

            thread_msg = f"\n\n[THREAD CONTEXT]"
            thread_msg += f"\nThis is message #{thread_depth} in an ongoing conversation thread"
            if thread_topic:
                thread_msg += f" about {thread_topic}"
            if is_heavy:
                thread_msg += "\nThis is a sensitive/heavy topic. Be empathetic and specific."
            elif thread_depth >= 3:
                thread_msg += "\nMaintain conversational continuity."

            system_prompt = system_prompt.rstrip() + thread_msg

        # --- Response mode instructions (from ContextResult tone) ---
        if not use_raw_mode:
            emotional_ctx = context.emotional_context
            if emotional_ctx:
                response_instructions = self._get_response_instructions(emotional_ctx)
                system_prompt = system_prompt.rstrip() + response_instructions

            # Escalation adaptation: append strategy-specific overrides
            if self.escalation_tracker:
                escalation_instructions = self.escalation_tracker.get_strategy_instructions()
                if escalation_instructions:
                    system_prompt = system_prompt.rstrip() + escalation_instructions

            session_headers = self._get_session_headers_instructions()
            system_prompt = system_prompt.rstrip() + session_headers

        # --- Thinking block instruction ---
        if not use_raw_mode:
            thinking_instruction = (
                "\n\n[IMPORTANT] Before your final response, include your reasoning "
                "in <thinking>...</thinking> tags. Walk through the context step-by-step, "
                "then provide your answer outside the tags."
            )
            system_prompt = system_prompt.rstrip() + thinking_instruction

        # --- 2) Build prompt context via PromptBuilder ---
        prompt_ctx = await self.prompt_builder.build_prompt_from_context(context)

        # Store memory_id_map for citation extraction
        self._current_memory_id_map = prompt_ctx.get('memory_id_map', {})

        # --- 3) Assemble final prompt ---
        user_input = context.file_context if context.has_files else context.original_query
        prompt = self.prompt_builder._assemble_prompt(
            context=prompt_ctx,
            user_input=user_input,
            system_prompt=system_prompt
        )

        if self.logger:
            duration = time.perf_counter() - _t_start
            self.logger.info(f"[BUILD_FULL_PROMPT] Completed in {duration:.2f}s")
            # Debug: Check if note_images was added to prompt_ctx by _assemble_prompt
            _note_imgs = prompt_ctx.get("note_images", []) if prompt_ctx else []
            if _note_imgs:
                self.logger.warning(f"[BUILD_FULL_PROMPT] note_images in prompt_ctx: {len(_note_imgs)} images")
            else:
                self.logger.warning(f"[BUILD_FULL_PROMPT] NO note_images in prompt_ctx! Keys: {list(prompt_ctx.keys()) if prompt_ctx else 'None'}")

        if return_raw_context:
            return prompt, system_prompt, prompt_ctx
        return prompt, system_prompt

    # ---------- 3) Prepare Prompt (legacy - use build_context instead) ----------

    async def prepare_prompt(
        self,
        user_input: str,
        files: Optional[List[Any]] = None,
        use_raw_mode: bool = False,
        return_context: bool = False,
    ) -> Union[Tuple[str, Optional[str]], Tuple[str, Optional[str], Dict[str, Any]]]:
        """
        Performs pre-generation steps: topic update, file processing, optional query
        rewrite, and prompt building.

        .. deprecated::
            Use :meth:`build_context` instead for cleaner separation of concerns.
            This method will be refactored to use ContextPipeline internally in a future version.

            New preferred flow::

                # Step 1: Build context (query analysis)
                context = await self.build_context(user_input, files, use_raw_mode)

                # Step 2: Build prompt using context
                prompt_ctx = await self.prompt_builder.build_prompt_from_context(context)

                # Step 3: Assemble final prompt
                prompt = self.prompt_builder._assemble_prompt(prompt_ctx, user_input)

        Args:
            user_input: The user's input text
            files: Optional list of uploaded files
            use_raw_mode: If True, skip RAG context gathering
            return_context: If True, also return the raw context dict for agentic search

        Returns:
            If return_context=False: (prompt, system_prompt)
            If return_context=True: (prompt, system_prompt, context_dict)
        """
        # Thin wrapper that delegates to the new ContextPipeline-based methods
        import time
        _t_start = time.perf_counter()

        # Step 1: Build context via ContextPipeline
        context = await self.build_context(
            user_input=user_input,
            files=files,
            use_raw_mode=use_raw_mode,
        )

        # Step 2: Raw mode returns early
        if use_raw_mode:
            return context.file_context or user_input, None

        # Step 3: Build full prompt (with optional raw context for agentic search)
        result = await self.build_full_prompt(
            context=context,
            use_raw_mode=use_raw_mode,
            return_raw_context=return_context,
        )

        if self.logger:
            duration = time.perf_counter() - _t_start
            self.logger.info(f"[PREPARE_PROMPT] Completed in {duration:.2f}s (via ContextPipeline)")

        return result

    # ---------- Memory Citation Methods ----------
    def _expand_citation_range(self, mem_id: str) -> List[str]:
        """
        Expand a range citation like MEM_RECENT_4-7 into individual IDs.

        Args:
            mem_id: Citation ID (e.g., "MEM_RECENT_4-7" or "MEM_RECENT_3")

        Returns:
            List of individual citation IDs (e.g., ["MEM_RECENT_4", "MEM_RECENT_5", "MEM_RECENT_6", "MEM_RECENT_7"])
        """
        # Check if it's a range citation (contains hyphen)
        if '-' in mem_id and not mem_id.startswith('PROFILE'):
            # Parse the range: MEM_RECENT_4-7 -> prefix="MEM_RECENT_", start=4, end=7
            match = re.match(r'([A-Z_]+_)(\d+)-(\d+)', mem_id)
            if match:
                prefix = match.group(1)
                start = int(match.group(2))
                end = int(match.group(3))

                # Generate individual IDs
                return [f"{prefix}{i}" for i in range(start, end + 1)]

        # Not a range, return as single-item list
        return [mem_id]

    def _extract_citations(self, response: str, memory_map: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract memory citations from response, handling both single citations and ranges.

        Args:
            response: Raw response with citation tags like [MEM_RECENT_3] or [MEM_RECENT_4-7]
            memory_map: Dictionary mapping citation IDs to memory metadata

        Returns:
            Tuple of (clean_response, citations_list)
            - clean_response: Response with citation tags removed
            - citations_list: List of cited memory metadata dicts
        """
        # Find all cited memory IDs (includes ranges like MEM_RECENT_4-7)
        cited_ids = set(self.citation_pattern.findall(response))

        citations = []
        seen_ids = set()  # Track which individual IDs we've already added

        if memory_map:
            for mem_id in cited_ids:
                # Expand ranges into individual IDs
                expanded_ids = self._expand_citation_range(mem_id)

                # Look up each individual ID in memory_map
                for individual_id in expanded_ids:
                    if individual_id in memory_map and individual_id not in seen_ids:
                        citations.append({
                            'memory_id': individual_id,
                            'type': memory_map[individual_id].get('type', 'unknown'),
                            'timestamp': memory_map[individual_id].get('timestamp', ''),
                            'content': memory_map[individual_id].get('content', '')[:200],  # Truncate for display
                            'relevance_score': memory_map[individual_id].get('relevance_score', 0.0),
                            'db_id': memory_map[individual_id].get('db_id', None)  # Include database ID for traceability
                        })
                        seen_ids.add(individual_id)

        # Remove citation tags for clean display (always, even if memory_map is empty)
        clean_response = self.citation_pattern.sub('', response)

        # Clean up multiple spaces left by removal
        clean_response = re.sub(r'\s+', ' ', clean_response)
        clean_response = clean_response.strip()

        self.logger.debug(f"[Citation] Extracted {len(citations)} citations from response (from {len(cited_ids)} citation tags)")

        return clean_response, citations

    # ---------- 3) Generate & Store ----------
    async def process_user_query(
        self,
        user_input: str,
        files: Optional[List[Any]] = None,
        use_raw_mode: bool = False,
        personality: Optional[str] = None,
        use_agentic_search: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Orchestrates the full request:
          - optional personality switch (if provided)
          - commands (early exit)
          - deictic pre-check (optional early clarification)
          - prepare_prompt
          - generate + store (or agentic search loop if enabled)
        Returns: (assistant_text, debug_info)

        Args:
            user_input: The user's query
            files: Optional list of files to process
            use_raw_mode: If True, skip memory/context gathering
            personality: Optional personality to switch to
            use_agentic_search: If True, use multi-round agentic search

        Note: Agentic search requires:
            - agentic_search.enabled = true in config
            - Web search trigger to indicate search is needed
            - Will fall back to standard flow if conditions not met
        """
        debug_info: Dict[str, Any] = {
            "start_time": datetime.now(),
            "user_input": user_input[:100],
            "files_count": len(files) if files else 0,
            "mode": "raw" if use_raw_mode else "enhanced",
        }

        try:
            # Personality hook: let GUI pass personality labels that flip the active config
            if personality and self.personality_manager:
                try:
                    self.personality_manager.switch_personality(personality)
                    if self.logger:
                        self.logger.info(f"Personality set to: {personality}")
                except Exception as e:
                    logger.error(f"[Orchestrator] Personality switch failed for '{personality}': {e}, using default")


            # --- Commands: early exit ---
            cmd = self.handle_commands(user_input)
            if cmd:
                # shape matches handler expectations: (text, debug_info)
                text, meta = cmd
                debug_info.update(meta)
                return text, debug_info

            # --- Deictic pre-check (clarify before we build/stream) ---
            if not use_raw_mode and is_deictic(user_input) and self.memory_system:
                try:
                    retrieval_result = await self.memory_system.get_memories(user_input, limit=10)
                    if retrieval_result and retrieval_result[0].get("metadata", {}).get("needs_clarification"):
                        response = "I'm not sure what you're referring to. Could you be more specific?"
                        try:
                            await self.memory_system.store_interaction(
                                query=user_input, response=response, tags=["clarification"]
                            )
                        except Exception as e:
                            logger.warning(f"[Orchestrator] Failed to store clarification interaction: {e}")
                        debug_info.update({
                            "response_length": len(response),
                            "end_time": datetime.now(),
                            "prompt_length": 0,
                        })
                        debug_info["duration"] = (debug_info["end_time"] - debug_info["start_time"]).total_seconds()
                        return response, debug_info
                except Exception as e:
                    if self.logger:
                        self.logger.debug(f"[Orchestrator] Deictic pre-retrieval failed or skipped: {e}")

            # --- Determine Model Name FIRST (needed for multimodal image loading) ---
            active_name_getter = getattr(self.model_manager, "get_active_model_name", None)
            model_name = active_name_getter() if callable(active_name_getter) else None
            model_name = model_name or "gpt-4-turbo"

            if self.logger:
                self.logger.info(f"[Orchestrator] Target model for response: {model_name}")

            # --- Build Prompt (NEW: via ContextPipeline) ---
            context = await self.build_context(
                user_input=user_input,
                files=files,
                use_raw_mode=use_raw_mode,
                personality=personality,
            )
            prompt, system_prompt, prompt_ctx = await self.build_full_prompt(
                context=context,
                use_raw_mode=use_raw_mode,
                return_raw_context=True,  # Get prompt context for images
            )

            # Extract images for multimodal models
            note_images = prompt_ctx.get("note_images", []) if prompt_ctx else []
            if self.logger:
                if note_images:
                    total_size = sum(len(img.get("data", "")) for img in note_images)
                    self.logger.warning(f"[Orchestrator] IMAGE DEBUG: {len(note_images)} images extracted from prompt_ctx, total base64={total_size//1024}KB")
                else:
                    self.logger.warning(f"[Orchestrator] IMAGE DEBUG: No images in prompt_ctx. Keys={list(prompt_ctx.keys()) if prompt_ctx else 'None'}")

            # Update escalation tracker with detected tone
            if self.escalation_tracker:
                need_type_str = None
                if context.emotional_context and hasattr(context.emotional_context, 'need_type'):
                    need_type_str = (
                        context.emotional_context.need_type.value
                        if hasattr(context.emotional_context.need_type, 'value')
                        else str(context.emotional_context.need_type)
                    )
                self.escalation_tracker.update(
                    context.tone_level,
                    user_input,
                    need_type=need_type_str,
                )

            debug_info["context_pipeline"] = {
                "tone_level": context.tone_level.value,
                "topics": context.topics[:3] if context.topics else [],
                "has_stm": context.has_stm,
                "has_thread": context.has_thread,
                "note_images_count": len(note_images),
            }
            if self.escalation_tracker:
                debug_info["escalation"] = self.escalation_tracker.get_debug_info()

            # Extract values needed for token limit logic
            is_heavy_topic = context.is_heavy_topic
            tone_level = context.tone_level

            # --- Generate Response ---
            # (model_name already determined earlier for image loading)

            # --- Agentic Search Check ---
            # If agentic search is requested and available, check if we should use it
            if use_agentic_search and not use_raw_mode and self.agentic_controller:
                try:
                    # Get web search decision from LLM-first trigger
                    from utils.web_search_trigger import analyze_for_web_search_llm

                    web_decision = await analyze_for_web_search_llm(
                        query=user_input,
                        model_manager=self.model_manager,
                        crisis_level=str(self.current_tone_level) if self.current_tone_level else None,
                        web_search_enabled=True,
                    )

                    if web_decision and web_decision.should_search and web_decision.search_terms:
                        if self.logger:
                            self.logger.info(
                                f"[Orchestrator] Using agentic search: terms={web_decision.search_terms}"
                            )

                        # Run agentic search loop
                        from core.agentic import ProgressEvent

                        full_response = ""
                        async for event_or_chunk in self.agentic_controller.run_agentic_search(
                            query=user_input,
                            system_prompt=system_prompt or "",
                            model_name=model_name,
                            initial_search_terms=web_decision.search_terms,
                            crisis_level=str(self.current_tone_level) if self.current_tone_level else None,
                        ):
                            if isinstance(event_or_chunk, ProgressEvent):
                                # Log progress events
                                if self.logger:
                                    self.logger.debug(
                                        f"[AgenticSearch] {event_or_chunk.event_type}: {event_or_chunk.message}"
                                    )
                                debug_info.setdefault("agentic_events", []).append({
                                    "type": event_or_chunk.event_type,
                                    "message": event_or_chunk.message,
                                    "round": event_or_chunk.round_number,
                                })
                            else:
                                # Accumulate response chunks
                                full_response += event_or_chunk

                        # Store interaction
                        if self.memory_system:
                            try:
                                await self.memory_system.store_interaction(
                                    query=user_input,
                                    response=full_response.strip(),
                                    tags=["agentic_search"]
                                )
                            except Exception:
                                pass

                        debug_info.update({
                            "response_length": len(full_response),
                            "end_time": datetime.now(),
                            "prompt_length": len(prompt),
                            "agentic_search_used": True,
                        })
                        debug_info["duration"] = (debug_info["end_time"] - debug_info["start_time"]).total_seconds()

                        return full_response.strip(), debug_info

                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"[Orchestrator] Agentic search failed, falling back: {e}")
                    debug_info["agentic_error"] = str(e)

            # ---------------------------------------------------------------------
            # Determine max_tokens based on tone level AND topic heaviness
            # ---------------------------------------------------------------------
            try:
                from config.app_config import DEFAULT_MAX_TOKENS, HEAVY_TOPIC_MAX_TOKENS

                # Heavy topics override tone-based limits
                if is_heavy_topic:
                    response_max_tokens = HEAVY_TOPIC_MAX_TOKENS
                    token_reason = "HEAVY topic"
                # Adjust max_tokens based on tone level for speed and brevity
                # tone_level is ToneLevel from context_pipeline
                elif tone_level == ToneLevel.CONVERSATIONAL:
                    response_max_tokens = 600  # Force brief responses in conversational mode
                    token_reason = "CONVERSATIONAL mode"
                elif tone_level == ToneLevel.CONCERN:
                    response_max_tokens = 1000  # Light support responses
                    token_reason = "CONCERN mode"
                elif tone_level == ToneLevel.ELEVATED:
                    response_max_tokens = 1500  # Allow more room for supportive responses
                    token_reason = "ELEVATED mode"
                elif tone_level == ToneLevel.CRISIS:
                    response_max_tokens = 2000  # Maximum room for crisis responses
                    token_reason = "CRISIS mode"
                else:
                    response_max_tokens = DEFAULT_MAX_TOKENS
                    token_reason = "DEFAULT"

                # Escalation tracker may override token budget for brevity
                if self.escalation_tracker:
                    budget_override = self.escalation_tracker.get_token_budget_override()
                    if budget_override is not None:
                        response_max_tokens = budget_override
                        token_reason = f"{self.escalation_tracker.current_strategy.value} (escalation override)"

                if self.logger:
                    self.logger.info(
                        f"[Orchestrator] Token limit: {response_max_tokens} ({token_reason})"
                    )
            except Exception as e:
                response_max_tokens = None  # Use model defaults
                if self.logger:
                    self.logger.debug(f"[Orchestrator] Failed to load token config: {e}")

            # ---------------------------------------------------------------------
            # Best-of / Duel / Ensemble generation (delegated to BestOfHandler)
            # ---------------------------------------------------------------------
            if self.best_of_handler.should_use_best_of(user_input, use_raw_mode):
                best_of_result = await self.best_of_handler.generate(
                    prompt=prompt,
                    user_input=user_input,
                    system_prompt=system_prompt,
                    model_name=model_name,
                    response_max_tokens=response_max_tokens
                )
                full_response = best_of_result.response
                debug_info["best_of_mode"] = best_of_result.mode
                debug_info["best_of_used"] = best_of_result.used_best_of
                # For duel mode, propagate metadata for thinking display
                if best_of_result.mode == "duel" and isinstance(best_of_result.metadata.get("raw"), dict):
                    duel_data = best_of_result.metadata["raw"]
                    debug_info["duel_thinking_a"] = duel_data.get("thinking_a", "")
                    debug_info["duel_thinking_b"] = duel_data.get("thinking_b", "")
                    debug_info["duel_winner"] = duel_data.get("winner", "")
                    debug_info["duel_models"] = duel_data.get("models", {})
            else:
                # Standard streaming path
                full_response = ""
                async for chunk in self.response_generator.generate_streaming_response(
                    prompt,
                    model_name,
                    system_prompt=system_prompt,
                    max_tokens=response_max_tokens,
                    images=note_images if note_images else None  # Pass images for multimodal models
                ):
                    full_response += (chunk + " ")
                full_response = full_response.strip()

            # --- Parse thinking block and extract final answer ---
            thinking_part, final_answer = ResponseParser.parse_thinking_block(full_response)
            # Strip XML-like wrappers (e.g., <result> … </result>) from final answer
            final_answer = ResponseParser.strip_xml_wrappers(final_answer)

            # Log thinking part for debugging if present
            if thinking_part:
                if self.logger:
                    self.logger.debug(f"[THINKING BLOCK]\n{thinking_part}")
                debug_info["thinking_length"] = len(thinking_part)

            # Store final answer (not the thinking part) in memory
            answer_for_storage = final_answer if final_answer else ResponseParser.strip_xml_wrappers(full_response)
            # Strip reflection blocks (they're stored separately as reflection memories)
            answer_for_storage = ResponseParser.strip_reflection_blocks(answer_for_storage)
            # Sanitize prompt header echoes before returning/storing
            answer_for_storage = ResponseParser.strip_prompt_artifacts(answer_for_storage)

            # Extract citations if enabled
            citations = []
            if self.enable_citations and hasattr(self, '_current_memory_id_map') and self._current_memory_id_map:
                raw_response_with_citations = answer_for_storage
                answer_for_storage, citations = self._extract_citations(
                    answer_for_storage,
                    self._current_memory_id_map
                )
                debug_info['raw_response_with_citations'] = raw_response_with_citations

            # Record response in escalation tracker for engagement detection
            if self.escalation_tracker:
                self.escalation_tracker.record_response(answer_for_storage)

            # --- Store Interaction ---
            if self.memory_system and not use_raw_mode:
                try:
                    await self.memory_system.store_interaction(
                        query=user_input,
                        response=answer_for_storage,
                        tags=["conversation"]
                    )
                except Exception as e:
                    logger.error(f"[Orchestrator] CRITICAL: Failed to store interaction - data loss: {e}")
            # Use instance logger
            if self.logger:
                self.logger.debug("[orchestrator] Persisted exchange; considering consolidation")

            # Summaries now run on shutdown; skip mid-session consolidation
            debug_info.update({
                "response_length": len(answer_for_storage),
                "full_response_length": len(full_response),
                "end_time": datetime.now(),
                "prompt_length": len(prompt),
                "citations": citations,  # Add extracted citations
                "citations_enabled": self.enable_citations,
            })
            debug_info["duration"] = (debug_info["end_time"] - debug_info["start_time"]).total_seconds()

            # Return only the final answer (thinking block removed)
            return answer_for_storage, debug_info

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing query: {e}", exc_info=True)
            if getattr(self, "conversation_logger", None):
                try:
                    self.conversation_logger.log_system_event("Error", str(e))
                except (IOError, OSError, AttributeError):
                    pass  # Logging failure shouldn't mask the original error
            debug_info["error"] = str(e)
            raise
