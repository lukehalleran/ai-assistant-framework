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
- System prompt flow:
  - Resolved via config/app_config.load_system_prompt and/or path override; forwarded to response generation as system role message.
- Side effects:
  - Writes conversation+metadata to corpus DB/Chroma via memory_system; logs events.
- Async behavior:
  - Uses async streaming from response_generator; overall methods are async.
"""
import os
import re
import processing.gate_system as gate_system
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
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

    @staticmethod
    def _parse_thinking_block(response: str) -> Tuple[str, str]:
        """
        Parse response to extract thinking block and final answer.

        Args:
            response: Full LLM response potentially containing <thinking>...</thinking>

        Returns:
            Tuple of (thinking_part, final_answer_part)
            - If no thinking block found, thinking_part is empty and final_answer_part is the full response
        """
        if not response or not isinstance(response, str):
            return "", response or ""

        # Look for </thinking> delimiter
        delimiter = "</thinking>"
        if delimiter in response:
            parts = response.split(delimiter, 1)
            if len(parts) == 2:
                thinking_raw = parts[0]
                final_answer = parts[1].strip()

                # Extract thinking content (remove opening tag if present)
                thinking_content = thinking_raw
                if "<thinking>" in thinking_raw:
                    thinking_content = thinking_raw.split("<thinking>", 1)[1]

                return thinking_content.strip(), final_answer

        # No thinking block found - return empty thinking and full response as answer
        return "", response

    @staticmethod
    def _strip_reflection_blocks(response: str) -> str:
        """
        Strip reflection blocks from response before storing/showing as conversation.

        Reflections are stored separately as reflection memories, so they shouldn't
        also be saved as part of the conversation response.

        Handles both formats:
        - <reflect>...</reflect>
        - [SYSTEM QUALITY REFLECTION]...

        Args:
            response: Full LLM response potentially containing reflection blocks

        Returns:
            Response with all reflection blocks removed
        """
        if not response or not isinstance(response, str):
            return response or ""

        import re
        # Remove <reflect>...</reflect> blocks
        cleaned = re.sub(r'<reflect>.*?</reflect>', '', response, flags=re.DOTALL)

        # Remove [SYSTEM QUALITY REFLECTION] and everything after it
        cleaned = re.sub(r'\[SYSTEM QUALITY REFLECTION\].*', '', cleaned, flags=re.DOTALL)

        # Remove standalone <reflection> tags (legacy format)
        cleaned = re.sub(r'<reflection>.*?</reflection>', '', cleaned, flags=re.DOTALL)

        # Clean up any extra whitespace left behind
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Collapse multiple blank lines
        return cleaned.strip()

    @staticmethod
    def _strip_xml_wrappers(text: str) -> str:
        """Remove simple XML-like wrappers such as <result>...</result>, <answer>...</answer>.

        Keeps inner content; tolerant if tags are missing.
        """
        if not text:
            return text
        try:
            import re
            s = text.strip()
            # Unwrap common tags if they span the whole string
            for tag in ("result", "answer", "final"):
                pattern = rf"^\s*<\s*{tag}[^>]*>([\s\S]*?)<\s*/\s*{tag}\s*>\s*$"
                m = re.match(pattern, s, flags=re.IGNORECASE)
                if m:
                    s = m.group(1).strip()
            return s
        except Exception:
            return text

    @staticmethod
    def _strip_prompt_artifacts(text: str) -> str:
        """Remove known bracketed prompt headers if the model echoes them.

        Conservative: removes header lines and their immediate block until a blank line.
        """
        if not text:
            return text
        try:
            import re
            header_patterns = [
                r"^\s*\[TIME CONTEXT\]",
                r"^\s*\[RECENT CONVERSATION[^\]]*\]",
                r"^\s*\[RELEVANT INFORMATION\]",
                r"^\s*\[RELEVANT MEMORIES\]",
                r"^\s*\[FACTS[ ^\]]*\]",
                r"^\s*\[RECENT FACTS\]",
                r"^\s*\[CURRENT MESSAGE FACTS\]",
                r"^\s*\[DIRECTIVES\]",
                r"^\s*\[CURRENT USER QUERY[ ^\]]*\]",
                r"^\s*\[USER INPUT\]",
                r"^\s*\[BACKGROUND KNOWLEDGE\]",
                r"^\s*\[CONVERSATION SUMMARIES[ ^\]]*\]",
                r"^\s*\[RECENT REFLECTIONS[ ^\]]*\]",
                r"^\s*\[SESSION REFLECTIONS[ ^\]]*\]",
            ]
            header_re = re.compile("(" + ")|(".join(header_patterns) + ")", re.IGNORECASE)
            lines = []
            skip_block = False
            for line in (text.splitlines() or []):
                if header_re.search(line):
                    skip_block = True
                    continue
                if skip_block:
                    if not line.strip():
                        skip_block = False
                    continue
                lines.append(line)
            return "\n".join(lines).strip()
        except Exception:
            return text

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

    # ---------- STM Helper Methods ----------
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

        # Detect topic changes - reset STM context on topic shift
        try:
            if self.topic_manager:
                # Get current topic from topic manager
                self.topic_manager.update_from_user_input(user_input)
                current_topic = self.topic_manager.get_primary_topic()

                # If we have a significant topic change, skip STM (clear contamination)
                if current_topic and self.last_stm_topic:
                    # Normalize for comparison
                    current_norm = current_topic.lower().strip()
                    last_norm = self.last_stm_topic.lower().strip()

                    if current_norm != last_norm and current_norm != "general":
                        self.logger.info(
                            f"[STM] Topic change detected: '{self.last_stm_topic}' -> '{current_topic}'. "
                            "Resetting STM context to prevent contamination."
                        )
                        self.last_stm_topic = current_topic
                        return False  # Skip STM on topic transitions

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
            except Exception:
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

    # ---------- 2) Commands & Topic ----------
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

    # ---------- 2) Prepare (files, rewrite, prompt) ----------

    async def prepare_prompt(
        self,
        user_input: str,
        files: Optional[List[Any]] = None,
        use_raw_mode: bool = False,
    ) -> Tuple[str, Optional[str]]:
        """
        Performs pre-generation steps: topic update, file processing, optional query
        rewrite, and prompt building.

        Returns:
            (prompt, system_prompt)
        """
        # ---------------------------------------------------------------------
        # 0) Topic inference (non-fatal; use single canonical topic string)
        # ---------------------------------------------------------------------
        try:
            if getattr(self, "topic_manager", None):
                # Update TopicManager’s internal state from this turn’s input
                self.topic_manager.update_from_user_input(user_input)

                # Ask for a single canonical topic string (or None)
                primary = self.topic_manager.get_primary_topic()

                # Switch only if different from current
                if primary and (primary.lower() != (self.current_topic or "general").lower()):
                    self.current_topic = primary
                    if getattr(self, "memory_system", None) is not None:
                        self.memory_system.current_topic = self.current_topic
                    if getattr(self, "logger", None):
                        self.logger.info(f"Topic switched to: {self.current_topic}")
        except Exception:
            # Topic inference must never block the flow
            pass

        # ---------------------------------------------------------------------
        # 0.5) Tone Detection (crisis vs. casual) - runs for both RAW and ENHANCED modes
        # ---------------------------------------------------------------------
        conversation_history = None
        if self.memory_system and hasattr(self.memory_system, "corpus_manager"):
            try:
                # Get recent conversation history for context-aware tone detection
                recent = self.memory_system.corpus_manager.get_recent_memories(count=3)
                conversation_history = recent if recent else None
            except Exception as e:
                if self.logger:
                    self.logger.debug(f"[Orchestrator] Failed to get conversation history for tone detection: {e}")

        # Detect emotional context (tone + need type)
        emotional_context = await analyze_emotional_context(
            message=user_input,
            conversation_history=conversation_history,
            model_manager=self.model_manager
        )

        # Log emotional context (backend only)
        emotional_log_msg = format_emotional_context_log(emotional_context, user_input)
        if self.logger:
            self.logger.info(f"[EMOTIONAL_CONTEXT] {emotional_log_msg}")

        # Log tone shifts
        if should_log_tone_shift(self.current_tone_level, emotional_context.crisis_level):
            shift_log = format_tone_shift_log(
                self.current_tone_level,
                emotional_context.crisis_level,
                emotional_context.tone_trigger
            )
            if self.logger:
                self.logger.warning(f"[TONE_SHIFT] {shift_log}")

        # Update current emotional context (will be used later when injecting response instructions)
        self.current_emotional_context = emotional_context
        self.current_tone_level = emotional_context.crisis_level  # Keep for compatibility

        # ---------------------------------------------------------------------
        # 1) File processing (enhanced path only)
        # ---------------------------------------------------------------------
        combined_text = user_input
        if files and not use_raw_mode and getattr(self, "file_processor", None):
            try:
                combined_text = await self.file_processor.process_files(user_input, files)
            except Exception:
                # Fail-open: use raw user_input if file processing has issues
                combined_text = user_input

        # ---------------------------------------------------------------------
        # 1.5) Inline fact extraction for heavy topics/long messages
        # ---------------------------------------------------------------------
        fresh_facts: List[Dict] = []
        is_heavy_topic = False  # Track for response token allocation
        if not use_raw_mode and getattr(self, "memory_system", None):
            try:
                # Use async query analysis with LLM for best accuracy
                from utils.query_checker import analyze_query_async
                qinfo = await analyze_query_async(user_input, model_manager=self.model_manager)
                is_heavy_topic = qinfo.is_heavy_topic  # Save for later use

                if qinfo.is_heavy_topic:
                    if getattr(self, "logger", None):
                        self.logger.info(
                            f"[Orchestrator] Heavy topic detected (len={len(user_input)}), "
                            "running inline fact extraction"
                        )

                    # Extract facts with timeout to avoid blocking
                    import asyncio
                    try:
                        fact_task = asyncio.create_task(
                            self.memory_system._extract_and_store_facts(
                                query=user_input,
                                response="",  # No response yet
                                truth_score=0.7  # Default for user-provided content
                            )
                        )

                        # Wait max 5 seconds
                        await asyncio.wait_for(fact_task, timeout=5.0)

                        # Retrieve just-extracted facts for prompt injection
                        fresh_facts = await self.memory_system.get_facts(
                            query=user_input,
                            limit=10  # Top 10 most relevant
                        )

                        if getattr(self, "logger", None) and fresh_facts:
                            self.logger.info(
                                f"[Orchestrator] Extracted {len(fresh_facts)} inline facts"
                            )

                    except asyncio.TimeoutError:
                        if getattr(self, "logger", None):
                            self.logger.warning("[Orchestrator] Inline fact extraction timed out")
                    except Exception as e:
                        if getattr(self, "logger", None):
                            self.logger.debug(f"[Orchestrator] Inline fact extraction failed: {e}")

            except Exception as e:
                # Never let fact extraction block the flow
                if getattr(self, "logger", None):
                    self.logger.debug(f"[Orchestrator] Heavy topic detection failed: {e}")

        # ---------------------------------------------------------------------
        # 2) Optional query rewrite (for retrieval/search phrasing)
        # ---------------------------------------------------------------------
        rewritten_query: Optional[str] = None
        qinfo = None
        try:
            qinfo = analyze_query(user_input)
        except Exception:
            qinfo = None
        # Only rewrite longer questions; skip tiny factoids to save latency
        # Allow config to disable query rewrite for lower latency
        _features = (self.config or {}).get("features", {}) if isinstance(self.config, dict) else {}
        _enable_rewrite = bool(_features.get("enable_query_rewrite", True))
        if _enable_rewrite and not use_raw_mode and qinfo and qinfo.is_question and qinfo.token_count >= 8:
            try:
                rewrite_prompt = (
                    'Rewrite the following user question into a concise, third-person declarative '
                    'statement suitable for a vector database search.\n\n'
                    f'User question: "{user_input}"\nRewritten statement:'
                )
                # Use a low-latency alias for quick rewrites; allow disabling timeout via config (<= 0)
                from config.app_config import REWRITE_TIMEOUT_S
                import asyncio as _a
                _rw_timeout = 0.0
                try:
                    _rw_timeout = float(REWRITE_TIMEOUT_S)
                except Exception:
                    _rw_timeout = 0.0
                _coro = self.model_manager.generate_once(
                    prompt=rewrite_prompt, model_name="gpt-4o-mini"
                )
                if _rw_timeout > 0:
                    rewritten_query = await _a.wait_for(_coro, timeout=_rw_timeout)
                else:
                    rewritten_query = await _coro
                if isinstance(rewritten_query, str):
                    rewritten_query = rewritten_query.strip().strip('"')
                else:
                    rewritten_query = user_input
            except Exception:
                rewritten_query = user_input

        # ---------------------------------------------------------------------
        # 3) Resolve system prompt (robust order + config-aware)
        # ---------------------------------------------------------------------
        SYSTEM_PROMPT_FALLBACK = (
            "You are Daemon, a helpful assistant with memory and RAG. "
            "Be direct, truthful, concise."
        )
        system_prompt: str = SYSTEM_PROMPT_FALLBACK

        # Merge persona config over base orchestrator config
        try:
            persona_cfg = self.personality_manager.get_current_config() if self.personality_manager else {}
        except Exception:
            persona_cfg = {}
        base_cfg = getattr(self, "config", {}) or {}
        merged_cfg = {**base_cfg, **(persona_cfg or {})}

        # Prefer centralized loader so it can read paths.* / prompts.* from cfg
        try:
            from config.app_config import load_system_prompt  # local import to avoid hard dep at import time
            loaded = load_system_prompt(merged_cfg)
            if isinstance(loaded, str) and loaded.strip():
                system_prompt = loaded
        except Exception:
            pass

        # Optional path override from persona or orchestrator
        override_path = None
        spf = (persona_cfg or {}).get("system_prompt_file")
        if isinstance(spf, str):
            override_path = spf
        elif isinstance(spf, dict):
            override_path = spf.get("system_prompt_file")

        if not override_path:
            override_path = getattr(self, "system_prompt_path", None)

        if override_path and isinstance(override_path, str):
            try:
                if os.path.exists(override_path):
                    with open(override_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    if text.strip():
                        system_prompt = text
            except Exception:
                pass

            if getattr(self, "logger", None):
                self.logger.info(
                    f"[orchestrator] Using system prompt len={len(system_prompt)}; "
                    f"head={repr(system_prompt[:80])}"
                )

        # ---------------------------------------------------------------------
        # 3.5) Runtime placeholder substitution for identity (name, pronouns)
        # ---------------------------------------------------------------------
        if isinstance(system_prompt, str) and system_prompt.strip():
            try:
                profile = getattr(self, 'user_profile', None)
                if profile:
                    identity = profile.identity

                    # Get name and pronouns, with defaults
                    name = identity.name if identity.name else "the user"
                    pronouns = identity.pronouns if identity.pronouns else "they/them"

                    # Map pronouns to variants (subject, object, possessive)
                    PRONOUN_MAP = {
                        "he/him": ("he", "him", "his"),
                        "she/her": ("she", "her", "her"),
                        "they/them": ("they", "them", "their"),
                    }
                    subj, obj, poss = PRONOUN_MAP.get(pronouns.lower(), ("they", "them", "their"))

                    # Replace placeholders
                    system_prompt = system_prompt.replace("{USER_NAME}", name)
                    system_prompt = system_prompt.replace("{USER_PRONOUNS}", pronouns)
                    system_prompt = system_prompt.replace("{PRONOUN_SUBJ}", subj)
                    system_prompt = system_prompt.replace("{PRONOUN_OBJ}", obj)
                    system_prompt = system_prompt.replace("{PRONOUN_POSS}", poss)

                    if getattr(self, "logger", None):
                        self.logger.debug(
                            f"[Orchestrator] Injected identity placeholders: "
                            f"name={name}, pronouns={pronouns}"
                        )
            except Exception as e:
                if getattr(self, "logger", None):
                    self.logger.debug(f"[Orchestrator] Profile placeholder injection failed: {e}")

        # ---------------------------------------------------------------------
        # 3.8) Citation instructions (if enabled) - INJECTED EARLY FOR PROMINENCE
        # ---------------------------------------------------------------------
        if self.enable_citations:
            citation_instruction = (
                "\n\n"
                "═══════════════════════════════════════════════════════════════\n"
                "MANDATORY MEMORY CITATION PROTOCOL (REQUIRED)\n"
                "═══════════════════════════════════════════════════════════════\n"
                "\n"
                "CRITICAL REQUIREMENT: You MUST cite every memory you reference in your response.\n"
                "\n"
                "Citation Format (use the exact index numbers shown in each section):\n"
                "• [MEM_RECENT_{n}] - For item n) in [RECENT CONVERSATION] section\n"
                "• [MEM_SEMANTIC_{n}] - For item n) in [RELEVANT MEMORIES] section\n"
                "• [SUM_RECENT_{n}] - For item n) in [RECENT SUMMARIES] section\n"
                "• [SUM_SEMANTIC_{n}] - For item n) in [SEMANTIC SUMMARIES] section\n"
                "• [REFL_RECENT_{n}] - For item n) in [RECENT REFLECTIONS] section\n"
                "• [REFL_SEMANTIC_{n}] - For item n) in [SEMANTIC REFLECTIONS] section\n"
                "• [PROFILE_CONTEXT] - For user profile information\n"
                "\n"
                "Examples:\n"
                "✓ \"You mentioned [MEM_RECENT_2] wanting to share this with OMSA professors.\"\n"
                "✓ \"Based on [MEM_RECENT_1] and [MEM_SEMANTIC_3], the dark theme is working well.\"\n"
                "✓ \"Your profile shows [PROFILE_CONTEXT] you're working on this project.\"\n"
                "\n"
                "Rules:\n"
                "1. ALWAYS cite when referencing specific facts, events, or statements from context\n"
                "2. Use the EXACT number shown in the prompt (e.g., if you see \"1) ...\", cite [MEM_RECENT_1])\n"
                "3. Include citations inline, immediately after the relevant statement\n"
                "4. Multiple citations are encouraged when combining information from multiple memories\n"
                "\n"
                "This is NOT optional - citations provide transparency and traceability for memory-augmented responses.\n"
                "═══════════════════════════════════════════════════════════════\n"
            )
            system_prompt = system_prompt.rstrip() + citation_instruction
            if self.logger:
                self.logger.debug("[PREPARE_PROMPT] Injected citation protocol (early position)")

        # ---------------------------------------------------------------------
        # 4) Append resolved topic hint to the end of the system prompt
        #     (kept simple; if no topic was inferred, default to 'general').
        # ---------------------------------------------------------------------
        try:
            topic_str = (getattr(self, "current_topic", None) or "general").strip()
            if isinstance(system_prompt, str) and system_prompt.strip():
                system_prompt = system_prompt.rstrip() + f"\n\nQuery topic: {topic_str}"
        except Exception:
            # Never let topic hint break the flow
            pass

        # ---------------------------------------------------------------------
        # 4.25) Inject conversation thread context for tone/style adjustment
        # ---------------------------------------------------------------------
        if not use_raw_mode and isinstance(system_prompt, str) and system_prompt.strip():
            try:
                if self.memory_system and hasattr(self.memory_system, 'get_thread_context'):
                    thread_ctx = self.memory_system.get_thread_context()

                    if thread_ctx and thread_ctx.get("thread_id"):
                        thread_depth = thread_ctx.get("thread_depth", 1)
                        is_heavy = thread_ctx.get("is_heavy_topic", False)
                        thread_topic = thread_ctx.get("thread_topic", "")

                        # Build thread context message
                        thread_msg = f"\n\n[THREAD CONTEXT]"
                        thread_msg += f"\nThis is message #{thread_depth} in an ongoing conversation thread"

                        if thread_topic:
                            thread_msg += f" about {thread_topic}"

                        if is_heavy:
                            thread_msg += "\nThis is a sensitive/heavy topic. "

                            if thread_depth >= 3:
                                thread_msg += (
                                    "You've been discussing this for multiple turns. "
                                    "Focus on specific details and avoid repeating general therapeutic questions. "
                                    "Reference concrete facts from earlier in the conversation."
                                )
                            else:
                                thread_msg += (
                                    "Be empathetic and specific. "
                                    "Engage with concrete details rather than generic therapeutic responses."
                                )
                        else:
                            # Non-heavy thread: subtle continuity hint
                            if thread_depth >= 3:
                                thread_msg += "\nMaintain conversational continuity and build on previous exchanges."

                        system_prompt = system_prompt.rstrip() + thread_msg

                        if getattr(self, "logger", None):
                            self.logger.debug(
                                f"[Orchestrator] Injected thread context: "
                                f"depth={thread_depth}, heavy={is_heavy}, topic={thread_topic}"
                            )
            except Exception as e:
                # Never let thread context injection break the flow
                if getattr(self, "logger", None):
                    self.logger.debug(f"[Orchestrator] Thread context injection failed: {e}")

        # ---------------------------------------------------------------------
        # 4.5) Add response mode instructions (emotional context: tone + need type)
        # ---------------------------------------------------------------------
        if not use_raw_mode and isinstance(system_prompt, str) and system_prompt.strip():
            # Get emotional context from the orchestrator's state
            emotional_ctx = getattr(self, "current_emotional_context", None)

            if self.logger:
                if emotional_ctx:
                    self.logger.info(
                        f"[PREPARE_PROMPT] Emotional context: "
                        f"Crisis={emotional_ctx.crisis_level.value}, Need={emotional_ctx.need_type.value}"
                    )
                else:
                    self.logger.info("[PREPARE_PROMPT] No emotional context set")

            if emotional_ctx:
                response_instructions = self._get_response_instructions(emotional_ctx)
                system_prompt = system_prompt.rstrip() + response_instructions
                if self.logger:
                    self.logger.info(
                        f"[PREPARE_PROMPT] Injected response instructions for "
                        f"{emotional_ctx.crisis_level.value}/{emotional_ctx.need_type.value}"
                    )
            else:
                if self.logger:
                    self.logger.warning("[PREPARE_PROMPT] No emotional context set - skipping response instructions")

            # Add session headers instructions for temporal reasoning and memory usage
            session_headers_instructions = self._get_session_headers_instructions()
            system_prompt = system_prompt.rstrip() + session_headers_instructions
            if self.logger:
                self.logger.debug("[PREPARE_PROMPT] Injected session headers instructions")

        # ---------------------------------------------------------------------
        # 4.6) Add thinking block instruction to system prompt
        # ---------------------------------------------------------------------
        if not use_raw_mode and isinstance(system_prompt, str) and system_prompt.strip():
            thinking_instruction = (
                "\n\n"
                "IMPORTANT: Before you provide your final answer, you must include a <thinking> block. "
                "Inside this block, detail your step-by-step reasoning and analysis of the user's request. "
                "After the </thinking> block, provide your final, concise, and helpful response to the user."
            )
            system_prompt = system_prompt.rstrip() + thinking_instruction

        # ---------------------------------------------------------------------
        # 4.8) Citation instructions - MOVED TO SECTION 3.8 FOR GREATER PROMINENCE
        # ---------------------------------------------------------------------
        # Citations are now injected early (section 3.8) so LLM sees them before
        # other instructions. This increases compliance with citation protocol.

        # ---------------------------------------------------------------------
        # 5) Raw mode: return plain text, no system prompt
        # ---------------------------------------------------------------------
        if use_raw_mode:
            return combined_text, None

        # ---------------------------------------------------------------------
        # 4.7) STM (Short-Term Memory) Pass - Analyze recent context
        # ---------------------------------------------------------------------
        stm_summary = None
        # Fetch MORE conversation history for STM (tone detection only needed 3)
        stm_conversation_history = None
        if self.memory_system and hasattr(self.memory_system, "corpus_manager"):
            try:
                stm_conversation_history = self.memory_system.corpus_manager.get_recent_memories(count=10) or []
            except Exception:
                pass

        if self.stm_analyzer and self._should_use_stm(stm_conversation_history, user_input):
            try:
                from config.app_config import STM_MAX_RECENT_MESSAGES
                # Get more recent messages for STM
                stm_recent = stm_conversation_history or []
                if self.memory_system and hasattr(self.memory_system, "corpus_manager"):
                    stm_recent = self.memory_system.corpus_manager.get_recent_memories(
                        count=STM_MAX_RECENT_MESSAGES
                    ) or []

                # Extract last assistant response for coherence
                last_assistant_response = None
                for mem in reversed(stm_recent):
                    if mem.get('response'):
                        last_assistant_response = mem.get('response')
                        break

                # Run STM analysis
                stm_summary = await self.stm_analyzer.analyze(
                    recent_memories=stm_recent,
                    user_query=user_input,
                    last_assistant_response=last_assistant_response
                )

                if self.logger:
                    self.logger.info(
                        f"[STM] Analysis complete: topic={stm_summary.get('topic')}, "
                        f"tone={stm_summary.get('tone')}, threads={len(stm_summary.get('open_threads', []))}"
                    )
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[STM] Analysis failed, continuing without STM: {e}")
                stm_summary = None

        # ---------------------------------------------------------------------
        # 5) Build prompt (unified)
        # ---------------------------------------------------------------------
        prompt = await self.prompt_builder.build_prompt(
            user_input=combined_text,
            search_query=rewritten_query,
            personality_config=persona_cfg,
            system_prompt=system_prompt,
            current_topic=getattr(self, "current_topic", "general"),
            fresh_facts=fresh_facts,  # Pass inline-extracted facts
            stm_summary=stm_summary,  # Pass STM context summary
        )

        # Capture memory_id_map for citation extraction before prompt is converted to string
        memory_id_map = {}
        if isinstance(prompt, dict):
            memory_id_map = prompt.get('memory_id_map', {})
            prompt = self.prompt_builder._assemble_prompt(
                context=prompt,
                user_input=combined_text,
                system_prompt=system_prompt
            )

        # Store memory_id_map for citation extraction
        self._current_memory_id_map = memory_id_map

        return prompt, system_prompt

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
        personality: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Orchestrates the full request:
          - optional personality switch (if provided)
          - commands (early exit)
          - deictic pre-check (optional early clarification)
          - prepare_prompt
          - generate + store
        Returns: (assistant_text, debug_info)
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
                except Exception:
                    if self.logger:
                        self.logger.warning(f"Could not set personality: {personality}")


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
                        except Exception:
                            pass
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

            # --- Build Prompt (unified path) ---
            prompt, system_prompt = await self.prepare_prompt(
                user_input=user_input, files=files, use_raw_mode=use_raw_mode
            )

            # --- Generate Response ---
            active_name_getter = getattr(self.model_manager, "get_active_model_name", None)
            model_name = active_name_getter() if callable(active_name_getter) else None
            model_name = model_name or "gpt-4-turbo"

            # Decide if we should use best-of (non-streaming) for quality; prefer runtime config
            try:
                from config.app_config import (
                    ENABLE_BEST_OF as DEFAULT_ENABLE_BEST_OF,
                    BEST_OF_N,
                    BEST_OF_TEMPS,
                    BEST_OF_MAX_TOKENS,
                    BEST_OF_MIN_QUESTION,
                    BEST_OF_MIN_TOKENS as _BEST_OF_MIN_TOKENS,
                )
            except Exception:
                DEFAULT_ENABLE_BEST_OF = True; BEST_OF_N = 2; BEST_OF_TEMPS = (0.2, 0.7); BEST_OF_MAX_TOKENS = 512; BEST_OF_MIN_QUESTION = True; _BEST_OF_MIN_TOKENS = 8

            _features = (self.config or {}).get("features", {}) if isinstance(self.config, dict) else {}
            _enable_bestof = bool(_features.get("enable_best_of", DEFAULT_ENABLE_BEST_OF))

            use_bestof = False
            try:
                qinfo = analyze_query(user_input)
                use_bestof = bool(
                    _enable_bestof and (
                        (qinfo.is_question and qinfo.token_count >= _BEST_OF_MIN_TOKENS)
                        or (not BEST_OF_MIN_QUESTION)
                    )
                )
            except Exception:
                use_bestof = bool(_enable_bestof)

            # ---------------------------------------------------------------------
            # Determine max_tokens based on tone level AND topic heaviness
            # ---------------------------------------------------------------------
            try:
                from config.app_config import DEFAULT_MAX_TOKENS, HEAVY_TOPIC_MAX_TOKENS
                from utils.tone_detector import CrisisLevel

                # Heavy topics override tone-based limits
                if is_heavy_topic:
                    response_max_tokens = HEAVY_TOPIC_MAX_TOKENS
                    token_reason = "HEAVY topic"
                # Adjust max_tokens based on tone/crisis level for speed and brevity
                elif tone_level == CrisisLevel.CONVERSATIONAL:
                    response_max_tokens = 600  # Force brief responses in conversational mode
                    token_reason = "CONVERSATIONAL mode"
                elif tone_level == CrisisLevel.SUPPORT:
                    response_max_tokens = 1500  # Allow more room for supportive responses
                    token_reason = "SUPPORT mode"
                elif tone_level == CrisisLevel.CRISIS:
                    response_max_tokens = 2000  # Maximum room for crisis responses
                    token_reason = "CRISIS mode"
                else:
                    response_max_tokens = DEFAULT_MAX_TOKENS
                    token_reason = "DEFAULT"

                if self.logger:
                    self.logger.info(
                        f"[Orchestrator] Token limit: {response_max_tokens} ({token_reason})"
                    )
            except Exception as e:
                response_max_tokens = None  # Use model defaults
                if self.logger:
                    self.logger.debug(f"[Orchestrator] Failed to load token config: {e}")

            if use_bestof and not use_raw_mode:
                try:
                    from config.app_config import (
                        BEST_OF_MODEL,
                        BEST_OF_LATENCY_BUDGET_S,
                        BEST_OF_GENERATOR_MODELS as DEF_GEN_MODELS,
                        BEST_OF_SELECTOR_MODELS as DEF_SEL_MODELS,
                        BEST_OF_SELECTOR_MAX_TOKENS as DEF_SEL_MAXTOK,
                        BEST_OF_SELECTOR_WEIGHTS as DEF_SEL_WEIGHTS,
                        BEST_OF_SELECTOR_TOP_K as DEF_SEL_TOPK,
                        BEST_OF_DUEL_MODE as DEF_DUEL_MODE,
                        BEST_OF_MAX_TOKENS as DEF_BEST_MAXTOK,
                    )
                except Exception:
                    BEST_OF_MODEL = None; BEST_OF_LATENCY_BUDGET_S = 2.0
                    DEF_GEN_MODELS = []
                    DEF_SEL_MODELS = []
                    DEF_SEL_MAXTOK = 64
                    DEF_SEL_WEIGHTS = {"heuristic": 1.0, "llm": 0.0}
                    DEF_SEL_TOPK = 0
                    DEF_DUEL_MODE = False
                    DEF_BEST_MAXTOK = 512

                import asyncio as _a
                # Run best-of with an optional latency budget; if disabled (<=0), do not time out
                try:
                    _budget = 0.0
                    try:
                        _budget = float(BEST_OF_LATENCY_BUDGET_S)
                    except Exception:
                        _budget = 0.0

                    # Feature overrides (runtime-configurable via GUI)
                    _runtime = (self.config or {}).get("features", {}) if isinstance(self.config, dict) else {}
                    GEN_MODELS = list(_runtime.get('best_of_generator_models', DEF_GEN_MODELS))
                    SEL_MODELS = list(_runtime.get('best_of_selector_models', DEF_SEL_MODELS))
                    SEL_MAXTOK = int(_runtime.get('best_of_selector_max_tokens', DEF_SEL_MAXTOK))
                    SEL_WEIGHTS = dict(_runtime.get('best_of_selector_weights', DEF_SEL_WEIGHTS)) if isinstance(_runtime.get('best_of_selector_weights', DEF_SEL_WEIGHTS), dict) else DEF_SEL_WEIGHTS
                    SEL_TOPK = int(_runtime.get('best_of_selector_top_k', DEF_SEL_TOPK))
                    BEST_MAXTOK = int(_runtime.get('best_of_max_tokens', DEF_BEST_MAXTOK))

                    # If multi-model generators are configured, use ensemble path
                    use_ensemble = bool(GEN_MODELS)
                    if self.logger:
                        try:
                            self.logger.info(
                                f"[BESTOF] ensemble={use_ensemble} gens={list(GEN_MODELS)} "
                                f"selectors={list(SEL_MODELS)} weights={dict(SEL_WEIGHTS)} "
                                f"top_k={int(SEL_TOPK)} model={(BEST_OF_MODEL or model_name)} max_tokens={int(BEST_MAXTOK)}"
                            )
                        except Exception:
                            pass
                    temps_used = tuple(BEST_OF_TEMPS) if isinstance(BEST_OF_TEMPS, (list, tuple)) else (0.2, 0.7)
                    # Optional strict duel mode (two generators + one judge)
                    _DUEL_MODE = bool(_runtime.get('best_of_duel_mode', DEF_DUEL_MODE))
                    use_duel = bool(_DUEL_MODE and len(GEN_MODELS) == 2 and len(SEL_MODELS) >= 1)
                    if _budget > 0:
                        if use_duel:
                            m1, m2 = list(GEN_MODELS)[:2]
                            judge = list(SEL_MODELS)[0]
                            best_task = _a.create_task(
                                self.response_generator.generate_duel_and_judge(
                                    prompt=prompt,
                                    model_a=m1,
                                    model_b=m2,
                                    judge_model=judge,
                                    system_prompt=system_prompt,
                                    question_text=user_input,
                                    context_hint=prompt,
                                    max_tokens=BEST_MAXTOK,
                                    temperature_a=(temps_used[0] if len(temps_used) > 0 else None),
                                    temperature_b=(temps_used[1] if len(temps_used) > 1 else None),
                                    judge_max_tokens=int(SEL_MAXTOK),
                                )
                            )
                        elif use_ensemble:
                            best_task = _a.create_task(
                                self.response_generator.generate_best_of_ensemble(
                                    prompt=prompt,
                                    generator_models=list(GEN_MODELS),
                                    system_prompt=system_prompt,
                                    question_text=user_input,
                                    context_hint=prompt,
                                    n_total=BEST_OF_N,
                                    temps=temps_used,
                                    max_tokens=BEST_MAXTOK,
                                    selector_models=list(SEL_MODELS),
                                    selector_max_tokens=int(SEL_MAXTOK),
                                    weight_heuristic=float(SEL_WEIGHTS.get("heuristic", 0.5)),
                                    weight_llm=float(SEL_WEIGHTS.get("llm", 0.5)),
                                    judge_top_k=int(SEL_TOPK),
                                )
                            )
                        else:
                            best_task = _a.create_task(
                                self.response_generator.generate_best_of(
                                    prompt=prompt,
                                    model_name=BEST_OF_MODEL or model_name,
                                    system_prompt=system_prompt,
                                    question_text=user_input,
                                    context_hint=prompt,
                                    n=BEST_OF_N,
                                    temps=temps_used,
                                    max_tokens=BEST_MAXTOK,
                                )
                            )
                        full_response = await _a.wait_for(best_task, timeout=_budget)
                    else:
                        if use_duel:
                            m1, m2 = list(GEN_MODELS)[:2]
                            judge = list(SEL_MODELS)[0]
                            full_response = await self.response_generator.generate_duel_and_judge(
                                prompt=prompt,
                                model_a=m1,
                                model_b=m2,
                                judge_model=judge,
                                system_prompt=system_prompt,
                                question_text=user_input,
                                context_hint=prompt,
                                max_tokens=BEST_MAXTOK,
                                temperature_a=(temps_used[0] if len(temps_used) > 0 else None),
                                temperature_b=(temps_used[1] if len(temps_used) > 1 else None),
                                judge_max_tokens=int(SEL_MAXTOK),
                            )
                        elif use_ensemble:
                            full_response = await self.response_generator.generate_best_of_ensemble(
                                prompt=prompt,
                                generator_models=list(GEN_MODELS),
                                system_prompt=system_prompt,
                                question_text=user_input,
                                context_hint=prompt,
                                n_total=BEST_OF_N,
                                temps=temps_used,
                                max_tokens=BEST_MAXTOK,
                                selector_models=list(SEL_MODELS),
                                selector_max_tokens=int(SEL_MAXTOK),
                                weight_heuristic=float(SEL_WEIGHTS.get("heuristic", 0.5)),
                                weight_llm=float(SEL_WEIGHTS.get("llm", 0.5)),
                                judge_top_k=int(SEL_TOPK),
                            )
                        else:
                            full_response = await self.response_generator.generate_best_of(
                                prompt=prompt,
                                model_name=BEST_OF_MODEL or model_name,
                                system_prompt=system_prompt,
                                question_text=user_input,
                                context_hint=prompt,
                                n=BEST_OF_N,
                                temps=temps_used,
                                max_tokens=BEST_MAXTOK,
                            )
                except Exception:
                    # Timeout or error: cancel best-of (if applicable) and proceed with streaming
                    try:
                        if 'best_task' in locals():
                            best_task.cancel()
                    except Exception:
                        pass
                    # Accumulate full response (no yielding yet - need to parse thinking block)
                    full_response = ""
                    async for chunk in self.response_generator.generate_streaming_response(
                        prompt, model_name, system_prompt=system_prompt, max_tokens=response_max_tokens
                    ):
                        full_response += (chunk + " ")
                    full_response = full_response.strip()
            else:
                # Accumulate full response (no yielding yet - need to parse thinking block first)
                full_response = ""
                async for chunk in self.response_generator.generate_streaming_response(
                    prompt, model_name, system_prompt=system_prompt, max_tokens=response_max_tokens
                ):
                    full_response += (chunk + " ")
                full_response = full_response.strip()

            # --- Parse thinking block and extract final answer ---
            thinking_part, final_answer = self._parse_thinking_block(full_response)
            # Strip XML-like wrappers (e.g., <result> … </result>) from final answer
            final_answer = self._strip_xml_wrappers(final_answer)

            # Log thinking part for debugging if present
            if thinking_part:
                if self.logger:
                    self.logger.debug(f"[THINKING BLOCK]\n{thinking_part}")
                debug_info["thinking_length"] = len(thinking_part)

            # Store final answer (not the thinking part) in memory
            answer_for_storage = final_answer if final_answer else self._strip_xml_wrappers(full_response)
            # Strip reflection blocks (they're stored separately as reflection memories)
            answer_for_storage = self._strip_reflection_blocks(answer_for_storage)
            # Sanitize prompt header echoes before returning/storing
            answer_for_storage = self._strip_prompt_artifacts(answer_for_storage)

            # Extract citations if enabled
            citations = []
            if self.enable_citations and hasattr(self, '_current_memory_id_map') and self._current_memory_id_map:
                raw_response_with_citations = answer_for_storage
                answer_for_storage, citations = self._extract_citations(
                    answer_for_storage,
                    self._current_memory_id_map
                )
                debug_info['raw_response_with_citations'] = raw_response_with_citations

            # --- Store Interaction ---
            if self.memory_system and not use_raw_mode:
                try:
                    await self.memory_system.store_interaction(
                        query=user_input,
                        response=answer_for_storage,
                        tags=["conversation"]
                    )
                except Exception:
                    pass
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
                except Exception:
                    pass
            debug_info["error"] = str(e)
            raise
