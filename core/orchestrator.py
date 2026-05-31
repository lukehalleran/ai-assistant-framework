"""
# core/orchestrator.py

Module Contract
- Purpose: High‑level request orchestrator. Prepares prompts (topic detection, optional file processing, query rewrite), invokes the model, and persists the interaction to memory.
- Inputs:
  - user_input: str; optional files; mode flags (raw/enhanced)
  - Wired collaborators: model_manager, response_generator, file_processor, prompt_builder, memory_system, topic/wiki/tokenizer managers
- Outputs:
  - process_user_query() → (assistant_text: str, debug_info: dict)
  - prepare_prompt() → (prompt_text: str, system_prompt: Optional[str])
- Key methods:
  - handle_commands(): simple topic switching commands
  - prepare_prompt(): topic update, file processing, optional query rewrite, resolve system prompt, build prompt via prompt_builder
  - build_full_prompt(): assembles final prompt; delegates system-prompt assembly to _build_system_prompt()
  - process_user_query(): thin coordinator — builds a _QueryFlow and routes through per-phase
    helper methods (behavior-preserving decomposition; final answer + debug_info returned).
  - _check_narrative_freshness(): Startup check for stale narrative context (>24h) [NEW 2026-01-17]
- process_user_query decomposition [REFACTOR 2026-05-30]:
  - State threads through a _QueryFlow dataclass; the coordinator calls, in order:
    _handle_command → _handle_deictic → _resolve_model_name → _build_prompt_phase →
    _maybe_document_generation → _maybe_agentic_search → _resolve_max_tokens →
    _generate_response → _postprocess_response → _store_interaction →
    _run_post_response_detectors → _finalize_debug.
  - Early-exit helpers (_handle_command, _handle_deictic, _maybe_document_generation,
    _maybe_agentic_search) return (text, debug_info) or None (None = fall through).
  - The outer try/except logs, sets debug_info["error"], and re-raises.
- System prompt flow:
  - Composed from file-based personality (config/prompts/default_personality.txt or custom_personality.txt) + immutable operating principles (config/prompts/operating_principles.txt) via load_personality_text() + load_operating_principles(). Falls back to load_system_prompt() if files fail.
  - Performs placeholder substitution: {USER_NAME}, {USER_PRONOUNS}, {PRONOUN_SUBJ}, {PRONOUN_OBJ}, {PRONOUN_POSS}. Forwarded to response generation as system role message.
  - Appends [THREAD CONTEXT] to system prompt for ongoing conversation threads (depth, topic, heavy-topic flag)
  - Proactive thread surfacing: on first message of session, appends instruction to naturally reference [UNRESOLVED THREADS] from prior sessions
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
  - Initializes GitHubManager for read-only GitHub API access [NEW 2026-05-19]
  - Accepts shared user_profile from MemoryCoordinator (avoids duplicate disk read) [CHANGED 2026-05-19]
"""
import re
import processing.gate_system as gate_system
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from core.response_parser import ResponseParser
from utils.logging_utils import get_logger
logger = get_logger("orchestrator")
from integrations.wikipedia_api import WikipediaAPI
from utils.tone_detector import CrisisLevel
from utils.emotional_context import EmotionalContext
from utils.need_detector import NeedType
from core.context_pipeline import ContextPipeline, ContextResult, ToneLevel
from core.best_of_handler import BestOfHandler
from core.escalation_tracker import EscalationTracker
from core.correction_detector import CorrectionDetector, CorrectionEvent
from core.citation_extractor import extract_citations as _ext_extract_citations, expand_citation_range as _ext_expand_citation_range
from core.tone_instructions import get_tone_instructions as _ext_get_tone_instructions, get_response_instructions as _ext_get_response_instructions, get_session_headers_instructions as _ext_get_session_headers_instructions
from core.truth_event_handler import get_recent_profile_facts as _ext_get_recent_profile_facts, apply_truth_event as _ext_apply_truth_event, cascade_entity_resolution as _ext_cascade_entity_resolution, apply_content_attributions as _ext_apply_content_attributions

SYSTEM_PROMPT = "..."  # safe fallback (replace with your real default)
wiki_api = WikipediaAPI()
gate_system.wikipedia_api = wiki_api  # This sets it globally
from utils.query_checker import is_deictic


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

    async def store_interaction(self, query: str, response: str, tags: Optional[List[str]] = None, **kwargs) -> None:
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

@dataclass
class _QueryFlow:
    """Threaded state for DaemonOrchestrator.process_user_query (behavior-preserving refactor)."""
    user_input: str
    files: Optional[List[Any]] = None
    use_raw_mode: bool = False
    personality: Optional[str] = None
    use_agentic_search: bool = False
    debug_info: Dict[str, Any] = field(default_factory=dict)
    model_name: Optional[str] = None
    context: Any = None
    prompt: str = ""
    system_prompt: Optional[str] = None
    prompt_ctx: Optional[Dict[str, Any]] = None
    note_images: List[Any] = field(default_factory=list)
    is_heavy_topic: bool = False
    tone_level: Any = None
    task_timings: Dict[str, Any] = field(default_factory=dict)
    gather_elapsed: float = 0.0
    t_ctx_elapsed: float = 0.0
    t_build_elapsed: float = 0.0
    t_gen_start: float = 0.0
    t_gen_elapsed: float = 0.0
    t_store_elapsed: float = 0.0
    response_max_tokens: Optional[int] = None
    full_response: str = ""
    answer_for_storage: str = ""
    citations: List[Any] = field(default_factory=list)


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
        topic_manager=None,
        wiki_manager=None,
        tokenizer_manager=None,
        conversation_logger=None,
        config: Optional[Dict[str, Any]] = None,
        user_profile=None,
    ):
        self.logger = get_logger("orchestrator")
        self.conversation_logger = conversation_logger  # kept for compatibility if referenced elsewhere
        self.model_manager = model_manager
        self.response_generator = response_generator
        self.file_processor = file_processor
        self.prompt_builder = prompt_builder
        self.memory_system = memory_system
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

        # Use shared user profile (avoids duplicate disk read)
        if user_profile is not None:
            self.user_profile = user_profile
        else:
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

        # Correction/Confirmation Detector — truth score evidence from user messages
        self.correction_detector = None
        try:
            from config.app_config import TRUTH_SCORER_ENABLED, TRUTH_SCORER_CORRECTION_DETECTION
            if TRUTH_SCORER_ENABLED and TRUTH_SCORER_CORRECTION_DETECTION:
                self.correction_detector = CorrectionDetector()
                self.logger.info("[Orchestrator] CorrectionDetector enabled")
            else:
                self.logger.debug("[Orchestrator] CorrectionDetector disabled via config")
        except Exception as e:
            self.logger.warning(f"[Orchestrator] Failed to initialize CorrectionDetector: {e}")

        # Response Planner — pre-answer planning + post-answer review gate
        self.response_planner = None
        self._current_response_plan = None
        try:
            from config.app_config import RESPONSE_PLANNING_ENABLED
            if RESPONSE_PLANNING_ENABLED:
                from core.response_planner import ResponsePlanner
                self.response_planner = ResponsePlanner(model_manager=model_manager)
                self.logger.info("[Orchestrator] ResponsePlanner enabled")
            else:
                self.logger.debug("[Orchestrator] ResponsePlanner disabled via config")
        except Exception as e:
            self.logger.warning(f"[Orchestrator] Failed to initialize ResponsePlanner: {e}")

        # Memory citation system
        self.enable_citations = False  # Will be set from GUI checkbox
        # Pattern matches citation formats: MEM_RECENT_3, MEM_SEMANTIC_4-7, SUM_RECENT_1, REFL_SEMANTIC_2, PROFILE_CONTEXT, WEB_1
        self.citation_pattern = re.compile(
            r'\[('
            r'WEB_\d+|'                   # WEB_1, WEB_2 (web search sources)
            r'MEM_\w+_\d+(?:-\d+)?|'      # MEM_RECENT_3, MEM_SEMANTIC_4-7
            r'SUM_\w+_\d+(?:-\d+)?|'      # SUM_RECENT_1, SUM_SEMANTIC_2-5
            r'REFL_\w+_\d+(?:-\d+)?|'     # REFL_RECENT_1, REFL_SEMANTIC_3
            r'FACT_\d+(?:-\d+)?|'         # FACT_3 (legacy)
            r'PROFILE_\w+'                # PROFILE_CONTEXT
            r')\]'
        )
        self._web_source_map: Dict[str, Dict[str, str]] = {}  # Set per-request by handlers

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

    # ---------- Truth Score Helper Methods ----------

    def _get_recent_profile_facts(self, limit: int = 30) -> list:
        """Gather recent current facts from user profile for correction/confirmation detection."""
        return _ext_get_recent_profile_facts(getattr(self, "user_profile", None), limit)

    def _apply_truth_event(self, event: CorrectionEvent) -> None:
        """Apply a correction/confirmation event to the matching profile fact."""
        return _ext_apply_truth_event(event, getattr(self, "user_profile", None))

    # ---------- Entity Resolution Methods ----------

    _CRISIS_KEYWORDS = frozenset({
        'die', 'died', 'death', 'dead', 'dying', 'icu', 'emergency',
        'hospital', 'dnr', 'critical', 'serious', 'make it', 'losing',
        'loss', 'crisis', 'panic', 'euthan', 'terminal', 'end of life',
    })

    def _cascade_entity_resolution(self, events: list) -> None:
        """Annotate crisis-era summaries/reflections with resolution metadata."""
        return _ext_cascade_entity_resolution(events, getattr(self, "memory_system", None))

    def _apply_content_attributions(self, attributions: list) -> None:
        """Apply retroactive attribution to the most recent shared content."""
        return _ext_apply_content_attributions(attributions, getattr(self, "memory_system", None))

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
        """Return mode-specific response instructions based on detected crisis level."""
        return _ext_get_tone_instructions(tone_level, getattr(self, "user_profile", None))

    def _get_response_instructions(self, ctx: EmotionalContext) -> str:
        """Generate response instructions based on combined emotional context."""
        return _ext_get_response_instructions(ctx, getattr(self, "user_profile", None))

    # ---------- 1b) Session Headers Instructions ----------
    def _get_session_headers_instructions(self) -> str:
        """Return concise instructions about temporal reasoning with prompt headers."""
        return _ext_get_session_headers_instructions()

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

            # Get ChromaDB store for memory search if available
            chroma_store = None
            if self.memory_system and hasattr(self.memory_system, 'chroma_store'):
                chroma_store = self.memory_system.chroma_store

            # Initialize file access manager if configured
            file_access_manager = None
            try:
                from config.app_config import (
                    FILE_ACCESS_ENABLED,
                    FILE_ACCESS_APPROVED_FOLDERS,
                    FILE_ACCESS_MAX_READ_BYTES,
                    FILE_ACCESS_MAX_GREP_RESULTS,
                    FILE_ACCESS_MAX_LIST_ENTRIES,
                    FILE_ACCESS_ALLOWED_EXTENSIONS,
                )
                if FILE_ACCESS_ENABLED and FILE_ACCESS_APPROVED_FOLDERS:
                    from core.file_access_manager import FileAccessManager
                    file_access_manager = FileAccessManager(
                        approved_folders=FILE_ACCESS_APPROVED_FOLDERS,
                        max_read_bytes=FILE_ACCESS_MAX_READ_BYTES,
                        max_grep_results=FILE_ACCESS_MAX_GREP_RESULTS,
                        max_list_entries=FILE_ACCESS_MAX_LIST_ENTRIES,
                        allowed_extensions=FILE_ACCESS_ALLOWED_EXTENSIONS,
                    )
                    if self.logger:
                        self.logger.info(
                            f"[Orchestrator] File access manager initialized "
                            f"({len(FILE_ACCESS_APPROVED_FOLDERS)} approved folders)"
                        )
            except ImportError as e:
                if self.logger:
                    self.logger.debug(f"[Orchestrator] File access manager not available: {e}")

            # Initialize git stats manager if configured
            git_stats_manager = None
            try:
                from config.app_config import (
                    GIT_STATS_ENABLED,
                    GIT_STATS_TIMEOUT,
                    GIT_STATS_MAX_OUTPUT_LINES,
                )
                if GIT_STATS_ENABLED:
                    from core.git_stats_manager import GitStatsManager
                    git_stats_manager = GitStatsManager(
                        timeout=GIT_STATS_TIMEOUT,
                        max_output_lines=GIT_STATS_MAX_OUTPUT_LINES,
                    )
                    if git_stats_manager.is_available():
                        if self.logger:
                            self.logger.info("[Orchestrator] Git stats manager initialized")
                    else:
                        git_stats_manager = None
                        if self.logger:
                            self.logger.debug("[Orchestrator] Git stats: not in a git repository")
            except ImportError as e:
                if self.logger:
                    self.logger.debug(f"[Orchestrator] Git stats manager not available: {e}")

            # Initialize GitHub API manager if configured
            github_manager = None
            try:
                from config.app_config import (
                    GITHUB_API_ENABLED,
                    GITHUB_API_TIMEOUT,
                    GITHUB_API_MAX_OUTPUT_LINES,
                    GITHUB_API_REPO,
                )
                if GITHUB_API_ENABLED:
                    from core.github_manager import GitHubManager
                    github_manager = GitHubManager(
                        repo=GITHUB_API_REPO if GITHUB_API_REPO else None,
                        timeout=GITHUB_API_TIMEOUT,
                        max_output_lines=GITHUB_API_MAX_OUTPUT_LINES,
                    )
                    if github_manager.is_available():
                        if self.logger:
                            self.logger.info("[Orchestrator] GitHub API manager initialized")
                    else:
                        github_manager = None
                        if self.logger:
                            self.logger.debug("[Orchestrator] GitHub API: gh CLI not available")
            except ImportError as e:
                if self.logger:
                    self.logger.debug(f"[Orchestrator] GitHub API manager not available: {e}")

            self._agentic_controller = AgenticSearchController(
                model_manager=self.model_manager,
                web_search_manager=web_search_manager,
                chroma_store=chroma_store,
                wolfram_manager=wolfram_manager,
                sandbox_manager=sandbox_manager,
                file_access_manager=file_access_manager,
                git_stats_manager=git_stats_manager,
                github_manager=github_manager,
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

    def _build_system_prompt(self, context: ContextResult, use_raw_mode: bool = False) -> str:
        """
        Assemble the system prompt from personality/principles + all contextual injections.

        Extracted verbatim from build_full_prompt (behavior-preserving). Reads only self
        collaborators (enable_citations, escalation_tracker, model_manager, time_manager,
        user_profile) and the ContextResult; returns the fully assembled system prompt.
        """
        SYSTEM_PROMPT_FALLBACK = (
            "You are Daemon, a helpful assistant with memory and RAG. "
            "Be direct, truthful, concise."
        )
        system_prompt: str = SYSTEM_PROMPT_FALLBACK

        # Load personality + operating principles (composed from separate files)
        try:
            from config.app_config import load_personality_text, load_operating_principles, PERSONALITY_CUSTOM_PATH
            from pathlib import Path as _Path
            _personality = load_personality_text()
            _principles = load_operating_principles()
            _using_custom = _Path(PERSONALITY_CUSTOM_PATH).exists()
            if _personality and _principles:
                system_prompt = _personality + "\n\n" + _principles
                logger.info(
                    f"[Orchestrator] System prompt composed: "
                    f"personality={'CUSTOM' if _using_custom else 'default'} ({len(_personality)} chars) "
                    f"+ principles ({len(_principles)} chars) = {len(system_prompt)} chars"
                )
            elif _personality:
                system_prompt = _personality
                logger.warning("[Orchestrator] Operating principles missing, using personality only")
            elif _principles:
                system_prompt = _principles
                logger.warning("[Orchestrator] Personality missing, using principles only")
        except (ImportError, Exception) as e:
            logger.warning(f"[Orchestrator] Personality/principles load failed: {e}")
            # Fallback to monolithic system_prompt.txt
            try:
                from config.app_config import load_system_prompt
                loaded = load_system_prompt({})
                if isinstance(loaded, str) and loaded.strip():
                    system_prompt = loaded
                    logger.info("[Orchestrator] Fell back to monolithic system_prompt.txt")
            except (ImportError, AttributeError):
                pass

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

            # --- Proactive thread surfacing (first message only) ---
            try:
                from config.app_config import THREAD_SURFACING_ENABLED
                if THREAD_SURFACING_ENABLED:
                    is_first_message = False
                    if hasattr(self, 'time_manager') and self.time_manager:
                        try:
                            gap = self.time_manager.time_since_previous_message()
                            is_first_message = isinstance(gap, str) and "N/A" in gap
                        except (AttributeError, TypeError):
                            pass
                    if is_first_message:
                        system_prompt = system_prompt.rstrip() + (
                            "\n\n## PROACTIVE THREAD SURFACING\n"
                            "The [UNRESOLVED THREADS] section in the prompt contains open threads from prior sessions "
                            "(commitments the user made, deadlines, unfinished topics, unanswered questions).\n"
                            "Since this is the START of a new session:\n"
                            "- Naturally weave 1-2 relevant threads into your greeting (don't list them all)\n"
                            "- Keep it conversational, NOT a bulleted task list\n"
                            "- Prioritize the user's current query first; mention threads secondarily\n"
                            "- If no threads are present or relevant, just greet normally\n"
                            "- Never fabricate threads — only reference what appears in [UNRESOLVED THREADS]\n"
                        )
            except (ImportError, Exception):
                pass

            # --- Codebase change awareness (first message only) ---
            try:
                from config.app_config import SESSION_DIFF_ENABLED
                if SESSION_DIFF_ENABLED:
                    _is_first = False
                    if hasattr(self, 'time_manager') and self.time_manager:
                        try:
                            _gap = self.time_manager.time_since_previous_message()
                            _is_first = isinstance(_gap, str) and "N/A" in _gap
                        except (AttributeError, TypeError):
                            pass
                    if _is_first:
                        system_prompt = system_prompt.rstrip() + (
                            "\n\n## CODEBASE CHANGE AWARENESS\n"
                            "The [CODEBASE CHANGES SINCE LAST SESSION] section lists files that "
                            "changed since your last conversation.\n"
                            "- If you discussed implementing a feature, check if relevant files "
                            "appear in the changes — if so, acknowledge the implementation.\n"
                            "- Do NOT list every change. Only mention changes relevant to conversation.\n"
                        )
            except (ImportError, Exception):
                pass

        # --- Thinking block instruction ---
        # Skip for models with native reasoning (Claude, DeepSeek-R1) — their
        # thinking is separated at the API level via extra_body.  The prompt
        # instruction is redundant and can cause the model to echo it.
        if not use_raw_mode:
            _active = getattr(self.model_manager, "get_active_model_name", lambda: None)()
            _has_native = (
                _active
                and hasattr(self.model_manager, "supports_reasoning")
                and self.model_manager.supports_reasoning(_active)
            )
            if not _has_native:
                thinking_instruction = (
                    "\n\n[IMPORTANT] Before your final response, include your reasoning "
                    "in <thinking>...</thinking> tags. Walk through the context step-by-step, "
                    "then provide your answer outside the tags."
                )
                system_prompt = system_prompt.rstrip() + thinking_instruction
        return system_prompt

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
        system_prompt = self._build_system_prompt(context, use_raw_mode)

        # --- 2) Build prompt context (+ optional response plan in parallel) ---
        _plan_result = None
        if self.response_planner and self.response_planner.should_plan(context):
            import asyncio as _aio
            _plan_task = _aio.create_task(
                self.response_planner.create_plan(
                    query=context.original_query,
                    context=context,
                )
            )
            _prompt_task = _aio.create_task(
                self.prompt_builder.build_prompt_from_context(context)
            )
            done, _ = await _aio.wait(
                [_plan_task, _prompt_task],
                timeout=35.0,
                return_when=_aio.ALL_COMPLETED,
            )
            try:
                prompt_ctx = _prompt_task.result()
            except Exception as e:
                logger.warning(f"[BUILD_FULL_PROMPT] Prompt build failed: {e}")
                prompt_ctx = {}
            try:
                _plan_result = _plan_task.result()
            except Exception as e:
                logger.debug(f"[BUILD_FULL_PROMPT] Response planner failed (non-fatal): {e}")
                _plan_result = None
        else:
            prompt_ctx = await self.prompt_builder.build_prompt_from_context(context)

        # Inject plan into system prompt
        if _plan_result:
            system_prompt = system_prompt.rstrip() + self.response_planner.format_plan_injection(_plan_result)
            logger.info(f"[BUILD_FULL_PROMPT] Response plan injected: {len(_plan_result.key_points)} points, tone={_plan_result.tone}")

        # Store plan on instance for review gate in handlers.py
        self._current_response_plan = _plan_result

        # Store memory_id_map for citation extraction
        self._current_memory_id_map = prompt_ctx.get('memory_id_map', {})
        # Store web source map for web citation extraction (populated by _assemble_prompt)
        self._web_source_map = prompt_ctx.get('_web_source_map', {})

        # Forward intent classification to prompt_ctx for agentic gate decisions
        if context.intent is not None:
            prompt_ctx['intent'] = context.intent

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
        _t_ctx = time.perf_counter()
        context = await self.build_context(
            user_input=user_input,
            files=files,
            use_raw_mode=use_raw_mode,
        )
        _ctx_elapsed = time.perf_counter() - _t_ctx

        # Step 2: Raw mode returns early
        if use_raw_mode:
            return context.file_context or user_input, None

        # Step 3: Build full prompt (with optional raw context for agentic search)
        _t_build = time.perf_counter()
        result = await self.build_full_prompt(
            context=context,
            use_raw_mode=use_raw_mode,
            return_raw_context=return_context,
        )
        _build_elapsed = time.perf_counter() - _t_build

        duration = time.perf_counter() - _t_start
        if self.logger:
            self.logger.info(f"[PREPARE_PROMPT] Completed in {duration:.2f}s (via ContextPipeline)")

        # Stash phase timings on instance for debug_info consumption
        self._last_phase_timings = {
            "context_pipeline": round(_ctx_elapsed, 3),
            "prompt_build": round(_build_elapsed, 3),
            "prepare_total": round(duration, 3),
        }
        # Extract per-task timings from prompt_ctx (stashed by builder)
        if return_context and isinstance(result, tuple) and len(result) >= 3:
            _pctx = result[2] or {}
            self._last_task_timings = _pctx.pop("_task_timings", {})
            self._last_gather_elapsed = _pctx.pop("_gather_elapsed", 0.0)
            _pctx.pop("_build_time", None)
        else:
            self._last_task_timings = {}
            self._last_gather_elapsed = 0.0

        return result

    # ---------- Memory Citation Methods ----------
    def _expand_citation_range(self, mem_id: str) -> List[str]:
        """Expand a range citation like MEM_RECENT_4-7 into individual IDs."""
        return _ext_expand_citation_range(mem_id)

    def _extract_citations(self, response: str, memory_map: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract memory and web citations from response."""
        return _ext_extract_citations(response, memory_map, getattr(self, "_web_source_map", None) or {})

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
        flow = _QueryFlow(
            user_input=user_input,
            files=files,
            use_raw_mode=use_raw_mode,
            personality=personality,
            use_agentic_search=use_agentic_search,
        )
        flow.debug_info = {
            "start_time": datetime.now(),
            "user_input": user_input[:100],
            "files_count": len(files) if files else 0,
            "mode": "raw" if use_raw_mode else "enhanced",
        }
        debug_info = flow.debug_info

        try:
            early = self._handle_command(flow)
            if early is not None:
                return early

            early = await self._handle_deictic(flow)
            if early is not None:
                return early

            self._resolve_model_name(flow)
            await self._build_prompt_phase(flow)

            early = await self._maybe_document_generation(flow)
            if early is not None:
                return early

            early = await self._maybe_agentic_search(flow)
            if early is not None:
                return early

            self._resolve_max_tokens(flow)
            await self._generate_response(flow)
            self._postprocess_response(flow)
            await self._store_interaction(flow)
            self._run_post_response_detectors(flow)
            self._finalize_debug(flow)

            return flow.answer_for_storage, debug_info

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

    def _handle_command(self, flow):
        """Commands: early exit. Returns (text, debug_info) or None to fall through."""
        user_input = flow.user_input
        debug_info = flow.debug_info
        cmd = self.handle_commands(user_input)
        if cmd:
            # shape matches handler expectations: (text, debug_info)
            text, meta = cmd
            debug_info.update(meta)
            return text, debug_info
        return None

    async def _handle_deictic(self, flow):
        """Deictic pre-check: clarify before we build/stream. Returns clarification tuple or None."""
        user_input = flow.user_input
        use_raw_mode = flow.use_raw_mode
        debug_info = flow.debug_info
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
        return None

    def _resolve_model_name(self, flow):
        """Determine target model name (needed for multimodal image loading)."""
        active_name_getter = getattr(self.model_manager, "get_active_model_name", None)
        model_name = active_name_getter() if callable(active_name_getter) else None
        model_name = model_name or "gpt-4-turbo"

        if self.logger:
            self.logger.info(f"[Orchestrator] Target model for response: {model_name}")
        flow.model_name = model_name

    async def _build_prompt_phase(self, flow):
        """Build context + full prompt (timed), update escalation tracker, stash phase state."""
        user_input = flow.user_input
        files = flow.files
        use_raw_mode = flow.use_raw_mode
        personality = flow.personality
        debug_info = flow.debug_info
        import time as _time_mod
        _t_ctx_start = _time_mod.perf_counter()
        context = await self.build_context(
            user_input=user_input,
            files=files,
            use_raw_mode=use_raw_mode,
            personality=personality,
        )
        _t_ctx_elapsed = _time_mod.perf_counter() - _t_ctx_start

        _t_build_start = _time_mod.perf_counter()
        prompt, system_prompt, prompt_ctx = await self.build_full_prompt(
            context=context,
            use_raw_mode=use_raw_mode,
            return_raw_context=True,  # Get prompt context for images
        )
        _t_build_elapsed = _time_mod.perf_counter() - _t_build_start

        # Extract per-task timings stashed by builder
        _task_timings = prompt_ctx.pop("_task_timings", {}) if prompt_ctx else {}
        _gather_elapsed = prompt_ctx.pop("_gather_elapsed", 0.0) if prompt_ctx else 0.0
        _builder_time = prompt_ctx.pop("_build_time", 0.0) if prompt_ctx else 0.0

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
        _t_gen_start = _time_mod.perf_counter()
        flow.context = context
        flow.prompt = prompt
        flow.system_prompt = system_prompt
        flow.prompt_ctx = prompt_ctx
        flow.note_images = note_images
        flow.is_heavy_topic = is_heavy_topic
        flow.tone_level = tone_level
        flow.t_ctx_elapsed = _t_ctx_elapsed
        flow.t_build_elapsed = _t_build_elapsed
        flow.task_timings = _task_timings
        flow.gather_elapsed = _gather_elapsed
        flow.t_gen_start = _t_gen_start

    async def _maybe_document_generation(self, flow):
        """Document-generation bypass. Returns (text, debug) on success, else None (fall through)."""
        user_input = flow.user_input
        use_raw_mode = flow.use_raw_mode
        use_agentic_search = flow.use_agentic_search
        debug_info = flow.debug_info
        system_prompt = flow.system_prompt
        model_name = flow.model_name
        prompt_ctx = flow.prompt_ctx
        prompt = flow.prompt
        _doc_intent = None
        if use_agentic_search and not use_raw_mode and self.agentic_controller:
            try:
                from knowledge.document_generator import detect_document_intent
                _doc_intent = detect_document_intent(user_input)
            except Exception:
                pass

        if _doc_intent and self.agentic_controller:
            try:
                if self.logger:
                    self.logger.info(
                        f"[Orchestrator] Document generation detected: "
                        f"topic={_doc_intent['topic']}, type={_doc_intent['doc_type']}"
                    )

                from core.agentic import ProgressEvent

                # Augment user query with explicit generate_document instruction
                doc_query = (
                    f"{user_input}\n\n"
                    f"[SYSTEM: The user wants a {_doc_intent['doc_type']} document. "
                    f"Use the generate_document tool with topic=\"{_doc_intent['topic']}\", "
                    f"doc_type=\"{_doc_intent['doc_type']}\""
                    + (f", focus=\"{_doc_intent['focus']}\"" if _doc_intent.get("focus") else "")
                    + ".]"
                )

                full_response = ""
                async for event_or_chunk in self.agentic_controller.run_agentic_search(
                    query=doc_query,
                    system_prompt=system_prompt or "",
                    model_name=model_name,
                    initial_search_terms=[_doc_intent["topic"]],
                    initial_context=prompt_ctx,
                    crisis_level=str(self.current_tone_level) if self.current_tone_level else None,
                ):
                    if isinstance(event_or_chunk, ProgressEvent):
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
                        full_response += event_or_chunk

                if self.memory_system:
                    try:
                        await self.memory_system.store_interaction(
                            query=user_input,
                            response=full_response.strip(),
                            tags=["document_generation"]
                        )
                    except Exception:
                        pass

                debug_info.update({
                    "response_length": len(full_response),
                    "end_time": datetime.now(),
                    "prompt_length": len(prompt),
                    "document_generation_used": True,
                    "doc_intent": _doc_intent,
                })
                debug_info["duration"] = (debug_info["end_time"] - debug_info["start_time"]).total_seconds()

                return full_response.strip(), debug_info

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[Orchestrator] Document generation failed, falling back: {e}")
                debug_info["doc_gen_error"] = str(e)
        return None

    async def _maybe_agentic_search(self, flow):
        """Agentic-search bypass. Returns (text, debug) on success, else None (fall through)."""
        user_input = flow.user_input
        use_raw_mode = flow.use_raw_mode
        use_agentic_search = flow.use_agentic_search
        debug_info = flow.debug_info
        system_prompt = flow.system_prompt
        model_name = flow.model_name
        prompt_ctx = flow.prompt_ctx
        prompt = flow.prompt
        _t_ctx_elapsed = flow.t_ctx_elapsed
        _t_build_elapsed = flow.t_build_elapsed
        _task_timings = flow.task_timings
        _gather_elapsed = flow.gather_elapsed
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
                        initial_context=prompt_ctx,
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
                    debug_info["phase_timings"] = {
                        "context_pipeline": round(_t_ctx_elapsed, 3),
                        "prompt_build": round(_t_build_elapsed, 3),
                        "llm_generation": round(debug_info["duration"] - _t_ctx_elapsed - _t_build_elapsed, 3),
                    }
                    debug_info["task_timings"] = {k: round(v, 3) for k, v in _task_timings.items()}
                    debug_info["gather_elapsed"] = round(_gather_elapsed, 3)

                    return full_response.strip(), debug_info

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[Orchestrator] Agentic search failed, falling back: {e}")
                debug_info["agentic_error"] = str(e)
        return None

    def _resolve_max_tokens(self, flow):
        """Determine max_tokens based on tone level and topic heaviness (+ escalation override)."""
        is_heavy_topic = flow.is_heavy_topic
        tone_level = flow.tone_level
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
        flow.response_max_tokens = response_max_tokens

    async def _generate_response(self, flow):
        """Best-of / duel / ensemble or standard streaming generation."""
        import time as _time_mod
        user_input = flow.user_input
        use_raw_mode = flow.use_raw_mode
        prompt = flow.prompt
        system_prompt = flow.system_prompt
        model_name = flow.model_name
        response_max_tokens = flow.response_max_tokens
        note_images = flow.note_images
        debug_info = flow.debug_info
        _t_gen_start = flow.t_gen_start
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

        _t_gen_elapsed = _time_mod.perf_counter() - _t_gen_start
        flow.full_response = full_response
        flow.t_gen_elapsed = _t_gen_elapsed

    def _postprocess_response(self, flow):
        """Parse thinking block, strip artifacts, extract citations, record response in escalation."""
        full_response = flow.full_response
        debug_info = flow.debug_info
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

        # Extract citations — auto-enable when web search results present
        citations = []
        has_web_sources = bool(getattr(self, '_web_source_map', None))
        should_extract = self.enable_citations or has_web_sources
        if should_extract and (
            (hasattr(self, '_current_memory_id_map') and self._current_memory_id_map)
            or has_web_sources
        ):
            raw_response_with_citations = answer_for_storage
            answer_for_storage, citations = self._extract_citations(
                answer_for_storage,
                self._current_memory_id_map if hasattr(self, '_current_memory_id_map') else {}
            )
            debug_info['raw_response_with_citations'] = raw_response_with_citations
            if has_web_sources:
                debug_info['web_source_map'] = self._web_source_map

        # Record response in escalation tracker for engagement detection
        if self.escalation_tracker:
            self.escalation_tracker.record_response(answer_for_storage)
        flow.answer_for_storage = answer_for_storage
        flow.citations = citations

    async def _store_interaction(self, flow):
        """Persist the exchange to memory (skipped in raw mode)."""
        import time as _time_mod
        user_input = flow.user_input
        use_raw_mode = flow.use_raw_mode
        answer_for_storage = flow.answer_for_storage
        _t_store_start = _time_mod.perf_counter()
        if self.memory_system and not use_raw_mode:
            try:
                await self.memory_system.store_interaction(
                    query=user_input,
                    response=answer_for_storage,
                    tags=["conversation"],
                    session_id=getattr(self.memory_system, 'session_id', None),
                )
            except Exception as e:
                self.logger.error(f"[Orchestrator] CRITICAL: Failed to store interaction - data loss: {e}")
        _t_store_elapsed = _time_mod.perf_counter() - _t_store_start
        # Use instance logger
        if self.logger:
            self.logger.debug("[orchestrator] Persisted exchange; considering consolidation")
        flow.t_store_elapsed = _t_store_elapsed

    def _run_post_response_detectors(self, flow):
        """Truth/correction/confirmation detection, staleness cascade, entity + attribution detection."""
        user_input = flow.user_input
        if self.user_profile and self.correction_detector:
            try:
                recent_facts = self._get_recent_profile_facts()
                events = self.correction_detector.detect_corrections(user_input, recent_facts)
                events += self.correction_detector.detect_confirmations(user_input, recent_facts)
                correction_events = []
                for event in events:
                    self._apply_truth_event(event)
                    if event.event_type == "correction":
                        correction_events.append(event)

                # --- Staleness cascade for corrections ---
                if correction_events:
                    try:
                        from config.app_config import STALENESS_ENABLED
                        claim_index = getattr(self.memory_system, 'claim_index', None) if self.memory_system else None
                        if STALENESS_ENABLED and claim_index:
                            from memory.claim_tracker import canonicalize_claim
                            entity_resolver = getattr(self.memory_system, 'entity_resolver', None)
                            for event in correction_events:
                                ck = canonicalize_claim(
                                    "user", event.relation,
                                    entity_resolver=entity_resolver,
                                )
                                affected = claim_index.cascade_staleness(
                                    ck,
                                    chroma_store=self.memory_system.chroma_store if self.memory_system else None,
                                )
                                if affected:
                                    self.logger.info(
                                        f"[Staleness] Cascade from correction on '{event.relation}': "
                                        f"{len(affected)} document(s) updated"
                                    )
                    except Exception as se:
                        self.logger.debug(f"[Staleness] Cascade failed (non-fatal): {se}")

            except Exception as e:
                self.logger.warning(f"[Orchestrator] Truth event detection failed: {e}")

        # --- Entity correction detection (non-profile entities) ---
        if self.correction_detector:
            try:
                entity_corrections = self.correction_detector.detect_entity_corrections(user_input)
                if entity_corrections:
                    self._cascade_entity_resolution(entity_corrections)
            except Exception as e:
                self.logger.debug(f"[Orchestrator] Entity correction detection failed (non-fatal): {e}")

        # --- Retroactive content attribution ---
        if self.correction_detector:
            try:
                attributions = self.correction_detector.detect_attributions(user_input)
                if attributions:
                    self._apply_content_attributions(attributions)
            except Exception as e:
                self.logger.debug(f"[Orchestrator] Attribution detection failed (non-fatal): {e}")

    def _finalize_debug(self, flow):
        """Finalize debug_info: response/prompt lengths, duration, phase + task timings."""
        debug_info = flow.debug_info
        answer_for_storage = flow.answer_for_storage
        full_response = flow.full_response
        prompt = flow.prompt
        citations = flow.citations
        _t_ctx_elapsed = flow.t_ctx_elapsed
        _t_build_elapsed = flow.t_build_elapsed
        _t_gen_elapsed = flow.t_gen_elapsed
        _t_store_elapsed = flow.t_store_elapsed
        _task_timings = flow.task_timings
        _gather_elapsed = flow.gather_elapsed
        debug_info.update({
            "response_length": len(answer_for_storage),
            "full_response_length": len(full_response),
            "end_time": datetime.now(),
            "prompt_length": len(prompt),
            "citations": citations,  # Add extracted citations
            "citations_enabled": self.enable_citations,
        })
        debug_info["duration"] = (debug_info["end_time"] - debug_info["start_time"]).total_seconds()

        # Phase-level timing for interpretability
        debug_info["phase_timings"] = {
            "context_pipeline": round(_t_ctx_elapsed, 3),
            "prompt_build": round(_t_build_elapsed, 3),
            "llm_generation": round(_t_gen_elapsed, 3),
            "memory_store": round(_t_store_elapsed, 3),
        }
        debug_info["task_timings"] = {k: round(v, 3) for k, v in _task_timings.items()}
        debug_info["gather_elapsed"] = round(_gather_elapsed, 3)
