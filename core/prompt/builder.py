"""
# core/prompt/builder.py

Module Contract
- Purpose: Main UnifiedPromptBuilder orchestrating context retrieval, token budget
  management, and prompt assembly coordination. Delegates formatting to PromptFormatter
  and hygiene to ContentHygiene.
- Key methods:
  - build_prompt(user_input, config, context_result, ...) -> Dict
    Main entry point: parallel retrieval → hygiene → token budget → returns context dict.
    Sets/clears intent weight overrides and graph refs on scorer around retrieval.
  - build_prompt_from_context(user_input, config, context_result, ...) -> Dict
    Lightweight path skipping full retrieval (uses pre-gathered context).
  - _llm_compress_oversized(context) -> Dict
    Pre-compresses items ≥3x over token limit via LLM before middle-out fallback.
  - _assemble_prompt(context, user_input, directives, system_prompt) -> str
    Delegates to PromptFormatter._assemble_prompt().
  - _build_feature_inventory(context) -> str
    Delegates to PromptFormatter._build_feature_inventory().
  - _hygiene_and_caps(context, stm_summary) -> Dict
    Delegates to ContentHygiene._hygiene_and_caps().
  - _backfill_recent_conversations(...) -> List
    Delegates to ContentHygiene._backfill_recent_conversations().
  - Post-budget floors (Step 7.1): Guarantees minimum recent_conversations (PROMPT_MIN_RECENT_FLOOR=5),
    summaries (PROMPT_MAX_SUMMARIES), and reflections (PROMPT_MAX_REFLECTIONS) survive budget trimming.
  - Skill activation (Step 5): Creates SkillActivationPolicy in __init__, over-fetches procedural
    skills by SKILL_ACTIVATION_FETCH_MULTIPLIER (3x), applies policy after parallel gather completes
    (intent suppression, score threshold, STM bonus, cooldown filter, cap to max_skills).
- Outputs:
  - Context dictionary with all assembled data, metadata, and performance metrics
  - Formatted prompt string via _assemble_prompt delegation
- Dependencies:
  - .context_gatherer.ContextGatherer (parallel async data retrieval)
  - .formatter.PromptFormatter (section assembly, feature inventory, moved module-level helpers)
  - .hygiene.ContentHygiene (dedup, caps, backfill)
  - .summarizer.LLMSummarizer (on-demand reflections and summaries)
  - .token_manager.TokenManager (budget enforcement, middle-out compression)
  - .base._FallbackMemoryCoordinator (testing fallback)
  - memory.skill_activation.SkillActivationPolicy (post-retrieval skill filter)
  - memory.skill_activation.SkillCooldownStore (JSON-backed cooldown tracking)
  - processing.gate_system (relevance filtering)
  - memory.memory_scorer (intent weight overrides, graph refs set/cleared per call)
- Re-exports from .formatter (backward compatibility):
  - _staleness_prefix, _is_multimodal_model, _load_upload_image
- Side effects:
  - Memory system queries and parallel data retrieval
  - LLM API calls for summarization, reflection, and oversized item compression
  - Sets/clears _intent_weight_overrides and _graph_memory/_entity_resolver on scorer
  - Comprehensive logging and performance metrics
"""

import os
import time
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from utils.time_manager import TimeManager
from utils.query_checker import analyze_query
from memory.memory_consolidator import MemoryConsolidator
from utils.logging_utils import get_logger

# Import the modular components
from .context_gatherer import (
    ContextGatherer,
    PROMPT_MAX_RECENT_SUMMARIES,
    PROMPT_MAX_SEMANTIC_SUMMARIES,
    PROMPT_MAX_RECENT_REFLECTIONS,
    PROMPT_MAX_SEMANTIC_REFLECTIONS
)
from .formatter import (
    PromptFormatter, _parse_bool, _dedupe_keep_order, _sanitize_embedded_headers,
    _staleness_prefix, _is_multimodal_model, _load_upload_image,
)
from .summarizer import LLMSummarizer
from .token_manager import TokenManager
from .base import _FallbackMemoryCoordinator
from .hygiene import ContentHygiene
from memory.skill_activation import SkillActivationPolicy, SkillCooldownStore

logger = get_logger("prompt_builder")

# Configuration loading
try:
    from config.app_config import config as _APP_CFG
    _MEM_CFG = (_APP_CFG.get("memory") or {})
except (ImportError, AttributeError) as e:
    logger.warning(f"[PromptBuilder] Could not load memory config: {e}, using defaults")
    _MEM_CFG = {}

def _cfg_int(key: str, default_val: int) -> int:
    try:
        v = _MEM_CFG.get(key, default_val)
        return int(v) if v is not None else int(default_val)
    except (ValueError, TypeError) as e:
        logger.debug(f"[PromptBuilder] Bad config value for '{key}': {e}, using default {default_val}")
        return int(default_val)

# Token and model configuration
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "4096"))
RESERVE_FOR_COMPLETION = int(os.getenv("RESERVE_FOR_COMPLETION", "1024"))

# Model-aware token budget (replaces static PROMPT_TOKEN_BUDGET = 15000)
try:
    from config.app_config import (
        PROMPT_TOKEN_BUDGET_OVERRIDE,
        PROMPT_TOKEN_BUDGET_DEFAULT,
        PROMPT_TOKEN_BUDGET_LOCAL,
        PROMPT_TOKEN_BUDGET_FLOOR,
        PROMPT_TOKEN_BUDGET_CEILING,
        PROMPT_TOKEN_BUDGET_CONTEXT_FRACTION,
    )
except ImportError:
    PROMPT_TOKEN_BUDGET_OVERRIDE = None
    PROMPT_TOKEN_BUDGET_DEFAULT = 40000
    PROMPT_TOKEN_BUDGET_LOCAL = 12000
    PROMPT_TOKEN_BUDGET_FLOOR = 8000
    PROMPT_TOKEN_BUDGET_CEILING = 60000
    PROMPT_TOKEN_BUDGET_CONTEXT_FRACTION = 0.25

# LLM compression config (smart compression for heavily oversized items)
try:
    from config.app_config import (
        LLM_COMPRESSION_ENABLED,
        LLM_COMPRESSION_MODEL,
        LLM_COMPRESSION_TIMEOUT,
        LLM_COMPRESSION_RATIO_THRESHOLD,
        LLM_COMPRESSION_MAX_BATCH,
    )
except ImportError:
    LLM_COMPRESSION_ENABLED = True
    LLM_COMPRESSION_MODEL = "gpt-4o-mini"
    LLM_COMPRESSION_TIMEOUT = 3.0
    LLM_COMPRESSION_RATIO_THRESHOLD = 3.0
    LLM_COMPRESSION_MAX_BATCH = 8


# ---------------------------------------------------------------------------
# Eval snapshot hook (gated, read-only, disabled by default)
# ---------------------------------------------------------------------------

def _eval_capture_enabled() -> bool:
    """Check if eval snapshot capture is enabled via environment variable."""
    return os.environ.get("DAEMON_EVAL_CAPTURE", "0") == "1"


def _eval_capture_strict() -> bool:
    """Check if eval capture should raise on errors (vs log warnings)."""
    return os.environ.get("DAEMON_EVAL_CAPTURE_STRICT", "0") == "1"


def _maybe_capture_eval_snapshot(
    context: Dict[str, Any],
    user_input: str,
    sections: list,
    final_prompt: str,
) -> None:
    """Gated eval snapshot hook. Does nothing unless DAEMON_EVAL_CAPTURE=1.

    This function is read-only: it does not mutate context, sections, or prompt.
    It captures the post-hygiene assembled prompt for eval replay and saves it
    to disk. Failures log warnings but do not break normal chat (unless strict mode).
    """
    if not _eval_capture_enabled():
        return

    try:
        # Lazy import to avoid loading eval modules during normal operation
        from eval.snapshots import SnapshotCapture, save_snapshot
        from eval.schema import PromptProvenance
        from eval.section_registry import match_header_to_key
        from datetime import datetime, timezone
        import subprocess

        # Build formatted_sections map from the sections list
        formatted_sections: Dict[str, str] = {}
        for section_text in sections:
            if not section_text:
                continue
            first_line = section_text.split("\n", 1)[0]
            key = match_header_to_key(first_line)
            if key:
                formatted_sections[key] = section_text
            else:
                # Try to detect [CURRENT USER QUERY] which has a nested structure
                if "[CURRENT USER QUERY]" in first_line:
                    formatted_sections["current_query"] = section_text

        # Build provenance
        git_hash = ""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                git_hash = result.stdout.strip()
        except Exception:
            pass

        provenance = PromptProvenance(
            model_name="",  # Not available in builder context
            git_commit_hash=git_hash,
            system_prompt_hash="",  # System prompt is in orchestrator
            capture_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Capture post_hygiene layer only (raw_retrieval would need pre-hygiene context)
        capture = SnapshotCapture()
        layer = capture.capture_layer(
            layer_name="post_hygiene",
            structured_context=context,
            formatted_sections=formatted_sections,
            prompt_text=final_prompt,
        )

        # Build minimal snapshot (single layer from builder hook)
        import uuid
        from eval.schema import PromptSnapshot

        snapshot = PromptSnapshot(
            snapshot_id=str(uuid.uuid4())[:8],
            query_text=user_input,
            query_timestamp=datetime.now(timezone.utc).isoformat(),
            processed_query=user_input,
            detected_intent="",
            detected_tone="",
            provenance=provenance,
            layers={"post_hygiene": layer},
            retrieval_metadata={},
            assembly_metadata={"section_count": len(sections)},
        )

        save_snapshot(snapshot)
        logger.info(f"[EVAL] Snapshot captured: {snapshot.snapshot_id} ({len(formatted_sections)} sections)")

    except Exception as e:
        if _eval_capture_strict():
            raise
        logger.warning(f"[EVAL] Snapshot capture failed (non-fatal): {e}")


def _compute_token_budget(model_manager) -> int:
    """Compute prompt token budget based on model context window.

    Priority: env-var override > model-aware fraction > default.
    """
    # 1. Explicit env-var override (legacy compat: PROMPT_TOKEN_BUDGET=15000)
    if PROMPT_TOKEN_BUDGET_OVERRIDE is not None:
        logger.info(f"[PromptBuilder] Token budget: {PROMPT_TOKEN_BUDGET_OVERRIDE} (env override)")
        return PROMPT_TOKEN_BUDGET_OVERRIDE

    # 2. No model_manager available — use default
    if model_manager is None:
        logger.info(f"[PromptBuilder] Token budget: {PROMPT_TOKEN_BUDGET_DEFAULT} (default, no model_manager)")
        return PROMPT_TOKEN_BUDGET_DEFAULT

    # 3. Model-aware computation
    try:
        ctx_limit = model_manager.get_context_limit()
        is_local = not model_manager.is_api_model(model_manager.get_active_model_name())

        raw = int(ctx_limit * PROMPT_TOKEN_BUDGET_CONTEXT_FRACTION)

        if is_local:
            budget = max(PROMPT_TOKEN_BUDGET_FLOOR, min(raw, PROMPT_TOKEN_BUDGET_LOCAL))
        else:
            budget = max(PROMPT_TOKEN_BUDGET_FLOOR, min(raw, PROMPT_TOKEN_BUDGET_CEILING))

        logger.info(
            f"[PromptBuilder] Token budget: {budget} "
            f"(model-aware, ctx={ctx_limit}, local={is_local})"
        )
        return budget
    except Exception as e:
        logger.warning(f"[PromptBuilder] Could not determine context limit: {e}, using default")
        return PROMPT_TOKEN_BUDGET_DEFAULT

# Content limits (aligned with ContextGatherer defaults and user expectations)
# - Recent conversations: 15
# - Relevant memories: 15 (semantic search results only)
# - Facts: 15 semantic + 15 recent
# - Summaries: 10 (hybrid)
# - Reflections: 10 (hybrid)
PROMPT_MAX_RECENT = _cfg_int("prompt_max_recent", 15)
PROMPT_MAX_MEMS = _cfg_int("prompt_max_mems", 15)
PROMPT_MAX_FACTS = _cfg_int("prompt_max_facts", 30)
PROMPT_MAX_RECENT_FACTS = _cfg_int("prompt_max_recent_facts", 30)
PROMPT_MAX_SUMMARIES = _cfg_int("prompt_max_summaries", 10)
PROMPT_MAX_REFLECTIONS = _cfg_int("prompt_max_reflections", 10)
PROMPT_MAX_DREAMS = _cfg_int("prompt_max_dreams", 3)
PROMPT_MAX_SEMANTIC = _cfg_int("prompt_max_semantic", 8)
PROMPT_MAX_WIKI = _cfg_int("prompt_max_wiki", 3)
USER_PROFILE_FACTS_PER_CATEGORY = _cfg_int("user_profile_facts_per_category", 3)
PROMPT_MAX_PERSONAL_NOTES = _cfg_int("prompt_max_personal_notes", 5)
PROMPT_MIN_RECENT_FLOOR = _cfg_int("prompt_min_recent_floor", 5)

# _staleness_prefix, _is_multimodal_model, _load_upload_image moved to formatter.py
# Re-exported above via: from .formatter import _staleness_prefix, _is_multimodal_model, _load_upload_image
PROMPT_MAX_REFERENCE_DOCS = _cfg_int("prompt_max_reference_docs", 15)
PROMPT_MAX_GIT_COMMITS = _cfg_int("prompt_max_git_commits", 10)
PROMPT_MAX_SKILLS = _cfg_int("prompt_max_skills", 5)
PROMPT_MAX_PROPOSALS = _cfg_int("prompt_max_proposals", 3)
PROMPT_MAX_USER_UPLOADS = _cfg_int("prompt_max_user_uploads", 5)
PROMPT_MAX_GRAPH_SENTENCES = _cfg_int("prompt_max_graph_sentences", 12)
PROMPT_MAX_SURFACED_THREADS = _cfg_int("prompt_max_surfaced_threads", 3)
PROMPT_MAX_PROACTIVE_INSIGHTS = _cfg_int("prompt_max_proactive_insights", 2)
PROMPT_MAX_VISUAL_MEMORIES = _cfg_int("prompt_max_visual_memories", 3)

# Feature toggles
REFLECTIONS_ON_DEMAND = _parse_bool(os.getenv("REFLECTIONS_ON_DEMAND", "0"))  # Off by default — blocks prompt build with LLM call
# Keep broad by default so we don't drop historical reflections
REFLECTIONS_SESSION_FILTER = _parse_bool(os.getenv("REFLECTIONS_SESSION_FILTER", "0"))
REFLECTIONS_TOPUP = _parse_bool(os.getenv("REFLECTIONS_TOPUP", "1"))

# Obsidian image loading for multimodal models
try:
    from config.app_config import (
        OBSIDIAN_INCLUDE_IMAGES,
        OBSIDIAN_MAX_IMAGES_PER_NOTE,
        PERSONAL_NOTES_GATE_THRESHOLD,
        REFERENCE_DOCS_GATE_THRESHOLD,
    )
except ImportError:
    OBSIDIAN_INCLUDE_IMAGES = True
    OBSIDIAN_MAX_IMAGES_PER_NOTE = 3
    PERSONAL_NOTES_GATE_THRESHOLD = 0.45
    REFERENCE_DOCS_GATE_THRESHOLD = 0.40



# Priority order for token budget management
PRIORITY_ORDER = [
    ("recent_conversations", 7),
    ("semantic_chunks", 6),
    ("personal_notes", 6),  # User's Obsidian notes - high priority
    ("user_uploads", 6),    # User uploaded files/images - high priority
    ("reference_docs", 5),  # Reference documents (system docs, project outlines)
    ("git_commits", 5),     # Git commit history (procedural memory)
    ("procedural_skills", 5),  # Reusable problem-solving patterns
    ("proposed_features", 3),  # Code proposals (trimmed before core context)
    ("unresolved_threads", 4),  # Open threads for proactive surfacing
    ("proactive_insights", 3),  # Cross-domain insights from knowledge graph
    ("memories", 5),
    ("semantic_facts", 4),
    ("fresh_facts", 4),
    ("summaries", 3),
    ("reflections", 2),
    ("wiki", 1),
    ("dreams", 2),
]




class UnifiedPromptBuilder:
    """
    Unified prompt builder that coordinates all prompt building functionality.

    This class orchestrates the entire prompt building process by:
    1. Gathering context from various sources (memories, facts, wiki, etc.)
    2. Managing token budgets and content prioritization
    3. Formatting and assembling the final prompt
    4. Providing LLM summarization capabilities
    """

    def __init__(self, memory_coordinator=None, model_manager=None, tokenizer_manager=None,
                 consolidator=None, time_manager=None, token_budget: int = None,
                 wiki_manager=None, topic_manager=None, gate_system=None, **kwargs):
        """
        Initialize the UnifiedPromptBuilder.

        Args:
            memory_coordinator: Coordinator for memory operations
            model_manager: Manager for LLM interactions
            tokenizer_manager: Manager for token counting
            consolidator: Memory consolidation manager
            time_manager: Time management utilities
            token_budget: Maximum tokens for prompt context (None = auto-compute from model)
        """
        # Core dependencies
        self.memory_coordinator = memory_coordinator or self._build_default_memory_coordinator()
        self.model_manager = model_manager
        self.tokenizer_manager = tokenizer_manager
        self.consolidator = consolidator or MemoryConsolidator(model_manager)
        self.time_manager = time_manager or TimeManager()

        # Additional managers (for backward compatibility)
        self.wiki_manager = wiki_manager
        self.topic_manager = topic_manager
        self.gate_system = gate_system

        # Token management — model-aware if token_budget not explicitly passed
        if token_budget is None:
            token_budget = _compute_token_budget(model_manager)
        self.token_budget = token_budget

        # Initialize modular components
        self.token_manager = TokenManager(
            model_manager=self.model_manager,
            tokenizer_manager=self.tokenizer_manager,
            token_budget=token_budget
        )

        self.context_gatherer = ContextGatherer(
            memory_coordinator=self.memory_coordinator,
            model_manager=self.model_manager,
            token_manager=self.token_manager,
            gate_system=self.gate_system,
            time_manager=self.time_manager
        )

        self.formatter = PromptFormatter(
            token_manager=self.token_manager,
            time_manager=self.time_manager
        )

        self.summarizer = LLMSummarizer(
            model_manager=self.model_manager,
            memory_coordinator=self.memory_coordinator
        )

        self._hygiene = ContentHygiene(
            memory_coordinator=self.memory_coordinator,
            context_gatherer=self.context_gatherer
        )

        # Skill activation policy (post-retrieval filtering + cooldown)
        try:
            from config.app_config import (
                SKILL_ACTIVATION_ENABLED, SKILL_ACTIVATION_MAX_SKILLS,
                SKILL_ACTIVATION_MIN_SCORE, SKILL_ACTIVATION_COOLDOWN_HOURS,
                SKILL_ACTIVATION_FETCH_MULTIPLIER, SKILL_ACTIVATION_STM_BONUS,
                SKILL_ACTIVATION_USE_STM,
            )
            self._skill_activation_policy = SkillActivationPolicy(
                cooldown_store=SkillCooldownStore(),
                min_score=SKILL_ACTIVATION_MIN_SCORE,
                cooldown_hours=SKILL_ACTIVATION_COOLDOWN_HOURS,
                max_skills=SKILL_ACTIVATION_MAX_SKILLS,
                stm_bonus=SKILL_ACTIVATION_STM_BONUS,
                enabled=SKILL_ACTIVATION_ENABLED,
            )
            self._skill_fetch_multiplier = SKILL_ACTIVATION_FETCH_MULTIPLIER
            self._skill_activation_use_stm = SKILL_ACTIVATION_USE_STM
        except ImportError:
            self._skill_activation_policy = None
            self._skill_fetch_multiplier = 1
            self._skill_activation_use_stm = False

        # State tracking
        self._prompt_token_usage = 0

    def _build_default_memory_coordinator(self):
        """Build a fallback memory coordinator if none provided."""
        logger.warning("No memory coordinator provided, using fallback")
        return _FallbackMemoryCoordinator()

    async def _llm_compress_oversized(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-pass: LLM-compress heavily oversized items before budget trimming.

        Only targets items >= ratio_threshold * max_tokens (default 3x).
        Mildly oversized items still handled by middle-out in token_manager.
        """
        if not LLM_COMPRESSION_ENABLED:
            return context
        if not self.model_manager or not hasattr(self.model_manager, 'generate_once'):
            return context

        from .token_manager import MEMORY_ITEM_MAX_TOKENS, SEMANTIC_ITEM_MAX_TOKENS, PRIORITY_ORDER as TM_PRIORITY_ORDER

        try:
            model_name = self.model_manager.get_active_model_name() if hasattr(self.model_manager, "get_active_model_name") else "default"
        except Exception:
            model_name = "default"

        # Scan all list sections for heavily oversized items
        candidates = []  # (section_name, index, item, item_tokens, max_tokens)
        for name, _prio in TM_PRIORITY_ORDER:
            val = context.get(name)
            if not val or not isinstance(val, list):
                continue
            if name in ("stm_summary", "user_profile", "narrative_state"):
                continue

            max_tokens = MEMORY_ITEM_MAX_TOKENS if name == "memories" else SEMANTIC_ITEM_MAX_TOKENS
            threshold = max_tokens * LLM_COMPRESSION_RATIO_THRESHOLD

            for i, item in enumerate(val):
                item_text = self.token_manager._extract_text(item)
                try:
                    t = self.token_manager.get_token_count(item_text, model_name)
                except Exception:
                    t = len(item_text.split())
                if t >= threshold:
                    candidates.append((name, i, item, t, max_tokens))

        if not candidates:
            return context

        # Sort by ratio (largest first), cap at max_batch
        candidates.sort(key=lambda c: c[3] / c[4], reverse=True)
        candidates = candidates[:LLM_COMPRESSION_MAX_BATCH]

        logger.info(f"[LLM-COMPRESS] {len(candidates)} items queued for LLM compression")

        # Build compression tasks
        async def _compress_one(section: str, idx: int, item, item_tokens: int, max_tok: int):
            item_text = self.token_manager._extract_text(item)
            target = max_tok
            prompt = (
                f"Compress the following text to approximately {target} tokens. "
                f"Preserve ALL key facts, names, dates, numbers, and decisions. "
                f"Output ONLY the compressed text, nothing else.\n\n"
                f"Text:\n{item_text}"
            )
            try:
                compressed = await asyncio.wait_for(
                    self.model_manager.generate_once(
                        prompt,
                        model_name=LLM_COMPRESSION_MODEL,
                        system_prompt="You are a precise text compressor. Output only the compressed text.",
                        max_tokens=target + 64,  # small buffer for token estimation mismatch
                        temperature=0.0,
                    ),
                    timeout=LLM_COMPRESSION_TIMEOUT,
                )
                if compressed and isinstance(compressed, str) and len(compressed.strip()) > 20:
                    try:
                        new_tokens = self.token_manager.get_token_count(compressed.strip(), model_name)
                    except Exception:
                        new_tokens = len(compressed.strip().split())
                    logger.info(
                        f"[LLM-COMPRESS] {section}[{idx}]: {item_tokens}→{new_tokens} tokens (LLM)"
                    )
                    return (section, idx, compressed.strip())
            except asyncio.TimeoutError:
                logger.warning(f"[LLM-COMPRESS] Timeout compressing {section}[{idx}], falling back to middle-out")
            except Exception as e:
                logger.warning(f"[LLM-COMPRESS] Failed {section}[{idx}]: {e}, falling back to middle-out")
            return None

        # Fire all compressions in parallel
        tasks = [
            _compress_one(section, idx, item, item_tokens, max_tok)
            for section, idx, item, item_tokens, max_tok in candidates
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Apply successful compressions back into context
        for result in results:
            if result is None or isinstance(result, Exception):
                continue
            section, idx, compressed_text = result
            items = context.get(section)
            if items and isinstance(items, list) and idx < len(items):
                original = items[idx]
                if isinstance(original, dict):
                    updated = dict(original)
                    for text_key in ('content', 'text', 'query', 'response'):
                        if text_key in updated:
                            updated[text_key] = compressed_text
                            break
                    else:
                        updated['content'] = compressed_text
                    items[idx] = updated
                else:
                    items[idx] = compressed_text

        return context

    async def build_prompt(self, user_input: str, config: Optional[Dict[str, Any]] = None,
                          search_query: Optional[str] = None, personality_config: Optional[Dict[str, Any]] = None,
                          system_prompt: Optional[str] = None, current_topic: Optional[str] = None,
                          fresh_facts: Optional[List[Any]] = None, memories: Optional[List[Any]] = None,
                          stm_summary: Optional[Dict[str, Any]] = None,
                          crisis_level: Optional[str] = None,
                          retrieval_overrides: Optional[Dict[str, int]] = None,
                          weight_overrides: Optional[Dict[str, float]] = None,
                          intent_type: Optional[str] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Build a complete prompt context for the given user input.

        This is the main entry point for prompt building. It gathers context
        from all sources, applies token budget management, and returns a
        structured context dict ready for formatting.

        Args:
            user_input: The user's query/input
            config: Optional configuration overrides
            crisis_level: Current crisis level (HIGH/MEDIUM suppresses web search)
            retrieval_overrides: Optional dict of {max_*: count} to override
                global PROMPT_MAX_* constants. Used by intent classifier.
            weight_overrides: Optional dict of {weight_name: value} to override
                global SCORE_WEIGHTS. Used by intent classifier.

        Returns:
            Dict containing the built prompt context with sections like:
            - recent_conversations
            - memories
            - facts
            - fresh_facts
            - summaries
            - reflections
            - wiki
            - semantic_chunks
            - dreams
            - web_search_results (if triggered)
        """
        start_time = time.time()
        config = config or {}

        # Clear memory_id_map at start of each query to prevent memory leaks
        if hasattr(self.context_gatherer, 'clear_memory_id_map'):
            self.context_gatherer.clear_memory_id_map()

        logger.info(f"Building prompt for user input: {len(user_input)} chars")

        try:
            # Pre-fork: Detect first message + gather codebase changes BEFORE small-talk check
            # This ensures even "Yo" gets codebase change awareness.
            is_first_message = False
            codebase_changes = {}
            if self.time_manager:
                try:
                    gap = self.time_manager.time_since_previous_message()
                    is_first_message = isinstance(gap, str) and "N/A" in gap
                except (AttributeError, TypeError):
                    pass
            if is_first_message:
                _since_dt = getattr(self.time_manager, 'last_session_end_time', None)
                try:
                    codebase_changes = await self.context_gatherer.get_codebase_changes(_since_dt)
                except Exception as e:
                    logger.debug(f"[BUILD_PROMPT] Codebase changes failed: {e}")

            # Step 1: Analyze the query
            query_analysis = {}
            try:
                query_analysis = analyze_query(user_input)
                logger.debug(f"Query analysis: {query_analysis}")
            except Exception as e:
                logger.warning(f"Query analysis failed: {e}")

            # Check if this is small-talk that doesn't need heavy retrieval
            is_small_talk = getattr(query_analysis, "is_small_talk", False)
            logger.warning(f"SMALL_TALK CHECK: is_small_talk={is_small_talk}")
            if is_small_talk:
                logger.warning("USING LIGHTWEIGHT CONTEXT - this will drop separated keys!")
                return await self._build_lightweight_context(user_input, stm_summary=stm_summary, codebase_changes=codebase_changes)

            # Step 2: Gather narrative context (synchronous, cheap file read)
            # Gated by intent override: max_narrative=0 skips entirely
            narrative_state = ""
            _ro_pre = retrieval_overrides or {}
            if _ro_pre.get("max_narrative", 1) > 0:
                try:
                    narrative_state = self.context_gatherer.get_narrative_context()
                    if narrative_state:
                        logger.debug(f"[PromptBuilder] Got narrative context ({len(narrative_state)} chars)")
                except Exception as e:
                    logger.debug(f"[PromptBuilder] Failed to get narrative context: {e}")

            # Apply intent-driven retrieval count overrides
            _ro = retrieval_overrides or {}
            eff_max_recent = _ro.get("max_recent", PROMPT_MAX_RECENT)
            eff_max_mems = _ro.get("max_mems", PROMPT_MAX_MEMS)
            eff_max_summaries_r = _ro.get("max_recent_summaries", PROMPT_MAX_RECENT_SUMMARIES)
            eff_max_summaries_s = _ro.get("max_semantic_summaries", PROMPT_MAX_SEMANTIC_SUMMARIES)
            # Convenience: "max_summaries" splits evenly into recent/semantic
            if "max_summaries" in _ro and "max_recent_summaries" not in _ro:
                half = max(1, _ro["max_summaries"] // 2)
                eff_max_summaries_r = half
                eff_max_summaries_s = _ro["max_summaries"] - half
            eff_max_reflections_r = _ro.get("max_recent_reflections", PROMPT_MAX_RECENT_REFLECTIONS)
            eff_max_reflections_s = _ro.get("max_semantic_reflections", PROMPT_MAX_SEMANTIC_REFLECTIONS)
            if "max_reflections" in _ro and "max_recent_reflections" not in _ro:
                half = max(1, _ro["max_reflections"] // 2)
                eff_max_reflections_r = half
                eff_max_reflections_s = _ro["max_reflections"] - half
            eff_max_dreams = _ro.get("max_dreams", PROMPT_MAX_DREAMS)
            eff_max_semantic = _ro.get("max_semantic", PROMPT_MAX_SEMANTIC)
            eff_max_wiki = _ro.get("max_wiki", PROMPT_MAX_WIKI)
            eff_max_skills = _ro.get("max_skills", PROMPT_MAX_SKILLS)
            eff_max_proposals = _ro.get("max_proposals", PROMPT_MAX_PROPOSALS)
            eff_max_git = _ro.get("max_git_commits", PROMPT_MAX_GIT_COMMITS)
            eff_max_surfaced_threads = _ro.get("max_surfaced_threads", PROMPT_MAX_SURFACED_THREADS)
            eff_max_reference_docs = _ro.get("max_reference_docs", PROMPT_MAX_REFERENCE_DOCS)
            eff_max_user_uploads = _ro.get("max_user_uploads", PROMPT_MAX_USER_UPLOADS)
            eff_max_proactive = _ro.get("max_proactive", PROMPT_MAX_PROACTIVE_INSIGHTS)
            eff_max_visual_memories = _ro.get("max_visual_memories", PROMPT_MAX_VISUAL_MEMORIES)
            # Defensive fallback: if no intent overrides available (intent=None),
            # suppress visual memory for very short messages (likely casual greetings)
            if not _ro and len(user_input.split()) <= 5:
                eff_max_visual_memories = 0
                logger.debug("[BUILD_PROMPT] No intent overrides + short message — suppressing visual memory")
            eff_max_personal_notes = _ro.get("max_personal_notes", PROMPT_MAX_PERSONAL_NOTES)

            if _ro:
                logger.info(f"[BUILD_PROMPT] Intent retrieval overrides: {_ro}")

            # Set intent-driven weight overrides on scorer (cleared after gather)
            scorer = getattr(self.memory_coordinator, 'scorer', None)
            if scorer and weight_overrides:
                scorer._intent_weight_overrides = weight_overrides
                logger.info(f"[BUILD_PROMPT] Intent weight overrides set on scorer")

            # Set graph references on scorer for graph-boosted scoring
            if scorer:
                scorer._graph_memory = getattr(self.memory_coordinator, 'graph_memory', None)
                scorer._entity_resolver = getattr(self.memory_coordinator, 'entity_resolver', None)

            # Apply intent-driven gate threshold override (cleared after gather)
            _gate_override = kwargs.get('_gate_threshold_override')
            _saved_gate_threshold = None
            _gate_obj = None
            if _gate_override is not None:
                # Access the gate system (triggers lazy init via property)
                _gs = self.context_gatherer.gate_system
                if _gs is not None:
                    # MultiStageGateSystem wraps CosineSimilarityGateSystem as .gate_system
                    _gate_obj = getattr(_gs, 'gate_system', _gs)
                    if hasattr(_gate_obj, 'cosine_threshold'):
                        _saved_gate_threshold = _gate_obj.cosine_threshold
                        _gate_obj.cosine_threshold = _gate_override
                        logger.info(f"[BUILD_PROMPT] Gate threshold override: {_saved_gate_threshold:.3f} -> {_gate_override:.3f}")

            # Step 3: Launch parallel data gathering tasks with per-task timing
            # Pre-embed the query once so all parallel ChromaDB lookups reuse it.
            chroma = getattr(self.memory_coordinator, 'chroma_store', None)
            if chroma and hasattr(chroma, 'clear_embedding_cache'):
                chroma.clear_embedding_cache()
                try:
                    chroma._cached_embed(user_input)
                except Exception:
                    pass  # Non-fatal; individual queries will embed as needed

            tasks = {}
            task_timings = {}

            async def _timed_task(name: str, coro):
                """Wrapper to time individual tasks"""
                _start = time.time()
                try:
                    result = await coro
                    task_timings[name] = time.time() - _start
                    return result
                except Exception as e:
                    task_timings[name] = time.time() - _start
                    raise e

            # Recent conversations
            tasks["recent"] = asyncio.create_task(
                _timed_task("recent", self.context_gatherer._get_recent_conversations(eff_max_recent))
            )

            # Query-relevant memories (semantic search results only)
            tasks["memories"] = asyncio.create_task(
                _timed_task("memories", self.context_gatherer._get_semantic_memories(user_input, eff_max_mems))
            )

            # User Profile (replaces semantic_facts + fresh_facts with categorized hybrid retrieval)
            # Increased max_tokens to 3000 to accommodate 12 facts per category (up to 144 facts total)
            tasks["user_profile"] = asyncio.create_task(
                _timed_task("user_profile", self.context_gatherer.get_user_profile_context(user_input, max_tokens=3000))
            )

            # Summaries (separated into recent + semantic)
            if eff_max_summaries_r > 0 or eff_max_summaries_s > 0:
                tasks["summaries"] = asyncio.create_task(
                    _timed_task("summaries", self.context_gatherer._get_summaries_separate(user_input, eff_max_summaries_r, eff_max_summaries_s))
                )

            # Dreams (if enabled)
            if eff_max_dreams > 0:
                tasks["dreams"] = asyncio.create_task(
                    _timed_task("dreams", self.context_gatherer._get_dreams(eff_max_dreams))
                )

            # Semantic chunks
            if eff_max_semantic > 0:
                tasks["semantic"] = asyncio.create_task(
                    _timed_task("semantic", self.context_gatherer._get_semantic_chunks(user_input, max_results=eff_max_semantic))
                )

            # Reflections (separated into recent + semantic)
            if eff_max_reflections_r > 0 or eff_max_reflections_s > 0:
                tasks["reflections"] = asyncio.create_task(
                    _timed_task("reflections", self.context_gatherer._get_reflections_separate(user_input, eff_max_reflections_r, eff_max_reflections_s))
                )

            # Wiki content
            if eff_max_wiki > 0:
                tasks["wiki"] = asyncio.create_task(
                    _timed_task("wiki", self.context_gatherer._get_wiki_content(user_input, eff_max_wiki))
                )

            # Personal notes from Obsidian vault
            # Check if model is multimodal to decide whether to load images
            current_model = getattr(self.model_manager, 'active_model_name', '') if self.model_manager else ''
            include_note_images = OBSIDIAN_INCLUDE_IMAGES and _is_multimodal_model(current_model)
            logger.debug(f"[PromptBuilder] image check: model={current_model}, OBSIDIAN_INCLUDE_IMAGES={OBSIDIAN_INCLUDE_IMAGES}, is_multimodal={_is_multimodal_model(current_model)}, include_note_images={include_note_images}")

            if eff_max_personal_notes > 0:
                tasks["personal_notes"] = asyncio.create_task(
                    _timed_task("personal_notes", self.context_gatherer.get_personal_notes(
                        user_input,
                        eff_max_personal_notes,
                        include_images=include_note_images,
                        max_images_per_note=OBSIDIAN_MAX_IMAGES_PER_NOTE
                    ))
                )

            # Reference documents (system docs, project outlines - excludes user uploads)
            # Suppressed when user uploads files so file content dominates context
            if not kwargs.get("_suppress_reference_docs", False) and eff_max_reference_docs > 0:
                tasks["reference_docs"] = asyncio.create_task(
                    _timed_task("reference_docs", self.context_gatherer.get_reference_docs(user_input, eff_max_reference_docs))
                )

            # User uploads (previously uploaded files/images)
            if eff_max_user_uploads > 0:
                tasks["user_uploads"] = asyncio.create_task(
                    _timed_task("user_uploads", self.context_gatherer.get_user_uploads(user_input, eff_max_user_uploads))
                )

            # Git commit history (procedural memory)
            if eff_max_git > 0:
                tasks["git_commits"] = asyncio.create_task(
                    _timed_task("git_commits", self.context_gatherer.get_git_commits(user_input, eff_max_git))
                )

            # Procedural skills (adaptive workflows)
            # Fetch wider window when activation policy is active so it can filter/rerank
            _skill_fetch_limit = eff_max_skills * self._skill_fetch_multiplier if self._skill_activation_policy else eff_max_skills
            if eff_max_skills > 0:
                tasks["procedural_skills"] = asyncio.create_task(
                    _timed_task("procedural_skills", self.context_gatherer.get_procedural_skills(user_input, _skill_fetch_limit))
                )

            # Proposed features (code proposals, only for project-related queries)
            if eff_max_proposals > 0:
                tasks["proposed_features"] = asyncio.create_task(
                    _timed_task("proposed_features", self.context_gatherer.get_proposed_features(user_input, eff_max_proposals))
                )

            # Knowledge graph context (entity relationships)
            tasks["graph_context"] = asyncio.create_task(
                _timed_task("graph_context", self.context_gatherer.get_graph_context(user_input, PROMPT_MAX_GRAPH_SENTENCES))
            )

            # Unresolved threads (proactive surfacing)
            tasks["unresolved_threads"] = asyncio.create_task(
                _timed_task("unresolved_threads", self.context_gatherer.get_unresolved_threads(eff_max_surfaced_threads))
            )

            # Proactive cross-domain insights from knowledge graph
            if eff_max_proactive > 0:
                tasks["proactive_insights"] = asyncio.create_task(
                    _timed_task("proactive_insights", self.context_gatherer.get_proactive_insights(user_input, eff_max_proactive))
                )

            # Visual memories (CLIP-based image search)
            if eff_max_visual_memories > 0:
                tasks["visual_memories"] = asyncio.create_task(
                    _timed_task("visual_memories", self.context_gatherer.get_visual_memories(user_input, eff_max_visual_memories))
                )

            # Web search (triggered based on query analysis, suppressed during crisis)
            tasks["web_search"] = asyncio.create_task(
                _timed_task("web_search", self.context_gatherer._get_web_search_results(user_input, crisis_level, intent_type=intent_type))
            )

            # Gather all results with timeout — use asyncio.wait so completed
            # tasks survive a timeout instead of wiping the entire context.
            _gather_start = time.time()
            try:
                done, pending = await asyncio.wait(
                    list(tasks.values()),
                    timeout=30.0,
                    return_when=asyncio.ALL_COMPLETED,
                )
                _gather_elapsed = time.time() - _gather_start

                gathered = {}
                timed_out_names = []
                for name, task in tasks.items():
                    if task in done:
                        try:
                            gathered[name] = task.result() or []
                            if name == "memories":
                                logger.debug(f"MEMORIES TASK: Got {len(gathered[name])} memories")
                            if name == "proposed_features":
                                logger.info(f"[PROPOSED_FEATURES] Task returned {len(gathered[name])} proposals")
                        except Exception as exc:
                            logger.warning("Context task %s failed: %s", name, exc)
                            gathered[name] = []
                    else:
                        task.cancel()
                        gathered[name] = []
                        timed_out_names.append(name)

                if timed_out_names:
                    logger.warning(
                        "Prompt context retrieval timed out; partial context used. Pending: %s",
                        sorted(timed_out_names),
                    )

                if task_timings:
                    sorted_timings = sorted(task_timings.items(), key=lambda x: x[1], reverse=True)
                    timing_str = " | ".join([f"{k}={v:.2f}s" for k, v in sorted_timings])
                    logger.info(
                        f"[BUILD_PROMPT TIMING] total={_gather_elapsed:.2f}s | {timing_str}"
                    )

            except Exception as _gather_exc:
                logger.warning("Unexpected error during context gathering: %s", _gather_exc)
                gathered = {name: [] for name in tasks.keys()}
            finally:
                # Clear intent weight overrides from scorer (set before gather)
                if scorer and weight_overrides:
                    scorer._intent_weight_overrides = None
                # Clear graph references from scorer
                if scorer:
                    scorer._graph_memory = None
                    scorer._entity_resolver = None
                # Restore gate threshold (set before gather)
                if _saved_gate_threshold is not None and _gate_obj is not None:
                    _gate_obj.cosine_threshold = _saved_gate_threshold

            # Step 3: Post-fetch processing

            # Apply skill activation policy (filter/rerank by intent, relevance, cooldown)
            if self._skill_activation_policy and "procedural_skills" in gathered:
                _stm_topics = None
                if self._skill_activation_use_stm and stm_summary:
                    _topic = stm_summary.get("topic", "")
                    _stm_topics = [_topic] if _topic and _topic.lower() != "general" else None
                gathered["procedural_skills"] = self._skill_activation_policy.filter(
                    gathered["procedural_skills"],
                    intent_type=intent_type,
                    stm_topics=_stm_topics,
                )

            # Handle separated summaries (recent + semantic)
            summaries_data = gathered.get("summaries", {})
            logger.debug(f"CONTEXT GATHERING: summaries_data type={type(summaries_data).__name__}, len={len(summaries_data) if isinstance(summaries_data, (list, dict)) else '?'}")
            if isinstance(summaries_data, dict):
                recent_summaries = summaries_data.get("recent", [])
                semantic_summaries = summaries_data.get("semantic", [])
                all_summaries = recent_summaries + semantic_summaries
                logger.debug(f"CONTEXT GATHERING: Extracted {len(recent_summaries)} recent, {len(semantic_summaries)} semantic summaries")
            else:
                # Backward compatibility for old format
                all_summaries = summaries_data or []
                recent_summaries = []
                semantic_summaries = []
                logger.debug(f"CONTEXT GATHERING: Using old format, got {len(all_summaries)} summaries")

            # Handle separated reflections (recent + semantic)
            reflections_data = gathered.get("reflections", {})
            if isinstance(reflections_data, dict):
                recent_reflections = reflections_data.get("recent", [])
                semantic_reflections = reflections_data.get("semantic", [])
                all_reflections = recent_reflections + semantic_reflections
            else:
                # Backward compatibility for old format
                all_reflections = reflections_data or []
                recent_reflections = []
                semantic_reflections = []

            # Filter reflections to session-level if enabled; if it empties the set,
            # fall back to original reflections to avoid dropping the section.
            if REFLECTIONS_SESSION_FILTER and all_reflections:
                session_reflections = [
                    r for r in all_reflections
                    if "session" in (r.get("tags", []) or []) or "session" in (r.get("source", "") or "")
                ]
                if not session_reflections:
                    session_reflections = all_reflections
            else:
                session_reflections = all_reflections

            # Sort reflections by timestamp (most recent first)
            try:
                session_reflections.sort(
                    key=lambda x: x.get("timestamp", ""),
                    reverse=True
                )
            except TypeError as e:
                logger.warning(f"[PromptBuilder] Could not sort reflections by timestamp: {e}")

            # Top-up with on-demand reflections if needed
            if (REFLECTIONS_TOPUP and REFLECTIONS_ON_DEMAND and
                len(session_reflections) < PROMPT_MAX_REFLECTIONS):

                try:
                    context_for_reflection = {
                        "memories": gathered.get("memories", []),
                        "fresh_facts": gathered.get("recent_facts", [])
                    }

                    on_demand_reflections = await self.summarizer._reflect_on_demand(
                        context_for_reflection,
                        user_input,
                        session_reflections
                    )

                    session_reflections.extend(on_demand_reflections)
                except Exception as e:
                    logger.warning(f"On-demand reflection failed: {e}")

            # Step 4: Build initial context
            gathered_memories = gathered.get("memories", [])
            logger.debug(f"CONTEXT BUILD: gathered memories count = {len(gathered_memories)}")

            # DEBUG: Check what's in recent conversations
            recent_convos = gathered.get("recent", [])
            logger.debug(f"[DEBUG RECENT] build_prompt: Got {len(recent_convos)} recent_conversations from gatherer")
            if recent_convos:
                # Log first 3 and last 3 with timestamps
                for i in range(min(3, len(recent_convos))):
                    mem = recent_convos[i]
                    ts = mem.get('timestamp', 'NO_TS')
                    query = mem.get('query', '')[:80]
                    logger.debug(f"[DEBUG RECENT] Item {i+1} (first): ts={ts}, query={query}...")
                if len(recent_convos) > 3:
                    for i in range(max(0, len(recent_convos) - 3), len(recent_convos)):
                        mem = recent_convos[i]
                        ts = mem.get('timestamp', 'NO_TS')
                        query = mem.get('query', '')[:80]
                        logger.debug(f"[DEBUG RECENT] Item {i+1} (last): ts={ts}, query={query}...")

            context = {
                "recent_conversations": recent_convos,
                "memories": gathered_memories,
                "user_profile": gathered.get("user_profile", ""),  # Replaces semantic_facts + fresh_facts
                "narrative_state": narrative_state,  # Temporal grounding (synthesized life context)
                "summaries": all_summaries,
                "recent_summaries": recent_summaries,
                "semantic_summaries": semantic_summaries,
                "reflections": session_reflections,
                "recent_reflections": recent_reflections,
                "semantic_reflections": semantic_reflections,
                "dreams": gathered.get("dreams", []),
                "semantic_chunks": gathered.get("semantic", []),
                "wiki": gathered.get("wiki", []),
                "personal_notes": gathered.get("personal_notes", []),  # User's Obsidian notes
                "reference_docs": gathered.get("reference_docs", []),  # System/project documentation
                "user_uploads": gathered.get("user_uploads", []),     # User uploaded files/images
                "git_commits": gathered.get("git_commits", []),      # Git commit history
                "procedural_skills": gathered.get("procedural_skills", []),  # Adaptive workflows
                "proposed_features": gathered.get("proposed_features", []),  # Code proposals
                "graph_context": gathered.get("graph_context", []),  # Knowledge graph relationships
                "unresolved_threads": gathered.get("unresolved_threads", []),  # Proactive thread surfacing
                "proactive_insights": gathered.get("proactive_insights", []),  # Cross-domain insights
                "visual_memories": gathered.get("visual_memories", {"text_results": [], "images": []}),  # CLIP visual memories
                "web_search_results": gathered.get("web_search"),  # Real-time web search results
                "codebase_changes": codebase_changes,  # Git changes since last session (first message only)
            }
            logger.debug(f"CONTEXT BUILT: recent_summaries={len(recent_summaries)}, semantic_summaries={len(semantic_summaries)}, recent_reflections={len(recent_reflections)}, semantic_reflections={len(semantic_reflections)}")
            logger.debug(f"CONTEXT BUILD: context memories count = {len(context['memories'])}")

            # Override with directly provided parameters (legacy interface)
            # Note: fresh_facts removed - now using user_profile instead
            if memories is not None:
                context["memories"] = memories

            # Step 5: Apply gating to filter by relevance
            try:
                # Avoid re-gating memories: ContextGatherer already applies
                # semantic filtering to the semantic half while preserving
                # the recency half. Re-gating here could drop the recents.

                # Do not gate wiki snippets here — wiki utility already applies
                # conservative cleaning and we prefer fail-open to ensure topical
                # knowledge flows into the prompt.

                # Allow semantic chunks to flow as-is; downstream token budgeting
                # and stitching will cap size. If we need gating later, prefer
                # the specialized filter_semantic_chunks in gate_system.

                # Gate personal notes through the multi-stage gate system
                personal_notes = context.get("personal_notes", [])
                if personal_notes and hasattr(self.context_gatherer, 'gate_system'):
                    try:
                        gated_notes = await self.context_gatherer.gate_system.filter_memories(
                            user_input, personal_notes
                        )
                        # Apply stricter relevance threshold for personal notes
                        # (general gate threshold is 0.18; personal notes need 0.30+)
                        pre_filter_count = len(gated_notes)
                        gated_notes = [n for n in gated_notes
                                       if n.get("relevance_score", 0) >= PERSONAL_NOTES_GATE_THRESHOLD]
                        context["personal_notes"] = gated_notes[:PROMPT_MAX_PERSONAL_NOTES]
                        logger.debug(f"Gated personal notes: {len(personal_notes)} -> {pre_filter_count} (gate) -> {len(context['personal_notes'])} (threshold={PERSONAL_NOTES_GATE_THRESHOLD})")
                    except Exception as gate_err:
                        logger.warning(f"Personal notes gating failed, keeping original: {gate_err}")

                # Filter reference docs by relevance threshold to prevent
                # semantically-distant content from polluting the prompt
                reference_docs = context.get("reference_docs", [])
                if reference_docs:
                    pre_count = len(reference_docs)
                    reference_docs = [d for d in reference_docs
                                      if d.get("relevance_score", 0) >= REFERENCE_DOCS_GATE_THRESHOLD]
                    context["reference_docs"] = reference_docs[:PROMPT_MAX_REFERENCE_DOCS]
                    logger.debug(f"Reference docs gate: {pre_count} -> {len(context['reference_docs'])} (threshold={REFERENCE_DOCS_GATE_THRESHOLD})")
            except Exception as e:
                logger.warning(f"Gating failed: {e}")

            # Step 6: Apply hygiene and caps
            logger.debug(f"BEFORE HYGIENE_AND_CAPS: memories count = {len(context.get('memories', []))}")
            context = await self._hygiene_and_caps(context, stm_summary=stm_summary)

            # Step 6.1: Top-up relevant memories if cross-effects reduced them too much.
            try:
                mems = context.get("memories", []) or []
                recents = context.get("recent_conversations", []) or []
                if len(mems) < PROMPT_MAX_MEMS:
                    # Pull extra recent conversations beyond the ones already shown
                    extra_recent = await self.context_gatherer._get_recent_conversations(PROMPT_MAX_RECENT + PROMPT_MAX_MEMS)
                    # Build keys for already used items
                    def _key(x):
                        return (str(x.get("query", "")) + str(x.get("response", ""))).strip().lower()

                    # CRITICAL: Check against BOTH recent_conversations AND existing memories to avoid duplicates
                    used = {_key(r) for r in recents}
                    used.update({_key(m) for m in mems})  # Also check against existing memories!

                    # Keep only items not already in either section
                    filler = []
                    skipped_count = 0
                    for item in extra_recent:
                        if _key(item) not in used:
                            filler.append(item)
                        else:
                            skipped_count += 1

                    needed = max(0, PROMPT_MAX_MEMS - len(mems))
                    if needed:
                        mems.extend(filler[:needed])
                        context["memories"] = mems
                        logger.debug(f"MEMORY TOP-UP: Added {min(needed, len(filler))} new memories (had {len(mems) - min(needed, len(filler))}, target {PROMPT_MAX_MEMS}), skipped {skipped_count} duplicates")
            except Exception as e:
                logger.warning(f"Memory top-up failed: {e}")

            logger.debug(f"AFTER MEMORY TOP-UP: memories count = {len(context.get('memories', []))}")

            # Step 6.2: Ensure minimum summaries and reflections by pulling directly from storage
            try:
                logger.debug(f"START OF SUMMARIES BLOCK: memories count = {len(context.get('memories', []))}")
                # Summaries — if we have too few, pull most recent without gating
                if len(context.get("summaries", []) or []) < PROMPT_MAX_SUMMARIES:
                    needed = PROMPT_MAX_SUMMARIES - len(context.get("summaries", []))
                    try:
                        # try memory_coordinator first (supports sync or async)
                        if hasattr(self.memory_coordinator, 'get_summaries'):
                            logger.debug(f"BEFORE get_summaries: memories count = {len(context.get('memories', []))}")
                            res = self.memory_coordinator.get_summaries(PROMPT_MAX_SUMMARIES * 2)
                            import asyncio as _asyncio
                            stored = await res if _asyncio.iscoroutine(res) else res
                            logger.debug(f"AFTER get_summaries: memories count = {len(context.get('memories', []))}, stored type = {type(stored).__name__}")
                        elif hasattr(self.memory_coordinator, 'corpus_manager') and hasattr(self.memory_coordinator.corpus_manager, 'get_summaries'):
                            stored = self.memory_coordinator.corpus_manager.get_summaries(PROMPT_MAX_SUMMARIES * 2)
                        else:
                            stored = []
                    except Exception as e:
                        logger.warning(f"[PromptBuilder] Summary retrieval failed: {e}")
                        stored = []

                    # Keep the newest not already in context
                    # Normalize stored schema (legacy may use 'response'/'text')
                    norm = []
                    for s in (stored or []):
                        if isinstance(s, dict):
                            if not s.get('content'):
                                c = s.get('response') or s.get('text')
                                if c:
                                    s = {**s, 'content': c}
                        norm.append(s)
                    stored = norm

                    have = { (s.get('content') or '').strip() for s in (context.get('summaries') or []) if isinstance(s, dict) }
                    add = []
                    for s in (stored or [])[::-1]:  # assume stored oldest->newest; reverse to pick newest first
                        if isinstance(s, dict) and (s.get('content') or '').strip() and (s.get('content').strip() not in have):
                            add.append(s)
                            have.add(s.get('content').strip())
                        if len(add) >= needed:
                            break
                    if add:
                        context['summaries'] = (context.get('summaries') or []) + add

                # Reflections — if too few, pull most recent historical reflections
                if len(context.get("reflections", []) or []) < PROMPT_MAX_REFLECTIONS:
                    needed = PROMPT_MAX_REFLECTIONS - len(context.get("reflections", []))
                    stored_refl = []
                    try:
                        if hasattr(self.memory_coordinator, 'get_reflections'):
                            # get_reflections may be async; try both
                            res = self.memory_coordinator.get_reflections(PROMPT_MAX_REFLECTIONS * 3)
                            if asyncio.iscoroutine(res):
                                stored_refl = await res
                            else:
                                stored_refl = res
                        elif hasattr(self.memory_coordinator, 'corpus_manager') and hasattr(self.memory_coordinator.corpus_manager, 'get_reflections'):
                            res2 = self.memory_coordinator.corpus_manager.get_reflections(PROMPT_MAX_REFLECTIONS * 3)
                            stored_refl = res2 if isinstance(res2, list) else list(res2)
                    except Exception as e:
                        logger.warning(f"[PromptBuilder] Reflection retrieval failed: {e}")
                        stored_refl = []

                    have_refl = { (r.get('content') or '').strip() for r in (context.get('reflections') or []) if isinstance(r, dict) }
                    add_refl = []
                    for r in (stored_refl or [])[::-1]:
                        if isinstance(r, dict):
                            content = (r.get('content') or '').strip()
                            if content and content not in have_refl:
                                add_refl.append(r)
                                have_refl.add(content)
                            if len(add_refl) >= needed:
                                break
                    if add_refl:
                        context['reflections'] = (context.get('reflections') or []) + add_refl
            except (TypeError, AttributeError, KeyError) as e:
                logger.debug(f"Reflection pre-budget top-up failed: {e}")

            # Step 6.9: LLM-compress heavily oversized items (async pre-pass)
            # Items >= 3x over their token limit get LLM summary instead of middle-out slicing.
            # Mildly oversized items (1x-3x) still use middle-out in token_manager.
            context = await self._llm_compress_oversized(context)

            # Step 7: Token budget management
            logger.debug(f"BEFORE TOKEN BUDGET: memories count = {len(context.get('memories', []))}")
            context = self.token_manager._manage_token_budget(context)
            logger.debug(f"AFTER TOKEN BUDGET: memories count = {len(context.get('memories', []))}")

            # Step 7.1: Post-budget floors for critical sections
            # Ensure recent conversations, summaries, and reflections are not
            # dropped entirely by budget trimming.
            try:
                # Recent conversations floor — guarantee session context survives
                recent_convos = context.get("recent_conversations", []) or []
                if len(recent_convos) < PROMPT_MIN_RECENT_FLOOR:
                    needed_recent = PROMPT_MIN_RECENT_FLOOR - len(recent_convos)
                    try:
                        stored_recent = await self.context_gatherer._get_recent_conversations(PROMPT_MIN_RECENT_FLOOR * 2)
                    except Exception as e:
                        logger.debug(f"Failed to fetch recent conversations for floor: {e}")
                        stored_recent = []
                    if stored_recent:
                        def _recent_key(x):
                            return (str(x.get("query", "")) + str(x.get("response", ""))).strip().lower()
                        have_keys = {_recent_key(r) for r in recent_convos}
                        add_recent = []
                        for r in stored_recent:
                            if isinstance(r, dict) and _recent_key(r) not in have_keys:
                                add_recent.append(r)
                                have_keys.add(_recent_key(r))
                            if len(add_recent) >= needed_recent:
                                break
                        if add_recent:
                            context['recent_conversations'] = (context.get('recent_conversations') or []) + add_recent
                            logger.info(f"[POST-BUDGET FLOOR] Restored {len(add_recent)} recent conversations (had {len(recent_convos)}, floor={PROMPT_MIN_RECENT_FLOOR})")

                # Summaries floor
                if len(context.get("summaries", []) or []) < PROMPT_MAX_SUMMARIES:
                    needed = PROMPT_MAX_SUMMARIES - len(context.get("summaries", []))
                    stored = []
                    try:
                        if hasattr(self.memory_coordinator, 'get_summaries'):
                            res = self.memory_coordinator.get_summaries(PROMPT_MAX_SUMMARIES * 3)
                            import asyncio as _asyncio
                            stored = await res if _asyncio.iscoroutine(res) else res
                        elif hasattr(self.memory_coordinator, 'corpus_manager') and hasattr(self.memory_coordinator.corpus_manager, 'get_summaries'):
                            stored = self.memory_coordinator.corpus_manager.get_summaries(PROMPT_MAX_SUMMARIES * 3)
                        else:
                            stored = []
                    except (AttributeError, TypeError) as e:
                        logger.debug(f"Failed to fetch summaries for floor: {e}")
                        stored = []

                    # Normalize stored schema
                    norm = []
                    for s in (stored or []):
                        if isinstance(s, dict) and not s.get('content'):
                            c = s.get('response') or s.get('text')
                            if c:
                                s = {**s, 'content': c}
                        norm.append(s)
                    stored = norm

                    have = { (s.get('content') or '').strip() for s in (context.get('summaries') or []) if isinstance(s, dict) }
                    add = []
                    for s in (stored or [])[::-1]:
                        if isinstance(s, dict):
                            content = (s.get('content') or '').strip()
                            if content and content not in have:
                                add.append(s)
                                have.add(content)
                            if len(add) >= needed:
                                break
                    if add:
                        context['summaries'] = (context.get('summaries') or []) + add

                logger.warning(f"AFTER SUMMARIES TOP-UP: memories count = {len(context.get('memories', []))}")

                # Reflections floor
                if len(context.get("reflections", []) or []) < PROMPT_MAX_REFLECTIONS:
                    needed = PROMPT_MAX_REFLECTIONS - len(context.get("reflections", []))
                    stored_refl = []
                    try:
                        if hasattr(self.memory_coordinator, 'get_reflections'):
                            res = self.memory_coordinator.get_reflections(PROMPT_MAX_REFLECTIONS * 3)
                            import asyncio as _asyncio
                            if _asyncio.iscoroutine(res):
                                stored_refl = await res
                            else:
                                stored_refl = res
                        elif hasattr(self.memory_coordinator, 'corpus_manager') and hasattr(self.memory_coordinator.corpus_manager, 'get_reflections'):
                            res2 = self.memory_coordinator.corpus_manager.get_reflections(PROMPT_MAX_REFLECTIONS * 3)
                            stored_refl = res2 if isinstance(res2, list) else list(res2)
                    except (AttributeError, TypeError) as e:
                        logger.debug(f"Failed to fetch reflections for floor: {e}")
                        stored_refl = []

                    # Normalize stored reflections schema
                    norm_r = []
                    for r in (stored_refl or []):
                        if isinstance(r, dict) and not r.get('content'):
                            c = r.get('response') or r.get('text')
                            if c:
                                r = {**r, 'content': c}
                        norm_r.append(r)
                    stored_refl = norm_r

                    have_refl = { (r.get('content') or '').strip() for r in (context.get('reflections') or []) if isinstance(r, dict) }
                    add_refl = []
                    for r in (stored_refl or [])[::-1]:
                        if isinstance(r, dict):
                            content = (r.get('content') or '').strip()
                            if content and content not in have_refl:
                                add_refl.append(r)
                                have_refl.add(content)
                            if len(add_refl) >= needed:
                                break
                    if add_refl:
                        context['reflections'] = (context.get('reflections') or []) + add_refl
            except (TypeError, AttributeError, KeyError) as e:
                logger.debug(f"Post-budget floor top-up failed: {e}")

            logger.warning(f"BEFORE FINAL ASSEMBLY: memories count = {len(context.get('memories', []))}")

            # Step 8: Final context assembly
            prompt_ctx = {
                "recent_conversations": context.get("recent_conversations", []),
                "memories": context.get("memories", []),
                "user_profile": context.get("user_profile", ""),  # Replaces semantic_facts + fresh_facts
                "narrative_state": context.get("narrative_state", ""),  # Temporal grounding (synthesized life context)
                "summaries": context.get("summaries", []),
                "recent_summaries": context.get("recent_summaries", []),
                "semantic_summaries": context.get("semantic_summaries", []),
                "reflections": context.get("reflections", []),
                "recent_reflections": context.get("recent_reflections", []),
                "semantic_reflections": context.get("semantic_reflections", []),
                "dreams": context.get("dreams", []),
                "semantic_chunks": context.get("semantic_chunks", []),
                "wiki": context.get("wiki", []),
                "personal_notes": context.get("personal_notes", []),  # User's Obsidian notes
                "reference_docs": context.get("reference_docs", []),  # Daemon self-knowledge docs
                "user_uploads": context.get("user_uploads", []),     # User uploaded files/images
                "git_commits": context.get("git_commits", []),      # Git commit history
                "procedural_skills": context.get("procedural_skills", []),  # Adaptive workflows
                "proposed_features": context.get("proposed_features", []),  # Code proposals
                "graph_context": context.get("graph_context", []),  # Knowledge graph relationships
                "unresolved_threads": context.get("unresolved_threads", []),  # Proactive thread surfacing
                "proactive_insights": context.get("proactive_insights", []),  # Cross-domain insights
                "visual_memories": context.get("visual_memories", {"text_results": [], "images": []}),  # CLIP visual memories
                "web_search_results": context.get("web_search_results"),  # Real-time web search results
                "stm_summary": context.get("stm_summary"),  # STM context summary (dict or None)
                "memory_id_map": self.context_gatherer.memory_id_map if hasattr(self.context_gatherer, 'memory_id_map') else {}
            }

            build_time = time.time() - start_time
            logger.info(f"Prompt built in {build_time:.2f}s")
            logger.warning(f"RETURNING CONTEXT: memories count = {len(prompt_ctx.get('memories', []))}")

            # Attach timing metadata for interpretability (underscore-prefixed to avoid collision)
            prompt_ctx["_task_timings"] = dict(task_timings)
            prompt_ctx["_gather_elapsed"] = locals().get('_gather_elapsed', 0.0)
            prompt_ctx["_build_time"] = build_time

            return prompt_ctx

        except Exception as e:
            logger.error(f"Prompt building failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return minimal context on error
            error_context = {
                "recent_conversations": [],
                "memories": [],
                "user_profile": "",
                "narrative_state": "",
                "summaries": [],
                "reflections": [],
                "dreams": [],
                "semantic_chunks": [],
                "wiki": [],
                "personal_notes": [],
                "user_uploads": [],
                "git_commits": [],
                "procedural_skills": [],
                "proposed_features": [],
                "graph_context": [],
                "unresolved_threads": [],
                "proactive_insights": [],
                "web_search_results": None,
                "memory_id_map": {}
            }
            # Include stm_summary if it was provided
            if stm_summary is not None:
                error_context["stm_summary"] = stm_summary
            return error_context

    async def build_prompt_from_context(
        self,
        context: "ContextResult",
        memories: Optional[List[Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build prompt from a ContextResult object.

        This method provides a clean interface for building prompts from the
        ContextPipeline's output. It maps ContextResult fields to the existing
        build_prompt parameters.

        Args:
            context: ContextResult from ContextPipeline.build()
            memories: Optional pre-retrieved memories (if not provided, will be gathered)
            config: Optional configuration overrides

        Returns:
            Dict containing the built prompt context

        Example:
            context = await context_pipeline.build(user_input, files)
            prompt_ctx = await prompt_builder.build_prompt_from_context(context)
            final_prompt = prompt_builder._assemble_prompt(prompt_ctx, user_input)
        """
        # Import here to avoid circular dependency
        from core.context_pipeline import ContextResult

        if not isinstance(context, ContextResult):
            raise TypeError(f"Expected ContextResult, got {type(context)}")

        # Extract intent overrides (if intent classifier ran)
        retrieval_overrides = {}
        weight_overrides = {}
        gate_threshold_override = None
        if hasattr(context, 'intent') and context.intent is not None:
            retrieval_overrides = context.intent.retrieval_overrides or {}
            weight_overrides = context.intent.weight_overrides or {}
            gate_threshold_override = getattr(context.intent, 'gate_threshold_override', None)

        # Extract intent type for web search gating
        _intent_type = None
        if context.intent is not None:
            _it = getattr(context.intent, 'intent_type', None)
            _intent_type = getattr(_it, 'value', str(_it)) if _it else None

        # Map ContextResult to build_prompt parameters
        # When files are uploaded, pass flag to suppress reference docs
        # so file content dominates the context window
        return await self.build_prompt(
            user_input=context.processed_query,
            config=config,
            search_query=context.processed_query if context.processed_query != context.original_query else None,
            current_topic=context.primary_topic,
            fresh_facts=context.extracted_facts if context.is_heavy_topic else None,
            memories=memories,
            stm_summary=context.stm_summary,
            crisis_level=context.crisis_level_str,
            retrieval_overrides=retrieval_overrides,
            weight_overrides=weight_overrides,
            intent_type=_intent_type,
            _suppress_reference_docs=context.has_files,
            _gate_threshold_override=gate_threshold_override,
        )

    async def _build_lightweight_context(self, user_input: str, stm_summary: Optional[Dict[str, Any]] = None,
                                          codebase_changes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build lightweight context for small-talk queries."""
        try:
            # Just get recent conversations for small-talk
            recent = await self.context_gatherer._get_recent_conversations(3)

            context = {
                "recent_conversations": recent,
                "memories": [],
                "user_profile": "",
                "narrative_state": "",  # No narrative context for small-talk
                "summaries": [],
                "recent_summaries": [],
                "semantic_summaries": [],
                "reflections": [],
                "recent_reflections": [],
                "semantic_reflections": [],
                "dreams": [],
                "semantic_chunks": [],
                "wiki": [],
                "personal_notes": [],  # No personal notes for small-talk
                "user_uploads": [],   # No uploads for small-talk
                "git_commits": [],
                "procedural_skills": [],
                "proposed_features": [],  # No proposals for small-talk
                "graph_context": [],  # No graph for small-talk
                "unresolved_threads": [],  # No threads for small-talk
                "proactive_insights": [],  # No insights for small-talk
                "web_search_results": None,  # No web search for small-talk
                "codebase_changes": codebase_changes or {},  # Git changes since last session
            }

            # Add STM summary if provided
            if stm_summary is not None:
                context["stm_summary"] = stm_summary

            # Add memory ID map for citations
            context["memory_id_map"] = self.context_gatherer.memory_id_map if hasattr(self.context_gatherer, 'memory_id_map') else {}

            # Ambiguity detection: check if short user message references a phrase
            # that appears in multiple sessions (prevents content conflation)
            try:
                from core.ambiguity_detector import AmbiguityDetector
                ambiguity = AmbiguityDetector.detect(
                    user_input,
                    context.get("recent_conversations", []),
                )
                if ambiguity.is_ambiguous:
                    context["disambiguation_notes"] = [ambiguity.disambiguation_note]
                    logger.info(
                        f"[AmbiguityDetector] Detected: '{ambiguity.ambiguous_phrase}' "
                        f"in {len(ambiguity.matching_entries)} entries across sessions"
                    )
            except Exception as e:
                logger.debug(f"[AmbiguityDetector] Detection failed (non-fatal): {e}")

            return context
        except Exception as e:
            logger.warning(f"Lightweight context building failed: {e}")
            return {
                "recent_conversations": [],
                "memories": [],
                "user_profile": "",
                "narrative_state": "",
                "summaries": [],
                "recent_summaries": [],
                "memory_id_map": {},
                "semantic_summaries": [],
                "reflections": [],
                "recent_reflections": [],
                "semantic_reflections": [],
                "dreams": [],
                "semantic_chunks": [],
                "wiki": [],
                "personal_notes": [],
                "user_uploads": [],
                "git_commits": [],
                "procedural_skills": [],
                "proposed_features": [],
                "graph_context": [],
                "unresolved_threads": [],
                "proactive_insights": [],
                "web_search_results": None,
                "codebase_changes": codebase_changes or {},
            }

    async def _hygiene_and_caps(self, context: Dict[str, Any], stm_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply deduplication and caps. Delegates to ContentHygiene."""
        return await self._hygiene._hygiene_and_caps(context, stm_summary)

    async def _backfill_recent_conversations(
        self,
        existing_items: List[Dict[str, Any]],
        seen_embeddings: List[tuple],
        seen_content: set,
        target_count: int,
        offset: int,
        embedder,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Backfill recent conversations. Delegates to ContentHygiene."""
        return await self._hygiene._backfill_recent_conversations(
            existing_items, seen_embeddings, seen_content,
            target_count, offset, embedder, similarity_threshold
        )

    def get_token_count(self, text: str, model_name: str) -> int:
        """Get token count for text."""
        return self.token_manager.get_token_count(text, model_name)

    def _extract_text(self, item: Any) -> str:
        """Extract text from various item formats."""
        return self.token_manager._extract_text(item)

    # Legacy support methods
    async def _gather_context(self, user_input: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Legacy context gathering method - delegates to build_prompt."""
        return await self.build_prompt(user_input, config)

    def _build_feature_inventory(self, context: Dict[str, Any]) -> str:
        """Build a compact feature inventory. Delegates to PromptFormatter."""
        return self.formatter._build_feature_inventory(context)

    def _assemble_prompt(self, context: Dict[str, Any] = None, user_input: str = "",
                        directives: str = "", system_prompt: str = "", **kwargs) -> str:
        """Assemble final prompt string from context. Delegates to PromptFormatter."""
        return self.formatter._assemble_prompt(context, user_input, directives, system_prompt, **kwargs)


# Legacy compatibility class
class PromptBuilder:
    """
    Legacy compatibility wrapper for UnifiedPromptBuilder.

    Provides the old interface for backwards compatibility.
    """

    def __init__(self, model_manager_or_memory_coordinator=None, model_manager=None, **kwargs):
        # Handle both old and new calling conventions
        if model_manager is None and hasattr(model_manager_or_memory_coordinator, 'generate'):
            # Old style: PromptBuilder(model_manager)
            model_manager = model_manager_or_memory_coordinator
            memory_coordinator = None
        else:
            # New style: PromptBuilder(memory_coordinator, model_manager)
            memory_coordinator = model_manager_or_memory_coordinator

        self.unified_builder = UnifiedPromptBuilder(
            memory_coordinator=memory_coordinator,
            model_manager=model_manager,
            **kwargs
        )
        # Expose common attributes for backward compatibility
        self.model_manager = model_manager

    def _assemble_prompt(self, user_input: str = "", context: Dict[str, Any] = None,
                        system_prompt: str = "", directives: str = "", **kwargs) -> str:
        """Expose _assemble_prompt method for backward compatibility.

        Handles both signatures:
        - Legacy: _assemble_prompt(user_input=..., context=..., system_prompt=...)
        - New: _assemble_prompt(context, user_input, directives)
        """
        # Debug logging
        logger.debug(f"_assemble_prompt called with: user_input={type(user_input)}, context={type(context)}")

        # Handle different calling conventions
        if context is None:
            context = {}

        # Use system_prompt as directives if directives not provided
        if system_prompt and not directives:
            directives = system_prompt

        return self.unified_builder._assemble_prompt(context, user_input, directives)

    async def build_prompt(self, user_input: str = "", config: Optional[Dict[str, Any]] = None,
                          memories=None, summaries=None, dreams=None, wiki_snippet=None,
                          semantic_chunks=None, semantic_memory_results=None,
                          time_context=None, recent_conversations=None, **kwargs) -> str:
        """Build prompt and return formatted string.

        Supports both new interface (user_input, config) and legacy interface
        (user_input with specific argument overrides).
        """
        logger.debug(f"PROMPT BUILD LEGACY: Got {len(memories) if memories else 0} memories from parameters")
        if any([memories is not None, summaries is not None, dreams is not None,
                wiki_snippet is not None, semantic_chunks is not None]):
            # Legacy interface - build context manually
            context = {
                "recent_conversations": recent_conversations or [],
                "memories": memories or [],
                "user_profile": "",
                "summaries": summaries or [],
                "reflections": [],
                "dreams": dreams or [],
                "semantic_chunks": semantic_chunks or [],
                "wiki": [{"content": wiki_snippet}] if wiki_snippet else [],
                "proposed_features": [],
                "graph_context": [],
                "unresolved_threads": [],
                "proactive_insights": [],
            }
            return self.unified_builder._assemble_prompt(context, user_input)
        else:
            # New interface - delegate to UnifiedPromptBuilder
            context = await self.unified_builder.build_prompt(user_input, config)
            return self.unified_builder._assemble_prompt(context, user_input)
