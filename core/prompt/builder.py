"""
# core/prompt/builder.py

Module Contract
- Purpose: Main UnifiedPromptBuilder orchestrating complete prompt assembly with parallel
  async retrieval, intent-driven overrides, graph-boosted scoring, and token budget management.
- Key methods:
  - build_prompt(user_input, config, context_result, user_input_for_search,
      max_recent, max_mems, max_facts, is_meta_conversational, weight_overrides,
      gate_threshold_overrides, session_files) -> Tuple[str, Dict]
    Main entry point: parallel retrieval → hygiene → format → token budget → assemble.
    Sets/clears intent weight overrides and graph refs on scorer around retrieval.
  - build_prompt_from_context(user_input, config, context_result, ...) -> Tuple[str, Dict]
    Lightweight path skipping full retrieval (uses pre-gathered context).
  - _llm_compress_oversized(context) -> Dict
    Pre-compresses items ≥3x over token limit via LLM before middle-out fallback.
  - _build_feature_inventory(context) -> str
    Generates [ACTIVE FEATURES] section from config flags and context counts.
  - _hygiene_and_caps(context, stm_summary) -> Dict
    Deduplication, caps, staleness prefixes, on-demand reflections.
  - Post-budget floors (Step 7.1): Guarantees minimum recent_conversations (PROMPT_MIN_RECENT_FLOOR=5),
    summaries (PROMPT_MAX_SUMMARIES), and reflections (PROMPT_MAX_REFLECTIONS) survive budget trimming.
- Outputs:
  - Complete formatted prompt string ready for LLM consumption
  - Context dictionary with all assembled data, metadata, and performance metrics
- Dependencies:
  - .context_gatherer.ContextGatherer (parallel async data retrieval)
  - .formatter.PromptFormatter (section assembly via _assemble_prompt)
  - .summarizer.LLMSummarizer (on-demand reflections and summaries)
  - .token_manager.TokenManager (budget enforcement, middle-out compression)
  - utils.time_manager.format_relative_timestamp (relative day labels on timestamps)
  - .base._FallbackMemoryCoordinator (testing fallback)
  - processing.gate_system (relevance filtering)
  - memory.memory_scorer (intent weight overrides, graph refs set/cleared per call)
- Side effects:
  - Memory system queries and parallel data retrieval
  - LLM API calls for summarization, reflection, and oversized item compression
  - Sets/clears _intent_weight_overrides and _graph_memory/_entity_resolver on scorer
  - Comprehensive logging and performance metrics

Prompt Section Order:
  [RECENT CONVERSATION] → [RELEVANT MEMORIES] → [USER PROFILE] → [SUMMARIES] →
  [REFLECTIONS] → [DREAMS] → [USER'S PERSONAL NOTES] → [USER UPLOADED ITEMS] →
  [DAEMON DOCUMENTATION] → [PROJECT COMMIT HISTORY] → [ADAPTIVE WORKFLOWS] →
  [PROPOSED FEATURES] → [KNOWLEDGE GRAPH] → [UNRESOLVED THREADS] →
  [PROACTIVE INSIGHTS] → [USER PROFILE] → [ACTIVE FEATURES] →
  [CODEBASE CHANGES SINCE LAST SESSION] →
  [WEB SEARCH RESULTS] → [RELEVANT INFORMATION] →
  [TIME CONTEXT] → [TEMPORAL GROUNDING] → [STM SUMMARY] → [CURRENT USER QUERY]
"""

import os
import time
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from utils.time_manager import TimeManager
from utils.query_checker import analyze_query
from memory.memory_consolidator import MemoryConsolidator
from utils.logging_utils import get_logger, log_and_time

# Import the modular components
from .context_gatherer import (
    ContextGatherer,
    PROMPT_MAX_RECENT_SUMMARIES,
    PROMPT_MAX_SEMANTIC_SUMMARIES,
    PROMPT_MAX_RECENT_REFLECTIONS,
    PROMPT_MAX_SEMANTIC_REFLECTIONS
)
from .formatter import PromptFormatter, _parse_bool, _dedupe_keep_order, _truncate_list, _sanitize_embedded_headers
from .summarizer import LLMSummarizer
from .token_manager import TokenManager
from .base import _FallbackMemoryCoordinator

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

def _staleness_prefix(item) -> str:
    """Return a staleness prefix for highly stale summaries/reflections.

    Items with staleness_ratio >= STALENESS_HISTORICAL_THRESHOLD get prefixed
    with [HISTORICAL — PARTIALLY OUTDATED] to signal the LLM to treat claims
    skeptically.
    """
    try:
        from config.app_config import STALENESS_ENABLED, STALENESS_HISTORICAL_THRESHOLD
        if not STALENESS_ENABLED:
            return ""
        if isinstance(item, dict):
            md = item.get("metadata", {}) or {}
            ratio = float(md.get("staleness_ratio", 0) or item.get("staleness_ratio", 0) or 0)
        else:
            return ""
        if ratio >= STALENESS_HISTORICAL_THRESHOLD:
            return "[HISTORICAL — PARTIALLY OUTDATED] "
    except Exception:
        pass
    return ""
PROMPT_MAX_REFERENCE_DOCS = _cfg_int("prompt_max_reference_docs", 15)
PROMPT_MAX_GIT_COMMITS = _cfg_int("prompt_max_git_commits", 10)
PROMPT_MAX_SKILLS = _cfg_int("prompt_max_skills", 5)
PROMPT_MAX_PROPOSALS = _cfg_int("prompt_max_proposals", 3)
PROMPT_MAX_USER_UPLOADS = _cfg_int("prompt_max_user_uploads", 5)
PROMPT_MAX_GRAPH_SENTENCES = _cfg_int("prompt_max_graph_sentences", 12)
PROMPT_MAX_SURFACED_THREADS = _cfg_int("prompt_max_surfaced_threads", 3)
PROMPT_MAX_PROACTIVE_INSIGHTS = _cfg_int("prompt_max_proactive_insights", 2)

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
        MULTIMODAL_MODELS,
        PERSONAL_NOTES_GATE_THRESHOLD,
    )
except ImportError:
    OBSIDIAN_INCLUDE_IMAGES = True
    OBSIDIAN_MAX_IMAGES_PER_NOTE = 3
    MULTIMODAL_MODELS = ["opus-4", "claude-3", "sonnet-4", "gpt-4o", "gemini"]
    PERSONAL_NOTES_GATE_THRESHOLD = 0.30


def _is_multimodal_model(model_id: str) -> bool:
    """Check if a model ID corresponds to a multimodal-capable model."""
    if not model_id:
        return False
    model_lower = model_id.lower()
    return any(pattern.lower() in model_lower for pattern in MULTIMODAL_MODELS)

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


def _load_upload_image(image_path: str) -> Optional[dict]:
    """
    Load a persisted upload image from disk as base64 for multimodal API calls.

    Args:
        image_path: Path to the image file on disk

    Returns:
        Dict with 'data', 'media_type', 'filename' keys, or None if loading fails
    """
    import base64
    from pathlib import Path

    try:
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            logger.debug(f"[_load_upload_image] File not found: {image_path}")
            return None

        # Enforce 5MB cap to avoid memory issues
        if path.stat().st_size > 5 * 1024 * 1024:
            logger.warning(f"[_load_upload_image] File too large (>5MB), skipping: {image_path}")
            return None

        # Determine media type from extension
        ext = path.suffix.lower()
        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        media_type = media_types.get(ext, 'application/octet-stream')

        data = base64.b64encode(path.read_bytes()).decode('utf-8')
        return {
            'data': data,
            'media_type': media_type,
            'filename': path.name,
        }
    except Exception as e:
        logger.warning(f"[_load_upload_image] Failed to load {image_path}: {e}")
        return None


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
            narrative_state = ""
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

            # Step 3: Launch parallel data gathering tasks with per-task timing
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
            tasks["summaries"] = asyncio.create_task(
                _timed_task("summaries", self.context_gatherer._get_summaries_separate(user_input, eff_max_summaries_r, eff_max_summaries_s))
            )

            # Dreams (if enabled)
            tasks["dreams"] = asyncio.create_task(
                _timed_task("dreams", self.context_gatherer._get_dreams(eff_max_dreams))
            )

            # Semantic chunks
            tasks["semantic"] = asyncio.create_task(
                _timed_task("semantic", self.context_gatherer._get_semantic_chunks(user_input, max_results=eff_max_semantic))
            )

            # Reflections (separated into recent + semantic)
            tasks["reflections"] = asyncio.create_task(
                _timed_task("reflections", self.context_gatherer._get_reflections_separate(user_input, eff_max_reflections_r, eff_max_reflections_s))
            )

            # Wiki content
            tasks["wiki"] = asyncio.create_task(
                _timed_task("wiki", self.context_gatherer._get_wiki_content(user_input, eff_max_wiki))
            )

            # Personal notes from Obsidian vault
            # Check if model is multimodal to decide whether to load images
            current_model = getattr(self.model_manager, 'active_model_name', '') if self.model_manager else ''
            include_note_images = OBSIDIAN_INCLUDE_IMAGES and _is_multimodal_model(current_model)
            logger.warning(f"[PromptBuilder] IMAGE DEBUG: model={current_model}, OBSIDIAN_INCLUDE_IMAGES={OBSIDIAN_INCLUDE_IMAGES}, is_multimodal={_is_multimodal_model(current_model)}, include_note_images={include_note_images}")

            tasks["personal_notes"] = asyncio.create_task(
                _timed_task("personal_notes", self.context_gatherer.get_personal_notes(
                    user_input,
                    PROMPT_MAX_PERSONAL_NOTES,
                    include_images=include_note_images,
                    max_images_per_note=OBSIDIAN_MAX_IMAGES_PER_NOTE
                ))
            )

            # Reference documents (system docs, project outlines - excludes user uploads)
            # Suppressed when user uploads files so file content dominates context
            if not kwargs.get("_suppress_reference_docs", False):
                tasks["reference_docs"] = asyncio.create_task(
                    _timed_task("reference_docs", self.context_gatherer.get_reference_docs(user_input, PROMPT_MAX_REFERENCE_DOCS))
                )

            # User uploads (previously uploaded files/images)
            tasks["user_uploads"] = asyncio.create_task(
                _timed_task("user_uploads", self.context_gatherer.get_user_uploads(user_input, PROMPT_MAX_USER_UPLOADS))
            )

            # Git commit history (procedural memory)
            tasks["git_commits"] = asyncio.create_task(
                _timed_task("git_commits", self.context_gatherer.get_git_commits(user_input, eff_max_git))
            )

            # Procedural skills (adaptive workflows)
            tasks["procedural_skills"] = asyncio.create_task(
                _timed_task("procedural_skills", self.context_gatherer.get_procedural_skills(user_input, eff_max_skills))
            )

            # Proposed features (code proposals, only for project-related queries)
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
            tasks["proactive_insights"] = asyncio.create_task(
                _timed_task("proactive_insights", self.context_gatherer.get_proactive_insights(user_input, PROMPT_MAX_PROACTIVE_INSIGHTS))
            )

            # Web search (triggered based on query analysis, suppressed during crisis)
            tasks["web_search"] = asyncio.create_task(
                _timed_task("web_search", self.context_gatherer._get_web_search_results(user_input, crisis_level, intent_type=intent_type))
            )

            # Gather all results with timeout
            _gather_start = time.time()
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks.values(), return_exceptions=True),
                    timeout=30.0
                )
                _gather_elapsed = time.time() - _gather_start

                # Map results back to names
                gathered = {}
                for i, (name, _) in enumerate(tasks.items()):
                    result = results[i]
                    if isinstance(result, Exception):
                        logger.warning(f"Task {name} failed: {result}")
                        gathered[name] = []
                    else:
                        gathered[name] = result or []
                        if name == "memories":
                            logger.debug(f"MEMORIES TASK: Got {len(result) if result else 0} memories")
                        if name == "proposed_features":
                            logger.info(f"[PROPOSED_FEATURES] Task returned {len(result) if result else 0} proposals")

                # Sort tasks by time (slowest first) for easy identification of bottlenecks
                sorted_timings = sorted(task_timings.items(), key=lambda x: x[1], reverse=True)
                timing_str = " | ".join([f"{k}={v:.2f}s" for k, v in sorted_timings])
                logger.info(
                    f"[BUILD_PROMPT TIMING] total={_gather_elapsed:.2f}s | {timing_str}"
                )

            except asyncio.TimeoutError:
                logger.warning("Data gathering timed out, using partial results")
                gathered = {name: [] for name in tasks.keys()}
            finally:
                # Clear intent weight overrides from scorer (set before gather)
                if scorer and weight_overrides:
                    scorer._intent_weight_overrides = None
                # Clear graph references from scorer
                if scorer:
                    scorer._graph_memory = None
                    scorer._entity_resolver = None

            # Step 3: Post-fetch processing

            # Handle separated summaries (recent + semantic)
            summaries_data = gathered.get("summaries", {})
            logger.warning(f"CONTEXT GATHERING: summaries_data = {summaries_data}, type = {type(summaries_data)}")
            if isinstance(summaries_data, dict):
                recent_summaries = summaries_data.get("recent", [])
                semantic_summaries = summaries_data.get("semantic", [])
                all_summaries = recent_summaries + semantic_summaries
                logger.warning(f"CONTEXT GATHERING: Extracted {len(recent_summaries)} recent, {len(semantic_summaries)} semantic summaries")
            else:
                # Backward compatibility for old format
                all_summaries = summaries_data or []
                recent_summaries = []
                semantic_summaries = []
                logger.warning(f"CONTEXT GATHERING: Using old format, got {len(all_summaries)} summaries")

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
            logger.warning(f"[DEBUG RECENT] build_prompt: Got {len(recent_convos)} recent_conversations from gatherer")
            if recent_convos:
                # Log first 3 and last 3 with timestamps
                for i in range(min(3, len(recent_convos))):
                    mem = recent_convos[i]
                    ts = mem.get('timestamp', 'NO_TS')
                    query = mem.get('query', '')[:80]
                    logger.warning(f"[DEBUG RECENT] Item {i+1} (first): ts={ts}, query={query}...")
                if len(recent_convos) > 3:
                    for i in range(max(0, len(recent_convos) - 3), len(recent_convos)):
                        mem = recent_convos[i]
                        ts = mem.get('timestamp', 'NO_TS')
                        query = mem.get('query', '')[:80]
                        logger.warning(f"[DEBUG RECENT] Item {i+1} (last): ts={ts}, query={query}...")

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

                # Reference docs bypass gating — they're authored system knowledge
                # already filtered by hybrid retrieval (keyword + semantic) in
                # get_documents().  Double-gating drops most docs because
                # conversational queries rarely cosine-match architecture docs.
                reference_docs = context.get("reference_docs", [])
                if reference_docs:
                    context["reference_docs"] = reference_docs[:PROMPT_MAX_REFERENCE_DOCS]
                    logger.debug(f"Reference docs (no gate): {len(reference_docs)} -> {len(context['reference_docs'])}")
            except Exception as e:
                logger.warning(f"Gating failed: {e}")

            # Step 6: Apply hygiene and caps
            logger.warning(f"BEFORE HYGIENE_AND_CAPS: memories count = {len(context.get('memories', []))}")
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
                        logger.warning(f"MEMORY TOP-UP: Added {min(needed, len(filler))} new memories (had {len(mems) - min(needed, len(filler))}, target {PROMPT_MAX_MEMS}), skipped {skipped_count} duplicates")
            except Exception as e:
                logger.warning(f"Memory top-up failed: {e}")

            logger.warning(f"AFTER MEMORY TOP-UP: memories count = {len(context.get('memories', []))}")

            # Step 6.2: Ensure minimum summaries and reflections by pulling directly from storage
            try:
                logger.warning(f"START OF SUMMARIES BLOCK: memories count = {len(context.get('memories', []))}, context id = {id(context)}")
                # Summaries — if we have too few, pull most recent without gating
                if len(context.get("summaries", []) or []) < PROMPT_MAX_SUMMARIES:
                    needed = PROMPT_MAX_SUMMARIES - len(context.get("summaries", []))
                    try:
                        # try memory_coordinator first (supports sync or async)
                        if hasattr(self.memory_coordinator, 'get_summaries'):
                            logger.warning(f"BEFORE get_summaries: memories count = {len(context.get('memories', []))}, context id = {id(context)}")
                            res = self.memory_coordinator.get_summaries(PROMPT_MAX_SUMMARIES * 2)
                            import asyncio as _asyncio
                            stored = await res if _asyncio.iscoroutine(res) else res
                            logger.warning(f"AFTER get_summaries: memories count = {len(context.get('memories', []))}, context id = {id(context)}, stored type = {type(stored)}")
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
            logger.warning(f"BEFORE TOKEN BUDGET: memories count = {len(context.get('memories', []))}")
            context = self.token_manager._manage_token_budget(context)
            logger.warning(f"AFTER TOKEN BUDGET: memories count = {len(context.get('memories', []))}")

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
                "web_search_results": context.get("web_search_results"),  # Real-time web search results
                "stm_summary": context.get("stm_summary"),  # STM context summary (dict or None)
                "memory_id_map": self.context_gatherer.memory_id_map if hasattr(self.context_gatherer, 'memory_id_map') else {}
            }

            build_time = time.time() - start_time
            logger.info(f"Prompt built in {build_time:.2f}s")
            logger.warning(f"RETURNING CONTEXT: memories count = {len(prompt_ctx.get('memories', []))}")

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
        if hasattr(context, 'intent') and context.intent is not None:
            retrieval_overrides = context.intent.retrieval_overrides or {}
            weight_overrides = context.intent.weight_overrides or {}

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
        """
        Apply deduplication and caps to all context sections.

        This ensures we don't have duplicate content and stay within
        reasonable limits for each content type.
        """
        # Debug: Log that we're starting dedup
        section_counts = {k: len(v) if isinstance(v, list) else 1 for k, v in context.items() if v}
        logger.info(f"[DEDUP START] Sections with content: {section_counts}")

        # Apply deduplication and caps to all sections
        sections_to_process = [
            "recent_conversations", "memories",
            "summaries", "recent_summaries", "semantic_summaries",
            "reflections", "recent_reflections", "semantic_reflections",
            "dreams", "semantic_chunks", "wiki"
        ]

        for section in sections_to_process:
            items = context.get(section, [])
            if not items:
                continue

            # Deduplicate
            if isinstance(items, list):
                # For memories and conversations, dedupe by content
                if section in ["recent_conversations", "memories"]:
                    original_count = len(items)
                    # Handle both content field (hybrid retriever) and query/response fields (corpus)
                    def dedup_key(x):
                        # Try content field first (from hybrid retriever)
                        content = x.get("content", "")
                        if content:
                            return content.strip().lower()
                        # Fallback to query/response
                        return str(x.get("response", "") + x.get("query", "")).strip().lower()

                    deduped = _dedupe_keep_order(items, key_fn=dedup_key)
                    logger.debug(f"ASSEMBLY DEDUP {section}: {original_count} -> {len(deduped)} items")
                else:
                    # For others, dedupe by string representation
                    deduped = _dedupe_keep_order(items)

                context[section] = deduped

        # Cross-section deduplication to catch content appearing in multiple sections
        # This is critical for avoiding duplicate ICE responses in conversations/memories
        # NOTE: We only dedup conversations/memories across each other, NOT summaries/reflections
        # because those need to stay in their dedicated sections with proper headers

        # String-based cross-section dedup (normalized first 500 chars).
        # Previously used embedding-based O(n²) cosine similarity which added 300-500ms.
        # String dedup catches the vast majority of exact/near-exact duplicates at ~0 cost.
        seen_content = set()

        cross_dedup_sections = [
            "recent_conversations", "memories", "personal_notes"
        ]

        # Track target counts for backfilling
        target_counts = {
            "recent_conversations": 10,  # Target number of unique recent conversations
            "memories": 30,  # Target number of unique memories
            "personal_notes": PROMPT_MAX_PERSONAL_NOTES  # Target number of personal notes
        }

        for section in cross_dedup_sections:
            items = context.get(section, [])
            if not items or not isinstance(items, list):
                continue

            target_count = target_counts.get(section, len(items))
            original_count = len(items)

            deduplicated = []
            for item in items:
                # Extract content for dedup check
                if isinstance(item, dict):
                    content = item.get("content", "")
                    if not content:
                        response = item.get("response", "")
                        content = response if response else str(item.get("query", ""))
                else:
                    content = str(item)

                # Normalize content for comparison
                normalized = content.strip().lower()
                for prefix in ["user:", "daemon:", "luke,"]:
                    if normalized.startswith(prefix):
                        normalized = normalized[len(prefix):].strip()

                dedup_key = normalized[:500]
                if dedup_key and dedup_key not in seen_content:
                    seen_content.add(dedup_key)
                    deduplicated.append(item)
                else:
                    logger.debug(f"CROSS-SECTION DEDUP: Skipped duplicate in {section} (key: {dedup_key[:80]}...)")

            original_count = len(items)
            if len(deduplicated) < original_count:
                logger.info(f"CROSS-SECTION DEDUP {section}: {original_count} -> {len(deduplicated)} items (removed {original_count - len(deduplicated)} duplicates)")

            context[section] = deduplicated

            # Backfill if we're below target after deduplication
            if len(deduplicated) < target_count and section == "recent_conversations":
                logger.info(f"[BACKFILL] {section} has {len(deduplicated)}/{target_count} items, fetching more...")

                backfill_result = await self._backfill_recent_conversations(
                    existing_items=deduplicated,
                    seen_embeddings=[],
                    seen_content=seen_content,
                    target_count=target_count,
                    offset=original_count,
                    embedder=None,
                    similarity_threshold=0.90
                )

                context[section] = backfill_result

        # Stitch semantic chunks by title
        semantic_chunks = context.get("semantic_chunks", [])
        if semantic_chunks:
            # Group by title and stitch content
            chunks_by_title = {}
            for chunk in semantic_chunks:
                title = chunk.get("title", "")
                if title:
                    if title not in chunks_by_title:
                        chunks_by_title[title] = chunk.copy()
                    else:
                        # Combine content
                        existing = chunks_by_title[title]
                        existing_content = existing.get("content", "")
                        new_content = chunk.get("content", "")
                        combined = f"{existing_content}\n\n{new_content}"

                        # Apply length limit
                        if len(combined) <= 4000:  # SEM_STITCH_MAX_CHARS
                            existing["content"] = combined

            context["semantic_chunks"] = list(chunks_by_title.values())

        # Add STM summary if provided
        if stm_summary is not None:
            context["stm_summary"] = stm_summary
            logger.debug(f"Added STM summary to context: topic={stm_summary.get('topic')}")

        return context

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
        """
        Backfill recent conversations to reach target count after deduplication.

        Fetches additional conversations from corpus and deduplicates them against
        existing items until we reach the target count or run out of conversations.

        Args:
            existing_items: Already deduplicated items
            seen_embeddings: List of (embedding, item) tuples for semantic dedup
            seen_content: Set of content keys for string-based dedup
            target_count: Target number of unique items
            offset: Starting offset in corpus
            embedder: Sentence embedder for semantic similarity
            similarity_threshold: Threshold for considering items duplicates

        Returns:
            List of deduplicated items (may be less than target_count if corpus exhausted)
        """
        import numpy as np

        deduplicated = existing_items.copy()
        batch_size = target_count - len(deduplicated)
        max_iterations = 10  # Safety limit
        iteration = 0

        logger.info(f"[BACKFILL] Starting with {len(deduplicated)} items, target={target_count}")

        while len(deduplicated) < target_count and iteration < max_iterations:
            iteration += 1

            # Fetch next batch from corpus
            try:
                if not self.memory_coordinator:
                    logger.warning("[BACKFILL] No memory_coordinator available")
                    break

                corpus_manager = getattr(self.memory_coordinator, 'corpus_manager', None)
                if not corpus_manager:
                    logger.warning("[BACKFILL] No corpus_manager in memory_coordinator")
                    break

                # Get more recent conversations from corpus
                all_recent = corpus_manager.get_recent_memories(
                    count=offset + batch_size
                )

                # Slice to get only the new batch
                if len(all_recent) <= offset:
                    logger.info(f"[BACKFILL] No more items in corpus (have {len(all_recent)}, offset={offset})")
                    break

                additional_items = all_recent[offset:offset + batch_size]

                if not additional_items:
                    logger.info(f"[BACKFILL] No more items available")
                    break

                logger.debug(f"[BACKFILL] Iteration {iteration}: fetched {len(additional_items)} items (offset={offset})")

                # Deduplicate new items against existing ones
                added_count = 0
                for item in additional_items:
                    # Extract content
                    if isinstance(item, dict):
                        content = item.get("content", "")
                        if not content:
                            response = item.get("response", "")
                            content = response if response else str(item.get("query", ""))
                    else:
                        content = str(item)

                    # Normalize
                    normalized = content.strip().lower()
                    for prefix in ["user:", "daemon:", "luke,"]:
                        if normalized.startswith(prefix):
                            normalized = normalized[len(prefix):].strip()

                    is_duplicate = False

                    # Check against existing deduplicated items
                    if embedder:
                        try:
                            item_embedding = embedder.encode(normalized[:512], convert_to_numpy=True)

                            for seen_emb, _ in seen_embeddings:
                                similarity = np.dot(item_embedding, seen_emb) / (
                                    np.linalg.norm(item_embedding) * np.linalg.norm(seen_emb) + 1e-8
                                )

                                if similarity >= similarity_threshold:
                                    is_duplicate = True
                                    logger.debug(f"[BACKFILL] Skipped duplicate (similarity={similarity:.3f})")
                                    break

                            if not is_duplicate:
                                seen_embeddings.append((item_embedding, item))
                                deduplicated.append(item)
                                added_count += 1

                                if len(deduplicated) >= target_count:
                                    break

                        except Exception as e:
                            logger.debug(f"[BACKFILL] Embedding failed: {e}")
                            # Fallback to string-based
                            dedup_key = normalized[:500]
                            if dedup_key and dedup_key not in seen_content:
                                seen_content.add(dedup_key)
                                deduplicated.append(item)
                                added_count += 1
                    else:
                        # String-based fallback
                        dedup_key = normalized[:500]
                        if dedup_key and dedup_key not in seen_content:
                            seen_content.add(dedup_key)
                            deduplicated.append(item)
                            added_count += 1

                    if len(deduplicated) >= target_count:
                        break

                logger.info(f"[BACKFILL] Iteration {iteration}: added {added_count} unique items, now have {len(deduplicated)}/{target_count}")

                # Update offset for next batch
                offset += len(additional_items)

                # If we didn't add any unique items, increase batch size
                if added_count == 0:
                    batch_size = min(batch_size * 2, 50)  # Double batch size up to 50
                else:
                    batch_size = target_count - len(deduplicated)

                if len(deduplicated) >= target_count:
                    break

            except Exception as e:
                logger.warning(f"[BACKFILL] Failed to fetch additional items: {e}")
                break

        logger.info(f"[BACKFILL] Complete: {len(deduplicated)}/{target_count} items after {iteration} iterations")
        return deduplicated

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
        """Build a compact feature inventory showing which systems are active and what they returned.

        Reads config flags and counts results from the context dict.
        No retrieval needed — purely reads config flags and context dict counts.

        Returns:
            Compact multi-line string grouped by category, or empty string.
        """
        try:
            from config import app_config as cfg

            def _on_off(flag: bool) -> str:
                return "ON" if flag else "OFF"

            def _count(key: str, fallback=None) -> str:
                """Get count annotation from context, e.g. '(3)' or ''."""
                val = context.get(key, fallback)
                if val is None:
                    return ""
                if isinstance(val, list):
                    return f"({len(val)})" if val else "(0)"
                if isinstance(val, str):
                    return f"({len(val.split(chr(10)))})" if val.strip() else "(0)"
                if isinstance(val, dict):
                    total = sum(len(v) for v in val.values() if isinstance(v, list))
                    return f"({total})" if total else "(0)"
                return ""

            lines = []

            # Memory category
            mem_parts = []
            kg_enabled = getattr(cfg, 'KNOWLEDGE_GRAPH_ENABLED', False)
            kg_ctx = context.get("graph_context", []) or []
            mem_parts.append(f"knowledge_graph={_on_off(kg_enabled)}{f'({len(kg_ctx)} edges)' if kg_ctx else ''}")
            mem_parts.append(f"fact_verification={_on_off(getattr(cfg, 'FACT_VERIFICATION_ENABLED', False))}")
            mem_parts.append(f"truth_scorer={_on_off(getattr(cfg, 'TRUTH_SCORER_ENABLED', True))}")
            mem_parts.append(f"dedup={_on_off(getattr(cfg, 'CROSS_DEDUP_ENABLED', False))}")
            lines.append("Memory: " + " | ".join(mem_parts))

            # Knowledge category
            know_parts = []
            git = context.get("git_commits", []) or []
            know_parts.append(f"git_commits={_on_off(getattr(cfg, 'GIT_MEMORY_ENABLED', False))}{f'({len(git)})' if git else ''}")
            notes = context.get("personal_notes", []) or []
            know_parts.append(f"obsidian={_on_off(bool(notes))}{f'({len(notes)} notes)' if notes else ''}")
            ref_docs = context.get("reference_docs", []) or []
            know_parts.append(f"reference_docs={_on_off(getattr(cfg, 'REFERENCE_DOCS_AUTO_SEED', False))}{f'({len(ref_docs)})' if ref_docs else ''}")
            web = context.get("web_search_results")
            know_parts.append(f"web_search={_on_off(getattr(cfg, 'WEB_SEARCH_ENABLED', False))}{_count('web_search_results')}")
            lines.append("Knowledge: " + " | ".join(know_parts))

            # Proactive category
            pro_parts = []
            threads = context.get("unresolved_threads", []) or []
            pro_parts.append(f"threads={_on_off(getattr(cfg, 'THREAD_SURFACING_ENABLED', False))}{f'({len(threads)} open)' if threads else ''}")
            insights = context.get("proactive_insights", []) or []
            pro_parts.append(f"insights={_on_off(getattr(cfg, 'PROACTIVE_SURFACING_ENABLED', False))}{f'({len(insights)})' if insights else ''}")
            pro_parts.append(f"narrative={_on_off(getattr(cfg, 'NARRATIVE_CONTEXT_ENABLED', True) if hasattr(cfg, 'NARRATIVE_CONTEXT_ENABLED') else bool(context.get('narrative_state')))}")
            lines.append("Proactive: " + " | ".join(pro_parts))

            # Analysis category
            ana_parts = []
            ana_parts.append(f"intent={_on_off(getattr(cfg, 'INTENT_ENABLED', False))}")
            ana_parts.append(f"escalation={_on_off(getattr(cfg, 'ESCALATION_ENABLED', False))}")
            skills = context.get("procedural_skills", []) or []
            ana_parts.append(f"skills={_on_off(getattr(cfg, 'PROCEDURAL_SKILLS_ENABLED', False))}{f'({len(skills)})' if skills else ''}")
            lines.append("Analysis: " + " | ".join(ana_parts))

            return "\n".join(lines)

        except Exception as e:
            logger.debug(f"[PromptBuilder] Feature inventory failed: {e}")
            return ""

    def _assemble_prompt(self, context: Dict[str, Any] = None, user_input: str = "",
                        directives: str = "", system_prompt: str = "", **kwargs) -> str:
        """
        Assemble final prompt string from context with numbering and timestamp-first entries.

        Section order (optimized for attention and token efficiency):
        1. Recent conversations (baseline context)
        2. Relevant memories (semantic hits)
        3. Recent summaries (compressed recent history)
        4. Semantic summaries (relevant compressed history)
        5. Background knowledge (wiki)
        6. Relevant information (semantic chunks)
        7. Recent reflections (meta insights)
        8. Semantic reflections (relevant meta insights)
        9. Dreams (if enabled)
        10. User profile (cheap, high attention, personalization)
        11. Time context (cheap, high attention, temporal grounding)
        12. STM summary (short-term context, maximum attention)
        13. Current user query (always last)

        Note: User profile and time context moved to positions 10-11 (from 4 and 1) to leverage
        recency bias in LLM attention while keeping token cost minimal.
        """
        if context is None:
            context = {}
        if system_prompt and not directives:
            directives = system_prompt

        from datetime import datetime
        logger.warning(f"PROMPT ASSEMBLY START: context has {len(context)} keys: {list(context.keys())}")
        logger.warning(f"PROMPT ASSEMBLY START: recent_summaries={len(context.get('recent_summaries', []))}, semantic_summaries={len(context.get('semantic_summaries', []))}")
        logger.warning(f"PROMPT ASSEMBLY START: stm_summary present = {context.get('stm_summary') is not None}, value = {context.get('stm_summary')}")

        def mem_parts(mem: Dict[str, Any]) -> tuple[str, str]:
            try:
                # Memory field structure varies by source:
                # - Hybrid retriever uses 'content' field
                # - Corpus manager uses 'query'/'response' fields
                # Try content field first (from hybrid retriever)
                content_field = mem.get("content", "")

                if content_field:
                    # Content field has full conversation text
                    content = content_field.strip()
                else:
                    # Fallback to query/response format
                    q = str(mem.get("query", ""))
                    r = str(mem.get("response", ""))

                    # Build the content
                    if q and r:
                        content = f"User: {q.strip()}\nDaemon: {r.strip()}"
                    elif r:
                        content = f"Daemon: {r.strip()}"
                    elif q:
                        content = f"User: {q.strip()}"
                    else:
                        content = str(mem)

                # Get timestamp (may be in root or metadata)
                ts = mem.get("timestamp", "")
                if not ts:
                    ts = mem.get("metadata", {}).get("timestamp", "")

                # Get tags
                tags = mem.get("tags", [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
                elif not tags:
                    tags = []
                tags_str = ", ".join(str(tag) for tag in tags) if tags else ""

                # Format timestamp with relative day label to prevent temporal hallucinations
                from utils.time_manager import format_relative_timestamp
                if isinstance(ts, datetime):
                    ts_str = format_relative_timestamp(ts)
                elif ts:
                    # Try parsing ISO string for relative formatting
                    try:
                        ts_str = format_relative_timestamp(datetime.fromisoformat(str(ts)))
                    except (ValueError, TypeError):
                        ts_str = str(ts)
                else:
                    ts_str = ""

                # Add tags
                if tags_str and content:
                    content += f"\nTags: {{{tags_str}}}"

                # Sanitize any embedded section headers to prevent prompt pollution
                content = _sanitize_embedded_headers(content)

                return content, ts_str
            except (AttributeError, TypeError, KeyError):
                return str(mem), ""

        sections: list[str] = []

        # Recent conversations
        recent = context.get("recent_conversations", []) or []
        logger.warning(f"[DEBUG RECENT] _assemble_prompt: Got {len(recent)} items in recent_conversations")
        recent_lines: list[str] = []
        for i, mem in enumerate(recent, start=1):
            content, ts = mem_parts(mem)
            # Debug: Log first 3 and last 3 items
            if i <= 3 or i > len(recent) - 3:
                logger.warning(f"[DEBUG RECENT] Item {i}: ts={ts}, content_preview={content[:100] if content else 'EMPTY'}...")
            recent_lines.append(f"{i}) {ts}: {content}" if ts else f"{i}) {content}")
        if recent_lines:
            logger.warning(f"[DEBUG RECENT] Adding [RECENT CONVERSATION] section with {len(recent_lines)} formatted entries")
            sections.append(f"[RECENT CONVERSATION] n={len(recent_lines)}\n" + "\n\n".join(recent_lines))

        # Relevant memories
        memories = context.get("memories", []) or []
        logger.warning(f"PROMPT BUILD: FINAL COUNT - Got {len(memories)} memories from context BEFORE ASSEMBLY")
        memory_lines: list[str] = []
        for i, mem in enumerate(memories, start=1):
            content, ts = mem_parts(mem)
            memory_lines.append(f"{i}) {ts}: {content}" if ts else f"{i}) {content}")
        if memory_lines:
            sections.append(f"[RELEVANT MEMORIES] n={len(memory_lines)}\n" + "\n\n".join(memory_lines))
            logger.warning(f"PROMPT BUILD: FINAL COUNT - [RELEVANT MEMORIES] section will contain {len(memory_lines)} memories")
        else:
            logger.warning("PROMPT BUILD: FINAL COUNT - No memories to display in [RELEVANT MEMORIES] section")

        # Recent Summaries
        recent_summaries = context.get("recent_summaries", []) or []
        logger.warning(f"PROMPT ASSEMBLY: Got {len(recent_summaries)} recent summaries")
        recent_sum_lines: list[str] = []
        for i, s in enumerate(recent_summaries, start=1):
            if isinstance(s, dict):
                content = s.get("content", "") or str(s)
                ts = s.get("timestamp", "")
            else:
                content = str(s)
                ts = ""
            if content:
                content = _sanitize_embedded_headers(content)
                prefix = _staleness_prefix(s)
                recent_sum_lines.append(f"{i}) {ts}: {prefix}{content}" if ts else f"{i}) {prefix}{content}")
        if recent_sum_lines:
            sections.append(f"[RECENT SUMMARIES] n={len(recent_sum_lines)}\n" + "\n\n".join(recent_sum_lines))
            logger.warning(f"PROMPT ASSEMBLY: Added recent summaries section with {len(recent_sum_lines)} items")
        else:
            logger.warning("PROMPT ASSEMBLY: No recent summaries to add")

        # Semantic Summaries
        semantic_summaries = context.get("semantic_summaries", []) or []
        semantic_sum_lines: list[str] = []
        for i, s in enumerate(semantic_summaries, start=1):
            if isinstance(s, dict):
                content = s.get("content", "") or str(s)
                ts = s.get("timestamp", "")
            else:
                content = str(s)
                ts = ""
            if content:
                content = _sanitize_embedded_headers(content)
                prefix = _staleness_prefix(s)
                semantic_sum_lines.append(f"{i}) {ts}: {prefix}{content}" if ts else f"{i}) {prefix}{content}")
        if semantic_sum_lines:
            sections.append(f"[SEMANTIC SUMMARIES] n={len(semantic_sum_lines)}\n" + "\n\n".join(semantic_sum_lines))

        # Recent Reflections
        recent_reflections = context.get("recent_reflections", []) or []
        recent_refl_lines: list[str] = []
        for i, r in enumerate(recent_reflections, start=1):
            if isinstance(r, dict):
                content = r.get("content", "") or str(r)
                ts = r.get("timestamp", "")
            else:
                content = str(r)
                ts = ""
            if content:
                content = _sanitize_embedded_headers(content)
                prefix = _staleness_prefix(r)
                recent_refl_lines.append(f"{i}) {ts}: {prefix}{content}" if ts else f"{i}) {prefix}{content}")
        if recent_refl_lines:
            sections.append(f"[RECENT REFLECTIONS] n={len(recent_refl_lines)}\n" + "\n\n".join(recent_refl_lines))

        # Semantic Reflections
        semantic_reflections = context.get("semantic_reflections", []) or []
        semantic_refl_lines: list[str] = []
        for i, r in enumerate(semantic_reflections, start=1):
            if isinstance(r, dict):
                content = r.get("content", "") or str(r)
                ts = r.get("timestamp", "")
            else:
                content = str(r)
                ts = ""
            if content:
                content = _sanitize_embedded_headers(content)
                prefix = _staleness_prefix(r)
                semantic_refl_lines.append(f"{i}) {ts}: {prefix}{content}" if ts else f"{i}) {prefix}{content}")
        if semantic_refl_lines:
            sections.append(f"[SEMANTIC REFLECTIONS] n={len(semantic_refl_lines)}\n" + "\n\n".join(semantic_refl_lines))

        # Wiki content
        wiki = context.get("wiki", []) or []
        wiki_lines: list[str] = []
        for i, w in enumerate(wiki, start=1):
            if isinstance(w, dict):
                content = w.get("content", "")
                title = w.get("title", "")
                block = f"**{title}**\n{content}" if title and content else (content or str(w))
            else:
                block = str(w)
            wiki_lines.append(f"{i}) {block}")
        if wiki_lines:
            sections.append(f"[BACKGROUND KNOWLEDGE] n={len(wiki_lines)}\n" + "\n\n".join(wiki_lines))

        # Web search results (real-time web content)
        web_search = context.get("web_search_results")
        if web_search is not None:
            try:
                # Handle WebSearchResult object
                if hasattr(web_search, 'has_results') and web_search.has_results:
                    pages = web_search.pages
                    from_cache = web_search.from_cache
                    ws_lines: list[str] = []
                    for i, page in enumerate(pages[:5], start=1):  # Limit to 5 results
                        title = page.title if hasattr(page, 'title') else page.get('title', '')
                        url = page.url if hasattr(page, 'url') else page.get('url', '')
                        content = (page.content if hasattr(page, 'content') else page.get('content', '')) or \
                                  (page.snippet if hasattr(page, 'snippet') else page.get('snippet', ''))
                        if content:
                            # Truncate long content
                            if len(content) > 2000:
                                content = content[:2000] + "..."
                            ws_lines.append(f"{i}) **{title}** ({url})\n{content}")
                    if ws_lines:
                        cache_note = " (cached)" if from_cache else ""
                        sections.append(f"[WEB SEARCH RESULTS] n={len(ws_lines)}{cache_note}\n" + "\n\n".join(ws_lines))
                        logger.info(f"[PROMPT ASSEMBLY] Added web search section with {len(ws_lines)} results")
            except Exception as e:
                logger.warning(f"[PROMPT ASSEMBLY] Failed to format web search results: {e}")

        # Semantic chunks
        chunks = context.get("semantic_chunks", []) or []
        sc_lines: list[str] = []
        for i, c in enumerate(chunks, start=1):
            if isinstance(c, dict):
                content = c.get("filtered_content", "") or c.get("content", "")
                title = c.get("title", "")
                block = f"**{title}**\n{content}" if title and content else (content or str(c))
            else:
                block = str(c)
            sc_lines.append(f"{i}) {block}")
        if sc_lines:
            sections.append(f"[RELEVANT INFORMATION] n={len(sc_lines)}\n" + "\n\n".join(sc_lines))

        # Dreams
        dreams = context.get("dreams", []) or []
        dr_lines: list[str] = []
        for i, d in enumerate(dreams, start=1):
            if isinstance(d, dict):
                content = d.get("content", "") or str(d)
                ts = d.get("timestamp", "")
            else:
                content = str(d)
                ts = ""
            dr_lines.append(f"{i}) {ts}: {content}" if ts else f"{i}) {content}")
        if dr_lines:
            sections.append(f"[DREAMS] n={len(dr_lines)}\n" + "\n\n".join(dr_lines))

        # Personal Notes from Obsidian vault
        personal_notes = context.get("personal_notes", []) or []
        pn_lines: list[str] = []
        note_images: list[dict] = []  # Collect images for multimodal models

        for i, note in enumerate(personal_notes, start=1):
            if isinstance(note, dict):
                title = note.get("metadata", {}).get("title", "")
                section = note.get("metadata", {}).get("section", "")
                tags = note.get("metadata", {}).get("tags", "")
                content = note.get("content", "")
                image_data = note.get("image_data", [])  # Base64 encoded images
                # Sanitize content to prevent embedded headers
                content = _sanitize_embedded_headers(content) if content else ""
            else:
                title, section, tags, content = "", "", "", str(note)
                image_data = []

            if content:
                # Build header: **Title** (Section) #tag1 #tag2
                header_parts = []
                if title:
                    header_parts.append(f"**{title}**")
                if section:
                    header_parts.append(f"({section})")
                if tags:
                    # Convert comma-separated tags to hashtag format
                    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
                    if tag_list:
                        header_parts.append(" ".join(f"#{t}" for t in tag_list))

                # Add relevance score so the LLM can see match strength
                relevance = note.get("relevance_score", 0.0)
                if relevance > 0:
                    header_parts.append(f"[relevance: {relevance:.2f}]")

                # Add image indicator if images are present
                if image_data:
                    header_parts.append(f"[{len(image_data)} image(s) attached]")
                    # Collect images with context about which note they belong to
                    for img in image_data:
                        note_images.append({
                            "note_index": i,
                            "note_title": title,
                            "note_section": section,
                            "filename": img.get("filename", ""),
                            "media_type": img.get("media_type", ""),
                            "data": img.get("data", ""),
                        })

                header = " ".join(header_parts) if header_parts else ""
                pn_lines.append(f"{i}) {header}\n{content}" if header else f"{i}) {content}")

        if pn_lines:
            sections.append(f"[USER'S PERSONAL NOTES] n={len(pn_lines)}\n" + "\n\n".join(pn_lines))

        # Store images in context for multimodal API calls
        if note_images:
            context["note_images"] = note_images
            total_data_size = sum(len(img.get("data", "")) for img in note_images)
            logger.warning(f"[PromptBuilder] IMAGE DEBUG: {len(note_images)} images collected, total base64 size={total_data_size//1024}KB")
        else:
            # Check why no images
            total_image_data = sum(len(note.get("image_data", [])) for note in personal_notes if isinstance(note, dict))
            logger.warning(f"[PromptBuilder] IMAGE DEBUG: No images in note_images list. personal_notes has {len(personal_notes)} notes, total image_data entries={total_image_data}")

        # User Uploaded Items (files and images uploaded during sessions)
        user_uploads = context.get("user_uploads", []) or []
        uu_lines: list[str] = []
        upload_images: list[dict] = []  # Collect images for multimodal models

        for i, upload in enumerate(user_uploads, start=1):
            if isinstance(upload, dict):
                meta = upload.get("metadata", {})
                title = meta.get("title", "")
                is_image = meta.get("is_image", False)
                media_type = meta.get("media_type", "")
                image_path = meta.get("image_path", "")
                content = upload.get("content", "")
                content = _sanitize_embedded_headers(content) if content else ""
            else:
                title, is_image, media_type, image_path, content = "", False, "", "", str(upload)

            if content:
                header_parts = []
                if title:
                    # Strip "upload:" prefix for cleaner display
                    display_title = title[7:] if title.startswith("upload:") else title
                    header_parts.append(f"**{display_title}**")
                if is_image:
                    header_parts.append(f"[image: {media_type}]")
                    # Load persisted image for multimodal API calls
                    if image_path:
                        img_data = _load_upload_image(image_path)
                        if img_data:
                            upload_images.append({
                                "note_index": 0,
                                "note_title": f"Upload: {title}",
                                "note_section": "",
                                "filename": img_data.get("filename", ""),
                                "media_type": img_data.get("media_type", ""),
                                "data": img_data.get("data", ""),
                            })
                header = " ".join(header_parts) if header_parts else ""
                uu_lines.append(f"{i}) {header}\n{content}" if header else f"{i}) {content}")

        if uu_lines:
            sections.append(f"[USER UPLOADED ITEMS] n={len(uu_lines)}\n" + "\n\n".join(uu_lines))

        # Merge upload images into note_images for multimodal API calls
        if upload_images:
            existing_images = context.get("note_images", [])
            existing_images.extend(upload_images)
            context["note_images"] = existing_images
            logger.debug(f"[PromptBuilder] Merged {len(upload_images)} upload images into note_images")

        # Reference Documents (system docs, project outlines, etc.)
        reference_docs = context.get("reference_docs", []) or []
        rd_lines: list[str] = []
        for i, doc in enumerate(reference_docs, start=1):
            if isinstance(doc, dict):
                title = doc.get("metadata", {}).get("title", "")
                section = doc.get("metadata", {}).get("section", "")
                file_type = doc.get("metadata", {}).get("file_type", "")
                content = doc.get("content", "")
                # Sanitize content to prevent embedded headers
                content = _sanitize_embedded_headers(content) if content else ""
            else:
                title, section, file_type, content = "", "", "", str(doc)

            if content:
                # Build header: **Title** (Section) [type]
                header_parts = []
                if title:
                    header_parts.append(f"**{title}**")
                if section:
                    header_parts.append(f"({section})")
                if file_type:
                    header_parts.append(f"[{file_type}]")
                header = " ".join(header_parts) if header_parts else ""
                rd_lines.append(f"{i}) {header}\n{content}" if header else f"{i}) {content}")

        if rd_lines:
            sections.append(f"[DAEMON DOCUMENTATION] n={len(rd_lines)}\n" + "\n\n".join(rd_lines))

        # Git commit history (procedural memory)
        git_commits = context.get("git_commits", []) or []
        gc_lines: list[str] = []
        for i, commit in enumerate(git_commits, start=1):
            if isinstance(commit, dict):
                content = commit.get("content", "")
                meta = commit.get("metadata", {})
                commit_hash = meta.get("commit_hash", "")
                author = meta.get("author", "")
                age = meta.get("age_relative", "")
                tags = meta.get("tags", "")
            else:
                content = str(commit)
                commit_hash, author, age, tags = "", "", "", ""

            if content:
                header_parts = []
                if commit_hash:
                    header_parts.append(f"[{commit_hash}]")
                if author:
                    header_parts.append(f"by {author}")
                if age:
                    header_parts.append(f"({age})")
                if tags:
                    tag_list = [t.strip() for t in tags.split(",") if t.strip() and t.strip() != "git-commit"]
                    if tag_list:
                        header_parts.append(" ".join(f"#{t}" for t in tag_list))
                header = " ".join(header_parts) if header_parts else ""
                gc_lines.append(f"{i}) {header}\n{content}" if header else f"{i}) {content}")

        if gc_lines:
            sections.append(f"[PROJECT COMMIT HISTORY] n={len(gc_lines)}\n" + "\n\n".join(gc_lines))

        # Procedural skills (adaptive workflows)
        proc_skills = context.get("procedural_skills", []) or []
        sk_lines: list[str] = []
        for i, skill in enumerate(proc_skills, start=1):
            if isinstance(skill, dict):
                meta = skill.get("metadata", {})
                trigger = meta.get("trigger", "")
                action = meta.get("action_pattern", "")
                category = meta.get("category", "")
                confidence = meta.get("confidence", "")
                tags_raw = meta.get("tags_json", "")
                created_at = meta.get("created_at", 0)
            else:
                trigger = str(skill)
                action, category, confidence, tags_raw, created_at = "", "", "", "", 0

            if trigger and action:
                parts = []
                if category:
                    parts.append(f"[{category}]")
                # Relative age from created_at epoch
                if created_at:
                    try:
                        import time as _time
                        age_secs = _time.time() - float(created_at)
                        if age_secs < 3600:
                            age_str = f"{int(age_secs / 60)} minutes ago"
                        elif age_secs < 86400:
                            age_str = f"{int(age_secs / 3600)} hours ago"
                        else:
                            age_str = f"{int(age_secs / 86400)} days ago"
                        parts.append(f"({age_str})")
                    except (ValueError, TypeError):
                        pass
                if confidence:
                    try:
                        parts.append(f"(conf={float(confidence):.0%})")
                    except (ValueError, TypeError):
                        pass
                if tags_raw:
                    try:
                        import json as _json
                        tag_list = _json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
                        if tag_list:
                            parts.append(" ".join(f"#{t}" for t in tag_list))
                    except Exception:
                        pass
                header = " ".join(parts) if parts else ""
                entry = f"{i}) {header}\nWHEN: {trigger}\nTHEN: {action}" if header else f"{i}) WHEN: {trigger}\nTHEN: {action}"
                sk_lines.append(entry)

        if sk_lines:
            sections.append(f"[ADAPTIVE WORKFLOWS] n={len(sk_lines)}\n" + "\n\n".join(sk_lines))

        # Proposed Features (code proposals surfaced for project-related queries)
        proposed_features = context.get("proposed_features", [])
        logger.info(f"[PROPOSED_FEATURES] _assemble_prompt: {len(proposed_features)} proposals in context")
        pf_lines = []
        for i, pf in enumerate(proposed_features, 1):
            meta = pf.get("metadata", {})
            title = meta.get("title", "Untitled")
            ptype = meta.get("proposal_type", "feature")
            priority = meta.get("priority", 5)
            tags_raw = meta.get("tags_json", "[]")
            reasoning = meta.get("reasoning", "")

            try:
                import json as _json
                tag_list = _json.loads(tags_raw) if isinstance(tags_raw, str) else (tags_raw or [])
            except Exception:
                tag_list = []

            tag_str = " ".join(f"#{t}" for t in tag_list) if tag_list else ""
            header = f"[{ptype}] P{priority}"
            if tag_str:
                header += f" {tag_str}"
            header += f" **{title}**"

            entry = f"{i}) {header}"
            if reasoning:
                entry += f"\n   Rationale: {reasoning[:200]}"
            pf_lines.append(entry)

        if pf_lines:
            sections.append(f"[PROPOSED FEATURES] n={len(pf_lines)}\n" + "\n\n".join(pf_lines))

        # Knowledge Graph context (entity relationships)
        graph_sentences = context.get("graph_context", []) or []
        if graph_sentences:
            graph_block = "\n".join(f"- {s}" for s in graph_sentences)
            sections.append(f"[KNOWLEDGE GRAPH] n={len(graph_sentences)}\n{graph_block}")

        # Unresolved threads (proactive surfacing)
        unresolved_threads = context.get("unresolved_threads", []) or []
        if unresolved_threads:
            thread_lines = []
            for t in unresolved_threads:
                ttype = t.get("thread_type", "unfinished")
                topic = t.get("topic", "")
                summary = t.get("summary", "")
                deadline = t.get("deadline_date")
                line = f"- [{ttype}] {topic}: {summary}"
                if deadline:
                    line += f" (deadline: {deadline})"
                thread_lines.append(line)
            sections.append(f"[UNRESOLVED THREADS] n={len(thread_lines)}\n" + "\n".join(thread_lines))

        # Proactive cross-domain insights
        proactive_insights = context.get("proactive_insights", []) or []
        if proactive_insights:
            insight_block = "\n".join(f"- {s}" for s in proactive_insights)
            sections.append(
                f"[PROACTIVE INSIGHTS] n={len(proactive_insights)}\n"
                "These are non-obvious connections across different areas of the user's life. "
                "Weave them in naturally IF relevant. Do NOT announce them as insights.\n"
                f"{insight_block}")

        # User Profile (replaces semantic_facts + fresh_facts)
        # MOVED: Placed here (after bulk knowledge, before query) for high attention with low token cost
        user_profile = context.get("user_profile", "")
        if user_profile and isinstance(user_profile, str):
            # Count facts (each fact ends with [timestamp])
            fact_count = user_profile.count('[20')  # Count timestamp brackets starting with [20xx
            sections.append(f"[USER PROFILE] n={fact_count}\n{user_profile}")

        # Active Features Inventory (always present, compact)
        feature_inventory = self._build_feature_inventory(context)
        if feature_inventory:
            sections.append(f"[ACTIVE FEATURES]\n{feature_inventory}")

        # Codebase changes since last session (first message only)
        codebase_changes = context.get("codebase_changes", {})
        if codebase_changes:
            cc_lines = []
            since_label = codebase_changes.get("since_label", "last session")
            committed = codebase_changes.get("committed", [])
            uncommitted_mod = codebase_changes.get("uncommitted_modified", [])
            uncommitted_new = codebase_changes.get("uncommitted_new", [])
            if committed:
                cc_lines.append(f"Committed ({len(committed)}):")
                for c in committed:
                    cc_lines.append(f"  - {c}")
            if uncommitted_mod:
                cc_lines.append(f"Modified uncommitted ({len(uncommitted_mod)}):")
                for f in uncommitted_mod:
                    cc_lines.append(f"  - {f}")
            if uncommitted_new:
                cc_lines.append(f"New untracked ({len(uncommitted_new)}):")
                for f in uncommitted_new:
                    cc_lines.append(f"  - {f}")
            if cc_lines:
                total = len(committed) + len(uncommitted_mod) + len(uncommitted_new)
                sections.append(
                    f"[CODEBASE CHANGES SINCE LAST SESSION] n={total} (since {since_label})\n"
                    + "\n".join(cc_lines))

        # Time context
        # MOVED: Placed here (right before STM and query) for temporal grounding with high attention
        try:
            time_ctx = self.formatter._get_time_context()  # prefer formatter's version if present
        except AttributeError:
            time_ctx = f"Current time: {datetime.now().strftime('%A, %Y-%m-%d %H:%M:%S')}"
        if time_ctx:
            sections.append(f"[TIME CONTEXT]\n{time_ctx}")

        # Temporal Grounding (Narrative Context) - synthesized life state for trajectory awareness
        narrative_state = context.get("narrative_state", "")
        if narrative_state and isinstance(narrative_state, str) and narrative_state.strip():
            sections.append(f"[TEMPORAL GROUNDING]\n{narrative_state}")
            logger.debug(f"[PROMPT ASSEMBLY] Added temporal grounding section ({len(narrative_state)} chars)")

        # STM (Short-Term Memory) Summary - placed right before query for maximum attention
        stm_summary = context.get("stm_summary")
        logger.warning(f"STM RENDERING CHECK: stm_summary = {stm_summary}")
        if stm_summary:
            logger.warning("STM RENDERING: Rendering STM section before query")
            stm_lines = []
            stm_lines.append(f"Topic: {stm_summary.get('topic', 'unknown')}")
            stm_lines.append(f"User Question: {stm_summary.get('user_question', '')}")
            stm_lines.append(f"Intent: {stm_summary.get('intent', '')}")
            stm_lines.append(f"Tone: {stm_summary.get('tone', 'neutral')}")

            open_threads = stm_summary.get('open_threads', [])
            if open_threads:
                stm_lines.append(f"Open Threads: {', '.join(open_threads)}")

            constraints = stm_summary.get('constraints', [])
            if constraints:
                stm_lines.append(f"Constraints: {', '.join(constraints)}")

            sections.append(f"[SHORT-TERM CONTEXT SUMMARY]\n" + "\n".join(stm_lines))
            logger.warning(f"STM RENDERING: Added STM section before query")
        else:
            logger.warning("STM RENDERING: No stm_summary in context, skipping section")

        # User input with last Q/A pair for coherence
        if user_input:
            query_section = f"[CURRENT USER QUERY]\n"

            # Attach last Q/A pair for maximum coherence (high attention area)
            recent = context.get("recent_conversations", [])
            if recent and len(recent) > 0:
                last_exchange = recent[0]  # First item is most recent (list ordered newest-first)
                last_q = last_exchange.get("query", "")
                last_a = last_exchange.get("response", "")
                if last_q and last_a:
                    query_section += f"[LAST EXCHANGE FOR CONTEXT]\n"
                    query_section += f"User: {last_q}\n"
                    query_section += f"Assistant: {last_a}\n\n"

            query_section += f"[CURRENT QUERY]\n{user_input}"
            sections.append(query_section)

        # DEBUG: Check for duplicate section headers before returning
        section_headers = [s.split('\n')[0] for s in sections if s]
        header_counts = {}
        for header in section_headers:
            if header.startswith('['):
                header_counts[header] = header_counts.get(header, 0) + 1

        duplicates = {h: c for h, c in header_counts.items() if c > 1}
        if duplicates:
            logger.error(f"[DEBUG RECENT] DUPLICATE SECTIONS DETECTED: {duplicates}")
            logger.error(f"[DEBUG RECENT] Total sections: {len(sections)}, section headers: {section_headers}")
        else:
            logger.warning(f"[DEBUG RECENT] No duplicate sections. Total sections: {len(sections)}")

        final_prompt = "\n\n".join(sections)

        # Count how many times "[RECENT CONVERSATION]" appears in final assembled prompt
        recent_conv_count = final_prompt.count("[RECENT CONVERSATION]")
        if recent_conv_count > 1:
            logger.error(f"[DEBUG RECENT] FINAL PROMPT HAS {recent_conv_count} [RECENT CONVERSATION] HEADERS!")
            # Find positions
            import re
            matches = [(m.start(), m.end()) for m in re.finditer(r'\[RECENT CONVERSATION\]', final_prompt)]
            logger.error(f"[DEBUG RECENT] Found at positions: {matches}")
            for i, (start, end) in enumerate(matches):
                context_start = max(0, start - 50)
                context_end = min(len(final_prompt), end + 200)
                logger.error(f"[DEBUG RECENT] Match {i+1} context: ...{final_prompt[context_start:context_end]}...")

        return final_prompt


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
