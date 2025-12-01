"""
# core/prompt/builder.py

Module Contract
- Purpose: Main UnifiedPromptBuilder orchestrating complete prompt assembly process.
- Inputs:
  - build_prompt(query: str, personality: str, mode: str) -> Tuple[str, Dict]
  - gather_context_async(query: str, **kwargs) -> Dict[str, Any]
  - apply_gating(context: Dict, query: str) -> Dict[str, Any]
- Outputs:
  - Complete formatted prompt ready for LLM consumption
  - Context dictionary with all assembled data and metadata
  - Performance metrics and debug information
- Behavior:
  - Coordinates all prompt building components (gatherer, formatter, summarizer, token manager)
  - Manages async context collection with parallel data fetching
  - Applies gating system for relevance filtering and content selection
  - Enforces token budgets and applies priority-based trimming
  - Handles different prompt modes (enhanced, raw, specialized)
  - Provides comprehensive error handling and graceful fallbacks
- Dependencies:
  - .context_gatherer.ContextGatherer (data collection)
  - .formatter.PromptFormatter (text assembly)
  - .summarizer.PromptSummarizer (LLM summarization)
  - .token_manager (budget management)
  - processing.gate_system (relevance filtering)
- Side effects:
  - Memory system queries and data retrieval
  - LLM API calls for summarization
  - Cache operations for performance
  - Comprehensive logging and metrics collection
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
from .formatter import PromptFormatter, _parse_bool, _dedupe_keep_order, _truncate_list
from .summarizer import LLMSummarizer
from .token_manager import TokenManager
from .base import _FallbackMemoryCoordinator

logger = get_logger("prompt_builder")

# Configuration loading
try:
    from config.app_config import config as _APP_CFG
    _MEM_CFG = (_APP_CFG.get("memory") or {})
except Exception:
    _MEM_CFG = {}

def _cfg_int(key: str, default_val: int) -> int:
    try:
        v = _MEM_CFG.get(key, default_val)
        return int(v) if v is not None else int(default_val)
    except Exception:
        return int(default_val)

# Token and model configuration
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "4096"))
RESERVE_FOR_COMPLETION = int(os.getenv("RESERVE_FOR_COMPLETION", "1024"))
# Set to 15000 to force middle-out compression and speed up Opus processing
PROMPT_TOKEN_BUDGET = int(os.getenv("PROMPT_TOKEN_BUDGET", "15000"))

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

# Feature toggles
REFLECTIONS_ON_DEMAND = _parse_bool(os.getenv("REFLECTIONS_ON_DEMAND", "1"))
# Keep broad by default so we don't drop historical reflections
REFLECTIONS_SESSION_FILTER = _parse_bool(os.getenv("REFLECTIONS_SESSION_FILTER", "0"))
REFLECTIONS_TOPUP = _parse_bool(os.getenv("REFLECTIONS_TOPUP", "1"))

# Priority order for token budget management
PRIORITY_ORDER = [
    ("recent_conversations", 7),
    ("semantic_chunks", 6),
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
                 consolidator=None, time_manager=None, token_budget: int = PROMPT_TOKEN_BUDGET,
                 wiki_manager=None, topic_manager=None, gate_system=None, **kwargs):
        """
        Initialize the UnifiedPromptBuilder.

        Args:
            memory_coordinator: Coordinator for memory operations
            model_manager: Manager for LLM interactions
            tokenizer_manager: Manager for token counting
            consolidator: Memory consolidation manager
            time_manager: Time management utilities
            token_budget: Maximum tokens for prompt context
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

        # Token management
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
            gate_system=self.gate_system
        )

        self.formatter = PromptFormatter(
            token_manager=self.token_manager
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

    async def build_prompt(self, user_input: str, config: Optional[Dict[str, Any]] = None,
                          search_query: Optional[str] = None, personality_config: Optional[Dict[str, Any]] = None,
                          system_prompt: Optional[str] = None, current_topic: Optional[str] = None,
                          fresh_facts: Optional[List[Any]] = None, memories: Optional[List[Any]] = None,
                          stm_summary: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Build a complete prompt context for the given user input.

        This is the main entry point for prompt building. It gathers context
        from all sources, applies token budget management, and returns a
        structured context dict ready for formatting.

        Args:
            user_input: The user's query/input
            config: Optional configuration overrides

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
        """
        start_time = time.time()
        config = config or {}

        logger.info(f"Building prompt for user input: {len(user_input)} chars")

        try:
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
                return await self._build_lightweight_context(user_input, stm_summary=stm_summary)

            # Step 2: Launch parallel data gathering tasks
            tasks = {}

            # Recent conversations
            tasks["recent"] = asyncio.create_task(
                self.context_gatherer._get_recent_conversations(PROMPT_MAX_RECENT)
            )

            # Query-relevant memories (semantic search results only)
            tasks["memories"] = asyncio.create_task(
                self.context_gatherer._get_semantic_memories(user_input, PROMPT_MAX_MEMS)
            )

            # User Profile (replaces semantic_facts + fresh_facts with categorized hybrid retrieval)
            tasks["user_profile"] = asyncio.create_task(
                self.context_gatherer.get_user_profile_context(user_input, max_tokens=500)
            )

            # Summaries (separated into recent + semantic)
            tasks["summaries"] = asyncio.create_task(
                self.context_gatherer._get_summaries_separate(user_input, PROMPT_MAX_RECENT_SUMMARIES, PROMPT_MAX_SEMANTIC_SUMMARIES)
            )

            # Dreams (if enabled)
            tasks["dreams"] = asyncio.create_task(
                self.context_gatherer._get_dreams(PROMPT_MAX_DREAMS)
            )

            # Semantic chunks
            tasks["semantic"] = asyncio.create_task(
                self.context_gatherer._get_semantic_chunks(
                    user_input, max_results=PROMPT_MAX_SEMANTIC
                )
            )

            # Reflections (separated into recent + semantic)
            tasks["reflections"] = asyncio.create_task(
                self.context_gatherer._get_reflections_separate(user_input, PROMPT_MAX_RECENT_REFLECTIONS, PROMPT_MAX_SEMANTIC_REFLECTIONS)
            )

            # Wiki content
            tasks["wiki"] = asyncio.create_task(
                self.context_gatherer._get_wiki_content(user_input, PROMPT_MAX_WIKI)
            )

            # Gather all results with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks.values(), return_exceptions=True),
                    timeout=30.0
                )

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

            except asyncio.TimeoutError:
                logger.warning("Data gathering timed out, using partial results")
                gathered = {name: [] for name in tasks.keys()}

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
            except Exception:
                pass

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
            logger.warning(f"[DEBUG] recent_conversations has {len(recent_convos)} items")
            if recent_convos:
                logger.warning(f"[DEBUG] First recent: {recent_convos[0].get('query', '')[:50]}...")
                logger.warning(f"[DEBUG] Last recent: {recent_convos[-1].get('query', '')[:50]}...")

            context = {
                "recent_conversations": recent_convos,
                "memories": gathered_memories,
                "user_profile": gathered.get("user_profile", ""),  # Replaces semantic_facts + fresh_facts
                "summaries": all_summaries,
                "recent_summaries": recent_summaries,
                "semantic_summaries": semantic_summaries,
                "reflections": session_reflections,
                "recent_reflections": recent_reflections,
                "semantic_reflections": semantic_reflections,
                "dreams": gathered.get("dreams", []),
                "semantic_chunks": gathered.get("semantic", []),
                "wiki": gathered.get("wiki", [])
            }
            logger.warning(f"CONTEXT BUILT: recent_summaries={len(recent_summaries)}, semantic_summaries={len(semantic_summaries)}, recent_reflections={len(recent_reflections)}, semantic_reflections={len(semantic_reflections)}")
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
                pass
            except Exception as e:
                logger.warning(f"Gating failed: {e}")

            # Step 6: Apply hygiene and caps
            logger.warning(f"BEFORE HYGIENE_AND_CAPS: memories count = {len(context.get('memories', []))}")
            context = await self._hygiene_and_caps(context, stm_summary=stm_summary)
            logger.warning(f"AFTER HYGIENE_AND_CAPS: context has {len(context)} keys: {list(context.keys())}")
            logger.warning(f"AFTER HYGIENE_AND_CAPS: memories count = {len(context.get('memories', []))}")
            logger.warning(f"AFTER HYGIENE_AND_CAPS: recent_summaries={len(context.get('recent_summaries', []))}, semantic_summaries={len(context.get('semantic_summaries', []))}")

            # Step 6.1: Top-up relevant memories if cross-effects reduced them too much.
            try:
                mems = context.get("memories", []) or []
                recents = context.get("recent_conversations", []) or []
                if len(mems) < PROMPT_MAX_MEMS:
                    # Pull extra recent conversations beyond the ones already shown
                    extra_recent = await self.context_gatherer._get_recent_conversations(PROMPT_MAX_RECENT + PROMPT_MAX_MEMS)
                    # Build keys for already used recent items
                    def _key(x):
                        return (str(x.get("query", "")) + str(x.get("response", ""))).strip().lower()
                    used = {_key(r) for r in recents}
                    # Keep only items not already in the recent section
                    filler = []
                    for item in extra_recent:
                        if _key(item) not in used:
                            filler.append(item)
                    needed = max(0, PROMPT_MAX_MEMS - len(mems))
                    if needed:
                        mems.extend(filler[:needed])
                        context["memories"] = mems
            except Exception:
                pass

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
                    except Exception:
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
                    except Exception:
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
            except Exception:
                pass

            # Step 7: Token budget management
            logger.warning(f"BEFORE TOKEN BUDGET: memories count = {len(context.get('memories', []))}")
            context = self.token_manager._manage_token_budget(context)
            logger.warning(f"AFTER TOKEN BUDGET: memories count = {len(context.get('memories', []))}")

            # Step 7.1: Post-budget floors for summaries and reflections
            # Ensure these sections are not dropped entirely by budget trimming.
            try:
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
                    except Exception:
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
                    except Exception:
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
            except Exception:
                pass

            logger.warning(f"BEFORE FINAL ASSEMBLY: memories count = {len(context.get('memories', []))}")

            # Step 8: Final context assembly
            prompt_ctx = {
                "recent_conversations": context.get("recent_conversations", []),
                "memories": context.get("memories", []),
                "user_profile": context.get("user_profile", ""),  # Replaces semantic_facts + fresh_facts
                "summaries": context.get("summaries", []),
                "recent_summaries": context.get("recent_summaries", []),
                "semantic_summaries": context.get("semantic_summaries", []),
                "reflections": context.get("reflections", []),
                "recent_reflections": context.get("recent_reflections", []),
                "semantic_reflections": context.get("semantic_reflections", []),
                "dreams": context.get("dreams", []),
                "semantic_chunks": context.get("semantic_chunks", []),
                "wiki": context.get("wiki", []),
                "stm_summary": context.get("stm_summary")  # STM context summary (dict or None)
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
                "summaries": [],
                "reflections": [],
                "dreams": [],
                "semantic_chunks": [],
                "wiki": []
            }
            # Include stm_summary if it was provided
            if stm_summary is not None:
                error_context["stm_summary"] = stm_summary
            return error_context

    async def _build_lightweight_context(self, user_input: str, stm_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build lightweight context for small-talk queries."""
        try:
            # Just get recent conversations for small-talk
            recent = await self.context_gatherer._get_recent_conversations(3)

            context = {
                "recent_conversations": recent,
                "memories": [],
                "user_profile": "",
                "summaries": [],
                "recent_summaries": [],
                "semantic_summaries": [],
                "reflections": [],
                "recent_reflections": [],
                "semantic_reflections": [],
                "dreams": [],
                "semantic_chunks": [],
                "wiki": []
            }

            # Add STM summary if provided
            if stm_summary is not None:
                context["stm_summary"] = stm_summary

            return context
        except Exception as e:
            logger.warning(f"Lightweight context building failed: {e}")
            return {
                "recent_conversations": [],
                "memories": [],
                "user_profile": "",
                "summaries": [],
                "recent_summaries": [],
                "semantic_summaries": [],
                "reflections": [],
                "recent_reflections": [],
                "semantic_reflections": [],
                "dreams": [],
                "semantic_chunks": [],
                "wiki": []
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

        # For conversations, we need semantic similarity because responses have minor variations
        # ("abundantly clear" vs "painfully clear", "storm" vs "tempest", etc.)
        seen_embeddings = []  # List of (embedding, original_item) tuples
        seen_content = set()  # Fallback for string-based dedup
        SIMILARITY_THRESHOLD = 0.90  # 90% similar = duplicate (lowered from 0.95 to catch more variations)

        # Get embedder for semantic dedup
        embedder = None
        try:
            embedder = self.model_manager.get_embedder() if hasattr(self.model_manager, "get_embedder") else None
            if embedder:
                logger.info("[DEDUP] Using semantic similarity for conversation deduplication")
        except Exception:
            logger.warning("[DEDUP] No embedder available, falling back to string-based dedup")

        cross_dedup_sections = [
            "recent_conversations", "memories"
        ]

        # Track target counts for backfilling
        target_counts = {
            "recent_conversations": 10,  # Target number of unique recent conversations
            "memories": 30  # Target number of unique memories
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
                        # Fallback to query/response (for conversations)
                        response = item.get("response", "")
                        content = response if response else str(item.get("query", ""))
                else:
                    content = str(item)

                # Normalize content for comparison
                normalized = content.strip().lower()
                for prefix in ["user:", "daemon:", "luke,"]:
                    if normalized.startswith(prefix):
                        normalized = normalized[len(prefix):].strip()

                is_duplicate = False

                # Use semantic similarity if embedder available
                if embedder:
                    try:
                        # Embed the content (use first 512 chars for speed)
                        item_embedding = embedder.encode(normalized[:512], convert_to_numpy=True)

                        # Check against all seen embeddings
                        for seen_emb, _ in seen_embeddings:
                            # Compute cosine similarity
                            import numpy as np
                            similarity = np.dot(item_embedding, seen_emb) / (
                                np.linalg.norm(item_embedding) * np.linalg.norm(seen_emb) + 1e-8
                            )

                            if similarity >= SIMILARITY_THRESHOLD:
                                is_duplicate = True
                                logger.debug(f"CROSS-SECTION DEDUP: Skipped semantic duplicate in {section} (similarity={similarity:.3f})")
                                break

                        if not is_duplicate:
                            seen_embeddings.append((item_embedding, item))
                            deduplicated.append(item)

                    except Exception as e:
                        logger.debug(f"[DEDUP] Embedding failed, using string fallback: {e}")
                        # Fallback to string-based dedup
                        dedup_key = normalized[:500]
                        if dedup_key and dedup_key not in seen_content:
                            seen_content.add(dedup_key)
                            deduplicated.append(item)
                        else:
                            is_duplicate = True

                else:
                    # String-based fallback
                    dedup_key = normalized[:500]
                    if dedup_key and dedup_key not in seen_content:
                        seen_content.add(dedup_key)
                        deduplicated.append(item)
                    else:
                        is_duplicate = True
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
                    seen_embeddings=seen_embeddings,
                    seen_content=seen_content,
                    target_count=target_count,
                    offset=original_count,
                    embedder=embedder,
                    similarity_threshold=SIMILARITY_THRESHOLD
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

    def _assemble_prompt(self, context: Dict[str, Any] = None, user_input: str = "",
                        directives: str = "", system_prompt: str = "", **kwargs) -> str:
        """Assemble final prompt string from context with numbering and timestamp-first entries."""
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

                # Format timestamp
                if isinstance(ts, datetime):
                    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
                elif ts:
                    ts_str = str(ts)
                else:
                    ts_str = ""

                # Add tags
                if tags_str and content:
                    content += f"\nTags: {{{tags_str}}}"

                return content, ts_str
            except Exception:
                return str(mem), ""

        sections: list[str] = []

        # Time context
        try:
            time_ctx = self.formatter._get_time_context()  # prefer formatter's version if present
        except Exception:
            time_ctx = f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        if time_ctx:
            sections.append(f"[TIME CONTEXT]\n{time_ctx}")

        # Recent conversations
        recent = context.get("recent_conversations", []) or []
        recent_lines: list[str] = []
        for i, mem in enumerate(recent, start=1):
            content, ts = mem_parts(mem)
            recent_lines.append(f"{i}) {ts}: {content}" if ts else f"{i}) {content}")
        if recent_lines:
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

        # User Profile (replaces semantic_facts + fresh_facts)
        user_profile = context.get("user_profile", "")
        if user_profile and isinstance(user_profile, str):
            sections.append(f"[USER PROFILE]\n{user_profile}")

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
                recent_sum_lines.append(f"{i}) {ts}: {content}" if ts else f"{i}) {content}")
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
                semantic_sum_lines.append(f"{i}) {ts}: {content}" if ts else f"{i}) {content}")
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
                recent_refl_lines.append(f"{i}) {ts}: {content}" if ts else f"{i}) {content}")
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
                semantic_refl_lines.append(f"{i}) {ts}: {content}" if ts else f"{i}) {content}")
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

        return "\n\n".join(sections)


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
                "wiki": [{"content": wiki_snippet}] if wiki_snippet else []
            }
            return self.unified_builder._assemble_prompt(context, user_input)
        else:
            # New interface - delegate to UnifiedPromptBuilder
            context = await self.unified_builder.build_prompt(user_input, config)
            return self.unified_builder._assemble_prompt(context, user_input)
