"""
Agentic Search Controller Module

Contract:
    - Provides AgenticSearchController for multi-round search loops
    - Manages ReAct cycle: Think → Multi-Act (parallel dispatch) → Observe → Repeat
    - Multi-action dispatch: LLM may request multiple independent tools per step;
      dispatched concurrently via asyncio.gather(), results accumulated in order
    - Emits ProgressEvent for UI updates
    - Enforces max_rounds limit (default 5, each tool call counts as one round)
    - Budget-enforced accumulated_context: _append_accumulated() trims oldest rounds
      when accumulated context exceeds context_budget_tokens (default 8000)
    - Budget-aware final prompt: _build_final_prompt() trims low-value sections
      (dreams, reflections, docs, summaries) if total exceeds ceiling
    - Falls back gracefully on search/API failures (partial failure: gather returns_exceptions=True)
    - Provenance: computes final_prompt_hash (SHA-256[:16]) on assembled prompt

Modular Architecture (2026-05-09):
    - AgenticFormatter (core/agentic/formatters.py): Pure stateless formatting methods
      for all result types (search, memory, file, wiki, etc.)
    - ToolExecutor (core/agentic/tools.py): Dispatch routing + low-level tool execution
      for all 10 tool types (web search, wolfram, sandbox, memory, files, git stats, etc.)
    - Controller retains: orchestration loop, prompt building, model interaction,
      quality heuristics, and delegation wrappers for backward compatibility

Dependencies:
    - core.agentic.formatters.AgenticFormatter (result formatting)
    - core.agentic.tools.ToolExecutor (tool dispatch + execution)
    - models.model_manager.ModelManager (for LLM generation)
    - knowledge.web_search_manager.WebSearchManager (for web searches)
    - knowledge.wolfram_manager.WolframManager (for computations, optional)
    - knowledge.sandbox_manager.SandboxManager (for code execution, optional)
    - memory.memory_expander.MemoryExpander (for memory expansion, optional)
    - core.prompt.token_manager.TokenManager (for budget enforcement)

Public Interface:
    - AgenticSearchController.run_agentic_search(skip_initial_search=False) -> AsyncGenerator[ProgressEvent|str]
    - AgenticSearchController.detect_protocol() -> SearchProtocol
"""

import asyncio
import hashlib
import logging
import re
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from core.agentic.types import (
    AgentState,
    AgenticSearchSession,
    ProgressEvent,
    SearchDecision,
    SearchProtocol,
    SearchRequest,
    SearchRound,
    _ToolResult,
    LOW_QUALITY_HINT_TEMPLATE,
    MAX_RELAXATION_HINT,
)
from core.agentic.protocols import (
    detect_protocol,
    get_protocol_handler,
    BaseProtocolHandler,
)
from core.agentic.formatters import AgenticFormatter
from core.agentic.tools import ToolExecutor

if TYPE_CHECKING:
    from models.model_manager import ModelManager
    from knowledge.web_search_manager import WebSearchManager, WebSearchResult
    from knowledge.wolfram_manager import WolframManager
    from knowledge.sandbox_manager import SandboxManager, PersistentSession, SandboxResult
    from core.prompt.token_manager import TokenManager
    from core.file_access_manager import FileAccessManager
    from core.git_stats_manager import GitStatsManager

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MAX_ROUNDS = 5
DEFAULT_CONTEXT_BUDGET_TOKENS = 8000
DEFAULT_COMPRESSION_MAX_TOKENS = 1500
DEFAULT_COMPRESSION_MODEL = "gpt-4o-mini"

# Pre-compiled patterns for query relaxation (avoid re-compiling per call)
_VERSION_PATTERN = re.compile(r'v?\d+(\.\d+)+')
_YEAR_PATTERN = re.compile(r'\b20\d{2}\b')
_ERROR_PATTERN = re.compile(r'error|exception|traceback|bug|issue', re.IGNORECASE)

# Stop words for relevance check
_STOP_WORDS = frozenset({'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'for', 'in', 'on', 'with', 'and', 'or'})


class AgenticSearchController:
    """
    Controls the ReAct-style agentic search loop.

    This controller manages multi-round search sessions where the LLM can
    iteratively gather information until it has enough to provide a
    comprehensive answer.

    The first search is automatic (triggered by the existing LLM-first trigger).
    Subsequent searches are model-driven via tool calls or XML markers.
    """

    VALID_MEMORY_COLLECTIONS = frozenset({
        "reference_docs", "facts", "conversations", "summaries",
        "reflections", "obsidian_notes", "procedural", "procedural_skills",
        "wiki_knowledge",
    })

    def __init__(
        self,
        model_manager: "ModelManager",
        web_search_manager: "WebSearchManager",
        chroma_store=None,
        wolfram_manager: Optional["WolframManager"] = None,
        sandbox_manager: Optional["SandboxManager"] = None,
        file_access_manager: Optional["FileAccessManager"] = None,
        git_stats_manager: Optional["GitStatsManager"] = None,
        token_manager: Optional["TokenManager"] = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        context_budget_tokens: int = DEFAULT_CONTEXT_BUDGET_TOKENS,
        compression_model: str = DEFAULT_COMPRESSION_MODEL,
    ):
        """
        Initialize the agentic search controller.

        Args:
            model_manager: LLM manager for generation
            web_search_manager: Web search manager for queries
            chroma_store: Optional ChromaDB store for memory search
            wolfram_manager: Optional Wolfram Alpha manager for computations
            sandbox_manager: Optional E2B sandbox manager for code execution
            file_access_manager: Optional file access manager for read/grep/list
            git_stats_manager: Optional git stats manager for repo activity queries
            token_manager: Optional token counter for budget enforcement
            max_rounds: Maximum search rounds allowed (default 5)
            context_budget_tokens: Token budget for accumulated context
            compression_model: Model to use for result compression
        """
        self.model_manager = model_manager
        self.web_search_manager = web_search_manager
        self.chroma_store = chroma_store
        self.wolfram_manager = wolfram_manager
        self.sandbox_manager = sandbox_manager
        self.file_access_manager = file_access_manager
        self.git_stats_manager = git_stats_manager
        self.token_manager = token_manager
        self.max_rounds = max_rounds
        self.context_budget_tokens = context_budget_tokens
        self.compression_model = compression_model

        # Memory expander (temporal window around a doc)
        self.memory_expander = None
        if chroma_store:
            try:
                from memory.memory_expander import MemoryExpander
                self.memory_expander = MemoryExpander(chroma_store)
            except Exception as e:
                logger.warning(f"[AgenticSearch] Could not init MemoryExpander: {e}")

        # Modular components (extracted from this class)
        self._formatter = AgenticFormatter()
        self._tool_executor = ToolExecutor(
            model_manager=model_manager,
            web_search_manager=web_search_manager,
            formatter=self._formatter,
            chroma_store=chroma_store,
            wolfram_manager=wolfram_manager,
            sandbox_manager=sandbox_manager,
            file_access_manager=file_access_manager,
            git_stats_manager=git_stats_manager,
            token_manager=token_manager,
            memory_expander=self.memory_expander,
            compression_model=compression_model,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text, using tokenizer if available."""
        if self.token_manager and hasattr(self.token_manager, 'get_token_count'):
            try:
                model_name = self.model_manager.get_active_model_name() if hasattr(self.model_manager, "get_active_model_name") else "default"
                return self.token_manager.get_token_count(text or "", model_name)
            except Exception:
                pass
        # Fallback: ~4 chars per token
        return len(text or "") // 4

    def _append_accumulated(self, session: "AgenticSearchSession", new_context: str) -> None:
        """Append to accumulated_context with budget enforcement.

        If adding new_context would exceed context_budget_tokens, trim
        the oldest accumulated content (from the front) to make room.
        """
        candidate = session.accumulated_context + "\n\n" + new_context if session.accumulated_context else new_context
        total_tokens = self._estimate_tokens(candidate)

        if total_tokens <= self.context_budget_tokens:
            session.accumulated_context = candidate
            return

        # Over budget — trim from the front (oldest rounds) to make room
        # Split into round blocks and drop from the front until under budget
        blocks = candidate.split("\n\n---\n")
        while len(blocks) > 1 and self._estimate_tokens("\n\n---\n".join(blocks)) > self.context_budget_tokens:
            blocks.pop(0)

        session.accumulated_context = "\n\n---\n".join(blocks)
        logger.info(
            f"[AgenticSearch] Trimmed accumulated_context to fit budget: "
            f"{total_tokens} -> {self._estimate_tokens(session.accumulated_context)} tokens "
            f"(budget={self.context_budget_tokens})"
        )

    def detect_protocol(self, model_name: str) -> SearchProtocol:
        """
        Determine which protocol to use based on model capabilities.

        Args:
            model_name: The model name or alias

        Returns:
            SearchProtocol indicating native tools or XML markers
        """
        return detect_protocol(model_name, self.model_manager.api_models)

    async def run_agentic_search(
        self,
        query: str,
        system_prompt: str,
        model_name: str,
        initial_search_terms: List[str],
        initial_context: Optional[Dict[str, Any]] = None,
        crisis_level: Optional[str] = None,
        skip_initial_search: bool = False,
        initial_urls: Optional[List[str]] = None,
    ) -> AsyncGenerator[Union[ProgressEvent, str], None]:
        """
        Execute the agentic search loop.

        Yields progress events during search phases and response chunks
        during final answer generation.

        Args:
            query: The user's original query
            system_prompt: Base system prompt
            model_name: Model to use for generation
            initial_search_terms: Search terms from LLM-first trigger
            initial_context: Optional pre-gathered context
            crisis_level: Current crisis/tone level
            skip_initial_search: If True, skip Round 1 web search (for computation-only queries)
            initial_urls: Optional list of URLs extracted from the user message to fetch directly

        Yields:
            ProgressEvent: Status updates for UI
            str: Final streamed response chunks
        """
        # Initialize session
        protocol = self.detect_protocol(model_name)
        session = AgenticSearchSession(
            query=query,
            max_rounds=self.max_rounds,
            protocol=protocol,
        )

        logger.info(
            f"[AgenticSearch] Starting session: query='{query[:50]}...', "
            f"protocol={protocol.value}, max_rounds={self.max_rounds}"
        )

        # Get protocol handler (pass tool availability for tool definitions)
        wolfram_available = self.wolfram_manager is not None and self.wolfram_manager.is_available()
        sandbox_available = self.sandbox_manager is not None and self.sandbox_manager.is_available()
        memory_available = self.chroma_store is not None
        file_access_available = self.file_access_manager is not None and self.file_access_manager.is_available()
        git_stats_available = self.git_stats_manager is not None and self.git_stats_manager.is_available()
        fetch_url_available = self.web_search_manager is not None and self.web_search_manager.is_available()
        handler = get_protocol_handler(
            protocol,
            wolfram_available=wolfram_available,
            sandbox_available=sandbox_available,
            memory_available=memory_available,
            file_access_available=file_access_available,
            git_stats_available=git_stats_available,
            fetch_url_available=fetch_url_available,
        )

        # Augment system prompt for agentic mode
        augmented_system_prompt = handler.augment_system_prompt(
            system_prompt, self.max_rounds
        )

        # Inject tool health summary so the LLM never confabulates about
        # its own capabilities (e.g. claiming FAISS works when drive is
        # disconnected).
        tool_health = self._tool_executor.get_tool_health()
        augmented_system_prompt += (
            f"\n\n[TOOL STATUS — DO NOT LIE ABOUT THESE]\n{tool_health}\n"
            "If a tool is UNAVAILABLE, you MUST tell the user it is unavailable "
            "when asked. Never claim a tool is working if its status says otherwise."
        )

        # Create persistent sandbox session if available (for variable persistence across turns)
        sandbox_session = None
        if sandbox_available:
            try:
                sandbox_session = await self.sandbox_manager.create_session()
                logger.info("[AgenticSearch] Created persistent sandbox session for ReAct loop")
            except Exception as e:
                logger.warning(f"[AgenticSearch] Failed to create sandbox session: {e}")
                # Continue without sandbox - will fall back gracefully

        try:
            # === ROUND 1: URL fetch or automatic search with trigger terms ===
            if initial_urls:
                # User message contains URLs — fetch them directly instead of searching
                session.state = AgentState.SEARCHING
                logger.info(f"[AgenticSearch] Round 1: fetching {len(initial_urls)} URL(s) from user message")

                for i, url in enumerate(initial_urls[:3]):  # Cap at 3 URLs
                    yield ProgressEvent(
                        event_type="fetching_url",
                        message=f"Fetching: {url}",
                        round_number=1,
                        metadata={"url": url}
                    )

                start_time = time.time()
                fetch_tasks = [
                    self._tool_executor._execute_fetch_url(url)
                    for url in initial_urls[:3]
                ]
                fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                fetch_duration = (time.time() - start_time) * 1000

                # Build accumulated context from fetched pages
                fetch_context_parts = []
                for url, result in zip(initial_urls[:3], fetch_results):
                    if isinstance(result, Exception):
                        content = f"[Error fetching {url}: {result}]"
                    else:
                        content = result
                    fetch_context_parts.append(
                        self._formatter.format_fetch_url_context(1, url, content)
                    )

                first_round = SearchRound(
                    round_number=1,
                    request=SearchRequest(
                        query=f"[Fetch URL] {initial_urls[0]}",
                        round_number=1
                    ),
                    results=None,
                    duration_ms=fetch_duration
                )
                first_round.summary = "\n\n".join(
                    r if not isinstance(r, Exception) else f"[Error: {r}]"
                    for r in fetch_results
                )
                session.rounds.append(first_round)
                session.accumulated_context = "\n\n".join(fetch_context_parts)

                yield ProgressEvent(
                    event_type="url_fetched",
                    message=f"Fetched {len(initial_urls[:3])} URL(s)",
                    round_number=1,
                    metadata={"duration_ms": fetch_duration}
                )

            elif skip_initial_search:
                # Skip Round 1 web search for computation-only queries
                logger.info("[AgenticSearch] Skipping initial search (computation-only mode)")
                session.accumulated_context = ""
                yield ProgressEvent(
                    event_type="thinking",
                    message="Entering Agentic Loop...",
                    round_number=1,
                    metadata={"skip_search": True}
                )
            else:
                session.state = AgentState.SEARCHING

                # Fallback to query if no search terms provided
                if not initial_search_terms:
                    initial_search_terms = [query]

                yield ProgressEvent(
                    event_type="searching",
                    message=f"Searching for: {initial_search_terms[0]}",
                    round_number=1,
                    metadata={"terms": initial_search_terms}
                )

                # Execute first search
                start_time = time.time()
                first_result = await self._execute_search(
                    initial_search_terms,
                    crisis_level=crisis_level
                )
                search_duration = (time.time() - start_time) * 1000

                # Record first round
                first_round = SearchRound(
                    round_number=1,
                    request=SearchRequest(
                        query=initial_search_terms[0],
                        round_number=1
                    ),
                    results=first_result,
                    duration_ms=search_duration
                )

                # Emit results found
                result_count = len(first_result.pages) if first_result and hasattr(first_result, 'pages') else 0
                yield ProgressEvent(
                    event_type="found_results",
                    message=f"Found {result_count} results",
                    round_number=1,
                    metadata={"result_count": result_count, "duration_ms": search_duration}
                )

                # Compress and accumulate context
                session.state = AgentState.OBSERVING
                compressed = await self._compress_results(first_result)
                first_round.summary = compressed
                session.rounds.append(first_round)
                session.accumulated_context = self._format_search_context(
                    1, initial_search_terms[0], compressed
                )

                # Check Round 1 result quality and set hint for next iteration
                is_low_quality, issue = self._is_low_quality_result(
                    first_result, initial_search_terms[0]
                )
                if is_low_quality:
                    session.low_quality_search_count += 1
                    suggestion = self._generate_relaxation_suggestion(initial_search_terms[0])
                    remaining = 2 - session.low_quality_search_count
                    session.relaxation_hint = LOW_QUALITY_HINT_TEMPLATE.format(
                        query=initial_search_terms[0],
                        issue=issue,
                        suggestion=suggestion,
                        remaining=remaining
                    )
                    logger.info(
                        f"[AgenticSearch] Round 1 low quality ({issue}), "
                        f"relaxation count: {session.low_quality_search_count}"
                    )

            # Compute context inventory once for the session
            session.context_inventory = self._compute_context_inventory(initial_context)
            if session.context_inventory:
                logger.debug(
                    f"[AgenticSearch] Context inventory computed: "
                    f"{session.context_inventory.count(chr(10))} sections"
                )

            # === ROUNDS 2-N: Model-driven iteration ===
            while session.can_continue and session.current_round <= self.max_rounds:
                session.state = AgentState.THINKING

                # Build prompt with accumulated context
                iteration_prompt = self._build_iteration_prompt(
                    query=query,
                    search_context=session.accumulated_context,
                    round_number=session.current_round,
                    session=session
                )

                # Generate with protocol-appropriate method
                decisions = await self._get_model_decision(
                    prompt=iteration_prompt,
                    system_prompt=augmented_system_prompt,
                    model_name=model_name,
                    handler=handler,
                    session=session
                )

                # Check for done signal — honor it immediately
                if any(d.is_done for d in decisions):
                    done_d = next((d for d in decisions if d.is_done), None)
                    session.model_signaled_done = True
                    session.done_reason = done_d.done_reason if done_d else None
                    logger.info(f"[AgenticSearch] Model signaled done: {session.done_reason}")
                    break

                # Filter to actual tool requests
                tool_decisions = [
                    d for d in decisions
                    if not d.is_done and not d.wants_answer
                ]
                if not tool_decisions:
                    logger.info("[AgenticSearch] Model ready to answer (implicit)")
                    break

                # Clamp to remaining round budget
                rounds_remaining = self.max_rounds - len(session.rounds)
                if rounds_remaining <= 0:
                    break
                if len(tool_decisions) > rounds_remaining:
                    tool_decisions = tool_decisions[:rounds_remaining]
                    logger.info(
                        f"[AgenticSearch] Clamped to {rounds_remaining} tools (max_rounds)"
                    )

                # Pre-filter expand_memory requests against session limit
                from config.app_config import EXPAND_MEMORY_ENABLED, EXPAND_MAX_PER_SESSION
                expand_budget = EXPAND_MAX_PER_SESSION - session.expand_count
                filtered_decisions = []
                for d in tool_decisions:
                    if d.wants_memory_expand and d.expand_memory_id:
                        if not EXPAND_MEMORY_ENABLED or not self.memory_expander:
                            logger.info("[AgenticSearch] expand_memory disabled, skipping")
                            continue
                        if expand_budget <= 0:
                            logger.info("[AgenticSearch] expand_memory limit reached, skipping")
                            continue
                        expand_budget -= 1
                    filtered_decisions.append(d)
                tool_decisions = filtered_decisions

                if not tool_decisions:
                    logger.info("[AgenticSearch] No dispatchable tools after filtering")
                    break

                # Assign round numbers and dispatch concurrently
                base_round = session.current_round
                session.state = AgentState.SEARCHING

                if len(tool_decisions) > 1:
                    logger.info(
                        f"[AgenticSearch] Parallel dispatch: {len(tool_decisions)} tools"
                    )

                tasks = [
                    self._dispatch_single(
                        d, base_round + i, session, crisis_level, sandbox_session
                    )
                    for i, d in enumerate(tool_decisions)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Yield events and accumulate results (deterministic order)
                session.state = AgentState.OBSERVING
                for tr in results:
                    if isinstance(tr, Exception):
                        logger.error(f"[AgenticSearch] Tool dispatch error: {tr}")
                        continue
                    for ev in tr.start_events:
                        yield ev
                    for ev in tr.end_events:
                        yield ev
                    if tr.round_data is not None:
                        session.rounds.append(tr.round_data)
                    if tr.formatted_context:
                        self._append_accumulated(session, tr.formatted_context)
                    if tr.memory_collection:
                        session.memory_search_counts[tr.memory_collection] = (
                            session.memory_search_counts.get(tr.memory_collection, 0) + 1
                        )
                    if tr.is_expand and tr.round_data is not None:
                        session.expand_count += 1

                # Relaxation tracking (web search results only)
                for tr in results:
                    if isinstance(tr, Exception):
                        continue
                    if tr.decision.wants_search and tr.round_data is not None:
                        self._update_relaxation_tracking(session, tr)

            # === FINAL GENERATION ===
            session.state = AgentState.GENERATING
            yield ProgressEvent(
                event_type="synthesizing",
                message="Generating comprehensive answer...",
                round_number=len(session.rounds),
                metadata={"total_rounds": len(session.rounds)}
            )

            # Generate final response
            async for chunk in self._generate_final_response(
                query=query,
                system_prompt=system_prompt,  # Use original system prompt for final
                model_name=model_name,
                session=session,
                initial_context=initial_context
            ):
                yield chunk

            session.state = AgentState.DONE
            session.end_time = datetime.now()
            self._last_session = session

            yield ProgressEvent(
                event_type="done",
                message="Search complete",
                round_number=len(session.rounds),
                metadata={
                    "total_rounds": len(session.rounds),
                    "total_duration_ms": session.total_duration_ms,
                    "search_duration_ms": session.total_search_duration_ms
                }
            )

        except Exception as e:
            session.state = AgentState.ERROR
            logger.error(f"[AgenticSearch] Error in agentic loop: {e}", exc_info=True)

            yield ProgressEvent(
                event_type="error",
                message=f"Search error: {str(e)}",
                round_number=session.current_round,
                metadata={"error": str(e)}
            )

            # Fallback: try to generate answer with whatever context we have
            if session.accumulated_context:
                yield ProgressEvent(
                    event_type="synthesizing",
                    message="Generating answer with available information...",
                    round_number=len(session.rounds)
                )

                async for chunk in self._generate_final_response(
                    query=query,
                    system_prompt=system_prompt,
                    model_name=model_name,
                    session=session,
                    initial_context=initial_context
                ):
                    yield chunk

        finally:
            # Always clean up the sandbox session
            if sandbox_session and not sandbox_session.is_closed:
                try:
                    await sandbox_session.close()
                    logger.info("[AgenticSearch] Closed sandbox session")
                except Exception as e:
                    logger.warning(f"[AgenticSearch] Error closing sandbox session: {e}")

    # ------------------------------------------------------------------
    # Parallel dispatch infrastructure
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Delegation wrappers (methods moved to ToolExecutor/AgenticFormatter)
    # Preserved for backward compatibility with tests that mock these.
    # ------------------------------------------------------------------

    async def _dispatch_single(self, decision, round_number, session, crisis_level, sandbox_session):
        """Route a single SearchDecision to the appropriate dispatch method.

        Uses self._dispatch_* methods (not tool_executor directly) so that
        tests can mock individual dispatch/execute methods on the controller.
        """
        if decision.wants_search and decision.search_query:
            return await self._dispatch_web_search(decision, round_number, crisis_level)
        elif decision.wants_wolfram and decision.wolfram_query:
            return await self._dispatch_wolfram(decision, round_number)
        elif decision.wants_sandbox and decision.sandbox_code:
            return await self._dispatch_sandbox(decision, round_number, sandbox_session)
        elif decision.wants_memory_search and decision.memory_query:
            return await self._dispatch_memory_search(decision, round_number)
        elif decision.wants_memory_expand and decision.expand_memory_id:
            return await self._dispatch_memory_expand(decision, round_number)
        elif decision.wants_file_read and decision.file_read_path:
            return await self._dispatch_file_read(decision, round_number)
        elif decision.wants_file_grep and decision.file_grep_pattern:
            return await self._dispatch_file_grep(decision, round_number)
        elif decision.wants_file_list and decision.file_list_path:
            return await self._dispatch_file_list(decision, round_number)
        elif decision.wants_full_document and decision.full_document_title:
            return await self._dispatch_full_document(decision, round_number)
        elif decision.wants_git_stats and decision.git_stats_query:
            return await self._dispatch_git_stats(decision, round_number)
        elif decision.wants_recall_image and decision.recall_image_query:
            return await self._tool_executor._dispatch_recall_image(decision, round_number)
        elif decision.wants_fetch_url and decision.fetch_url:
            return await self._tool_executor._dispatch_fetch_url(decision, round_number)
        else:
            return _ToolResult(
                decision=decision, round_data=None,
                formatted_context="", start_events=[], end_events=[],
            )

    async def _dispatch_web_search(self, decision, round_number, crisis_level=None):
        """Dispatch web search. Calls self._execute_search/_format_* for mock compatibility."""
        start_events = [ProgressEvent(event_type="searching", message=f"Searching for: {decision.search_query}",
                                       round_number=round_number, metadata={"query": decision.search_query, "reason": decision.search_reason})]
        start_time = time.time()
        result = await self._execute_search([decision.search_query], crisis_level=crisis_level)
        duration = (time.time() - start_time) * 1000
        round_data = SearchRound(round_number=round_number, request=SearchRequest(query=decision.search_query, reason=decision.search_reason, round_number=round_number), results=result, duration_ms=duration)
        result_count = len(result.pages) if result and hasattr(result, 'pages') else 0
        compressed = await self._compress_results(result)
        round_data.summary = compressed
        end_events = [ProgressEvent(event_type="found_results", message=f"Found {result_count} results", round_number=round_number, metadata={"result_count": result_count})]
        return _ToolResult(decision=decision, round_data=round_data, formatted_context=self._format_search_context(round_number, decision.search_query, compressed), start_events=start_events, end_events=end_events)

    async def _dispatch_wolfram(self, decision, round_number):
        return await self._tool_executor._dispatch_wolfram(decision, round_number)

    async def _dispatch_sandbox(self, decision, round_number, sandbox_session=None):
        return await self._tool_executor._dispatch_sandbox(decision, round_number, sandbox_session)

    async def _dispatch_memory_search(self, decision, round_number):
        """Dispatch memory search. Calls self._execute_memory_search/_format_* for mock compat."""
        collection = decision.memory_collection or "facts"
        start_events = [ProgressEvent(event_type="searching_memory", message=f"Searching {collection}: {decision.memory_query}",
                                       round_number=round_number, metadata={"query": decision.memory_query, "collection": collection, "reason": decision.memory_reason})]
        start_time = time.time()
        memory_result = await self._execute_memory_search(decision.memory_query, collection)
        duration = (time.time() - start_time) * 1000
        round_data = SearchRound(round_number=round_number, request=SearchRequest(query=f"[Memory: {collection}] {decision.memory_query}", reason=decision.memory_reason, round_number=round_number), results=None, duration_ms=duration)
        round_data.summary = memory_result
        end_events = [ProgressEvent(event_type="found_results", message=f"Found memory results from {collection}", round_number=round_number, metadata={"collection": collection, "duration_ms": duration})]
        return _ToolResult(decision=decision, round_data=round_data, formatted_context=self._format_memory_context(round_number, collection, decision.memory_query, memory_result), start_events=start_events, end_events=end_events, memory_collection=collection)

    async def _dispatch_memory_expand(self, decision, round_number):
        return await self._tool_executor._dispatch_memory_expand(decision, round_number)

    async def _dispatch_file_read(self, decision, round_number):
        return await self._tool_executor._dispatch_file_read(decision, round_number)

    async def _dispatch_file_grep(self, decision, round_number):
        return await self._tool_executor._dispatch_file_grep(decision, round_number)

    async def _dispatch_file_list(self, decision, round_number):
        return await self._tool_executor._dispatch_file_list(decision, round_number)

    async def _dispatch_full_document(self, decision, round_number):
        return await self._tool_executor._dispatch_full_document(decision, round_number)

    async def _dispatch_git_stats(self, decision, round_number):
        return await self._tool_executor._dispatch_git_stats(decision, round_number)

    def _update_relaxation_tracking(
        self, session: AgenticSearchSession, tr: _ToolResult
    ) -> None:
        """Update relaxation hints after a web search result."""
        search_result = tr.round_data.results
        query = tr.decision.search_query
        is_low_quality, issue = self._is_low_quality_result(search_result, query)
        if is_low_quality:
            session.low_quality_search_count += 1
            if session.low_quality_search_count > 2:
                session.relaxation_hint = MAX_RELAXATION_HINT
                logger.info(
                    "[AgenticSearch] Max relaxation attempts reached, forcing synthesis"
                )
            else:
                suggestion = self._generate_relaxation_suggestion(query)
                remaining = 2 - session.low_quality_search_count
                session.relaxation_hint = LOW_QUALITY_HINT_TEMPLATE.format(
                    query=query, issue=issue,
                    suggestion=suggestion, remaining=remaining
                )
                logger.info(
                    f"[AgenticSearch] Low quality result ({issue}), "
                    f"relaxation count: {session.low_quality_search_count}"
                )
        else:
            session.low_quality_search_count = 0
            session.relaxation_hint = None
            logger.debug("[AgenticSearch] Good search results, reset relaxation counter")

    # ------------------------------------------------------------------
    # Execution/compression delegation wrappers (moved to ToolExecutor)
    # ------------------------------------------------------------------

    async def _execute_search(self, search_terms, crisis_level=None):
        return await self._tool_executor._execute_search(search_terms, crisis_level)

    async def _compress_results(self, result, max_tokens=DEFAULT_COMPRESSION_MAX_TOKENS):
        return await self._tool_executor._compress_results(result, max_tokens)

    async def _get_model_decision(
        self,
        prompt: str,
        system_prompt: str,
        model_name: str,
        handler: BaseProtocolHandler,
        session: AgenticSearchSession
    ) -> List[SearchDecision]:
        """
        Get the model's decision(s) on what to do next.

        Returns a list of SearchDecision objects. When the model requests
        multiple independent tools in one step, each gets its own entry.

        Args:
            prompt: The prompt to send
            system_prompt: System prompt with agentic instructions
            model_name: Model to use
            handler: Protocol handler for parsing
            session: Current session state

        Returns:
            List of SearchDecision(s) indicating model's choice(s)
        """
        try:
            if session.protocol == SearchProtocol.NATIVE_TOOLS:
                # Use tool calling
                response = await self._generate_with_tools(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model_name=model_name,
                    tools=handler.get_tools()
                )
            else:
                # Use standard generation for XML markers
                response = await self.model_manager.generate_once(
                    prompt=prompt,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    max_tokens=500,  # Limit for decision phase
                    temperature=0.3
                )

            return handler.parse_response(response)

        except Exception as e:
            logger.error(f"[AgenticSearch] Decision generation failed: {e}")
            # On error, signal to answer with current context
            return [SearchDecision(wants_answer=True)]

    async def _generate_with_tools(
        self,
        prompt: str,
        system_prompt: str,
        model_name: str,
        tools: List[Dict]
    ) -> Any:
        """
        Generate with tool calling support.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            model_name: Model to use
            tools: Tool definitions

        Returns:
            Raw response with potential tool calls
        """
        # Check if model_manager has tool support
        if hasattr(self.model_manager, 'generate_once_with_tools'):
            return await self.model_manager.generate_once_with_tools(
                prompt=prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                tools=tools,
                tool_choice="auto"
            )
        else:
            # Fallback to standard generation
            logger.warning("[AgenticSearch] Tool calling not available, using standard generation")
            response = await self.model_manager.generate_once(
                prompt=prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.3
            )
            return response

    async def _generate_final_response(
        self,
        query: str,
        system_prompt: str,
        model_name: str,
        session: AgenticSearchSession,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate the final response using accumulated search context.

        Args:
            query: Original user query
            system_prompt: System prompt
            model_name: Model to use
            session: Session with accumulated context
            initial_context: Additional context (memories, etc.)

        Yields:
            Response text chunks
        """
        # Build final prompt with all context
        final_prompt = self._build_final_prompt(
            query=query,
            session=session,
            initial_context=initial_context
        )

        # Hash prompt for provenance
        session.final_prompt_hash = hashlib.sha256(final_prompt.encode()).hexdigest()[:16]

        # Extract images from initial context for multimodal models
        _images = None
        if initial_context and isinstance(initial_context, dict):
            _note_images = initial_context.get("note_images", [])
            if _note_images:
                _images = _note_images
                logger.info(f"[AgenticSearch] Passing {len(_images)} images to final response")

        # Stream the response
        try:
            # generate_async returns a coroutine that yields a stream
            stream = await self.model_manager.generate_async(
                prompt=final_prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                max_tokens=4096,
                images=_images,
            )

            # Handle different return types
            _was_reasoning = False
            if hasattr(stream, '__aiter__'):
                # It's an async iterator (OpenAI stream)
                async for chunk in stream:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        # Skip reasoning-only chunks (API-level thinking separation)
                        delta_reasoning = getattr(delta, 'reasoning_content', '') or getattr(delta, 'reasoning', '') or ''
                        delta_content = getattr(delta, 'content', '') or ''
                        if delta_reasoning and not delta_content:
                            if not _was_reasoning:
                                _was_reasoning = True
                                yield "<thinking>"
                            continue
                        if _was_reasoning and delta_content:
                            _was_reasoning = False
                            yield "</thinking>"
                        if delta_content:
                            yield delta_content
                    elif isinstance(chunk, str):
                        yield chunk
            elif isinstance(stream, str):
                # It's a complete string (local model or stub)
                yield stream
            else:
                # Try to iterate as sync iterator
                for chunk in stream:
                    if isinstance(chunk, str):
                        yield chunk

        except Exception as e:
            logger.error(f"[AgenticSearch] Final generation failed: {e}")
            yield f"I apologize, but I encountered an error generating the response: {str(e)}"

    def _compute_context_inventory(self, initial_context: Optional[Dict[str, Any]]) -> str:
        """
        Compute a short summary of what the RAG pipeline already gathered.

        This prevents the agentic loop from re-searching for information
        that's already available in the prompt context.

        Args:
            initial_context: The pre-gathered context dict from the prompt builder

        Returns:
            A concise inventory string listing available context sections
        """
        if not initial_context:
            return ""

        lines = []

        user_profile = initial_context.get('user_profile', '')
        if user_profile and isinstance(user_profile, str) and user_profile.strip():
            # Count lines as rough proxy for fact count
            fact_count = len([l for l in user_profile.strip().split('\n') if l.strip()])
            lines.append(f"- [USER PROFILE]: {fact_count} categorized facts")

        recent_summaries = initial_context.get('recent_summaries', [])
        if recent_summaries:
            lines.append(f"- [RECENT SUMMARIES]: {len(recent_summaries)} session summaries")

        semantic_summaries = initial_context.get('semantic_summaries', [])
        if semantic_summaries:
            lines.append(f"- [SEMANTIC SUMMARIES]: {len(semantic_summaries)} topically relevant summaries")

        # Handle both list and dict format for summaries
        summaries = initial_context.get('summaries', [])
        if isinstance(summaries, dict):
            if not recent_summaries and summaries.get('recent'):
                lines.append(f"- [RECENT SUMMARIES]: {len(summaries['recent'])} session summaries")
            if not semantic_summaries and summaries.get('semantic'):
                lines.append(f"- [SEMANTIC SUMMARIES]: {len(summaries['semantic'])} topically relevant summaries")

        recent_reflections = initial_context.get('recent_reflections', [])
        reflections = initial_context.get('reflections', [])
        if recent_reflections:
            lines.append(f"- [RECENT REFLECTIONS]: {len(recent_reflections)} reflections")
        elif isinstance(reflections, list) and reflections:
            lines.append(f"- [REFLECTIONS]: {len(reflections)} reflections")

        personal_notes = initial_context.get('personal_notes', [])
        if personal_notes:
            lines.append(f"- [PERSONAL NOTES]: {len(personal_notes)} Obsidian notes")

        memories = initial_context.get('memories', [])
        if memories:
            lines.append(f"- [RELEVANT MEMORIES]: {len(memories)} conversation memories")

        recent = initial_context.get('recent_conversations', [])
        if recent:
            lines.append(f"- [RECENT CONVERSATIONS]: {len(recent)} recent exchanges")

        reference_docs = initial_context.get('reference_docs', [])
        if reference_docs:
            lines.append(f"- [DAEMON DOCUMENTATION]: {len(reference_docs)} reference docs")

        dreams = initial_context.get('dreams', [])
        if dreams:
            lines.append(f"- [RECENT DREAMS]: {len(dreams)} dream entries")

        visual_mems = initial_context.get('visual_memories', {})
        vm_count = len(visual_mems.get('text_results', [])) if isinstance(visual_mems, dict) else 0
        if vm_count:
            lines.append(f"- [VISUAL MEMORIES]: {vm_count} images already retrieved")

        git_commits = initial_context.get('git_commits', [])
        if git_commits:
            lines.append(f"- [PROJECT COMMIT HISTORY]: {len(git_commits)} commits")

        graph_context = initial_context.get('graph_context', [])
        if graph_context:
            lines.append(f"- [KNOWLEDGE GRAPH]: {len(graph_context)} relationship sentences")

        threads = initial_context.get('unresolved_threads', [])
        if threads:
            lines.append(f"- [UNRESOLVED THREADS]: {len(threads)} open threads")

        insights = initial_context.get('proactive_insights', [])
        if insights:
            lines.append(f"- [PROACTIVE INSIGHTS]: {len(insights)} insights")

        if not lines:
            return ""

        header = "Context already gathered by retrieval pipeline:"
        footer = "Do NOT re-search for information already covered above. Use search_memory to fill gaps in specific collections not yet covered."
        return f"{header}\n" + "\n".join(lines) + f"\n{footer}"

    def _build_iteration_prompt(
        self,
        query: str,
        search_context: str,
        round_number: int,
        session: Optional[AgenticSearchSession] = None
    ) -> str:
        """Build prompt for iteration decision."""
        _now = datetime.now()
        _time_ctx = _now.strftime("Today is %A, %Y-%m-%d %H:%M. ")
        parts = [f"""{_time_ctx}User Question: {query}

Search Results So Far:
{search_context}

You are in round {round_number} of up to {self.max_rounds} search rounds."""]

        # Include context inventory so the LLM knows what RAG already gathered
        if session and session.context_inventory:
            parts.append(session.context_inventory)

        # Include relaxation hint if present (guides LLM to broader queries or synthesis)
        if session and session.relaxation_hint:
            parts.append(session.relaxation_hint)

        # Include memory diversity hint if a collection has been over-searched
        if session and session.memory_search_counts:
            for coll, count in session.memory_search_counts.items():
                if count >= 2:
                    parts.append(
                        f"You've already searched '{coll}' {count} times. "
                        "Try a different collection (summaries, conversations, reflections) "
                        "for broader coverage."
                    )

        # Inject tool health so the LLM knows what's actually working
        tool_health = self._tool_executor.get_tool_health()
        parts.append(
            f"[TOOL STATUS — report these accurately, never claim a tool works if it says UNAVAILABLE]\n{tool_health}"
        )

        parts.append("""Based on the search results above:
1. If you have enough information to fully answer the question, signal you're done and answer.
2. If you need more specific information, request another search with a focused query.
3. Consider what's missing: different aspects, more recent data, or more specific details.

What would you like to do?""")

        return "\n\n".join(parts)

    def _build_final_prompt(
        self,
        query: str,
        session: AgenticSearchSession,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the final prompt with all accumulated context including RAG data."""
        parts = []

        # Add RAG context if available (from prompt builder)
        if initial_context:
            # Recent conversations (historical context)
            recent = initial_context.get('recent_conversations', [])
            if recent:
                recent_text = self._format_recent_conversations(recent)
                if recent_text:
                    parts.append(f"[RECENT CONVERSATION — HISTORICAL CONTEXT ONLY, DO NOT RESPOND TO THESE]\n{recent_text}")

            # Relevant memories (semantic search results)
            memories = initial_context.get('memories', [])
            if memories:
                mem_text = self._format_memories(memories)
                if mem_text:
                    parts.append(f"[RELEVANT MEMORIES]\n{mem_text}")

            # User profile (categorized facts)
            user_profile = initial_context.get('user_profile', '')
            if user_profile and isinstance(user_profile, str) and user_profile.strip():
                parts.append(
                    f"[USER PROFILE]\n"
                    "Stored facts — reference naturally but do not add names, apps, or details not written here.\n"
                    f"{user_profile}")

            # Summaries (recent + semantic)
            # Builder provides: summaries (flat list), recent_summaries, semantic_summaries
            recent_summaries = initial_context.get('recent_summaries', [])
            semantic_summaries = initial_context.get('semantic_summaries', [])
            # Fallback: if using old dict format
            summaries = initial_context.get('summaries', [])
            if isinstance(summaries, dict):
                recent_summaries = recent_summaries or summaries.get('recent', [])
                semantic_summaries = semantic_summaries or summaries.get('semantic', [])
            if recent_summaries:
                sum_text = self._format_summaries(recent_summaries)
                if sum_text:
                    parts.append(f"[RECENT SUMMARIES]\n{sum_text}")
            if semantic_summaries:
                sum_text = self._format_summaries(semantic_summaries)
                if sum_text:
                    parts.append(f"[SEMANTIC SUMMARIES]\n{sum_text}")

            # Personal notes from Obsidian
            personal_notes = initial_context.get('personal_notes', [])
            if personal_notes:
                notes_text = self._format_personal_notes(personal_notes)
                if notes_text:
                    parts.append(f"[USER'S PERSONAL NOTES]\n{notes_text}")

            # Dreams
            dreams = initial_context.get('dreams', [])
            if dreams:
                dreams_text = self._format_dreams(dreams)
                if dreams_text:
                    parts.append(f"[RECENT DREAMS]\n{dreams_text}")

            # Reference docs (Daemon self-knowledge)
            reference_docs = initial_context.get('reference_docs', [])
            if reference_docs:
                doc_lines = []
                for i, doc in enumerate(reference_docs, start=1):
                    if isinstance(doc, dict):
                        content = doc.get('content', '')
                        meta = doc.get('metadata', {}) if isinstance(doc.get('metadata'), dict) else {}
                        title = meta.get('title', '')
                        section = meta.get('section', '')
                        if content:
                            header_parts = []
                            if title:
                                header_parts.append(f"**{title}**")
                            if section:
                                header_parts.append(f"({section})")
                            header = " ".join(header_parts) if header_parts else ""
                            doc_lines.append(f"{i}) {header}\n{content.strip()}" if header else f"{i}) {content.strip()}")
                    elif isinstance(doc, str) and doc.strip():
                        doc_lines.append(doc.strip())
                if doc_lines:
                    parts.append(f"[DAEMON DOCUMENTATION]\n" + "\n\n".join(doc_lines))

            # Reflections
            # Builder provides: reflections (flat list), recent_reflections, semantic_reflections
            recent_reflections = initial_context.get('recent_reflections', [])
            reflections = initial_context.get('reflections', [])
            # Fallback: if using old dict format
            if isinstance(reflections, dict):
                recent_reflections = recent_reflections or reflections.get('recent', [])
            elif isinstance(reflections, list) and not recent_reflections:
                recent_reflections = reflections
            if recent_reflections:
                ref_text = self._format_reflections(recent_reflections)
                if ref_text:
                    parts.append(f"[RECENT REFLECTIONS]\n{ref_text}")

        # Time context (critical for temporal queries — model needs today's date)
        _now = datetime.now()
        parts.append(f"[TIME CONTEXT]\nCurrent time: {_now.strftime('%A, %Y-%m-%d %H:%M:%S')}")

        # Add search results
        if session.accumulated_context:
            parts.append(f"[WEB SEARCH RESULTS - {len(session.rounds)} rounds]\n{session.accumulated_context}")

        # Add the query
        parts.append(f"[CURRENT USER QUERY — RESPOND TO THIS]\n{query}")

        # Tool health — so the LLM never confabulates about its own capabilities
        tool_health = self._tool_executor.get_tool_health()
        parts.append(
            f"[TOOL STATUS — report these accurately, never claim a tool works if it says UNAVAILABLE]\n{tool_health}"
        )

        # Instructions
        has_web = bool(session.accumulated_context)
        citation_line = (
            "- Cite web sources using [WEB_N] markers (e.g., 'According to Reuters [WEB_1]...'). "
            "Every factual claim from web sources MUST include a [WEB_N] citation."
            if has_web else "- Cite web sources when stating facts from search results"
        )
        parts.append(f"""Please provide a comprehensive answer based on ALL context above:
- Use your memories, facts, and personal notes to personalize the response
{citation_line}
- Note any uncertainties or conflicting information
- Focus on answering the user's specific question
- If asked about tool status, ONLY report what [TOOL STATUS] says — do NOT rely on prior conversation""")

        # Budget enforcement: if assembled prompt is too large, trim low-value sections
        # while preserving recent conversations and agentic search results.
        # Use 2x the context_budget_tokens as the ceiling for the full final prompt
        # (context_budget_tokens governs just the agentic results; full prompt gets more room).
        prompt_ceiling = self.context_budget_tokens * 5  # ~40K tokens for default 8K budget
        assembled = "\n\n".join(parts)
        total_tokens = self._estimate_tokens(assembled)
        if total_tokens > prompt_ceiling:
            # Trim sections in priority order: dreams, reflections, reference docs, summaries
            # These are the sections least critical for answering the immediate query
            trimmable_prefixes = [
                "[RECENT DREAMS]",
                "[RECENT REFLECTIONS]",
                "[DAEMON DOCUMENTATION]",
                "[SEMANTIC SUMMARIES]",
                "[RECENT SUMMARIES]",
                "[USER'S PERSONAL NOTES]",
            ]
            for prefix in trimmable_prefixes:
                parts = [p for p in parts if not p.startswith(prefix)]
                assembled = "\n\n".join(parts)
                total_tokens = self._estimate_tokens(assembled)
                if total_tokens <= prompt_ceiling:
                    break
            if total_tokens > prompt_ceiling:
                logger.warning(
                    f"[AgenticSearch] Final prompt still over ceiling after trimming: "
                    f"{total_tokens}/{prompt_ceiling} tokens"
                )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Format delegation wrappers (moved to AgenticFormatter)
    # ------------------------------------------------------------------

    def _format_recent_conversations(self, conversations):
        return self._formatter.format_recent_conversations(conversations)

    def _format_memories(self, memories):
        return self._formatter.format_memories(memories)

    def _format_summaries(self, summaries):
        return self._formatter.format_summaries(summaries)

    def _format_personal_notes(self, notes):
        return self._formatter.format_personal_notes(notes)

    def _format_dreams(self, dreams):
        return self._formatter.format_dreams(dreams)

    def _format_reflections(self, reflections):
        return self._formatter.format_reflections(reflections)

    def _is_low_quality_result(self, result, query: str):
        """Check if search result is low quality (empty, irrelevant, or sparse)."""
        if result is None:
            return True, "no results returned"
        pages = getattr(result, 'pages', []) if result else []
        if not pages:
            return True, "empty results"
        if len(pages) < 2:
            return True, "very few results"
        return False, ""

    def _generate_relaxation_suggestion(self, query: str) -> str:
        """Generate a suggestion for query relaxation."""
        if len(query.split()) > 6:
            return "Try a shorter, more focused query"
        return "Try alternative phrasing or broader terms"

    def _format_search_context(self, round_number, query, content):
        return self._formatter.format_search_context(round_number, query, content)

    def _format_wolfram_context(self, round_number, query, content):
        return self._formatter.format_wolfram_context(round_number, query, content)

    def _format_sandbox_context(self, round_number, purpose, content):
        return self._formatter.format_sandbox_context(round_number, purpose, content)

    async def _execute_memory_search(self, query, collection):
        return await self._tool_executor._execute_memory_search(query, collection)

    def _search_wiki_faiss(self, query, k=8):
        return self._tool_executor._search_wiki_faiss(query, k)

    def _format_wiki_faiss_results(self, results):
        return self._formatter.format_wiki_faiss_results(results)

    def _format_memory_results(self, results, collection):
        return self._formatter.format_memory_results(results, collection)

    def _format_memory_context(self, round_num, collection, query, results):
        return self._formatter.format_memory_context(round_num, collection, query, results)

    def _execute_memory_expand(self, memory_id, window=3, collection=None):
        return self._tool_executor._execute_memory_expand(memory_id, window, collection)

    def _format_expanded_results(self, result):
        return self._formatter.format_expanded_results(result)

    def _format_expand_context(self, round_num, memory_id, results):
        return self._formatter.format_expand_context(round_num, memory_id, results)

    async def _execute_file_read(self, filepath, start_line=None, end_line=None):
        return await self._tool_executor._execute_file_read(filepath, start_line, end_line)

    async def _execute_file_grep(self, pattern, folder=None, file_glob=None):
        return await self._tool_executor._execute_file_grep(pattern, folder, file_glob)

    async def _execute_file_list(self, dirpath, recursive=False):
        return await self._tool_executor._execute_file_list(dirpath, recursive)

    def _format_file_context(self, round_num, operation, content):
        return self._formatter.format_file_context(round_num, operation, content)

    async def _execute_full_document_retrieval(self, title):
        return await self._tool_executor._execute_full_document_retrieval(title)

    def _format_full_document_context(self, round_num, title, content):
        return self._formatter.format_full_document_context(round_num, title, content)

    async def _execute_git_stats(self, query):
        return await self._tool_executor._execute_git_stats(query)

    def _format_git_stats_context(self, round_num, query, content):
        return self._formatter.format_git_stats_context(round_num, query, content)

    async def _execute_wolfram(self, query):
        return await self._tool_executor._execute_wolfram(query)
