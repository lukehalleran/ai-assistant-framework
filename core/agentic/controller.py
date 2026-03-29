"""
Agentic Search Controller Module

Contract:
    - Provides AgenticSearchController for multi-round search loops
    - Manages ReAct cycle: Think → Act (search/compute/code) → Observe → Repeat
    - Emits ProgressEvent for UI updates
    - Enforces max_rounds limit (default 5)
    - Compresses search results to fit context budget
    - Budget-enforced accumulated_context: _append_accumulated() trims oldest rounds
      when accumulated context exceeds context_budget_tokens (default 8000) [NEW 2026-03-28]
    - Budget-aware final prompt: _build_final_prompt() trims low-value sections
      (dreams, reflections, docs, summaries) if total exceeds ceiling [NEW 2026-03-28]
    - Falls back gracefully on search/API failures

Tool Support:
    - Web search via WebSearchManager
    - Wolfram Alpha computation via WolframManager (optional)
    - Python code execution via SandboxManager (optional) [NEW 2026-01-22]
      - Persistent sessions for variable persistence across turns
      - Automatic cleanup in finally block
    - Memory search via ChromaDB collections (optional)
    - Memory expansion via MemoryExpander (optional) [NEW 2026-03]
      - Expands a search result to show surrounding turns (timestamp window)
      - For summaries, retrieves original source conversations
      - Gated by EXPAND_MAX_PER_SESSION per session
    - File access via file_read/file_grep/file_list tools (optional) [NEW 2026-03-26]

Provenance [NEW 2026-03-26]:
    - Computes final_prompt_hash (SHA-256[:16]) on the assembled prompt for audit trail
    - Saves completed session to _last_session for handler access after execute_search()
    - Formats memory citations with [MEM_RECENT_N] / [MEM_SEMANTIC_N] markers for citation extraction

Key Parameters:
    - skip_initial_search: bool - Skip Round 1 web search for computation-only queries [NEW 2026-01-22]

Dependencies:
    - models.model_manager.ModelManager (for LLM generation)
    - knowledge.web_search_manager.WebSearchManager (for web searches)
    - knowledge.wolfram_manager.WolframManager (for computations, optional)
    - knowledge.sandbox_manager.SandboxManager (for code execution, optional) [NEW 2026-01-22]
    - memory.memory_expander.MemoryExpander (for memory expansion, optional) [NEW 2026-03]
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
    LOW_QUALITY_HINT_TEMPLATE,
    MAX_RELAXATION_HINT,
)
from core.agentic.protocols import (
    detect_protocol,
    get_protocol_handler,
    BaseProtocolHandler,
)

if TYPE_CHECKING:
    from models.model_manager import ModelManager
    from knowledge.web_search_manager import WebSearchManager, WebSearchResult
    from knowledge.wolfram_manager import WolframManager
    from knowledge.sandbox_manager import SandboxManager, PersistentSession, SandboxResult
    from core.prompt.token_manager import TokenManager
    from core.file_access_manager import FileAccessManager

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
    })

    def __init__(
        self,
        model_manager: "ModelManager",
        web_search_manager: "WebSearchManager",
        chroma_store=None,
        wolfram_manager: Optional["WolframManager"] = None,
        sandbox_manager: Optional["SandboxManager"] = None,
        file_access_manager: Optional["FileAccessManager"] = None,
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
        handler = get_protocol_handler(
            protocol,
            wolfram_available=wolfram_available,
            sandbox_available=sandbox_available,
            memory_available=memory_available,
            file_access_available=file_access_available,
        )

        # Augment system prompt for agentic mode
        augmented_system_prompt = handler.augment_system_prompt(
            system_prompt, self.max_rounds
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
            # === ROUND 1: Automatic search with trigger terms (unless skipped) ===
            if skip_initial_search:
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
                decision = await self._get_model_decision(
                    prompt=iteration_prompt,
                    system_prompt=augmented_system_prompt,
                    model_name=model_name,
                    handler=handler,
                    session=session
                )

                if decision.wants_search and decision.search_query:
                    # Model wants another search
                    yield ProgressEvent(
                        event_type="searching",
                        message=f"Searching for: {decision.search_query}",
                        round_number=session.current_round,
                        metadata={"query": decision.search_query, "reason": decision.search_reason}
                    )

                    session.state = AgentState.SEARCHING
                    start_time = time.time()
                    result = await self._execute_search(
                        [decision.search_query],
                        crisis_level=crisis_level
                    )
                    search_duration = (time.time() - start_time) * 1000

                    # Record round
                    round_data = SearchRound(
                        round_number=session.current_round,
                        request=SearchRequest(
                            query=decision.search_query,
                            reason=decision.search_reason,
                            round_number=session.current_round
                        ),
                        results=result,
                        duration_ms=search_duration
                    )

                    result_count = len(result.pages) if result and hasattr(result, 'pages') else 0
                    yield ProgressEvent(
                        event_type="found_results",
                        message=f"Found {result_count} results",
                        round_number=session.current_round,
                        metadata={"result_count": result_count}
                    )

                    # Compress and accumulate
                    session.state = AgentState.OBSERVING
                    compressed = await self._compress_results(result)
                    round_data.summary = compressed
                    session.rounds.append(round_data)

                    # Add to accumulated context (budget-enforced)
                    self._append_accumulated(session, self._format_search_context(
                        session.current_round - 1,  # Already incremented by rounds.append
                        decision.search_query,
                        compressed
                    ))

                    # Check result quality and update relaxation hint
                    is_low_quality, issue = self._is_low_quality_result(
                        result, decision.search_query
                    )
                    if is_low_quality:
                        session.low_quality_search_count += 1
                        if session.low_quality_search_count > 2:
                            session.relaxation_hint = MAX_RELAXATION_HINT
                            logger.info(
                                "[AgenticSearch] Max relaxation attempts reached, "
                                "forcing synthesis"
                            )
                        else:
                            suggestion = self._generate_relaxation_suggestion(decision.search_query)
                            remaining = 2 - session.low_quality_search_count
                            session.relaxation_hint = LOW_QUALITY_HINT_TEMPLATE.format(
                                query=decision.search_query,
                                issue=issue,
                                suggestion=suggestion,
                                remaining=remaining
                            )
                            logger.info(
                                f"[AgenticSearch] Low quality result ({issue}), "
                                f"relaxation count: {session.low_quality_search_count}"
                            )
                    else:
                        # Good results - reset counter and clear hint
                        session.low_quality_search_count = 0
                        session.relaxation_hint = None
                        logger.debug("[AgenticSearch] Good search results, reset relaxation counter")

                elif decision.wants_wolfram and decision.wolfram_query:
                    # Model wants a Wolfram Alpha computation
                    yield ProgressEvent(
                        event_type="computing",
                        message=f"Computing: {decision.wolfram_query}",
                        round_number=session.current_round,
                        metadata={"query": decision.wolfram_query, "reason": decision.wolfram_reason}
                    )

                    session.state = AgentState.SEARCHING  # Reuse SEARCHING state
                    start_time = time.time()
                    wolfram_result = await self._execute_wolfram(decision.wolfram_query)
                    compute_duration = (time.time() - start_time) * 1000

                    # Record round
                    round_data = SearchRound(
                        round_number=session.current_round,
                        request=SearchRequest(
                            query=decision.wolfram_query,
                            reason=decision.wolfram_reason,
                            round_number=session.current_round
                        ),
                        results=None,  # Wolfram results stored as summary
                        duration_ms=compute_duration
                    )

                    yield ProgressEvent(
                        event_type="computed",
                        message="Computation complete",
                        round_number=session.current_round,
                        metadata={"duration_ms": compute_duration}
                    )

                    # Store result and accumulate
                    session.state = AgentState.OBSERVING
                    round_data.summary = wolfram_result
                    session.rounds.append(round_data)

                    # Add to accumulated context (budget-enforced)
                    self._append_accumulated(session, self._format_wolfram_context(
                        session.current_round - 1,
                        decision.wolfram_query,
                        wolfram_result
                    ))

                elif decision.wants_sandbox and decision.sandbox_code:
                    # Model wants to execute Python code in sandbox
                    purpose = decision.sandbox_purpose or "executing code"
                    yield ProgressEvent(
                        event_type="executing_code",
                        message=f"Running Python: {purpose}",
                        round_number=session.current_round,
                        metadata={"purpose": purpose}
                    )

                    session.state = AgentState.SEARCHING  # Reuse SEARCHING state
                    start_time = time.time()

                    # Execute in persistent session if available, otherwise ephemeral
                    if sandbox_session and not sandbox_session.is_closed:
                        sandbox_result = await sandbox_session.run(decision.sandbox_code)
                    elif self.sandbox_manager and self.sandbox_manager.is_available():
                        sandbox_result = await self.sandbox_manager.execute_code(decision.sandbox_code)
                    else:
                        # No sandbox available - create error result
                        from knowledge.sandbox_manager import SandboxResult
                        sandbox_result = SandboxResult(
                            code=decision.sandbox_code,
                            success=False,
                            error="Code sandbox not available (E2B not configured)"
                        )

                    execution_duration = (time.time() - start_time) * 1000

                    # Record round
                    round_data = SearchRound(
                        round_number=session.current_round,
                        request=SearchRequest(
                            query=f"[Python: {purpose}]",
                            reason=purpose,
                            round_number=session.current_round
                        ),
                        results=None,  # Sandbox results stored as summary
                        duration_ms=execution_duration
                    )

                    if sandbox_result.success:
                        yield ProgressEvent(
                            event_type="code_executed",
                            message=f"Code executed ({sandbox_result.execution_time:.1f}s)",
                            round_number=session.current_round,
                            metadata={"duration_ms": execution_duration}
                        )
                    else:
                        yield ProgressEvent(
                            event_type="code_error",
                            message="Execution error (see details)",
                            round_number=session.current_round,
                            metadata={"error": sandbox_result.error}
                        )

                    # Store result and accumulate
                    session.state = AgentState.OBSERVING
                    formatted_result = self.sandbox_manager.format_for_prompt(
                        sandbox_result, purpose
                    ) if self.sandbox_manager else str(sandbox_result.error or sandbox_result.stdout)
                    round_data.summary = formatted_result
                    session.rounds.append(round_data)

                    # Add to accumulated context (budget-enforced)
                    self._append_accumulated(session, self._format_sandbox_context(
                        session.current_round - 1,
                        purpose,
                        formatted_result
                    ))

                elif decision.wants_memory_search and decision.memory_query:
                    # Model wants to search internal memory/knowledge base
                    collection = decision.memory_collection or "facts"
                    yield ProgressEvent(
                        event_type="searching_memory",
                        message=f"Searching {collection}: {decision.memory_query}",
                        round_number=session.current_round,
                        metadata={
                            "query": decision.memory_query,
                            "collection": collection,
                            "reason": decision.memory_reason,
                        }
                    )

                    session.state = AgentState.SEARCHING
                    start_time = time.time()
                    memory_result = await self._execute_memory_search(
                        decision.memory_query, collection
                    )
                    search_duration = (time.time() - start_time) * 1000

                    round_data = SearchRound(
                        round_number=session.current_round,
                        request=SearchRequest(
                            query=f"[Memory: {collection}] {decision.memory_query}",
                            reason=decision.memory_reason,
                            round_number=session.current_round
                        ),
                        results=None,
                        duration_ms=search_duration
                    )

                    yield ProgressEvent(
                        event_type="found_results",
                        message=f"Found memory results from {collection}",
                        round_number=session.current_round,
                        metadata={"collection": collection, "duration_ms": search_duration}
                    )

                    session.state = AgentState.OBSERVING
                    round_data.summary = memory_result
                    session.rounds.append(round_data)

                    # Track per-collection search counts for diversity enforcement
                    session.memory_search_counts[collection] = (
                        session.memory_search_counts.get(collection, 0) + 1
                    )

                    self._append_accumulated(session, self._format_memory_context(
                        session.current_round - 1,
                        collection,
                        decision.memory_query,
                        memory_result
                    ))

                elif decision.wants_memory_expand and decision.expand_memory_id:
                    # Model wants to expand a memory hit for surrounding context
                    from config.app_config import EXPAND_MEMORY_ENABLED, EXPAND_MAX_PER_SESSION
                    memory_id = decision.expand_memory_id
                    if not EXPAND_MEMORY_ENABLED or not self.memory_expander:
                        logger.info("[AgenticSearch] expand_memory disabled or no expander")
                        # Treat as implicit answer to avoid stalling
                        break
                    if session.expand_count >= EXPAND_MAX_PER_SESSION:
                        logger.info("[AgenticSearch] expand_memory limit reached (%d/%d)",
                                    session.expand_count, EXPAND_MAX_PER_SESSION)
                        break

                    is_summary = (decision.expand_collection == "summaries")
                    recall_label = "Recalling Long Term Memory..." if is_summary else "Recalling..."

                    yield ProgressEvent(
                        event_type="expanding_memory",
                        message=recall_label,
                        round_number=session.current_round,
                        metadata={
                            "memory_id": memory_id,
                            "collection": decision.expand_collection,
                            "reason": decision.expand_reason,
                        }
                    )

                    session.state = AgentState.SEARCHING
                    start_time = time.time()
                    expand_result = self._execute_memory_expand(
                        memory_id, decision.expand_window, decision.expand_collection
                    )
                    duration = (time.time() - start_time) * 1000

                    formatted = self._format_expanded_results(expand_result)
                    n_turns = len(expand_result.get("turns", []))

                    round_data = SearchRound(
                        round_number=session.current_round,
                        request=SearchRequest(
                            query=f"[Expand Memory] {memory_id[:8]}",
                            reason=decision.expand_reason,
                            round_number=session.current_round
                        ),
                        results=None,
                        duration_ms=duration
                    )

                    done_label = (
                        f"Recalled {n_turns} memories from long term"
                        if is_summary else f"Recalled {n_turns} surrounding turns"
                    )
                    yield ProgressEvent(
                        event_type="memory_expanded",
                        message=done_label,
                        round_number=session.current_round,
                        metadata={"duration_ms": duration}
                    )

                    session.state = AgentState.OBSERVING
                    round_data.summary = formatted
                    session.rounds.append(round_data)
                    session.expand_count += 1
                    self._append_accumulated(session, self._format_expand_context(
                        session.current_round - 1, memory_id, formatted
                    ))

                elif decision.wants_file_read and decision.file_read_path:
                    # Model wants to read a file from disk
                    yield ProgressEvent(
                        event_type="reading_file",
                        message=f"Reading {decision.file_read_path}",
                        round_number=session.current_round,
                        metadata={"path": decision.file_read_path, "reason": decision.file_read_reason}
                    )

                    session.state = AgentState.SEARCHING
                    start_time = time.time()
                    file_result = await self._execute_file_read(
                        decision.file_read_path,
                        decision.file_read_start_line,
                        decision.file_read_end_line,
                    )
                    duration = (time.time() - start_time) * 1000

                    round_data = SearchRound(
                        round_number=session.current_round,
                        request=SearchRequest(
                            query=f"[File Read] {decision.file_read_path}",
                            reason=decision.file_read_reason,
                            round_number=session.current_round
                        ),
                        results=None,
                        duration_ms=duration
                    )

                    yield ProgressEvent(
                        event_type="file_read",
                        message=f"Read {decision.file_read_path}",
                        round_number=session.current_round,
                        metadata={"duration_ms": duration}
                    )

                    session.state = AgentState.OBSERVING
                    round_data.summary = file_result
                    session.rounds.append(round_data)
                    self._append_accumulated(session, self._format_file_context(
                        session.current_round - 1,
                        f"file_read: {decision.file_read_path}",
                        file_result
                    ))

                elif decision.wants_file_grep and decision.file_grep_pattern:
                    # Model wants to grep files on disk
                    yield ProgressEvent(
                        event_type="searching_files",
                        message=f"Grepping for '{decision.file_grep_pattern}'",
                        round_number=session.current_round,
                        metadata={"pattern": decision.file_grep_pattern, "reason": decision.file_grep_reason}
                    )

                    session.state = AgentState.SEARCHING
                    start_time = time.time()
                    grep_result = await self._execute_file_grep(
                        decision.file_grep_pattern,
                        decision.file_grep_folder,
                        decision.file_grep_glob,
                    )
                    duration = (time.time() - start_time) * 1000

                    round_data = SearchRound(
                        round_number=session.current_round,
                        request=SearchRequest(
                            query=f"[File Grep] {decision.file_grep_pattern}",
                            reason=decision.file_grep_reason,
                            round_number=session.current_round
                        ),
                        results=None,
                        duration_ms=duration
                    )

                    yield ProgressEvent(
                        event_type="files_searched",
                        message=f"Grep complete for '{decision.file_grep_pattern}'",
                        round_number=session.current_round,
                        metadata={"duration_ms": duration}
                    )

                    session.state = AgentState.OBSERVING
                    round_data.summary = grep_result
                    session.rounds.append(round_data)
                    self._append_accumulated(session, self._format_file_context(
                        session.current_round - 1,
                        f"file_grep: {decision.file_grep_pattern}",
                        grep_result
                    ))

                elif decision.wants_file_list and decision.file_list_path:
                    # Model wants to list a directory
                    yield ProgressEvent(
                        event_type="listing_files",
                        message=f"Listing {decision.file_list_path}",
                        round_number=session.current_round,
                        metadata={"path": decision.file_list_path, "reason": decision.file_list_reason}
                    )

                    session.state = AgentState.SEARCHING
                    start_time = time.time()
                    list_result = await self._execute_file_list(
                        decision.file_list_path,
                        decision.file_list_recursive,
                    )
                    duration = (time.time() - start_time) * 1000

                    round_data = SearchRound(
                        round_number=session.current_round,
                        request=SearchRequest(
                            query=f"[File List] {decision.file_list_path}",
                            reason=decision.file_list_reason,
                            round_number=session.current_round
                        ),
                        results=None,
                        duration_ms=duration
                    )

                    yield ProgressEvent(
                        event_type="files_listed",
                        message=f"Listed {decision.file_list_path}",
                        round_number=session.current_round,
                        metadata={"duration_ms": duration}
                    )

                    session.state = AgentState.OBSERVING
                    round_data.summary = list_result
                    session.rounds.append(round_data)
                    self._append_accumulated(session, self._format_file_context(
                        session.current_round - 1,
                        f"file_list: {decision.file_list_path}",
                        list_result
                    ))

                elif decision.is_done:
                    # Model signals it has enough info
                    session.model_signaled_done = True
                    session.done_reason = decision.done_reason
                    logger.info(f"[AgenticSearch] Model signaled done: {decision.done_reason}")
                    break

                else:
                    # Model wants to answer (no explicit done signal)
                    logger.info("[AgenticSearch] Model ready to answer (implicit)")
                    break

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

    async def _execute_search(
        self,
        search_terms: List[str],
        crisis_level: Optional[str] = None
    ) -> Any:
        """
        Execute web search with given terms.

        Args:
            search_terms: List of search queries
            crisis_level: Current crisis level

        Returns:
            WebSearchResult or MultiSearchResult
        """
        from knowledge.web_search_manager import WebSearchDepth

        if len(search_terms) == 1:
            return await self.web_search_manager.search(
                query=search_terms[0],
                depth=WebSearchDepth.STANDARD,
                crisis_level=crisis_level
            )
        else:
            # Multiple terms - use multi_search
            return await self.web_search_manager.multi_search(
                query=search_terms[0],  # Primary query
                depth=WebSearchDepth.STANDARD,
                auto_decompose=False  # We already have decomposed terms
            )

    async def _compress_results(
        self,
        result: Any,
        max_tokens: int = DEFAULT_COMPRESSION_MAX_TOKENS
    ) -> str:
        """
        Compress search results to fit context budget.

        Args:
            result: WebSearchResult to compress
            max_tokens: Maximum tokens for compressed output

        Returns:
            Compressed text representation of results
        """
        if not result or not hasattr(result, 'pages') or not result.pages:
            return "No results found."

        # Format results
        formatted = result.get_formatted_content(max_chars=6000)

        # Estimate token count (rough: 4 chars per token)
        estimated_tokens = len(formatted) // 4

        if estimated_tokens <= max_tokens:
            return formatted

        # Need to compress via LLM
        try:
            summary_prompt = f"""Summarize these search results concisely, preserving key facts, dates, and sources:

{formatted}

Provide a focused summary with the most important information."""

            summary = await self.model_manager.generate_once(
                prompt=summary_prompt,
                model_name=self.compression_model,
                max_tokens=max_tokens,
                temperature=0.3
            )

            return summary if summary else formatted[:max_tokens * 4]

        except Exception as e:
            logger.warning(f"[AgenticSearch] Compression failed: {e}")
            # Fallback to truncation
            return formatted[:max_tokens * 4]

    async def _get_model_decision(
        self,
        prompt: str,
        system_prompt: str,
        model_name: str,
        handler: BaseProtocolHandler,
        session: AgenticSearchSession
    ) -> SearchDecision:
        """
        Get the model's decision on what to do next.

        Args:
            prompt: The prompt to send
            system_prompt: System prompt with agentic instructions
            model_name: Model to use
            handler: Protocol handler for parsing
            session: Current session state

        Returns:
            SearchDecision indicating model's choice
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
            return SearchDecision(wants_answer=True)

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

        # Stream the response
        try:
            # generate_async returns a coroutine that yields a stream
            stream = await self.model_manager.generate_async(
                prompt=final_prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                max_tokens=4096
            )

            # Handle different return types
            if hasattr(stream, '__aiter__'):
                # It's an async iterator (OpenAI stream)
                async for chunk in stream:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            yield delta.content
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
        parts = [f"""User Question: {query}

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
                parts.append(f"[USER PROFILE]\n{user_profile}")

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

        # Add search results
        if session.accumulated_context:
            parts.append(f"[WEB SEARCH RESULTS - {len(session.rounds)} rounds]\n{session.accumulated_context}")

        # Add the query
        parts.append(f"[CURRENT USER QUERY — RESPOND TO THIS]\n{query}")

        # Instructions
        parts.append("""Please provide a comprehensive answer based on ALL context above:
- Use your memories, facts, and personal notes to personalize the response
- Cite web sources when stating facts from search results
- Note any uncertainties or conflicting information
- Focus on answering the user's specific question""")

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

    def _format_recent_conversations(self, conversations: List[Dict]) -> str:
        """Format recent conversations for the prompt with citation markers."""
        if not conversations:
            return ""
        lines = []
        for i, conv in enumerate(conversations, 1):
            ts = conv.get('timestamp', '')
            user_msg = conv.get('query', conv.get('user', ''))
            assistant_msg = conv.get('response', conv.get('assistant', ''))
            if user_msg:
                lines.append(f"[MEM_RECENT_{i}] {ts}: User: {user_msg[:500]}")
                if assistant_msg:
                    lines.append(f"   Daemon: {assistant_msg[:500]}")
        return "\n".join(lines)

    def _format_memories(self, memories: List[Dict]) -> str:
        """Format memories for the prompt with citation markers."""
        if not memories:
            return ""
        lines = []
        for i, mem in enumerate(memories, 1):
            ts = mem.get('timestamp', '')
            content = mem.get('content', mem.get('query', ''))
            response = mem.get('response', '')
            if content:
                lines.append(f"[MEM_SEMANTIC_{i}] {ts}: {content[:400]}")
                if response:
                    lines.append(f"   Response: {response[:400]}")
        return "\n".join(lines)

    def _format_summaries(self, summaries: List[Dict]) -> str:
        """Format summaries for the prompt."""
        if not summaries:
            return ""
        lines = []
        for i, s in enumerate(summaries, 1):
            content = s.get('content', s.get('summary', ''))
            ts = s.get('timestamp', '')
            if content:
                lines.append(f"{i}) [{ts}] {content[:600]}")
        return "\n".join(lines)

    def _format_personal_notes(self, notes: List[Dict]) -> str:
        """Format personal notes from Obsidian for the prompt."""
        if not notes:
            return ""
        lines = []
        for i, note in enumerate(notes, 1):
            title = note.get('metadata', {}).get('title', 'Untitled')
            content = note.get('content', '')[:500]
            tags = note.get('metadata', {}).get('tags', '')
            if content:
                tag_str = f" [tags: {tags}]" if tags else ""
                lines.append(f"{i}) {title}{tag_str}: {content}")
        return "\n".join(lines)

    def _format_dreams(self, dreams: List[Dict]) -> str:
        """Format dreams for the prompt."""
        if not dreams:
            return ""
        lines = []
        for i, d in enumerate(dreams, 1):
            content = d.get('content', d.get('dream', ''))
            ts = d.get('timestamp', '')
            if content:
                lines.append(f"{i}) [{ts}] {content[:400]}")
        return "\n".join(lines)

    def _format_reflections(self, reflections: List[Dict]) -> str:
        """Format reflections for the prompt."""
        if not reflections:
            return ""
        lines = []
        for i, r in enumerate(reflections, 1):
            content = r.get('content', r.get('reflection', ''))
            ts = r.get('timestamp', '')
            if content:
                lines.append(f"{i}) [{ts}] {content[:400]}")
        return "\n".join(lines)

    def _format_search_context(
        self,
        round_number: int,
        query: str,
        content: str
    ) -> str:
        """Format a single search round for context."""
        return f"[Search Round {round_number}] Query: {query}\n{content}"

    def _format_wolfram_context(
        self,
        round_number: int,
        query: str,
        content: str
    ) -> str:
        """Format a single Wolfram Alpha computation for context."""
        return f"[Computation Round {round_number}] Query: {query}\n{content}"

    def _format_sandbox_context(
        self,
        round_number: int,
        purpose: str,
        content: str
    ) -> str:
        """Format a single Python code execution for context."""
        return f"[Code Execution Round {round_number}] Purpose: {purpose}\n{content}"

    async def _execute_memory_search(self, query: str, collection: str) -> str:
        """Execute raw semantic search against a ChromaDB collection."""
        from config.app_config import AGENTIC_MEMORY_SEARCH_LIMIT

        if not self.chroma_store:
            return "[Memory search unavailable]"

        if collection not in self.VALID_MEMORY_COLLECTIONS:
            return f"[Invalid collection: {collection}. Valid: {', '.join(sorted(self.VALID_MEMORY_COLLECTIONS))}]"

        try:
            results = self.chroma_store.query_collection(
                collection_name=collection,
                query_text=query,
                n_results=AGENTIC_MEMORY_SEARCH_LIMIT,
            )

            if not results:
                return f"[No results found in {collection} for: {query}]"

            return self._format_memory_results(results, collection)

        except Exception as e:
            logger.warning(f"[AgenticSearch] Memory search failed: {e}")
            return f"[Memory search error: {e}]"

    def _format_memory_results(self, results: list, collection: str) -> str:
        """Format ChromaDB results into readable text for the LLM."""
        lines = []
        for i, r in enumerate(results, 1):
            content = r.get("content", "").strip()
            score = r.get("relevance_score", 0.0)
            meta = r.get("metadata", {})
            doc_id = r.get("id", "")

            header_parts = [f"[{i}]"]
            if doc_id:
                header_parts.append(f"(id: {doc_id})")
            if collection == "reference_docs":
                title = meta.get("title", "")
                section = meta.get("section", "")
                if title:
                    header_parts.append(title)
                if section:
                    header_parts.append(f"({section})")
            elif collection == "facts":
                subject = meta.get("subject", "")
                relation = meta.get("relation", "")
                if subject and relation:
                    header_parts.append(f"{subject} — {relation}")
            elif collection in ("conversations", "summaries", "reflections"):
                ts = meta.get("timestamp", "")
                if ts:
                    header_parts.append(ts[:19])

            header_parts.append(f"(score: {score:.2f})")
            header_parts.append(f"[{collection}]")
            header = " ".join(header_parts)

            if len(content) > 500:
                content = content[:500] + "..."

            lines.append(f"{header}\n{content}")

        return "\n\n".join(lines)

    def _format_memory_context(
        self, round_num: int, collection: str, query: str, results: str
    ) -> str:
        """Format memory search results for accumulated context."""
        return (
            f"[MEMORY SEARCH — Round {round_num} — {collection}]\n"
            f"Query: {query}\n"
            f"Results:\n{results}"
        )

    # ------------------------------------------------------------------
    # Memory expansion execution
    # ------------------------------------------------------------------

    def _execute_memory_expand(
        self, memory_id: str, window: int = 3, collection: Optional[str] = None
    ) -> dict:
        """Run MemoryExpander.expand() and return the result dict."""
        if not self.memory_expander:
            return {"anchor_id": memory_id, "turns": [], "error": "Expander not available"}
        try:
            from config.app_config import EXPAND_MAX_WINDOW
            window = max(1, min(window, EXPAND_MAX_WINDOW))
            return self.memory_expander.expand(memory_id, window, collection)
        except Exception as e:
            logger.warning(f"[AgenticSearch] Memory expand failed: {e}")
            return {"anchor_id": memory_id, "turns": [], "error": str(e)}

    def _format_expanded_results(self, result: dict) -> str:
        """Format expand result dict into readable text for the LLM."""
        error = result.get("error")
        turns = result.get("turns", [])
        collection = result.get("collection", "?")
        method = result.get("expansion_method", "timestamp_window")
        total = result.get("total_in_collection", 0)

        if method == "source_docs":
            # Summary expansion — first turn is the summary anchor, rest are source conversations
            anchor_turns = [t for t in turns if t.get("is_anchor")]
            source_turns = [t for t in turns if not t.get("is_anchor")]
            lines = [f"[Summary expanded to {len(source_turns)} source conversations]"]
            if error:
                lines.append(f"Note: {error}")
            if anchor_turns:
                lines.append(f"--- SUMMARY ---")
                lines.append(anchor_turns[0].get("content", ""))
            if source_turns:
                lines.append(f"\n--- ORIGINAL CONVERSATIONS ({len(source_turns)}) ---")
                for t in source_turns:
                    ts = t.get("timestamp", "")[:19]
                    tid = t.get("id", "")[:8]
                    content = t.get("content", "")
                    lines.append(f"[{tid}] {ts}")
                    lines.append(content)
                    lines.append("")
        else:
            lines = [f"[Expanded from {collection} | method: {method} | {len(turns)} turns shown / {total} total]"]
            if error:
                lines.append(f"Note: {error}")
            for t in turns:
                marker = "  <<<< TARGET" if t.get("is_anchor") else ""
                ts = t.get("timestamp", "")[:19]
                tid = t.get("id", "")[:8]
                content = t.get("content", "")
                lines.append(f"--- [{tid}] {ts}{marker} ---")
                lines.append(content)

        return "\n".join(lines)

    def _format_expand_context(self, round_num: int, memory_id: str, results: str) -> str:
        """Format expanded results for accumulated context."""
        return (
            f"[MEMORY EXPANSION — Round {round_num} — {memory_id[:8]}]\n"
            f"{results}"
        )

    # ------------------------------------------------------------------
    # File access execution
    # ------------------------------------------------------------------

    async def _execute_file_read(
        self, filepath: str, start_line: Optional[int] = None, end_line: Optional[int] = None
    ) -> str:
        """Read a file via FileAccessManager."""
        if not self.file_access_manager:
            return "[File access not configured]"
        try:
            result = await self.file_access_manager.read_file(filepath, start_line, end_line)
            return self.file_access_manager.format_read_for_prompt(result)
        except Exception as e:
            logger.warning(f"[AgenticSearch] File read failed: {e}")
            return f"[File read error: {e}]"

    async def _execute_file_grep(
        self, pattern: str, folder: Optional[str] = None, file_glob: Optional[str] = None
    ) -> str:
        """Grep files via FileAccessManager."""
        if not self.file_access_manager:
            return "[File access not configured]"
        try:
            result = await self.file_access_manager.grep_files(
                pattern, folder, file_glob or "*.py"
            )
            return self.file_access_manager.format_grep_for_prompt(result)
        except Exception as e:
            logger.warning(f"[AgenticSearch] File grep failed: {e}")
            return f"[File grep error: {e}]"

    async def _execute_file_list(
        self, dirpath: str, recursive: bool = False
    ) -> str:
        """List directory via FileAccessManager."""
        if not self.file_access_manager:
            return "[File access not configured]"
        try:
            result = await self.file_access_manager.list_directory(dirpath, recursive)
            return self.file_access_manager.format_list_for_prompt(result)
        except Exception as e:
            logger.warning(f"[AgenticSearch] File list failed: {e}")
            return f"[File list error: {e}]"

    def _format_file_context(
        self, round_num: int, operation: str, content: str
    ) -> str:
        """Format file access results for accumulated context."""
        return (
            f"[FILE ACCESS — Round {round_num}]\n"
            f"Operation: {operation}\n"
            f"Result:\n{content}"
        )

    def _is_low_quality_result(
        self,
        search_result: Any,
        query: str
    ) -> Tuple[bool, str]:
        """
        Check if search results are too weak to be useful.

        Args:
            search_result: WebSearchResult from web_search_manager
            query: The search query that was executed

        Returns:
            Tuple of (is_low_quality, issue_description)
        """
        # Handle WebSearchResult - access the .pages attribute
        pages = getattr(search_result, 'pages', None) or []

        if not pages:
            return True, "no results"
        if len(pages) == 1:
            return True, "only 1 result"

        # Extract query terms (filter stop words)
        query_terms = [w for w in query.lower().split() if w not in _STOP_WORDS]

        if not query_terms:
            return False, "ok"

        # Get top result content (check first 2000 chars for speed)
        top = pages[0]
        top_content = (
            getattr(top, 'content', '') or
            getattr(top, 'snippet', '') or
            getattr(top, 'text', '') or
            ''
        ).lower()[:2000]

        # Fast substring check instead of set intersection
        matches = sum(1 for term in query_terms if term in top_content)

        if matches < len(query_terms) * 0.3:
            return True, "results don't match query terms"

        return False, "ok"

    def _generate_relaxation_suggestion(self, query: str) -> str:
        """
        Generate a suggestion for how to relax/broaden the query.

        Args:
            query: The original search query

        Returns:
            A suggestion string for query reformulation
        """
        # Check for version numbers (e.g., "3.12", "v2.0.1")
        if _VERSION_PATTERN.search(query):
            return "Remove version numbers and try a more general query"

        # Check for year/date specifics
        if _YEAR_PATTERN.search(query):
            return "Remove year specifics or try a broader time range"

        # Check for very long queries (likely too specific)
        if query.count(' ') > 5:  # Faster than split + len
            return "Simplify to core subject + 1-2 keywords"

        # Check for quoted exact phrases
        if '"' in query:
            return "Remove exact phrase quotes and search for individual terms"

        # Check for technical jargon patterns
        if _ERROR_PATTERN.search(query):
            return "Try searching for the error message or symptom instead of the context"

        # Default suggestion
        return "Try broader category terms or synonyms"

    async def _execute_wolfram(self, query: str) -> str:
        """
        Execute Wolfram Alpha query with fallback to web search.

        Args:
            query: The computation query

        Returns:
            Formatted result string for context
        """
        if not self.wolfram_manager:
            logger.warning("[AgenticSearch] Wolfram Alpha not configured, falling back to web search")
            # Fallback to web search for computation explanation
            result = await self._execute_search([f"{query} calculation explanation"])
            return await self._compress_results(result)

        result = await self.wolfram_manager.query(query)

        if result.success:
            formatted = self.wolfram_manager.format_for_prompt(result)
            logger.info(
                f"[AgenticSearch] Wolfram query '{query[:40]}...' succeeded in {result.execution_time:.2f}s"
            )
            return formatted

        # Fallback to web search on failure
        logger.warning(f"[AgenticSearch] Wolfram failed ({result.error}), falling back to web search")
        fallback_result = await self._execute_search([f"{query} explanation solution"])
        return await self._compress_results(fallback_result)
