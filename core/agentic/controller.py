"""
Agentic Search Controller Module

Contract:
    - Provides AgenticSearchController for multi-round search loops
    - Manages ReAct cycle: Think → Search → Observe → Repeat
    - Emits ProgressEvent for UI updates
    - Enforces max_rounds limit (default 5)
    - Compresses search results to fit context budget
    - Falls back gracefully on search/API failures

Dependencies:
    - models.model_manager.ModelManager (for LLM generation)
    - knowledge.web_search_manager.WebSearchManager (for searches)
    - core.prompt.token_manager.TokenManager (for budget enforcement)

Public Interface:
    - AgenticSearchController.run_agentic_search() -> AsyncGenerator[ProgressEvent|str]
    - AgenticSearchController.detect_protocol() -> SearchProtocol
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, TYPE_CHECKING

from core.agentic.types import (
    AgentState,
    AgenticSearchSession,
    ProgressEvent,
    SearchDecision,
    SearchProtocol,
    SearchRequest,
    SearchRound,
)
from core.agentic.protocols import (
    detect_protocol,
    get_protocol_handler,
    BaseProtocolHandler,
)

if TYPE_CHECKING:
    from models.model_manager import ModelManager
    from knowledge.web_search_manager import WebSearchManager, WebSearchResult
    from core.prompt.token_manager import TokenManager

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MAX_ROUNDS = 5
DEFAULT_CONTEXT_BUDGET_TOKENS = 8000
DEFAULT_COMPRESSION_MAX_TOKENS = 1500
DEFAULT_COMPRESSION_MODEL = "gpt-4o-mini"


class AgenticSearchController:
    """
    Controls the ReAct-style agentic search loop.

    This controller manages multi-round search sessions where the LLM can
    iteratively gather information until it has enough to provide a
    comprehensive answer.

    The first search is automatic (triggered by the existing LLM-first trigger).
    Subsequent searches are model-driven via tool calls or XML markers.
    """

    def __init__(
        self,
        model_manager: "ModelManager",
        web_search_manager: "WebSearchManager",
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
            token_manager: Optional token counter for budget enforcement
            max_rounds: Maximum search rounds allowed (default 5)
            context_budget_tokens: Token budget for accumulated context
            compression_model: Model to use for result compression
        """
        self.model_manager = model_manager
        self.web_search_manager = web_search_manager
        self.token_manager = token_manager
        self.max_rounds = max_rounds
        self.context_budget_tokens = context_budget_tokens
        self.compression_model = compression_model

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

        # Get protocol handler
        handler = get_protocol_handler(protocol)

        # Augment system prompt for agentic mode
        augmented_system_prompt = handler.augment_system_prompt(
            system_prompt, self.max_rounds
        )

        try:
            # === ROUND 1: Automatic search with trigger terms ===
            session.state = AgentState.SEARCHING

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

            # === ROUNDS 2-N: Model-driven iteration ===
            while session.can_continue and session.current_round <= self.max_rounds:
                session.state = AgentState.THINKING

                # Build prompt with accumulated context
                iteration_prompt = self._build_iteration_prompt(
                    query=query,
                    search_context=session.accumulated_context,
                    round_number=session.current_round
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

                    # Add to accumulated context
                    session.accumulated_context += "\n\n" + self._format_search_context(
                        session.current_round - 1,  # Already incremented by rounds.append
                        decision.search_query,
                        compressed
                    )

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

    def _build_iteration_prompt(
        self,
        query: str,
        search_context: str,
        round_number: int
    ) -> str:
        """Build prompt for iteration decision."""
        return f"""User Question: {query}

Search Results So Far:
{search_context}

You are in round {round_number} of up to {self.max_rounds} search rounds.

Based on the search results above:
1. If you have enough information to fully answer the question, signal you're done and answer.
2. If you need more specific information, request another search with a focused query.
3. Consider what's missing: different aspects, more recent data, or more specific details.

What would you like to do?"""

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
            summaries = initial_context.get('summaries', {})
            if summaries:
                recent_summaries = summaries.get('recent', [])
                semantic_summaries = summaries.get('semantic', [])
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

            # Reflections
            reflections = initial_context.get('reflections', {})
            if reflections:
                recent_refs = reflections.get('recent', [])
                if recent_refs:
                    ref_text = self._format_reflections(recent_refs)
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

        return "\n\n".join(parts)

    def _format_recent_conversations(self, conversations: List[Dict]) -> str:
        """Format recent conversations for the prompt."""
        if not conversations:
            return ""
        lines = []
        for i, conv in enumerate(conversations, 1):
            ts = conv.get('timestamp', '')
            user_msg = conv.get('query', conv.get('user', ''))
            assistant_msg = conv.get('response', conv.get('assistant', ''))
            if user_msg:
                lines.append(f"{i}) {ts}: User: {user_msg[:500]}")
                if assistant_msg:
                    lines.append(f"   Daemon: {assistant_msg[:500]}")
        return "\n".join(lines)

    def _format_memories(self, memories: List[Dict]) -> str:
        """Format memories for the prompt."""
        if not memories:
            return ""
        lines = []
        for i, mem in enumerate(memories, 1):
            ts = mem.get('timestamp', '')
            content = mem.get('content', mem.get('query', ''))
            response = mem.get('response', '')
            if content:
                lines.append(f"{i}) {ts}: {content[:400]}")
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
