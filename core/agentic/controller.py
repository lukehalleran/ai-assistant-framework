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
      (dreams, reflections, docs, summaries) if total exceeds ceiling. Recent
      conversation is framed as this session's ground truth (a contradicting web
      result must be surfaced as a conflict, not silently trusted), while still
      forbidding replies to old turns as if they were the current message.
    - Session-grounded decisions: _compute_recent_conversation_digest() builds a
      short content digest of the most recent turns (not just the inventory's
      counts), injected into every _build_iteration_prompt() so the loop won't search
      to re-derive — or contradict — a fact already settled, or ask the user to
      re-explain it. Ordering is timestamp-aware (2026-08-02: the gatherer's
      recent_conversations is NEWEST-first; the old tail-slice fed the decision
      rounds the N OLDEST turns, and via decision-answer reuse the final reply
      asked the user to re-explain 20-minute-old context twice in one day)
    - Falls back gracefully on search/API failures (partial failure: gather returns_exceptions=True)
    - Reasoning-only recovery [NEW 2026-06-14]: _generate_final_response() tracks whether any
      visible content was emitted; if the model streamed only reasoning (deepseek-v4 etc. can
      swallow the whole answer into the reasoning channel, yielding just the synthetic "<thinking>"
      marker), it closes the dangling marker and retries once via _recover_reasoning_only_response()
      → generate_once(disable_reasoning=True). Prevents the GUI "caught by the thinking filter" dead-end.
      Extended 2026-07-03: also recovers when the model dumps a literal tagged reasoning block
      (<reasoning>…</reasoning> etc.) in the CONTENT channel with no answer after it — the channel
      check can't see that case, so the assembled visible text is checked post-stream via
      ResponseParser.sanitize_for_storage() (empties → retry without native reasoning).
    - Interleaved-reasoning leak defense [NEW 2026-06-28]: _generate_final_response() streams via
      core.reasoning_stream_filter.InterleavedReasoningFilter. Reasoning models (glm-5.2 observed)
      can interleave reason → draft → reason → real answer; the old "yield every content delta" loop
      fused the discarded draft onto the answer ("synthesis system.Let me check…"). The filter holds
      the leading content run until confirmed non-draft and drops a short run cut off by resumed
      reasoning (restored at finish() if nothing replaces it). See conv 0f6d70c7.
    - Premature-done guard [NEW 2026-06-28]: the loop's done-check no longer honors <done/> on
      round 1 when nothing was gathered (no rounds, empty accumulated_context) and no answer text
      was provided. It nudges once to force real tool use first — glm-5.2 was signaling done on
      round 1 without searching, so memory-seeking queries got a promissory non-answer.
    - Decision-answer reuse [NEW 2026-07-15]: when the loop exits because the model answered
      instead of calling tools (implicit wants_answer or done + answer text), the decision
      round's text is vetted by _usable_decision_answer() (≥200 chars post-sanitize, ends at a
      sentence boundary, not promissory "let me check…" narration, no action dispatched that
      round) and, if it passes, IS the final response — the second full-context synthesis call
      is skipped (~20-30s saved). Config: agentic_search.reuse_decision_answer (default true),
      agentic_search.decision_max_tokens (default 1600, both decision paths — high enough that
      complete answers don't truncate; capped answers fail the boundary check and fall back).
      Provenance: final_prompt_hash is set to the sentinel "decision-answer-reuse".
    - Latency guards [NEW 2026-07-24]: the rounds-2-N loop is bounded two ways so a
      slow/misbehaving model can't hang the turn (observed: kimi-3 narrating tool
      intent in prose instead of emitting XML markers, ~55-60s/round, hung a turn
      ~2 min until the user hit Retry). (1) _get_model_decision() wraps each
      decision-LLM call in asyncio.wait_for(AGENTIC_ROUND_TIMEOUT_S, default 75s);
      on timeout it returns wants_answer=True so the loop exits into final
      synthesis (backstop vs. a stalled connection). (2) A wall-clock deadline
      (AGENTIC_LOOP_TIMEOUT_S, default 120s) is checked at the top of the loop;
      once exceeded no new round starts and the loop synthesizes from gathered
      context. Config: agentic_search.round_timeout_s / loop_timeout_s.
    - Sandbox lifecycle [fixed 2026-07-24]: the persistent E2B session is recycled
      in _get_sandbox_session() by cheap local checks (is_closed, then age_seconds
      vs _sandbox_session_timeout) then a backend liveness probe (session.is_alive
      → E2B is_running). Prior bugs: the age recycle read `.age` (nonexistent; dead)
      and is_closed couldn't see a server-side kill, so a dead handle was reused.
    - Provenance: computes final_prompt_hash (SHA-256[:16]) on assembled prompt

Modular Architecture (2026-05-09):
    - AgenticFormatter (core/agentic/formatters.py): Pure stateless formatting methods
      for all result types (search, memory, file, wiki, etc.)
    - ToolExecutor (core/agentic/tools.py): Dispatch routing + low-level tool execution
      for all 20 tool types (web search, wolfram, sandbox, memory, files, git stats, contacts, etc.)
    - Controller retains: orchestration loop, prompt building, model interaction,
      quality heuristics, and delegation wrappers for backward compatibility

Dependencies:
    - core.agentic.formatters.AgenticFormatter (result formatting)
    - core.agentic.tools.ToolExecutor (tool dispatch + execution)
    - utils.python_fs_guard.agent_mode (Python filesystem guard context)
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
from core.reasoning_stream_filter import InterleavedReasoningFilter
from utils.python_fs_guard import agent_mode as _fs_agent_mode
from utils.ordered_slice import oldest_first as _ordered_oldest_first

if TYPE_CHECKING:
    from models.model_manager import ModelManager
    from knowledge.web_search_manager import WebSearchManager, WebSearchResult
    from knowledge.wolfram_manager import WolframManager
    from knowledge.sandbox_manager import SandboxManager, PersistentSession, SandboxResult
    from core.prompt.token_manager import TokenManager
    from core.file_access_manager import FileAccessManager
    from core.git_stats_manager import GitStatsManager
    from core.github_manager import GitHubManager

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


# Forced write-action detection + deterministic param backfill now live in the action registry
# (core/actions/registry.py) — the single source of truth, so adding an action is one place.
# Re-exported here for the controller body and for tests that import these names.
from core.actions.registry import (  # noqa: E402
    detect_action_intent,
    backfill_params,
    _extract_issue_fields_from_query,
)


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
        github_manager: Optional["GitHubManager"] = None,
        token_manager: Optional["TokenManager"] = None,
        corpus_manager=None,
        user_profile=None,
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
            github_manager: Optional GitHub API manager for read-only repo queries
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
        self.github_manager = github_manager
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

        # Persistent sandbox session — survives across agentic runs within the
        # same conversation so variables, dataframes, and files carry over.
        # Created lazily on first sandbox use; closed on shutdown or timeout.
        self._sandbox_session = None
        self._sandbox_session_timeout = 600  # 10 minutes idle → close

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
            github_manager=github_manager,
            token_manager=token_manager,
            memory_expander=self.memory_expander,
            corpus_manager=corpus_manager,
            user_profile=user_profile,
            compression_model=compression_model,
        )

    async def _get_sandbox_session(self):
        """Get or create a persistent sandbox session."""
        # Check if existing session is still usable, recycling a stale one.
        # NOTE (2026-07-24): this block had two defects. (1) The age recycle was
        # DEAD — it checked `.age`, but PersistentSession exposes `age_seconds`,
        # so hasattr(...,'age') was always False and a long-lived session was
        # never closed here. (2) `is_closed` only reflects an explicit local
        # close(), so a sandbox E2B killed server-side (idle ~5 min / crash) read
        # as alive and the next run() failed. Now: cheap local checks first
        # (is_closed, then age_seconds), then a best-effort backend liveness
        # probe (is_alive → E2B is_running) only for a session young enough to
        # otherwise reuse — so we pay one probe per reusing turn, not per round.
        if self._sandbox_session is not None:
            _drop_reason = None
            if self._sandbox_session.is_closed:
                _drop_reason = ""  # already closed; just detach
            elif getattr(self._sandbox_session, 'age_seconds', 0) > self._sandbox_session_timeout:
                _drop_reason = "timed out"
            elif not self._sandbox_session.is_alive():
                _drop_reason = "died server-side"
            if _drop_reason is not None:
                if _drop_reason:
                    logger.info(f"[AgenticSearch] Sandbox session {_drop_reason}, recreating")
                    try:
                        await self._sandbox_session.close()
                    except Exception:
                        pass
                self._sandbox_session = None

        if self._sandbox_session is None and self.sandbox_manager:
            try:
                self._sandbox_session = await self.sandbox_manager.create_session()
                logger.info("[AgenticSearch] Created persistent sandbox session")
            except Exception as e:
                logger.warning(f"[AgenticSearch] Failed to create sandbox session: {e}")

        return self._sandbox_session

    async def close_sandbox(self):
        """Close the persistent sandbox session. Call on shutdown."""
        if self._sandbox_session and not self._sandbox_session.is_closed:
            try:
                await self._sandbox_session.close()
                logger.info("[AgenticSearch] Closed persistent sandbox session")
            except Exception as e:
                logger.warning(f"[AgenticSearch] Error closing sandbox: {e}")
            self._sandbox_session = None

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

    @staticmethod
    def _email_search_is_available() -> bool:
        """Cheap runtime capability check used when exposing native tools."""
        try:
            from core.email.service import get_email_service

            service = get_email_service()
            return any(provider.is_configured() for provider in service.providers)
        except Exception:
            return False

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
        fetch_fastpath: bool = False,
        gate_modes: Optional[List[str]] = None,
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
            fetch_fastpath: Skip model decision rounds after a substantive direct fetch.
            gate_modes: List of trigger modes ("web_search", "memory", "computation", etc.)

        Yields:
            ProgressEvent: Status updates for UI
            str: Final streamed response chunks
        """
        # Initialize session
        # Audit F21 (2026-08-31): the regenerate stash is per-turn state — a
        # turn that never reaches _generate_final_response must not let
        # regenerate_final_answer fire against the PREVIOUS turn's prompt.
        self._last_final_prompt = None
        self._last_final_system_prompt = None
        self._last_final_model = None
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
        github_available = self.github_manager is not None and self.github_manager.is_available()
        fetch_url_available = self.web_search_manager is not None and self.web_search_manager.is_available()
        email_search_available = self._email_search_is_available()
        try:
            from config.app_config import INTERNET_ACTIONS_ENABLED
            actions_available = INTERNET_ACTIONS_ENABLED
        except ImportError:
            actions_available = False
        handler = get_protocol_handler(
            protocol,
            wolfram_available=wolfram_available,
            sandbox_available=sandbox_available,
            memory_available=memory_available,
            file_access_available=file_access_available,
            git_stats_available=git_stats_available,
            github_available=github_available,
            fetch_url_available=fetch_url_available,
            actions_available=actions_available,
            email_search_available=email_search_available,
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

        # Inject internet actions availability
        try:
            from config.app_config import INTERNET_ACTIONS_ENABLED
            if INTERNET_ACTIONS_ENABLED:
                from core.actions.registry import enabled_action_types
                _action_types = ", ".join(at.value for at in enabled_action_types())
                augmented_system_prompt += (
                    "\n\n[AVAILABLE ACTIONS]\n"
                    "You can propose write actions requiring user confirmation via the propose_action tool.\n"
                    "Propose when you notice:\n"
                    "- An upcoming deadline mentioned in context or threads\n"
                    "- A follow-up the user said they'd do but hasn't yet\n"
                    "- Information that should be shared with someone mentioned in conversation\n"
                    "- The user explicitly asks you to send/create/post something\n\n"
                    f"Available action types: {_action_types}\n"
                    "Propose at most ONE logical action per turn. A request for several "
                    "calendar events is ONE batch proposal, so the user sees one complete "
                    "confirmation instead of several hidden approvals.\n\n"
                    "CRITICAL — DO NOT JUST DRAFT, AND DO NOT OVER-RESEARCH: When the user EXPLICITLY asks "
                    "you to create/file an issue, send a message, comment on a PR, or post something, your "
                    "FIRST action MUST be to call the propose_action tool — not to research. Writing the "
                    "issue/message as your text answer does NOTHING; only a propose_action call gives the "
                    "user an Approve button. For GitHub issues/PR comments the repo is AUTO-DETECTED from "
                    "the local git remote — do NOT look it up, and do NOT read files or call git_stats / "
                    "github to 'write a better body'; the title and body the user gave you are enough. "
                    "Call propose_action immediately as your first action. For a GitHub issue: "
                    "propose_action(action_type=\"github_create_issue\", subject=<title>, message=<body>)."
                )
        except ImportError:
            pass

        # Get persistent sandbox session (survives across agentic runs in the conversation)
        sandbox_session = None
        if sandbox_available:
            try:
                sandbox_session = await self._get_sandbox_session()
                if sandbox_session:
                    logger.info("[AgenticSearch] Using persistent sandbox session")
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
                session.round_telemetry.append({
                    "round": 1, "action": "fetch_url", "decision_ms": 0,
                    "tool_ms": round(fetch_duration), "timed_out": False,
                })

                # A plainly shared URL does not need a planning round after a
                # successful fetch.  Keep the normal loop as the fallback for
                # short/error-only fetches.
                try:
                    from config.app_config import AGENTIC_FETCH_FASTPATH_MIN_CHARS
                except ImportError:
                    AGENTIC_FETCH_FASTPATH_MIN_CHARS = 400
                substantive = any(
                    not isinstance(result, Exception)
                    and len(str(result)) >= AGENTIC_FETCH_FASTPATH_MIN_CHARS
                    for result in fetch_results
                )
                if fetch_fastpath and substantive:
                    session.model_signaled_done = True
                    session.fetch_fastpath_fired = True
                    session.round_telemetry[-1]["action"] = "fetch_fastpath"
                    logger.info(
                        "[AgenticSearch] Fetch fastpath: skipping decision rounds "
                        "-> direct synthesis"
                    )

                yield ProgressEvent(
                    event_type="url_fetched",
                    message=f"Fetched {len(initial_urls[:3])} URL(s)",
                    round_number=1,
                    metadata={"duration_ms": fetch_duration}
                )

            elif skip_initial_search or not initial_search_terms:
                # Skip the Round 1 *web* search. Two cases land here:
                #   1. skip_initial_search — the gate routed us to
                #      memory/knowledge/tools/computation rather than the web, so
                #      the loop chooses its own tools instead of a blind opening
                #      web query. (Not "computation-only" — memory-seeking
                #      queries land here too.)
                #   2. no seed terms — the trigger said "search" but distilled no
                #      terms. Blind-searching the raw user message verbatim is
                #      almost always low quality (filler, pronouns, no distilled
                #      intent) and once mislabelled a casual message as news, so
                #      we let the loop distill its own query instead.
                if not initial_search_terms and not skip_initial_search:
                    # Web-routed but no distilled seed terms: remember it so the
                    # loop can insist on at least one real search before accepting
                    # a tool-less first answer (answering purely from priors).
                    session._web_mode_no_seed = True
                    logger.info(
                        "[AgenticSearch] No seed terms — skipping blind verbatim "
                        "web search; loop will distill its own query"
                    )
                else:
                    logger.info("[AgenticSearch] Skipping initial web search (loop will pick tools)")
                session.accumulated_context = ""
                yield ProgressEvent(
                    event_type="thinking",
                    message="Entering Agentic Loop...",
                    round_number=1,
                    metadata={"skip_search": True}
                )
            else:
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
                session.round_telemetry.append({
                    "round": 1, "action": "web_search", "decision_ms": 0,
                    "tool_ms": round(search_duration), "timed_out": False,
                })
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

            # Compute a short digest of this session's recent turns (content, not just
            # the counts in the inventory) so the per-round decision can see what was
            # already established and avoid searching to re-derive it (or contradicting it).
            session.recent_conversation_digest = self._compute_recent_conversation_digest(
                initial_context
            )

            # Detect explicit write-action intent. If present, force the model to call
            # propose_action on the first decision round (native-tools protocol only) so
            # research-eager models don't spend every round reading code and never act.
            _forced_action = detect_action_intent(query)  # ActionType or None (from the registry)
            _force_propose_pending = _forced_action is not None
            if _forced_action:
                # Forced action rounds need more than the tiny general-purpose
                # recent-turn digest. Follow-ups such as "create the calendar
                # events" depend on the prior answer's full date table and the
                # user's intervening "day of" preference.
                session.action_context_digest = self._compute_action_context(
                    initial_context
                )
                logger.info(
                    f"[AgenticSearch] Explicit action intent ({_forced_action.value}) — forcing "
                    f"propose_action on first decision round"
                )

            # Answer text written during the decision round that ended the loop.
            # When substantive (see _usable_decision_answer), it IS the final
            # response — the second full-context synthesis call is skipped.
            _decision_answer_text: Optional[str] = None

            # One-shot: a decision-round timeout with NOTHING gathered yet on a
            # tool-triggered session dispatches the requested search
            # deterministically instead of "answering with current context"
            # (2026-08-27: an explicit "can we do a web search" turn hit the
            # 75s timeout and ended with zero tools, then spent 280s
            # synthesizing an answer it had no evidence for). A SECOND timeout
            # falls through to synthesis as before.
            _timeout_fallback_used = False

            # Wall-clock budget for the rounds-2-N loop. A slow/misbehaving model
            # (observed 2026-07-24: kimi-3 narrating tool intent in prose instead
            # of emitting XML markers, ~55-60s/round) could otherwise run every
            # round to max_rounds and hang the turn for minutes. Once exceeded,
            # stop starting new rounds and fall through to final synthesis with
            # whatever was gathered.
            from config.app_config import AGENTIC_LOOP_TIMEOUT_S
            _loop_deadline = time.monotonic() + AGENTIC_LOOP_TIMEOUT_S

            # === ROUNDS 2-N: Model-driven iteration ===
            while session.can_continue and session.current_round <= self.max_rounds:
                if time.monotonic() > _loop_deadline:
                    logger.warning(
                        f"[AgenticSearch] Loop wall-clock budget "
                        f"({AGENTIC_LOOP_TIMEOUT_S:.0f}s) exceeded after "
                        f"{len(session.rounds)} round(s) — stopping and "
                        f"synthesizing from gathered context"
                    )
                    break
                session.state = AgentState.THINKING

                # Build prompt with accumulated context
                iteration_prompt = self._build_iteration_prompt(
                    query=query,
                    search_context=session.accumulated_context,
                    round_number=session.current_round,
                    session=session
                )

                # Force propose_action once when an explicit action was requested. Forced
                # tool_choice alone isn't honored by every provider (e.g. deepseek-v4), so we
                # ALSO restrict the offered tools to just propose_action — with no research
                # tools available, the model can only propose. (Native-tools protocol only;
                # tools_override is ignored on the XML path.)
                _round_tool_choice: Any = "auto"
                _round_tools_override: Optional[List[Dict]] = None
                _round_system_prompt = augmented_system_prompt
                _round_prompt = iteration_prompt
                if _force_propose_pending:
                    from core.actions.registry import ACTION_SPECS
                    _spec = ACTION_SPECS.get(_forced_action)
                    _hint = (_spec.field_hint if _spec and _spec.field_hint else "the required fields")
                    _round_tool_choice = {"type": "function", "function": {"name": "propose_action"}}
                    _ptool = getattr(handler, "propose_action_tool", None)
                    if _ptool is not None:
                        _round_tools_override = [_ptool]
                        # Use the user's actual request as the prompt (not the generic "what tool
                        # next?" iteration prompt) so the model fills the content fields from it.
                        _round_prompt = iteration_prompt + "\n\n" + (
                            "[ACTION EXECUTION DIRECTIVE]\n"
                            f"The user asked: {query}\n\n"
                            f"Call propose_action now to do exactly this, filling in ALL content "
                            f"fields from the request and conversation context above. For several "
                            f"calendar events, use one events[] batch."
                        )
                        _round_system_prompt = augmented_system_prompt + (
                            f"\n\n[ACTION REQUIRED] The user explicitly asked you to perform a write "
                            f"action ({_forced_action.value}). Call propose_action NOW and FILL IN the "
                            f"content fields from the user's request — for this action: {_hint}. "
                            f"Do NOT leave required fields empty, and do NOT specify a repo (auto-detected)."
                        )
                    else:
                        # XML-markers protocol (2026-08-29): tool_choice/tools_override are
                        # ignored here and "propose_action" is native-tools vocabulary the
                        # model has never seen — the live forced calendar round produced
                        # NOTHING and fell through to implicit-ready. Give the model the
                        # actual marker syntax with the spec's required fields as
                        # attributes, one marker per item.
                        _round_prompt = iteration_prompt + "\n\n" + self._build_xml_action_force_prompt(
                            query, _forced_action, _spec)
                        _round_system_prompt = augmented_system_prompt + (
                            f"\n\n[ACTION REQUIRED] The user explicitly asked you to perform a "
                            f"write action ({_forced_action.value}). Emit the <action> marker(s) "
                            f"NOW exactly as instructed — for this action: {_hint}. Do NOT "
                            f"narrate or answer in prose; markers only."
                        )
                    _force_propose_pending = False  # force on this round only

                # Generate with protocol-appropriate method and record the
                # decision latency even when the model returns no tools.
                decision_started = time.monotonic()
                decisions = await self._get_model_decision(
                    prompt=_round_prompt,
                    system_prompt=_round_system_prompt,
                    model_name=model_name,
                    handler=handler,
                    session=session,
                    tool_choice=_round_tool_choice,
                    tools_override=_round_tools_override,
                )
                decision_ms = (time.monotonic() - decision_started) * 1000
                decision_timed_out = any(
                    getattr(decision, "timed_out", False) for decision in decisions
                )
                def _decision_action(decision: SearchDecision) -> str:
                    names = (
                        ("web_search", decision.wants_search),
                        ("wolfram", decision.wants_wolfram),
                        ("sandbox", decision.wants_sandbox),
                        ("memory_search", decision.wants_memory_search),
                        ("memory_expand", decision.wants_memory_expand),
                        ("file_read", decision.wants_file_read),
                        ("file_grep", decision.wants_file_grep),
                        ("file_list", decision.wants_file_list),
                        ("fetch_url", decision.wants_fetch_url),
                        ("github", decision.wants_github),
                        ("action", decision.wants_action),
                        ("pattern_scan", decision.wants_pattern_scan),
                    )
                    selected = [name for name, enabled in names if enabled]
                    return ",".join(selected) or ("answer" if decision.wants_answer else "done")
                telemetry_entry = {
                    "round": session.current_round,
                    "action": ",".join(_decision_action(decision) for decision in decisions),
                    "decision_ms": round(decision_ms),
                    "tool_ms": 0,
                    "timed_out": decision_timed_out,
                }
                session.round_telemetry.append(telemetry_entry)

                # Decision-round timeout with nothing gathered: the user's
                # request explicitly triggered the tool loop, so a stalled
                # decision call must not silently become "answer from context".
                # Substitute a deterministic search from the trigger's own
                # seed terms (or the query itself) — once. Route depends on
                # the trigger mode: web_search → web, memory → memory.
                from config.app_config import AGENTIC_TIMEOUT_TOOL_FALLBACK
                if (
                    decisions
                    and getattr(decisions[0], "timed_out", False)
                    and not session.rounds
                    and not _timeout_fallback_used
                    and AGENTIC_TIMEOUT_TOOL_FALLBACK
                ):
                    _timeout_fallback_used = True
                    _fb_terms = [
                        t.strip()
                        for t in (initial_search_terms or [])
                        if t and t.strip()
                    ][:2] or [self._fallback_terms_from_query(query)]
                    _is_memory_mode = gate_modes and "memory" in gate_modes

                    if _is_memory_mode:
                        # Memory-routed session: substitute memory search
                        logger.warning(
                            f"[AgenticSearch] Decision round timed out with zero "
                            f"tools dispatched — running memory search "
                            f"deterministically: {_fb_terms[0] if _fb_terms else 'all'}"
                        )
                        yield ProgressEvent(
                            event_type="round_start",
                            message="Model stalled — searching memory directly",
                            round_number=session.current_round,
                            metadata={"query": _fb_terms[0] if _fb_terms else query},
                        )
                        decisions = [
                            SearchDecision(
                                wants_memory_search=True,
                                memory_query=_fb_terms[0] if _fb_terms else query,
                                memory_collection="all",
                                memory_reason=(
                                    "decision-round timeout — dispatching memory search "
                                    "deterministically"
                                ),
                            )
                        ]
                    elif fetch_url_available:
                        # Web-routed session: substitute web search
                        logger.warning(
                            f"[AgenticSearch] Decision round timed out with zero "
                            f"tools dispatched — running web search "
                            f"deterministically: {_fb_terms}"
                        )
                        yield ProgressEvent(
                            event_type="round_start",
                            message="Model stalled — running the requested search directly",
                            round_number=session.current_round,
                            metadata={"terms": _fb_terms},
                        )
                        decisions = [
                            SearchDecision(
                                wants_search=True,
                                search_query=t,
                                search_reason=(
                                    "decision-round timeout — dispatching the "
                                    "explicitly requested search deterministically"
                                ),
                            )
                            for t in _fb_terms
                        ]

                # Dispatch action proposals BEFORE honoring done signal.
                # The model often sends propose_action + signal_done together;
                # if we break on done first, the action never gets dispatched.
                _action_decisions = [
                    d for d in decisions
                    if d.wants_action and d.action_type
                ]
                _action_decisions = self._coalesce_action_decisions(_action_decisions)
                if _action_decisions:
                    for _ad in _action_decisions:
                        # Backfill blank fields from the user's request for any action whose spec
                        # has a deterministic extractor (e.g. github issue title/body). Models call
                        # propose_action but unreliably leave content empty under a large context.
                        try:
                            from core.actions.types import ActionType as _AT
                            _bf = backfill_params(_AT(_ad.action_type), query)
                        except ValueError:
                            _bf = {}
                        if _bf:
                            _params = dict(_ad.action_params or {})
                            _filled = [k for k, v in _bf.items() if not _params.get(k) and v]
                            for _k in _filled:
                                _params[_k] = _bf[_k]
                            if _filled:
                                _ad.action_params = _params
                                logger.info(
                                    f"[AgenticSearch] Backfilled {_filled} from query for {_ad.action_type}"
                                )
                        _ad_round = session.current_round
                        telemetry_entry.setdefault("rounds", []).append(_ad_round)
                        _ad_result = await self._dispatch_single(
                            _ad, _ad_round, session, crisis_level, sandbox_session
                        )
                        for ev in _ad_result.start_events:
                            yield ev
                        for ev in _ad_result.end_events:
                            yield ev
                        if _ad_result.round_data is not None:
                            session.rounds.append(_ad_result.round_data)
                        if _ad_result.formatted_context:
                            self._append_accumulated(session, _ad_result.formatted_context)
                        logger.info(f"[AgenticSearch] Dispatched action before done: {_ad.action_type}")
                    # Audit F13 (2026-08-31): once an action dispatched this
                    # session, the forced-action retry must never re-arm — a
                    # later tool-less round used to re-force and produce a
                    # DUPLICATE proposal.
                    session._action_dispatched = True
                    _force_propose_pending = False

                # Web-mode-without-seed guard: the trigger routed this query to
                # the web but distilled no seed terms, so Round 1 was skipped.
                # If the model's very first decision is a tool-less answer (done
                # or implicit), it is answering from priors with zero web results
                # — nudge once to distill and run a real search first.
                _wants_tools = any(
                    not d.is_done and not d.wants_answer
                    and not (d.wants_action and d.action_type)
                    for d in decisions
                )
                if (getattr(session, '_web_mode_no_seed', False)
                        and not _wants_tools
                        and len(session.rounds) == 0
                        and not getattr(session, '_web_nudge_sent', False)):
                    session._web_nudge_sent = True
                    logger.info(
                        "[AgenticSearch] Web-routed query about to be answered "
                        "with no web search — nudging to search first"
                    )
                    session.accumulated_context += (
                        "\n\n[SYSTEM]: This request was routed here because it "
                        "needs fresh information from the web, but you have not "
                        "searched yet. Distill a focused query and call the web "
                        "search tool now, e.g.:\n"
                        "<search>your distilled query</search>\n"
                        "Then answer from the results."
                    )
                    continue  # retry this round; the model should search now

                # Check for done signal — honor it, but guard against a
                # PREMATURE done. Some models (glm-5.2 observed 2026-06-28) emit
                # <done/> on round 1 without running a single tool, so for a query
                # the gate routed to agentic search precisely because it needs
                # lookup, the loop ends having gathered nothing and the final
                # synthesis produces a useless promissory non-answer ("Let me
                # check what you've been up to…"). When done arrives before any
                # tool has run, with no context gathered and no answer text,
                # nudge once to force real tool use before accepting done.
                if any(d.is_done for d in decisions):
                    done_d = next((d for d in decisions if d.is_done), None)
                    _nothing_gathered = (
                        len(session.rounds) == 0
                        and not (session.accumulated_context or "").strip()
                    )
                    _has_answer = any((d.partial_response or "").strip() for d in decisions)
                    if (_nothing_gathered and not _has_answer
                            and not getattr(session, '_done_nudge_sent', False)):
                        session._done_nudge_sent = True
                        logger.info(
                            "[AgenticSearch] Premature done on round 1 with nothing "
                            "gathered — nudging to use tools before accepting done"
                        )
                        session.accumulated_context += (
                            "\n\n[SYSTEM]: You signaled completion but have not gathered "
                            "any information yet. Do NOT signal done before using tools — "
                            "the user's request requires looking things up. Call the "
                            "appropriate XML tool markers now, for example:\n"
                            "<memory collection=\"conversations\">synthesis system</memory>\n"
                            "<git_stats>recent commits</git_stats>\n"
                            "<github>recent activity</github>\n"
                            "Use the tools, then answer from what you find."
                        )
                        continue  # retry this round; the model should call tools now
                    session.model_signaled_done = True
                    session.done_reason = done_d.done_reason if done_d else None
                    # Answer text alongside done is a reuse candidate — unless an
                    # action was dispatched this round (its result arrived AFTER
                    # the text was written, so the text can't reflect it).
                    if not _action_decisions:
                        _decision_answer_text = "".join(
                            d.partial_response or "" for d in decisions
                        ).strip() or None
                    logger.info(f"[AgenticSearch] Model signaled done: {session.done_reason}")
                    break

                # Filter to actual tool requests (exclude already-dispatched actions)
                tool_decisions = [
                    d for d in decisions
                    if not d.is_done and not d.wants_answer
                    and not (d.wants_action and d.action_type)  # already handled above
                ]
                if not tool_decisions:
                    # If this is round 1 and the model just narrated instead of
                    # using tools, retry once with an explicit nudge. Some models
                    # (e.g. DeepSeek) emit plain text describing tool calls instead
                    # of the actual XML markers on the first attempt.
                    if (len(session.rounds) == 0
                            and not getattr(session, '_tool_nudge_sent', False)):
                        # Check if the response mentions tools/actions
                        _answer_text = ''
                        for d in decisions:
                            if d.partial_response:
                                _answer_text += d.partial_response
                        _tool_mentions = any(w in _answer_text.lower() for w in (
                            'github', 'git_stats', 'search', 'let me pull',
                            'let me grab', 'let me run', 'let me check',
                            'list_repos', 'commits', 'lines added',
                            'propose_action', 'send_email', 'send email',
                            'send_telegram', 'send_discord',
                        ))
                        if _tool_mentions:
                            logger.info(
                                "[AgenticSearch] Model narrated tool intent without "
                                "using XML markers — retrying with nudge"
                            )
                            session._tool_nudge_sent = True
                            # Append the response + nudge to get the model to
                            # actually emit XML markers this time
                            _nudge = (
                                "\n\n[SYSTEM]: You described what tools you would use "
                                "but did NOT actually call them. You MUST use the XML "
                                "tool markers to execute tools. For example:\n"
                                "<github>open issues</github>\n"
                                "<git_stats>commits this week</git_stats>\n"
                                "<memory collection=\"facts\">user github</memory>\n"
                                "<action type=\"send_email\" recipient=\"user@example.com\" "
                                "reason=\"user asked\">message body</action>\n"
                                "Do NOT describe what you will do — just call the tools "
                                "now using the XML format above."
                            )
                            session.accumulated_context += (
                                f"\n\n[Your previous response (NOT executed)]:\n"
                                f"{_answer_text[:500]}\n{_nudge}"
                            )
                            continue  # Retry the loop iteration

                    # Forced-action retry (2026-08-29): the user explicitly
                    # requested a write action, the forced round produced no
                    # action marker, and nothing else was gathered — silence
                    # here must not become "ready to answer". Re-arm the force
                    # (now protocol-aware) and retry exactly once.
                    if (_forced_action is not None and not _action_decisions
                            and not getattr(session, '_action_dispatched', False)
                            and not getattr(session, '_action_force_retry_sent', False)):
                        session._action_force_retry_sent = True
                        _force_propose_pending = True
                        logger.info(
                            "[AgenticSearch] Forced action round produced no "
                            "action marker — retrying once"
                        )
                        continue

                    if not _action_decisions:
                        _decision_answer_text = "".join(
                            d.partial_response or "" for d in decisions
                        ).strip() or None
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

                tool_started = time.monotonic()
                tasks = [
                    self._dispatch_single(
                        d, base_round + i, session, crisis_level, sandbox_session
                    )
                    for i, d in enumerate(tool_decisions)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                telemetry_entry["tool_ms"] = round((time.monotonic() - tool_started) * 1000)
                # Audit F32 (2026-08-31): parallel rounds are numbered
                # base_round+i — record them so the provenance join can match
                # (it used to match only entry["round"], losing decision_ms
                # for every round after the first of a multi-tool iteration).
                telemetry_entry.setdefault("rounds", []).extend(
                    base_round + i for i in range(len(tool_decisions)))

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

            # Reuse a substantive decision-round answer instead of paying a
            # second full-context synthesis call (the observed pattern: a 32s
            # decision call whose answer text was discarded, followed by a 24s
            # re-generation of essentially the same answer).
            from config.app_config import AGENTIC_REUSE_DECISION_ANSWER
            _reused_answer = (
                self._usable_decision_answer(_decision_answer_text)
                if (AGENTIC_REUSE_DECISION_ANSWER and _decision_answer_text)
                else None
            )
            if _reused_answer:
                session.decision_answer_reuse_fired = True
                session.final_prompt_hash = "decision-answer-reuse"
                logger.info(
                    f"[AgenticSearch] Reusing decision-round answer "
                    f"({len(_reused_answer)} chars) — final synthesis call skipped"
                )
                yield _reused_answer
            else:
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
            # Sandbox session is persistent — do NOT close it here.
            # It will be reused across agentic runs within the conversation.
            # Cleanup happens via close_sandbox() at shutdown or on timeout.
            pass

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

        Runs inside agent_mode() so Python filesystem guards are active —
        destructive operations on protected repo paths will raise PermissionError.
        """
        with _fs_agent_mode():
            return await self._dispatch_single_inner(
                decision, round_number, session, crisis_level, sandbox_session
            )

    async def _dispatch_single_inner(self, decision, round_number, session, crisis_level, sandbox_session):
        """Inner dispatch — iterates the SHARED DISPATCH_TABLE (core.agentic.tools) so this router
        cannot drift from ToolExecutor.dispatch_single. Each handler resolves to the controller's
        own method if it defines one (preserving test-mockability), otherwise to the ToolExecutor's.
        Always runs under agent_mode() context.
        """
        from core.agentic.tools import DISPATCH_TABLE, reroute_url_search
        decision = reroute_url_search(decision)
        for predicate, handler_name, arg_builder in DISPATCH_TABLE:
            if predicate(decision):
                handler = getattr(self, handler_name, None) or getattr(self._tool_executor, handler_name)
                return await handler(*arg_builder(decision, round_number, crisis_level, sandbox_session))
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
        session: AgenticSearchSession,
        tool_choice: Any = "auto",
        tools_override: Optional[List[Dict]] = None,
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
        # Per-round backstop against a stalled connection / hung provider call.
        # Generous (default 75s) so it never cuts a legitimate full-length
        # decision round; on timeout we answer with whatever context we have.
        from config.app_config import AGENTIC_ROUND_TIMEOUT_S
        try:
            if session.protocol == SearchProtocol.NATIVE_TOOLS:
                # Use tool calling. tools_override lets a caller restrict the tool set
                # (e.g. to just propose_action) so research-eager models can't wander.
                _tools = tools_override if tools_override is not None else handler.get_tools()
                response = await asyncio.wait_for(
                    self._generate_with_tools(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model_name=model_name,
                        tools=_tools,
                        tool_choice=tool_choice,
                    ),
                    timeout=AGENTIC_ROUND_TIMEOUT_S,
                )
            else:
                # Use standard generation for XML markers.
                # IMPORTANT: Use a dedicated non-reasoning call for the decision
                # phase. Models with native reasoning (DeepSeek) burn the token
                # budget on chain-of-thought, leaving nothing for XML markers.
                # The decision phase just needs to emit tool tags, not reason.
                response = await asyncio.wait_for(
                    self._generate_decision_no_reasoning(
                        prompt=prompt,
                        model_name=model_name,
                        system_prompt=system_prompt,
                    ),
                    timeout=AGENTIC_ROUND_TIMEOUT_S,
                )

            return handler.parse_response(response)

        except asyncio.TimeoutError:
            logger.warning(
                f"[AgenticSearch] Decision generation timed out after "
                f"{AGENTIC_ROUND_TIMEOUT_S:.0f}s"
            )
            # Marked timed_out so the loop can distinguish a stalled decision
            # call from the model's own ready-to-answer signal: on a
            # tool-triggered session with nothing gathered yet, the loop
            # dispatches the requested tool deterministically instead of
            # answering with current context.
            return [SearchDecision(wants_answer=True, timed_out=True)]
        except Exception as e:
            logger.error(f"[AgenticSearch] Decision generation failed: {e}")
            # On error, signal to answer with current context
            return [SearchDecision(wants_answer=True)]

    async def _generate_decision_no_reasoning(
        self,
        prompt: str,
        model_name: str,
        system_prompt: str,
    ) -> str:
        """Generate a decision response WITHOUT native reasoning.

        For the agentic iteration phase, models like DeepSeek burn their
        token budget on chain-of-thought reasoning, leaving no room for
        actual XML tool markers. This method bypasses reasoning and uses
        a higher token limit so the model can emit tool tags directly.
        """
        target_model = model_name or self.model_manager.active_model_name
        full_model = self.model_manager.api_models.get(target_model, target_model)

        if self.model_manager.async_client is None:
            return ""

        messages = [
            # Strip the prompt-cache breakpoint marker — this path builds the
            # system message directly (no cache split), so the marker must not
            # reach the model. No-op when absent.
            {"role": "system", "content": self.model_manager._strip_cache_breakpoint(system_prompt)},
            {"role": "user", "content": prompt},
        ]

        from config.app_config import AGENTIC_DECISION_MAX_TOKENS
        create_kwargs = dict(
            model=full_model,
            messages=messages,
            max_tokens=AGENTIC_DECISION_MAX_TOKENS,
            temperature=0.3,
            stream=False,
        )
        # Explicitly do NOT add reasoning params

        try:
            response = await self.model_manager.async_client.chat.completions.create(
                **create_kwargs
            )
            if response and response.choices:
                return response.choices[0].message.content or ""
            return ""
        except Exception as e:
            logger.error(f"[AgenticSearch] Decision generation (no-reasoning) failed: {e}")
            return ""

    async def _generate_with_tools(
        self,
        prompt: str,
        system_prompt: str,
        model_name: str,
        tools: List[Dict],
        tool_choice: Any = "auto",
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
            from config.app_config import AGENTIC_DECISION_MAX_TOKENS
            return await self.model_manager.generate_once_with_tools(
                prompt=prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=AGENTIC_DECISION_MAX_TOKENS,
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

    # Promissory tool-intent phrasing marks a decision-round text as a PLAN,
    # not an answer — it must go through the real synthesis call. 2026-08-28
    # live failure: "The first round of results missed the mark — … Let me aim
    # at the actual research on …" shipped (and stored) as the FINAL response.
    # It defeated the old guard twice: the check scanned only the first 150
    # chars (the "Let me…" sat just past the window) and 'aim' wasn't in the
    # substring verb list. Now: a word-bounded regex over the WHOLE text —
    # a plan sentence anywhere means the model didn't finish; rejecting only
    # costs the synthesis-call latency, never correctness. "let me know" and
    # bare "I'll" without a tool-intent verb deliberately do NOT match.
    _PROMISSORY_RE = re.compile(
        r"\b(?:let\s+me|i'?ll|i\s+will|i'?m\s+going\s+to|gonna)\s+"
        r"(?:pull|grab|run|re-?run|check|search|re-?search|query|look|aim|"
        r"dig|find|fetch|try|retry|refine|adjust|narrow|broaden|redo|"
        r"verify|target)\b",
        re.IGNORECASE,
    )
    # Loop-meta narration about the quality of prior tool ROUNDS ("The first
    # round of results missed the mark…") — commentary on the search process,
    # not an answer to the user. Head-anchored: real answers can mention
    # "results" later in the body, but round-postmortems open with it.
    _LOOP_META_RE = re.compile(
        r"\b(?:first|second|third|next|another|last|that|this)\s+round\s+of\b"
        r"|\bmissed\s+the\s+mark\b"
        r"|\bresults?\s+(?:missed|didn'?t\s+(?:return|match|help)|came\s+back\s+empty)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _build_xml_action_force_prompt(query: str, forced_action, spec) -> str:
        """Forced-round prompt for the XML-markers protocol: a concrete
        <action> example whose attributes are the spec's required/optional
        fields, one marker per item (a calendar request can carry several
        events — each is its own marker)."""
        _fields = list(getattr(spec, "required", ()) or ())
        _attr_example = " ".join(f'{f}="<{f}>"' for f in _fields) or 'recipient="<who>"'
        _type = forced_action.value
        _calendar_hint = ""
        if _type == "calendar_update_event":
            # The example must SHOW the change fields (2026-09-01 live: the
            # required-only example taught summary+date and the model emitted
            # a changeless update that could only fail after approval).
            _attr_example += (' new_start_time="<new-start>" new_end_time="<new-end>"')
            _calendar_hint = (
                " summary + date (YYYY-MM-DD) identify the EXISTING event. Put the "
                "CHANGES in new_* fields: new_start_time and new_end_time TOGETHER "
                "as the user's local wall-clock times WITHOUT a UTC offset (e.g. "
                "2026-09-09T13:00:00); new_summary/new_description/new_location as "
                "needed. At least one new_* field is required."
            )
        if _type == "calendar_delete_event":
            _calendar_hint = (
                " summary + date (YYYY-MM-DD) identify the EXISTING event to "
                "remove — exactly those two fields, nothing else is needed."
            )
        if _type == "calendar_create_event":
            _attr_example += ' all_day="<true-or-false>" time_zone="<IANA-timezone-if-timed>"'
            # Timezone rule (2026-09-01 live: "1 PM" with no source timezone
            # was emitted as 13:00 ET because the old hint's only example was
            # "ET = America/New_York" — the executed Google event landed an
            # hour early on the user's Central calendar). Local is the
            # DEFAULT; a source-named zone is the exception.
            _calendar_hint = (
                " If the user selected day-of/all-day entries, set all_day=\"true\", "
                "start_time to YYYY-MM-DD, and end_time to the NEXT date because "
                "Google's all-day end is exclusive. For timed events, times with NO "
                "explicit timezone in the request or source material are the USER'S "
                "LOCAL time — write them WITHOUT a UTC offset and omit time_zone. "
                "Only when the source material explicitly names a zone (e.g. a "
                "syllabus stating ET = America/New_York) set time_zone to that IANA "
                "zone; never silently reinterpret a stated zone as local."
            )
        return (
            f"The user asked: {query}\n\n"
            f"Perform exactly this request by emitting one or more <action> "
            f"markers — nothing else, no prose. Syntax (fill every field from "
            f"the request and the conversation context above):\n"
            f'<action type="{_type}" {_attr_example} reason="user asked">optional details</action>\n'
            f"Timed datetimes are the user's LOCAL wall-clock time in ISO 8601 "
            f"WITHOUT a UTC offset (e.g. 2026-09-13T23:59:00) unless the source "
            f"names a zone.{_calendar_hint} If the request "
            f"covers multiple items (several events, several messages), emit one "
            f"<action> marker per item, each with ALL fields filled."
        )

    @staticmethod
    def _coalesce_action_decisions(decisions: List[SearchDecision]) -> List[SearchDecision]:
        """Collapse several calendar-event calls into one approval proposal.

        The pending store exposes one proposal ID to the UI/API and defaults
        to five entries, while a course schedule can easily contain seven or
        more deadlines. Keeping one decision per event therefore made later
        events unreachable (and overflowed the store). A calendar batch is one
        user-authorized logical action and retains every event in params.events.
        """
        if len(decisions) < 2:
            return decisions
        calendar = [
            d for d in decisions
            if str(getattr(d.action_type, "value", d.action_type) or "")
            == "calendar_create_event"
        ]
        if len(calendar) < 2:
            return decisions

        events: List[Dict[str, Any]] = []
        seen = set()
        for decision in calendar:
            params = dict(decision.action_params or {})
            # Flatten a native events[] call if one appears alongside other
            # calendar decisions; XML produces one marker per item.
            candidates = params.get("events") if isinstance(params.get("events"), list) else [params]
            for event in candidates:
                if not isinstance(event, dict):
                    continue
                clean = {k: v for k, v in event.items() if k != "events"}
                key = (
                    str(clean.get("summary", "")),
                    str(clean.get("start_time", "")),
                    str(clean.get("end_time", "")),
                )
                if key in seen:
                    continue
                seen.add(key)
                events.append(clean)

        if len(events) < 2:
            # Either everything was a duplicate or malformed; preserve the
            # first actual decision and drop duplicate calendar calls.
            first = calendar[0]
            return [d for d in decisions if d is first or d not in calendar]

        batch = SearchDecision(
            wants_action=True,
            action_type="calendar_create_event",
            action_params={"events": events},
            action_summary=f"calendar_create_event: {len(events)} events",
            action_reason=next(
                (d.action_reason for d in calendar if d.action_reason),
                "User requested multiple calendar events",
            ),
        )
        output: List[SearchDecision] = []
        inserted = False
        for decision in decisions:
            if decision in calendar:
                if not inserted:
                    output.append(batch)
                    inserted = True
                continue
            output.append(decision)
        return output

    def narration_shaped_final(self, text: str) -> bool:
        """Is a FINAL synthesis output mid-loop narration instead of an
        answer? (2026-08-29: the synthesis call shipped 'let me grab the full
        text back out of memory…' as the final reply — the 08-28 promissory
        guards only covered the decision-answer REUSE path.) Short promissory
        text or a loop-meta opener qualifies; long substantive answers that
        merely contain 'let me check' mid-prose do not."""
        t = (text or "").strip()
        if not t:
            return False
        if len(t) < 600 and self._PROMISSORY_RE.search(t):
            return True
        if self._LOOP_META_RE.search(t[:200]):
            return True
        return False

    async def regenerate_final_answer(self) -> Optional[str]:
        """One bounded no-reasoning retry of the final synthesis after a
        narration-shaped output. Returns vetted text or None (caller keeps
        the original). Uses the stashed final prompt from the last
        _generate_final_response call."""
        final_prompt = getattr(self, "_last_final_prompt", None)
        if not final_prompt:
            return None
        directive = (
            "\n\n[SYSTEM]: Your previous output described what you were about "
            "to do instead of answering. All tool rounds are CLOSED — no more "
            "searching, retrieving, or checking is possible. Using ONLY the "
            "context above, write the complete final answer to the user's "
            "request now (include the concrete results — dates, catalog, "
            "outcomes — plus anything you could not do and why). Never "
            "describe an action you are about to take."
        )
        try:
            recovered = await self.model_manager.generate_once(
                prompt=final_prompt + directive,
                model_name=getattr(self, "_last_final_model", None),
                system_prompt=getattr(self, "_last_final_system_prompt", None) or "",
                max_tokens=8192,
                disable_reasoning=True,
            )
        except Exception as e:
            logger.error(f"[AgenticSearch] Narration recovery failed: {e}")
            return None
        from core.response_parser import ResponseParser
        recovered = (ResponseParser.sanitize_for_storage(recovered or "") or "").strip()
        if len(recovered) < 200 or self.narration_shaped_final(recovered):
            logger.warning(
                "[AgenticSearch] Narration recovery produced unusable output "
                f"({len(recovered)} chars) — keeping original")
            return None
        logger.info(
            f"[AgenticSearch] Recovered narration-shaped final response "
            f"({len(recovered)} chars via no-reasoning retry)")
        return recovered

    def _usable_decision_answer(self, text: str) -> Optional[str]:
        """Vet a decision-round answer for reuse as the final response.

        Returns the sanitized text when it is a complete, substantive answer,
        or None to fall back to the full synthesis call. Guards:
        - substance: ≥ 200 chars after reasoning-tag sanitization (short
          fragments and "Ok."-style stubs re-generate instead)
        - truncation: must end at a sentence/formatting boundary — the
          decision call has a token cap and finish_reason is not surfaced,
          so a mid-sentence ending is treated as capped output
        - narration: promissory openers ("Let me check…") are plans the
          model failed to execute, never final answers
        """
        from core.response_parser import ResponseParser
        candidate = (text or "").strip()
        if len(candidate) < 200:
            return None
        sanitized = (ResponseParser.sanitize_for_storage(candidate) or "").strip()
        if len(sanitized) < 200:
            return None
        if sanitized[-1] not in '.!?"\')]}`*:;…”’':
            logger.info(
                "[AgenticSearch] Decision answer looks truncated "
                "(no terminal punctuation) — falling back to synthesis call"
            )
            return None
        if self._PROMISSORY_RE.search(sanitized):
            logger.info(
                "[AgenticSearch] Decision answer contains promissory "
                "tool-intent phrasing — plan, not answer; falling back to "
                "synthesis call"
            )
            return None
        if self._LOOP_META_RE.search(sanitized[:200]):
            logger.info(
                "[AgenticSearch] Decision answer opens with loop-meta "
                "narration about prior rounds — falling back to synthesis call"
            )
            return None
        return sanitized

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

        # Stash for regenerate_final_answer (narration-shaped final recovery)
        self._last_final_prompt = final_prompt
        self._last_final_system_prompt = system_prompt
        self._last_final_model = model_name

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
                max_tokens=8192,
                images=_images,
            )

            # Handle different return types. The InterleavedReasoningFilter
            # suppresses reasoning-only chunks (emitting <thinking>/</thinking>
            # markers) AND defends against interleaved drafts: glm-5.2 etc. can
            # stream reason → draft → reason → real answer, which the old
            # "yield every content delta" loop fused into a leak like
            # "synthesis system.Let me check…". See core/reasoning_stream_filter.py.
            _rfilter = InterleavedReasoningFilter()
            _visible_parts: List[str] = []
            if hasattr(stream, '__aiter__'):
                # It's an async iterator (OpenAI stream)
                async for chunk in stream:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        delta_reasoning = getattr(delta, 'reasoning_content', '') or getattr(delta, 'reasoning', '') or ''
                        delta_content = getattr(delta, 'content', '') or ''
                        for _kind, _text in _rfilter.feed(delta_reasoning, delta_content):
                            _visible_parts.append(_text)
                            yield _text
                    elif isinstance(chunk, str):
                        if chunk:
                            for _kind, _text in _rfilter.feed('', chunk):
                                _visible_parts.append(_text)
                                yield _text

                # Flush any content the filter is still holding back.
                for _kind, _text in _rfilter.finish():
                    _visible_parts.append(_text)
                    yield _text

                # Reasoning-only recovery: the model streamed reasoning but never
                # any visible content (deepseek-v4 etc. occasionally swallow the
                # entire answer into the reasoning channel, leaving content empty).
                # Without this, the loop returns just "<thinking>" and the GUI shows
                # the "caught by the thinking filter" dead-end. Close the dangling
                # marker and retry once with native reasoning disabled so the model
                # emits its answer as normal content.
                if _rfilter.reasoning_seen and not _rfilter.content_emitted:
                    if _rfilter.in_reasoning:
                        yield "</thinking>"
                    logger.warning(
                        "[AgenticSearch] Final response was reasoning-only (no content); "
                        "retrying without native reasoning"
                    )
                    async for _rc in self._recover_reasoning_only_response(
                        final_prompt, model_name, system_prompt
                    ):
                        if _rc:
                            yield _rc
                else:
                    # Literal-tag reasoning-only leak: some models dump their whole
                    # chain of thought as <reasoning>…</reasoning> in the CONTENT
                    # channel (so content_emitted is True and the channel-based check
                    # above can't see it) and then stop without an answer. If the
                    # assembled visible text sanitizes down to nothing, recover the
                    # same way. (Observed 2026-07-03: stored response was one raw
                    # <reasoning> block.)
                    from core.response_parser import ResponseParser
                    _assembled = "".join(_visible_parts)
                    if _assembled.strip() and not ResponseParser.sanitize_for_storage(_assembled):
                        logger.warning(
                            "[AgenticSearch] Final response was a literal tagged "
                            "reasoning block with no answer; retrying without "
                            "native reasoning"
                        )
                        async for _rc in self._recover_reasoning_only_response(
                            final_prompt, model_name, system_prompt
                        ):
                            if _rc:
                                yield _rc
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

    async def _recover_reasoning_only_response(
        self, final_prompt: str, model_name: str, system_prompt: str
    ) -> AsyncGenerator[str, None]:
        """Recover an answer when the model returned reasoning but zero content.

        Retries the final synthesis once (non-streaming) with native reasoning
        disabled, forcing reasoning models (deepseek-v4 etc.) to emit their answer
        as normal content. Yields the recovered text, or nothing if recovery also
        fails — the caller already closed the dangling <thinking> marker, so a
        no-op leaves a clean (empty) stream for the GUI fallback to handle.
        """
        try:
            recovered = await self.model_manager.generate_once(
                prompt=final_prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                max_tokens=8192,
                disable_reasoning=True,
            )
        except Exception as e:
            logger.error(f"[AgenticSearch] Reasoning-only recovery failed: {e}")
            return
        recovered = (recovered or "").strip()
        if recovered:
            logger.info(f"[AgenticSearch] Recovered {len(recovered)} chars via no-reasoning retry")
            yield recovered
        else:
            logger.warning("[AgenticSearch] Reasoning-only recovery produced no content")

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

    # Decision-phase digest budget: a few of the most recent turns, hard-truncated.
    # The inventory only reports counts; this gives the loop the actual content so it
    # can avoid searching to re-derive an in-session fact (or contradicting one).
    _DIGEST_MAX_TURNS = 4
    _DIGEST_MSG_CHARS = 220
    _ACTION_CONTEXT_MAX_TURNS = 3
    _ACTION_CONTEXT_MSG_CHARS = 2200
    _ACTION_CONTEXT_TOTAL_CHARS = 7000

    @staticmethod
    def _ordered_recent_conversations(
        initial_context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return recent conversation turns oldest-first, newest last."""
        if not initial_context:
            return []
        recent = initial_context.get('recent_conversations', []) or []
        if not recent:
            return []

        def _conv_ts(conv):
            raw = str(conv.get('timestamp', '') or '')
            try:
                return datetime.fromisoformat(raw.replace(' ', 'T', 1)[:26])
            except (ValueError, TypeError):
                return None

        ordered = list(recent)
        stamps = [_conv_ts(c) for c in ordered]
        if all(s is not None for s in stamps):
            # utils.ordered_slice.oldest_first (single source of truth for
            # "sort by timestamp") — same semantics as the hand-rolled
            # sorted(zip(...)) this replaced, since every stamp parses here.
            return _ordered_oldest_first(ordered, _conv_ts)
        # No generic per-item fallback applies when timestamps are MISSING
        # (not merely unparseable-per-item): the gatherer's contract in that
        # case is "the whole list is newest-first", so the correct recovery
        # is a full reversal, not treating unstamped items as individually
        # "oldest" (utils.ordered_slice's default) — pinned by
        # test_agentic_digest_order.py::test_unparseable_timestamps_assume_newest_first.
        return list(reversed(ordered))

    @staticmethod
    def _head_tail(text: str, limit: int) -> str:
        """Bound text without dropping the closing question/preference cue."""
        value = (text or "").strip()
        if len(value) <= limit:
            return value
        head = max(1, limit // 2)
        tail = max(1, limit - head)
        return value[:head] + "\n[…snipped…]\n" + value[-tail:]

    def _compute_action_context(
        self, initial_context: Optional[Dict[str, Any]],
    ) -> str:
        """Richer recent-turn context used only by explicit write actions.

        General decision rounds intentionally use a tiny digest. Write-action
        follow-ups are different: exact dates, recipients, draft bodies, and
        the assistant's final choice question often live after character 220.
        """
        ordered = self._ordered_recent_conversations(initial_context)
        if not ordered:
            return ""
        lines: List[str] = []
        for conv in ordered[-self._ACTION_CONTEXT_MAX_TURNS:]:
            user_msg = conv.get('query', conv.get('user', '')) or ''
            assistant_msg = conv.get('response', conv.get('assistant', '')) or ''
            if user_msg:
                lines.append(
                    "User: " + self._head_tail(
                        str(user_msg), self._ACTION_CONTEXT_MSG_CHARS
                    )
                )
            if assistant_msg:
                lines.append(
                    "Daemon: " + self._head_tail(
                        str(assistant_msg), self._ACTION_CONTEXT_MSG_CHARS
                    )
                )
        rendered = "\n\n".join(lines)
        rendered = self._head_tail(rendered, self._ACTION_CONTEXT_TOTAL_CHARS)
        return (
            "[ACTION CONTEXT — AUTHORITATIVE EARLIER TURNS]\n"
            "Resolve pronouns, selected options, event details, recipients, and "
            "dates from these turns. Do not ask again for a choice already answered.\n"
            + rendered
        )

    def _compute_recent_conversation_digest(
        self, initial_context: Optional[Dict[str, Any]]
    ) -> str:
        """Build a short, hard-truncated digest of this session's recent turns.

        Returns "" when there are no recent conversations. Kept deliberately
        small (last few turns, each message clipped) because it is added to
        every decision-round prompt.
        """
        if not initial_context:
            return ""
        ordered = self._ordered_recent_conversations(initial_context)
        if not ordered:
            return ""
        tail = ordered[-self._DIGEST_MAX_TURNS:]
        lines = []
        for conv in tail:
            user_msg = (conv.get('query', conv.get('user', '')) or '').strip()
            assistant_msg = (conv.get('response', conv.get('assistant', '')) or '').strip()
            if not user_msg:
                continue
            lines.append(f"- User: {user_msg[:self._DIGEST_MSG_CHARS]}")
            if assistant_msg:
                lines.append(f"  Daemon: {assistant_msg[:self._DIGEST_MSG_CHARS]}")
        if not lines:
            return ""

        header = (
            "[RECENT CONVERSATION — EARLIER TURNS] What was said in the most recent "
            "turns, including just-ended sessions (most recent last). Pronouns and "
            "references like \"that\"/\"earlier\" in the user's question usually point "
            "here. If these already establish a fact relevant to the question, USE it — "
            "do NOT search to re-derive what's settled, do NOT ask the user to re-explain "
            "something these turns already state, and do NOT request a search whose answer "
            "contradicts what was established here without good reason."
        )
        return header + "\n" + "\n".join(lines)

    @staticmethod
    def _detect_tool_hints(query: str) -> str:
        """Detect tool name mentions in a query and return usage hints.

        When the user explicitly mentions tools by name, the model should
        prioritize calling those tools rather than narrating about them.
        """
        q = query.lower()
        hints = []
        if any(w in q for w in ('github', 'issues', 'pull request', 'pr ', 'prs',
                                 'releases', 'actions', 'workflow')):
            hints.append('Use <github>your query</github> (or the github tool) to query GitHub.')
        if any(w in q for w in ('git stats', 'git stat', 'commits', 'loc ',
                                 'lines of code', 'lines added', 'lines changed',
                                 'files changed')):
            hints.append('Use <git_stats>your query</git_stats> (or the git_stats tool) for repo stats.')
        if any(w in q for w in ('search memory', 'remember', 'recall', 'my facts')):
            hints.append('Use <memory collection="facts">query</memory> (or the search_memory tool).')
        if not hints:
            return ''
        return (
            '\n[TOOL HINT]: The user is asking you to USE these tools, not describe them. '
            'Call them now:\n' + '\n'.join(f'- {h}' for h in hints)
        )

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

        # Detect tool mentions and add hints
        _tool_hints = self._detect_tool_hints(query)

        parts = [f"""{_time_ctx}User Question: {query}{_tool_hints}

Search Results So Far:
{search_context}

You are in round {round_number} of up to {self.max_rounds} search rounds."""]

        # Include context inventory so the LLM knows what RAG already gathered
        if session and session.context_inventory:
            parts.append(session.context_inventory)

        # Include this session's recent-turn digest (content, not just counts) so the
        # decision is grounded in what was already established this session.
        if session and session.recent_conversation_digest:
            parts.append(session.recent_conversation_digest)

        # Explicit write actions get a bounded richer digest. This is absent
        # from ordinary search rounds, so it does not inflate normal prompts.
        if session and getattr(session, "action_context_digest", ""):
            parts.append(session.action_context_digest)

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
1. If you have enough information to fully answer the question, write your COMPLETE final answer to the user now — finished, user-facing prose (it may be shown to them verbatim), with no preamble about what you did or plan to do.
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
                    parts.append(
                        "[RECENT CONVERSATION — THIS SESSION'S HISTORY]\n"
                        "Context only — do not reply to these turns as if they were the "
                        "current message. But they are established ground truth for this "
                        "session: if a search result contradicts what was already settled "
                        "here, surface the conflict and trust the session unless the new "
                        "evidence is clearly stronger — do NOT silently override it.\n"
                        "Entries ending in [...truncated] are PREVIEWS, not complete "
                        "messages — never quote one as if it were the full text; use "
                        "search_memory (conversations) to retrieve the full stored "
                        "message first.\n"
                        f"{recent_text}"
                    )

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

        # Check if an action was proposed during this session
        _has_pending_action = False
        try:
            from core.agentic.tools import ToolExecutor
            _store = ToolExecutor._get_pending_actions_store()
            _pending = _store.get_pending()
            if _pending:
                _has_pending_action = True
        except Exception:
            pass

        # Instructions
        has_web = bool(session.accumulated_context)
        has_wiki = bool(getattr(self._tool_executor, "_current_wiki_source_map", None))
        citation_line = (
            "- Cite web sources using [WEB_N] markers (e.g., 'According to Reuters [WEB_1]...'). "
            "Every factual claim from web sources MUST include a [WEB_N] citation.\n"
            "- Search results may be skewed toward the user's geographic area. NEVER assume an "
            "institution or business found in results (a school, bank, clinic, company) is the "
            "user's own unless the user or memory named it — no phone numbers, procedures, or "
            "identities attributed to 'their' institution from location-matched results; say the "
            "institution is unidentified and ask."
            if has_web else "- Cite web sources when stating facts from search results"
        )
        if has_wiki:
            citation_line = (
                "- Cite Wikipedia content using the [WIKI_N] markers from the context headers "
                "(e.g., 'first recorded in 610 AD [WIKI_1]...'). NEVER write a bare [Wikipedia] "
                "tag — use the numbered [WIKI_N] marker so the source can be linked.\n"
            ) + citation_line
        action_instruction = ""
        if _has_pending_action:
            action_instruction = (
                "\n- IMPORTANT: You proposed an action that is now awaiting user confirmation. "
                "Briefly confirm what you proposed and let the user know they can approve or reject it. "
                "Do NOT narrate about your tools or capabilities — just confirm the proposed action."
            )
        date_grounding_line = (
            "- For today's date and day of the week, [TIME CONTEXT] above is authoritative; "
            "if a web source states a conflicting day/date, defer to [TIME CONTEXT] (do not "
            "copy a weekday from a snippet)."
            if has_web else
            "- Use [TIME CONTEXT] above as the authoritative source for today's date and day of the week."
        )
        parts.append(f"""Please provide a comprehensive answer based on ALL context above:
- Use your memories, facts, and personal notes to personalize the response
{citation_line}
{date_grounding_line}
- Note any uncertainties or conflicting information
- Focus on answering the user's specific question
- If asked about tool status, ONLY report what [TOOL STATUS] says — do NOT rely on prior conversation{action_instruction}""")

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

    @staticmethod
    def _fallback_terms_from_query(query: str) -> str:
        """Distill a search string from the raw query for the decision-timeout
        fallback: strip the search-request preamble ("can we do a web search
        and attempt to confirm ...") so the searchable content remains."""
        q = (query or "").strip()
        _PREAMBLE_RE = re.compile(
            r"^(?:(?:hey|hi|ok(?:ay)?|so|please|also)[,\s]+)*"
            r"(?:can|could|would|will)?\s*(?:you|we)?\s*"
            r"(?:please\s+)?(?:do|run|try|perform)?\s*"
            r"(?:a\s+)?(?:web\s+|internet\s+|online\s+)?search(?:es)?\s*"
            r"(?:and\s+(?:attempt\s+to\s+|try\s+to\s+)?)?"
            r"(?:to\s+)?(?:confirm|verify|find(?:\s+out)?|check|look\s+up)?\s*",
            re.IGNORECASE,
        )
        stripped = _PREAMBLE_RE.sub("", q, count=1).strip(" ?.!,")
        return stripped if len(stripped.split()) >= 2 else q.strip(" ?.!,")

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
