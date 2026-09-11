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
from core.agentic.tools import LazySandboxSession, ToolExecutor
from core.action_claim_guard import UNVERIFIED_CLAIM_MARKER
from core.reasoning_stream_filter import InterleavedReasoningFilter
from utils.python_fs_guard import agent_mode as _fs_agent_mode
from utils.ordered_slice import oldest_first as _ordered_oldest_first
from utils.text_budget import fit_text_to_tokens

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

# F8 (2026-09-08 homework-session audit): the section-trim ladder in
# _build_final_prompt is section-level (dreams/reflections/docs/summaries)
# and can be fully exhausted while a single oversized [CURRENT USER QUERY]
# part — a large attachment merged straight into the query text — is what's
# still pushing the assembled prompt over the ceiling (records 10/14 logged
# "40559/40000 tokens after trimming" and shipped the over-budget prompt
# anyway). `_squeeze_query_part_for_ceiling` below middle-outs ONLY that
# part's body as a last resort.
_CURRENT_QUERY_HEADER = "[CURRENT USER QUERY — RESPOND TO THIS]"
_MIDDLE_OUT_SNIP_RE = re.compile(r"\n… \[middle-out snipped (\d+) chars\] …\n")
_UPLOAD_TITLE_IN_TEXT_RE = re.compile(r'upload:([^"\)\s]+)')

# Git-cued document hint (2026-09-10, round 2, A9): "Cool. Managed to push
# today and there is a new doc I think will be helpful" followed by "Can
# you take a look?" resolved the unnamed document to a 5-day-old upload via
# the reuse pool instead of the repository file the user had just pushed —
# no cue reached the loop at all. When the current OR previous USER turn
# carries a git cue AND a doc noun, the loop is hinted to check the repo
# FIRST.
_GIT_CUE_RE = re.compile(r"\b(?:push(?:ed|ing)?|commit(?:ted|ting)?|repo(?:sitory)?|pr|merged)\b", re.IGNORECASE)
_DOC_NOUN_RE = re.compile(r"\b(?:docs?|documents?|files?|mds?|readme|handoffs?)\b", re.IGNORECASE)


# Forced write-action detection + deterministic param backfill now live in the action registry
# (core/actions/registry.py) — the single source of truth, so adding an action is one place.
# Re-exported here for the controller body and for tests that import these names.
from core.actions.registry import (  # noqa: E402
    detect_action_intent,
    backfill_params,
    calendar_datetime_shape_errors,
    extract_calendar_title,
    resolve_weekday_time,
    resolved_fields_note,
    _extract_issue_fields_from_query,
)


def _pending_cards_note(action_verb: str = "call propose_action") -> str:
    """[PENDING CARDS] line for a forced action-round prompt (2026-09-10,
    round 3, A11): truthful about whether an approval card actually
    exists, so the model cannot lean on a PRIOR turn's "queued"/"locked
    in"/"re-queued" wording as if it minted a real proposal — none of the
    round-3 live failures (T1/T2/T3) had an actual card behind that
    narration. Lists real cards when they exist.

    ``action_verb`` is protocol-appropriate: "call propose_action" for the
    native-tools builder (the ONLY place "propose_action" — native-tools
    vocabulary — may appear), "emit the <action> marker" for the XML
    builder — the XML forced prompt must never leak native-tools wording.
    """
    try:
        from core.agentic.tools import ToolExecutor
        pending = ToolExecutor._get_pending_actions_store().get_all_pending()
    except Exception:
        pending = []
    if not pending:
        return (
            "[PENDING CARDS] none — any earlier 'queued'/'locked in'/"
            "'re-queued' wording in the conversation was NOT backed by a "
            f"card; you must {action_verb} now."
        )
    lines = []
    for p in pending:
        _t = getattr(p.action_type, "value", p.action_type)
        lines.append(f"- {_t}: {p.summary or p.action_id}")
    return "[PENDING CARDS]\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Note-body extraction (2026-09-10, round 3, A14)
# ---------------------------------------------------------------------------
# Live: "jot down a note for this session: TA sessions are Saturdays at 11
# CT," routed correctly (A3) but create_daemon_note saved a hallucinated
# calendar claim instead of the user's own words — the model was free to
# write whatever it wanted into the note body. This is deliberately
# query_checker-free (self-contained cue table, not a reach into
# utils.query_checker's private regexes) — prefers the text after a
# colon/dash separator following the note-save cue ("... for this session:
# X" -> "X"); with no separator, strips the leading imperative + filler and
# keeps what remains.
_NOTE_BODY_CUE_RE = re.compile(
    r"^\s*(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|please\s+)?"
    r"(?:jot\s+(?:down|this)\s+(?:a\s+)?note|"
    r"save\s+(?:this\s+)?(?:as\s+)?(?:a\s+)?note|"
    r"write\s+(?:this\s+)?(?:down\s+)?(?:as\s+)?(?:a\s+)?note|"
    r"make\s+(?:a\s+)?note|"
    r"remember\s+this(?:\s+for\s+(?:me|later|next\s+time))?|"
    r"note\s+to\s+self)\b",
    re.IGNORECASE,
)
# A colon anywhere, or a dash surrounded by SPACES (never a bare hyphen
# inside a range like "9-11") — the clearest boundary between the
# instruction and the content itself.
_NOTE_BODY_SEPARATOR_RE = re.compile(r":\s*|\s[-–—]\s")
_NOTE_BODY_LEADING_FILLER_RE = re.compile(
    r"^\s*(?:for\s+(?:this\s+session|me|later|next\s+time)\s*)?(?:that\s+)?[,:\-–—\s]*",
    re.IGNORECASE,
)


def extract_note_body(query: str) -> str:
    """Deterministic note-body extraction for a gate-detected note-save
    request — see module comment above for the exact rule. Empty input
    yields an empty body (never a caller crash)."""
    text = (query or "").strip()
    if not text:
        return ""
    m = _NOTE_BODY_CUE_RE.match(text)
    rest = text[m.end():] if m else text
    sep = _NOTE_BODY_SEPARATOR_RE.search(rest)
    if sep:
        body = rest[sep.end():]
    else:
        body = _NOTE_BODY_LEADING_FILLER_RE.sub("", rest, count=1)
    body = body.strip().rstrip(",.;").strip()
    return body or text.rstrip(",.;").strip()


def note_fallback_title(body: str) -> str:
    """Truncate a note-fallback BODY to a title of at most 60 characters,
    breaking at a word boundary rather than mid-word (2026-09-11, round 5,
    A19 — mirrors the A11 calendar fallback's own resolvable-from-the-
    request-alone approach: the body itself is the only text this
    deterministic path has, so it also serves as the title, shortened)."""
    text = (body or "").strip()
    if len(text) <= 60:
        return text
    truncated = text[:60]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.strip()


def _backfill_fill_keys(params: dict, bf: dict, wd_bf: dict) -> list:
    """Keys the deterministic backfill may write (2026-09-10 referee fix):
    blank fields as before, PLUS a calendar start/end the model supplied in
    a shape-invalid form (a date-less "15:00:00") when the user's own words
    resolved it (``wd_bf``) — otherwise the proposal is rejected at parse and
    costs a retry round for a value the request already determined."""
    out = []
    for k, v in bf.items():
        if not v:
            continue
        if not params.get(k):
            out.append(k)
        elif k in ("start_time", "end_time") and k in wd_bf and calendar_datetime_shape_errors({k: params.get(k)}):
            out.append(k)
    return out

def _should_ground_calendar_times(session, this_round_forced_type: Optional[str]) -> bool:
    """Whether THIS round's calendar_create_event decisions must be checked
    against the gathered-context time pool (2026-09-10, narrowed same day by
    referee follow-up).

    A forced round always grounds — that's the original guessed-17:00
    incident. But once a session has DECLINED a guessed time
    (``session._action_force_declined`` set by that same check below), the
    loop continues UNFORCED and the model was previously free to re-propose
    the same guessed time on a later, unforced round with no check at all.
    Once a session has declined once, every LATER calendar_create_event
    decision in the session is grounded too. Callers still separately
    require ``session.action_context_digest`` to be truthy (no pool built =
    nothing to check against).
    """
    return bool(this_round_forced_type) or bool(getattr(session, "_action_force_declined", False))


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

        # A single tool response may exceed the budget, or contain no round
        # delimiter at all. Dropping old blocks alone cannot bound that case.
        from utils.text_budget import fit_text_to_tokens
        session.accumulated_context = fit_text_to_tokens(
            "\n\n---\n".join(blocks), self.context_budget_tokens, self._estimate_tokens,
        )
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
        forced_action: Optional[str] = None,
        action_query_ws: Optional[str] = None,
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
            forced_action: ActionType.value to force on the first decision round when
                the QUERY itself carries no action pattern — the gate's prior-turn
                offer affirmation (2026-09-07: "please create" after "Want me to
                create the recurring event?"). A same-turn detect_action_intent hit wins.
            action_query_ws (2026-09-10, round 3, A10/A11/A14 sibling): the
                caller's already whitespace-normalized user text (never the
                merged/attachment-bearing `query`) — used ONLY for
                action-detection/backfill (detect_action_intent,
                resolve_weekday_time, extract_calendar_title, the note-body
                extraction) so a client-side soft line-wrap or attached-file
                content can never defeat those deterministic checks. Falls
                back to `query` when not supplied (every pre-existing caller).

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
        self._action_query_ws = action_query_ws if action_query_ws is not None else query
        protocol = self.detect_protocol(model_name)
        session = AgenticSearchSession(
            query=query,
            max_rounds=self.max_rounds,
            protocol=protocol,
        )
        # A14 (2026-09-10, round 3): a gate-detected note-save request's
        # saved note body must be the USER'S stated content, never the
        # model's own elaboration on the create_daemon_note call — live:
        # "jot down a note for this session: TA sessions are Saturdays at
        # 11 CT," ended up saving a hallucinated "calendar event already
        # created" claim instead. Computed once per session; consumed at
        # dispatch time in _dispatch_single_inner.
        session.note_body_override = None
        try:
            from utils.query_checker import is_note_save_request
            if is_note_save_request(self._action_query_ws):
                session.note_body_override = extract_note_body(self._action_query_ws)
        except Exception as e:
            logger.debug(f"[AgenticSearch] Note-body extraction skipped: {e}")

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

        # Lazy sandbox acquisition (2026-09-08, F5): sandbox_available (=
        # sandbox_manager.is_available()) is a cheap local flag, but actually
        # acquiring a session is a remote E2B create-session call. Acquiring
        # it here unconditionally — before round 1, for every agentic turn
        # where the flag was true — created a sandbox even on turns that
        # never executed code (live: 26 created, 13 closed with zero
        # executions). LazySandboxSession defers the real call to
        # ToolExecutor._dispatch_sandbox, the only site that ever needs it;
        # _get_sandbox_session's recycling/age/liveness handling and
        # close_sandbox() are unchanged, and the "Using persistent sandbox
        # session" log now fires on first successful acquisition there.
        sandbox_session = LazySandboxSession(self._get_sandbox_session) if sandbox_available else None

        try:
            # Tracks whether round 1 itself ran a web search (as opposed to a
            # URL fetch or a skip-to-loop branch) — used by the A3 pre-
            # gathered-web seeding below, which must not double-seed when the
            # loop's own round 1 already searched the web for this turn.
            _round1_was_web_search = False

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
                _round1_was_web_search = True
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

            # A3 (2026-09-06): seed this turn's pre-gathered base web
            # evidence into the loop when round 1 did NOT itself run a web
            # search (memory/tool routing, or a URL fetch). Without this,
            # initial_context["web_search_results"] — the prompt builder's
            # own base retrieval for this turn — never reaches
            # accumulated_context, the decision prompt, or the final prompt;
            # decision-answer reuse then answers from counts + a short digest
            # alone, having never actually seen the evidence.
            if not _round1_was_web_search:
                _base_web = (initial_context or {}).get("web_search_results")
                if getattr(_base_web, "has_results", False):
                    _merge_web_ids = getattr(self._tool_executor, "_merge_web_ids", None)
                    if _merge_web_ids is None:
                        logger.debug(
                            "[AgenticSearch] Pre-gathered web_search_results present "
                            "but tool executor lacks _merge_web_ids — skipping seed"
                        )
                    else:
                        try:
                            # _merge_web_ids assigns ids FIRST (continuing the
                            # session-wide map) — render_prenumbered_web_sources
                            # formats that already-numbered list without ever
                            # calling assign_web_ids a second time (which would
                            # mint a second, colliding set of ids for the same
                            # pages; see knowledge/web_search_manager.py).
                            _numbered = _merge_web_ids(_base_web.pages)
                            from knowledge.web_search_manager import render_prenumbered_web_sources  # lazy import: matches tools.py's existing lazy web_search_manager imports
                            _seed_lines, _ = render_prenumbered_web_sources(
                                _numbered, max_sources=8, max_chars_per_source=2000,
                            )
                            if _seed_lines:
                                self._append_accumulated(
                                    session,
                                    "[Pre-gathered web results — base retrieval for this turn]\n"
                                    + "\n\n".join(_seed_lines)
                                )
                                session.seeded_base_web = True
                                logger.info(
                                    f"[AgenticSearch] Seeded {len(_seed_lines)} "
                                    f"pre-gathered base web source(s) into the loop"
                                )
                        except Exception as e:
                            logger.debug(
                                f"[AgenticSearch] Pre-gathered web seeding failed "
                                f"(non-fatal): {e}"
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

            # A9 (round 2): the previous USER turn's raw text — checked
            # alongside the current query for a git-cued document hint
            # (_detect_tool_hints), computed once since it does not change
            # across rounds.
            _prev_user_text_for_hints = self._previous_user_query(initial_context)

            # Detect explicit write-action intent. If present, force the model to call
            # propose_action on the first decision round (native-tools protocol only) so
            # research-eager models don't spend every round reading code and never act.
            # A10 (2026-09-10, round 3): action-detection reads the caller's
            # already-normalized `self._action_query_ws` (falls back to
            # `query` when the caller supplied none — every pre-existing
            # test/call site) so a client-side soft line-wrap can never
            # defeat this check the way it defeated the shape predicates.
            _forced_action = detect_action_intent(self._action_query_ws)  # ActionType or None (from the registry)
            # A6 (round 2): True when the CURRENT query text is not itself a
            # self-contained action request — the force came from the gate's
            # prior-turn offer/retry/clarification-answer arm, so `query`
            # ("Yes 1 hour", "yes") is a short follow-up, not the request
            # itself. The forced-round prompt then labels it [USER ANSWER]
            # instead of "The user asked" (the actual request lives in the
            # action-context digest computed below).
            _forced_via_gate = _forced_action is None and bool(forced_action)
            if _forced_action is None and forced_action:
                try:
                    from core.actions.types import ActionType as _AT
                    _forced_action = _AT(forced_action)
                    logger.info(
                        f"[AgenticSearch] Forcing {forced_action} from the gate's "
                        "prior-turn offer affirmation"
                    )
                except ValueError:
                    logger.warning(
                        f"[AgenticSearch] Unknown forced_action {forced_action!r} ignored")
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
                    session=session,
                    prev_user_text=_prev_user_text_for_hints,
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
                # Snapshot BEFORE the block below consumes it — this is what
                # gets passed to _get_model_decision so parsing can pin/coerce
                # the action_type ONLY on a round that is actually forcing
                # (F12, 2026-09-09: never outside a forced round).
                _this_round_forced_type: Optional[str] = (
                    _forced_action.value if (_force_propose_pending and _forced_action) else None
                )
                if _force_propose_pending:
                    from core.actions.registry import ACTION_SPECS, build_forced_tool_schema
                    _spec = ACTION_SPECS.get(_forced_action)
                    _hint = (_spec.field_hint if _spec and _spec.field_hint else "the required fields")
                    _prior_reject_reason = session.last_action_reject_reason
                    _reject_note = (
                        f" Your previous attempt was REJECTED: {_prior_reject_reason}."
                        if _prior_reject_reason else ""
                    )
                    session.last_action_reject_reason = None  # consumed
                    _round_tool_choice = {"type": "function", "function": {"name": "propose_action"}}
                    # Forced-round schema (F12): scoped to exactly the required
                    # type/fields so the model cannot silently substitute a
                    # sibling action_type (the generic tool's enum previously
                    # had no calendar_update/delete_event entries at all).
                    _ptool = build_forced_tool_schema(_forced_action) or getattr(
                        handler, "propose_action_tool", None)
                    if getattr(handler, "propose_action_tool", None) is not None:
                        _round_tools_override = [_ptool]
                        # Use the user's actual request as the prompt (not the generic "what tool
                        # next?" iteration prompt) so the model fills the content fields from it.
                        # A10 (round 3): `self._action_query_ws` (whitespace-
                        # normalized bare user text, falls back to `query`)
                        # rather than raw `query` — a client soft line-wrap
                        # must not defeat resolved_fields_note/
                        # resolve_weekday_time inside this builder; the full
                        # attachment-merged `query` is still visible to the
                        # model via the iteration prompt/action digest above.
                        _round_prompt = iteration_prompt + "\n\n" + self._build_native_action_prompt(
                            self._action_query_ws, _forced_action, is_followup=_forced_via_gate)
                        _round_system_prompt = augmented_system_prompt + (
                            f"\n\n[ACTION REQUIRED] The user explicitly asked you to perform a write "
                            f"action ({_forced_action.value}) and ONLY that action_type — do not "
                            f"substitute a sibling type. Call propose_action NOW with "
                            f"action_type=\"{_forced_action.value}\" and FILL IN the content fields "
                            f"from the user's request — for this action: {_hint}. Do NOT leave "
                            f"required fields empty, and do NOT specify a repo (auto-detected). "
                            f"Never INVENT a time, date, or recipient that is not stated in the "
                            f"request or the context above — a guessed value is rejected."
                            f"{_reject_note}"
                        )
                    else:
                        # XML-markers protocol (2026-08-29): tool_choice/tools_override are
                        # ignored here and "propose_action" is native-tools vocabulary the
                        # model has never seen — the live forced calendar round produced
                        # NOTHING and fell through to implicit-ready. Give the model the
                        # actual marker syntax with the spec's required fields as
                        # attributes, one marker per item.
                        # A10 (round 3): same self._action_query_ws rationale
                        # as the native-tools branch above.
                        _round_prompt = iteration_prompt + "\n\n" + self._build_xml_action_force_prompt(
                            self._action_query_ws, _forced_action, _spec,
                            reject_reason=_prior_reject_reason,
                            is_followup=_forced_via_gate)
                        _round_system_prompt = augmented_system_prompt + (
                            f"\n\n[ACTION REQUIRED] The user explicitly asked you to perform a "
                            f"write action ({_forced_action.value}). Emit the <action> marker(s) "
                            f"NOW exactly as instructed — for this action: {_hint}. Do NOT "
                            f"narrate or answer in prose; markers only.{_reject_note}"
                        )
                    _force_propose_pending = False  # force on this round only

                # Receipt (2026-09-06, A5): hash of the LAST iteration prompt
                # actually sent to _get_model_decision. Overwritten every
                # round so it ends up naming the round that produced whatever
                # answer (reused or not) the loop exits with.
                session.decision_prompt_hash = hashlib.sha256(
                    _round_prompt.encode("utf-8", "ignore")
                ).hexdigest()[:16]

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
                    forced_action_type=_this_round_forced_type,
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
                # Forced-round time grounding (2026-09-10): a forced
                # calendar proposal whose clock time appears nowhere in the
                # request, the conversation/action digests, or the gathered
                # tool output is a GUESS (live: "I only put professors hours
                # in calander" → a TA session invented at 17:00 with the
                # model's own reasoning saying "should be confirmed"). Drop
                # it, record why, and never re-force this session — the
                # loop continues unforced so the model can look the time up
                # or ask; the no-card backstop keeps the reply honest.
                # A22 (2026-09-11, round 6): the check didn't recognize its
                # OWN deterministic resolution — a forced round proposed
                # exactly resolve_weekday_time's 1-hour-default end time and
                # got declined as an invented guess anyway.
                # ground_calendar_params_by_resolution grounds/replaces
                # against that resolution before the pool-text check gets
                # the final say; calendar_times_ungrounded itself is
                # untouched (called internally, still pure).
                if (
                    _action_decisions
                    and _should_ground_calendar_times(session, _this_round_forced_type)
                    and session.action_context_digest
                ):
                    from core.actions.registry import ground_calendar_params_by_resolution
                    _pool = "\n".join(str(x or "") for x in (
                        query, session.action_context_digest,
                        session.recent_conversation_digest, session.accumulated_context))
                    _kept = []
                    for _ad in _action_decisions:
                        _t = str(getattr(_ad.action_type, "value", _ad.action_type) or "")
                        if _t == "calendar_create_event":
                            _grounded, _replaced, _bad = ground_calendar_params_by_resolution(
                                _ad.action_params or {}, self._action_query_ws, _pool)
                            if _replaced:
                                _ad.action_params = _grounded
                                for _label, _old, _new in _replaced:
                                    logger.info(
                                        f"[AgenticSearch] ungrounded {_label}={_old} "
                                        f"replaced by request resolution {_new}"
                                    )
                        else:
                            _bad = []
                        if _bad:
                            _reason = (
                                f"forced {_t} not proposed: {', '.join(_bad)} appears nowhere in "
                                "the request or gathered context — a guessed time is worse than "
                                "no card; ask the user for the time or look it up")
                            _ad.action_reject_reason = _reason
                            session._action_force_declined = True
                            self._append_accumulated(session, f"[ACTION NOT PROPOSED] {_reason}")
                            logger.warning(f"[AgenticSearch] {_reason}")
                            continue
                        _kept.append(_ad)
                    _action_decisions = _kept
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
                        # Weekday + clock-time backfill (2026-09-10, A2): a
                        # calendar_create_event request naming a weekday and
                        # a bare hour ("Tuesdays at 3, through Dec 4") with
                        # no explicit date — resolve_weekday_time fills the
                        # date the model otherwise has to guess (or, live,
                        # left as a dateless bare clock time that the new
                        # proposal-time shape check now rejects outright).
                        _wd_bf = {}
                        if _ad.action_type == "calendar_create_event":
                            try:
                                # A10 (round 3): normalized bare user text —
                                # see the run_agentic_search docstring entry
                                # for action_query_ws.
                                _wd_bf = resolve_weekday_time(self._action_query_ws)
                            except Exception as e:
                                logger.debug(
                                    f"[AgenticSearch] Weekday/time backfill failed (non-fatal): {e}"
                                )
                                _wd_bf = {}
                            for _wk, _wv in _wd_bf.items():
                                _bf.setdefault(_wk, _wv)
                        if _bf:
                            _params = dict(_ad.action_params or {})
                            _filled = _backfill_fill_keys(_params, _bf, _wd_bf)
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
                            and not getattr(session, '_action_force_declined', False)
                            and not getattr(session, '_action_force_retry_sent', False)):
                        session._action_force_retry_sent = True
                        _force_propose_pending = True
                        # F12 (2026-09-09): carry WHY the proposal was
                        # rejected into the retry — a wrong-type propose_action
                        # (e.g. calendar_create_event proposed while
                        # calendar_delete_event was required) used to be
                        # silently dropped and the retry was a blind re-ask.
                        _reject_d = next(
                            (d for d in decisions if getattr(d, "action_reject_reason", None)),
                            None,
                        )
                        session.last_action_reject_reason = (
                            _reject_d.action_reject_reason if _reject_d else None
                        )
                        logger.info(
                            "[AgenticSearch] Forced action round produced no "
                            "action marker — retrying once"
                            + (f" (rejected: {session.last_action_reject_reason})"
                               if session.last_action_reject_reason else "")
                        )
                        continue

                    # A11 deterministic fallback (2026-09-10, round 3): the
                    # forced round AND its one retry (above) BOTH produced no
                    # calendar decision — live: "put a recurring calendar
                    # event ... for the MGT study group, Tuesdays at 3,
                    # through Dec 4" silently fell through to "ready to
                    # answer" twice, and the final synthesis narrated a
                    # queue that never happened. When the request's own
                    # weekday+clock-time is resolvable AND a title is
                    # extractable, mint the proposal ourselves through the
                    # SAME dispatch path a model decision takes — still
                    # human-gated (a pending card, never auto-executed).
                    if (
                        _forced_action is not None
                        and str(getattr(_forced_action, "value", _forced_action))
                        == "calendar_create_event"
                        and not _action_decisions
                        and not getattr(session, '_action_dispatched', False)
                        and not getattr(session, '_action_force_declined', False)
                        and getattr(session, '_action_force_retry_sent', False)
                        and not getattr(session, '_action_force_fallback_sent', False)
                    ):
                        session._action_force_fallback_sent = True
                        _fb_wd = resolve_weekday_time(self._action_query_ws)
                        _fb_title = extract_calendar_title(self._action_query_ws)
                        if _fb_wd and _fb_title:
                            _fb_params = {
                                "summary": _fb_title,
                                "start_time": _fb_wd["start_time"],
                                "end_time": _fb_wd["end_time"],
                            }
                            if _fb_wd.get("recurrence"):
                                _fb_params["recurrence"] = _fb_wd["recurrence"]
                            _fb_decision = SearchDecision(
                                wants_action=True,
                                action_type="calendar_create_event",
                                action_params=_fb_params,
                                action_reason=(
                                    "deterministic fallback: model declined "
                                    "to propose"
                                ),
                            )
                            logger.warning(
                                "[AgenticSearch] Forced calendar round + retry "
                                "both declined to propose — minting the "
                                f"proposal deterministically: {_fb_params}"
                            )
                            _fb_round = session.current_round
                            telemetry_entry.setdefault("rounds", []).append(_fb_round)
                            _fb_result = await self._dispatch_single(
                                _fb_decision, _fb_round, session, crisis_level,
                                sandbox_session,
                            )
                            for ev in _fb_result.start_events:
                                yield ev
                            for ev in _fb_result.end_events:
                                yield ev
                            if _fb_result.round_data is not None:
                                session.rounds.append(_fb_result.round_data)
                            if _fb_result.formatted_context:
                                self._append_accumulated(session, _fb_result.formatted_context)
                            session._action_dispatched = True
                            continue  # let the model produce its final answer

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

            # A19 (2026-09-11, round 5): deterministic note-save fallback —
            # the A11 calendar pattern, one chokepoint placed right after
            # the round loop so it covers ALL THREE ways the loop can end
            # (explicit done, implicit ready-to-answer, or max-rounds
            # exhaustion) without duplicating the check at each exit. Live:
            # a gate-detected note-save request ("jot down a note for this
            # session: TA sessions are Saturdays at 11 CT,") ran two
            # implicit-ready-to-answer rounds with NO
            # `Native tool create_daemon_note` call at all — the tool HINT
            # offered to the model is advisory only, the model declined it
            # twice, and the final reply narrated the note as already
            # saved. session.note_body_override is set only for a
            # gate-detected note-save request (session init above);
            # session._note_dispatched is set at the single dispatch
            # chokepoint (_dispatch_single_inner / ToolExecutor.dispatch_single)
            # whenever a create_daemon_note decision — model-authored or
            # this fallback — actually runs, so this check has one source
            # of truth rather than re-deriving it from session.rounds.
            if (
                getattr(session, "note_body_override", None)
                and not getattr(session, "_note_dispatched", False)
                and not getattr(session, "_note_force_fallback_sent", False)
            ):
                session._note_force_fallback_sent = True
                _fb_note_body = session.note_body_override
                _fb_note_decision = SearchDecision(
                    wants_create_daemon_note=True,
                    daemon_note_title=note_fallback_title(_fb_note_body),
                    daemon_note_category="implementation",
                    daemon_note_summary=_fb_note_body,
                    daemon_note_user_requested=True,
                    daemon_note_reason=(
                        "deterministic fallback: model declined to save"
                    ),
                )
                logger.warning(
                    "[AgenticSearch] Note-save request never dispatched "
                    "create_daemon_note across the round loop — minting "
                    f"the note deterministically: {_fb_note_body!r}"
                )
                _fb_note_round = session.current_round
                _fb_note_result = await self._dispatch_single(
                    _fb_note_decision, _fb_note_round, session, crisis_level,
                    sandbox_session,
                )
                for ev in _fb_note_result.start_events:
                    yield ev
                for ev in _fb_note_result.end_events:
                    yield ev
                if _fb_note_result.round_data is not None:
                    session.rounds.append(_fb_note_result.round_data)
                if _fb_note_result.formatted_context:
                    self._append_accumulated(session, _fb_note_result.formatted_context)
                # The model's final answer must reflect what the note
                # fallback just did, not a stale narration captured before
                # it ran (the exact live R5 confabulation) — discard any
                # decision-round reuse candidate so a real final-generation
                # call runs. A11's in-loop `continue` re-queries the model
                # for the same reason; there is no further round to
                # re-query here, so full synthesis IS "let the model
                # produce its final answer".
                _decision_answer_text = None

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
            _candidate_reused_answer = (
                self._usable_decision_answer(_decision_answer_text)
                if (AGENTIC_REUSE_DECISION_ANSWER and _decision_answer_text)
                else None
            )
            # A4/B1 (2026-09-06/07): reuse is permitted only when the decision
            # round that produced this text actually saw admitted evidence —
            # never when initial_context carried memories/uploads/web results/
            # ... (a _RETRIEVAL_EVIDENCE_KEYS entry) the decision prompt never
            # rendered (only a bounded digest + counts). Background-only
            # context (profile/summaries/reflections) does not block reuse.
            _reused_answer = None
            if _candidate_reused_answer:
                if self._decision_saw_admitted_evidence(session, initial_context):
                    _reused_answer = _candidate_reused_answer
                else:
                    _missing_key = self._first_unmet_retrieval_key(initial_context, session)
                    session.reuse_skipped_reason = (
                        f"decision prompt lacked admitted evidence: {_missing_key}"
                        if _missing_key
                        else "decision prompt lacked admitted evidence"
                    )
                    logger.info(
                        "[AgenticSearch] Decision answer would have been reused, "
                        "but initial_context carries admitted evidence the "
                        "decision round never saw — falling back to full synthesis"
                    )

            if _reused_answer:
                session.decision_answer_reuse_fired = True
                session.answer_call = "decision_reuse"
                # Real hash of the actual prompt the answering call saw — the
                # old sentinel string "decision-answer-reuse" recorded nothing
                # about which call produced the answer (2026-09-06, A5).
                session.final_prompt_hash = session.decision_prompt_hash
                _reuse_sections: List[str] = []
                if session.recent_conversation_digest:
                    _reuse_sections.append("[RECENT CONVERSATION — EARLIER TURNS]")
                if session.accumulated_context and session.accumulated_context.strip():
                    _reuse_sections.append("Search Results So Far")
                if session.context_inventory:
                    _reuse_sections.append("Context inventory")
                session.visible_sources = self._compute_visible_sources(_reuse_sections)
                # web_search_results counts as "rendered" for the reuse path
                # only when A3 actually seeded it into accumulated_context —
                # otherwise it is exactly the evidence the digest/inventory
                # summarized but never showed in full.
                _reuse_rendered_keys = {"web_search_results"} if session.seeded_base_web else set()
                session.omitted_sections = self._omitted_admitted_sections(
                    initial_context, _reuse_rendered_keys
                )
                logger.info(
                    f"[AgenticSearch] Reusing decision-round answer "
                    f"({len(_reused_answer)} chars) — final synthesis call skipped"
                )
                yield _reused_answer
            else:
                # Generate final response
                session.answer_call = "final_synthesis"
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

                session.answer_call = "error_fallback"
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
        # A14 (2026-09-10, round 3): a gate-detected note-save request's
        # body is the USER'S stated content — computed once at session
        # start (run_agentic_search) — never the model's own elaboration.
        # Overridden here, the single chokepoint both routers dispatch
        # through, rather than inside ToolExecutor._dispatch_create_daemon_note
        # (which has no session access).
        if getattr(decision, "wants_create_daemon_note", False):
            # A19 (2026-09-11, round 5): mark the session as having actually
            # dispatched create_daemon_note THIS turn — whichever path
            # drives it (a model decision or the post-loop deterministic
            # fallback in run_agentic_search) — so that fallback's
            # "was a note ever dispatched" check has a single source of
            # truth instead of re-deriving it from session.rounds.
            session._note_dispatched = True
            if getattr(session, "note_body_override", None):
                decision.daemon_note_summary = session.note_body_override
                # A15 (round 4): this note's body came from an explicit user
                # request, not model initiative — the executor's autonomy
                # guardrails (session cap, semantic dedup) must not veto it.
                decision.daemon_note_user_requested = True
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
        forced_action_type: Optional[str] = None,
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
            forced_action_type: ActionType.value this round is forcing (F12,
                2026-09-09), or None. Passed to the protocol handler so a
                propose_action call naming a DIFFERENT type gets coerced
                (when its params fit the required spec) or rejected with a
                reason — never outside a forced round.

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

            return handler.parse_response(response, forced_action_type=forced_action_type)

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

        For the agentic iteration phase, models like DeepSeek/Kimi burn their
        token budget on chain-of-thought reasoning, leaving no room for
        actual XML tool markers. This method bypasses reasoning and uses
        a higher token limit so the model can emit tool tags directly.

        2026-09-06 (fix 1.5): delegates to THE deployed
        ``model_manager.generate_once(disable_reasoning=True)`` instead of
        building the API call directly. A direct call that simply omits the
        ``reasoning`` key does NOT disable reasoning for reasoning-by-default
        models (kimi-k3, the active model) — live: three memory_search
        decision rounds took 45.6s of decision time for 45ms of tool time.
        The direct call also bypassed ``resolve_top_p`` (kimi-k3 mandates
        top_p=0.95 or 400s) and the ``extra_body={"usage":{"include":True}}``
        cache-usage accounting ``generate_once`` already sends.
        """
        from config.app_config import AGENTIC_DECISION_MAX_TOKENS
        try:
            return await self.model_manager.generate_once(
                prompt=prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                max_tokens=AGENTIC_DECISION_MAX_TOKENS,
                temperature=0.3,
                disable_reasoning=True,
            )
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
                disable_reasoning=True,
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
    # Numbered ("1." / "2)") or bulleted ("-", "*", "•") line opener — used by
    # the B2 enumerated-question-list guard in _usable_decision_answer.
    _ENUMERATED_LINE_RE = re.compile(r"^(?:\d{1,2}[.)]|[-*•])\s+")

    _LOOP_META_RE = re.compile(
        r"\b(?:first|second|third|next|another|last|that|this)\s+round\s+of\b"
        r"|\bmissed\s+the\s+mark\b"
        r"|\bresults?\s+(?:missed|didn'?t\s+(?:return|match|help)|came\s+back\s+empty)\b",
        re.IGNORECASE,
    )

    # Context keys carrying admitted evidence the prompt builder already
    # gathered (2026-09-06, A4/A5). Decision-answer reuse must not stand in
    # for a real synthesis call when one of these was non-empty but never
    # reached the decision prompt (only a bounded digest + counts did).
    #
    # B1 (2026-09-07): split into background vs. retrieval evidence. Tool
    # results alone never prove the BASE retrieval evidence was seen — a
    # turn-6 incident had the loop list repo files with file_list/file_grep,
    # which counted as "evidence seen" via session.accumulated_context while
    # the base retrieval's memories/notes/uploads never reached the decision
    # prompt, and the reused answer narrated instead of using them. Only a
    # non-empty RETRIEVAL key may now block reuse; background context
    # (profile/summary/reflection digest lines already visible to the
    # decision prompt via context_inventory) never does.
    _BACKGROUND_EVIDENCE_KEYS: Tuple[str, ...] = (
        "user_profile", "recent_summaries", "recent_reflections",
    )
    # "user_uploads" joins this tuple (2026-09-07, B1): the [USER UPLOADED
    # ITEMS] section (core/prompt/formatter.py, context key "user_uploads"
    # per core/prompt/builder.py / token_manager.py) carries real retrieved
    # content exactly like memories/reference_docs and was simply missing
    # from evidence tracking — a decision round that never saw the user's
    # uploaded homework could still be "reused" as the final answer.
    _RETRIEVAL_EVIDENCE_KEYS: Tuple[str, ...] = (
        "memories", "personal_notes", "semantic_summaries", "reference_docs",
        "web_search_results", "graph_context", "relevant_emails",
        "google_calendar", "user_uploads",
    )
    # Unchanged total (background + retrieval) so the A5 omitted_sections
    # receipt keeps reporting every admitted-evidence key, not just the
    # ones that can block reuse.
    _ADMITTED_EVIDENCE_KEYS: Tuple[str, ...] = (
        _BACKGROUND_EVIDENCE_KEYS + _RETRIEVAL_EVIDENCE_KEYS
    )

    # Keys _build_final_prompt renders directly (when non-empty) into the
    # final synthesis prompt, independent of session.accumulated_context.
    # "web_search_results" is handled separately (rendered only via
    # accumulated_context — see _omitted_admitted_sections).
    _FINAL_PROMPT_DIRECT_RENDERED_KEYS = frozenset({
        "memories", "user_profile", "recent_summaries", "semantic_summaries",
        "reference_docs", "recent_reflections", "personal_notes", "user_uploads",
    })

    _UPLOAD_CHUNK_MAX_CHARS = 1500

    @staticmethod
    def _is_upload_roster_marker(item: Any) -> bool:
        return isinstance(item, dict) and (item.get('metadata') or {}).get('type') == 'upload_roster'

    @classmethod
    def _upload_roster_line(cls, user_uploads: Any) -> str:
        """'Homework1-2.pdf (2026-09-05), …' from the gatherer's roster marker
        (core/prompt/gatherer_knowledge.get_user_uploads), or ''."""
        for item in user_uploads or []:
            if cls._is_upload_roster_marker(item):
                roster = (item.get('metadata') or {}).get('roster') or []
                return ", ".join(
                    f"{r.get('title', '')} ({r.get('date', '')})" for r in roster if r.get('title')
                )
        return ""

    @classmethod
    def _format_user_uploads(cls, user_uploads: Any) -> str:
        """Numbered admitted upload chunks (title + capped content; image
        stubs by name only) followed by the roster line."""
        lines: List[str] = []
        n = 0
        for item in user_uploads or []:
            if cls._is_upload_roster_marker(item) or not isinstance(item, dict):
                continue
            meta = item.get('metadata') or {}
            title = str(meta.get('title', '') or '')
            if title.startswith('upload:'):
                title = title[len('upload:'):]
            content = str(item.get('content', '') or '').strip()
            n += 1
            if meta.get('is_image') or content.startswith('User uploaded image:'):
                lines.append(f"{n}) **{title}** (image upload)")
                continue
            if len(content) > cls._UPLOAD_CHUNK_MAX_CHARS:
                content = content[:cls._UPLOAD_CHUNK_MAX_CHARS] + "…"
            lines.append(f"{n}) **{title}**\n{content}" if title else f"{n}) {content}")
        roster = cls._upload_roster_line(user_uploads)
        if roster:
            lines.append(
                "Recently uploaded files (full text retrievable by title with "
                f"get_full_document): {roster}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _context_value_nonempty(value: Any) -> bool:
        """True when an initial_context section value carries anything —
        handles list/dict/str sections and dataclass-shaped results
        (WebSearchResult's own has_results, since a WebSearchResult is
        neither falsy-by-default nor list/dict/str)."""
        if value is None:
            return False
        if hasattr(value, "has_results"):
            try:
                return bool(value.has_results)
            except Exception:
                return bool(value)
        return bool(value)

    def _decision_saw_admitted_evidence(
        self,
        session: "AgenticSearchSession",
        initial_context: Optional[Dict[str, Any]],
    ) -> bool:
        """B1 (2026-09-07, supersedes A4): tool results alone never prove the
        BASE retrieval evidence was seen — a turn where the loop only listed
        repo files still counted session.accumulated_context as "evidence
        seen" while the base retrieval's memories/notes/uploads never
        reached the decision prompt. True (reuse permitted) iff NO
        _RETRIEVAL_EVIDENCE_KEYS entry is non-empty in initial_context —
        background-only context (profile/summary/reflection digest lines,
        already visible to the decision prompt via context_inventory) may
        still be reused, same as before. Any non-empty retrieval key blocks
        reuse regardless of accumulated_context, since that key is real
        retrieved content the decision prompt's bounded digest never
        rendered.
        """
        return self._first_unmet_retrieval_key(initial_context, session) is None

    def _first_unmet_retrieval_key(
        self,
        initial_context: Optional[Dict[str, Any]],
        session: Optional["AgenticSearchSession"] = None,
    ) -> Optional[str]:
        """B1 (2026-09-07): the first _RETRIEVAL_EVIDENCE_KEYS entry that is
        non-empty in initial_context AND was never rendered into the decision
        prompt — names the evidence the decision round never saw, for
        session.reuse_skipped_reason. The one retrieval key the decision
        round CAN see is the pre-gathered base web result: A3 seeding
        (session.seeded_base_web) renders it verbatim into
        accumulated_context before round 2, so it is exempt (Fable referee,
        2026-09-07 — blocking on it would undo A3's purpose). Every other
        retrieval key reaches the decision prompt only as a bounded digest
        + counts, so any non-empty one blocks reuse."""
        if not initial_context:
            return None
        seeded_web = bool(session is not None and getattr(session, "seeded_base_web", False))
        for key in self._RETRIEVAL_EVIDENCE_KEYS:
            if key == "web_search_results" and seeded_web:
                continue
            if self._context_value_nonempty(initial_context.get(key)):
                return key
        return None

    def _compute_visible_sources(self, sections: List[str]) -> Dict[str, Any]:
        """A5 receipt: {"web_ids": ..., "sections": ...} at answer time."""
        source_map = getattr(self._tool_executor, "_current_web_source_map", None) or {}
        return {"web_ids": sorted(source_map.keys()), "sections": list(sections)}

    def _omitted_admitted_sections(
        self,
        initial_context: Optional[Dict[str, Any]],
        rendered_keys: Any,
    ) -> List[str]:
        """A5 receipt: _ADMITTED_EVIDENCE_KEYS entries that were non-empty in
        initial_context but not among rendered_keys (the sections the
        answering call actually rendered)."""
        if not initial_context:
            return []
        return [
            key for key in self._ADMITTED_EVIDENCE_KEYS
            if self._context_value_nonempty(initial_context.get(key)) and key not in rendered_keys
        ]

    @staticmethod
    def _build_native_action_prompt(
        query: str, forced_action, is_followup: bool = False
    ) -> str:
        """Native-tools forced-round directive appended to `_round_prompt`.

        A5 (round 2, BC-30-adjacent): includes the [RESOLVED FIELDS] block
        from `resolved_fields_note` for a calendar_create_event whose
        weekday+time `resolve_weekday_time` can ground deterministically —
        live, a forced round asked "how long does the study group run?"
        because the resolved start/end/recurrence were only ever applied
        POST-HOC to the model's own decision, never shown to the model
        itself, so it treated end_time as unstated.

        A6 (round 2): `is_followup` labels the query as a [USER ANSWER]
        instead of "The user asked" when this forced round was reached via
        the gate's prior-turn offer/retry/clarification-answer arm — the
        CURRENT text ("Yes 1 hour") is a short reply, not the request
        itself (the actual request lives in the action-context digest
        already present earlier in this prompt).
        """
        lead = f"[USER ANSWER] {query}" if is_followup else f"The user asked: {query}"
        text = (
            "[ACTION EXECUTION DIRECTIVE]\n"
            f"{lead}\n\n"
            f"Call propose_action now to do exactly this, filling in ALL content "
            f"fields from the request and conversation context above. For several "
            f"calendar events, use one events[] batch.\n\n"
            f"{_pending_cards_note()}"
        )
        if getattr(forced_action, "value", None) == "calendar_create_event":
            note = resolved_fields_note(query)
            if note:
                text += "\n\n" + note
        return text

    @staticmethod
    def _build_xml_action_force_prompt(
        query: str, forced_action, spec, reject_reason: Optional[str] = None,
        is_followup: bool = False,
    ) -> str:
        """Forced-round prompt for the XML-markers protocol: a concrete
        <action> example whose attributes are the spec's required/optional
        fields, one marker per item (a calendar request can carry several
        events — each is its own marker). `reject_reason` (F12, 2026-09-09):
        when the immediately-prior forced attempt was rejected, name why so
        the single retry is not a blind re-ask. `is_followup` (A6, round 2):
        see `_build_native_action_prompt`."""
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
        _reject_note = f" Your previous attempt was REJECTED: {reject_reason}." if reject_reason else ""
        _lead = f"[USER ANSWER] {query}" if is_followup else f"The user asked: {query}"
        _note = resolved_fields_note(query) if _type == "calendar_create_event" else ""
        return (
            f"{_lead}\n\n"
            f"Perform exactly this request by emitting one or more <action> "
            f"markers — nothing else, no prose. Syntax (fill every field from "
            f"the request and the conversation context above):\n"
            f'<action type="{_type}" {_attr_example} reason="user asked">optional details</action>\n'
            f"Timed datetimes are the user's LOCAL wall-clock time in ISO 8601 "
            f"WITHOUT a UTC offset (e.g. 2026-09-13T23:59:00) unless the source "
            f"names a zone.{_calendar_hint} If the request "
            f"covers multiple items (several events, several messages), emit one "
            f"<action> marker per item, each with ALL fields filled. Never INVENT a "
            f"time, date, or recipient that is not stated in the request or context — "
            f"a guessed value is rejected. Use "
            f'type="{_type}" exactly — do not substitute a different '
            f"action_type.{_reject_note}\n\n"
            f"{_pending_cards_note('emit the <action> marker')}"
            + (f"\n\n{_note}" if _note else "")
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
        - question-dominated (2026-09-07, B2): ≥2 lines ending in "?" with
          fewer than 2 non-question lines is a clarification list, not an
          answer
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
        # B2 (2026-09-07): a decision round can produce a well-punctuated,
        # non-promissory CLARIFICATION LIST instead of an answer — a turn-6
        # incident ("I checked the uploads directory ... Can you tell me:
        # 1. ... 2. ... 3. ...") passed every guard above (ends in "?", no
        # promissory verb, no loop-meta opener) and was reused as the final
        # response. ≥2 lines ending in "?" with fewer than 2 non-question
        # lines means the reply is mostly a list of questions, not an
        # answer, when admitted evidence exists to answer from.
        # Fable referee (2026-09-07): the live turn-6 reply also carried a
        # "possibilities" bullet list and an intro paragraph (six non-question
        # lines), so the line-ratio test alone would have let it through. An
        # ENUMERATED question list (two or more numbered/bulleted lines that
        # end in "?") is a clarification request by shape regardless of how
        # much prose surrounds it.
        _lines = [ln.strip() for ln in sanitized.splitlines() if ln.strip()]
        _q_lines = [ln for ln in _lines if ln.endswith("?")]
        _non_q_lines = [ln for ln in _lines if not ln.endswith("?")]
        _enum_q_lines = [ln for ln in _q_lines if self._ENUMERATED_LINE_RE.match(ln)]
        if len(_enum_q_lines) >= 2 or (len(_q_lines) >= 2 and len(_non_q_lines) < 2):
            logger.info(
                "[AgenticSearch] Decision answer is question-dominated "
                "(clarification list, not an answer) — falling back to "
                "synthesis call"
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

        # Receipts (2026-09-06, A5): what the answering call actually saw.
        # answer_call itself is set by the caller (decision_reuse never
        # reaches this method; final_synthesis vs error_fallback both call it,
        # so the caller's pre-set value must not be overwritten here).
        session.visible_sources = self._compute_visible_sources(
            getattr(session, "_final_prompt_section_headers", [])
        )
        _final_rendered_keys = set(self._FINAL_PROMPT_DIRECT_RENDERED_KEYS)
        if session.accumulated_context and session.accumulated_context.strip():
            _final_rendered_keys.add("web_search_results")
        session.omitted_sections = self._omitted_admitted_sections(
            initial_context, _final_rendered_keys
        )

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

        # User uploads (2026-09-07): the inventory used to omit this section
        # entirely, so the loop never learned which uploaded files EXIST and
        # searched reference_docs semantically for "the first assignment"
        # four times without finding Homework1-2.pdf. The roster titles are
        # rendered verbatim — a title is the argument get_full_document needs.
        user_uploads = initial_context.get('user_uploads', [])
        if user_uploads:
            _real = [u for u in user_uploads if not self._is_upload_roster_marker(u)]
            _roster = self._upload_roster_line(user_uploads)
            _line = f"- [USER UPLOADED ITEMS]: {len(_real)} admitted upload chunks"
            if _roster:
                _line += (
                    "; uploaded files retrievable IN FULL with "
                    f"get_full_document(title): {_roster}"
                )
            lines.append(_line)

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
    def _previous_user_query(initial_context: Optional[Dict[str, Any]]) -> str:
        """The most recent PRIOR user turn's raw text (A9, round 2) — used
        only to detect a git-push cue for an otherwise-unnamed document
        follow-up ("Can you take a look?" right after "Managed to push
        today and there is a new doc...")."""
        ordered = AgenticSearchController._ordered_recent_conversations(initial_context)
        if not ordered:
            return ""
        last = ordered[-1]
        return str(last.get('query', last.get('user', '')) or '')

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

    @staticmethod
    def _clip_preserving_claim_marker(text: str, limit: int) -> str:
        """Clip `text` to `limit` chars, but when it ends with the action-
        claim marker (2026-09-11, round 5, B13 sibling: core.action_claim_
        guard.UNVERIFIED_CLAIM_MARKER), clip the message BODY instead of
        the raw tail so the marker itself always survives — a hard
        char-cap silently dropping the exact flag a laundered claim needs
        would defeat the point of annotating it upstream in
        core/prompt/gatherer_memory.py. Ordinary (unmarked) text keeps the
        prior plain `text[:limit]` behavior."""
        suffix = "\n" + UNVERIFIED_CLAIM_MARKER
        if text.endswith(suffix):
            body = text[: -len(suffix)]
            return body[: max(0, limit - len(suffix))] + suffix
        return text[:limit]

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
                lines.append(
                    "  Daemon: "
                    + self._clip_preserving_claim_marker(assistant_msg, self._DIGEST_MSG_CHARS)
                )
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
    def _detect_tool_hints(query: str, prev_user_text: str = "") -> str:
        """Detect tool name mentions in a query and return usage hints.

        When the user explicitly mentions tools by name, the model should
        prioritize calling those tools rather than narrating about them.

        ``prev_user_text`` (A9, round 2): the immediately-prior USER turn's
        raw text, checked for the git-cued document hint below alongside
        the current query — default "" preserves the pre-A9 single-turn
        behavior for every other caller/test.
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
        # Note-save request (2026-09-10, A3): the gate's note-save Tier-1 arm
        # routes here via modes=['tools'] with no forced tool the way a
        # write-action round has — live: "jot down a note for this session:
        # …" reached the loop and the model replied that it can't save
        # notes at all, never calling create_daemon_note.
        try:
            from utils.query_checker import is_note_save_request
            if is_note_save_request(query):
                hints.append(
                    'Use the create_daemon_note tool to save this note now — '
                    'do not just say you will or that you cannot.'
                )
        except Exception:
            pass
        # Git-cued document hint (2026-09-10, round 2, A9): the CURRENT or
        # PREVIOUS user turn names a git action AND a doc noun together —
        # live: "Managed to push today and there is a new doc..." followed
        # by "Can you take a look?" resolved to a 5-day-old uploaded PDF via
        # the reuse pool, never the repository file just pushed.
        def _has_git_doc_cue(text: str) -> bool:
            return bool(text) and bool(_GIT_CUE_RE.search(text)) and bool(_DOC_NOUN_RE.search(text))
        if _has_git_doc_cue(query) or _has_git_doc_cue(prev_user_text):
            hints.append(
                'The document is probably a REPOSITORY file (the user pushed '
                'today): use file_list/file_grep on docs/ and the newest git '
                'commits BEFORE the upload pool; if the loop cannot identify '
                'it, ask which file.'
            )
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
        session: Optional[AgenticSearchSession] = None,
        prev_user_text: str = "",
    ) -> str:
        """Build prompt for iteration decision."""
        _now = datetime.now()
        _time_ctx = _now.strftime("Today is %A, %Y-%m-%d %H:%M. ")

        # Detect tool mentions and add hints
        _tool_hints = self._detect_tool_hints(query, prev_user_text)

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

    def _squeeze_query_part_for_ceiling(
        self, parts: List[str], prompt_ceiling: int
    ) -> Tuple[List[str], int]:
        """F8 last resort: middle-out ONLY the `[CURRENT USER QUERY]` part's
        body to bring the assembled prompt under `prompt_ceiling` once the
        section-level trim ladder above has run out of sections to drop.

        Uses `self.token_manager._middle_out` (model-aware tokenizer) when a
        token manager is attached; otherwise falls back to the same
        character-based head/tail fit (`utils.text_budget.fit_text_to_tokens`)
        using `self._estimate_tokens` as the counter. The user's own head
        words and the tail of the attachment survive; every other part is
        untouched, and the `* 5` ceiling multiplier is never changed here.

        Returns `(parts, total_tokens)` — the original list/count, unchanged,
        when there is no query part or squeezing it wouldn't help.
        """
        assembled_tokens = self._estimate_tokens("\n\n".join(parts))
        idx = next(
            (i for i, p in enumerate(parts) if p.startswith(_CURRENT_QUERY_HEADER)), None
        )
        if idx is None:
            return parts, assembled_tokens

        header_prefix = f"{_CURRENT_QUERY_HEADER}\n"
        body = parts[idx][len(header_prefix):]
        other_tokens = self._estimate_tokens(
            "\n\n".join(parts[:idx] + parts[idx + 1:])
        )

        name_match = _UPLOAD_TITLE_IN_TEXT_RE.search(body)
        doc_name = name_match.group(1) if name_match else "the attachment"

        def _marker(cut_chars: int) -> str:
            return (
                f"\n[… {cut_chars} characters of attached material omitted from this "
                f'call; full text retrievable via get_full_document(title="upload:{doc_name}") …]\n'
            )

        # fit_text_to_tokens/_middle_out size head+tail so head+GENERIC
        # marker+tail fits the budget handed to them; OUR marker (below) is
        # longer than their built-in "middle-out snipped N chars" one, so
        # swapping it in afterward can push the result back over budget
        # (observed: 40006/40000). Reserve this marker's own token cost
        # up front — using a worst-case (7-digit cut count) template so the
        # reservation doesn't itself depend on the exact cut size — so the
        # post-swap total still fits.
        _marker_budget_estimate = self._estimate_tokens(_marker(9_999_999))
        # Headroom for the header line + join separators; never below a
        # floor that would erase the query outright.
        target_body_tokens = max(
            prompt_ceiling - other_tokens - _marker_budget_estimate - 8, 256
        )

        try:
            if self.token_manager is not None and hasattr(self.token_manager, "_middle_out"):
                shrunk = self.token_manager._middle_out(body, target_body_tokens, force=True)
            else:
                shrunk = fit_text_to_tokens(body, target_body_tokens, self._estimate_tokens)
        except Exception as e:
            logger.warning(f"[AgenticSearch] Query-part squeeze failed, leaving prompt as-is: {e}")
            return parts, assembled_tokens

        if shrunk == body:
            return parts, assembled_tokens  # nothing to gain

        cut_chars = max(len(body) - len(shrunk), 0)
        marker = _marker(cut_chars)
        # Replace the generic middle-out marker (if the fit produced one)
        # with the get_full_document-aware one so the model knows how to
        # retrieve the omitted material; append it when no marker fit at all.
        if _MIDDLE_OUT_SNIP_RE.search(shrunk):
            shrunk = _MIDDLE_OUT_SNIP_RE.sub(marker, shrunk, count=1)
        else:
            shrunk = shrunk + marker

        new_parts = list(parts)
        new_parts[idx] = header_prefix + shrunk
        new_tokens = self._estimate_tokens("\n\n".join(new_parts))
        logger.warning(
            f"[AgenticSearch] Squeezed [CURRENT USER QUERY] to fit ceiling: "
            f"{assembled_tokens} -> {new_tokens} tokens ({prompt_ceiling} ceiling), "
            f"{cut_chars} characters cut from the query body"
        )
        return new_parts, new_tokens

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

            # User uploads (2026-09-07): never rendered here before — the
            # final synthesis answered "I don't have the actual HW1
            # instructions" with the roster naming Homework1-2.pdf sitting in
            # the base prompt it never saw.
            user_uploads = initial_context.get('user_uploads', [])
            if user_uploads:
                uu_text = self._format_user_uploads(user_uploads)
                if uu_text:
                    parts.append(f"[USER UPLOADED ITEMS]\n{uu_text}")

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

        # Add search results. Header is source-kind-neutral (2026-09-06):
        # accumulated_context can hold memory/file/computation/fetched-page
        # results with no web search at all (or pre-gathered base web results
        # seeded by A3 with no agentic web round) — the old
        # "[WEB SEARCH RESULTS - N rounds]" label plus the unconditional
        # "every claim MUST cite [WEB_N]" instruction below misdescribed a
        # memory-only loop's own tool results as web search.
        if session.accumulated_context:
            parts.append(
                f"[TOOL RESULTS - {len(session.rounds)} rounds] (each block is "
                f"labeled by its source kind: web search, memory, file, "
                f"computation, fetched page, pre-gathered web)\n"
                f"{session.accumulated_context}"
            )

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

        # Instructions. has_web is keyed off actual WEB sources having been
        # numbered (2026-09-06) — accumulated_context alone no longer implies
        # web evidence (a memory/file/computation-only loop was getting the
        # "every claim MUST cite [WEB_N]" instruction it had nothing to cite).
        has_web = bool(getattr(self._tool_executor, "_current_web_source_map", None))
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

            # F8: the section ladder is exhausted but a single oversized
            # [CURRENT USER QUERY] part (a large attachment) can still be
            # over ceiling on its own — squeeze that part's body as a last
            # resort before giving up.
            if total_tokens > prompt_ceiling:
                parts, total_tokens = self._squeeze_query_part_for_ceiling(parts, prompt_ceiling)

            if total_tokens > prompt_ceiling:
                logger.warning(
                    f"[AgenticSearch] Final prompt still over ceiling after trimming: "
                    f"{total_tokens}/{prompt_ceiling} tokens"
                )

        # Provenance receipt (2026-09-06): record the top-level bracketed
        # section headers that actually made it into the assembled prompt
        # (post-trim), for session.visible_sources["sections"]. Only the
        # FIRST LINE of each top-level block is checked — nested content
        # (e.g. inline [WEB_N] citation lines inside the tool-results block)
        # must never be mistaken for a top-level section header.
        _section_headers: List[str] = []
        for _part in parts:
            _first_line = _part.split("\n", 1)[0]
            _hm = re.match(r"^\[[^\]]+\]", _first_line)
            if _hm:
                _section_headers.append(_hm.group(0))
        session._final_prompt_section_headers = _section_headers

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
