"""
# core/prompt/token_manager.py

Module Contract
- Purpose: Token management and budget control for prompt building system.
- Inputs:
  - extract_text_from_context(context: Dict[str, Any]) -> str
  - trim_to_budget(context: Dict[str, Any], budget: int) -> Dict[str, Any]
  - count_tokens(text: str) -> int
- Outputs:
  - Extracted text content from structured context
  - Budget-compliant context with trimmed content
  - Token counts for text validation
- Behavior:
  - Extracts text from various context sections (conversations, memories, facts, reference_docs, narrative_state, etc.)
  - Applies priority-based trimming when content exceeds token budget
  - Uses middle-out compression for mildly oversized text blocks (1x-3x over limit).
    _middle_out never returns a result larger than its input (829→900 "compression" observed
    pre-2026-07-25) and its snip marker emits real newlines (was a literal "\\n")
  - Heavily oversized items (≥3x) are pre-compressed by LLM in builder._llm_compress_oversized() [NEW 2026-03-26]
  - Budget resolution (2026-07-25): API-model budgets cap at PROMPT_TOKEN_BUDGET_DEFAULT (the
    ctx-fraction path can only LOWER it — it previously overrode the experiment-validated 10000
    up to 15360); context limits resolve via model_manager.MODEL_CONTEXT_LIMITS (no hardcoded 128K)
  - Logs the TRUE context total including unmetered sections (metered usage under-reports ~25%)
    without changing trim semantics
  - Preserves most important content while respecting limits
  - narrative_state: Priority 8 (high), capped at NARRATIVE_STATE_MAX_TOKENS (500) [NEW 2026-01-17]
- Dependencies:
  - models.tokenizer_manager.TokenizerManager (token counting)
  - utils.logging_utils (logging)
- Side effects:
  - Logging of token usage and trimming actions

Priority Order (highest to lowest — 2026-08-14: meters the keys the formatter
RENDERS; the combined summaries/reflections keys were dead-metered before, so
the four rendered summary/reflection sections were unmetered AND untrimmable):
  - stm_summary: 10 (metadata only, never trimmed)
  - user_profile: 9 (critical identity, naturally bounded)
  - narrative_state: 8 (temporal grounding, 500 token cap) / web_search_results: 8
  - recent_conversations / graph_context / unresolved_threads: 7
  - google_calendar / upcoming_schedule: 7
  - semantic_chunks / personal_notes / user_uploads / disambiguation_notes: 6
  - reference_docs / memories: 5
  - procedural_skills / facts: 4
  - recent_summaries / semantic_summaries / proposed_features / git_commits / proactive_insights: 3
  - git_commits from local repository status retrieval: 8 in both budget passes
  - recent_reflections / semantic_reflections / daemon_self_notes / dreams / codebase_changes: 2
  - wiki: 1
UNRENDERED_CONTEXT_KEYS (summaries, reflections, stm_summary, memory_id_map) are
excluded from metering AND the true-total log; test_budget_meters_rendered_sections.py
asserts formatter↔PRIORITY_ORDER parity so a new rendered section can't go unmetered.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from utils.logging_utils import get_logger
from utils.text_budget import fit_text_to_tokens
from .base import _parse_bool

# Module-level import verified cycle-free (2026-09-06): knowledge.web_search_manager
# does not import core.prompt.* (or anything that does) at module level, so this
# is a plain dependency, not one of the four documented lazy-import cases in
# CLAUDE.md "Import doctrine".
from knowledge.web_search_manager import render_numbered_web_sources, trim_web_search_result

logger = get_logger("prompt_token_manager")

# Constants for token management
PRIORITY_ORDER = [
    ("stm_summary",          10),  # Highest priority - STM context should never be trimmed
    ("user_profile",          9),  # Critical identity context, naturally bounded (~1-3K)
    ("narrative_state",       8),  # Temporal grounding - high priority, capped at 500 tokens
    ("recent_conversations",  7),
    ("graph_context",         7),  # Knowledge graph entities, small (~200-800 tokens)
    ("unresolved_threads",    7),  # Continuity threads, small (~100-500 tokens)
    ("semantic_chunks",       6),
    ("personal_notes",        6),  # User's Obsidian notes - high priority
    ("user_uploads",          6),  # User explicitly uploaded content
    ("reference_docs",        5),  # User uploaded reference documents
    ("memories",              5),
    ("web_search_results",    8),  # Real-time web content — high priority, user explicitly asked for current info
    ("google_calendar",       7),  # Real-time calendar events, small + time-sensitive
    ("relevant_emails",       7),  # Relevant emails from Gmail/Outlook, small + real-time
    ("upcoming_schedule",     7),  # Gated schedule events, small
    ("disambiguation_notes",  6),  # Cross-session phrase disambiguation, small
    ("procedural_skills",     4),  # Adaptive workflows
    ("facts",                 4),
    # The formatter renders the SPLIT summary/reflection keys; the combined
    # "summaries"/"reflections" keys are never rendered. Until 2026-08-14 the
    # combined keys were metered here instead — so the four rendered sections
    # were invisible to the budget AND untrimmable (turn observed at 17.5K
    # true tokens against a 10K budget).
    ("recent_summaries",      3),
    ("semantic_summaries",    3),
    ("proposed_features",     3),  # Code proposals (trimmed before core context)
    ("git_commits",           3),  # Project commit history
    ("proactive_insights",    3),  # Cross-domain insights, naturally bounded
    ("recent_reflections",    2),  # Below summaries
    ("semantic_reflections",  2),
    ("daemon_self_notes",     2),  # Non-ground-truth self notes, small
    ("dreams",                2),  # Still included; trimmed early if needed
    ("codebase_changes",      2),  # First message only, session diff
    ("wiki",                  1),
]

# Context keys that are inputs/intermediates the formatter never renders —
# excluded from both metering and the true-total visibility log so unrendered
# content can't inflate either number. ("summaries"/"reflections" are the
# legacy combined keys kept for back-compat; the split keys above are the
# rendered ones.)
UNRENDERED_CONTEXT_KEYS = frozenset({
    "summaries", "reflections", "stm_summary", "memory_id_map",
    "web_search_decision",
})

# Max tokens for narrative_state section (temporal grounding)
NARRATIVE_STATE_MAX_TOKENS = int(os.getenv("NARRATIVE_STATE_MAX_TOKENS", "500"))

# Configuration loading helpers
try:
    from config.app_config import config as _APP_CFG
    _MEM_CFG = (_APP_CFG.get("memory") or {})
except (ImportError, AttributeError):
    _MEM_CFG = {}

def _cfg_int(key: str, default_val: int) -> int:
    try:
        v = _MEM_CFG.get(key, default_val)
        return int(v) if v is not None else int(default_val)
    except (ValueError, TypeError):
        return int(default_val)

# Token limits and configuration
ENABLE_MIDDLE_OUT = _parse_bool(os.getenv("ENABLE_MIDDLE_OUT", "1"))
USER_INPUT_MAX_TOKENS = int(os.getenv("USER_INPUT_MAX_TOKENS", "4096"))
MEMORY_ITEM_MAX_TOKENS = int(os.getenv("MEMORY_ITEM_MAX_TOKENS", "512"))
SEMANTIC_ITEM_MAX_TOKENS = int(os.getenv("SEMANTIC_ITEM_MAX_TOKENS", "800"))
# The profile gatherer already selects up to 3000 tokens of identity context.
# It is a whole section, not an individual retrieved snippet; do not silently
# reduce that allocation when enforcing measured snippet caps.
USER_PROFILE_MAX_TOKENS = 3000
# Per-field cap for the QUERY of a conversation-shaped item ({query, response}
# corpus entries — recent_conversations and the memories fallback shape).
# 2026-09-04: the formatter renders "User: <query>\nDaemon: <response>" but
# _extract_text read only the RESPONSE key, so a 401,972-char attachment-heavy
# query rode into the NEXT turn's [RECENT CONVERSATION] unmetered and
# untrimmed — a one-line follow-up question billed 146K prompt tokens and the
# attachment retest 279K. The query field is now capped on its own (head+tail
# middle-out keeps the user's words and the end of the paste) and metering
# counts the rendered pair exactly as the formatter emits it.
CONVERSATION_QUERY_MAX_TOKENS = int(os.getenv("CONVERSATION_QUERY_MAX_TOKENS", "600"))

# Structured-section adapters (2026-09-06 evidence-transport fix).
#
# Verified defect: the non-list branch of _manage_token_budget used to do
# `item_text = str(val)` for ANY non-list section, and — once it exceeded
# SEMANTIC_ITEM_MAX_TOKENS (800) — wrote that middle-out'd STRING back into
# `trimmed[name]`. A WebSearchResult dataclass repr for more than about a
# page of content always exceeds 800 tokens, so `web_search_results` silently
# became a `str`, and the formatter's inline web-search block
# (`hasattr(web_search, 'has_results')`) then rendered NOTHING — the base
# prompt's web evidence vanished with no error. The generic string
# write-back itself dates to 2025-11-28 (d7e0f7b); `web_search_results` was
# only added to PRIORITY_ORDER on 2026-03-26 (e0f08e0), which is when this
# became live-reachable.
#
# STRUCTURED_SECTION_ADAPTERS lets a registered section be METERED against
# what will actually be rendered (never a str(dataclass) repr) and, if that
# exceeds WEB_SEARCH_SECTION_MAX_TOKENS, SHRUNK via a bounded ladder instead
# of ever being coerced to a string. Each entry is
# (meter_fn(val, count_tokens) -> int, shrink_fn(val, step) -> val|None).
WEB_SEARCH_SECTION_MAX_TOKENS = int(os.getenv("WEB_SEARCH_SECTION_MAX_TOKENS", "3200"))

# Fixed (max_sources, max_chars_per_source) ladder the web_search_results
# adapter shrinks down when the rendered section is still too large. Step 0
# matches render_numbered_web_sources' own default bound (8, 2000), so the
# very first application does not change the metered size by itself — it
# only locks the same bound into the stored value; real reduction begins at
# step 1. Past the last entry the section is dropped (None), never stringified.
_WEB_SEARCH_SHRINK_LADDER: List[Tuple[int, int]] = [
    (8, 2000), (6, 1500), (4, 1000), (2, 800), (1, 600),
]


def _meter_web_search_results(val: Any, count_tokens: Callable[[str], int]) -> int:
    """meter_fn for STRUCTURED_SECTION_ADAPTERS['web_search_results']: token
    count of the EXACT text render_numbered_web_sources would produce for
    this value (the same call the formatter's inline web-search block makes),
    never a str(dataclass) repr. Empty/errored results (has_results False)
    count 0 so the section passes through untouched."""
    if not getattr(val, "has_results", False):
        return 0
    pages = getattr(val, "pages", None) or []
    lines, _ = render_numbered_web_sources(pages, max_sources=8, max_chars_per_source=2000)
    if not lines:
        return 0
    return count_tokens("\n\n".join(lines))


def _shrink_web_search_results(val: Any, step: int) -> Optional[Any]:
    """shrink_fn for STRUCTURED_SECTION_ADAPTERS['web_search_results']: apply
    the next rung of _WEB_SEARCH_SHRINK_LADDER; past the end returns None —
    the section is dropped, never coerced to a string."""
    if step < 0 or step >= len(_WEB_SEARCH_SHRINK_LADDER):
        return None
    max_sources, max_chars = _WEB_SEARCH_SHRINK_LADDER[step]
    return trim_web_search_result(val, max_sources=max_sources, max_chars_per_source=max_chars)


STRUCTURED_SECTION_ADAPTERS: Dict[str, Tuple[Callable[[Any, Callable[[str], int]], int], Callable[[Any, int], Optional[Any]]]] = {
    "web_search_results": (_meter_web_search_results, _shrink_web_search_results),
}


class TokenManager:
    """Handles token counting, budgets, and text compression for prompt building."""

    def __init__(self, model_manager, tokenizer_manager, token_budget: int):
        self.model_manager = model_manager
        self.tokenizer_manager = tokenizer_manager
        self.token_budget = token_budget
        self._prompt_token_usage = 0

    def get_token_count(self, text: str, model_name: str) -> int:
        """Delegate to tokenizer_manager; keeps compatibility with your models."""
        return self.tokenizer_manager.count_tokens(text or "", model_name)

    # Single source of truth for which dict field carries an item's text.
    # _extract_text READS this key and the oversized-item compressor WRITES
    # back to it — they must agree (2026-09-01: the write-back had its own
    # list with 'query' in it, so a conversation entry's compressed RESPONSE
    # overwrote its QUERY while the full response stayed — the rendered entry
    # DOUBLED while metering thought it shrank; latent since 2025-11, first
    # fired when insight-mode reports became oversized recent turns).
    _TEXT_KEYS = ("content", "text", "response", "filtered_content")

    def _text_key(self, item: dict) -> Optional[str]:
        for key in self._TEXT_KEYS:
            if key in item and item[key]:
                return key
        return None

    @staticmethod
    def _is_conversation_item(item: Any) -> bool:
        """A {query, response} corpus entry with no pre-rendered content/text
        field — the shape formatter.mem_parts renders as User:/Daemon: lines."""
        return (
            isinstance(item, dict)
            and not item.get("content")
            and not item.get("text")
            and bool(item.get("query"))
        )

    @staticmethod
    def _render_conversation_item(item: dict) -> str:
        """Mirror formatter.mem_parts' query/response fallback byte-for-byte so
        metering sees what the prompt will actually carry."""
        q = str(item.get("query", "") or "").strip()
        r = str(item.get("response", "") or "").strip()
        if q and r:
            return f"User: {q}\nDaemon: {r}"
        if r:
            return f"Daemon: {r}"
        return f"User: {q}"

    def _extract_text(self, item: Any) -> str:
        """Extract text from various item formats for token counting."""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            if self._is_conversation_item(item):
                return self._render_conversation_item(item)
            key = self._text_key(item)
            return str(item[key]) if key else str(item)
        return str(item)

    def _cap_conversation_item(self, item: dict, response_max_tokens: int, model_name: str) -> dict:
        """Cap a conversation entry's query and response FIELDS independently
        (2026-09-04). Each field is middle-out'd in place under its own cap so
        the formatter's rendering shape is untouched; a joined blob is never
        written back into a single key (the 2026-09-01 write-back bug class)."""
        capped = dict(item)
        q = str(capped.get("query", "") or "")
        if q and self.get_token_count(q, model_name) > CONVERSATION_QUERY_MAX_TOKENS:
            capped["query"] = self._middle_out(q, CONVERSATION_QUERY_MAX_TOKENS, force=True)
        r = str(capped.get("response", "") or "")
        if r and self.get_token_count(r, model_name) > response_max_tokens:
            capped["response"] = self._middle_out(r, response_max_tokens, force=True)
        return capped

    def _middle_out(self, text: str, max_tokens: int, head_ratio: float = 0.6, force: bool = False) -> str:
        """Compress text by keeping the head and tail, trimming the middle.

        Slice by characters, then measure candidates with the active tokenizer.
        The marker is included in the cap; four characters per token is only
        an initial estimate, never a budget guarantee.

        Args:
            text: Text to potentially compress
            max_tokens: Maximum tokens for this item
            head_ratio: Ratio of head to tail (default 0.6 = 60% head, 40% tail)
            force: If True, apply compression regardless of prompt size. If False (default),
                   only compress when total prompt would exceed 20K tokens.
        """
        if not ENABLE_MIDDLE_OUT:
            return text
        if max_tokens <= 0:
            return ""

        # Only apply middle-out if we're above the token budget
        # unless explicitly forced
        if not force and hasattr(self, '_prompt_token_usage'):
            if self._prompt_token_usage < self.token_budget:
                return text

        model_name = "default"
        try:
            if hasattr(self.model_manager, "get_active_model_name"):
                model_name = self.model_manager.get_active_model_name()
            toks = self.get_token_count(text or "", model_name)
        except (AttributeError, RuntimeError):
            toks = len((text or "").split())
        if toks <= max_tokens:
            return text
        return fit_text_to_tokens(
            text or "", max_tokens,
            lambda value: self.get_token_count(value, model_name), head_ratio,
        )

    def _manage_token_budget(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trim sections in increasing priority order until we fit within token budget.
        For lists, we remove items from the tail in conservative chunks (25%) to
        avoid over-trimming; for strings, we blank the whole section if needed.
        """
        model_name = self.model_manager.get_active_model_name()
        trimmed = dict(context)
        current_tokens = 0
        # Per-section ladder position for STRUCTURED_SECTION_ADAPTERS entries,
        # shared between the first pass (which may walk several rungs in one
        # go to satisfy WEB_SEARCH_SECTION_MAX_TOKENS) and the second pass
        # (which advances exactly one rung per pass) so they never repeat or
        # skip a step.
        _adapter_shrink_steps: Dict[str, int] = {}

        # Helper to count tokens for any item
        def _item_tokens(item: Any) -> int:
            text = self._extract_text(item)
            return self.get_token_count(text, model_name)

        # First pass: optimistic inclusion with per-item compression
        priority_order = list(PRIORITY_ORDER)
        if any(
            isinstance(item, dict)
            and item.get("metadata", {}).get("retrieval_source") == "local_git"
            for item in context.get("git_commits", []) or []
        ):
            # Current repository status is direct evidence for this turn.
            # Admit it before historical conversation, within the same budget.
            priority_order = [(name, 8 if name == "git_commits" else priority)
                              for name, priority in priority_order]
        for name, _prio in sorted(priority_order, key=lambda entry: entry[1], reverse=True):
            val = trimmed.get(name)
            if not val:
                continue

            # Special handling: stm_summary is metadata, not content - preserve without token counting
            if name == "stm_summary":
                logger.debug(f"[TOKEN BUDGET] Preserving stm_summary (metadata, no token cost)")
                continue

            # Special handling: narrative_state has a hard cap of 500 tokens
            if name == "narrative_state" and val:
                item_text = str(val)
                t = self.get_token_count(item_text, model_name)
                if t > NARRATIVE_STATE_MAX_TOKENS:
                    item_text = self._middle_out(item_text, NARRATIVE_STATE_MAX_TOKENS, force=True)
                    t = self.get_token_count(item_text, model_name)
                    trimmed[name] = item_text
                    logger.debug(f"[TOKEN BUDGET] Capped narrative_state to {NARRATIVE_STATE_MAX_TOKENS} tokens")
                current_tokens += t
                continue

            logger.debug(f"[TOKEN BUDGET] Processing section '{name}' with {len(val) if isinstance(val, list) else 1} items, current_tokens={current_tokens}")
            if isinstance(val, list):
                kept = []
                for i, item in enumerate(val):
                    max_item_tokens = MEMORY_ITEM_MAX_TOKENS if name == "memories" else SEMANTIC_ITEM_MAX_TOKENS
                    if self._is_conversation_item(item) and ENABLE_MIDDLE_OUT:
                        # Conversation-shaped entry ({query, response}): cap the
                        # two FIELDS independently and meter the rendered pair
                        # (2026-09-04) — see CONVERSATION_QUERY_MAX_TOKENS.
                        _before = self.get_token_count(self._extract_text(item), model_name)
                        item = self._cap_conversation_item(item, max_item_tokens, model_name)
                        item_text = self._extract_text(item)
                        t = self.get_token_count(item_text, model_name)
                        if t < _before:
                            logger.debug(f"[MIDDLE-OUT] Capped conversation entry {name}[{i}]: {_before} → {t} tokens")
                    else:
                        # Get item text and token count
                        item_text = self._extract_text(item)
                        t = self.get_token_count(item_text, model_name)

                        # Apply middle-out to oversized individual items before considering removal
                        if t > max_item_tokens and ENABLE_MIDDLE_OUT:
                            compressed_text = self._middle_out(item_text, max_item_tokens, force=True)
                            # Update item with compressed text — written back to
                            # the SAME key _extract_text read (see _TEXT_KEYS).
                            if isinstance(item, dict):
                                compressed_item = dict(item)
                                _wb_key = self._text_key(compressed_item)
                                if _wb_key:
                                    compressed_item[_wb_key] = compressed_text
                                item = compressed_item
                            else:
                                item = compressed_text
                            # Recount tokens after compression
                            # A dictionary with no recognized text key was
                            # not changed. Meter the object actually retained.
                            t = _item_tokens(item)
                            logger.debug(f"[MIDDLE-OUT] Compressed {name}[{i}]: {self.get_token_count(item_text, model_name)} → {t} tokens")

                    if current_tokens + t <= self.token_budget:
                        kept.append(item)
                        current_tokens += t
                    else:
                        if name == "memories":
                            logger.debug(f"[TOKEN BUDGET] Stopped adding memories at item {i}/{len(val)}: budget={self.token_budget}, current={current_tokens}, item_tokens={t}")
                        break
                if name == "memories":
                    logger.debug(f"[TOKEN BUDGET] Kept {len(kept)}/{len(val)} memories, budget={self.token_budget}, used={current_tokens}")
                trimmed[name] = kept
            elif isinstance(val, dict):
                # Dict-shaped sections (e.g. a raw {"pages": [...]} web
                # result, or codebase_changes) are metered but NEVER
                # coerced/written back — unchanged from prior behavior. This
                # is explicitly NOT an adapter case even for
                # "web_search_results": only the dataclass shape is adapted.
                item_text = str(val)
                t = self.get_token_count(item_text, model_name)
                if current_tokens + t <= self.token_budget:
                    current_tokens += t
                else:
                    # We'll consider dropping this later in the second pass.
                    pass
            elif name in STRUCTURED_SECTION_ADAPTERS:
                # Registered structured section (2026-09-06 evidence-transport
                # fix): meter what will ACTUALLY be rendered and shrink down a
                # fixed ladder if it's still too large — never str()-and-
                # write-back (see STRUCTURED_SECTION_ADAPTERS doc above).
                meter_fn, shrink_fn = STRUCTURED_SECTION_ADAPTERS[name]

                def _count_fn(text: str, _mn: str = model_name) -> int:
                    return self.get_token_count(text, _mn)

                t = meter_fn(val, _count_fn)
                step = _adapter_shrink_steps.get(name, 0)
                while t > WEB_SEARCH_SECTION_MAX_TOKENS:
                    shrunk = shrink_fn(val, step)
                    step += 1
                    if shrunk is None:
                        val = None
                        t = 0
                        break
                    val = shrunk
                    t = meter_fn(val, _count_fn)
                _adapter_shrink_steps[name] = step
                trimmed[name] = val
                if val is not None:
                    logger.debug(
                        f"[TOKEN BUDGET] Structured section '{name}' metered "
                        f"at {t} tokens (adapter step={step})"
                    )
                if val is not None and current_tokens + t <= self.token_budget:
                    current_tokens += t
                else:
                    # Over budget (or dropped by the ladder already) — the
                    # second pass continues shrinking/dropping if needed.
                    pass
            else:
                # Plain string sections (like wiki content): apply middle-out
                # if too large. Any OTHER unregistered structured object type
                # is metered by str() length but NEVER written back — an
                # unmeterable section must not be silently stringified into
                # the prompt either.
                item_text = str(val)
                t = self.get_token_count(item_text, model_name)
                if isinstance(val, str):
                    section_cap = USER_PROFILE_MAX_TOKENS if name == "user_profile" else SEMANTIC_ITEM_MAX_TOKENS
                    if t > section_cap and ENABLE_MIDDLE_OUT:
                        item_text = self._middle_out(item_text, section_cap, force=True)
                        t = self.get_token_count(item_text, model_name)
                        trimmed[name] = item_text
                        logger.debug(f"[MIDDLE-OUT] Compressed {name} section: {self.get_token_count(str(val), model_name)} → {t} tokens")
                else:
                    logger.debug(
                        f"[TOKEN BUDGET] Unregistered structured section "
                        f"'{name}' (type={type(val).__name__}) — metering by "
                        f"str() length, never writing back a stringified value"
                    )

                if current_tokens + t <= self.token_budget:
                    current_tokens += t
                else:
                    # We'll consider dropping this later in the second pass.
                    pass

        # If we're still over (due to some large strings), trim by priority
        def _total_tokens(ctx: Dict[str, Any]) -> int:
            total = 0
            for name, _ in PRIORITY_ORDER:
                v = ctx.get(name)
                if not v:
                    continue
                # Skip stm_summary - it's metadata, not content
                if name == "stm_summary":
                    continue
                if isinstance(v, list):
                    for it in v:
                        total += _item_tokens(it)
                elif name in STRUCTURED_SECTION_ADAPTERS and not isinstance(v, (str, dict)):
                    meter_fn, _shrink_fn = STRUCTURED_SECTION_ADAPTERS[name]
                    total += meter_fn(v, lambda text, _mn=model_name: self.get_token_count(text, _mn))
                else:
                    total += self.get_token_count(str(v), model_name)
            return total

        usage = _total_tokens(trimmed)
        logger.debug(f"[PROMPT] Token budget (pre-trim check): {usage}/{self.token_budget}")

        if usage > self.token_budget:
            # Second pass: iterative trim from lowest priority upward (max 3 passes)
            for _pass in range(3):
                if usage <= self.token_budget:
                    break
                for name, prio in sorted(priority_order, key=lambda x: x[1]):  # low → high
                    v = trimmed.get(name)
                    if not v:
                        continue

                    if isinstance(v, list) and v:
                        # Drop a conservative slice from the tail
                        drop_n = max(1, int(len(v) * 0.25))
                        trimmed[name] = v[:-drop_n]
                    elif name in STRUCTURED_SECTION_ADAPTERS and not isinstance(v, (str, dict)):
                        # Adapter section: advance exactly ONE ladder step per
                        # pass (never blanked/stringified whole); past the
                        # ladder's end the shrink_fn drops it to None.
                        _, shrink_fn = STRUCTURED_SECTION_ADAPTERS[name]
                        step = _adapter_shrink_steps.get(name, 0)
                        trimmed[name] = shrink_fn(v, step)
                        _adapter_shrink_steps[name] = step + 1
                    elif isinstance(v, (str, dict)) and v:
                        # Structured sections (e.g. codebase_changes) must
                        # retain their schema; drop them whole if necessary.
                        trimmed[name] = {} if isinstance(v, dict) else ""

                    usage = _total_tokens(trimmed)
                    if usage <= self.token_budget:
                        break

            if usage > self.token_budget:
                logger.warning(
                    f"[TOKEN BUDGET] Still over budget after 3 trim passes: "
                    f"{usage}/{self.token_budget} tokens"
                )

        logger.debug(f"[PROMPT] Token budget: {usage}/{self.token_budget}")
        self._prompt_token_usage = usage

        # Visibility-only true total for anything still outside PRIORITY_ORDER
        # (visual_memories, note_images, …) plus the deliberately-unmetered
        # stm_summary. 2026-08-14: the four rendered summary/reflection split
        # keys + calendar/schedule moved INTO metering (they were the bulk of
        # the ~25%-and-worse under-report — a 17.5K true prompt against a 10K
        # budget), so this residual should now be small; a large number here
        # means a new rendered section was added without a PRIORITY_ORDER row.
        try:
            metered_names = {name for name, _ in PRIORITY_ORDER}
            unmetered = 0
            for key, v in trimmed.items():
                if key in metered_names and key != "stm_summary":
                    continue
                # Unrendered inputs and _-prefixed metadata never reach the
                # prompt — counting them would inflate the "true total".
                if key in UNRENDERED_CONTEXT_KEYS or key.startswith("_"):
                    continue
                if not v:
                    continue
                if isinstance(v, list):
                    unmetered += sum(_item_tokens(it) for it in v)
                elif isinstance(v, (str, dict)):
                    unmetered += self.get_token_count(
                        self._extract_text(v) if isinstance(v, dict) else v, model_name
                    )
            if unmetered:
                logger.info(
                    f"[TOKEN BUDGET] True context total ≈ {usage + unmetered} tokens "
                    f"(metered {usage}/{self.token_budget} + {unmetered} unmetered)"
                )
        except Exception as e:
            logger.debug(f"[TOKEN BUDGET] True-total accounting failed (non-fatal): {e}")
        return trimmed
