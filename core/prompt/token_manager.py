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
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from utils.logging_utils import get_logger
from utils.text_budget import fit_text_to_tokens
from .base import _parse_bool
from memory.fact_source import is_test_turn, probes_hidden_from_turn
# ONE session-boundary rule (formatter owns it; builder._session_truth_from_probe
# applies the same walk). Module-level import verified cycle-free 2026-10-15.
from .formatter import (
    _detect_session_boundary, _parse_entry_timestamp, SESSION_BOUNDARY_GAP_HOURS,
)

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
    # 2026-10-08 (BC-24): equal-priority sections are admitted in LIST order, so
    # the small time-sensitive ones sit BEFORE history — a 20-turn
    # recent_conversations + session_timeline filled 9,987 of the 10,000 budget
    # and emptied the calendar and threads (they are small; history flexes).
    ("google_calendar",       7),  # Real-time calendar events, small + time-sensitive
    ("unresolved_threads",    7),  # Continuity threads, small (~100-500 tokens)
    # 2026-10-12 (B4): the user's OWN statements relevant to a factual-recall
    # question about them - tiny (<=3 short lines), outranks assistant history.
    ("user_stated_evidence",  8),
    # 2026-10-20 (BC-75/BC-52/BC-46/BC-99): [YOUR STATED FIGURES] -- the user's own
    # figures with date + modality, <=12 short lines, capped at claim_ledger.max_block_tokens
    # upstream. Same tier as the user's own recalled statements: outranks assistant history.
    ("stated_figures",        8),
    # 2026-10-27 (BC-103/BC-100): [YOUR DAY COUNTS] -- <=day_counts.max_lines (3) short lines,
    # the user's own day counts already projected to today. Same tier as the user's figures.
    ("day_counts",            8),
    ("recent_conversations",  7),
    # 2026-09-27 (BC-46, BC-30, BC-51): [EARLIER TODAY] recall-timeline lines
    # a truncated [RECENT CONVERSATION] window dropped (temporal_recall-only,
    # capped at 40 short lines) + the scalar session-truth values backing the
    # truncated-header honesty fix and the [TIME CONTEXT] "session began"
    # line. The scalars cost only a few tokens each (str() of a bool/int/
    # timestamp); rows exist to keep formatter↔PRIORITY_ORDER parity green
    # (test_budget_meters_rendered_sections.py fails on any unmetered
    # rendered-key read, scalar or not).
    ("session_timeline",        7),
    ("recent_window_truncated", 7),
    ("session_turns_total",     7),
    ("session_started_at",      7),
    ("graph_context",         7),  # Knowledge graph entities, small (~200-800 tokens)
    ("semantic_chunks",       6),
    ("personal_notes",        6),  # User's Obsidian notes - high priority
    ("user_uploads",          6),  # User explicitly uploaded content
    ("reference_docs",        5),  # User uploaded reference documents
    ("memories",              5),
    ("web_search_results",    8),  # Real-time web content — high priority, user explicitly asked for current info
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
    # 2026-10-24 (BC-46): the one-line "[NOT SHOWN FOR SPACE]" note naming
    # sections this budget emptied (computed AFTER the passes below; ~30 tokens, deliberately outside the
    # budget-compliance total).
    ("budget_omitted_note",  10),
]

# Small, time-sensitive sections: metered before history in the first pass
# (list position in PRIORITY_ORDER) and trimmed last among their priority ties
# in the second pass.
_SMALL_TIME_SENSITIVE = frozenset({"google_calendar", "unresolved_threads"})

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

def _cfg_float(key: str, default_val: float) -> float:
    try:
        v = _MEM_CFG.get(key, default_val)
        return float(v) if v is not None else float(default_val)
    except (ValueError, TypeError):
        return float(default_val)

# 2026-10-15: PRIOR-session recent conversation may use at most this share of
# the prompt budget. recent_conversations (priority 7) is admitted before
# memories/summaries/reflections, so 26 prior-session turns (~6.5K of a 10K
# budget) emptied every long-term section (live probe 14:48; 47% of October
# builds had 0 memories). Current-session turns are never capped by this.
RECENT_PRIOR_SESSION_MAX_SHARE = _cfg_float("prompt_recent_prior_session_max_share", 0.40)
# Section-wide cap (current + prior) on recent_conversations. The newest
# RECENT_ALWAYS_ADMIT items are exempt (total budget only) — same YAML key and
# default as builder.PROMPT_MIN_RECENT_FLOOR (not imported: cycle).
RECENT_MAX_SHARE = _cfg_float("prompt_recent_max_share", 0.50)
RECENT_ALWAYS_ADMIT = _cfg_int("prompt_min_recent_floor", 5)

# 2026-10-15 (BC-99, BC-24): survival floors for the rendered summary/reflection
# sections. The builder tops these sections up to the floor BEFORE the budget
# runs, so the budget must RESERVE room for them or priority admission empties
# the very items the floor guarantees. Section -> memory-config key of its floor
# (the SAME keys builder.PROMPT_MIN_*_FLOOR read; defaults mirror the builder's).
# Listed in PRIORITY_ORDER order (reserve is granted first-come up to the bound).
# 2026-10-20 (BC-99 reopened): `memories` joins -- 37 of 54 builds 10-18..10-20
# emptied [RELEVANT MEMORIES] (priority 5 is reached after ~8.7K of the 10K budget
# is spent), including distress turns and a recall question. Its floor is the
# dedicated budget-RESERVE key prompt_memories_reserve_floor (2), distinct from
# the builder's pre-budget top-up minimum prompt_memory_topup_floor.
FLOOR_RESERVE_MAX_SHARE_DEFAULT = 0.25  # one value: YAML default, schema default, code fallback
FLOOR_PROTECTED_SECTIONS = (
    ("memories", "prompt_memories_reserve_floor", 2),
    ("recent_summaries", "prompt_min_summaries_floor", 2),
    ("recent_reflections", "prompt_min_reflections_floor", 1),
)


# 2026-10-24 (BC-99/BC-24): a turn routed to LIVE local git history (status
# report or own-code activity question) reserves room for its first N commits.
# CONDITIONAL: ambient git_commits (index retrieval, priority 3) stay emptiable.
GIT_ROUTED_RESERVE = ("git_commits", "prompt_git_commits_reserve_floor", 5)

# Human labels for the omitted-context note (unlisted keys fall back to the key
# with underscores spaced out).
_OMITTED_LABELS = {
    "git_commits": "commit history",
    "semantic_summaries": "semantic summaries",
    "recent_summaries": "recent summaries",
    "semantic_reflections": "semantic reflections",
    "recent_reflections": "recent reflections",
    "daemon_self_notes": "your own notes",
    "proactive_insights": "proactive insights",
    "proposed_features": "proposed features",
    "user_uploads": "uploaded items",
    "personal_notes": "personal notes",
    "reference_docs": "reference docs",
    "memories": "relevant memories",
    "google_calendar": "calendar events",
    "upcoming_schedule": "scheduled events",
    "unresolved_threads": "open threads",
    "web_search_results": "web results",
    "stated_figures": "your stated figures",
    "day_counts": "your day counts",
    "recent_conversations": "recent conversation",
}
OMITTED_NOTE_MAX_SECTIONS = 4


def _is_routed_git(context: Dict[str, Any]) -> bool:
    """True when git_commits carries live local history (a routed turn)."""
    return any(
        isinstance(item, dict)
        and (item.get("metadata") or {}).get("retrieval_source") == "local_git"
        for item in (context.get("git_commits") or [])
    )


def build_omitted_note(emptied: List[str], context: Dict[str, Any]) -> str:
    """ONE short line naming (human labels, counts) the non-empty sections the
    budget emptied, highest-priority first, so absence is not read as evidence."""
    prio = {name: p for name, p in PRIORITY_ORDER}
    rows = []
    for name in sorted(emptied, key=lambda n: -prio.get(n, 0)):
        val = context.get(name)
        if not val or name in ("budget_omitted_note", "stm_summary"):
            continue
        count = len(val) if isinstance(val, list) else 1
        rows.append(f"{_OMITTED_LABELS.get(name, name.replace('_', ' '))} ({count})")
    if not rows:
        return ""
    shown, extra = rows[:OMITTED_NOTE_MAX_SECTIONS], len(rows) - OMITTED_NOTE_MAX_SECTIONS
    listing = ", ".join(shown) + (f", +{extra} more" if extra > 0 else "")
    return (f"[NOT SHOWN FOR SPACE] {listing} - absence here is not evidence "
            "that nothing happened \u2014 context for you; do not mention it unless "
            "the user asks what you can see.")


def _live_memory_cfg(key: str, default_val: Any) -> Any:
    """Read a memory-section config key at CALL time (BC-82: live, not frozen)."""
    try:
        from config.app_config import config as cfg  # lazy import: live-config read
        v = (cfg.get("memory") or {}).get(key, default_val)
        return default_val if v is None else v
    except (ImportError, AttributeError, TypeError):  # degrades: falls back to the built-in default
        return default_val


def current_session_prefix_len(items: List[Any], now: datetime) -> int:
    """Length of the newest-first prefix of `items` that is the CURRENT session.

    Same rule as builder._session_truth_from_probe: item 0 is current only if
    it is not a session boundary from `now`; item i>0 continues the prefix
    while it is not a boundary from item i-1. `now` is explicit (BC-94).
    """
    prev_ts: Optional[datetime] = None
    n = 0
    for idx, item in enumerate(items or []):
        ts = _parse_entry_timestamp(item) if isinstance(item, dict) else None
        # An item whose time cannot be read is never PROVEN prior-session
        # (formatter's rule would call it a boundary): it stays in the
        # uncapped prefix, so untimestamped lists behave exactly as before.
        if ts is not None:
            if idx == 0:
                if _detect_session_boundary(ts, now, SESSION_BOUNDARY_GAP_HOURS):
                    return 0
            elif prev_ts is not None and _detect_session_boundary(prev_ts, ts, SESSION_BOUNDARY_GAP_HOURS):
                break
            prev_ts = ts
        n += 1
    return n

def drop_prior_session_test_turns(items: List[Any], now: Optional[datetime] = None,
                                  current_query: Optional[str] = None) -> List[Any]:
    """Drop owner ``[test]...[/test]`` probe turns that are NOT in the CURRENT
    session from a newest-first recent-conversation list (2026-10-18).

    The current session is `current_session_prefix_len` (the ONE session
    notion the budget and the formatter share), so a multi-turn probe keeps
    its own previous exchange while a probe from an earlier session — whose
    stored reply later probes copied verbatim — never reaches the prompt.
    Real turns of both sessions are kept. THE predicate is
    ``memory.fact_source.is_test_turn``.

    ``current_query`` is the query of the turn being built (2026-10-28, BC-58):
    when it is given and is NOT itself a probe, the turn is a REAL one and
    NO probe turn is visible to it, same session or not (a real "Hey" an hour
    after six probes carried all of them as its recent conversation). A probe
    current turn, or ``None`` (a caller outside a turn), keeps the rule above.
    """
    items = list(items or [])
    if probes_hidden_from_turn(current_query):
        return [e for e in items if not is_test_turn(e)]
    keep = current_session_prefix_len(items, now or datetime.now())
    return [e for i, e in enumerate(items) if i < keep or not is_test_turn(e)]


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
        # A deliberately skipped search carries a short deterministic note the
        # formatter renders (BC-04); meter exactly that text.
        _note = getattr(val, "note", None)
        return count_tokens(str(_note)) if _note else 0
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

    def _manage_token_budget(
        self, context: Dict[str, Any], now: Optional[datetime] = None
    ) -> Dict[str, Any]:
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
        def _meter_list_item(name: str, i: int, item: Any):
            """Cap + meter one list item exactly as the first pass does (shared with
            the floor-reserve computation so the two can never meter differently)."""
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
            return item, t

        # BC-99 / BC-24: reserve room for the floor-protected sections' first
        # `floor` items; sections admitted ahead of them see budget - reserve.
        _reserve: Dict[str, int] = {}
        try:
            _max_share = float(_live_memory_cfg("prompt_floor_reserve_max_share", FLOOR_RESERVE_MAX_SHARE_DEFAULT))
        except (TypeError, ValueError):  # degrades: default share
            _max_share = FLOOR_RESERVE_MAX_SHARE_DEFAULT
        _reserve_bound = int(_max_share * self.token_budget)
        _reserve_total = 0
        _floors: Dict[str, int] = {}
        _protected = list(FLOOR_PROTECTED_SECTIONS)
        if _is_routed_git(context):
            _protected.insert(0, GIT_ROUTED_RESERVE)
        for _rname, _rkey, _rdef in _protected:
            try:
                _floor = int(_live_memory_cfg(_rkey, _rdef))
            except (TypeError, ValueError):  # degrades: default floor
                _floor = _rdef
            _floors[_rname] = max(0, _floor)
            _rval = context.get(_rname)
            if _floor <= 0 or not isinstance(_rval, list) or not _rval:
                continue
            _need = sum(_meter_list_item(_rname, _ri, _rit)[1]
                        for _ri, _rit in enumerate(_rval[:_floor]))
            if _reserve_total + _need > _reserve_bound:
                logger.info(
                    f"[TOKEN BUDGET] Floor reserve for '{_rname}' skipped: {_need} tokens would "
                    f"exceed bound {_reserve_bound} ({_max_share:.2f} of budget; reserved so far {_reserve_total})"
                )
                continue
            _reserve[_rname] = _need
            _reserve_total += _need

        def _held_for(name: str) -> int:
            """Tokens other sections' floor reserves hold back from `name`'s admission."""
            return sum(v for k, v in _reserve.items() if k != name)

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
                # Prior-session share cap (recent_conversations only).
                _prior_start = len(val)
                _prior_cap = 0
                _prior_used = 0
                _prior_kept = 0
                _prior_capped = False
                _capped_by = ""
                _sect_cap = 0
                _sect_used = 0
                if name == "recent_conversations":
                    _prior_start = current_session_prefix_len(val, now or datetime.now())
                    _prior_cap = int(RECENT_PRIOR_SESSION_MAX_SHARE * self.token_budget)
                    _sect_cap = int(RECENT_MAX_SHARE * self.token_budget)
                for i, item in enumerate(val):
                    item, t = _meter_list_item(name, i, item)
                    if name == "recent_conversations":
                        if i >= _prior_start and _prior_used + t > _prior_cap:
                            _prior_capped, _capped_by = True, "prior-session"
                            break
                        if i >= RECENT_ALWAYS_ADMIT and _sect_used + t > _sect_cap:
                            _prior_capped, _capped_by = True, "section"
                            break
                    # Floor-protected sections: this section's own reserve is released
                    # here; other sections' reserves stay held back from this pass.
                    if current_tokens + t + _held_for(name) <= self.token_budget:
                        kept.append(item)
                        current_tokens += t
                        _sect_used += t
                        if i >= _prior_start:
                            _prior_used += t
                            _prior_kept += 1
                    else:
                        if name == "memories":
                            logger.debug(f"[TOKEN BUDGET] Stopped adding memories at item {i}/{len(val)}: budget={self.token_budget}, current={current_tokens}, item_tokens={t}")
                        break
                if _prior_capped:
                    logger.info(
                        f"[TOKEN BUDGET] recent_conversations {_capped_by} cap ended section: kept "
                        f"{len(kept)}/{len(val)} items, {_prior_kept}/{len(val) - _prior_start} prior "
                        f"(prior cap {_prior_cap}, section cap {_sect_cap} tokens); "
                        f"current session {_prior_start} items"
                    )
                if name == "memories":
                    logger.debug(f"[TOKEN BUDGET] Kept {len(kept)}/{len(val)} memories, budget={self.token_budget}, used={current_tokens}")
                trimmed[name] = kept
                _reserve.pop(name, None)
            elif isinstance(val, dict):
                # Dict-shaped sections (e.g. a raw {"pages": [...]} web
                # result, or codebase_changes) are metered but NEVER
                # coerced/written back — unchanged from prior behavior. This
                # is explicitly NOT an adapter case even for
                # "web_search_results": only the dataclass shape is adapted.
                item_text = str(val)
                t = self.get_token_count(item_text, model_name)
                if current_tokens + t + _held_for(name) <= self.token_budget:
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
                if val is not None and current_tokens + t + _held_for(name) <= self.token_budget:
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

                if current_tokens + t + _held_for(name) <= self.token_budget:
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
                # Ties: the small time-sensitive rows trim LAST (they sit early
                # in list order for the first pass, which would otherwise make
                # them the FIRST tied rows trimmed here).
                for name, prio in sorted(
                    priority_order,
                    key=lambda x: (x[1], x[0] in _SMALL_TIME_SENSITIVE),
                ):  # low → high
                    v = trimmed.get(name)
                    if not v:
                        continue

                    if isinstance(v, list) and v:
                        # Drop a conservative slice from the tail
                        # Floor-protected sections never drop below their floor
                        # (reserve doctrine, BC-99): trim only the tail beyond it.
                        _fl = _floors.get(name, 0)
                        if len(v) <= _fl:
                            continue
                        drop_n = min(max(1, int(len(v) * 0.25)), len(v) - _fl)
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

        # 2026-10-08 (BC-25, BC-47): name the sections this budget EMPTIED
        # (non-empty in, empty out). Downstream consumers (the agentic reuse
        # gate) cannot tell "never retrieved" from "retrieved then trimmed
        # away" on the post-budget context alone. '_'-prefixed: never rendered
        # or metered (see the true-total loop below).
        _emptied = sorted(
            name for name, _ in PRIORITY_ORDER
            if name != "stm_summary" and context.get(name) and not trimmed.get(name)
        )
        trimmed.pop("budget_omitted_note", None)
        if _emptied:
            trimmed["_budget_emptied"] = _emptied
            _note = build_omitted_note(_emptied, context)
            if _note:
                trimmed["budget_omitted_note"] = _note
                # Not added to `usage`: <=~40 tokens, emitted only when the budget
                # already cut something, and usage is the budget-compliance number.
            logger.info(f"[TOKEN BUDGET] Sections emptied by budget: {_emptied}"
                        + (f" (floor reserve {_reserve_total} tokens)" if _reserve_total else ""))

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
                    f"(metered {usage}/{self.token_budget} + {unmetered} unmetered"
                    + (f"; floor reserve {_reserve_total}" if _reserve_total else "") + ")"
                )
        except Exception as e:  # degrades: true-total token accounting log line is skipped
            logger.debug(f"[TOKEN BUDGET] True-total accounting failed (non-fatal): {e}")
        return trimmed
