# utils/daily_notes_generator.py
"""
DailyNotesGenerator - Generate daily summary notes from Daemon conversations.

Module Contract:
- Purpose: Automatically generate structured daily notes from conversation history,
           written from Daemon's perspective in YAML + bullet format with LLM-based tag generation.
- Inputs:
  - generate_for_date(date: date) -> GenerationResult: Generate note for specific date
  - generate_yesterday_if_missing() -> Optional[GenerationResult]: Startup catch-up (called by GUI on launch)
  - note_exists(date: date) -> bool: Check if note already exists
- Outputs:
  - Markdown files in Obsidian vault: ~/Documents/[user vault]/Daily/M D YY Daily Note.md
  - GenerationResult with success status, path, stats
- Behavior:
  - Filters corpus by timestamp to get day's conversations
  - Calculates active duration (estimated actual usage time, not wall-clock span) [NEW 2026-01-18]
  - Calls LLM to generate structured summary:
    - Main Quest, Side Quests, Emotional State, Key Decisions, etc.
    - Life Events section: Work, Study, Sleep, Exercise, Other [NEW 2026-01-18]
  - Generates contextual tags using TagGenerator (3-10 tags) [NEW 2026-01-22]
  - Writes atomic markdown with YAML frontmatter:
    - usage_intensity (was: intensity) [RENAMED 2026-01-18]
    - span_hours (wall-clock), active_hours (estimated usage) [NEW 2026-01-18]
    - tags: system tags + content tags [NEW 2026-01-22]
  - Auto-update: if note exists but conversation count has grown by >= DAILY_NOTES_UPDATE_MIN_NEW (default 3),
    regenerates with all conversations for the day. Fixes stale notes from early-session shutdowns. [NEW 2026-05-20]
  - Skips if note exists and conversation count delta is below threshold (unless force=True)
  - Triggers narrative context refresh after daily note creation [NEW 2026-01-17]
- Key Methods:
  - _calculate_active_duration(convos): Estimate actual usage time [NEW 2026-01-18]
    - Reading time: ~200 words/min for responses
    - Typing time: ~40 words/min for queries
    - Gap time: Capped at 30 seconds (idle time excluded)
  - _calculate_intensity(convos, active_hours): 1-10 score based on count/active duration/complexity
- Dependencies:
  - memory.corpus_manager (conversation retrieval, narrative context persistence)
  - models.model_manager (LLM generation)
  - utils.tag_generator (tag generation for notes) [NEW 2026-01-22]
  - memory.memory_consolidator (narrative synthesis) [NEW 2026-01-17]
  - config.app_config (paths and settings)
- Side effects:
  - Writes files to Obsidian vault
  - LLM API calls (note generation + tag generation)
  - Logging
  - _trigger_narrative_refresh(): Regenerates narrative context after daily note creation [NEW 2026-01-17]
"""

import os
import json
import logging
from pathlib import Path
from utils.safe_json import atomic_write_text
from utils.personal_claim_provenance import annotate_personal_claim_memory, marker_suffix
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re

from utils.status_claims import authoritative_facts_block, remove_conflicting_claims
import utils.status_claims as status_claims
import utils.streak_claims as streak_claims
import utils.tag_generator as _tag_generator
from utils.retrieval_outcome import RetrievalError
from utils.notes_common import FALLBACK_MODELS, SPEAKER_AUTHORITY_RULES

logger = logging.getLogger(__name__)


_DURATION_AMOUNT = (
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
)
_STATUS_DURATION_RE = re.compile(
    rf"\b(?:for\s+)?(?:the\s+)?(?:last\s+)?(?:a\s+)?"
    rf"{_DURATION_AMOUNT}(?:\s*(?:-|to)\s*{_DURATION_AMOUNT})?\s*"
    r"(?:days?|weeks?|months?)\b",
    re.IGNORECASE,
)
_STATUS_WORD_RE = re.compile(
    r"\b(?:stable|stabilized|stability|functional|functioning|functionality|"
    r"function|healthy|health|recovered|recovery|baseline|good\s+to\s+go)\b",
    re.IGNORECASE,
)
_CONDITIONAL_RE = re.compile(
    r"\b(?:if|might|may|could|would|risk|in case|do not expect|don't expect)\b",
    re.IGNORECASE,
)
_DRAFT_CONTEXT_RE = re.compile(
    r"\b(?:draft|email|subject:|look ok|how (?:does|do|we) .*look|send this)\b",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]?\d{4}(?!\d)")
_LONG_ID_RE = re.compile(r"(?<!\d)\d{7,}(?!\d)")


def _claim_sentence(text: str, start: int, end: int, max_chars: int = 360) -> str:
    """Return the sentence-like window containing a matched temporal claim."""
    left_candidates = [text.rfind(mark, 0, start) for mark in ("\n", ".", "?", "!")]
    left = max(left_candidates) + 1
    right_candidates = [
        pos for mark in ("\n", ".", "?", "!")
        if (pos := text.find(mark, end)) >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    snippet = " ".join(text[left:right].split())
    if len(snippet) > max_chars:
        match_mid = max(0, (start + end) // 2 - left)
        half = max_chars // 2
        clip_start = max(0, match_mid - half)
        clip_end = min(len(snippet), clip_start + max_chars)
        snippet = snippet[clip_start:clip_end]
        if clip_start:
            snippet = "…" + snippet
        if clip_end < len(" ".join(text[left:right].split())):
            snippet += "…"
    return snippet


def build_temporal_claim_audit(convos: List[Dict[str, Any]]) -> str:
    """Extract user-authored status/duration claims for synthesis reconciliation.

    This is deliberately mechanical: it does not decide which duration is true.
    It preserves the exact predicates and marks conditional or draft context so
    the summarizer cannot silently collapse competing statements into one fact.
    """
    claims = []
    seen = set()
    streak_lines = []
    for conv in convos:
        # The user's own typed text when the turn carried attachments
        # (2026-09-05) — transcripts/CSV rows are not the user's claims.
        query = str(conv.get("user_text") or conv.get("query") or "")
        if not query:
            continue
        # Redact before locating sentence boundaries: a period inside an email
        # address must not terminate the window and leak a partial address.
        audit_text = _EMAIL_RE.sub("[EMAIL]", query)
        audit_text = _PHONE_RE.sub("[PHONE]", audit_text)
        audit_text = _LONG_ID_RE.sub("[ID]", audit_text)
        ts = conv.get("timestamp")
        if isinstance(ts, datetime):
            time_str = ts.strftime("%H:%M")
        elif isinstance(ts, str):
            try:
                time_str = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%H:%M")
            except ValueError:
                time_str = "??:??"
        else:
            time_str = "??:??"

        draft_context = bool(_DRAFT_CONTEXT_RE.search(query))
        for match in _STATUS_DURATION_RE.finditer(audit_text):
            snippet = _claim_sentence(audit_text, match.start(), match.end())
            if not _STATUS_WORD_RE.search(snippet):
                continue
            key = snippet.casefold()
            if key in seen:
                continue
            seen.add(key)
            if draft_context:
                status = "draft-or-pasted-text"
            elif _CONDITIONAL_RE.search(snippet):
                status = "conditional"
            else:
                status = "direct-user-statement"
            claims.append(f'- [{time_str}] [{status}] "{snippet}"')
        # Streak / day counts (2026-09-05): the user's own count for THIS
        # date. Listed separately so the note writes "day N (his count)"
        # instead of copying an assistant reply's number ("six solid days"
        # was copied into the Sep 3 note a day after the user said day 6).
        try:
            for sc in streak_claims.extract_streak_claims(audit_text, ts if ts else None):
                key = ("streak", sc.count, sc.snippet.casefold())
                if key in seen:
                    continue
                seen.add(key)
                streak_lines.append(
                    f'- [{time_str}] [streak-count] "{sc.snippet}" → the user\'s own count on this date: day {sc.count}'
                )
        except Exception:
            pass

    if not claims and not streak_lines:
        return "(No mechanically detected status-duration claims.)"
    body = "\n".join(claims[:12] + streak_lines[:6])
    return (
        "Mechanically detected user-text claims (not independently verified):\n"
        + body
        + "\nPreserve the exact status predicate and duration for each item. "
        "Draft/pasted or conditional text is not an established user fact. "
        "If direct statements appear inconsistent, report the discrepancy; "
        "do not silently choose or average one. "
        "A [streak-count] line is the user's own count ON THIS DATE — write it as "
        "\"day N (his count)\" and never substitute a number from an assistant reply "
        "or an earlier day."
    )


# ---------------------------------------------------------------------------
# Tool-use attribution (2026-10-12, X3): the note pass used to see only
# query/response text, so it inferred which tool did what -- a note credited
# Wolfram with a tip calculation that no Wolfram call produced (Wolfram ran on
# a different turn). The turn telemetry records the tools that ACTUALLY ran;
# they are handed to the note per exchange, with a name-a-tool-only-if-listed
# rule, and a deterministic LOG-ONLY post-check flags a registry tool name in
# the note that ran in no turn that day.
# ---------------------------------------------------------------------------

# Control/terminal tools that are not "work" a note would attribute.
_NON_ATTRIBUTABLE_TOOLS = frozenset({"done_searching"})


def known_tool_names() -> List[str]:
    """Agentic tool names from the registry itself (native tool definitions +
    DISPATCH_TABLE handler names) -- no hand-kept list to drift."""
    # lazy import: layering (utils/ must not import core.agentic at module load)
    from core.agentic import types as _atypes
    from core.agentic.tools import DISPATCH_TABLE as _TABLE  # lazy import: layering
    names = set()
    for attr in dir(_atypes):
        if attr.endswith("_TOOL_DEFINITION"):
            fn = (getattr(_atypes, attr, None) or {}).get("function") or {}
            if fn.get("name"):
                names.add(str(fn["name"]))
    for _pred, handler, _args in _TABLE:
        base = str(handler).replace("_dispatch_", "", 1)
        if base != "api_search":  # shared handler; the specific names come from the definitions
            names.add(base)
    return sorted(n for n in names if n not in _NON_ATTRIBUTABLE_TOOLS)


def _tool_tokens(name: str) -> frozenset:
    return frozenset(t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t)


def _tool_names_match(a: str, b: str) -> bool:
    """Same tool under the registry's two spellings (wolfram / wolfram_alpha,
    memory_search / search_memory): equal token sets, or one is a token-prefix
    of the other."""
    ta, tb = _tool_tokens(a), _tool_tokens(b)
    if not ta or not tb:
        return False
    return ta == tb or ta <= tb or tb <= ta


def load_turn_tool_receipts(target_date: date, path: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """Per-turn {ts, query, tools} for `target_date` from the turn telemetry
    JSONL (stream-parsed). None = the record is unavailable (missing/unreadable)
    -- distinct from [] (readable, no turns that day)."""
    if path is None:
        from config.app_config import TURN_TELEMETRY_PATH as path  # lazy import: live-config
    try:
        out: List[Dict[str, Any]] = []
        day = target_date.isoformat()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, dict) or rec.get("test_env"):
                    continue
                if str(rec.get("ts", ""))[:10] != day:
                    continue
                tools: List[str] = []
                for call in rec.get("tool_calls") or []:
                    t = str((call or {}).get("tool", "")).strip()
                    if t and t not in tools and t not in _NON_ATTRIBUTABLE_TOOLS:
                        tools.append(t)
                out.append({"ts": rec.get("ts"), "query": str(rec.get("query", "")).strip(), "tools": tools})
        return out
    except (OSError, ValueError):  # degrades: tool record unavailable, the note is told not to name tools
        return None


def tools_by_exchange(convos: List[Dict[str, Any]],
                      receipts: Optional[List[Dict[str, Any]]]) -> Optional[Dict[int, List[str]]]:
    """Exchange number (1-based, as _format_conversations numbers them) -> the
    tools that ran for it. Joined on the exact stripped query, in order, each
    receipt consumed once. An exchange with no matching receipt maps to [] only
    when the record exists at all."""
    if receipts is None:
        return None
    pool = list(receipts)
    result: Dict[int, List[str]] = {}
    for i, conv in enumerate(convos, 1):
        q = str(conv.get("query", "")).strip()
        match = next((r for r in pool if r["query"] == q), None)
        if match is not None:
            pool.remove(match)
        result[i] = list(match["tools"]) if match else []
    return result


def build_tool_use_record(convos: List[Dict[str, Any]], receipts: Optional[List[Dict[str, Any]]]) -> str:
    """Prompt block listing, per exchange, the tools that actually ran."""
    mapping = tools_by_exchange(convos, receipts)
    if mapping is None:
        return ("TOOLS THAT ACTUALLY RAN: record unavailable. Do NOT name any "
                "tool, API or calculator as having done anything.")
    ran = {i: t for i, t in mapping.items() if t}
    lines = ["TOOLS THAT ACTUALLY RAN (from the turn log; exchanges not listed ran NO tool):"]
    if ran:
        lines.extend(f"- Exchange {i}: {', '.join(t)}" for i, t in sorted(ran.items()))
    else:
        lines.append("- None: no tool ran in any exchange today.")
    lines.append("Name a tool, API or calculator only if it is listed for that exchange. "
                 "Arithmetic or facts in a reply without a listed tool came from the model itself.")
    return "\n".join(lines)


def find_unrun_tool_mentions(note_text: str, receipts: Optional[List[Dict[str, Any]]]) -> List[str]:
    """LOG-ONLY post-check: registry tool names the note mentions that ran in
    NO turn that day. [] when the record is unavailable (nothing to check
    against) or nothing is flagged."""
    if receipts is None:
        return []
    ran = {t for r in receipts for t in r["tools"]}
    flagged: List[str] = []
    for name in known_tool_names():
        pattern = r"\b" + r"[\s_\-]+".join(re.escape(t) for t in re.split(r"[^A-Za-z0-9]+", name) if t) + r"\b"
        if not re.search(pattern, note_text or "", re.IGNORECASE):
            continue
        if any(_tool_names_match(name, r) for r in ran):
            continue
        flagged.append(name)
    return flagged


@dataclass
class GenerationResult:
    """Result of daily note generation."""
    success: bool = False
    date: Optional[date] = None
    output_path: Optional[Path] = None
    conversation_count: int = 0
    intensity: int = 0
    duration_hours: float = 0.0
    skipped_reason: Optional[str] = None  # "already_exists", "no_conversations", "disabled"
    error: Optional[str] = None
    unrun_tool_mentions: List[str] = field(default_factory=list)  # LOG-ONLY post-check (2026-10-12)


# ---------------------------------------------------------------------------
# Module-level path resolvers (shared by DailyNotesGenerator and STM injection)
# ---------------------------------------------------------------------------
# These exist as module-level functions so callers (e.g. STMAnalyzer) can read
# daily notes without instantiating DailyNotesGenerator (which would lazy-load
# corpus_manager + model_manager). Both helpers are read-only and never trigger
# generation or narrative refresh.

def get_daily_note_path(target_date: date, vault_path: Optional[Path] = None) -> Optional[Path]:
    """Resolve the actual on-disk path for a daily note, checking all known
    layouts (flat, weekly-at-root, monthly/weekly).

    Returns None if no file exists for the given date.
    """
    if vault_path is None:
        try:
            from utils.notes_common import daily_notes_base  # lazy import: live-config
            base = daily_notes_base()
        except ImportError:
            return None
    else:
        base = Path(vault_path).expanduser()

    filename = f"{target_date.month} {target_date.day} {target_date.strftime('%y')} Daily Note.md"

    # Layout 1: flat
    p = base / filename
    if p.exists():
        return p

    # Layout 2: weekly folder at root (legacy)
    monday = target_date - timedelta(days=target_date.weekday())
    week_num = monday.isocalendar()[1]
    week_folder = f"Week {week_num} {monday.strftime('%b %Y')}"
    p = base / week_folder / filename
    if p.exists():
        return p

    # Layout 3: monthly/weekly folder (current)
    month_folder = f"{monday.strftime('%B %Y')}"
    p = base / month_folder / week_folder / filename
    if p.exists():
        return p

    return None


def read_daily_note(target_date: date, vault_path: Optional[Path] = None) -> Optional[str]:
    """Read the markdown text for a daily note, or None if missing or unreadable."""
    p = get_daily_note_path(target_date, vault_path)
    if p is None:
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"[DailyNotes] Failed to read {p}: {e}")
        return None


# LLM prompt template
SYSTEM_PROMPT_TEMPLATE = '''You are Daemon, an AI companion writing a daily note about your conversations with __USER_NAME__ today.

CONVERSATIONS FROM {date}:
{formatted_conversations}

TEMPORAL STATUS CLAIM AUDIT:
{temporal_claim_audit}

{tool_use_record}

{authoritative_status_facts}

STATISTICS:
- Conversations: {count}
- Time span: {first_time} to {last_time} ({span_hours:.1f} hours wall-clock)
- Active time: {active_hours:.1f} hours (estimated actual usage)

Write a daily note in this EXACT structure:

## Summary
2-3 sentences capturing the day's theme from your perspective as Daemon.

## Main Quest: [Primary Focus]
The main thing we worked on today. 3-5 bullets about progress, challenges, outcomes.

## Side Quests
Other topics discussed. Format: **Topic**: one-line description. If only one main topic, write "None today - fully focused on the Main Quest."

## Life Events
Track these specific life activities ONLY if mentioned in conversations. Do NOT assume they didn't happen if not mentioned - just note what was or wasn't discussed:

- **Work**: If mentioned: duration, what was done, how it went. If not mentioned: "Not discussed today."
- **Study**: If mentioned: what subject/material, duration, how it went, any progress. If not mentioned: "Not discussed today."
- **Sleep**: If mentioned: quality, duration, any issues. If not mentioned: "Not discussed today."
- **Exercise/Health**: If mentioned: what activity, how it went. If not mentioned: "Not discussed today."
- **Other Events**: Any other notable life events mentioned (social activities, appointments, errands, etc.). If none mentioned: "None mentioned."

## Emotional State
__USER_NAME__'s mood throughout the day. Note any shifts. Be specific and observant.

## Key Decisions
Explicit choices made today. Bullet list. If none, write "None today."

## Knowledge Gained
New facts, concepts, or insights that came up. Bullet list. If none, write "None today."

## Open Threads
Unresolved questions or things to follow up on. Bullet list. If none, write "All threads resolved."

## Intensity: X/10
One line: count, duration, and complexity assessment.

IMPORTANT:
- Write from YOUR perspective as Daemon ("Today we...", "__USER_NAME__ seemed...")
- Be factual - only include what actually happened in the conversations
- For Life Events: ONLY report what was explicitly mentioned. Never assume activities happened or didn't happen. If __USER_NAME__ didn't mention work/study/sleep, say "Not discussed today" - do NOT say "__USER_NAME__ didn't work today."
- If a section has nothing relevant, acknowledge it briefly
- Keep it concise but informative
- Preserve exact time spans and the state they modify (for example, "stable"
  versus "functional"). Never silently replace one with the other.
- Treat text in drafts, pasted emails, quotations, and hypothetical/conditional
  statements as attributed text, not as independent confirmation of a user fact.
- If the TEMPORAL STATUS CLAIM AUDIT contains competing direct statements,
  state the discrepancy explicitly or preserve both with their distinct predicates.
- If an AUTHORITATIVE CURRENT FACTS block is present above, never write a
  sentence that contradicts it (e.g. do not claim someone withdrew from
  school/a program while a current enrollment fact is listed there).
- Tools: name a tool, API or calculator as having done something ONLY if the
  TOOLS THAT ACTUALLY RAN block above lists it for that exchange.
- Do NOT include any preamble or meta-commentary, just the note content

__SPEAKER_AUTHORITY_RULES__
'''


class DailyNotesGenerator:
    """Generate daily summary notes from Daemon conversations."""

    def __init__(self, corpus_manager=None, model_manager=None, vault_path: str = None,
                 tag_generator=None, user_profile=None):
        """
        Initialize DailyNotesGenerator.

        Args:
            corpus_manager: CorpusManager instance (lazy-loaded if None)
            model_manager: ModelManager instance (lazy-loaded if None)
            vault_path: Path to Obsidian vault (defaults to config)
            tag_generator: TagGenerator instance (lazy-loaded if None)
            user_profile: UserProfile instance for the status-claim conflict
                guard (2026-09-04; lazy-loaded if None)
        """
        self._corpus_manager = corpus_manager
        self._model_manager = model_manager
        self._tag_generator = tag_generator
        self._user_profile = user_profile

        # Load config
        try:
            from config.app_config import (  # lazy import: live-config
                OBSIDIAN_VAULT_PATH,
                DAILY_NOTES_ENABLED,
                DAILY_NOTES_FOLDER,
                DAILY_NOTES_MODEL,
                DAILY_NOTES_MAX_TOKENS,
                TAG_GENERATION_ENABLED,
            )
            self.vault_path = Path(vault_path or OBSIDIAN_VAULT_PATH).expanduser()
            self.enabled = DAILY_NOTES_ENABLED
            self.daily_folder = DAILY_NOTES_FOLDER
            self.model_name = DAILY_NOTES_MODEL
            self.max_tokens = DAILY_NOTES_MAX_TOKENS
            self.tag_generation_enabled = TAG_GENERATION_ENABLED
        except ImportError:
            self.vault_path = Path(vault_path or "~/Documents/Notes").expanduser()
            self.enabled = True
            self.daily_folder = "Daily"
            self.model_name = "sonnet-4.5"
            self.max_tokens = 800
            self.tag_generation_enabled = True

        # Explicit vault_path override keeps vault/folder; otherwise the shared
        # resolver (vault when Obsidian is on + present, else the data root).
        if vault_path:
            self.output_dir = self.vault_path / self.daily_folder
        else:
            try:
                from utils.notes_common import daily_notes_base  # lazy import: live-config
                self.output_dir = daily_notes_base()
            except Exception:  # degrades: falls back to vault/folder layout
                self.output_dir = self.vault_path / self.daily_folder
        logger.debug(f"[DailyNotes] Initialized: vault={self.vault_path}, folder={self.daily_folder}")

    @property
    def corpus_manager(self):
        """Lazy-load CorpusManager."""
        if self._corpus_manager is None:
            try:
                from memory.corpus_manager import CorpusManager  # lazy import: cycle
                from config.app_config import CORPUS_FILE  # lazy import: live-config
                self._corpus_manager = CorpusManager(CORPUS_FILE)
                logger.debug("[DailyNotes] CorpusManager lazy-loaded")
            except Exception as e:
                logger.error(f"[DailyNotes] Failed to load CorpusManager: {e}")
                raise
        return self._corpus_manager

    @property
    def model_manager(self):
        """Lazy-load ModelManager."""
        if self._model_manager is None:
            try:
                from models.model_manager import ModelManager  # lazy import: layering
                self._model_manager = ModelManager()
                logger.debug("[DailyNotes] ModelManager lazy-loaded")
            except Exception as e:
                logger.error(f"[DailyNotes] Failed to load ModelManager: {e}")
                raise
        return self._model_manager

    @property
    def tag_generator(self):
        """Lazy-load TagGenerator."""
        if self._tag_generator is None:
            try:
                self._tag_generator = _tag_generator.TagGenerator(model_manager=self.model_manager)
                logger.debug("[DailyNotes] TagGenerator lazy-loaded")
            except Exception as e:
                logger.warning(f"[DailyNotes] Failed to load TagGenerator: {e}")
                # Non-critical, can continue without tag generation
                self._tag_generator = None
        return self._tag_generator

    @property
    def user_profile(self):
        """Lazy-load UserProfile (read-only use here — this class never
        saves it). Failure is non-critical: callers fall back to an empty
        status-facts list, which disables the status-claim guard for that
        call but never blocks daily-note generation."""
        if getattr(self, "_user_profile", None) is None:
            try:
                from memory.user_profile import UserProfile  # lazy import: startup-cost
                self._user_profile = UserProfile()
            except Exception as e:
                logger.debug(f"[DailyNotes] UserProfile unavailable for status-claim guard: {e}")
                return None
        return self._user_profile

    def _current_status_facts(self) -> List[Dict[str, Any]]:
        """Collect CURRENT enrollment/employment/residence facts for the
        status-claim conflict guard (utils/status_claims.py). Raises
        RetrievalError when the profile is unavailable or the read fails;
        a genuine empty filter result still returns []."""
        profile = self.user_profile
        if profile is None:
            raise RetrievalError(source="status_facts", reason="profile_unavailable")
        try:
            current = profile.get_current_view()
        except Exception as e:
            logger.debug(f"[DailyNotes] Could not read profile facts for status guard: {e}")
            raise RetrievalError(source="status_facts", reason=type(e).__name__) from e

        facts: List[Dict[str, Any]] = []
        for cat_facts in (current or {}).values():
            for f in cat_facts or []:
                if isinstance(f, dict) and f.get("relation") in status_claims.STATUS_RELATIONS:
                    facts.append(f)
        return facts

    def _format_filename(self, target_date: date) -> str:
        """Format filename to match existing convention: 'M D YY Daily Note.md'."""
        return f"{target_date.month} {target_date.day} {target_date.strftime('%y')} Daily Note.md"

    def _get_week_folder_name(self, target_date: date) -> str:
        """Format weekly folder name: 'Week 3 Jan 2026'."""
        monday = target_date - timedelta(days=target_date.weekday())
        week_num = monday.isocalendar()[1]
        return f"Week {week_num} {monday.strftime('%b %Y')}"

    def _get_month_folder_name(self, target_date: date) -> str:
        """Format monthly folder name based on Monday of the week: 'January 2026'."""
        monday = target_date - timedelta(days=target_date.weekday())
        return f"{monday.strftime('%B %Y')}"

    def _get_note_path(self, target_date: date) -> Path:
        """Get note path - uses monthly/weekly folder structure."""
        filename = self._format_filename(target_date)
        month_folder = self._get_month_folder_name(target_date)
        week_folder = self._get_week_folder_name(target_date)
        return self.output_dir / month_folder / week_folder / filename

    def note_exists(self, target_date: date) -> bool:
        """Check if note already exists for date (flat, weekly-at-root, or monthly/weekly)."""
        return self._find_existing_note_path(target_date) is not None

    def _find_existing_note_path(self, target_date: date) -> Optional[Path]:
        """Find the actual filesystem path of an existing note, or None."""
        filename = self._format_filename(target_date)

        # Check monthly/weekly folder (current layout — most likely)
        month_folder = self._get_month_folder_name(target_date)
        week_folder = self._get_week_folder_name(target_date)
        path = self.output_dir / month_folder / week_folder / filename
        if path.exists():
            return path

        # Check weekly folder at root (legacy layout)
        path = self.output_dir / week_folder / filename
        if path.exists():
            return path

        # Check flat directory
        path = self.output_dir / filename
        if path.exists():
            return path

        return None

    def _get_existing_conversation_count(self, target_date: date) -> int:
        """Read the conversation count from an existing note's YAML frontmatter."""
        note_path = self._find_existing_note_path(target_date)
        if not note_path:
            return 0

        try:
            content = note_path.read_text(encoding='utf-8')
            if not content.startswith("---"):
                return 0

            end_idx = content.find("---", 3)
            if end_idx < 0:
                return 0

            frontmatter = content[3:end_idx]
            for line in frontmatter.split("\n"):
                line = line.strip()
                if line.startswith("conversations:"):
                    return int(line.split(":", 1)[1].strip())
        except Exception as e:
            logger.debug(f"[DailyNotes] Could not read frontmatter for {target_date}: {e}")

        return 0

    def _should_auto_update(self, target_date: date, current_count: int) -> bool:
        """Check if an existing note should be regenerated due to significant new data."""
        from config.app_config import DAILY_NOTES_UPDATE_MIN_NEW  # lazy import: live-config

        existing_count = self._get_existing_conversation_count(target_date)
        new_conversations = current_count - existing_count

        if new_conversations >= DAILY_NOTES_UPDATE_MIN_NEW:
            logger.info(
                f"[DailyNotes] Auto-update triggered for {target_date}: "
                f"{existing_count} → {current_count} conversations "
                f"(+{new_conversations}, threshold={DAILY_NOTES_UPDATE_MIN_NEW})"
            )
            return True

        return False

    def _get_conversations_for_date(self, target_date: date) -> List[Dict[str, Any]]:
        """Filter corpus entries by date."""
        conversations = []

        for entry in self.corpus_manager.corpus:
            ts = entry.get("timestamp")
            if ts is None:
                continue

            # Handle both datetime objects and ISO strings
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    continue
            elif not isinstance(ts, datetime):
                continue

            # Compare dates
            if ts.date() == target_date:
                conversations.append(entry)

        # Sort chronologically
        conversations.sort(key=lambda x: x.get("timestamp", datetime.min))
        return conversations

    def _format_conversations(self, convos: List[Dict[str, Any]]) -> str:
        """Format conversations for LLM prompt."""
        formatted = []

        for i, conv in enumerate(convos, 1):
            marked = annotate_personal_claim_memory(conv)
            query = conv.get("query", "").strip()
            response = conv.get("response", "").strip()
            ts = conv.get("timestamp")

            # Format timestamp
            if isinstance(ts, datetime):
                time_str = ts.strftime("%H:%M")
            elif isinstance(ts, str):
                try:
                    time_str = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%H:%M")
                except ValueError:
                    time_str = "??:??"
            else:
                time_str = "??:??"

            # Truncate long messages
            q_max, a_max = 300, 400
            if len(query) > q_max:
                query = query[:q_max] + "..."
            if len(response) > a_max:
                response = response[:a_max] + "..."
            if marked.get("response") != conv.get("response"):
                response += "\n" + marker_suffix(conv.get("response") or "", marked.get("response") or "")

            formatted.append(f"[{time_str}] Exchange {i}:\nUser: {query}\nDaemon: {response}")

        return "\n\n".join(formatted)

    def _calculate_active_duration(self, convos: List[Dict[str, Any]]) -> float:
        """
        Calculate active usage duration in hours.

        For each conversation:
        - Estimate reading time: ~200 words/min for response
        - Estimate typing time: ~40 words/min for query
        - Cap gap between conversations at 30 seconds (idle time doesn't count)

        Returns:
            Active duration in hours
        """
        if not convos:
            return 0.0

        total_seconds = 0.0
        MAX_GAP_SECONDS = 30  # Cap idle time between exchanges

        for i, conv in enumerate(convos):
            query = conv.get("query", "")
            response = conv.get("response", "")

            # Estimate time for this exchange
            # Reading response: ~200 words/min = ~1000 chars/min
            read_time = len(response) / 1000 * 60  # seconds
            # Typing query: ~40 words/min = ~200 chars/min
            type_time = len(query) / 200 * 60  # seconds
            # Minimum 10 seconds per exchange
            exchange_time = max(10, read_time + type_time)

            total_seconds += exchange_time

            # Add capped gap time to next conversation
            if i < len(convos) - 1:
                curr_ts = conv.get("timestamp")
                next_ts = convos[i + 1].get("timestamp")

                if curr_ts and next_ts:
                    # Parse timestamps
                    if isinstance(curr_ts, str):
                        curr_ts = datetime.fromisoformat(curr_ts.replace("Z", "+00:00"))
                    if isinstance(next_ts, str):
                        next_ts = datetime.fromisoformat(next_ts.replace("Z", "+00:00"))

                    if isinstance(curr_ts, datetime) and isinstance(next_ts, datetime):
                        gap = (next_ts - curr_ts).total_seconds()
                        # Only count up to MAX_GAP_SECONDS of idle time
                        total_seconds += min(gap, MAX_GAP_SECONDS)

        return total_seconds / 3600  # Convert to hours

    def _calculate_intensity(self, convos: List[Dict[str, Any]], active_hours: float) -> int:
        """
        Calculate intensity score 1-10 based on:
        - Conversation count
        - Active duration (not wall-clock time)
        - Average message length (complexity proxy)
        """
        if not convos:
            return 0

        count = len(convos)

        # Average message length as complexity proxy
        total_chars = sum(
            len(c.get("query", "")) + len(c.get("response", ""))
            for c in convos
        )
        avg_chars = total_chars / count if count > 0 else 0

        # Scoring components
        count_score = min(count / 20, 1.0) * 4  # Max 4 points for 20+ conversations
        duration_score = min(active_hours / 2, 1.0) * 3  # Max 3 points for 2+ active hours
        complexity_score = min(avg_chars / 1000, 1.0) * 3  # Max 3 points for avg 1000+ chars

        intensity = int(round(count_score + duration_score + complexity_score))
        return max(1, min(10, intensity))  # Clamp to 1-10

    def _extract_main_quest(self, llm_response: str) -> str:
        """Extract main quest title from LLM response for frontmatter."""
        # Look for "## Main Quest: [Title]"
        match = re.search(r'## Main Quest:\s*(.+?)(?:\n|$)', llm_response)
        if match:
            return match.group(1).strip()
        return "General Conversation"

    def _build_frontmatter(self, target_date: date, count: int, span_hours: float,
                           active_hours: float, intensity: int, main_quest: str,
                           content_tags: List[str] = None) -> str:
        """Generate YAML frontmatter."""
        # System tags are always included
        system_tags = ['daily', 'daemon-generated']

        # Add content tags if provided
        if content_tags:
            all_tags = system_tags + content_tags
        else:
            all_tags = system_tags

        # Format tags for YAML (quoted strings to handle hyphens)
        tags_str = ', '.join(f'"{tag}"' for tag in all_tags)

        return f"""---
date: {target_date.isoformat()}
author: daemon
source_type: daemon_daily_summary
derived_from: conversation_corpus
usage_intensity: {intensity}
conversations: {count}
span_hours: {span_hours:.1f}
active_hours: {active_hours:.1f}
main_quest: "{main_quest}"
tags: [{tags_str}]
generated: {datetime.now().isoformat()}
---
"""

    def _write_note(self, target_date: date, content: str) -> Path:
        """Atomic write to Daily/M D YY Daily Note.md (or weekly folder if exists)."""
        # Get the appropriate path (weekly folder or flat)
        note_path = self._get_note_path(target_date)

        # Ensure parent directory exists
        note_path.parent.mkdir(parents=True, exist_ok=True)

        atomic_write_text(note_path, content)
        logger.info(f"[DailyNotes] Written: {note_path}")
        return note_path

    async def generate_for_date(self, target_date: date, force: bool = False) -> GenerationResult:
        """
        Generate daily note for specific date.

        Args:
            target_date: Date to generate note for
            force: If True, overwrite existing note

        Returns:
            GenerationResult with success status and details
        """
        result = GenerationResult(date=target_date)

        # Check if disabled
        if not self.enabled:
            result.skipped_reason = "disabled"
            logger.info(f"[DailyNotes] Skipped {target_date}: feature disabled")
            return result

        # Get conversations for date (needed for both new and auto-update checks)
        convos = self._get_conversations_for_date(target_date)
        result.conversation_count = len(convos)

        # Check if already exists — but allow auto-update when new data is significant
        if not force and self.note_exists(target_date):
            if self._should_auto_update(target_date, len(convos)):
                logger.info(f"[DailyNotes] Auto-updating {target_date} with {len(convos)} conversations")
            else:
                result.skipped_reason = "already_exists"
                result.output_path = self._find_existing_note_path(target_date)
                logger.info(f"[DailyNotes] Skipped {target_date}: note already exists")
                return result

        if not convos:
            result.skipped_reason = "no_conversations"
            logger.info(f"[DailyNotes] Skipped {target_date}: no conversations")
            return result

        # Calculate stats
        first_ts = convos[0].get("timestamp")
        last_ts = convos[-1].get("timestamp")

        if isinstance(first_ts, str):
            first_ts = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
        if isinstance(last_ts, str):
            last_ts = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))

        # Wall-clock span (for display: "10:00 to 20:00")
        span_hours = (last_ts - first_ts).total_seconds() / 3600 if first_ts and last_ts else 0
        # Active duration (estimated actual usage time)
        active_hours = self._calculate_active_duration(convos)
        result.duration_hours = active_hours  # Use active duration for stats
        result.intensity = self._calculate_intensity(convos, active_hours)

        # Format for LLM
        formatted = self._format_conversations(convos)
        temporal_claim_audit = build_temporal_claim_audit(convos)
        tool_receipts = load_turn_tool_receipts(target_date)
        tool_use_record = build_tool_use_record(convos, tool_receipts)
        first_time = first_ts.strftime("%H:%M") if first_ts else "??:??"
        last_time = last_ts.strftime("%H:%M") if last_ts else "??:??"

        # Status-claim conflict guard (2026-09-04): current enrollment/
        # employment/residence facts, injected as an authoritative block
        # AND used as a post-generation contradiction check below. A failed
        # read is a hard stop (F12b): a guard-less note is never written.
        try:
            status_facts = self._current_status_facts()
        except RetrievalError as e:
            logger.warning(
                f"[DailyNotes] Note not generated for {target_date}: status guard unavailable ({e.source}: {e.reason})"
            )
            result.error = "status_guard_unavailable"
            return result
        status_block = authoritative_facts_block(status_facts)

        # Build prompt with user display name substitution
        from utils.user_identity import get_user_display_name  # lazy import: live-config read
        user_name = get_user_display_name()
        prompt_template = SYSTEM_PROMPT_TEMPLATE.replace(
            "__SPEAKER_AUTHORITY_RULES__", SPEAKER_AUTHORITY_RULES).replace("__USER_NAME__", user_name)
        prompt = prompt_template.format(
            date=target_date.strftime("%B %d, %Y"),
            formatted_conversations=formatted,
            temporal_claim_audit=temporal_claim_audit,
            tool_use_record=tool_use_record,
            authoritative_status_facts=status_block,
            count=len(convos),
            first_time=first_time,
            last_time=last_time,
            span_hours=span_hours,
            active_hours=active_hours,
        )

        # Call LLM with fallback models
        # Try primary model first, then fallback to alternatives if it fails
        # Expanded list includes Claude, Gemini, and newer GPT models for better reliability
        models_to_try = [self.model_name] + [m for m in FALLBACK_MODELS if m != self.model_name]

        llm_response = None
        last_error = None

        for model in models_to_try:
            try:
                logger.info(f"[DailyNotes] Generating note for {target_date} ({len(convos)} conversations) using {model}")
                llm_response = await self.model_manager.generate_once(
                    prompt,
                    max_tokens=self.max_tokens,
                    model_name=model,
                )

                # Check for API error responses (model_manager returns these as content, not exceptions)
                if llm_response and llm_response.startswith("[API unavailable]"):
                    logger.warning(f"[DailyNotes] Model {model} unavailable, trying next...")
                    last_error = f"Model {model} unavailable"
                    llm_response = None
                    continue

                # Check for empty response (like glm-4.6 returning nothing)
                if llm_response and llm_response.startswith("[Error:"):
                    logger.warning(f"[DailyNotes] Model {model} returned error, trying next...")
                    last_error = f"Model {model} returned error"
                    llm_response = None
                    continue

                # Validate we got a reasonable response (not empty or just the prompt echoed back)
                if not llm_response or len(llm_response.strip()) < 100:
                    logger.warning(f"[DailyNotes] Model {model} returned too-short response ({len(llm_response) if llm_response else 0} chars), trying next...")
                    last_error = f"Model {model} returned too-short response"
                    llm_response = None
                    continue

                # Success! Break out of the loop
                logger.info(f"[DailyNotes] Successfully generated note using {model}")
                break

            except Exception as e:
                logger.warning(f"[DailyNotes] Model {model} failed: {e}, trying next...")
                last_error = str(e)
                llm_response = None
                continue

        # If all models failed, return error
        if not llm_response:
            result.error = f"All LLM models failed. Last error: {last_error}"
            logger.error(f"[DailyNotes] All models failed for {target_date}: {last_error}")
            return result

        llm_response, status_conflicts = remove_conflicting_claims(llm_response, status_facts)
        for c in status_conflicts:
            logger.warning(
                f'[DailyNotes] Removed status-claim conflict ({c.family}) for {target_date}: '
                f'"{c.claim_text}" contradicts current {c.relation}={c.value}'
            )

        # LOG-ONLY tool-attribution post-check (2026-10-12): never edits the note.
        try:
            result.unrun_tool_mentions = find_unrun_tool_mentions(llm_response, tool_receipts)
            if result.unrun_tool_mentions:
                logger.warning(
                    f"[DailyNotes] {target_date}: note names tool(s) that ran in no turn: "
                    f"{', '.join(result.unrun_tool_mentions)}"
                )
        except Exception as e:  # degrades: tool-attribution check skipped
            logger.debug(f"[DailyNotes] tool-attribution check skipped: {e}")

        # Extract main quest for frontmatter
        main_quest = self._extract_main_quest(llm_response)

        # Generate contextual tags
        content_tags = []
        if self.tag_generation_enabled and self.tag_generator:
            try:
                tag_metadata = {
                    'main_quest': main_quest,
                    'intensity': result.intensity,
                    'conversations': len(convos),
                    'active_hours': active_hours,
                }
                tag_result = await self.tag_generator.generate_tags(
                    llm_response,
                    note_type="daily",
                    metadata=tag_metadata
                )
                content_tags = tag_result.tags
                logger.info(f"[DailyNotes] Generated {len(content_tags)} tags: {', '.join(content_tags)}")
            except Exception as e:
                logger.warning(f"[DailyNotes] Tag generation failed: {e}, continuing without tags")

        # Build full note
        frontmatter = self._build_frontmatter(
            target_date, len(convos), span_hours, active_hours, result.intensity, main_quest,
            content_tags=content_tags
        )
        header = f"\n# Daily Note - {target_date.strftime('%B %d, %Y')}\n\n"
        full_content = frontmatter + header + llm_response.strip() + "\n"

        # Write to vault
        try:
            result.output_path = self._write_note(target_date, full_content)
            result.success = True
            logger.info(f"[DailyNotes] Success: {result.output_path}")

            # Trigger narrative context refresh after new daily note
            await self._trigger_narrative_refresh()

        except Exception as e:
            result.error = str(e)
            logger.error(f"[DailyNotes] Failed to write note: {e}")

        return result

    async def _trigger_narrative_refresh(self) -> None:
        """
        Trigger narrative context refresh after daily note creation.

        This is a non-blocking operation - failures are logged but don't
        affect the daily note generation result.
        """
        try:
            from config.app_config import NARRATIVE_CONTEXT_ENABLED  # lazy import: live-config
            if not NARRATIVE_CONTEXT_ENABLED:
                return

            from memory.memory_consolidator import MemoryConsolidator  # lazy import: layering
            consolidator = MemoryConsolidator(self.model_manager)

            # Recent user statements feed the streak ledger (2026-09-05).
            user_statements = []
            try:
                user_statements = self.corpus_manager.get_recent_memories(60)
            except Exception as e:
                logger.debug(f"[DailyNotes] No user statements for the streak ledger: {e}")

            # Generate narrative from Obsidian notes (no corpus fallback needed for fresh trigger)
            narrative = await consolidator.generate_narrative_context(user_statements=user_statements)

            if narrative:
                self.corpus_manager.save_narrative_context(narrative)
                logger.info("[DailyNotes] Narrative context refreshed after daily note creation")

        except Exception as e:
            # Non-critical: log and continue
            logger.warning(f"[DailyNotes] Failed to refresh narrative context: {e}")

    async def generate_yesterday_if_missing(self) -> Optional[GenerationResult]:
        """
        Startup catch-up: generate yesterday's note if missing.

        Returns:
            GenerationResult if generated, None if skipped
        """
        yesterday = date.today() - timedelta(days=1)

        if self.note_exists(yesterday):
            logger.debug(f"[DailyNotes] Yesterday's note exists, skipping catch-up")
            return None

        logger.info(f"[DailyNotes] Catch-up: generating yesterday's note ({yesterday})")
        return await self.generate_for_date(yesterday)

    async def generate_date_range(self, start_date: date, end_date: date,
                                   force: bool = False) -> List[GenerationResult]:
        """
        Generate notes for a date range.

        Args:
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)
            force: If True, overwrite existing notes

        Returns:
            List of GenerationResult for each date
        """
        results = []
        current = start_date

        while current <= end_date:
            result = await self.generate_for_date(current, force=force)
            results.append(result)
            current += timedelta(days=1)

        return results
