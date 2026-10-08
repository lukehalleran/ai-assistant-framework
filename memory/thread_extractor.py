# memory/thread_extractor.py
"""
LLM-based extraction of open threads from session conversations.

Module Contract
- Purpose: Uses LLM to identify open loops (commitments, deadlines, unfinished
  topics, unanswered questions) from conversation transcripts, and detect when
  existing open threads have been resolved.
- Inputs:
  - session_conversations: list of conversation dicts (query/response pairs)
  - open_threads: existing open threads — used for resolution detection AND
    shown to the extraction prompt as "already tracked" so the LLM doesn't
    re-extract a task that already has a thread
  - model_manager: LLM abstraction for generate_once()
- Outputs:
  - List of new OpenThread objects extracted from conversations
  - List of (thread_id, resolution) tuples for resolved threads
- Key behaviors:
  - Two separate LLM calls: extraction + resolution detection
  - Few-shot prompt examples for each ThreadType
  - Robust JSON parsing with find("[") / rfind("]") pattern
  - Resolution detection skipped if no existing open threads
  - Resolution prompt instructs the LLM to resolve ALL duplicate threads a
    completion applies to, not just the closest match
  - Uses temperature=0.0 for deterministic extraction
- Dependencies:
  - memory.thread_models (data models)
  - models.model_manager (LLM calls)
  - config.app_config (model alias)
"""

import json
import re
import time
from datetime import datetime
from typing import List, Optional, Tuple

from utils.logging_utils import get_logger
from memory import fact_source
from memory.thread_models import DisputedResolution, OpenThread, ThreadType, ThreadStatus

logger = get_logger("thread_extractor")


EXTRACTION_PROMPT = """You are a conversation analyst. Review the conversation below and extract open threads — things that are unresolved, promised, or need follow-up.

Thread types:
- "commitment": Something the user said they would do (e.g., "I'll study for the exam", "I need to call my doctor")
- "deadline": Something with an explicit or implied deadline (e.g., "exam next Tuesday", "presentation on Friday")
- "unfinished": A topic that was started but not completed or resolved
- "question": A question the user asked that wasn't fully answered, or that they need to find out

For each thread, output this JSON format. Output a JSON array:
[
  {{
    "topic": "short label (3-10 words)",
    "summary": "brief description of the open thread",
    "thread_type": "commitment|deadline|unfinished|question",
    "urgency": 0.0-1.0,
    "resolution_hint": "what would close this thread",
    "deadline_date": "YYYY-MM-DD or null"
  }}
]

Examples:
- User says "I need to study for my exam next Tuesday" → {{"topic": "Study for exam", "summary": "User needs to study for an exam happening next Tuesday", "thread_type": "deadline", "urgency": 0.8, "resolution_hint": "User confirms they studied or the exam passed", "deadline_date": "2026-03-24"}}
- User says "I should call my doctor about the results" → {{"topic": "Call doctor about results", "summary": "User mentioned needing to call their doctor about test results", "thread_type": "commitment", "urgency": 0.6, "resolution_hint": "User confirms they called", "deadline_date": null}}
- Discussion about a project plan that was left incomplete → {{"topic": "Project plan discussion", "summary": "Was discussing project architecture but conversation moved on", "thread_type": "unfinished", "urgency": 0.3, "resolution_hint": "Resume the project plan discussion", "deadline_date": null}}

Rules:
- Only extract genuinely open threads — not things that were resolved in the conversation
- Do NOT extract a thread for any task listed under ALREADY TRACKED below — those are already stored. This includes the same underlying task described in different words. Re-extracting creates duplicates that keep resurfacing after the user finishes the task.
- Urgency 0.0-1.0: deadlines coming soon = high, casual mentions = low
- Output ONLY a valid JSON array, no other text
- If no open threads exist, output []
- Maximum 5 threads per session
- GROUNDING: lines starting "Assistant:" are context only, never evidence. A thread must be grounded in what the USER said; an assistant statement is never a source for a commitment, deadline, date or time. Keep a time zone exactly as the user wrote it (e.g. "2:30 EST") — never convert it.
- TEMPORAL: Today's date is {today}. When the user mentions relative dates ("tomorrow", "next Tuesday", "this weekend"), resolve them to absolute dates in deadline_date AND in the summary. Example: "I have an exam tomorrow" on 2026-05-19 → deadline_date: "2026-05-20", summary: "User has an exam on Tue 2026-05-20"

ALREADY TRACKED (open threads that already exist — do NOT re-extract these tasks):
{tracked_threads}

CONVERSATION:
{conversation_text}

Open threads (JSON array only):"""


RESOLUTION_PROMPT = """You are a conversation analyst. Given existing open threads and a recent conversation, determine which threads (if any) have been resolved.

A thread is resolved when:
- The user explicitly says they completed the task ("I studied", "I called the doctor")
- The deadline has passed and the user discussed the outcome
- The topic was fully addressed in this conversation
- The user explicitly cancels or drops the commitment

A thread is DISPUTED (not resolved) when the user says its premise is wrong or was never true (e.g. "that's not a standing meeting", "I never agreed to that") — it was neither done nor cancelled.

Existing open threads:
{threads_json}

Recent conversation:
{conversation_text}

For each resolved thread, output this JSON format. Output a JSON array:
[
  {{"thread_id": "the-thread-id", "resolution": "brief description of how it was resolved", "outcome": "resolved|disputed"}}
]

Rules:
- Only mark threads as resolved if there is clear evidence in the conversation
- Lines starting "Assistant:" are context only; the user's own words are the evidence
- Use "outcome": "disputed" only when the USER disputes the thread's premise; otherwise "resolved"
- Do NOT mark a thread resolved just because it wasn't mentioned
- The thread list may contain DUPLICATES — several entries describing the same underlying task in different words (e.g. "Homework due Friday" and "Last 2 homework questions"). When the conversation resolves a task, output EVERY thread that task resolves, not just the single closest match.
- Output ONLY a valid JSON array, no other text
- If no threads were resolved, output []
- Today's date is {today}. Use this to judge whether deadlines have passed.

Resolved threads (JSON array only):"""


def _build_conversation_text(session_conversations: List[dict], max_chars: int = 6000) -> str:
    """Build a conversation excerpt string from session dicts."""
    excerpts = []
    for e in session_conversations:
        q = (e.get("query") or "").strip()
        a = (e.get("response") or "").strip()
        if q or a:
            lines = []
            if q:
                lines.append(f"User: {q[:400]}")
            if a:
                lines.append(f"Assistant: {a[:500]}")
            excerpts.append("\n".join(lines))

    text = "\n\n".join(excerpts)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


# ---------------------------------------------------------------------------
# User-authored provenance for new threads (2026-10-08, BC-75/BC-51)
# ---------------------------------------------------------------------------

_MONTH_DAY_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(\d{1,2})\b",
    re.IGNORECASE,
)
_WEEKDAY_TOKEN_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tues|thurs|thur)\b",
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(?:([ap])\.?m\b\.?)?", re.IGNORECASE)
_ZONES = "EST|EDT|CST|CDT|MST|MDT|PST|PDT|AKST|AKDT|HST|UTC|GMT|ET|CT|MT|PT"
_ZONE_TIME_RE = re.compile(
    rf"\b(\d{{1,2}}(?::\d{{2}})?)\s*(?:[AaPp]\.?[Mm]\.?)?\s*({_ZONES})\b")
_PROVENANCE_TYPES = frozenset({ThreadType.DEADLINE, ThreadType.COMMITMENT})
_EXCERPT_CHARS = 200


def _clock_key(hour: str, minute: Optional[str]) -> str:
    return f"{int(hour)}:{minute or '00'}"


def _date_time_tokens(text: str) -> set:
    """Weekday names, month-day pairs and clock times found in text."""
    out = set()
    for m in _WEEKDAY_TOKEN_RE.finditer(text or ""):
        out.add("wd:" + m.group(1).lower()[:3])
    for m in _MONTH_DAY_RE.finditer(text or ""):
        out.add(f"md:{m.group(1).lower()[:3]}{int(m.group(2))}")
    for m in _CLOCK_RE.finditer(text or ""):
        if m.group(2) or m.group(3):  # "2:30" / "2pm" — never a bare number
            out.add("t:" + _clock_key(m.group(1), m.group(2)))
    return out


def _user_support(topic: str, summary: str, user_texts: List[str]) -> Optional[Tuple[str, str]]:
    """(excerpt, zone) when USER text supports the thread, else None.

    Supported = a date/time token of the thread (weekday, month-day, clock
    time) OR >=2 of its topic's key words (all of them when it has fewer)
    appears in a user line. ``zone`` is a time-zone abbreviation the user
    wrote next to one of the thread's clock times ("" when none).
    """
    thread_tokens = _date_time_tokens(f"{topic} {summary}")
    topic_words = fact_source._tokens(topic)
    need = min(2, len(topic_words))
    lines = [ln.strip() for t in user_texts for ln in t.splitlines() if ln.strip()]
    excerpt = ""
    for ln in lines:
        if thread_tokens & _date_time_tokens(ln):
            excerpt = ln
            break
    if not excerpt and need:
        for ln in lines:
            if len(topic_words & fact_source._tokens(ln)) >= need:
                excerpt = ln
                break
    if not excerpt:
        return None
    zone = ""
    for ln in lines:
        for m in _ZONE_TIME_RE.finditer(ln):
            mm = re.match(r"(\d{1,2})(?::(\d{2}))?", m.group(1))
            if "t:" + _clock_key(mm.group(1), mm.group(2)) in thread_tokens:
                zone = m.group(2)
                break
        if zone:
            break
    return excerpt[:_EXCERPT_CHARS], zone


class ThreadExtractionError(RuntimeError):
    """The LLM call failed or returned nothing parseable.

    Distinct from a successful "no threads" ([]): the caller (shutdown) logs
    the failure instead of reading it as an empty session (BC-47). Messages
    never carry response text (privacy).
    """


def _parse_json_array_strict(raw: Optional[str]) -> Optional[List[dict]]:
    """Robust JSON array parsing with find("[") / rfind("]") pattern.

    Returns None when no array can be parsed; a parsed EMPTY array is [] (a
    genuine "nothing found").
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Find the JSON array boundaries
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start:end + 1])
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    return None


def _parse_json_array(raw: str) -> List[dict]:
    """Lenient wrapper: unparseable input reads as []."""
    return _parse_json_array_strict(raw) or []


class ThreadExtractor:
    """
    LLM-based extractor for open threads from conversations.

    Two-phase approach:
    1. extract_new_threads(): identify new open loops from session
    2. detect_resolutions(): check if existing threads were addressed
    """

    def __init__(self, model_manager=None):
        self.model_manager = model_manager

    async def extract_new_threads(
        self,
        session_conversations: List[dict],
        open_threads: Optional[List[OpenThread]] = None,
    ) -> List[OpenThread]:
        """
        Extract new open threads from session conversations.

        Args:
            session_conversations: List of conversation dicts with query/response
            open_threads: Already-tracked open threads, shown to the LLM as
                "do not re-extract" — without this the extractor re-creates a
                thread for a task that is already tracked (or was just resolved)
                every session the task is mentioned

        Returns:
            List of new OpenThread objects

        Raises:
            ThreadExtractionError: the LLM call failed or its response was
                empty/unparseable (distinct from a genuine empty list).
        """
        if not self.model_manager or not hasattr(self.model_manager, "generate_once"):
            return []

        if not session_conversations:
            return []

        conversation_text = _build_conversation_text(session_conversations)
        if not conversation_text.strip():
            return []

        tracked_lines = [
            f"- {t.topic}: {(t.summary or '')[:150]}"
            for t in (open_threads or [])[:20]
        ]
        tracked_threads = "\n".join(tracked_lines) if tracked_lines else "(none)"

        today_str = datetime.now().strftime("%A, %Y-%m-%d")
        prompt = EXTRACTION_PROMPT.format(
            conversation_text=conversation_text,
            today=today_str,
            tracked_threads=tracked_threads,
        )

        try:
            model_alias = self._get_model_alias()
            raw = await self.model_manager.generate_once(
                prompt,
                model_name=model_alias if model_alias else None,
                max_tokens=800,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning(f"[ThreadExtractor] LLM extraction failed: {type(e).__name__}")
            raise ThreadExtractionError(
                f"generate_once failed: {type(e).__name__}"
            ) from e

        if not raw:
            raise ThreadExtractionError("generate_once returned an empty response")

        items = _parse_json_array_strict(raw)
        if items is None:
            raise ThreadExtractionError(f"unparseable response ({len(raw)} chars)")
        threads = []
        now = time.time()
        user_texts = [text for _i, text, _tid in fact_source.iter_user_messages(session_conversations)]

        for item in items[:5]:  # cap at 5
            try:
                thread_type_str = item.get("thread_type", "unfinished")
                try:
                    thread_type = ThreadType(thread_type_str)
                except ValueError:
                    thread_type = ThreadType.UNFINISHED

                deadline = item.get("deadline_date")
                if deadline and not isinstance(deadline, str):
                    deadline = None
                if deadline and deadline.lower() in ("null", "none", ""):
                    deadline = None

                topic = item.get("topic", "Unknown thread")[:200]
                summary = item.get("summary", "")[:1000]
                support = _user_support(topic, summary, user_texts)
                if thread_type in _PROVENANCE_TYPES and support is None:
                    logger.info(
                        f"[ThreadExtractor] dropped assistant-only thread "
                        f"(type={thread_type.value})"
                    )
                    continue
                source_excerpt, zone = support or ("", "")
                if zone and not re.search(rf"\b{zone}\b", f"{topic} {summary}"):
                    summary = f"{summary[:1000 - 20 - len(zone)]} ({zone} as written)"

                thread = OpenThread(
                    topic=topic,
                    summary=summary,
                    source_summary=source_excerpt,
                    thread_type=thread_type,
                    urgency=max(0.0, min(1.0, float(item.get("urgency", 0.5)))),
                    mentioned_at=now,
                    last_referenced=now,
                    resolution_hint=item.get("resolution_hint", "")[:500],
                    deadline_date=deadline,
                )
                threads.append(thread)
            except (ValueError, KeyError, TypeError) as e:
                logger.debug(f"[ThreadExtractor] Skipping invalid thread: {e}")
                continue

        if threads:
            logger.info(f"[ThreadExtractor] Extracted {len(threads)} new thread(s)")

        return threads

    async def detect_resolutions(
        self,
        session_conversations: List[dict],
        open_threads: List[OpenThread],
    ) -> List[Tuple[str, str]]:
        """
        Detect which existing open threads were resolved in this session.

        Args:
            session_conversations: Recent conversation dicts
            open_threads: Currently open threads to check

        Returns:
            List of (thread_id, resolution_description) tuples

        Raises:
            ThreadExtractionError: as for extract_new_threads().
        """
        if not self.model_manager or not hasattr(self.model_manager, "generate_once"):
            return []

        if not open_threads:
            return []

        if not session_conversations:
            return []

        conversation_text = _build_conversation_text(session_conversations)
        if not conversation_text.strip():
            return []

        # Build threads JSON for the prompt
        threads_data = []
        for t in open_threads[:20]:  # cap context
            threads_data.append({
                "thread_id": t.thread_id,
                "topic": t.topic,
                "summary": t.summary,
                "thread_type": t.thread_type.value,
            })

        threads_json = json.dumps(threads_data, indent=2)
        today_str = datetime.now().strftime("%A, %Y-%m-%d")
        prompt = RESOLUTION_PROMPT.format(
            threads_json=threads_json,
            conversation_text=conversation_text,
            today=today_str,
        )

        try:
            model_alias = self._get_model_alias()
            raw = await self.model_manager.generate_once(
                prompt,
                model_name=model_alias if model_alias else None,
                max_tokens=400,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning(f"[ThreadExtractor] LLM resolution detection failed: {type(e).__name__}")
            raise ThreadExtractionError(
                f"generate_once failed: {type(e).__name__}"
            ) from e

        if not raw:
            raise ThreadExtractionError("generate_once returned an empty response")

        items = _parse_json_array_strict(raw)
        if items is None:
            raise ThreadExtractionError(f"unparseable response ({len(raw)} chars)")
        resolutions = []

        # Build set of valid thread IDs for validation
        valid_ids = {t.thread_id for t in open_threads}

        for item in items:
            thread_id = item.get("thread_id", "")
            resolution = item.get("resolution", "")
            if str(item.get("outcome", "")).lower() == "disputed":
                resolution = DisputedResolution(resolution)
            if thread_id and thread_id in valid_ids:
                resolutions.append((thread_id, resolution))

        if resolutions:
            logger.info(f"[ThreadExtractor] Detected {len(resolutions)} resolution(s)")

        return resolutions

    @staticmethod
    def _get_model_alias() -> str:
        """Get model alias from config."""
        try:
            from config.app_config import THREAD_MODEL_ALIAS
            return THREAD_MODEL_ALIAS
        except ImportError:
            return ""
