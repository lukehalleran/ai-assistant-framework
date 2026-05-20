# memory/thread_extractor.py
"""
LLM-based extraction of open threads from session conversations.

Module Contract
- Purpose: Uses LLM to identify open loops (commitments, deadlines, unfinished
  topics, unanswered questions) from conversation transcripts, and detect when
  existing open threads have been resolved.
- Inputs:
  - session_conversations: list of conversation dicts (query/response pairs)
  - open_threads: existing open threads for resolution detection
  - model_manager: LLM abstraction for generate_once()
- Outputs:
  - List of new OpenThread objects extracted from conversations
  - List of (thread_id, resolution) tuples for resolved threads
- Key behaviors:
  - Two separate LLM calls: extraction + resolution detection
  - Few-shot prompt examples for each ThreadType
  - Robust JSON parsing with find("[") / rfind("]") pattern
  - Resolution detection skipped if no existing open threads
  - Uses temperature=0.0 for deterministic extraction
- Dependencies:
  - memory.thread_models (data models)
  - models.model_manager (LLM calls)
  - config.app_config (model alias)
"""

import json
import time
from datetime import datetime
from typing import List, Optional, Tuple

from utils.logging_utils import get_logger
from memory.thread_models import OpenThread, ThreadType, ThreadStatus

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
- Urgency 0.0-1.0: deadlines coming soon = high, casual mentions = low
- Output ONLY a valid JSON array, no other text
- If no open threads exist, output []
- Maximum 5 threads per session
- TEMPORAL: Today's date is {today}. When the user mentions relative dates ("tomorrow", "next Tuesday", "this weekend"), resolve them to absolute dates in deadline_date AND in the summary. Example: "I have an exam tomorrow" on 2026-05-19 → deadline_date: "2026-05-20", summary: "User has an exam on Tue 2026-05-20"

CONVERSATION:
{conversation_text}

Open threads (JSON array only):"""


RESOLUTION_PROMPT = """You are a conversation analyst. Given existing open threads and a recent conversation, determine which threads (if any) have been resolved.

A thread is resolved when:
- The user explicitly says they completed the task ("I studied", "I called the doctor")
- The deadline has passed and the user discussed the outcome
- The topic was fully addressed in this conversation
- The user explicitly cancels or drops the commitment

Existing open threads:
{threads_json}

Recent conversation:
{conversation_text}

For each resolved thread, output this JSON format. Output a JSON array:
[
  {{"thread_id": "the-thread-id", "resolution": "brief description of how it was resolved"}}
]

Rules:
- Only mark threads as resolved if there is clear evidence in the conversation
- Do NOT mark a thread resolved just because it wasn't mentioned
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


def _parse_json_array(raw: str) -> List[dict]:
    """Robust JSON array parsing with find("[") / rfind("]") pattern."""
    if not raw or not raw.strip():
        return []

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
        return []

    try:
        parsed = json.loads(text[start:end + 1])
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    return []


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
        self, session_conversations: List[dict]
    ) -> List[OpenThread]:
        """
        Extract new open threads from session conversations.

        Args:
            session_conversations: List of conversation dicts with query/response

        Returns:
            List of new OpenThread objects
        """
        if not self.model_manager or not hasattr(self.model_manager, "generate_once"):
            return []

        if not session_conversations:
            return []

        conversation_text = _build_conversation_text(session_conversations)
        if not conversation_text.strip():
            return []

        today_str = datetime.now().strftime("%A, %Y-%m-%d")
        prompt = EXTRACTION_PROMPT.format(
            conversation_text=conversation_text,
            today=today_str,
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
            logger.warning(f"[ThreadExtractor] LLM extraction failed: {e}")
            return []

        if not raw:
            return []

        items = _parse_json_array(raw)
        threads = []
        now = time.time()

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

                thread = OpenThread(
                    topic=item.get("topic", "Unknown thread")[:200],
                    summary=item.get("summary", "")[:1000],
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
            logger.warning(f"[ThreadExtractor] LLM resolution detection failed: {e}")
            return []

        if not raw:
            return []

        items = _parse_json_array(raw)
        resolutions = []

        # Build set of valid thread IDs for validation
        valid_ids = {t.thread_id for t in open_threads}

        for item in items:
            thread_id = item.get("thread_id", "")
            resolution = item.get("resolution", "")
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
