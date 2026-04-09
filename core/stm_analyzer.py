"""
# core/stm_analyzer.py

Module Contract
- Purpose: Short-Term Memory analyzer that generates concise JSON summaries of recent conversation context
- Inputs:
  - recent_memories: List of recent conversation dicts (query/response pairs).
    Caller is expected to time-window these (typically last 24h via
    CorpusManager.get_recent_within_hours), capped by STM_MAX_RECENT_MESSAGES.
  - user_query: Current user input
  - last_assistant_response: Optional last assistant message for context
- Side inputs (read internally, not passed):
  - Last STM_INJECT_DAILY_NOTES_DAYS daily notes from the Obsidian vault, read
    via utils.daily_notes_generator.read_daily_note(). Used to give STM
    cross-day recall disambiguation. Gracefully degrades when notes are
    missing (e.g. session starts before catch-up has run).
- Outputs:
  - Dict with: topic, user_question, intent, tone, reference_type, temporal_facts, open_threads, constraints
  - reference_type ∈ {new_event, recall, clarification, correction, unclear} — disambiguates whether the current message is a new report or a restatement of prior context. Defaults to "unclear" when uncertain.
  - temporal_facts: normalized facts about user's current state, with collapse-toward-fewer-events disambiguation rule applied.
- Key pieces:
  - analyze(): Main async method that calls LLM to analyze context
  - _format_memories(): Converts memory dicts to readable conversation text with relative day labels
  - _parse_json(): Robust JSON parser with fallback handling
- Side effects:
  - None beyond LLM API call and logging
"""
import json
from typing import List, Dict, Any, Optional
from utils.logging_utils import get_logger

logger = get_logger("stm_analyzer")


class STMAnalyzer:
    """
    Analyzes short-term conversation context using a lightweight LLM pass.

    Returns structured JSON summaries to help the main model understand
    immediate conversation state without re-reading full message history.
    """

    def __init__(self, model_manager, model_name: str = "gpt-4o-mini"):
        """
        Initialize STM analyzer.

        Args:
            model_manager: ModelManager instance for LLM calls
            model_name: Model to use for analysis (default: gpt-4o-mini for speed/cost)
        """
        self.model_manager = model_manager
        self.model_name = model_name
        logger.info(f"[STMAnalyzer] Initialized with model: {model_name}")

    async def analyze(
        self,
        recent_memories: List[Dict[str, Any]],
        user_query: str,
        last_assistant_response: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze short-term conversation context.

        Args:
            recent_memories: Recent conversation turns (list of dicts with 'query'/'response')
            user_query: Current user input
            last_assistant_response: Optional last assistant message for additional context

        Returns:
            Dict with fields:
                - topic: Current conversation topic (brief)
                - user_question: What user is asking now
                - intent: What they really want
                - tone: Emotional tone
                - open_threads: Unresolved questions (list)
                - constraints: Any constraints noted (list)
        """
        conversation_text = self._format_memories(recent_memories)
        daily_notes_text = self._get_recent_daily_notes_text()

        # Daily notes section is included only when at least one note was found.
        # When sessions start before catch-up has run, this gracefully degrades
        # to the old (recent-conversation-only) behavior.
        notes_section = ""
        if daily_notes_text:
            notes_section = (
                "\n\nRecent daily notes (Daemon-generated EOD summaries — use these "
                "to identify whether the current message restates an event that has "
                "already happened on a prior day):\n"
                f"{daily_notes_text}\n"
            )

        # Build STM analysis prompt
        prompt = f"""Analyze this conversation's SHORT-TERM context only.

Recent conversation:
{conversation_text}
{notes_section}
Current user query: {user_query}

Return ONLY valid JSON with these fields:
- "topic": current conversation topic (brief, 2-5 words)
- "user_question": what user is asking now, paraphrased neutrally (one sentence). If they are restating something, paraphrase the act of restatement, not the underlying claim.
- "intent": what they really want from this turn (one sentence)
- "tone": emotional tone (one word: neutral/casual/concerned/frustrated/excited)
- "reference_type": ONE of:
    * "new_event"     — user is reporting something not present in recent conversation
    * "recall"        — user is restating, re-emphasizing, or returning to an event already in recent conversation
    * "clarification" — user is adding a small detail to an existing topic
    * "correction"    — user is contradicting a claim the assistant made
    * "unclear"       — cannot tell from context (DEFAULT when uncertain)
  Treating a recall as a new event is the more dangerous error. Default to "recall" or "unclear" when in doubt. If the user names actors, locations, or events you do NOT see in recent conversation, classify as "unclear" — you may simply lack the source memory.
- "temporal_facts": list of normalized facts about the user's current state with explicit time anchors. Resolve ambiguous references conservatively: collapse toward fewer events, not more. (list of strings, may be empty)
- "open_threads": unresolved questions or topics from conversation (list of strings)
- "constraints": any constraints mentioned (time, tools, safety, etc.) (list of strings)

CRITICAL DISAMBIGUATION RULES:
1. If the current message names the same actors + action + outcome as something in recent conversation, classify as "recall" — not a new event.
2. Resolve ambiguous temporal phrases by collapsing toward fewer events. "Did not sleep" + "fight mode all night" on the same morning = ONE bad night, not two.
3. Do NOT invent a count or pattern. If you only have evidence of one occurrence of something, say so explicitly in temporal_facts.
4. When the user uses phrases like "told them what happened", "this situation", "that thing" without naming a new event, assume they are referencing something already known — classify as "recall" or "unclear".

Example (new event):
{{
  "topic": "Python debugging",
  "user_question": "How to fix the timeout error",
  "intent": "Get practical solution to immediate technical problem",
  "tone": "frustrated",
  "reference_type": "new_event",
  "temporal_facts": ["user is currently debugging a timeout error"],
  "open_threads": ["Performance optimization", "Error handling strategy"],
  "constraints": ["Limited to standard library", "Production environment"]
}}

Example (recall — user re-emphasizing something already discussed):
{{
  "topic": "Police response",
  "user_question": "User is re-emphasizing that police declined to act on the prior incident",
  "intent": "Express continued distress about the same event, not report a new one",
  "tone": "concerned",
  "reference_type": "recall",
  "temporal_facts": ["one prior incident of police inaction is in recent context; no evidence of a second"],
  "open_threads": [],
  "constraints": []
}}

Return JSON only, no markdown or extra text:"""

        try:
            logger.debug(f"[STMAnalyzer] Running analysis for query: {user_query[:50]}...")

            # Call model manager using generate_once for non-streaming response
            content = await self.model_manager.generate_once(
                prompt=prompt,
                model_name=self.model_name,
                system_prompt="You are a context analyzer that returns only valid JSON.",
                max_tokens=300,
                temperature=0.3  # Lower temp for more structured output
            )

            # Parse and return
            parsed = self._parse_json(content)
            logger.debug(f"[STMAnalyzer] Analysis complete: topic={parsed.get('topic')}, tone={parsed.get('tone')}")
            return parsed

        except Exception as e:
            logger.error(f"[STMAnalyzer] Analysis failed: {e}")
            return self._empty_summary()

    def _get_recent_daily_notes_text(self, num_days: Optional[int] = None) -> str:
        """Read the last N daily notes from the Obsidian vault and format them
        as a single text block for STM injection.

        Returns "" if the feature is disabled, no notes are available, or the
        helper module can't be imported. Pure read-only — never triggers
        generation or narrative refresh.
        """
        try:
            from config.app_config import STM_INJECT_DAILY_NOTES_DAYS
        except ImportError:
            STM_INJECT_DAILY_NOTES_DAYS = 0

        n = num_days if num_days is not None else STM_INJECT_DAILY_NOTES_DAYS
        if n <= 0:
            return ""

        try:
            from utils.daily_notes_generator import read_daily_note
        except ImportError:
            return ""

        from datetime import date, timedelta
        today = date.today()
        parts: List[str] = []

        for offset in range(1, n + 1):
            target = today - timedelta(days=offset)
            text = read_daily_note(target)
            if not text:
                continue
            label = "yesterday" if offset == 1 else f"{offset} days ago"
            parts.append(
                f"--- Daily note for {target.isoformat()} ({label}) ---\n{text.strip()}"
            )

        if not parts:
            logger.debug("[STMAnalyzer] No daily notes available for injection")
            return ""

        logger.debug(f"[STMAnalyzer] Injecting {len(parts)} daily note(s) into STM input")
        return "\n\n".join(parts)

    def _format_memories(self, memories: List[Dict]) -> str:
        """
        Convert memory dicts to readable conversation text with temporal markers.

        Args:
            memories: List of conversation dicts with 'query', 'response', and 'timestamp' keys

        Returns:
            Formatted conversation string with relative day labels
        """
        from datetime import datetime
        from utils.time_manager import format_relative_timestamp

        lines = []
        for mem in memories:
            query = mem.get('query', '').strip()
            response = mem.get('response', '').strip()

            # Extract timestamp for temporal context
            ts = mem.get('timestamp', '')
            ts_prefix = ""
            if ts:
                try:
                    if isinstance(ts, datetime):
                        ts_prefix = f"[{format_relative_timestamp(ts)}] "
                    elif isinstance(ts, str):
                        ts_prefix = f"[{format_relative_timestamp(datetime.fromisoformat(ts))}] "
                except (ValueError, TypeError):
                    pass

            if query:
                # Truncate very long messages for STM context
                query_short = query[:200] + "..." if len(query) > 200 else query
                lines.append(f"{ts_prefix}User: {query_short}")

            if response:
                # Truncate very long responses
                response_short = response[:200] + "..." if len(response) > 200 else response
                lines.append(f"Assistant: {response_short}")

        # Keep last 10 lines max (5 exchanges)
        return '\n'.join(lines[-10:])

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        """
        Robust JSON parser with fallback handling.

        Handles markdown-wrapped JSON, extra text, and malformed output.

        Args:
            raw: Raw LLM response text

        Returns:
            Parsed dict or empty summary on failure
        """
        try:
            # Try to extract JSON if wrapped in markdown
            if '```json' in raw:
                start = raw.find('```json') + 7
                end = raw.find('```', start)
                raw = raw[start:end].strip()
            elif '```' in raw:
                start = raw.find('```') + 3
                end = raw.find('```', start)
                raw = raw[start:end].strip()

            # Try direct parsing
            parsed = json.loads(raw)

            # Validate required fields
            required = ['topic', 'user_question', 'intent', 'tone', 'open_threads', 'constraints']
            for field in required:
                if field not in parsed:
                    logger.warning(f"[STMAnalyzer] Missing field '{field}' in JSON, using empty summary")
                    return self._empty_summary()

            # Backfill optional fields added later (graceful degradation if older
            # model output omits them)
            parsed.setdefault('reference_type', 'unclear')
            parsed.setdefault('temporal_facts', [])

            return parsed

        except json.JSONDecodeError as e:
            logger.warning(f"[STMAnalyzer] Failed to parse JSON: {e}. Raw: {raw[:100]}...")
            return self._empty_summary()

    def _empty_summary(self) -> Dict[str, Any]:
        """
        Fallback summary when analysis fails.

        Returns:
            Empty but valid STM summary dict
        """
        return {
            "topic": "unknown",
            "user_question": "",
            "intent": "",
            "tone": "neutral",
            "reference_type": "unclear",
            "temporal_facts": [],
            "open_threads": [],
            "constraints": []
        }
