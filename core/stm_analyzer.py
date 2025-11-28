"""
# core/stm_analyzer.py

Module Contract
- Purpose: Short-Term Memory analyzer that generates concise JSON summaries of recent conversation context
- Inputs:
  - recent_memories: List of recent conversation dicts (query/response pairs)
  - user_query: Current user input
  - last_assistant_response: Optional last assistant message for context
- Outputs:
  - Dict with: topic, user_question, intent, tone, open_threads, constraints
- Key pieces:
  - analyze(): Main async method that calls LLM to analyze context
  - _format_memories(): Converts memory dicts to readable conversation text
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

        # Build STM analysis prompt
        prompt = f"""Analyze this conversation's SHORT-TERM context only.

Recent conversation:
{conversation_text}

Current user query: {user_query}

Return ONLY valid JSON with these fields:
- "topic": current conversation topic (brief, 2-5 words)
- "user_question": what user is asking now (one sentence)
- "intent": what they really want (one sentence)
- "tone": emotional tone (one word: neutral/casual/concerned/frustrated/excited)
- "open_threads": unresolved questions or topics from conversation (list of strings)
- "constraints": any constraints mentioned (time, tools, safety, etc.) (list of strings)

Example:
{{
  "topic": "Python debugging",
  "user_question": "How to fix the timeout error",
  "intent": "Get practical solution to immediate technical problem",
  "tone": "frustrated",
  "open_threads": ["Performance optimization", "Error handling strategy"],
  "constraints": ["Limited to standard library", "Production environment"]
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

    def _format_memories(self, memories: List[Dict]) -> str:
        """
        Convert memory dicts to readable conversation text.

        Args:
            memories: List of conversation dicts with 'query' and 'response' keys

        Returns:
            Formatted conversation string
        """
        lines = []
        for mem in memories:
            query = mem.get('query', '').strip()
            response = mem.get('response', '').strip()

            if query:
                # Truncate very long messages for STM context
                query_short = query[:200] + "..." if len(query) > 200 else query
                lines.append(f"User: {query_short}")

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
            "open_threads": [],
            "constraints": []
        }
