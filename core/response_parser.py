"""
Response parsing and text cleaning utilities.

Extracted from DaemonOrchestrator to follow Single Responsibility Principle.
These are pure functions with no side effects - they take text and return cleaned text.

Module Contract:
- Purpose: Parse and clean LLM responses (thinking blocks, reflections, XML wrappers, artifacts)
- Inputs: Raw response strings from LLM
- Outputs: Cleaned response strings
- Key methods:
  - parse_thinking_block(response) → (thinking, answer): Tag-based extraction (<thinking>/<think>/<output>),
    then heuristic fallback (_detect_untagged_thinking) for models that dump reasoning without tags
  - _detect_untagged_thinking(response) → (thinking, answer): Pattern-based detection of untagged
    chain-of-thought (meta-reasoning phrases, instruction echoes). Requires ≥2 distinct pattern hits
    and a clean split point (blank-line gap + ≥20 chars answer).
  - likely_untagged_thinking(response) → bool: Fast streaming-time check — same heuristic patterns
    but no split required. Returns True when ≥2 patterns present. Used by handlers.py to suppress
    thinking during streaming before parse_thinking_block() can find a clean split.
  - has_incomplete_thinking_block(response) → bool: Streaming suppression (open tag, no close tag yet)
  - extract_incomplete_thinking(response) → str: Extract in-progress thinking for streaming indicator
  - strip_thinking_tag_leaks(text) → str: Remove partial/malformed thinking tags
  - strip_reflection_blocks(response) → str: Remove <reflect> and [SYSTEM QUALITY REFLECTION] blocks
  - strip_xml_wrappers(text) → str: Unwrap <result>/<answer>/<output>/<response> wrappers
  - strip_prompt_artifacts(text) → str: Remove echoed prompt section headers
- Side effects: None (pure functions)
"""

import re
from typing import Tuple


class ResponseParser:
    """
    Utility class for parsing and cleaning LLM responses.

    All methods are static and stateless - no instance required.
    """

    # Regex to catch leaked thinking tags: <think>, </think>, <thinking>, </thinking>,
    # and partial/malformed variants like /think>, /thinking>, <|think|>, etc.
    _THINK_TAG_LEAK_RE = re.compile(
        r'<\|?/?think(?:ing)?\|?>|'   # <think>, </think>, <thinking>, </thinking>, <|think|>
        r'/?think(?:ing)?>',            # /think>, /thinking> (missing opening <)
        re.IGNORECASE,
    )

    # Opening think tags for streaming detection
    _THINK_OPEN_TAGS = [("<thinking>", "</thinking>"), ("<think>", "</think>")]

    @staticmethod
    def has_incomplete_thinking_block(response: str) -> bool:
        """Check if response has an opening think tag but no closing tag yet.

        Used during streaming to detect that we're inside a thinking block
        before the closing tag has arrived, so thinking content isn't leaked
        as regular text.
        """
        if not response:
            return False
        lower = response.lower()
        for open_tag, close_tag in ResponseParser._THINK_OPEN_TAGS:
            if open_tag in lower and close_tag not in lower:
                return True
        return False

    @staticmethod
    def extract_incomplete_thinking(response: str) -> str:
        """Extract content after opening think tag when closing tag hasn't arrived.

        For streaming use: returns the thinking-in-progress text so it can be
        shown as a thinking indicator rather than leaked as regular output.
        """
        if not response:
            return ""
        lower = response.lower()
        for open_tag, _ in ResponseParser._THINK_OPEN_TAGS:
            idx = lower.find(open_tag)
            if idx >= 0:
                return response[idx + len(open_tag):].strip()
        return ""

    @staticmethod
    def strip_thinking_tag_leaks(text: str) -> str:
        """Remove leaked/partial thinking tags from model output.

        Some models (DeepSeek, Qwen, GLM) use <think>...</think> and occasionally
        leak partial tags like '/think>' into the visible response. This strips
        any remnant thinking tags that weren't caught by parse_thinking_block.
        """
        if not text:
            return text
        cleaned = ResponseParser._THINK_TAG_LEAK_RE.sub('', text)
        # Collapse any leading whitespace left behind
        return cleaned.lstrip('\n').strip() if cleaned != text else text

    @staticmethod
    def _detect_untagged_thinking(response: str) -> Tuple[str, str]:
        """Heuristic fallback: detect untagged thinking dumped before the real answer.

        Scans lines from the top for meta-reasoning patterns.  When enough
        distinct patterns are found in a prefix, everything from the first
        match through a double-newline gap is treated as thinking.

        Returns (thinking, answer) or ("", "") if nothing detected.
        """
        if not response or len(response) < 80:
            return "", ""

        lines = response.split('\n')
        if len(lines) < 3:
            return "", ""

        # Find spans of lines that match heuristic patterns
        hit_patterns: set = set()
        last_hit_line = -1
        first_hit_line = -1

        for i, line in enumerate(lines):
            for j, pat in enumerate(ResponseParser._THINKING_HEURISTIC_PATTERNS):
                if pat.search(line):
                    hit_patterns.add(j)
                    if first_hit_line < 0:
                        first_hit_line = i
                    last_hit_line = i
                    break  # one pattern per line is enough

        if len(hit_patterns) < ResponseParser._HEURISTIC_MIN_HITS:
            return "", ""

        # Extend the boundary to the next blank-line gap after the last hit,
        # which typically separates reasoning from the actual response.
        split_line = last_hit_line + 1
        for i in range(last_hit_line + 1, len(lines)):
            if not lines[i].strip():
                split_line = i + 1
                break
            split_line = i + 1

        # Require that the remaining answer is at least 20 chars
        # (guard against stripping the entire response)
        answer = '\n'.join(lines[split_line:]).strip()
        if len(answer) < 20:
            return "", ""

        thinking = '\n'.join(lines[first_hit_line:split_line]).strip()
        return thinking, answer

    # Pattern to detect <output>...</output> wrapper (used by some providers
    # to separate thinking from answer when thinking is returned inline)
    _OUTPUT_WRAPPER_RE = re.compile(
        r'<\s*output\s*>([\s\S]*?)<\s*/\s*output\s*>',
        re.IGNORECASE,
    )

    # Heuristic patterns indicating untagged thinking / internal reasoning.
    # Each pattern is a compiled regex.  If >=HEURISTIC_MIN_HITS of these
    # appear in a *prefix* of the response (before the real answer), we
    # treat everything up to (and including) the last match as thinking.
    _THINKING_HEURISTIC_PATTERNS = [
        # Meta-reasoning about what to do / how to respond
        re.compile(r"(?:^|\n)\s*(?:I should |I need to |Let me think|Let me (?:consider|check|look)|I'll )", re.IGNORECASE),
        # Third-person references to the user in internal reasoning
        re.compile(r"(?:^|\n)\s*(?:He'?s saying|She'?s saying|They'?re saying|The user is |The user (?:wants|asked|said|seems))", re.IGNORECASE),
        # Planning / strategy language
        re.compile(r"(?:^|\n)\s*(?:What would (?:actually )?be useful|How should I |I could mention|I (?:shouldn't|should not) )", re.IGNORECASE),
        # System prompt instruction echo (tail end of the thinking instruction)
        re.compile(r"Walk through the context step-by-step", re.IGNORECASE),
        # Explicit "not now" / conversational meta-analysis
        re.compile(r"(?:^|\n)\s*(?:This is (?:a casual|just a)|not asking me to|flagging that)", re.IGNORECASE),
        # Bullet-style reasoning about approach
        re.compile(r"(?:^|\n)\s*-\s+(?:Explicitly |Temporal |Source |Emotional |Deciding )", re.IGNORECASE),
    ]
    _HEURISTIC_MIN_HITS = 2  # need at least 2 distinct patterns to trigger

    @staticmethod
    def likely_untagged_thinking(response: str) -> bool:
        """Fast streaming check: does accumulated text look like untagged thinking?

        Unlike _detect_untagged_thinking(), this does NOT require a clean split
        point or ≥20 chars of answer.  It only checks whether the heuristic
        patterns appear often enough to indicate reasoning is being dumped.

        Used during streaming to suppress leaked thinking before the full
        split can be computed.
        """
        if not response or len(response) < 60:
            return False
        lines = response.split('\n')
        if len(lines) < 2:
            return False
        hit_patterns: set = set()
        for line in lines:
            for j, pat in enumerate(ResponseParser._THINKING_HEURISTIC_PATTERNS):
                if pat.search(line):
                    hit_patterns.add(j)
                    if len(hit_patterns) >= ResponseParser._HEURISTIC_MIN_HITS:
                        return True
                    break  # one pattern per line
        return False

    @staticmethod
    def parse_thinking_block(response: str) -> Tuple[str, str]:
        """
        Parse response to extract thinking block and final answer.

        Handles:
        - <thinking>...</thinking> (Anthropic/OpenAI style)
        - <think>...</think> (DeepSeek/Qwen/GLM style)
        - <output>...</output> wrapper (some OpenRouter providers wrap the
          answer in <output> when thinking is returned inline)

        Args:
            response: Full LLM response potentially containing thinking blocks

        Returns:
            Tuple of (thinking_part, final_answer_part)
            - If no thinking block found, thinking_part is empty and final_answer_part is the full response
        """
        if not response or not isinstance(response, str):
            return "", response or ""

        # Try both tag variants: <thinking> and <think>
        for close_tag, open_tag in [("</thinking>", "<thinking>"), ("</think>", "<think>")]:
            if close_tag in response:
                parts = response.split(close_tag, 1)
                if len(parts) == 2:
                    thinking_raw = parts[0]
                    final_answer = parts[1].strip()

                    # Extract thinking content (remove opening tag if present)
                    thinking_content = thinking_raw
                    if open_tag in thinking_raw:
                        thinking_content = thinking_raw.split(open_tag, 1)[1]

                    # Clean any remaining tag leaks from the final answer
                    final_answer = ResponseParser.strip_thinking_tag_leaks(final_answer)

                    return thinking_content.strip(), final_answer

        # Check for <output> wrapper — thinking before it, answer inside it
        m = ResponseParser._OUTPUT_WRAPPER_RE.search(response)
        if m:
            answer = m.group(1).strip()
            thinking = response[:m.start()].strip()
            if answer:
                return thinking, answer

        # No tags found — try heuristic detection of untagged thinking.
        # Models sometimes dump chain-of-thought without wrapping it in tags.
        thinking, answer = ResponseParser._detect_untagged_thinking(response)
        if thinking and answer:
            return thinking, ResponseParser.strip_thinking_tag_leaks(answer)

        # Nothing detected — strip any leaked partial tags
        cleaned = ResponseParser.strip_thinking_tag_leaks(response)
        return "", cleaned

    @staticmethod
    def strip_reflection_blocks(response: str) -> str:
        """
        Strip reflection blocks from response before storing/showing as conversation.

        Reflections are stored separately as reflection memories, so they shouldn't
        also be saved as part of the conversation response.

        Handles both formats:
        - <reflect>...</reflect>
        - [SYSTEM QUALITY REFLECTION]...

        Args:
            response: Full LLM response potentially containing reflection blocks

        Returns:
            Response with all reflection blocks removed
        """
        if not response or not isinstance(response, str):
            return response or ""

        # Remove <reflect>...</reflect> blocks
        cleaned = re.sub(r'<reflect>.*?</reflect>', '', response, flags=re.DOTALL)

        # Remove [SYSTEM QUALITY REFLECTION] and everything after it
        cleaned = re.sub(r'\[SYSTEM QUALITY REFLECTION\].*', '', cleaned, flags=re.DOTALL)

        # Remove standalone <reflection> tags (legacy format)
        cleaned = re.sub(r'<reflection>.*?</reflection>', '', cleaned, flags=re.DOTALL)

        # Clean up any extra whitespace left behind
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Collapse multiple blank lines
        return cleaned.strip()

    @staticmethod
    def strip_xml_wrappers(text: str) -> str:
        """Remove simple XML-like wrappers such as <result>...</result>, <answer>...</answer>.

        Keeps inner content; tolerant if tags are missing.
        """
        if not text:
            return text
        try:
            s = text.strip()
            # Unwrap common tags if they span the whole string
            for tag in ("result", "answer", "final", "output", "response"):
                pattern = rf"^\s*<\s*{tag}[^>]*>([\s\S]*?)<\s*/\s*{tag}\s*>\s*$"
                m = re.match(pattern, s, flags=re.IGNORECASE)
                if m:
                    s = m.group(1).strip()
            return s
        except Exception:
            return text

    @staticmethod
    def strip_prompt_artifacts(text: str) -> str:
        """Remove known bracketed prompt headers if the model echoes them.

        Conservative: removes header lines and their immediate block until a blank line.
        """
        if not text:
            return text
        try:
            header_patterns = [
                r"^\s*\[TIME CONTEXT\]",
                r"^\s*\[RECENT CONVERSATION[^\]]*\]",
                r"^\s*\[RELEVANT INFORMATION\]",
                r"^\s*\[RELEVANT MEMORIES\]",
                r"^\s*\[FACTS[ ^\]]*\]",
                r"^\s*\[RECENT FACTS\]",
                r"^\s*\[CURRENT MESSAGE FACTS\]",
                r"^\s*\[DIRECTIVES\]",
                r"^\s*\[CURRENT USER QUERY[ ^\]]*\]",
                r"^\s*\[USER INPUT\]",
                r"^\s*\[BACKGROUND KNOWLEDGE\]",
                r"^\s*\[CONVERSATION SUMMARIES[ ^\]]*\]",
                r"^\s*\[RECENT REFLECTIONS[ ^\]]*\]",
                r"^\s*\[SESSION REFLECTIONS[ ^\]]*\]",
            ]
            header_re = re.compile("(" + ")|(".join(header_patterns) + ")", re.IGNORECASE)
            lines = []
            skip_block = False
            for line in (text.splitlines() or []):
                if header_re.search(line):
                    skip_block = True
                    continue
                if skip_block:
                    if not line.strip():
                        skip_block = False
                    continue
                lines.append(line)
            return "\n".join(lines).strip()
        except Exception:
            return text
