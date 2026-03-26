"""
Response parsing and text cleaning utilities.

Extracted from DaemonOrchestrator to follow Single Responsibility Principle.
These are pure functions with no side effects - they take text and return cleaned text.

Module Contract:
- Purpose: Parse and clean LLM responses (thinking blocks, reflections, XML wrappers, artifacts)
- Inputs: Raw response strings from LLM
- Outputs: Cleaned response strings
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
    def parse_thinking_block(response: str) -> Tuple[str, str]:
        """
        Parse response to extract thinking block and final answer.

        Handles both <thinking>...</thinking> (Anthropic/OpenAI style) and
        <think>...</think> (DeepSeek/Qwen/GLM style).

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

        # No thinking block found — still strip any leaked partial tags
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
            for tag in ("result", "answer", "final"):
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
