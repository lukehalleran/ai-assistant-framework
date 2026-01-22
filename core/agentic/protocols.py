"""
Search Protocol Handlers

Contract:
    - Abstracts difference between native tool calling and XML markers
    - BaseProtocolHandler defines interface
    - NativeToolsHandler for OpenAI/Anthropic
    - XMLMarkerHandler for local models
    - detect_protocol() chooses based on model name

Public Interface:
    - detect_protocol(model_name: str) -> SearchProtocol
    - BaseProtocolHandler.parse_response() -> SearchDecision
    - NativeToolsHandler.parse_response() -> SearchDecision
    - XMLMarkerHandler.parse_response() -> SearchDecision

Dependencies:
    - core.agentic.types (SearchProtocol, SearchDecision)
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.agentic.types import SearchDecision, SearchProtocol

logger = logging.getLogger(__name__)

# Models known to support native tool calling
NATIVE_TOOL_MODELS = [
    # OpenAI
    "gpt-4", "gpt-4o", "gpt-4-turbo", "gpt-5",
    "openai/gpt-4", "openai/gpt-4o", "openai/gpt-4-turbo", "openai/gpt-5",
    # Anthropic
    "claude-3", "claude-opus", "claude-sonnet", "claude-haiku",
    "anthropic/claude-3", "anthropic/claude-opus", "anthropic/claude-sonnet",
    # DeepSeek (supports function calling)
    "deepseek-chat", "deepseek-coder",
]


def detect_protocol(model_name: str, api_models: Optional[Dict[str, str]] = None) -> SearchProtocol:
    """
    Determine which protocol to use based on model capabilities.

    Args:
        model_name: The model name or alias
        api_models: Optional mapping of aliases to full model names

    Returns:
        SearchProtocol indicating native tools or XML markers
    """
    # Resolve alias to full model name if mapping provided
    full_model = model_name
    if api_models and model_name in api_models:
        full_model = api_models[model_name]

    model_lower = full_model.lower()

    # Check if model supports native tool calling
    for prefix in NATIVE_TOOL_MODELS:
        if prefix.lower() in model_lower:
            logger.debug(f"[AgenticProtocol] Model {model_name} supports native tools")
            return SearchProtocol.NATIVE_TOOLS

    logger.debug(f"[AgenticProtocol] Model {model_name} will use XML markers")
    return SearchProtocol.XML_MARKERS


class BaseProtocolHandler(ABC):
    """Abstract base class for protocol handlers."""

    @abstractmethod
    def parse_response(self, response: Any) -> SearchDecision:
        """
        Parse LLM response to extract search decision.

        Args:
            response: Raw response from LLM (format depends on protocol)

        Returns:
            SearchDecision indicating what the model wants to do
        """
        pass

    @abstractmethod
    def get_tools(self) -> Optional[List[Dict]]:
        """
        Get tool definitions for this protocol.

        Returns:
            List of tool definitions for native protocol, None for XML
        """
        pass

    @abstractmethod
    def augment_system_prompt(self, system_prompt: str, max_rounds: int) -> str:
        """
        Augment system prompt with protocol-specific instructions.

        Args:
            system_prompt: Original system prompt
            max_rounds: Maximum number of search rounds allowed

        Returns:
            Augmented system prompt
        """
        pass


class NativeToolsHandler(BaseProtocolHandler):
    """
    Handler for native tool/function calling (OpenAI, Anthropic).

    Parses tool_calls from LLM response to detect search and Wolfram requests.
    """

    def __init__(self, wolfram_available: bool = False):
        from core.agentic.types import SEARCH_TOOL_DEFINITION, DONE_TOOL_DEFINITION, WOLFRAM_TOOL_DEFINITION
        self.search_tool = SEARCH_TOOL_DEFINITION
        self.done_tool = DONE_TOOL_DEFINITION
        self.wolfram_tool = WOLFRAM_TOOL_DEFINITION
        self.wolfram_available = wolfram_available

    def parse_response(self, response: Any) -> SearchDecision:
        """
        Parse native tool call response.

        Args:
            response: LLM response object with potential tool_calls

        Returns:
            SearchDecision based on tool calls or lack thereof
        """
        # Handle different response formats
        tool_calls = None

        # OpenAI format: response.tool_calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls = response.tool_calls
        # Dict format
        elif isinstance(response, dict) and 'tool_calls' in response:
            tool_calls = response['tool_calls']

        if not tool_calls:
            # No tool calls - model wants to answer directly
            content = self._extract_content(response)
            return SearchDecision(
                wants_answer=True,
                partial_response=content
            )

        # Parse first tool call
        tool_call = tool_calls[0]

        # Handle different tool call formats
        if hasattr(tool_call, 'function'):
            # OpenAI format
            func_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
        elif isinstance(tool_call, dict):
            # Dict format
            func = tool_call.get('function', {})
            func_name = func.get('name', '')
            try:
                args = json.loads(func.get('arguments', '{}'))
            except (json.JSONDecodeError, TypeError):
                args = {}
        else:
            logger.warning(f"[AgenticProtocol] Unknown tool call format: {type(tool_call)}")
            return SearchDecision(wants_answer=True)

        # Process tool call
        if func_name == "web_search":
            query = args.get("query", "")
            reason = args.get("reason")
            if query:
                logger.debug(f"[AgenticProtocol] Native tool search request: {query}")
                return SearchDecision(
                    wants_search=True,
                    search_query=query,
                    search_reason=reason
                )
            else:
                logger.warning("[AgenticProtocol] web_search called without query")
                return SearchDecision(wants_answer=True)

        elif func_name == "done_searching":
            reason = args.get("reason")
            logger.debug(f"[AgenticProtocol] Native tool done signal: {reason}")
            return SearchDecision(
                is_done=True,
                done_reason=reason
            )

        elif func_name == "wolfram_alpha":
            query = args.get("query", "")
            reason = args.get("reason")
            if query:
                logger.debug(f"[AgenticProtocol] Native tool Wolfram request: {query}")
                return SearchDecision(
                    wants_wolfram=True,
                    wolfram_query=query,
                    wolfram_reason=reason
                )
            else:
                logger.warning("[AgenticProtocol] wolfram_alpha called without query")
                return SearchDecision(wants_answer=True)

        else:
            logger.warning(f"[AgenticProtocol] Unknown tool called: {func_name}")
            return SearchDecision(wants_answer=True)

    def _extract_content(self, response: Any) -> Optional[str]:
        """Extract text content from response."""
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict):
            return response.get('content')
        elif isinstance(response, str):
            return response
        return None

    def get_tools(self) -> List[Dict]:
        """Return tool definitions for API calls."""
        tools = [self.search_tool, self.done_tool]
        if self.wolfram_available:
            tools.append(self.wolfram_tool)
        return tools

    def augment_system_prompt(self, system_prompt: str, max_rounds: int) -> str:
        """
        For native tools, minimal prompt augmentation needed.

        The tools themselves describe their purpose.
        """
        if self.wolfram_available:
            addition = (
                "\n\n[AGENTIC TOOLS MODE]\n"
                "You have access to web_search, wolfram_alpha, and done_searching tools. "
                f"Use web_search for current info, wolfram_alpha for math/science computations "
                f"(up to {max_rounds} tool uses total). "
                "Use done_searching when you have enough information to answer."
            )
        else:
            addition = (
                "\n\n[AGENTIC SEARCH MODE]\n"
                "You have access to web_search and done_searching tools. "
                f"Use web_search to find current information (up to {max_rounds} times). "
                "Use done_searching when you have enough information to answer."
            )
        return system_prompt + addition


class XMLMarkerHandler(BaseProtocolHandler):
    """
    Handler for XML marker-based search requests (local models).

    Parses <search>query</search>, <wolfram>query</wolfram>, and <done/> markers from text.
    """

    # Regex patterns for marker detection
    SEARCH_PATTERN = re.compile(r'<search>(.*?)</search>', re.DOTALL | re.IGNORECASE)
    WOLFRAM_PATTERN = re.compile(r'<wolfram>(.*?)</wolfram>', re.DOTALL | re.IGNORECASE)
    DONE_PATTERN = re.compile(r'<done\s*/?>', re.IGNORECASE)

    def parse_response(self, response: Any) -> SearchDecision:
        """
        Parse XML markers from text response.

        Args:
            response: Text response from LLM

        Returns:
            SearchDecision based on markers found
        """
        # Extract text content
        text = self._to_text(response)
        if not text:
            return SearchDecision(wants_answer=True)

        # Check for Wolfram marker first (more specific tool)
        wolfram_match = self.WOLFRAM_PATTERN.search(text)
        if wolfram_match:
            query = wolfram_match.group(1).strip()
            if query:
                logger.debug(f"[AgenticProtocol] XML wolfram marker found: {query}")
                return SearchDecision(
                    wants_wolfram=True,
                    wolfram_query=query
                )

        # Check for search marker
        search_match = self.SEARCH_PATTERN.search(text)
        if search_match:
            query = search_match.group(1).strip()
            if query:
                logger.debug(f"[AgenticProtocol] XML search marker found: {query}")
                return SearchDecision(
                    wants_search=True,
                    search_query=query
                )

        # Check for done marker
        if self.DONE_PATTERN.search(text):
            logger.debug("[AgenticProtocol] XML done marker found")
            return SearchDecision(is_done=True)

        # No markers - model wants to answer
        return SearchDecision(
            wants_answer=True,
            partial_response=text
        )

    def _to_text(self, response: Any) -> Optional[str]:
        """Convert response to text string."""
        if isinstance(response, str):
            return response
        elif hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict):
            return response.get('content', '')
        return None

    def get_tools(self) -> None:
        """XML protocol doesn't use tool definitions."""
        return None

    def augment_system_prompt(self, system_prompt: str, max_rounds: int) -> str:
        """
        Add XML marker instructions to system prompt.
        """
        from core.agentic.types import AGENTIC_SYSTEM_PROMPT_INJECTION

        injection = AGENTIC_SYSTEM_PROMPT_INJECTION.format(max_rounds=max_rounds)
        return system_prompt + "\n\n" + injection


def get_protocol_handler(
    protocol: SearchProtocol,
    wolfram_available: bool = False
) -> BaseProtocolHandler:
    """
    Factory function to get appropriate protocol handler.

    Args:
        protocol: The protocol to use
        wolfram_available: Whether Wolfram Alpha is configured

    Returns:
        Protocol handler instance
    """
    if protocol == SearchProtocol.NATIVE_TOOLS:
        return NativeToolsHandler(wolfram_available=wolfram_available)
    else:
        return XMLMarkerHandler()
