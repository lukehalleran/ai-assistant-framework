"""
Search Protocol Handlers

Contract:
    - Abstracts difference between native tool calling and XML markers
    - BaseProtocolHandler defines interface
    - NativeToolsHandler for OpenAI/Anthropic function calling
    - XMLMarkerHandler for local models using XML tags
    - detect_protocol() chooses based on model name
    - get_protocol_handler() factory with tool availability flags

Public Interface:
    - detect_protocol(model_name: str) -> SearchProtocol
    - get_protocol_handler(protocol, wolfram_available, sandbox_available) -> BaseProtocolHandler
    - BaseProtocolHandler.parse_response() -> SearchDecision
    - NativeToolsHandler.parse_response() -> SearchDecision (parses tool_calls)
    - XMLMarkerHandler.parse_response() -> SearchDecision (parses XML markers)

Supported Tools:
    - web_search / <search>: Web search queries
    - execute_wolfram / <wolfram>: Wolfram Alpha computations
    - execute_python / <python>: E2B sandbox code execution [NEW 2026-01-22]
    - search_memory / <memory>: ChromaDB memory/knowledge base search
    - expand_memory / <expand_memory>: Expand a memory hit to surrounding context [NEW 2026-03]
    - file_read / <file_read>: Read file contents from approved directories
    - file_grep / <file_grep>: Search for patterns across files
    - file_list / <file_list>: List directory contents
    - signal_done / <done>: Signal task completion

Dependencies:
    - core.agentic.types (SearchProtocol, SearchDecision, tool definitions)
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

    Parses tool_calls from LLM response to detect search, Wolfram, and sandbox requests.
    """

    def __init__(self, wolfram_available: bool = False, sandbox_available: bool = False, memory_available: bool = False, file_access_available: bool = False):
        from core.agentic.types import (
            SEARCH_TOOL_DEFINITION,
            DONE_TOOL_DEFINITION,
            WOLFRAM_TOOL_DEFINITION,
            SANDBOX_TOOL_DEFINITION,
            MEMORY_SEARCH_TOOL_DEFINITION,
            EXPAND_MEMORY_TOOL_DEFINITION,
            FILE_READ_TOOL_DEFINITION,
            FILE_GREP_TOOL_DEFINITION,
            FILE_LIST_TOOL_DEFINITION,
        )
        self.search_tool = SEARCH_TOOL_DEFINITION
        self.done_tool = DONE_TOOL_DEFINITION
        self.wolfram_tool = WOLFRAM_TOOL_DEFINITION
        self.sandbox_tool = SANDBOX_TOOL_DEFINITION
        self.memory_tool = MEMORY_SEARCH_TOOL_DEFINITION
        self.expand_memory_tool = EXPAND_MEMORY_TOOL_DEFINITION
        self.file_read_tool = FILE_READ_TOOL_DEFINITION
        self.file_grep_tool = FILE_GREP_TOOL_DEFINITION
        self.file_list_tool = FILE_LIST_TOOL_DEFINITION
        self.wolfram_available = wolfram_available
        self.sandbox_available = sandbox_available
        self.memory_available = memory_available
        self.file_access_available = file_access_available

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

        elif func_name == "execute_python":
            code = args.get("code", "")
            purpose = args.get("purpose")
            if code:
                logger.debug(f"[AgenticProtocol] Native tool sandbox request: {purpose or 'code execution'}")
                return SearchDecision(
                    wants_sandbox=True,
                    sandbox_code=code,
                    sandbox_purpose=purpose
                )
            else:
                logger.warning("[AgenticProtocol] execute_python called without code")
                return SearchDecision(wants_answer=True)

        elif func_name == "search_memory":
            query = args.get("query", "")
            collection = args.get("collection", "facts")
            reason = args.get("reason")
            if query:
                logger.debug(f"[AgenticProtocol] Native tool memory search: {collection}/{query}")
                return SearchDecision(
                    wants_memory_search=True,
                    memory_query=query,
                    memory_collection=collection,
                    memory_reason=reason
                )
            else:
                logger.warning("[AgenticProtocol] search_memory called without query")
                return SearchDecision(wants_answer=True)

        elif func_name == "expand_memory":
            memory_id = args.get("memory_id", "")
            reason = args.get("reason")
            if memory_id:
                logger.debug(f"[AgenticProtocol] Native tool expand_memory: {memory_id[:8]}")
                return SearchDecision(
                    wants_memory_expand=True,
                    expand_memory_id=memory_id,
                    expand_window=int(args.get("window", 3)),
                    expand_collection=args.get("collection"),
                    expand_reason=reason,
                )
            else:
                logger.warning("[AgenticProtocol] expand_memory called without memory_id")
                return SearchDecision(wants_answer=True)

        elif func_name == "file_read":
            filepath = args.get("filepath", "")
            reason = args.get("reason")
            if filepath:
                logger.debug(f"[AgenticProtocol] Native tool file read: {filepath}")
                return SearchDecision(
                    wants_file_read=True,
                    file_read_path=filepath,
                    file_read_start_line=args.get("start_line"),
                    file_read_end_line=args.get("end_line"),
                    file_read_reason=reason
                )
            else:
                logger.warning("[AgenticProtocol] file_read called without filepath")
                return SearchDecision(wants_answer=True)

        elif func_name == "file_grep":
            pattern = args.get("pattern", "")
            reason = args.get("reason")
            if pattern:
                logger.debug(f"[AgenticProtocol] Native tool file grep: {pattern}")
                return SearchDecision(
                    wants_file_grep=True,
                    file_grep_pattern=pattern,
                    file_grep_folder=args.get("folder"),
                    file_grep_glob=args.get("file_glob"),
                    file_grep_reason=reason
                )
            else:
                logger.warning("[AgenticProtocol] file_grep called without pattern")
                return SearchDecision(wants_answer=True)

        elif func_name == "file_list":
            dirpath = args.get("dirpath", "")
            reason = args.get("reason")
            if dirpath:
                logger.debug(f"[AgenticProtocol] Native tool file list: {dirpath}")
                return SearchDecision(
                    wants_file_list=True,
                    file_list_path=dirpath,
                    file_list_recursive=args.get("recursive", False),
                    file_list_reason=reason
                )
            else:
                logger.warning("[AgenticProtocol] file_list called without dirpath")
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
        if self.sandbox_available:
            tools.append(self.sandbox_tool)
        if self.memory_available:
            tools.append(self.memory_tool)
            tools.append(self.expand_memory_tool)
        if self.file_access_available:
            tools.extend([self.file_read_tool, self.file_grep_tool, self.file_list_tool])
        return tools

    def augment_system_prompt(self, system_prompt: str, max_rounds: int) -> str:
        """
        For native tools, minimal prompt augmentation needed.

        The tools themselves describe their purpose.
        """
        tool_list = ["web_search"]
        if self.wolfram_available:
            tool_list.append("wolfram_alpha")
        if self.sandbox_available:
            tool_list.append("execute_python")
        if self.memory_available:
            tool_list.append("search_memory")
            tool_list.append("expand_memory")
        if self.file_access_available:
            tool_list.extend(["file_read", "file_grep", "file_list"])
        tool_list.append("done_searching")

        tools_str = ", ".join(tool_list)
        memory_guidance = (
            " Use search_memory for internal/personal questions "
            "(your own docs, user facts, past conversations). "
            "For user profile/biographical questions, prefer summaries and conversations "
            "over facts — summaries contain rich narrative context while facts stores "
            "individual triples (name=X, age=33). Diversify across collections. "
            "Use web_search for external/current events. Use both when needed."
        ) if self.memory_available else ""
        addition = (
            f"\n\n[AGENTIC TOOLS MODE]\n"
            f"You have access to {tools_str} tools. "
            f"Use web_search for current info, wolfram_alpha for quick math/science computations, "
            f"execute_python for multi-step calculations and data analysis "
            f"(up to {max_rounds} tool uses total).{memory_guidance} "
            "Use done_searching when you have enough information to answer."
        )
        return system_prompt + addition


class XMLMarkerHandler(BaseProtocolHandler):
    """
    Handler for XML marker-based search requests (local models).

    Parses <search>query</search>, <wolfram>query</wolfram>, <python>code</python>,
    and <done/> markers from text.
    """

    # Regex patterns for marker detection
    SEARCH_PATTERN = re.compile(r'<search>(.*?)</search>', re.DOTALL | re.IGNORECASE)
    WOLFRAM_PATTERN = re.compile(r'<wolfram>(.*?)</wolfram>', re.DOTALL | re.IGNORECASE)
    DONE_PATTERN = re.compile(r'<done\s*/?>', re.IGNORECASE)
    # Python sandbox pattern with optional purpose attribute
    # Matches: <python>code</python> or <python purpose="description">code</python>
    PYTHON_PATTERN = re.compile(
        r'<python(?:\s+purpose=["\']([^"\']*)["\'])?\s*>(.*?)</python>',
        re.DOTALL | re.IGNORECASE
    )
    # Memory search pattern with optional collection attribute
    # Matches: <memory collection="facts">query</memory> or <memory>query</memory>
    MEMORY_PATTERN = re.compile(
        r'<memory(?:\s+collection=["\']([^"\']*)["\'])?\s*>(.*?)</memory>',
        re.DOTALL | re.IGNORECASE
    )
    # File read pattern: <file_read path="filepath">optional reason</file_read>
    FILE_READ_PATTERN = re.compile(
        r'<file_read\s+path=["\']([^"\']+)["\'](?:\s+start_line=["\'](\d+)["\'])?(?:\s+end_line=["\'](\d+)["\'])?\s*>(.*?)</file_read>',
        re.DOTALL | re.IGNORECASE
    )
    # File grep pattern: <file_grep pattern="pat" glob="*.py">optional folder</file_grep>
    FILE_GREP_PATTERN = re.compile(
        r'<file_grep\s+pattern=["\']([^"\']+)["\'](?:\s+glob=["\']([^"\']*)["\'])?\s*>(.*?)</file_grep>',
        re.DOTALL | re.IGNORECASE
    )
    # File list pattern: <file_list path="dirpath" recursive="false"/>
    FILE_LIST_PATTERN = re.compile(
        r'<file_list\s+path=["\']([^"\']+)["\'](?:\s+recursive=["\']([^"\']*)["\'])?\s*/?>',
        re.IGNORECASE
    )
    # Expand memory pattern: <expand_memory id="abc12345" collection="conversations" window="3">reason</expand_memory>
    EXPAND_MEMORY_PATTERN = re.compile(
        r'<expand_memory\s+id=["\']([^"\']+)["\']'
        r'(?:\s+collection=["\']([^"\']*)["\'])?'
        r'(?:\s+window=["\'](\d+)["\'])?'
        r'\s*>(.*?)</expand_memory>',
        re.DOTALL | re.IGNORECASE
    )

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

        # Check for Python sandbox marker first (most specific for multi-step computation)
        python_match = self.PYTHON_PATTERN.search(text)
        if python_match:
            purpose = python_match.group(1)  # May be None if no purpose attr
            code = python_match.group(2).strip()
            if code:
                logger.debug(f"[AgenticProtocol] XML python marker found: {purpose or 'code execution'}")
                return SearchDecision(
                    wants_sandbox=True,
                    sandbox_code=code,
                    sandbox_purpose=purpose
                )

        # Check for Wolfram marker (quick calculations)
        wolfram_match = self.WOLFRAM_PATTERN.search(text)
        if wolfram_match:
            query = wolfram_match.group(1).strip()
            if query:
                logger.debug(f"[AgenticProtocol] XML wolfram marker found: {query}")
                return SearchDecision(
                    wants_wolfram=True,
                    wolfram_query=query
                )

        # Check for memory search marker
        memory_match = self.MEMORY_PATTERN.search(text)
        if memory_match:
            collection = memory_match.group(1) or "facts"
            query = memory_match.group(2).strip()
            if query:
                logger.debug(f"[AgenticProtocol] XML memory marker found: {collection}/{query}")
                return SearchDecision(
                    wants_memory_search=True,
                    memory_query=query,
                    memory_collection=collection
                )

        # Check for expand_memory marker
        expand_match = self.EXPAND_MEMORY_PATTERN.search(text)
        if expand_match:
            memory_id = expand_match.group(1).strip()
            collection = expand_match.group(2) or None
            window = int(expand_match.group(3)) if expand_match.group(3) else 3
            if memory_id:
                logger.debug(f"[AgenticProtocol] XML expand_memory marker found: {memory_id[:8]}")
                return SearchDecision(
                    wants_memory_expand=True,
                    expand_memory_id=memory_id,
                    expand_window=window,
                    expand_collection=collection,
                )

        # Check for file read marker
        file_read_match = self.FILE_READ_PATTERN.search(text)
        if file_read_match:
            filepath = file_read_match.group(1).strip()
            start_line = int(file_read_match.group(2)) if file_read_match.group(2) else None
            end_line = int(file_read_match.group(3)) if file_read_match.group(3) else None
            if filepath:
                logger.debug(f"[AgenticProtocol] XML file_read marker found: {filepath}")
                return SearchDecision(
                    wants_file_read=True,
                    file_read_path=filepath,
                    file_read_start_line=start_line,
                    file_read_end_line=end_line,
                )

        # Check for file grep marker
        file_grep_match = self.FILE_GREP_PATTERN.search(text)
        if file_grep_match:
            pattern = file_grep_match.group(1).strip()
            glob = file_grep_match.group(2)
            folder = file_grep_match.group(3).strip() if file_grep_match.group(3) else None
            if pattern:
                logger.debug(f"[AgenticProtocol] XML file_grep marker found: {pattern}")
                return SearchDecision(
                    wants_file_grep=True,
                    file_grep_pattern=pattern,
                    file_grep_folder=folder if folder else None,
                    file_grep_glob=glob if glob else None,
                )

        # Check for file list marker
        file_list_match = self.FILE_LIST_PATTERN.search(text)
        if file_list_match:
            dirpath = file_list_match.group(1).strip()
            recursive = file_list_match.group(2) or "false"
            if dirpath:
                logger.debug(f"[AgenticProtocol] XML file_list marker found: {dirpath}")
                return SearchDecision(
                    wants_file_list=True,
                    file_list_path=dirpath,
                    file_list_recursive=recursive.lower() == "true",
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
    wolfram_available: bool = False,
    sandbox_available: bool = False,
    memory_available: bool = False,
    file_access_available: bool = False,
) -> BaseProtocolHandler:
    """
    Factory function to get appropriate protocol handler.

    Args:
        protocol: The protocol to use
        wolfram_available: Whether Wolfram Alpha is configured
        sandbox_available: Whether E2B sandbox is configured
        memory_available: Whether ChromaDB memory search is available
        file_access_available: Whether file access manager is configured

    Returns:
        Protocol handler instance
    """
    if protocol == SearchProtocol.NATIVE_TOOLS:
        return NativeToolsHandler(
            wolfram_available=wolfram_available,
            sandbox_available=sandbox_available,
            memory_available=memory_available,
            file_access_available=file_access_available,
        )
    else:
        return XMLMarkerHandler()
