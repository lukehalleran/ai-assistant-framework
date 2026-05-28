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
    - get_protocol_handler(protocol, wolfram_available, sandbox_available, ...) -> BaseProtocolHandler
    - BaseProtocolHandler.parse_response() -> List[SearchDecision]
    - NativeToolsHandler.parse_response() -> List[SearchDecision] (parses ALL tool_calls)
    - XMLMarkerHandler.parse_response() -> List[SearchDecision] (parses ALL XML markers)

Supported Tools:
    - web_search / <search>: Web search queries
    - execute_wolfram / <wolfram>: Wolfram Alpha computations
    - execute_python / <python>: E2B sandbox code execution [NEW 2026-01-22]
    - search_memory / <memory>: ChromaDB memory/knowledge base search
    - expand_memory / <expand_memory>: Expand a memory hit to surrounding context [NEW 2026-03]
    - file_read / <file_read>: Read file contents from approved directories
    - file_grep / <file_grep>: Search for patterns across files
    - file_list / <file_list>: List directory contents
    - git_stats / <git_stats>: Git repository activity stats
    - github / <github>: Read-only GitHub API (issues, PRs, actions, releases, search)
    - get_full_document / <get_full_document>: Retrieve complete uploaded document by title
    - fetch_url / <fetch_url>: Fetch web page content by URL
    - propose_action / <propose_action>: Propose internet write actions (email, telegram, discord, calendar)
    - lookup_contact / <lookup_contact>: Google Contacts + Gmail header lookup [NEW 2026-05-28]
      Tool name aliases: search_contacts, find_contact, search_gmail, search_email, gmail_search, search_inbox
      Also parsed from <invoke name="..."> XML fallback
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
    def parse_response(self, response: Any) -> List[SearchDecision]:
        """
        Parse LLM response to extract search decisions.

        Returns a list of SearchDecision objects. When the LLM requests
        multiple independent tools in one step, each gets its own entry.
        Single-tool responses return a 1-element list (backward compatible).

        Args:
            response: Raw response from LLM (format depends on protocol)

        Returns:
            List of SearchDecision(s) — one per tool call requested
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

    def __init__(self, wolfram_available: bool = False, sandbox_available: bool = False, memory_available: bool = False, file_access_available: bool = False, git_stats_available: bool = False, github_available: bool = False, fetch_url_available: bool = False, actions_available: bool = False):
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
            GIT_STATS_TOOL_DEFINITION,
            GITHUB_TOOL_DEFINITION,
            GET_FULL_DOCUMENT_TOOL_DEFINITION,
            RECALL_IMAGE_TOOL_DEFINITION,
            FETCH_URL_TOOL_DEFINITION,
            STACKEXCHANGE_TOOL_DEFINITION,
            ARXIV_TOOL_DEFINITION,
            PUBMED_TOOL_DEFINITION,
            HACKERNEWS_TOOL_DEFINITION,
            GENERATE_DOCUMENT_TOOL_DEFINITION,
            CREATE_DAEMON_NOTE_TOOL_DEFINITION,
            PROPOSE_ACTION_TOOL_DEFINITION,
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
        self.git_stats_tool = GIT_STATS_TOOL_DEFINITION
        self.github_tool = GITHUB_TOOL_DEFINITION
        self.full_document_tool = GET_FULL_DOCUMENT_TOOL_DEFINITION
        self.recall_image_tool = RECALL_IMAGE_TOOL_DEFINITION
        self.fetch_url_tool = FETCH_URL_TOOL_DEFINITION
        self.stackexchange_tool = STACKEXCHANGE_TOOL_DEFINITION
        self.arxiv_tool = ARXIV_TOOL_DEFINITION
        self.pubmed_tool = PUBMED_TOOL_DEFINITION
        self.hackernews_tool = HACKERNEWS_TOOL_DEFINITION
        self.generate_document_tool = GENERATE_DOCUMENT_TOOL_DEFINITION
        self.create_daemon_note_tool = CREATE_DAEMON_NOTE_TOOL_DEFINITION
        self.propose_action_tool = PROPOSE_ACTION_TOOL_DEFINITION
        self.wolfram_available = wolfram_available
        self.sandbox_available = sandbox_available
        self.memory_available = memory_available
        self.file_access_available = file_access_available
        self.git_stats_available = git_stats_available
        self.github_available = github_available
        self.fetch_url_available = fetch_url_available
        self.actions_available = actions_available

    def parse_response(self, response: Any) -> List[SearchDecision]:
        """
        Parse native tool call response.

        Processes ALL tool_calls from the LLM response, returning one
        SearchDecision per tool call. Single-tool responses return a
        1-element list (backward compatible).

        Args:
            response: LLM response object with potential tool_calls

        Returns:
            List of SearchDecision(s) based on tool calls
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
            # No tool calls from API — but model may have emitted the call
            # as text (common with OpenRouter proxied models). Try to parse
            # propose_action from the text content before giving up.
            content = self._extract_content(response)
            if content:
                text_decisions = self._parse_text_tool_calls(content)
                if text_decisions:
                    logger.info(
                        f"[AgenticProtocol] Parsed {len(text_decisions)} tool call(s) "
                        f"from text content (API returned no tool_calls)"
                    )
                    return text_decisions
            # Model wants to answer directly
            return [SearchDecision(
                wants_answer=True,
                partial_response=content
            )]

        # Parse ALL tool calls
        decisions = []
        for tool_call in tool_calls:
            decision = self._parse_single_tool_call(tool_call)
            if decision is not None:
                decisions.append(decision)

        if not decisions:
            return [SearchDecision(wants_answer=True)]

        return decisions

    def _parse_single_tool_call(self, tool_call: Any) -> Optional[SearchDecision]:
        """
        Parse a single tool call into a SearchDecision.

        Args:
            tool_call: A single tool call object (OpenAI or dict format)

        Returns:
            SearchDecision or None if the tool call is malformed
        """
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
            return None

        # Process tool call
        if func_name == "web_search":
            query = args.get("query", "")
            reason = args.get("reason")
            site = args.get("site")
            if query:
                logger.debug(f"[AgenticProtocol] Native tool search request: {query}" + (f" site={site}" if site else ""))
                return SearchDecision(
                    wants_search=True,
                    search_query=query,
                    search_site=site,
                    search_reason=reason
                )
            else:
                logger.warning("[AgenticProtocol] web_search called without query")
                return None

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
                return None

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
                return None

        elif func_name == "search_memory":
            query = args.get("query", "")
            collection = args.get("collection", "facts")
            reason = args.get("reason")
            if not query:
                # Fallback: scan args for any string value
                for k, v in args.items():
                    if k not in ("collection", "reason") and isinstance(v, str) and v.strip():
                        query = v.strip()
                        break
            if not query:
                logger.warning(f"[AgenticProtocol] search_memory called with empty args={args}, skipping")
                return None
            logger.debug(f"[AgenticProtocol] Native tool memory search: {collection}/{query}")
            return SearchDecision(
                wants_memory_search=True,
                memory_query=query,
                memory_collection=collection,
                memory_reason=reason
            )

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
                return None

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
                return None

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
                return None

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
                return None

        elif func_name == "git_stats":
            query = args.get("query", "") or args.get("input", "") or args.get("q", "")
            reason = args.get("reason")
            if not query:
                # Some models pass the query as the only arg with a non-standard key
                for k, v in args.items():
                    if k not in ("reason",) and isinstance(v, str) and v.strip():
                        query = v.strip()
                        break
            if not query:
                # Model called git_stats with empty args — use a sensible default
                query = "recent commits"
                logger.warning(f"[AgenticProtocol] git_stats called with empty args, defaulting to '{query}'")
            logger.debug(f"[AgenticProtocol] Native tool git_stats: {query}")
            return SearchDecision(
                wants_git_stats=True,
                git_stats_query=query,
                git_stats_reason=reason,
            )

        elif func_name == "get_full_document":
            title = args.get("title", "")
            reason = args.get("reason")
            if title:
                logger.debug(f"[AgenticProtocol] Native tool get_full_document: {title}")
                return SearchDecision(
                    wants_full_document=True,
                    full_document_title=title,
                    full_document_reason=reason,
                )
            else:
                logger.warning("[AgenticProtocol] get_full_document called without title")
                return None

        elif func_name == "recall_image":
            query = args.get("query", "")
            reason = args.get("reason")
            if query:
                logger.debug(f"[AgenticProtocol] Native tool recall_image: {query}")
                return SearchDecision(
                    wants_recall_image=True,
                    recall_image_query=query,
                    recall_image_reason=reason,
                )
            else:
                logger.warning("[AgenticProtocol] recall_image called without query")
                return None

        elif func_name == "fetch_url":
            url = args.get("url", "")
            reason = args.get("reason")
            if url:
                logger.debug(f"[AgenticProtocol] Native tool fetch_url: {url}")
                return SearchDecision(
                    wants_fetch_url=True,
                    fetch_url=url,
                    fetch_url_reason=reason,
                )
            else:
                logger.warning("[AgenticProtocol] fetch_url called without url")
                return None

        elif func_name == "github":
            query = args.get("query", "") or args.get("input", "") or args.get("q", "")
            reason = args.get("reason")
            if not query:
                for k, v in args.items():
                    if k not in ("reason",) and isinstance(v, str) and v.strip():
                        query = v.strip()
                        break
            if not query:
                query = "repo info"
                logger.warning(f"[AgenticProtocol] github called with empty args, defaulting to '{query}'")
            logger.debug(f"[AgenticProtocol] Native tool github: {query}")
            return SearchDecision(
                wants_github=True,
                github_query=query,
                github_reason=reason,
            )

        elif func_name == "search_stackexchange":
            query = args.get("query", "")
            site = args.get("site", "stackoverflow")
            reason = args.get("reason")
            if query:
                return SearchDecision(
                    wants_stackexchange=True,
                    stackexchange_query=query,
                    stackexchange_site=site,
                    stackexchange_reason=reason,
                )
            return None

        elif func_name == "search_arxiv":
            query = args.get("query", "")
            reason = args.get("reason")
            if query:
                return SearchDecision(
                    wants_arxiv=True,
                    arxiv_query=query,
                    arxiv_reason=reason,
                )
            return None

        elif func_name == "search_pubmed":
            query = args.get("query", "")
            reason = args.get("reason")
            if query:
                return SearchDecision(
                    wants_pubmed=True,
                    pubmed_query=query,
                    pubmed_reason=reason,
                )
            return None

        elif func_name == "search_hackernews":
            query = args.get("query", "")
            reason = args.get("reason")
            if query:
                return SearchDecision(
                    wants_hackernews=True,
                    hackernews_query=query,
                    hackernews_reason=reason,
                )
            return None

        elif func_name == "generate_document":
            topic = args.get("topic", "")
            doc_type = args.get("doc_type", "report")
            focus = args.get("focus")
            reason = args.get("reason")
            if topic:
                logger.debug(f"[AgenticProtocol] Native tool generate_document: {topic} ({doc_type})")
                return SearchDecision(
                    wants_generate_document=True,
                    generate_document_topic=topic,
                    generate_document_type=doc_type,
                    generate_document_focus=focus,
                    generate_document_reason=reason,
                )
            else:
                logger.warning("[AgenticProtocol] generate_document called without topic")
                return None

        elif func_name == "create_daemon_note":
            title = args.get("title", "")
            category = args.get("category", "implementation")
            summary = args.get("summary", "")
            reason = args.get("reason")
            if title and summary:
                logger.debug(f"[AgenticProtocol] Native tool create_daemon_note: {title} ({category})")
                return SearchDecision(
                    wants_create_daemon_note=True,
                    daemon_note_title=title,
                    daemon_note_category=category,
                    daemon_note_summary=summary,
                    daemon_note_reason=reason,
                )
            else:
                logger.warning(f"[AgenticProtocol] create_daemon_note called without title/summary")
                return None

        elif func_name == "propose_action":
            action_type = args.get("action_type", "")
            message = args.get("message", "")
            reason = args.get("reason", "")
            recipient = args.get("recipient", "")
            subject = args.get("subject", "")

            # Calendar events use summary/start_time/end_time instead of message
            is_calendar = action_type == "calendar_create_event"
            cal_summary = args.get("summary", "")

            if action_type and (message or (is_calendar and cal_summary)):
                params = {}
                if message:
                    params["message"] = message
                if recipient:
                    params["recipient"] = recipient
                if subject:
                    params["subject"] = subject

                # Forward calendar-specific params
                if is_calendar:
                    for key in ("summary", "description", "start_time", "end_time",
                                "time_zone", "calendar_id", "location"):
                        val = args.get(key)
                        if val:
                            params[key] = val

                if is_calendar and cal_summary:
                    display = f"calendar_create_event: {cal_summary}"
                    summary_text = display
                elif recipient:
                    summary_text = f"{action_type} to {recipient}: {message[:60]}"
                else:
                    summary_text = f"{action_type}: {message[:80]}"
                logger.info(f"[AgenticProtocol] Native tool propose_action: {summary_text}")
                return SearchDecision(
                    wants_action=True,
                    action_type=action_type,
                    action_params=params,
                    action_summary=summary_text,
                    action_reason=reason,
                )
            else:
                logger.warning("[AgenticProtocol] propose_action called without action_type/message")
                return None

        elif func_name in ("lookup_contact", "search_contacts", "find_contact",
                           "search_gmail", "search_email", "gmail_search", "search_inbox"):
            name = args.get("name", "") or args.get("query", "")
            reason = args.get("reason")
            if name:
                logger.debug(f"[AgenticProtocol] Native tool lookup_contact (via {func_name}): {name}")
                return SearchDecision(
                    wants_lookup_contact=True,
                    lookup_contact_name=name,
                    lookup_contact_reason=reason,
                )
            else:
                logger.warning(f"[AgenticProtocol] {func_name} called without name/query")
                return None

        else:
            logger.warning(f"[AgenticProtocol] Unknown tool called: {func_name}")
            return None

    def _parse_text_tool_calls(self, content: str) -> List[SearchDecision]:
        """Parse tool calls embedded as text when the API didn't return tool_calls.

        Handles patterns like:
            [propose_action: send_email]
            {"to": "...", "message": "...", ...}

        Also handles XML-style <action> tags as fallback.
        """
        decisions = []

        # Pattern 1: [propose_action: <type>] followed by JSON
        action_match = re.search(
            r'\[propose_action:\s*(\w+)\]\s*\{',
            content,
        )
        if action_match:
            action_type = action_match.group(1)
            json_start = content.index('{', action_match.start())
            # Find matching closing brace
            depth = 0
            json_end = json_start
            for i, ch in enumerate(content[json_start:], json_start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        json_end = i + 1
                        break
            try:
                params = json.loads(content[json_start:json_end])
            except (json.JSONDecodeError, ValueError):
                params = {}
            if params:
                message = params.get("body") or params.get("message", "")
                recipient = params.get("to") or params.get("recipient", "")
                subject = params.get("subject", "")
                reason = params.get("reason", "User requested")
                is_calendar = action_type == "calendar_create_event"
                cal_summary = params.get("summary", "")

                if message or (is_calendar and cal_summary):
                    action_params = {}
                    if message:
                        action_params["message"] = message
                    if recipient:
                        action_params["recipient"] = recipient
                    if subject:
                        action_params["subject"] = subject
                    # Forward calendar-specific params
                    if is_calendar:
                        for key in ("summary", "description", "start_time", "end_time",
                                    "time_zone", "calendar_id", "location"):
                            val = params.get(key)
                            if val:
                                action_params[key] = val

                    if is_calendar and cal_summary:
                        summary = f"calendar_create_event: {cal_summary}"
                    elif recipient:
                        summary = f"{action_type} to {recipient}: {message[:60]}"
                    else:
                        summary = f"{action_type}: {message[:80]}"

                    # Normalize action type: only add send_ prefix for messaging types
                    if not action_type.startswith("send_") and action_type in (
                        "telegram", "discord", "email",
                    ):
                        action_type = f"send_{action_type}"

                    logger.info(f"[AgenticProtocol] Parsed text propose_action: {summary}")
                    decisions.append(SearchDecision(
                        wants_action=True,
                        action_type=action_type,
                        action_params=action_params,
                        action_summary=summary,
                        action_reason=reason,
                    ))

        # Pattern 2: XML-style <invoke name="..."> (Anthropic XML function calling)
        if not decisions:
            for invoke_match in re.finditer(
                r'<invoke\s+name="(\w+)"[^>]*>(.*?)</invoke>',
                content,
                re.DOTALL,
            ):
                func_name = invoke_match.group(1)
                body = invoke_match.group(2)
                # Extract <parameter name="key">value</parameter> pairs
                args = {}
                for param_match in re.finditer(
                    r'<parameter\s+name="(\w+)"[^>]*>(.*?)</parameter>',
                    body,
                    re.DOTALL,
                ):
                    args[param_match.group(1)] = param_match.group(2).strip()

                parsed = self._parse_single_tool_call({
                    "function": {"name": func_name, "arguments": json.dumps(args)}
                })
                if parsed is not None:
                    logger.info(f"[AgenticProtocol] Parsed XML invoke '{func_name}' from text")
                    decisions.append(parsed)

        # Pattern 3: XML-style <action type="..."> (legacy XML tag format)
        if not decisions:
            action_xml = re.search(
                r'<action\s+type="(\w+)"[^>]*>(.*?)</action>',
                content,
                re.DOTALL,
            )
            if action_xml:
                action_type = action_xml.group(1)
                message = action_xml.group(2).strip()
                # Extract attributes
                recipient = ""
                subject = ""
                reason = ""
                for attr_match in re.finditer(r'(\w+)="([^"]*)"', action_xml.group(0)):
                    k, v = attr_match.group(1), attr_match.group(2)
                    if k == "recipient":
                        recipient = v
                    elif k == "subject":
                        subject = v
                    elif k == "reason":
                        reason = v
                if message:
                    action_params = {"message": message}
                    if recipient:
                        action_params["recipient"] = recipient
                    if subject:
                        action_params["subject"] = subject
                    summary = f"{action_type} to {recipient}: {message[:60]}" if recipient else f"{action_type}: {message[:80]}"
                    logger.info(f"[AgenticProtocol] Parsed XML action from text: {summary}")
                    decisions.append(SearchDecision(
                        wants_action=True,
                        action_type=action_type,
                        action_params=action_params,
                        action_summary=summary,
                        action_reason=reason or "User requested",
                    ))

        return decisions

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
            tools.append(self.full_document_tool)
        if self.file_access_available:
            tools.extend([self.file_read_tool, self.file_grep_tool, self.file_list_tool])
        if self.git_stats_available:
            tools.append(self.git_stats_tool)
        if self.github_available:
            tools.append(self.github_tool)
        if self.fetch_url_available:
            tools.append(self.fetch_url_tool)
        # Always available — free public APIs, no auth
        tools.extend([
            self.stackexchange_tool,
            self.arxiv_tool,
            self.pubmed_tool,
            self.hackernews_tool,
        ])
        # Document generation + self-notes — always available
        tools.append(self.generate_document_tool)
        tools.append(self.create_daemon_note_tool)
        # Internet actions — only when enabled
        if self.actions_available:
            tools.append(self.propose_action_tool)
        # NOTE: recall_image tool deliberately excluded from iteration tools.
        # Visual memories are already retrieved by the builder's parallel pipeline
        # and included in the initial context. Adding recall_image here causes
        # redundant agentic rounds that burn API credits. The tool dispatch and
        # definition remain wired for future use when explicit image search is
        # needed beyond what the builder retrieves.
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
            tool_list.append("get_full_document")
        if self.file_access_available:
            tool_list.extend(["file_read", "file_grep", "file_list"])
        if self.git_stats_available:
            tool_list.append("git_stats")
        if self.github_available:
            tool_list.append("github")
        if self.fetch_url_available:
            tool_list.append("fetch_url")
        tool_list.extend(["search_stackexchange", "search_arxiv", "search_pubmed", "search_hackernews"])
        tool_list.append("done_searching")

        tools_str = ", ".join(tool_list)
        memory_guidance = (
            " Use search_memory for internal/personal questions "
            "(your own docs, user facts, past conversations) and wiki_knowledge "
            "for encyclopedic/factual questions (history, science, geography, etc.). "
            "For user profile/biographical questions, prefer summaries and conversations "
            "over facts — summaries contain rich narrative context while facts stores "
            "individual triples (name=X, age=33). Diversify across collections. "
            "Use web_search for external/current events. Use both when needed."
        ) if self.memory_available else ""
        fetch_url_guidance = (
            " IMPORTANT: When you know a specific URL (from profile facts, conversation, "
            "or the user's message), use fetch_url to read it directly — do NOT web_search for it. "
            "web_search finds pages; fetch_url reads them. "
            "Example: if you know the user's GitHub is https://github.com/user/repo, "
            "call fetch_url(url='https://github.com/user/repo'), don't search for 'user GitHub repo'."
        ) if self.fetch_url_available else ""
        github_guidance = (
            " Use the github tool for questions about this repository's issues, PRs, "
            "CI/CD status, releases, workflows, labels, milestones, and contributors. "
            "It queries the GitHub API directly — much faster and more structured than "
            "web_search for repo-specific data. Use natural language: "
            "'open issues labeled bug', 'PR #42', 'failed CI runs', 'latest release', "
            "'search code for TODO', 'contributors'."
        ) if self.github_available else ""
        specialized_search_guidance = (
            " For technical/programming questions, prefer search_stackexchange over web_search — "
            "it returns structured Q&A with vote scores and accepted answers. "
            "For academic/research questions, use search_arxiv (CS, ML, physics, math) "
            "or search_pubmed (biomedical, health, clinical). "
            "For tech industry news/opinions, use search_hackernews. "
            "web_search with site parameter (e.g. site='reddit.com') is best for "
            "personal experiences and community opinions."
        )
        addition = (
            f"\n\n[AGENTIC TOOLS MODE]\n"
            f"You have access to {tools_str} tools. "
            f"Use web_search for current info, wolfram_alpha for quick math/science computations, "
            f"execute_python for multi-step calculations and data analysis "
            f"(up to {max_rounds} tool uses total). "
            f"IMPORTANT: The Python sandbox is PERSISTENT across messages — variables, "
            f"dataframes, and imports from previous execute_python calls are still alive. "
            f"Reuse them instead of recreating from scratch."
            f"{memory_guidance}{fetch_url_guidance}{github_guidance} "
            f"{specialized_search_guidance} "
            "You may call multiple tools in a single step when the queries are independent "
            "(e.g., web_search AND search_stackexchange simultaneously). "
            "Use done_searching when you have enough information to answer."
        )
        return system_prompt + addition


def _strip_xml_tags(text: str) -> str:
    """Strip XML tags from text, returning just the text content.

    Handles cases where models wrap content in nested XML tags like
    <github><action>list_repos</action></github> instead of <github>list repos</github>.
    """
    return re.sub(r'<[^>]+>', ' ', text).strip()


def _extract_nested_tag(text: str, tag: str) -> Optional[str]:
    """Extract the text content of a nested <tag>...</tag> from XML body."""
    m = re.search(rf'<{tag}\s*>(.*?)</{tag}>', text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


class XMLMarkerHandler(BaseProtocolHandler):
    """
    Handler for XML marker-based search requests (local models).

    Parses <search>query</search>, <wolfram>query</wolfram>, <python>code</python>,
    and <done/> markers from text.
    """

    # Regex patterns for marker detection
    # Primary: <search>query</search>
    # Aliases: <web_search>query</web_search>, <web_search query="...">..., <search query="...">...
    SEARCH_PATTERN = re.compile(r'<(?:search|web_search)>(.*?)</(?:search|web_search)>', re.DOTALL | re.IGNORECASE)
    SEARCH_ATTR_PATTERN = re.compile(r'<(?:search|web_search)\s+query=["\']([^"\']+)["\']\s*/?>', re.DOTALL | re.IGNORECASE)
    # Primary: <memory>query</memory>
    # Alias: <search_memory>query</search_memory>, <search_memory query="...">...
    MEMORY_ATTR_PATTERN = re.compile(r'<search_memory\s+query=["\']([^"\']+)["\']\s*/?>', re.DOTALL | re.IGNORECASE)
    # Nested-tag fallback: <search_memory><query>...</query><collection>...</collection></search_memory>
    MEMORY_NESTED_PATTERN = re.compile(
        r'<search_memory\s*>(.*?)</search_memory>',
        re.DOTALL | re.IGNORECASE
    )
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
    # Git stats pattern: <git_stats>query</git_stats>
    GIT_STATS_PATTERN = re.compile(
        r'<git_stats>(.*?)</git_stats>',
        re.DOTALL | re.IGNORECASE
    )
    # GitHub API pattern: <github>query</github>
    GITHUB_PATTERN = re.compile(
        r'<github>(.*?)</github>',
        re.DOTALL | re.IGNORECASE
    )
    # Fetch URL pattern: <fetch_url url="https://example.com">reason</fetch_url>
    FETCH_URL_PATTERN = re.compile(
        r'<fetch_url\s+url=["\']([^"\']+)["\']\s*>(.*?)</fetch_url>',
        re.DOTALL | re.IGNORECASE
    )
    # Get full document pattern: <get_full_document title="doc title">reason</get_full_document>
    GET_FULL_DOCUMENT_PATTERN = re.compile(
        r'<get_full_document\s+title=["\']([^"\']+)["\']\s*>(.*?)</get_full_document>',
        re.DOTALL | re.IGNORECASE
    )
    # Expand memory pattern: <expand_memory id="abc12345" collection="conversations" window="3">reason</expand_memory>
    EXPAND_MEMORY_PATTERN = re.compile(
        r'<expand_memory\s+id=["\']([^"\']+)["\']'
        r'(?:\s+collection=["\']([^"\']*)["\'])?'
        r'(?:\s+window=["\'](\d+)["\'])?'
        r'\s*>(.*?)</expand_memory>',
        re.DOTALL | re.IGNORECASE
    )
    # Internet action pattern: <action type="send_telegram" recipient="@luke">message</action>
    ACTION_PATTERN = re.compile(
        r'<action\s+type=["\']([^"\']+)["\']'
        r'(?:\s+recipient=["\']([^"\']*)["\'])?'
        r'(?:\s+subject=["\']([^"\']*)["\'])?'
        r'(?:\s+reason=["\']([^"\']*)["\'])?'
        r'\s*>(.*?)</action>',
        re.DOTALL | re.IGNORECASE
    )
    # Contact lookup pattern: <lookup_contact name="Meaghan">reason</lookup_contact>
    LOOKUP_CONTACT_PATTERN = re.compile(
        r'<lookup_contact\s+name=["\']([^"\']+)["\']\s*>(.*?)</lookup_contact>',
        re.DOTALL | re.IGNORECASE
    )
    # Propose action pattern: <propose_action type="send_email" recipient="..." subject="..." reason="...">body</propose_action>
    PROPOSE_ACTION_PATTERN = re.compile(
        r'<propose_action\s+type=["\']([^"\']+)["\']'
        r'(?:\s+recipient=["\']([^"\']*)["\'])?'
        r'(?:\s+subject=["\']([^"\']*)["\'])?'
        r'(?:\s+reason=["\']([^"\']*)["\'])?'
        r'\s*>(.*?)</propose_action>',
        re.DOTALL | re.IGNORECASE
    )
    # Fallback: Anthropic-style <invoke name="tool"><parameter name="key">val</parameter></invoke>
    INVOKE_PATTERN = re.compile(
        r'<invoke\s+name=["\'](\w+)["\'][^>]*>(.*?)</invoke>',
        re.DOTALL | re.IGNORECASE
    )
    INVOKE_PARAM_PATTERN = re.compile(
        r'<parameter\s+name=["\'](\w+)["\'][^>]*>(.*?)</parameter>',
        re.DOTALL | re.IGNORECASE
    )

    def parse_response(self, response: Any) -> List[SearchDecision]:
        """
        Parse XML markers from text response.

        Collects ALL markers found in the response. If a <done/> marker
        is present, returns only the done signal (tools are ignored).
        Single-marker responses return a 1-element list.

        Args:
            response: Text response from LLM

        Returns:
            List of SearchDecision(s) based on markers found
        """
        # Extract text content
        text = self._to_text(response)
        if not text:
            return [SearchDecision(wants_answer=True)]

        # Check for done marker first — if present, honor it immediately
        if self.DONE_PATTERN.search(text):
            logger.debug("[AgenticProtocol] XML done marker found")
            return [SearchDecision(is_done=True)]

        decisions = []

        # Check for Python sandbox markers
        for python_match in self.PYTHON_PATTERN.finditer(text):
            purpose = python_match.group(1)  # May be None if no purpose attr
            code = python_match.group(2).strip()
            if code:
                logger.debug(f"[AgenticProtocol] XML python marker found: {purpose or 'code execution'}")
                decisions.append(SearchDecision(
                    wants_sandbox=True,
                    sandbox_code=code,
                    sandbox_purpose=purpose
                ))

        # Check for Wolfram markers
        for wolfram_match in self.WOLFRAM_PATTERN.finditer(text):
            query = wolfram_match.group(1).strip()
            if query:
                logger.debug(f"[AgenticProtocol] XML wolfram marker found: {query}")
                decisions.append(SearchDecision(
                    wants_wolfram=True,
                    wolfram_query=query
                ))

        # Check for memory search markers
        for memory_match in self.MEMORY_PATTERN.finditer(text):
            collection = memory_match.group(1) or "facts"
            query = memory_match.group(2).strip()
            if query:
                logger.debug(f"[AgenticProtocol] XML memory marker found: {collection}/{query}")
                decisions.append(SearchDecision(
                    wants_memory_search=True,
                    memory_query=query,
                    memory_collection=collection
                ))

        # Check for expand_memory markers
        for expand_match in self.EXPAND_MEMORY_PATTERN.finditer(text):
            memory_id = expand_match.group(1).strip()
            collection = expand_match.group(2) or None
            window = int(expand_match.group(3)) if expand_match.group(3) else 3
            if memory_id:
                logger.debug(f"[AgenticProtocol] XML expand_memory marker found: {memory_id[:8]}")
                decisions.append(SearchDecision(
                    wants_memory_expand=True,
                    expand_memory_id=memory_id,
                    expand_window=window,
                    expand_collection=collection,
                ))

        # Check for get_full_document markers
        for full_doc_match in self.GET_FULL_DOCUMENT_PATTERN.finditer(text):
            title = full_doc_match.group(1).strip()
            reason = full_doc_match.group(2).strip() or None
            if title:
                logger.debug(f"[AgenticProtocol] XML get_full_document marker found: {title}")
                decisions.append(SearchDecision(
                    wants_full_document=True,
                    full_document_title=title,
                    full_document_reason=reason,
                ))

        # Check for file read markers
        for file_read_match in self.FILE_READ_PATTERN.finditer(text):
            filepath = file_read_match.group(1).strip()
            start_line = int(file_read_match.group(2)) if file_read_match.group(2) else None
            end_line = int(file_read_match.group(3)) if file_read_match.group(3) else None
            if filepath:
                logger.debug(f"[AgenticProtocol] XML file_read marker found: {filepath}")
                decisions.append(SearchDecision(
                    wants_file_read=True,
                    file_read_path=filepath,
                    file_read_start_line=start_line,
                    file_read_end_line=end_line,
                ))

        # Check for file grep markers
        for file_grep_match in self.FILE_GREP_PATTERN.finditer(text):
            pattern = file_grep_match.group(1).strip()
            glob = file_grep_match.group(2)
            folder = file_grep_match.group(3).strip() if file_grep_match.group(3) else None
            if pattern:
                logger.debug(f"[AgenticProtocol] XML file_grep marker found: {pattern}")
                decisions.append(SearchDecision(
                    wants_file_grep=True,
                    file_grep_pattern=pattern,
                    file_grep_folder=folder if folder else None,
                    file_grep_glob=glob if glob else None,
                ))

        # Check for git stats markers
        for git_stats_match in self.GIT_STATS_PATTERN.finditer(text):
            raw = git_stats_match.group(1).strip()
            # Handle nested tags: <git_stats><query>...</query></git_stats>
            query = _strip_xml_tags(raw) if '<' in raw else raw
            if not query:
                query = "recent commits"
                logger.warning(f"[AgenticProtocol] XML git_stats empty, defaulting to '{query}'")
            logger.debug(f"[AgenticProtocol] XML git_stats marker found: {query}")
            decisions.append(SearchDecision(
                wants_git_stats=True,
                git_stats_query=query,
            ))

        # Check for GitHub API markers
        for github_match in self.GITHUB_PATTERN.finditer(text):
            raw = github_match.group(1).strip()
            # Handle nested tags: <github><action>list_repos</action></github>
            query = _strip_xml_tags(raw) if '<' in raw else raw
            if not query:
                query = "repo info"
                logger.warning(f"[AgenticProtocol] XML github empty, defaulting to '{query}'")
            logger.debug(f"[AgenticProtocol] XML github marker found: {query}")
            decisions.append(SearchDecision(
                wants_github=True,
                github_query=query,
            ))

        # Check for fetch_url markers
        for fetch_url_match in self.FETCH_URL_PATTERN.finditer(text):
            url = fetch_url_match.group(1).strip()
            reason = fetch_url_match.group(2).strip() or None
            if url:
                logger.debug(f"[AgenticProtocol] XML fetch_url marker found: {url}")
                decisions.append(SearchDecision(
                    wants_fetch_url=True,
                    fetch_url=url,
                    fetch_url_reason=reason,
                ))

        # Check for file list markers
        for file_list_match in self.FILE_LIST_PATTERN.finditer(text):
            dirpath = file_list_match.group(1).strip()
            recursive = file_list_match.group(2) or "false"
            if dirpath:
                logger.debug(f"[AgenticProtocol] XML file_list marker found: {dirpath}")
                decisions.append(SearchDecision(
                    wants_file_list=True,
                    file_list_path=dirpath,
                    file_list_recursive=recursive.lower() == "true",
                ))

        # Check for search markers (content-style: <search>query</search> or <web_search>query</web_search>)
        for search_match in self.SEARCH_PATTERN.finditer(text):
            query = search_match.group(1).strip()
            if query:
                logger.debug(f"[AgenticProtocol] XML search marker found: {query}")
                decisions.append(SearchDecision(
                    wants_search=True,
                    search_query=query
                ))

        # Check for attribute-style search markers (<web_search query="..."> or <search query="...">)
        for search_attr_match in self.SEARCH_ATTR_PATTERN.finditer(text):
            query = search_attr_match.group(1).strip()
            if query:
                logger.debug(f"[AgenticProtocol] XML search attr marker found: {query}")
                decisions.append(SearchDecision(
                    wants_search=True,
                    search_query=query
                ))

        # Check for attribute-style memory markers (<search_memory query="...">)
        for memory_attr_match in self.MEMORY_ATTR_PATTERN.finditer(text):
            query = memory_attr_match.group(1).strip()
            if query:
                logger.debug(f"[AgenticProtocol] XML search_memory attr marker found: {query}")
                decisions.append(SearchDecision(
                    wants_memory_search=True,
                    memory_query=query,
                ))

        # Nested-tag fallback: <search_memory><query>X</query><collection>Y</collection></search_memory>
        for nested_match in self.MEMORY_NESTED_PATTERN.finditer(text):
            body = nested_match.group(1)
            query = _extract_nested_tag(body, "query")
            collection = _extract_nested_tag(body, "collection") or "facts"
            if query:
                logger.debug(f"[AgenticProtocol] XML search_memory nested marker found: {collection}/{query}")
                decisions.append(SearchDecision(
                    wants_memory_search=True,
                    memory_query=query,
                    memory_collection=collection,
                ))

        # Check for action markers: <action type="send_telegram" recipient="@luke">message</action>
        for action_match in self.ACTION_PATTERN.finditer(text):
            action_type = action_match.group(1).strip()
            recipient = action_match.group(2) or ""
            subject = action_match.group(3) or ""
            reason = action_match.group(4) or ""
            message = action_match.group(5).strip()
            if action_type and message:
                params = {"message": message}
                if recipient:
                    params["recipient"] = recipient
                if subject:
                    params["subject"] = subject
                summary = f"{action_type}: {message[:80]}"
                if recipient:
                    summary = f"{action_type} to {recipient}: {message[:60]}"
                logger.info(f"[AgenticProtocol] XML action marker found: {summary}")
                decisions.append(SearchDecision(
                    wants_action=True,
                    action_type=action_type,
                    action_params=params,
                    action_summary=summary,
                    action_reason=reason,
                ))

        # Check for <propose_action> markers
        for pa_match in self.PROPOSE_ACTION_PATTERN.finditer(text):
            action_type = pa_match.group(1).strip()
            recipient = pa_match.group(2) or ""
            subject = pa_match.group(3) or ""
            reason = pa_match.group(4) or ""
            message = pa_match.group(5).strip()
            if action_type:
                params = {}
                if message:
                    params["message"] = message
                if recipient:
                    params["recipient"] = recipient
                if subject:
                    params["subject"] = subject
                summary = f"{action_type} to {recipient}: {message[:60]}" if recipient else f"{action_type}: {message[:80]}"
                logger.info(f"[AgenticProtocol] XML propose_action marker found: {summary}")
                decisions.append(SearchDecision(
                    wants_action=True,
                    action_type=action_type,
                    action_params=params,
                    action_summary=summary,
                    action_reason=reason or "User requested",
                ))

        # Check for <lookup_contact> markers
        for lc_match in self.LOOKUP_CONTACT_PATTERN.finditer(text):
            name = lc_match.group(1).strip()
            reason = lc_match.group(2).strip()
            if name:
                logger.debug(f"[AgenticProtocol] XML lookup_contact marker found: {name}")
                decisions.append(SearchDecision(
                    wants_lookup_contact=True,
                    lookup_contact_name=name,
                    lookup_contact_reason=reason,
                ))

        # Fallback: <invoke name="..."> (Anthropic-style function calls emitted as text)
        if not decisions:
            for invoke_match in self.INVOKE_PATTERN.finditer(text):
                func_name = invoke_match.group(1)
                body = invoke_match.group(2)
                args = {}
                for param_match in self.INVOKE_PARAM_PATTERN.finditer(body):
                    args[param_match.group(1)] = param_match.group(2).strip()
                # Delegate to NativeToolsHandler parsing for known tools
                if func_name in ("lookup_contact", "search_contacts", "find_contact",
                               "search_gmail", "search_email", "gmail_search", "search_inbox"):
                    name = args.get("name", "") or args.get("query", "")
                    if name:
                        logger.debug(f"[AgenticProtocol] XML invoke lookup_contact (via {func_name}): {name}")
                        decisions.append(SearchDecision(
                            wants_lookup_contact=True,
                            lookup_contact_name=name,
                            lookup_contact_reason=args.get("reason"),
                        ))
                elif func_name == "propose_action":
                    action_type = args.get("action_type") or args.get("type", "")
                    if action_type:
                        params = {}
                        for k in ("message", "recipient", "subject"):
                            if args.get(k):
                                params[k] = args[k]
                        summary = f"{action_type}: {args.get('message', '')[:80]}"
                        logger.info(f"[AgenticProtocol] XML invoke propose_action: {summary}")
                        decisions.append(SearchDecision(
                            wants_action=True,
                            action_type=action_type,
                            action_params=params,
                            action_summary=summary,
                            action_reason=args.get("reason", "User requested"),
                        ))
                else:
                    logger.debug(f"[AgenticProtocol] XML invoke unknown tool: {func_name}")

        # No markers found - model wants to answer
        if not decisions:
            return [SearchDecision(
                wants_answer=True,
                partial_response=text
            )]

        return decisions

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
    git_stats_available: bool = False,
    github_available: bool = False,
    fetch_url_available: bool = False,
    actions_available: bool = False,
) -> BaseProtocolHandler:
    """
    Factory function to get appropriate protocol handler.

    Args:
        protocol: The protocol to use
        wolfram_available: Whether Wolfram Alpha is configured
        sandbox_available: Whether E2B sandbox is configured
        memory_available: Whether ChromaDB memory search is available
        file_access_available: Whether file access manager is configured
        git_stats_available: Whether git stats manager is configured
        github_available: Whether GitHub API manager is configured
        fetch_url_available: Whether web search manager supports URL extraction
        actions_available: Whether internet write actions are enabled

    Returns:
        Protocol handler instance
    """
    if protocol == SearchProtocol.NATIVE_TOOLS:
        return NativeToolsHandler(
            wolfram_available=wolfram_available,
            sandbox_available=sandbox_available,
            memory_available=memory_available,
            file_access_available=file_access_available,
            git_stats_available=git_stats_available,
            github_available=github_available,
            fetch_url_available=fetch_url_available,
            actions_available=actions_available,
        )
    else:
        return XMLMarkerHandler()
