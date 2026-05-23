"""
Agentic Search Tool Executor

Contract:
    - Provides ToolExecutor for dispatching and executing agentic search tools
    - Routes SearchDecision objects to appropriate tool handlers
    - Handles web search, Wolfram, sandbox, memory search/expand, file access,
      full document retrieval, and git stats
    - Extracted from AgenticSearchController to reduce god-object size

Dependencies:
    - core.agentic.formatters.AgenticFormatter (result formatting)
    - core.agentic.types (SearchDecision, _ToolResult, ProgressEvent, etc.)
    - Various manager classes via constructor injection
"""

import logging
import time
from typing import Any, List, Optional, TYPE_CHECKING

from core.agentic.types import (
    ProgressEvent,
    SearchDecision,
    SearchRequest,
    SearchRound,
    _ToolResult,
)
from core.agentic.formatters import AgenticFormatter

if TYPE_CHECKING:
    from models.model_manager import ModelManager
    from knowledge.web_search_manager import WebSearchManager
    from knowledge.wolfram_manager import WolframManager
    from knowledge.sandbox_manager import SandboxManager
    from core.prompt.token_manager import TokenManager
    from core.file_access_manager import FileAccessManager
    from core.git_stats_manager import GitStatsManager
    from core.github_manager import GitHubManager
    from memory.memory_expander import MemoryExpander

logger = logging.getLogger(__name__)

DEFAULT_COMPRESSION_MAX_TOKENS = 1500


class ToolExecutor:
    """
    Dispatches and executes agentic search tools.

    Routes SearchDecision objects to the appropriate handler, executes the
    underlying operation via injected managers, and returns formatted results.
    """

    VALID_MEMORY_COLLECTIONS = frozenset({
        "reference_docs", "facts", "conversations", "summaries",
        "reflections", "obsidian_notes", "procedural", "procedural_skills",
        "wiki_knowledge",
    })

    def __init__(
        self,
        model_manager: "ModelManager",
        web_search_manager: "WebSearchManager",
        formatter: AgenticFormatter,
        chroma_store=None,
        wolfram_manager: Optional["WolframManager"] = None,
        sandbox_manager: Optional["SandboxManager"] = None,
        file_access_manager: Optional["FileAccessManager"] = None,
        git_stats_manager: Optional["GitStatsManager"] = None,
        github_manager: Optional["GitHubManager"] = None,
        token_manager: Optional["TokenManager"] = None,
        memory_expander: Optional["MemoryExpander"] = None,
        compression_model: str = "gpt-4o-mini",
    ):
        self.model_manager = model_manager
        self.web_search_manager = web_search_manager
        self.formatter = formatter
        self.chroma_store = chroma_store
        self.wolfram_manager = wolfram_manager
        self.sandbox_manager = sandbox_manager
        self.file_access_manager = file_access_manager
        self.git_stats_manager = git_stats_manager
        self.github_manager = github_manager
        self.token_manager = token_manager
        self.memory_expander = memory_expander
        self.compression_model = compression_model

        # Web source map for citation tracking across rounds
        self._current_web_source_map = {}

    def get_tool_health(self) -> str:
        """Return a status summary of tool backends for the agentic system prompt.

        Reports which tools are available, degraded, or unavailable so the LLM
        never confabulates about its own capabilities.
        """
        lines = []

        # Web search
        if self.web_search_manager and self.web_search_manager.is_available():
            lines.append("web_search: AVAILABLE")
        else:
            lines.append("web_search: UNAVAILABLE (no API key or Tavily client error)")

        # FAISS Wikipedia index
        try:
            from knowledge.semantic_search import is_faiss_available
            if is_faiss_available():
                lines.append("wiki_knowledge (FAISS 41M vectors): AVAILABLE")
            else:
                lines.append(
                    "wiki_knowledge (FAISS 41M vectors): UNAVAILABLE "
                    "(index files not found — drive may be disconnected). "
                    "Only sparse ChromaDB wiki_knowledge is available as fallback."
                )
        except Exception:
            lines.append("wiki_knowledge (FAISS): UNAVAILABLE (import error)")

        # Memory / ChromaDB
        if self.chroma_store:
            lines.append("memory_search (ChromaDB): AVAILABLE")
        else:
            lines.append("memory_search (ChromaDB): UNAVAILABLE")

        # Wolfram
        if self.wolfram_manager:
            lines.append("wolfram: AVAILABLE")
        else:
            lines.append("wolfram: UNAVAILABLE")

        # File access
        if self.file_access_manager:
            lines.append("file_access: AVAILABLE")
        else:
            lines.append("file_access: UNAVAILABLE")

        # Git stats
        if self.git_stats_manager:
            lines.append("git_stats: AVAILABLE")
        else:
            lines.append("git_stats: UNAVAILABLE")

        # Memory expander
        if self.memory_expander:
            lines.append("expand_memory: AVAILABLE")
        else:
            lines.append("expand_memory: UNAVAILABLE")

        # Visual memory
        try:
            from config.app_config import VISUAL_MEMORY_ENABLED
            if VISUAL_MEMORY_ENABLED:
                lines.append("recall_image: AVAILABLE")
            else:
                lines.append("recall_image: DISABLED")
        except Exception:
            lines.append("recall_image: DISABLED")

        # GitHub API
        if self.github_manager:
            lines.append("github: AVAILABLE (issues, PRs, actions, releases, search — read-only)")
        else:
            lines.append("github: UNAVAILABLE (gh CLI not installed or not authenticated)")

        # Dedicated search APIs (always available — free, no auth)
        lines.append("search_stackexchange: AVAILABLE (Stack Overflow, ServerFault, etc.)")
        lines.append("search_arxiv: AVAILABLE (academic papers)")
        lines.append("search_pubmed: AVAILABLE (biomedical literature)")
        lines.append("search_hackernews: AVAILABLE (tech news/discussion)")

        # Document generation
        try:
            from config.app_config import DOCUMENT_GENERATION_ENABLED
            if DOCUMENT_GENERATION_ENABLED:
                lines.append("generate_document: AVAILABLE (research topic → save markdown report/summary to documents/)")
            else:
                lines.append("generate_document: DISABLED")
        except Exception:
            lines.append("generate_document: DISABLED")

        # Daemon self-notes
        try:
            from config.app_config import DAEMON_NOTES_ENABLED
            if DAEMON_NOTES_ENABLED:
                lines.append("create_daemon_note: AVAILABLE (save working context for future sessions → daemon_notes/)")
            else:
                lines.append("create_daemon_note: DISABLED")
        except Exception:
            lines.append("create_daemon_note: DISABLED")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dispatch router
    # ------------------------------------------------------------------

    async def dispatch_single(
        self,
        decision: SearchDecision,
        round_number: int,
        session: Any,
        crisis_level: Optional[str],
        sandbox_session: Optional[Any],
    ) -> _ToolResult:
        """Route a single SearchDecision to the appropriate dispatch method."""
        if decision.wants_search and decision.search_query:
            # If the search query contains a URL, reroute to fetch_url
            import re as _re
            q = decision.search_query.strip()
            _url_match = _re.search(r'(https?://[^\s<>"\')\]]+)', q)
            if _url_match:
                url = _url_match.group(1)
                logger.info(f"[ToolExecutor] Rerouting URL from web_search to fetch_url: {url}")
                decision = SearchDecision(
                    wants_fetch_url=True,
                    fetch_url=url,
                    fetch_url_reason=decision.search_reason,
                )
                return await self._dispatch_fetch_url(decision, round_number)
            return await self._dispatch_web_search(decision, round_number, crisis_level)
        elif decision.wants_wolfram and decision.wolfram_query:
            return await self._dispatch_wolfram(decision, round_number)
        elif decision.wants_sandbox and decision.sandbox_code:
            return await self._dispatch_sandbox(decision, round_number, sandbox_session)
        elif decision.wants_memory_search and decision.memory_query:
            return await self._dispatch_memory_search(decision, round_number)
        elif decision.wants_memory_expand and decision.expand_memory_id:
            return await self._dispatch_memory_expand(decision, round_number)
        elif decision.wants_file_read and decision.file_read_path:
            return await self._dispatch_file_read(decision, round_number)
        elif decision.wants_file_grep and decision.file_grep_pattern:
            return await self._dispatch_file_grep(decision, round_number)
        elif decision.wants_file_list and decision.file_list_path:
            return await self._dispatch_file_list(decision, round_number)
        elif decision.wants_full_document and decision.full_document_title:
            return await self._dispatch_full_document(decision, round_number)
        elif decision.wants_git_stats and decision.git_stats_query:
            return await self._dispatch_git_stats(decision, round_number)
        elif decision.wants_recall_image and decision.recall_image_query:
            return await self._dispatch_recall_image(decision, round_number)
        elif decision.wants_fetch_url and decision.fetch_url:
            return await self._dispatch_fetch_url(decision, round_number)
        elif decision.wants_stackexchange and decision.stackexchange_query:
            return await self._dispatch_api_search(decision, round_number, "stackexchange")
        elif decision.wants_arxiv and decision.arxiv_query:
            return await self._dispatch_api_search(decision, round_number, "arxiv")
        elif decision.wants_pubmed and decision.pubmed_query:
            return await self._dispatch_api_search(decision, round_number, "pubmed")
        elif decision.wants_hackernews and decision.hackernews_query:
            return await self._dispatch_api_search(decision, round_number, "hackernews")
        elif decision.wants_github and decision.github_query:
            return await self._dispatch_github(decision, round_number)
        elif decision.wants_generate_document and decision.generate_document_topic:
            return await self._dispatch_generate_document(decision, round_number)
        elif decision.wants_create_daemon_note and decision.daemon_note_title:
            return await self._dispatch_create_daemon_note(decision, round_number)
        else:
            return _ToolResult(
                decision=decision, round_data=None,
                formatted_context="", start_events=[], end_events=[],
            )

    # ------------------------------------------------------------------
    # Dispatch methods
    # ------------------------------------------------------------------

    async def _dispatch_web_search(
        self, decision: SearchDecision, round_number: int,
        crisis_level: Optional[str],
    ) -> _ToolResult:
        start_events = [ProgressEvent(
            event_type="searching",
            message=f"Searching for: {decision.search_query}",
            round_number=round_number,
            metadata={"query": decision.search_query, "reason": decision.search_reason}
        )]

        start_time = time.time()
        include_domains = [decision.search_site] if decision.search_site else None
        result = await self._execute_search(
            [decision.search_query], crisis_level=crisis_level,
            include_domains=include_domains,
        )
        search_duration = (time.time() - start_time) * 1000

        round_data = SearchRound(
            round_number=round_number,
            request=SearchRequest(
                query=decision.search_query,
                reason=decision.search_reason,
                round_number=round_number
            ),
            results=result,
            duration_ms=search_duration
        )

        result_count = len(result.pages) if result and hasattr(result, 'pages') else 0
        compressed = await self._compress_results(result)
        round_data.summary = compressed

        end_events = [ProgressEvent(
            event_type="found_results",
            message=f"Found {result_count} results",
            round_number=round_number,
            metadata={"result_count": result_count}
        )]

        return _ToolResult(
            decision=decision,
            round_data=round_data,
            formatted_context=self.formatter.format_search_context(
                round_number, decision.search_query, compressed
            ),
            start_events=start_events,
            end_events=end_events,
        )

    async def _dispatch_wolfram(
        self, decision: SearchDecision, round_number: int,
    ) -> _ToolResult:
        start_events = [ProgressEvent(
            event_type="computing",
            message=f"Computing: {decision.wolfram_query}",
            round_number=round_number,
            metadata={"query": decision.wolfram_query, "reason": decision.wolfram_reason}
        )]

        start_time = time.time()
        wolfram_result = await self._execute_wolfram(decision.wolfram_query)
        compute_duration = (time.time() - start_time) * 1000

        round_data = SearchRound(
            round_number=round_number,
            request=SearchRequest(
                query=decision.wolfram_query,
                reason=decision.wolfram_reason,
                round_number=round_number
            ),
            results=None,
            duration_ms=compute_duration
        )
        round_data.summary = wolfram_result

        end_events = [ProgressEvent(
            event_type="computed",
            message="Computation complete",
            round_number=round_number,
            metadata={"duration_ms": compute_duration}
        )]

        return _ToolResult(
            decision=decision,
            round_data=round_data,
            formatted_context=self.formatter.format_wolfram_context(
                round_number, decision.wolfram_query, wolfram_result
            ),
            start_events=start_events,
            end_events=end_events,
        )

    async def _dispatch_sandbox(
        self, decision: SearchDecision, round_number: int,
        sandbox_session: Optional[Any],
    ) -> _ToolResult:
        purpose = decision.sandbox_purpose or "executing code"
        start_events = [ProgressEvent(
            event_type="executing_code",
            message=f"Running Python: {purpose}",
            round_number=round_number,
            metadata={"purpose": purpose}
        )]

        start_time = time.time()

        # Execute in persistent session if available, otherwise ephemeral
        if sandbox_session and not sandbox_session.is_closed:
            sandbox_result = await sandbox_session.run(decision.sandbox_code)
        elif self.sandbox_manager and self.sandbox_manager.is_available():
            sandbox_result = await self.sandbox_manager.execute_code(decision.sandbox_code)
        else:
            from knowledge.sandbox_manager import SandboxResult
            sandbox_result = SandboxResult(
                code=decision.sandbox_code,
                success=False,
                error="Code sandbox not available (E2B not configured)"
            )

        execution_duration = (time.time() - start_time) * 1000

        round_data = SearchRound(
            round_number=round_number,
            request=SearchRequest(
                query=f"[Python: {purpose}]",
                reason=purpose,
                round_number=round_number
            ),
            results=None,
            duration_ms=execution_duration
        )

        if sandbox_result.success:
            end_events = [ProgressEvent(
                event_type="code_executed",
                message=f"Code executed ({sandbox_result.execution_time:.1f}s)",
                round_number=round_number,
                metadata={"duration_ms": execution_duration}
            )]
        else:
            end_events = [ProgressEvent(
                event_type="code_error",
                message="Execution error (see details)",
                round_number=round_number,
                metadata={"error": sandbox_result.error}
            )]

        formatted_result = self.sandbox_manager.format_for_prompt(
            sandbox_result, purpose
        ) if self.sandbox_manager else str(sandbox_result.error or sandbox_result.stdout)
        round_data.summary = formatted_result

        return _ToolResult(
            decision=decision,
            round_data=round_data,
            formatted_context=self.formatter.format_sandbox_context(
                round_number, purpose, formatted_result
            ),
            start_events=start_events,
            end_events=end_events,
        )

    async def _dispatch_memory_search(
        self, decision: SearchDecision, round_number: int,
    ) -> _ToolResult:
        collection = decision.memory_collection or "facts"
        start_events = [ProgressEvent(
            event_type="searching_memory",
            message=f"Searching {collection}: {decision.memory_query}",
            round_number=round_number,
            metadata={
                "query": decision.memory_query,
                "collection": collection,
                "reason": decision.memory_reason,
            }
        )]

        start_time = time.time()
        memory_result = await self._execute_memory_search(
            decision.memory_query, collection
        )
        search_duration = (time.time() - start_time) * 1000

        round_data = SearchRound(
            round_number=round_number,
            request=SearchRequest(
                query=f"[Memory: {collection}] {decision.memory_query}",
                reason=decision.memory_reason,
                round_number=round_number
            ),
            results=None,
            duration_ms=search_duration
        )
        round_data.summary = memory_result

        end_events = [ProgressEvent(
            event_type="found_results",
            message=f"Found memory results from {collection}",
            round_number=round_number,
            metadata={"collection": collection, "duration_ms": search_duration}
        )]

        return _ToolResult(
            decision=decision,
            round_data=round_data,
            formatted_context=self.formatter.format_memory_context(
                round_number, collection, decision.memory_query, memory_result
            ),
            start_events=start_events,
            end_events=end_events,
            memory_collection=collection,
        )

    async def _dispatch_memory_expand(
        self, decision: SearchDecision, round_number: int,
    ) -> _ToolResult:
        memory_id = decision.expand_memory_id
        is_summary = (decision.expand_collection == "summaries")
        recall_label = "Recalling Long Term Memory..." if is_summary else "Recalling..."

        start_events = [ProgressEvent(
            event_type="expanding_memory",
            message=recall_label,
            round_number=round_number,
            metadata={
                "memory_id": memory_id,
                "collection": decision.expand_collection,
                "reason": decision.expand_reason,
            }
        )]

        start_time = time.time()
        expand_result = self._execute_memory_expand(
            memory_id, decision.expand_window, decision.expand_collection
        )
        duration = (time.time() - start_time) * 1000

        formatted = self.formatter.format_expanded_results(expand_result)
        n_turns = len(expand_result.get("turns", []))

        round_data = SearchRound(
            round_number=round_number,
            request=SearchRequest(
                query=f"[Expand Memory] {memory_id[:8]}",
                reason=decision.expand_reason,
                round_number=round_number
            ),
            results=None,
            duration_ms=duration
        )
        round_data.summary = formatted

        done_label = (
            f"Recalled {n_turns} memories from long term"
            if is_summary else f"Recalled {n_turns} surrounding turns"
        )
        end_events = [ProgressEvent(
            event_type="memory_expanded",
            message=done_label,
            round_number=round_number,
            metadata={"duration_ms": duration}
        )]

        return _ToolResult(
            decision=decision,
            round_data=round_data,
            formatted_context=self.formatter.format_expand_context(
                round_number, memory_id, formatted
            ),
            start_events=start_events,
            end_events=end_events,
            is_expand=True,
        )

    async def _dispatch_file_read(
        self, decision: SearchDecision, round_number: int,
    ) -> _ToolResult:
        start_events = [ProgressEvent(
            event_type="reading_file",
            message=f"Reading {decision.file_read_path}",
            round_number=round_number,
            metadata={"path": decision.file_read_path, "reason": decision.file_read_reason}
        )]

        start_time = time.time()
        file_result = await self._execute_file_read(
            decision.file_read_path,
            decision.file_read_start_line,
            decision.file_read_end_line,
        )
        duration = (time.time() - start_time) * 1000

        round_data = SearchRound(
            round_number=round_number,
            request=SearchRequest(
                query=f"[File Read] {decision.file_read_path}",
                reason=decision.file_read_reason,
                round_number=round_number
            ),
            results=None,
            duration_ms=duration
        )
        round_data.summary = file_result

        end_events = [ProgressEvent(
            event_type="file_read",
            message=f"Read {decision.file_read_path}",
            round_number=round_number,
            metadata={"duration_ms": duration}
        )]

        return _ToolResult(
            decision=decision,
            round_data=round_data,
            formatted_context=self.formatter.format_file_context(
                round_number, f"file_read: {decision.file_read_path}", file_result
            ),
            start_events=start_events,
            end_events=end_events,
        )

    async def _dispatch_file_grep(
        self, decision: SearchDecision, round_number: int,
    ) -> _ToolResult:
        start_events = [ProgressEvent(
            event_type="searching_files",
            message=f"Grepping for '{decision.file_grep_pattern}'",
            round_number=round_number,
            metadata={"pattern": decision.file_grep_pattern, "reason": decision.file_grep_reason}
        )]

        start_time = time.time()
        grep_result = await self._execute_file_grep(
            decision.file_grep_pattern,
            decision.file_grep_folder,
            decision.file_grep_glob,
        )
        duration = (time.time() - start_time) * 1000

        round_data = SearchRound(
            round_number=round_number,
            request=SearchRequest(
                query=f"[File Grep] {decision.file_grep_pattern}",
                reason=decision.file_grep_reason,
                round_number=round_number
            ),
            results=None,
            duration_ms=duration
        )
        round_data.summary = grep_result

        end_events = [ProgressEvent(
            event_type="files_searched",
            message=f"Grep complete for '{decision.file_grep_pattern}'",
            round_number=round_number,
            metadata={"duration_ms": duration}
        )]

        return _ToolResult(
            decision=decision,
            round_data=round_data,
            formatted_context=self.formatter.format_file_context(
                round_number, f"file_grep: {decision.file_grep_pattern}", grep_result
            ),
            start_events=start_events,
            end_events=end_events,
        )

    async def _dispatch_file_list(
        self, decision: SearchDecision, round_number: int,
    ) -> _ToolResult:
        start_events = [ProgressEvent(
            event_type="listing_files",
            message=f"Listing {decision.file_list_path}",
            round_number=round_number,
            metadata={"path": decision.file_list_path, "reason": decision.file_list_reason}
        )]

        start_time = time.time()
        list_result = await self._execute_file_list(
            decision.file_list_path,
            decision.file_list_recursive,
        )
        duration = (time.time() - start_time) * 1000

        round_data = SearchRound(
            round_number=round_number,
            request=SearchRequest(
                query=f"[File List] {decision.file_list_path}",
                reason=decision.file_list_reason,
                round_number=round_number
            ),
            results=None,
            duration_ms=duration
        )
        round_data.summary = list_result

        end_events = [ProgressEvent(
            event_type="files_listed",
            message=f"Listed {decision.file_list_path}",
            round_number=round_number,
            metadata={"duration_ms": duration}
        )]

        return _ToolResult(
            decision=decision,
            round_data=round_data,
            formatted_context=self.formatter.format_file_context(
                round_number, f"file_list: {decision.file_list_path}", list_result
            ),
            start_events=start_events,
            end_events=end_events,
        )

    async def _dispatch_full_document(
        self, decision: SearchDecision, round_number: int,
    ) -> _ToolResult:
        title = decision.full_document_title
        start_events = [ProgressEvent(
            event_type="retrieving_document",
            message=f"Retrieving full document: {title}",
            round_number=round_number,
            metadata={"title": title, "reason": decision.full_document_reason}
        )]

        start_time = time.time()
        doc_result = await self._execute_full_document_retrieval(title)
        duration = (time.time() - start_time) * 1000

        round_data = SearchRound(
            round_number=round_number,
            request=SearchRequest(
                query=f"[Full Document] {title}",
                reason=decision.full_document_reason,
                round_number=round_number
            ),
            results=None,
            duration_ms=duration
        )
        round_data.summary = doc_result

        end_events = [ProgressEvent(
            event_type="document_retrieved",
            message=f"Retrieved full document: {title}",
            round_number=round_number,
            metadata={"duration_ms": duration}
        )]

        return _ToolResult(
            decision=decision,
            round_data=round_data,
            formatted_context=self.formatter.format_full_document_context(
                round_number, title, doc_result
            ),
            start_events=start_events,
            end_events=end_events,
        )

    async def _dispatch_git_stats(
        self, decision: SearchDecision, round_number: int,
    ) -> _ToolResult:
        start_events = [ProgressEvent(
            event_type="querying_git",
            message=f"Git stats: {decision.git_stats_query}",
            round_number=round_number,
            metadata={"query": decision.git_stats_query, "reason": decision.git_stats_reason}
        )]

        start_time = time.time()
        git_result = await self._execute_git_stats(decision.git_stats_query)
        duration = (time.time() - start_time) * 1000

        round_data = SearchRound(
            round_number=round_number,
            request=SearchRequest(
                query=f"[Git Stats] {decision.git_stats_query}",
                reason=decision.git_stats_reason,
                round_number=round_number
            ),
            results=None,
            duration_ms=duration
        )
        round_data.summary = git_result

        end_events = [ProgressEvent(
            event_type="git_stats_done",
            message="Git stats retrieved",
            round_number=round_number,
            metadata={"duration_ms": duration}
        )]

        return _ToolResult(
            decision=decision,
            round_data=round_data,
            formatted_context=self.formatter.format_git_stats_context(
                round_number, decision.git_stats_query, git_result
            ),
            start_events=start_events,
            end_events=end_events,
        )

    async def _dispatch_recall_image(
        self, decision: SearchDecision, round_number: int,
    ) -> _ToolResult:
        start_events = [ProgressEvent(
            event_type="recalling_image",
            message=f"Searching visual memory: {decision.recall_image_query}",
            round_number=round_number,
            metadata={"query": decision.recall_image_query, "reason": decision.recall_image_reason}
        )]

        start_time = time.time()
        visual_result = await self._execute_recall_image(decision.recall_image_query)
        duration = (time.time() - start_time) * 1000

        round_data = SearchRound(
            round_number=round_number,
            request=SearchRequest(
                query=f"[Visual Memory] {decision.recall_image_query}",
                reason=decision.recall_image_reason,
                round_number=round_number
            ),
            results=None,
            duration_ms=duration
        )
        round_data.summary = visual_result.get("summary", "No visual memories found.")

        end_events = [ProgressEvent(
            event_type="recall_image_done",
            message=f"Found {visual_result.get('count', 0)} visual memories",
            round_number=round_number,
            metadata={"duration_ms": duration, "count": visual_result.get("count", 0)}
        )]

        return _ToolResult(
            decision=decision,
            round_data=round_data,
            formatted_context=visual_result.get("formatted", ""),
            start_events=start_events,
            end_events=end_events,
        )

    async def _dispatch_fetch_url(
        self, decision: SearchDecision, round_number: int,
    ) -> _ToolResult:
        start_events = [ProgressEvent(
            event_type="fetching_url",
            message=f"Fetching: {decision.fetch_url}",
            round_number=round_number,
            metadata={"url": decision.fetch_url, "reason": decision.fetch_url_reason}
        )]

        start_time = time.time()
        fetch_result = await self._execute_fetch_url(decision.fetch_url)
        duration = (time.time() - start_time) * 1000

        round_data = SearchRound(
            round_number=round_number,
            request=SearchRequest(
                query=f"[Fetch URL] {decision.fetch_url}",
                reason=decision.fetch_url_reason,
                round_number=round_number
            ),
            results=None,
            duration_ms=duration
        )
        round_data.summary = fetch_result

        end_events = [ProgressEvent(
            event_type="url_fetched",
            message=f"Fetched content from URL",
            round_number=round_number,
            metadata={"duration_ms": duration}
        )]

        return _ToolResult(
            decision=decision,
            round_data=round_data,
            formatted_context=self.formatter.format_fetch_url_context(
                round_number, decision.fetch_url, fetch_result
            ),
            start_events=start_events,
            end_events=end_events,
        )

    async def _dispatch_github(
        self, decision: SearchDecision, round_number: int,
    ) -> _ToolResult:
        start_events = [ProgressEvent(
            event_type="querying_github",
            message=f"GitHub: {decision.github_query}",
            round_number=round_number,
            metadata={"query": decision.github_query, "reason": decision.github_reason}
        )]

        start_time = time.time()
        github_result = await self._execute_github(decision.github_query)
        duration = (time.time() - start_time) * 1000

        round_data = SearchRound(
            round_number=round_number,
            request=SearchRequest(
                query=f"[GitHub] {decision.github_query}",
                reason=decision.github_reason,
                round_number=round_number
            ),
            results=None,
            duration_ms=duration
        )
        round_data.summary = github_result

        end_events = [ProgressEvent(
            event_type="github_done",
            message="GitHub query complete",
            round_number=round_number,
            metadata={"duration_ms": duration}
        )]

        return _ToolResult(
            decision=decision,
            round_data=round_data,
            formatted_context=self.formatter.format_github_context(
                round_number, decision.github_query, github_result
            ),
            start_events=start_events,
            end_events=end_events,
        )

    async def _dispatch_generate_document(
        self, decision: SearchDecision, round_number: int,
    ) -> _ToolResult:
        topic = decision.generate_document_topic or "unknown"
        doc_type = decision.generate_document_type or "report"

        start_events = [ProgressEvent(
            event_type="generating_document",
            message=f"Generating {doc_type}: {topic}",
            round_number=round_number,
            metadata={"topic": topic, "doc_type": doc_type, "reason": decision.generate_document_reason}
        )]

        start_time = time.time()
        doc_result = await self._execute_generate_document(
            topic, doc_type, decision.generate_document_focus
        )
        duration = (time.time() - start_time) * 1000

        round_data = SearchRound(
            round_number=round_number,
            request=SearchRequest(
                query=f"[Generate Document] {topic}",
                reason=decision.generate_document_reason,
                round_number=round_number
            ),
            results=None,
            duration_ms=duration
        )
        round_data.summary = doc_result

        end_events = [ProgressEvent(
            event_type="document_generated",
            message=f"Document saved: {topic}",
            round_number=round_number,
            metadata={"duration_ms": duration}
        )]

        formatted = (
            f"\n---\n**Round {round_number}: Document Generated**\n"
            f"{doc_result}\n---\n"
        )

        return _ToolResult(
            decision=decision,
            round_data=round_data,
            formatted_context=formatted,
            start_events=start_events,
            end_events=end_events,
        )

    async def _execute_generate_document(
        self, topic: str, doc_type: str, focus: str | None = None,
    ) -> str:
        """Execute document generation via DocumentGenerator."""
        from config import app_config

        if not app_config.DOCUMENT_GENERATION_ENABLED:
            return "[Document generation is disabled in config]"

        try:
            from knowledge.document_generator import DocumentGenerator

            generator = DocumentGenerator(
                model_manager=self.model_manager,
                web_search_manager=self.web_search_manager,
                chroma_store=self.chroma_store,
            )
            result = await generator.generate(
                topic=topic,
                doc_type=doc_type if doc_type in ("report", "summary") else "report",
                focus=focus,
            )
            return (
                f"Document saved: {result.path}\n"
                f"Title: {result.title}\n"
                f"Type: {result.doc_type}\n"
                f"Sources: {len(result.sources)}\n"
                f"Sections: {result.sections_count}\n"
                f"Words: {result.word_count}"
            )
        except Exception as e:
            logger.warning(f"[AgenticSearch] Document generation failed: {e}")
            return f"[Document generation error: {e}]"

    # ------------------------------------------------------------------
    async def _dispatch_create_daemon_note(
        self, decision: SearchDecision, round_number: int,
    ) -> _ToolResult:
        title = decision.daemon_note_title or "untitled"

        start_events = [ProgressEvent(
            event_type="saving_note",
            message=f"Saving self-note: {title}",
            round_number=round_number,
            metadata={"title": title, "category": decision.daemon_note_category}
        )]

        start_time = time.time()
        note_result = await self._execute_create_daemon_note(
            title, decision.daemon_note_category or "implementation",
            decision.daemon_note_summary or "",
        )
        duration = (time.time() - start_time) * 1000

        round_data = SearchRound(
            round_number=round_number,
            request=SearchRequest(
                query=f"[Self-Note] {title}",
                reason=decision.daemon_note_reason,
                round_number=round_number
            ),
            results=None,
            duration_ms=duration
        )
        round_data.summary = note_result

        end_events = [ProgressEvent(
            event_type="note_saved",
            message=f"Self-note saved: {title}",
            round_number=round_number,
            metadata={"duration_ms": duration}
        )]

        formatted = (
            f"\n---\n**Round {round_number}: Self-Note Saved**\n"
            f"{note_result}\n---\n"
        )

        return _ToolResult(
            decision=decision,
            round_data=round_data,
            formatted_context=formatted,
            start_events=start_events,
            end_events=end_events,
        )

    # Class-level singleton for autonomous note session tracking.
    # Shared across ToolExecutor instances within the same process so
    # the per-session cap survives across agentic rounds.
    _daemon_notes_manager: Optional[Any] = None

    async def _execute_create_daemon_note(
        self, title: str, category: str, summary: str,
    ) -> str:
        """Execute autonomous self-note creation with guardrails.

        Uses create_autonomous_note() which enforces:
        - Per-session cap (max 3)
        - Semantic dedup (skip if >0.85 similarity to existing note)
        - Session ID tracking for audit trail
        """
        from config import app_config

        if not app_config.DAEMON_NOTES_ENABLED:
            return "[Self-notes disabled in config]"

        if not summary or len(summary.strip()) < 10:
            return "[Self-note skipped: summary too short]"

        try:
            from knowledge.daemon_notes_manager import DaemonNotesManager

            # Reuse manager instance for session cap tracking
            if ToolExecutor._daemon_notes_manager is None:
                ToolExecutor._daemon_notes_manager = DaemonNotesManager(
                    model_manager=self.model_manager,
                    chroma_store=self.chroma_store,
                )

            manager = ToolExecutor._daemon_notes_manager
            # Ensure model_manager/chroma_store are current
            manager.model_manager = self.model_manager
            manager.chroma_store = self.chroma_store

            # Get session ID from current web source map timestamp or fallback
            session_id = str(int(time.time()))

            result = await manager.create_autonomous_note(
                title=title,
                category=category if category in ("implementation", "architecture", "research", "decisions") else "implementation",
                summary=summary,
                confidence="tentative",
                session_id=session_id,
                status="tentative",
            )

            if result is None:
                return "[Self-note skipped by guardrails (session cap or dedup)]"

            return (
                f"Self-note saved: {result.title}\n"
                f"Path: {result.path}\n"
                f"Category: {result.category}\n"
                f"Status: {result.status}"
            )
        except Exception as e:
            logger.warning(f"[AgenticSearch] Self-note creation failed: {e}")
            return f"[Self-note error: {e}]"

    # Low-level execution methods
    # ------------------------------------------------------------------

    async def _execute_search(
        self,
        search_terms: List[str],
        crisis_level: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
    ) -> Any:
        """Execute web search with given terms."""
        from knowledge.web_search_manager import WebSearchDepth

        if len(search_terms) == 1:
            return await self.web_search_manager.search(
                query=search_terms[0],
                depth=WebSearchDepth.STANDARD,
                crisis_level=crisis_level,
                include_domains=include_domains,
            )
        else:
            return await self.web_search_manager.multi_search(
                query=search_terms[0],
                depth=WebSearchDepth.STANDARD,
                auto_decompose=False,
                sub_queries=search_terms,
            )

    async def _compress_results(
        self,
        result: Any,
        max_tokens: int = DEFAULT_COMPRESSION_MAX_TOKENS
    ) -> str:
        """Compress search results to fit context budget."""
        if not result or not hasattr(result, 'pages') or not result.pages:
            return "No results found."

        from knowledge.web_search_manager import assign_web_ids, format_web_sources_with_ids
        numbered_sources, web_source_map = assign_web_ids(result.pages)
        self._current_web_source_map.update(web_source_map)
        formatted = format_web_sources_with_ids(numbered_sources, max_chars=6000)

        estimated_tokens = len(formatted) // 4

        if estimated_tokens <= max_tokens:
            return formatted

        try:
            summary_prompt = f"""Summarize these search results concisely, preserving key facts, dates, and sources:

{formatted}

Provide a focused summary with the most important information."""

            summary = await self.model_manager.generate_once(
                prompt=summary_prompt,
                model_name=self.compression_model,
                max_tokens=max_tokens,
                temperature=0.3
            )

            return summary if summary else formatted[:max_tokens * 4]

        except Exception as e:
            logger.warning(f"[AgenticSearch] Compression failed: {e}")
            return formatted[:max_tokens * 4]

    async def _execute_wolfram(self, query: str) -> str:
        """Execute Wolfram Alpha query with fallback to web search."""
        if not self.wolfram_manager:
            logger.warning("[AgenticSearch] Wolfram Alpha not configured, falling back to web search")
            result = await self._execute_search([f"{query} calculation explanation"])
            return await self._compress_results(result)

        result = await self.wolfram_manager.query(query)

        if result.success:
            formatted = self.wolfram_manager.format_for_prompt(result)
            logger.info(
                f"[AgenticSearch] Wolfram query '{query[:40]}...' succeeded in {result.execution_time:.2f}s"
            )
            return formatted

        logger.warning(f"[AgenticSearch] Wolfram failed ({result.error}), falling back to web search")
        fallback_result = await self._execute_search([f"{query} explanation solution"])
        return await self._compress_results(fallback_result)

    async def _execute_memory_search(self, query: str, collection: str) -> str:
        """Execute raw semantic search against a ChromaDB collection."""
        from config.app_config import AGENTIC_MEMORY_SEARCH_LIMIT

        if not self.chroma_store:
            return "[Memory search unavailable]"

        if collection not in self.VALID_MEMORY_COLLECTIONS:
            return f"[Invalid collection: {collection}. Valid: {', '.join(sorted(self.VALID_MEMORY_COLLECTIONS))}]"

        try:
            results = self.chroma_store.query_collection(
                collection_name=collection,
                query_text=query,
                n_results=AGENTIC_MEMORY_SEARCH_LIMIT,
            )

            # For wiki_knowledge: always prefer FAISS semantic search (41M vectors)
            # over ChromaDB which has sparse/irrelevant legacy data.
            if collection == "wiki_knowledge":
                faiss_results = self._search_wiki_faiss(query, k=AGENTIC_MEMORY_SEARCH_LIMIT)
                if faiss_results:
                    # Track wiki titles for session enrichment
                    from knowledge.wiki_tracker import WikiArticleTracker
                    tracker = WikiArticleTracker.get_instance()
                    for r in faiss_results:
                        t = r.get("title", "")
                        if t:
                            tracker.track(t, r.get("text", "")[:500])
                    logger.info(f"[AgenticSearch] wiki_knowledge using FAISS index "
                                f"({len(faiss_results)} results)")
                    return self.formatter.format_wiki_faiss_results(faiss_results)
                else:
                    # FAISS returned nothing — check if the index is actually available
                    from knowledge.semantic_search import is_faiss_available
                    if not is_faiss_available():
                        faiss_warning = (
                            "[⚠ FAISS WIKIPEDIA INDEX UNAVAILABLE — the 41M-vector "
                            "index could not be loaded (drive may be disconnected or "
                            "files missing). Results below are from the sparse ChromaDB "
                            "wiki_knowledge collection only. DO NOT tell the user that "
                            "Wikipedia/FAISS search is working — it is NOT. If the user "
                            "asked whether Wikipedia search works, tell them it is "
                            "currently unavailable.]\n\n"
                        )
                        logger.warning("[AgenticSearch] FAISS index unavailable for "
                                       "wiki_knowledge search — falling back to ChromaDB")
                    else:
                        faiss_warning = ""
                    if not results:
                        if faiss_warning:
                            return faiss_warning + f"[No results found in {collection} for: {query}]"
                        return f"[No results found in {collection} for: {query}]"
                    return faiss_warning + self.formatter.format_memory_results(results, collection)

            if not results:
                return f"[No results found in {collection} for: {query}]"

            return self.formatter.format_memory_results(results, collection)

        except Exception as e:
            logger.warning(f"[AgenticSearch] Memory search failed: {e}")
            return f"[Memory search error: {e}]"

    def _search_wiki_faiss(self, query: str, k: int = 8) -> list[dict]:
        """Search the FAISS Wikipedia index (41M vectors) as fallback for wiki_knowledge."""
        try:
            from knowledge.semantic_search import semantic_search_with_neighbors
            return semantic_search_with_neighbors(query, k=k)
        except Exception as e:
            logger.warning(f"[AgenticSearch] FAISS wiki search failed: {e}")
            return []

    def _execute_memory_expand(
        self, memory_id: str, window: int = 3, collection: Optional[str] = None
    ) -> dict:
        """Run MemoryExpander.expand() and return the result dict."""
        if not self.memory_expander:
            return {"anchor_id": memory_id, "turns": [], "error": "Expander not available"}
        try:
            from config.app_config import EXPAND_MAX_WINDOW
            window = max(1, min(window, EXPAND_MAX_WINDOW))
            return self.memory_expander.expand(memory_id, window, collection)
        except Exception as e:
            logger.warning(f"[AgenticSearch] Memory expand failed: {e}")
            return {"anchor_id": memory_id, "turns": [], "error": str(e)}

    async def _execute_file_read(
        self, filepath: str, start_line: Optional[int] = None, end_line: Optional[int] = None
    ) -> str:
        """Read a file via FileAccessManager."""
        if not self.file_access_manager:
            return "[File access not configured]"
        try:
            result = await self.file_access_manager.read_file(filepath, start_line, end_line)
            return self.file_access_manager.format_read_for_prompt(result)
        except Exception as e:
            logger.warning(f"[AgenticSearch] File read failed: {e}")
            return f"[File read error: {e}]"

    async def _execute_file_grep(
        self, pattern: str, folder: Optional[str] = None, file_glob: Optional[str] = None
    ) -> str:
        """Grep files via FileAccessManager."""
        if not self.file_access_manager:
            return "[File access not configured]"
        try:
            result = await self.file_access_manager.grep_files(
                pattern, folder, file_glob or "*.py"
            )
            return self.file_access_manager.format_grep_for_prompt(result)
        except Exception as e:
            logger.warning(f"[AgenticSearch] File grep failed: {e}")
            return f"[File grep error: {e}]"

    async def _execute_file_list(
        self, dirpath: str, recursive: bool = False
    ) -> str:
        """List directory via FileAccessManager."""
        if not self.file_access_manager:
            return "[File access not configured]"
        try:
            result = await self.file_access_manager.list_directory(dirpath, recursive)
            return self.file_access_manager.format_list_for_prompt(result)
        except Exception as e:
            logger.warning(f"[AgenticSearch] File list failed: {e}")
            return f"[File list error: {e}]"

    async def _execute_full_document_retrieval(self, title: str) -> str:
        """Retrieve all chunks of a document by title, reassembled in order."""
        if not self.chroma_store:
            return "[Full document retrieval unavailable — no memory store]"
        try:
            from knowledge.reference_docs_manager import ReferenceDocsManager
            manager = ReferenceDocsManager(chroma_store=self.chroma_store)
            content = manager.get_full_document(title)
            if content:
                max_chars = 60000
                if len(content) > max_chars:
                    content = content[:max_chars] + f"\n\n[... truncated at {max_chars} chars — document continues ...]"
                resolved = manager._fuzzy_resolve_title(title)
                actual_title = resolved if resolved else title
                return f"[Full Document: {actual_title}]\n{content}"
            else:
                titles = manager.list_document_titles()
                if titles:
                    return f"[No document found matching '{title}'. Available titles: {', '.join(titles[:20])}]"
                return f"[No document found matching '{title}'. No documents in reference_docs collection.]"
        except Exception as e:
            logger.warning(f"[AgenticSearch] Full document retrieval failed: {e}")
            return f"[Full document retrieval error: {e}]"

    async def _execute_git_stats(self, query: str) -> str:
        """Execute a git stats query via GitStatsManager."""
        if not self.git_stats_manager:
            return "[Git stats not configured]"
        try:
            result = await self.git_stats_manager.execute_query(query)
            return self.git_stats_manager.format_for_prompt(result)
        except Exception as e:
            logger.warning(f"[AgenticSearch] Git stats failed: {e}")
            return f"[Git stats error: {e}]"

    async def _execute_github(self, query: str) -> str:
        """Execute a GitHub query via GitHubManager."""
        if not self.github_manager:
            return "[GitHub API not configured]"
        try:
            result = await self.github_manager.execute_query(query)
            return self.github_manager.format_for_prompt(result)
        except Exception as e:
            logger.warning(f"[AgenticSearch] GitHub query failed: {e}")
            return f"[GitHub query error: {e}]"

    async def _execute_fetch_url(self, url: str) -> str:
        """Fetch page content from a URL via Tavily extract API.

        Also registers the fetched page in the web source map for [WEB_N] citations.
        """
        if not self.web_search_manager:
            return "[Web search manager not configured — cannot fetch URLs]"
        try:
            pages = await self.web_search_manager._tavily_extract([url])
            if not pages:
                return f"[Could not fetch content from {url}]"
            page = pages[0]
            title = page.title or url
            content = page.content or page.snippet or ""
            if not content:
                return f"[Page at {url} returned no extractable content]"

            # Register in web source map for citation tracking
            try:
                from knowledge.web_search_manager import assign_web_ids
                numbered, source_map = assign_web_ids(pages)
                self._current_web_source_map.update(source_map)
                # Use the assigned WEB_N id in the output
                if numbered:
                    web_id = numbered[0].web_id
                    return f"[{web_id}] Title: {title}\nURL: {url}\n\n{content}"
            except Exception:
                pass  # Citation registration is non-critical

            return f"Title: {title}\nURL: {url}\n\n{content}"
        except Exception as e:
            logger.warning(f"[AgenticSearch] URL fetch failed for {url}: {e}")
            return f"[URL fetch error: {e}]"

    async def _execute_recall_image(self, query: str) -> dict:
        """Search visual memory for images matching the query."""
        try:
            from config.app_config import VISUAL_MEMORY_ENABLED
            if not VISUAL_MEMORY_ENABLED:
                return {"summary": "[Visual memory not enabled]", "formatted": "", "count": 0}

            from knowledge.clip_manager import get_clip_manager
            from knowledge.visual_memory_store import VisualMemoryStore
            from knowledge.visual_retrieval import VisualRetriever

            clip = get_clip_manager()
            chroma = getattr(self, '_chroma_store', None)
            if not chroma and self.memory_coordinator:
                chroma = getattr(self.memory_coordinator, 'chroma_store', None)
            store = VisualMemoryStore(chroma_store=chroma)
            retriever = VisualRetriever(clip, store)

            result = await retriever.retrieve_visual_memories(query, max_images=3)
            text_results = result.get("text_results", [])

            if not text_results:
                return {"summary": "No matching visual memories found.", "formatted": "", "count": 0}

            # Format for agentic context
            lines = [f"[VISUAL MEMORY SEARCH] query: {query}"]
            for i, r in enumerate(text_results, start=1):
                caption = r.get("caption", "")
                source = r.get("source", "")
                score = r.get("score", 0.0)
                entities = r.get("entity_ids", [])
                parts = [f"[{source}]"]
                if entities:
                    parts.append(f"entities: {', '.join(entities)}")
                parts.append(f"[relevance: {score:.2f}]")
                lines.append(f"{i}) {' '.join(parts)}\n   {caption}")

            formatted = "\n\n".join(lines)
            summary = f"Found {len(text_results)} visual memories for '{query}'"

            return {"summary": summary, "formatted": formatted, "count": len(text_results)}

        except ImportError:
            return {"summary": "[Visual memory dependencies not installed]", "formatted": "", "count": 0}
        except Exception as e:
            logger.warning(f"[AgenticSearch] Visual memory recall failed: {e}")
            return {"summary": f"[Visual memory error: {e}]", "formatted": "", "count": 0}

    # ------------------------------------------------------------------
    # Dedicated API search tools (no auth required)
    # ------------------------------------------------------------------

    async def _dispatch_api_search(
        self, decision: SearchDecision, round_number: int, tool_name: str,
    ) -> _ToolResult:
        """Unified dispatcher for Stack Exchange, arXiv, PubMed, Hacker News."""
        query_map = {
            "stackexchange": decision.stackexchange_query,
            "arxiv": decision.arxiv_query,
            "pubmed": decision.pubmed_query,
            "hackernews": decision.hackernews_query,
        }
        query = query_map.get(tool_name, "")
        start_events = [ProgressEvent(
            event_type="searching",
            message=f"Searching {tool_name}: {query}",
            round_number=round_number,
            metadata={"tool": tool_name, "query": query}
        )]

        start_time = time.time()
        try:
            if tool_name == "stackexchange":
                formatted = await self._execute_stackexchange(
                    query, site=decision.stackexchange_site or "stackoverflow"
                )
            elif tool_name == "arxiv":
                formatted = await self._execute_arxiv(query)
            elif tool_name == "pubmed":
                formatted = await self._execute_pubmed(query)
            elif tool_name == "hackernews":
                formatted = await self._execute_hackernews(query)
            else:
                formatted = f"Unknown API tool: {tool_name}"
        except Exception as e:
            logger.warning(f"[ToolExecutor] {tool_name} search failed: {e}")
            formatted = f"[{tool_name} search error: {e}]"

        duration_ms = (time.time() - start_time) * 1000
        end_events = [ProgressEvent(
            event_type="tool_complete",
            message=f"{tool_name} search complete ({duration_ms:.0f}ms)",
            round_number=round_number,
        )]
        return _ToolResult(
            decision=decision, round_data=None,
            formatted_context=formatted,
            start_events=start_events, end_events=end_events,
        )

    async def _execute_stackexchange(self, query: str, site: str = "stackoverflow") -> str:
        """Search Stack Exchange API. No auth needed."""
        import asyncio
        import urllib.parse
        import httpx

        url = (
            f"https://api.stackexchange.com/2.3/search/advanced"
            f"?order=desc&sort=votes&q={urllib.parse.quote(query)}"
            f"&site={site}&filter=withbody&pagesize=5"
        )
        loop = asyncio.get_event_loop()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            data = resp.json()

        items = data.get("items", [])
        if not items:
            return f"[Stack Exchange] No results for: {query}"

        lines = [f"[STACK EXCHANGE — {site}] {query}\n"]
        for i, item in enumerate(items[:5], 1):
            title = item.get("title", "")
            score = item.get("score", 0)
            answered = item.get("is_answered", False)
            accepted = "ACCEPTED" if item.get("accepted_answer_id") else ""
            link = item.get("link", "")
            # Extract text from body HTML (simple strip)
            import re
            body = re.sub(r"<[^>]+>", "", item.get("body", ""))[:500]
            lines.append(
                f"{i}. [{score} votes] {'[ANSWERED]' if answered else ''} {accepted}\n"
                f"   {title}\n   {body.strip()}\n   {link}\n"
            )
        return "\n".join(lines)

    async def _execute_arxiv(self, query: str) -> str:
        """Search arXiv API. No auth needed."""
        import urllib.parse
        import httpx
        import xml.etree.ElementTree as ET

        url = (
            f"http://export.arxiv.org/api/query"
            f"?search_query=all:{urllib.parse.quote(query)}"
            f"&start=0&max_results=5&sortBy=relevance&sortOrder=descending"
        )
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", ns)

        if not entries:
            return f"[arXiv] No results for: {query}"

        lines = [f"[arXiv SEARCH] {query}\n"]
        for i, entry in enumerate(entries[:5], 1):
            title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")[:400]
            authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
            link = ""
            for l in entry.findall("atom:link", ns):
                if l.get("title") == "pdf":
                    link = l.get("href", "")
                    break
            if not link:
                link = entry.findtext("atom:id", "", ns) or ""
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += f" et al. ({len(authors)} authors)"
            lines.append(
                f"{i}. {title}\n"
                f"   {author_str}\n"
                f"   {summary}\n"
                f"   {link}\n"
            )
        return "\n".join(lines)

    async def _execute_pubmed(self, query: str) -> str:
        """Search PubMed E-utilities. No auth needed."""
        import urllib.parse
        import httpx
        import xml.etree.ElementTree as ET

        # Step 1: search for IDs
        search_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=pubmed&term={urllib.parse.quote(query)}&retmax=5&sort=relevance&retmode=xml"
        )
        async with httpx.AsyncClient(timeout=15.0) as client:
            search_resp = await client.get(search_url)

        root = ET.fromstring(search_resp.text)
        ids = [id_el.text for id_el in root.findall(".//Id") if id_el.text]

        if not ids:
            return f"[PubMed] No results for: {query}"

        # Step 2: fetch summaries
        fetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=pubmed&id={','.join(ids)}&rettype=abstract&retmode=xml"
        )
        async with httpx.AsyncClient(timeout=15.0) as client:
            fetch_resp = await client.get(fetch_url)

        articles_root = ET.fromstring(fetch_resp.text)
        lines = [f"[PUBMED SEARCH] {query}\n"]

        for i, article in enumerate(articles_root.findall(".//PubmedArticle"), 1):
            title = article.findtext(".//ArticleTitle") or "No title"
            abstract = article.findtext(".//AbstractText") or "No abstract"
            pmid = article.findtext(".//PMID") or ""
            authors_el = article.findall(".//Author")
            authors = []
            for a in authors_el[:3]:
                last = a.findtext("LastName") or ""
                init = a.findtext("Initials") or ""
                if last:
                    authors.append(f"{last} {init}".strip())
            author_str = ", ".join(authors)
            if len(authors_el) > 3:
                author_str += f" et al."
            lines.append(
                f"{i}. {title}\n"
                f"   {author_str}\n"
                f"   {abstract[:400]}\n"
                f"   https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n"
            )
        return "\n".join(lines)

    async def _execute_hackernews(self, query: str) -> str:
        """Search Hacker News via Algolia API. No auth needed."""
        import urllib.parse
        import httpx

        url = (
            f"https://hn.algolia.com/api/v1/search"
            f"?query={urllib.parse.quote(query)}&tags=story&hitsPerPage=5"
        )
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            data = resp.json()

        hits = data.get("hits", [])
        if not hits:
            return f"[Hacker News] No results for: {query}"

        lines = [f"[HACKER NEWS SEARCH] {query}\n"]
        for i, hit in enumerate(hits[:5], 1):
            title = hit.get("title", "")
            points = hit.get("points", 0)
            comments = hit.get("num_comments", 0)
            url = hit.get("url", "")
            hn_url = f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
            lines.append(
                f"{i}. [{points} pts, {comments} comments] {title}\n"
                f"   {url}\n"
                f"   Discussion: {hn_url}\n"
            )
        return "\n".join(lines)
