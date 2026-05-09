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
        self.token_manager = token_manager
        self.memory_expander = memory_expander
        self.compression_model = compression_model

        # Web source map for citation tracking across rounds
        self._current_web_source_map = {}

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
        result = await self._execute_search(
            [decision.search_query], crisis_level=crisis_level
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

    # ------------------------------------------------------------------
    # Low-level execution methods
    # ------------------------------------------------------------------

    async def _execute_search(
        self,
        search_terms: List[str],
        crisis_level: Optional[str] = None
    ) -> Any:
        """Execute web search with given terms."""
        from knowledge.web_search_manager import WebSearchDepth

        if len(search_terms) == 1:
            return await self.web_search_manager.search(
                query=search_terms[0],
                depth=WebSearchDepth.STANDARD,
                crisis_level=crisis_level
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

            # For wiki_knowledge: always prefer FAISS semantic search (40M vectors)
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

            if not results:
                return f"[No results found in {collection} for: {query}]"

            return self.formatter.format_memory_results(results, collection)

        except Exception as e:
            logger.warning(f"[AgenticSearch] Memory search failed: {e}")
            return f"[Memory search error: {e}]"

    def _search_wiki_faiss(self, query: str, k: int = 8) -> list[dict]:
        """Search the FAISS Wikipedia index (40M vectors) as fallback for wiki_knowledge."""
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
