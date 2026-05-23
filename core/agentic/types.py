"""
Agentic Search Type Definitions

Contract:
    - Defines data structures for agentic search sessions
    - All types are immutable dataclasses where possible
    - ProgressEvent is the standard UI update format

Public Types:
    - AgentState (enum): IDLE, THINKING, SEARCHING, OBSERVING, GENERATING, DONE, ERROR
    - SearchProtocol (enum): NATIVE_TOOLS, XML_MARKERS
    - SearchRequest, SearchRound, SearchDecision (dataclasses)
    - AgenticSearchSession (session state container)
      - final_prompt_hash: str — SHA-256[:16] of final assembled prompt for provenance [NEW 2026-03-26]
      - get_provenance_summary() → dict — serializable audit trail with per-round action classification [NEW 2026-03-26]
      - _classify_round_action(query) → str — classifies round by query prefix (memory_search, sandbox, file_read, etc.)
    - ProgressEvent (UI update events)
    - SEARCH_TOOL_DEFINITION, DONE_TOOL_DEFINITION, WOLFRAM_TOOL_DEFINITION, SANDBOX_TOOL_DEFINITION, MEMORY_SEARCH_TOOL_DEFINITION, EXPAND_MEMORY_TOOL_DEFINITION (tool schemas)
    - FILE_READ_TOOL_DEFINITION, FILE_GREP_TOOL_DEFINITION, FILE_LIST_TOOL_DEFINITION (file access tool schemas)
    - GET_FULL_DOCUMENT_TOOL_DEFINITION (full document retrieval tool schema)
    - GIT_STATS_TOOL_DEFINITION (git repository stats tool schema)
    - RECALL_IMAGE_TOOL_DEFINITION (visual memory CLIP image search tool schema)
    - FETCH_URL_TOOL_DEFINITION (direct URL content fetching tool schema)
    - GITHUB_TOOL_DEFINITION (read-only GitHub API tool schema)
    - GENERATE_DOCUMENT_TOOL_DEFINITION (research & save document tool schema)

SearchDecision Fields (extended for multi-tool support):
    - wants_search, search_query, search_reason (web search)
    - wants_wolfram, wolfram_query, wolfram_reason (Wolfram Alpha computation)
    - wants_sandbox, sandbox_code, sandbox_purpose (E2B Python sandbox) [NEW 2026-01-22]
    - wants_memory_search, memory_query, memory_collection, memory_reason (ChromaDB memory search)
    - wants_memory_expand, expand_memory_id, expand_window, expand_collection, expand_reason (memory expansion)
    - wants_file_read, file_read_path, file_read_start_line, file_read_end_line, file_read_reason (file read)
    - wants_file_grep, file_grep_pattern, file_grep_folder, file_grep_glob, file_grep_reason (file grep)
    - wants_file_list, file_list_path, file_list_recursive, file_list_reason (file list)
    - wants_git_stats, git_stats_query, git_stats_reason (git repository stats)
    - wants_full_document, full_document_title, full_document_reason (full document retrieval)
    - wants_recall_image, recall_image_query, recall_image_reason (CLIP visual memory search)
    - wants_fetch_url, fetch_url, fetch_url_reason (direct URL content fetching)
    - is_done, done_reason, wants_answer, partial_response

Dependencies:
    - None (pure data types)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge.web_search_manager import WebSearchResult


class AgentState(Enum):
    """Current state of the agentic search loop."""
    IDLE = "idle"
    THINKING = "thinking"      # LLM deciding what to do
    SEARCHING = "searching"    # Executing web search
    OBSERVING = "observing"    # Processing search results
    GENERATING = "generating"  # Generating final answer
    DONE = "done"
    ERROR = "error"


class SearchProtocol(Enum):
    """How search requests are communicated with the LLM."""
    NATIVE_TOOLS = "native_tools"    # OpenAI/Anthropic function calling
    XML_MARKERS = "xml_markers"      # <search>query</search> for local models


@dataclass
class SearchRequest:
    """A search request from the LLM."""
    query: str
    reason: Optional[str] = None
    round_number: int = 1
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SearchRound:
    """One iteration of the search loop."""
    round_number: int
    request: SearchRequest
    results: Optional[Any] = None  # WebSearchResult or MultiSearchResult
    summary: Optional[str] = None  # Compressed version for context
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class SearchDecision:
    """
    Result of parsing LLM output for search decisions.

    One or more of these may be True when the model requests parallel tools.
    When multiple flags are set, the controller dispatches them concurrently
    via asyncio.gather(). Protocol handlers return List[SearchDecision] —
    each element has exactly one wants_* flag set.

    Tool flags:
    - wants_search: Model wants to perform a web search
    - wants_wolfram: Model wants to perform a Wolfram Alpha computation
    - wants_sandbox: Model wants to execute Python code in sandbox
    - wants_memory_search: Model wants to search internal memory
    - wants_memory_expand: Model wants to expand a memory hit
    - wants_file_read/grep/list: File access tools
    - wants_git_stats: Git repository stats
    - wants_full_document: Full document retrieval
    - wants_fetch_url: Direct URL content fetching

    Terminal flags:
    - is_done: Model signals it has enough information
    - wants_answer: Model wants to provide final answer (no explicit done signal)
    """
    # Web search
    wants_search: bool = False
    search_query: Optional[str] = None
    search_site: Optional[str] = None  # Domain filter (e.g. "stackoverflow.com")
    search_reason: Optional[str] = None
    # Wolfram Alpha (quick calculations)
    wants_wolfram: bool = False
    wolfram_query: Optional[str] = None
    wolfram_reason: Optional[str] = None
    # Code Sandbox (multi-step computation)
    wants_sandbox: bool = False
    sandbox_code: Optional[str] = None
    sandbox_purpose: Optional[str] = None
    # Memory search (internal knowledge base)
    wants_memory_search: bool = False
    memory_query: Optional[str] = None
    memory_collection: Optional[str] = None
    memory_reason: Optional[str] = None
    # Memory expansion (temporal context around a doc)
    wants_memory_expand: bool = False
    expand_memory_id: Optional[str] = None
    expand_window: int = 3
    expand_collection: Optional[str] = None
    expand_reason: Optional[str] = None
    # File read (approved-folder file access)
    wants_file_read: bool = False
    file_read_path: Optional[str] = None
    file_read_start_line: Optional[int] = None
    file_read_end_line: Optional[int] = None
    file_read_reason: Optional[str] = None
    # File grep (approved-folder search)
    wants_file_grep: bool = False
    file_grep_pattern: Optional[str] = None
    file_grep_folder: Optional[str] = None
    file_grep_glob: Optional[str] = None
    file_grep_reason: Optional[str] = None
    # File list (approved-folder directory listing)
    wants_file_list: bool = False
    file_list_path: Optional[str] = None
    file_list_recursive: bool = False
    file_list_reason: Optional[str] = None
    # Git stats (local repo activity queries)
    wants_git_stats: bool = False
    git_stats_query: Optional[str] = None
    git_stats_reason: Optional[str] = None
    # Full document retrieval (all chunks of an uploaded document)
    wants_full_document: bool = False
    full_document_title: Optional[str] = None
    full_document_reason: Optional[str] = None
    # Visual memory recall (CLIP image search)
    wants_recall_image: bool = False
    recall_image_query: Optional[str] = None
    recall_image_reason: Optional[str] = None
    # URL fetching (direct page content retrieval)
    wants_fetch_url: bool = False
    fetch_url: Optional[str] = None
    fetch_url_reason: Optional[str] = None
    # Stack Exchange search
    wants_stackexchange: bool = False
    stackexchange_query: Optional[str] = None
    stackexchange_site: str = "stackoverflow"
    stackexchange_reason: Optional[str] = None
    # arXiv paper search
    wants_arxiv: bool = False
    arxiv_query: Optional[str] = None
    arxiv_reason: Optional[str] = None
    # PubMed biomedical literature search
    wants_pubmed: bool = False
    pubmed_query: Optional[str] = None
    pubmed_reason: Optional[str] = None
    # Hacker News search
    wants_hackernews: bool = False
    hackernews_query: Optional[str] = None
    hackernews_reason: Optional[str] = None
    # GitHub API (read-only: issues, PRs, actions, releases, search, etc.)
    wants_github: bool = False
    github_query: Optional[str] = None
    github_reason: Optional[str] = None
    # Document generation (research & save markdown document)
    wants_generate_document: bool = False
    generate_document_topic: Optional[str] = None
    generate_document_type: Optional[str] = None  # "report" | "summary"
    generate_document_focus: Optional[str] = None
    generate_document_reason: Optional[str] = None
    # Daemon self-note (save working context for future sessions)
    wants_create_daemon_note: bool = False
    daemon_note_title: Optional[str] = None
    daemon_note_category: Optional[str] = None  # implementation | architecture | research | decisions
    daemon_note_summary: Optional[str] = None
    daemon_note_reason: Optional[str] = None
    # Completion
    is_done: bool = False
    done_reason: Optional[str] = None
    wants_answer: bool = False
    partial_response: Optional[str] = None


@dataclass
class AgenticSearchSession:
    """
    Complete session state for an agentic search interaction.

    Tracks the full lifecycle of a multi-round search session including
    all search rounds, accumulated context, and final output.
    """
    query: str
    state: AgentState = AgentState.IDLE
    rounds: List[SearchRound] = field(default_factory=list)
    accumulated_context: str = ""
    total_tokens_used: int = 0
    max_rounds: int = 5
    protocol: SearchProtocol = SearchProtocol.XML_MARKERS

    # LLM signals
    model_signaled_done: bool = False
    done_reason: Optional[str] = None

    # Final output
    final_response: Optional[str] = None

    # Query relaxation tracking
    low_quality_search_count: int = 0
    relaxation_hint: Optional[str] = None

    # Context awareness tracking
    memory_search_counts: Dict[str, int] = field(default_factory=dict)
    expand_count: int = 0
    context_inventory: str = ""

    # Provenance
    final_prompt_hash: str = ""

    # Metadata
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def current_round(self) -> int:
        """Get the current round number (1-indexed)."""
        return len(self.rounds) + 1

    @property
    def can_continue(self) -> bool:
        """Check if the session can continue with more search rounds."""
        return (
            not self.model_signaled_done and
            self.current_round <= self.max_rounds and
            self.state not in (AgentState.DONE, AgentState.ERROR)
        )

    @property
    def total_duration_ms(self) -> float:
        """Get total session duration in milliseconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() * 1000

    @property
    def total_search_duration_ms(self) -> float:
        """Get total time spent searching across all rounds."""
        return sum(r.duration_ms for r in self.rounds)

    def get_provenance_summary(self) -> Dict[str, Any]:
        """Return a serializable provenance dict for audit logging."""
        agentic_rounds = []
        for r in self.rounds:
            rd: Dict[str, Any] = {
                "round": r.round_number,
                "duration_ms": r.duration_ms,
            }
            if r.request:
                query = r.request.query or ""
                rd["action"] = self._classify_round_action(query)
                rd["query"] = query[:120]
                rd["reason"] = r.request.reason[:120] if r.request.reason else ""
            if r.error:
                rd["error"] = str(r.error)[:200]
            agentic_rounds.append(rd)
        return {
            "total_rounds": len(self.rounds),
            "protocol": self.protocol.value if self.protocol else "",
            "agentic_rounds": agentic_rounds,
            "model_signaled_done": self.model_signaled_done,
            "done_reason": self.done_reason or "",
            "context_inventory": self.context_inventory[:500] if self.context_inventory else "",
            "memory_search_counts": dict(self.memory_search_counts) if self.memory_search_counts else {},
            "expand_count": self.expand_count,
            "final_prompt_hash": self.final_prompt_hash,
            "total_duration_ms": self.total_duration_ms,
        }

    @staticmethod
    def _classify_round_action(query: str) -> str:
        """Classify the action type from the SearchRequest query prefix."""
        if query.startswith("[Memory:"):
            return "memory_search"
        if query.startswith("[Python:"):
            return "sandbox"
        if query.startswith("[File Read]"):
            return "file_read"
        if query.startswith("[File Grep]"):
            return "file_grep"
        if query.startswith("[File List]"):
            return "file_list"
        if query.startswith("[Expand Memory]"):
            return "expand_memory"
        if query.startswith("[Git Stats]"):
            return "git_stats"
        if query.startswith("[Full Document]"):
            return "full_document"
        if query.startswith("[Fetch URL]"):
            return "fetch_url"
        if query.startswith("[GitHub]"):
            return "github"
        if query.startswith("[Generate Document]"):
            return "generate_document"
        if query.startswith("[Self-Note]"):
            return "daemon_self_note"
        return "web_search"


@dataclass
class ProgressEvent:
    """
    Progress update for UI display.

    Event types:
    - "searching": Starting a search
    - "found_results": Search completed with results
    - "synthesizing": Generating final answer
    - "done": Session complete
    - "error": An error occurred
    """
    event_type: str
    message: str
    round_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


# Tool definitions for native protocol (OpenAI/Anthropic format)
SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use when you need "
            "real-time data, recent events, or information that may have "
            "changed since your training cutoff. "
            "Use 'site' to target specific domains for better results "
            "(e.g. 'stackoverflow.com' for code questions, 'reddit.com' for opinions)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and include relevant context."
                },
                "site": {
                    "type": "string",
                    "description": (
                        "Optional: restrict to a specific domain. "
                        "Examples: 'stackoverflow.com', 'reddit.com', 'github.com', "
                        "'news.ycombinator.com', 'arxiv.org', 'pubmed.ncbi.nlm.nih.gov'."
                    )
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this search is needed."
                }
            },
            "required": ["query"]
        }
    }
}

DONE_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "done_searching",
        "description": (
            "Signal that you have gathered enough information from web searches "
            "and are ready to provide a comprehensive final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why you have enough information to answer."
                }
            },
            "required": []
        }
    }
}

WOLFRAM_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "wolfram_alpha",
        "description": (
            "Compute mathematical expressions, solve equations, perform calculus, "
            "get scientific data, handle unit conversions, statistical analysis, "
            "and any query requiring numerical computation or data lookup. "
            "Examples: 'solve x^2 + 5x - 6 = 0', 'integrate sin(x) from 0 to pi', "
            "'convert 100 miles to kilometers', 'population of France', "
            "'plot sin(x) * exp(-x/10) from 0 to 20'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Mathematical, scientific, or data query to compute"
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this computation is needed."
                }
            },
            "required": ["query"]
        }
    }
}

SANDBOX_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "Execute Python code in a secure sandbox environment. "
            "Use for: multi-step calculations, data processing with pandas/numpy, "
            "creating visualizations with matplotlib, symbolic math with sympy, "
            "any computation requiring multiple lines of code or intermediate results. "
            "Pre-installed packages: numpy, pandas, matplotlib, scipy, sympy, scikit-learn, "
            "requests, beautifulsoup4. "
            "IMPORTANT: Variables persist within this conversation - you can define a variable "
            "in one execution and use it in the next."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use print() to display results."
                },
                "purpose": {
                    "type": "string",
                    "description": "Brief explanation of what this code does (for logging/context)"
                }
            },
            "required": ["code"]
        }
    }
}

MEMORY_SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_memory",
        "description": (
            "Search your own memory and knowledge base. Collections and what they contain:\n"
            "- summaries: Rich narrative session summaries — best for profile overviews, biographical questions, 'tell me about' queries\n"
            "- conversations: Raw past conversation turns — best for 'did we discuss', temporal recall, specific exchanges\n"
            "- facts: Individual extracted triples (e.g. name=Luke, age=33) — best for specific single facts\n"
            "- reflections: End-of-session reflections and insights\n"
            "- reference_docs: Your own architecture/documentation\n"
            "- obsidian_notes: User's personal Obsidian vault notes\n"
            "- wiki_knowledge: Pre-embedded Wikipedia articles — best for factual/encyclopedic questions about real-world topics\n"
            "- procedural: Git commit history and how-to knowledge\n"
            "- procedural_skills: Learned reusable problem-solving patterns\n"
            "IMPORTANT: Diversify across collections — avoid searching the same collection repeatedly. "
            "For profile/biographical questions, prefer summaries and conversations over facts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific about what you're looking for."
                },
                "collection": {
                    "type": "string",
                    "description": "Which memory collection to search.",
                    "enum": [
                        "reference_docs", "facts", "conversations",
                        "summaries", "reflections", "obsidian_notes",
                        "wiki_knowledge", "procedural", "procedural_skills"
                    ]
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this memory search is needed."
                }
            },
            "required": ["query", "collection"]
        }
    }
}

EXPAND_MEMORY_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "expand_memory",
        "description": (
            "Expand a memory hit to see its surrounding context (neighboring turns). "
            "Use AFTER search_memory when a result looks relevant but you need more "
            "context to understand it fully (e.g., what was said before/after). "
            "Provide the FULL doc ID from search results (shown in parentheses as 'id: ...'). "
            "For summaries, this retrieves the original conversations that were compressed into the summary. "
            "IMPORTANT: Do NOT loop — one expand per memory is enough. "
            "Do NOT expand memories you have already expanded."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The full document ID to expand (copy the complete ID from search results)"
                },
                "collection": {
                    "type": "string",
                    "description": "The collection this memory belongs to (from search results header).",
                    "enum": [
                        "conversations", "summaries", "reflections",
                        "facts", "obsidian_notes"
                    ]
                },
                "window": {
                    "type": "integer",
                    "description": "Number of neighbors on each side (1-5, default 3)"
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why expansion is needed."
                }
            },
            "required": ["memory_id"]
        }
    }
}

FILE_READ_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "file_read",
        "description": (
            "Read the contents of a file from an approved directory. "
            "Use for viewing source code, configs, logs, notes, or any text file on disk. "
            "Supports optional line range to read specific sections of large files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the file to read (relative or absolute)"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional: start reading from this line (1-indexed)"
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional: stop reading at this line"
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this file is needed."
                }
            },
            "required": ["filepath"]
        }
    }
}

FILE_GREP_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "file_grep",
        "description": (
            "Search for a text pattern across files in approved directories. "
            "Returns matching lines with surrounding context. "
            "Use for finding where something is defined, used, or referenced in the codebase."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex supported)"
                },
                "folder": {
                    "type": "string",
                    "description": "Optional: limit search to this folder"
                },
                "file_glob": {
                    "type": "string",
                    "description": "File pattern to search, e.g. '*.py', '*.md'. Default: '*'"
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case-sensitive search. Default: false"
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this search is needed."
                }
            },
            "required": ["pattern"]
        }
    }
}

FILE_LIST_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "file_list",
        "description": (
            "List files in an approved directory. "
            "Use to orient yourself before reading or grepping — "
            "see what files exist, their sizes, and directory structure."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dirpath": {
                    "type": "string",
                    "description": "Directory path to list"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Include subdirectories. Default: false"
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why listing this directory."
                }
            },
            "required": ["dirpath"]
        }
    }
}

GET_FULL_DOCUMENT_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_full_document",
        "description": (
            "Retrieve the COMPLETE text of a previously uploaded document by its title. "
            "Use when a search_memory hit from reference_docs shows a document is relevant "
            "but you only see a fragment. This fetches ALL chunks reassembled in order. "
            "The title must match exactly (use the title shown in search results metadata)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Exact document title (as shown in search result metadata)"
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why the full document is needed."
                }
            },
            "required": ["title"]
        }
    }
}

GIT_STATS_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "git_stats",
        "description": (
            "Query the local git repository for activity statistics. "
            "Use for: commit counts, recent commits, contributors, files changed, "
            "branch info, diff stats, and other temporal git questions. "
            "Read-only — never modifies the repository."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language description of what git stats to look up, "
                        "e.g. 'commits this week', 'files changed today', "
                        "'top contributors this month'"
                    )
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this git query is needed."
                }
            },
            "required": ["query"]
        }
    }
}

RECALL_IMAGE_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "recall_image",
        "description": (
            "Search visual memory for images matching a description. "
            "Returns relevant images from uploaded photos, Obsidian note diagrams, "
            "and other personal images. Use when the user asks about visual content, "
            "photos, diagrams, people, pets, or when visual context would help."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language description of the image to find, "
                        "e.g. 'my cat', 'the math diagram', 'family photo'"
                    )
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why visual recall is needed."
                }
            },
            "required": ["query"]
        }
    }
}

FETCH_URL_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": (
            "Fetch the content of a web page by its URL. "
            "Use when the user provides a specific URL to visit, or when you need "
            "to read the actual content of a page (e.g., a GitHub repo, article, docs). "
            "Web search finds pages; fetch_url reads them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch (must start with http:// or https://)"
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this URL needs to be fetched."
                }
            },
            "required": ["url"]
        }
    }
}

STACKEXCHANGE_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_stackexchange",
        "description": (
            "Search Stack Exchange for technical Q&A. Returns top-voted answers "
            "with accepted-answer flags. Best for programming, sysadmin, math, "
            "and science questions. Default site is stackoverflow."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Technical question or keywords to search."
                },
                "site": {
                    "type": "string",
                    "description": "Stack Exchange site (default: stackoverflow). Others: serverfault, superuser, math, stats, unix, askubuntu."
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this search is needed."
                }
            },
            "required": ["query"]
        }
    }
}

ARXIV_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_arxiv",
        "description": (
            "Search arXiv for academic papers. Returns titles, authors, "
            "abstracts, and links. Use for research questions, ML/AI papers, "
            "physics, math, CS, statistics, and quantitative biology."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Academic search query — paper titles, topics, or author names."
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this search is needed."
                }
            },
            "required": ["query"]
        }
    }
}

PUBMED_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_pubmed",
        "description": (
            "Search PubMed for biomedical and life science literature. "
            "Returns paper titles, authors, abstracts, and PMIDs. "
            "Use for medical, health, biology, and clinical research questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Biomedical search query — conditions, treatments, genes, etc."
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this search is needed."
                }
            },
            "required": ["query"]
        }
    }
}

HACKERNEWS_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_hackernews",
        "description": (
            "Search Hacker News for tech discussions, startup news, and developer opinions. "
            "Returns story titles, points, comment counts, and URLs. "
            "Best for tech industry trends, developer tools, and community opinions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for Hacker News stories and discussions."
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this search is needed."
                }
            },
            "required": ["query"]
        }
    }
}

GITHUB_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "github",
        "description": (
            "Query the GitHub repository for issues, pull requests, actions, "
            "releases, workflows, labels, milestones, contributors, and code search. "
            "Read-only — never modifies the repository. "
            "Use natural language: 'open issues labeled bug', 'PR #42', "
            "'failed CI runs', 'latest release', 'search code for TODO', etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language description of what to look up on GitHub, "
                        "e.g. 'open issues', 'PR #12 diff', 'recent workflow runs', "
                        "'closed issues labeled enhancement', 'contributors', "
                        "'search code for authenticate'"
                    )
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this GitHub query is needed."
                }
            },
            "required": ["query"]
        }
    }
}

GENERATE_DOCUMENT_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "generate_document",
        "description": (
            "Research a topic and save a structured markdown document (report or summary). "
            "Gathers sources from web search, memory, and notes, then drafts a document "
            "with inline citations, YAML frontmatter, and a Sources section. "
            "Saves to documents/ with versioned filenames."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to research and write about."
                },
                "doc_type": {
                    "type": "string",
                    "enum": ["report", "summary"],
                    "description": "Type of document: 'report' (multi-section) or 'summary' (concise)."
                },
                "focus": {
                    "type": "string",
                    "description": "Optional narrowing focus, e.g. 'economic impact' or 'recent developments'."
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this document is needed."
                }
            },
            "required": ["topic", "doc_type"]
        }
    }
}

CREATE_DAEMON_NOTE_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "create_daemon_note",
        "description": (
            "Save a structured self-note for future Daemon sessions. "
            "Use this when you discover something worth remembering: implementation decisions, "
            "architectural patterns, gotchas, or important context about the user's project. "
            "Notes are saved to disk and embedded for future retrieval. "
            "They are NOT user-facing facts — they are your own working context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short descriptive title for the note (3-100 chars)."
                },
                "category": {
                    "type": "string",
                    "enum": ["implementation", "architecture", "research", "decisions"],
                    "description": "Category: implementation (how things work), architecture (design patterns), research (findings), decisions (choices made)."
                },
                "summary": {
                    "type": "string",
                    "description": "2-5 sentence summary of the key information to remember."
                },
                "reason": {
                    "type": "string",
                    "description": "Why this is worth noting for future sessions."
                }
            },
            "required": ["title", "category", "summary"]
        }
    }
}

# System prompt injection for local models
AGENTIC_SYSTEM_PROMPT_INJECTION = """
[AGENTIC TOOLS ENABLED]
You have access to web search, Wolfram Alpha, and Python code execution. Use these tools to gather information:

**Available Tools:**
1. **Web Search**: <search>your query</search>
   Use for: current events, explanations, general knowledge, how-to guides, opinions.

2. **Wolfram Alpha**: <wolfram>your query</wolfram>
   Use for: quick math calculations, equations, unit conversions, scientific data, statistics.
   Best for one-shot computations where you just need a numerical answer.
   Examples: "solve x^2 - 4 = 0", "integrate x^2 dx", "convert 50 mph to km/h", "GDP of Germany"

3. **Python Code Execution**: <python purpose="brief description">your code</python>
   Use for: multi-step calculations, data processing with pandas/numpy, creating visualizations,
   symbolic math with sympy, any logic requiring multiple lines of code.
   Pre-installed: numpy, pandas, matplotlib, scipy, sympy, scikit-learn, requests, bs4
   IMPORTANT: Variables persist across executions in this conversation!
   Example:
   <python purpose="analyze fibonacci sequence">
   def fib(n):
       a, b = 0, 1
       result = []
       for _ in range(n):
           result.append(a)
           a, b = b, a + b
       return result
   sequence = fib(20)
   print(f"First 20 Fibonacci: {{sequence}}")
   print(f"Mean: {{sum(sequence)/len(sequence):.2f}}")
   </python>

4. **Done**: <done/> - Signal you have enough information to answer.

5. **Memory Search**: <memory collection="collection_name">your query</memory>
   Use for: your own docs (reference_docs), user facts (facts), past conversations (conversations),
   summaries (summaries), reflections (reflections), Obsidian notes (obsidian_notes),
   Wikipedia articles (wiki_knowledge), git history (procedural), learned skills (procedural_skills).
   Prefer this over web search when the answer is likely already in your memory.

6. **File Read**: <file_read path="filepath">optional reason</file_read>
   Read the contents of a file from disk. Use for viewing source code, configs, logs, notes.
   Example: <file_read path="core/orchestrator.py">checking main request handler</file_read>

7. **File Grep**: <file_grep pattern="search_pattern" glob="*.py">optional folder</file_grep>
   Search for a text pattern across files. Returns matching lines with context.
   Example: <file_grep pattern="def generate_response" glob="*.py">core/</file_grep>

8. **File List**: <file_list path="directory" recursive="false"/>
   List files in a directory. Use to orient before reading or grepping.
   Example: <file_list path="core/agentic/" recursive="false"/>

9. **Expand Memory**: <expand_memory id="full_doc_id" collection="conversations" window="3">reason</expand_memory>
   After a search_memory result looks relevant, expand it to see surrounding turns/source conversations.
   Copy the FULL doc ID from search results. Do NOT loop — one expand per memory is enough.
   Example: <expand_memory id="a1b2c3d4-e5f6-7890-abcd-ef1234567890" collection="conversations" window="3">need full conversation context</expand_memory>

10. **Git Stats**: <git_stats>your query</git_stats>
   Use for: commit counts, recent commits, contributors, files changed, branch activity, diff stats.
   Supports time ranges: "this week", "today", "last 30 days", "this month", etc.
   Example: <git_stats>how many commits this week</git_stats>

11. **Get Full Document**: <get_full_document title="exact document title">reason</get_full_document>
    Retrieve the complete text of an uploaded document by its exact title.
    Use after search_memory returns a fragment from reference_docs and you need the whole document.
    Example: <get_full_document title="upload:ISYE_6501_Syllabus.pdf">need full assignment schedule</get_full_document>

12. **Fetch URL**: <fetch_url url="https://example.com">reason</fetch_url>
    Fetch and read the content of a specific web page by URL.
    Use when the user provides a link, or when you need to read a page directly.
    Web search finds pages; fetch_url reads them.
    Example: <fetch_url url="https://github.com/user/repo">user asked me to check their repo</fetch_url>

13. **GitHub**: <github>your query</github>
    Query the GitHub repository for issues, PRs, actions, releases, workflows, labels,
    milestones, contributors, and code search. Read-only — never modifies the repository.
    Use natural language queries.
    Examples:
    <github>open issues labeled bug</github>
    <github>PR #42</github>
    <github>recent workflow runs</github>
    <github>search code for authenticate</github>
    <github>contributors</github>
    <github>closed PRs</github>

**Tool Selection Guidelines:**
| Task Type | Best Tool |
|-----------|-----------|
| Quick calculation (e.g., "what's 17% of 234?") | Wolfram |
| Unit conversion (e.g., "convert 5 miles to km") | Wolfram |
| Physical constants (e.g., "speed of light") | Wolfram |
| Multi-step math with intermediate values | Python |
| Data analysis, statistics | Python |
| Generate charts/visualizations | Python |
| Current events, factual lookup | Search |
| Internal docs, architecture questions | Memory (reference_docs) |
| User profile overview, "tell me about myself" | Memory (summaries, conversations) |
| Individual facts (name, age, specific detail) | Memory (facts) |
| Past conversations, "did we discuss" | Memory (conversations, summaries) |
| User's personal notes | Memory (obsidian_notes) |
| Encyclopedic/factual knowledge (history, science, etc.) | Memory (wiki_knowledge) |
| View source code, configs, logs | File Read |
| Find where something is defined/used | File Grep |
| Explore directory structure | File List |
| See context around a memory hit | Expand Memory |
| Git repository activity, commit counts | Git Stats |
| Full uploaded document (syllabus, PDF, etc.) | Get Full Document |
| Read a specific URL (GitHub, article, docs) | Fetch URL |
| GitHub issues, PRs, CI status, releases | GitHub |
| GitHub code search, contributors, labels | GitHub |

**IMPORTANT — fetch_url vs search:**
When you know a specific URL (from profile facts, memory, or the user's message), use fetch_url to read it directly.
Do NOT use search to find a page when you already have its URL. Search finds pages; fetch_url reads them.
Example: if user profile says github_url=https://github.com/user/repo, use <fetch_url url="https://github.com/user/repo">reading user's repo</fetch_url>

You can use tools up to {max_rounds} times total. Be specific with queries.
You may use multiple tools in a single response when the queries are independent (e.g., <search>...</search> and <memory>...</memory> together).

## Query Reformulation

When search returns empty, sparse, or irrelevant results:

1. **Identify the problem:**
   - Too specific? (exact product names, version numbers, niche jargon)
   - Too narrow? (combining multiple constraints)
   - Wrong framing? (searching for solution vs. searching for problem)

2. **Relaxation strategies:**
   - Remove version/date specifics: "Python 3.12 asyncio bug" → "Python asyncio bug"
   - Use category terms: "ChromaDB embedding error" → "vector database embedding error"
   - Split compound queries: "fast lightweight local LLM" → "lightweight local LLM"
   - Try synonyms: "timeout" → "deadline exceeded" or "connection failed"
   - Search for the error message directly if you have one

3. **Limits:**
   - Max 2 reformulation attempts per topic
   - After 2 attempts, answer with whatever information you have
   - Don't keep searching if results are consistently poor — synthesize and note uncertainty
""".strip()

# Query relaxation hint templates
LOW_QUALITY_HINT_TEMPLATE = """⚠️ Previous search for "{query}" returned {issue}. Consider:
- {suggestion}
- Relaxation attempts remaining: {remaining}/2"""

MAX_RELAXATION_HINT = """ℹ️ Search relaxation limit reached. Answer using:
- Information gathered so far
- Your training knowledge
- Explicit acknowledgment of gaps

Do not attempt further searches on this topic."""


@dataclass
class _ToolResult:
    """Internal: result from a single tool dispatch coroutine.

    Used by the parallel dispatch pattern in AgenticSearchController.
    Each _dispatch_* method returns one of these instead of yielding
    ProgressEvents directly, so the caller can gather() multiple
    dispatches and yield events in deterministic order afterward.
    """
    decision: SearchDecision
    round_data: Optional[SearchRound]  # None if tool was skipped/errored
    formatted_context: str             # Ready for _append_accumulated()
    start_events: List[ProgressEvent]  # "Searching for X...", etc.
    end_events: List[ProgressEvent]    # "Found N results", etc.
    memory_collection: Optional[str] = None  # For search count tracking
    is_expand: bool = False            # For expand_count tracking
