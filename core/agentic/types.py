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
    - ProgressEvent (UI update events)
    - SEARCH_TOOL_DEFINITION, DONE_TOOL_DEFINITION, WOLFRAM_TOOL_DEFINITION, SANDBOX_TOOL_DEFINITION, MEMORY_SEARCH_TOOL_DEFINITION (tool schemas)

SearchDecision Fields (extended for multi-tool support):
    - wants_search, search_query, search_reason (web search)
    - wants_wolfram, wolfram_query, wolfram_reason (Wolfram Alpha computation)
    - wants_sandbox, sandbox_code, sandbox_purpose (E2B Python sandbox) [NEW 2026-01-22]
    - wants_memory_search, memory_query, memory_collection, memory_reason (ChromaDB memory search)
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

    One of these should be True:
    - wants_search: Model wants to perform a web search
    - wants_wolfram: Model wants to perform a Wolfram Alpha computation
    - wants_sandbox: Model wants to execute Python code in sandbox
    - is_done: Model signals it has enough information
    - wants_answer: Model wants to provide final answer (no explicit done signal)
    """
    # Web search
    wants_search: bool = False
    search_query: Optional[str] = None
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
    context_inventory: str = ""

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
            "changed since your training cutoff."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and include relevant context."
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
                        "procedural", "procedural_skills"
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
   git history (procedural), learned skills (procedural_skills).
   Prefer this over web search when the answer is likely already in your memory.

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

You can use tools up to {max_rounds} times total. Be specific with queries.

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
