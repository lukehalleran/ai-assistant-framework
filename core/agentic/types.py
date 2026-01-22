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
    - SEARCH_TOOL_DEFINITION, DONE_TOOL_DEFINITION (tool schemas)

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
    - is_done: Model signals it has enough information
    - wants_answer: Model wants to provide final answer (no explicit done signal)
    """
    wants_search: bool = False
    search_query: Optional[str] = None
    search_reason: Optional[str] = None
    wants_wolfram: bool = False
    wolfram_query: Optional[str] = None
    wolfram_reason: Optional[str] = None
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

# System prompt injection for local models
AGENTIC_SYSTEM_PROMPT_INJECTION = """
[AGENTIC TOOLS ENABLED]
You have access to web search and Wolfram Alpha. Use these tools to gather information:

**Available Tools:**
1. **Web Search**: <search>your query</search>
   Use for: current events, explanations, general knowledge, how-to guides, opinions.

2. **Wolfram Alpha**: <wolfram>your query</wolfram>
   Use for: math calculations, equations, unit conversions, scientific data, statistics.
   Examples: "solve x^2 - 4 = 0", "integrate x^2 dx", "convert 50 mph to km/h", "GDP of Germany"

3. **Done**: <done/> - Signal you have enough information to answer.

**Tool Selection:**
- Math/calculations/conversions/scientific data → use <wolfram>
- Current events/explanations/general knowledge → use <search>
- Complex queries may need both tools in sequence

You can use tools up to {max_rounds} times total. Be specific with queries.
""".strip()
