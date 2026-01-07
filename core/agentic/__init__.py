"""
Agentic Search Package

This package provides multi-round, LLM-driven web search capabilities.
The agentic search loop allows models to iteratively gather information
until they have enough to provide a comprehensive answer.

Contract:
    - Provides AgenticSearchController for orchestrating search loops
    - Supports both native tool calling (API models) and XML markers (local models)
    - Emits ProgressEvent for UI status updates
    - Enforces configurable round limits (default 5)
    - Compresses accumulated context to fit token budgets

Usage:
    from core.agentic import AgenticSearchController, ProgressEvent

    controller = AgenticSearchController(model_manager, web_search_manager)
    async for event in controller.run_agentic_search(query, ...):
        if isinstance(event, ProgressEvent):
            # Handle progress update
        else:
            # Handle response chunk
"""

from core.agentic.types import (
    AgentState,
    SearchProtocol,
    SearchRequest,
    SearchRound,
    SearchDecision,
    AgenticSearchSession,
    ProgressEvent,
    SEARCH_TOOL_DEFINITION,
    DONE_TOOL_DEFINITION,
)
from core.agentic.protocols import (
    detect_protocol,
    BaseProtocolHandler,
    NativeToolsHandler,
    XMLMarkerHandler,
)
from core.agentic.controller import AgenticSearchController

__all__ = [
    # Types
    "AgentState",
    "SearchProtocol",
    "SearchRequest",
    "SearchRound",
    "SearchDecision",
    "AgenticSearchSession",
    "ProgressEvent",
    "SEARCH_TOOL_DEFINITION",
    "DONE_TOOL_DEFINITION",
    # Protocols
    "detect_protocol",
    "BaseProtocolHandler",
    "NativeToolsHandler",
    "XMLMarkerHandler",
    # Controller
    "AgenticSearchController",
]
