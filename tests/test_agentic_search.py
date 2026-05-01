"""
Tests for Agentic Search System

Tests cover:
- Type definitions and data structures
- Protocol handlers (native tools and XML markers)
- Controller logic and flow
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# Test Types Module
# =============================================================================

class TestAgentState:
    """Tests for AgentState enum."""

    def test_agent_states_exist(self):
        """Verify all expected states are defined."""
        from core.agentic.types import AgentState

        assert AgentState.IDLE.value == "idle"
        assert AgentState.THINKING.value == "thinking"
        assert AgentState.SEARCHING.value == "searching"
        assert AgentState.OBSERVING.value == "observing"
        assert AgentState.GENERATING.value == "generating"
        assert AgentState.DONE.value == "done"
        assert AgentState.ERROR.value == "error"


class TestSearchProtocol:
    """Tests for SearchProtocol enum."""

    def test_protocol_types(self):
        """Verify protocol types are defined."""
        from core.agentic.types import SearchProtocol

        assert SearchProtocol.NATIVE_TOOLS.value == "native_tools"
        assert SearchProtocol.XML_MARKERS.value == "xml_markers"


class TestSearchRequest:
    """Tests for SearchRequest dataclass."""

    def test_search_request_defaults(self):
        """Test default values for SearchRequest."""
        from core.agentic.types import SearchRequest

        request = SearchRequest(query="test query")
        assert request.query == "test query"
        assert request.reason is None
        assert request.round_number == 1
        assert isinstance(request.timestamp, datetime)

    def test_search_request_with_values(self):
        """Test SearchRequest with all values specified."""
        from core.agentic.types import SearchRequest

        request = SearchRequest(
            query="test query",
            reason="need more info",
            round_number=3
        )
        assert request.query == "test query"
        assert request.reason == "need more info"
        assert request.round_number == 3


class TestSearchRound:
    """Tests for SearchRound dataclass."""

    def test_search_round_defaults(self):
        """Test default values for SearchRound."""
        from core.agentic.types import SearchRound, SearchRequest

        request = SearchRequest(query="test")
        round_data = SearchRound(round_number=1, request=request)

        assert round_data.round_number == 1
        assert round_data.request.query == "test"
        assert round_data.results is None
        assert round_data.summary is None
        assert round_data.duration_ms == 0.0
        assert round_data.error is None


class TestSearchDecision:
    """Tests for SearchDecision dataclass."""

    def test_search_decision_defaults(self):
        """Test default values for SearchDecision."""
        from core.agentic.types import SearchDecision

        decision = SearchDecision()
        assert decision.wants_search is False
        assert decision.search_query is None
        assert decision.is_done is False
        assert decision.wants_answer is False

    def test_search_decision_wants_search(self):
        """Test SearchDecision for a search request."""
        from core.agentic.types import SearchDecision

        decision = SearchDecision(
            wants_search=True,
            search_query="latest news",
            search_reason="need current info"
        )
        assert decision.wants_search is True
        assert decision.search_query == "latest news"
        assert decision.search_reason == "need current info"

    def test_search_decision_is_done(self):
        """Test SearchDecision for done signal."""
        from core.agentic.types import SearchDecision

        decision = SearchDecision(
            is_done=True,
            done_reason="have enough info"
        )
        assert decision.is_done is True
        assert decision.done_reason == "have enough info"


class TestAgenticSearchSession:
    """Tests for AgenticSearchSession dataclass."""

    def test_session_defaults(self):
        """Test default session state."""
        from core.agentic.types import AgenticSearchSession, AgentState, SearchProtocol

        session = AgenticSearchSession(query="test query")

        assert session.query == "test query"
        assert session.state == AgentState.IDLE
        assert session.rounds == []
        assert session.accumulated_context == ""
        assert session.max_rounds == 5
        assert session.protocol == SearchProtocol.XML_MARKERS
        assert session.model_signaled_done is False
        assert session.final_response is None

    def test_session_current_round(self):
        """Test current_round property."""
        from core.agentic.types import AgenticSearchSession, SearchRound, SearchRequest

        session = AgenticSearchSession(query="test")
        assert session.current_round == 1

        # Add a round
        session.rounds.append(SearchRound(
            round_number=1,
            request=SearchRequest(query="search 1")
        ))
        assert session.current_round == 2

    def test_session_can_continue(self):
        """Test can_continue property."""
        from core.agentic.types import AgenticSearchSession, AgentState

        session = AgenticSearchSession(query="test", max_rounds=3)

        # Initial state - can continue
        assert session.can_continue is True

        # Model signals done - cannot continue
        session.model_signaled_done = True
        assert session.can_continue is False

        # Reset and test state blocking
        session.model_signaled_done = False
        session.state = AgentState.DONE
        assert session.can_continue is False

        # Reset and test max rounds
        session.state = AgentState.IDLE
        from core.agentic.types import SearchRound, SearchRequest
        for i in range(3):
            session.rounds.append(SearchRound(
                round_number=i + 1,
                request=SearchRequest(query=f"search {i}")
            ))
        assert session.can_continue is False  # current_round is 4, max is 3


class TestProgressEvent:
    """Tests for ProgressEvent dataclass."""

    def test_progress_event_defaults(self):
        """Test default ProgressEvent values."""
        from core.agentic.types import ProgressEvent

        event = ProgressEvent(
            event_type="searching",
            message="Searching for: test query"
        )

        assert event.event_type == "searching"
        assert event.message == "Searching for: test query"
        assert event.round_number == 0
        assert event.metadata == {}
        assert isinstance(event.timestamp, datetime)

    def test_progress_event_with_metadata(self):
        """Test ProgressEvent with metadata."""
        from core.agentic.types import ProgressEvent

        event = ProgressEvent(
            event_type="found_results",
            message="Found 5 results",
            round_number=2,
            metadata={"result_count": 5, "cached": True}
        )

        assert event.event_type == "found_results"
        assert event.round_number == 2
        assert event.metadata["result_count"] == 5
        assert event.metadata["cached"] is True


class TestToolDefinitions:
    """Tests for tool definition constants."""

    def test_search_tool_definition(self):
        """Test SEARCH_TOOL_DEFINITION structure."""
        from core.agentic.types import SEARCH_TOOL_DEFINITION

        assert SEARCH_TOOL_DEFINITION["type"] == "function"
        assert SEARCH_TOOL_DEFINITION["function"]["name"] == "web_search"
        assert "query" in SEARCH_TOOL_DEFINITION["function"]["parameters"]["properties"]

    def test_done_tool_definition(self):
        """Test DONE_TOOL_DEFINITION structure."""
        from core.agentic.types import DONE_TOOL_DEFINITION

        assert DONE_TOOL_DEFINITION["type"] == "function"
        assert DONE_TOOL_DEFINITION["function"]["name"] == "done_searching"


# =============================================================================
# Test Protocols Module
# =============================================================================

class TestDetectProtocol:
    """Tests for detect_protocol function."""

    def test_detect_native_tools_gpt4(self):
        """GPT-4 models should use native tools."""
        from core.agentic.protocols import detect_protocol, SearchProtocol

        assert detect_protocol("gpt-4o") == SearchProtocol.NATIVE_TOOLS
        assert detect_protocol("gpt-4-turbo") == SearchProtocol.NATIVE_TOOLS

    def test_detect_native_tools_claude(self):
        """Claude models should use native tools."""
        from core.agentic.protocols import detect_protocol, SearchProtocol

        assert detect_protocol("claude-3-opus") == SearchProtocol.NATIVE_TOOLS
        assert detect_protocol("claude-sonnet") == SearchProtocol.NATIVE_TOOLS

    def test_detect_xml_markers_local(self):
        """Unknown/local models should use XML markers."""
        from core.agentic.protocols import detect_protocol, SearchProtocol

        assert detect_protocol("llama-3") == SearchProtocol.XML_MARKERS
        assert detect_protocol("mistral-7b") == SearchProtocol.XML_MARKERS
        assert detect_protocol("unknown-model") == SearchProtocol.XML_MARKERS

    def test_detect_with_api_models_mapping(self):
        """Test with api_models alias mapping."""
        from core.agentic.protocols import detect_protocol, SearchProtocol

        api_models = {
            "my-gpt": "openai/gpt-4o-mini",
            "my-local": "local/llama-3"
        }

        assert detect_protocol("my-gpt", api_models) == SearchProtocol.NATIVE_TOOLS
        assert detect_protocol("my-local", api_models) == SearchProtocol.XML_MARKERS


class TestXMLMarkerHandler:
    """Tests for XMLMarkerHandler."""

    def test_parse_search_marker(self):
        """Test parsing search marker from response."""
        from core.agentic.protocols import XMLMarkerHandler

        handler = XMLMarkerHandler()
        response = "I need more information. <search>latest SpaceX news</search>"
        decision = handler.parse_response(response)

        assert decision.wants_search is True
        assert decision.search_query == "latest SpaceX news"

    def test_parse_done_marker(self):
        """Test parsing done marker from response."""
        from core.agentic.protocols import XMLMarkerHandler

        handler = XMLMarkerHandler()
        response = "I have enough information now. <done/>"
        decision = handler.parse_response(response)

        assert decision.is_done is True

    def test_parse_done_marker_with_space(self):
        """Test parsing done marker with space."""
        from core.agentic.protocols import XMLMarkerHandler

        handler = XMLMarkerHandler()
        response = "Done searching. <done />"
        decision = handler.parse_response(response)

        assert decision.is_done is True

    def test_parse_no_markers(self):
        """Test parsing response without markers."""
        from core.agentic.protocols import XMLMarkerHandler

        handler = XMLMarkerHandler()
        response = "Here is the answer to your question..."
        decision = handler.parse_response(response)

        assert decision.wants_answer is True
        assert decision.partial_response == response

    def test_parse_empty_response(self):
        """Test parsing empty response."""
        from core.agentic.protocols import XMLMarkerHandler

        handler = XMLMarkerHandler()
        decision = handler.parse_response("")

        assert decision.wants_answer is True

    def test_parse_multiline_search(self):
        """Test parsing multiline search query."""
        from core.agentic.protocols import XMLMarkerHandler

        handler = XMLMarkerHandler()
        response = "Let me search for that.\n<search>\nSpaceX Starship\nlaunch date 2026\n</search>"
        decision = handler.parse_response(response)

        assert decision.wants_search is True
        assert "SpaceX Starship" in decision.search_query

    def test_get_tools_returns_none(self):
        """XML handler should not return tools."""
        from core.agentic.protocols import XMLMarkerHandler

        handler = XMLMarkerHandler()
        assert handler.get_tools() is None

    def test_augment_system_prompt(self):
        """Test system prompt augmentation."""
        from core.agentic.protocols import XMLMarkerHandler

        handler = XMLMarkerHandler()
        original = "You are a helpful assistant."
        augmented = handler.augment_system_prompt(original, max_rounds=5)

        assert original in augmented
        assert "<search>" in augmented
        assert "<done/>" in augmented
        assert "5" in augmented  # max_rounds


class TestNativeToolsHandler:
    """Tests for NativeToolsHandler."""

    def test_parse_web_search_tool_call(self):
        """Test parsing web_search tool call."""
        from core.agentic.protocols import NativeToolsHandler

        handler = NativeToolsHandler()

        # Mock OpenAI-style response
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "web_search"
        mock_tool_call.function.arguments = '{"query": "latest AI news", "reason": "need current info"}'
        mock_response.tool_calls = [mock_tool_call]

        decision = handler.parse_response(mock_response)

        assert decision.wants_search is True
        assert decision.search_query == "latest AI news"
        assert decision.search_reason == "need current info"

    def test_parse_done_tool_call(self):
        """Test parsing done_searching tool call."""
        from core.agentic.protocols import NativeToolsHandler

        handler = NativeToolsHandler()

        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "done_searching"
        mock_tool_call.function.arguments = '{"reason": "have enough info"}'
        mock_response.tool_calls = [mock_tool_call]

        decision = handler.parse_response(mock_response)

        assert decision.is_done is True
        assert decision.done_reason == "have enough info"

    def test_parse_no_tool_calls(self):
        """Test parsing response without tool calls."""
        from core.agentic.protocols import NativeToolsHandler

        handler = NativeToolsHandler()

        mock_response = MagicMock()
        mock_response.tool_calls = None
        mock_response.content = "Here is the answer..."

        decision = handler.parse_response(mock_response)

        assert decision.wants_answer is True
        assert decision.partial_response == "Here is the answer..."

    def test_parse_dict_format(self):
        """Test parsing dict format response."""
        from core.agentic.protocols import NativeToolsHandler

        handler = NativeToolsHandler()

        response = {
            "tool_calls": [{
                "function": {
                    "name": "web_search",
                    "arguments": '{"query": "test query"}'
                }
            }]
        }

        decision = handler.parse_response(response)

        assert decision.wants_search is True
        assert decision.search_query == "test query"

    def test_get_tools(self):
        """Test that tools are returned."""
        from core.agentic.protocols import NativeToolsHandler

        handler = NativeToolsHandler()
        tools = handler.get_tools()

        assert len(tools) == 2
        assert tools[0]["function"]["name"] == "web_search"
        assert tools[1]["function"]["name"] == "done_searching"

    def test_augment_system_prompt(self):
        """Test system prompt augmentation for native tools."""
        from core.agentic.protocols import NativeToolsHandler

        handler = NativeToolsHandler()
        original = "You are a helpful assistant."
        augmented = handler.augment_system_prompt(original, max_rounds=5)

        assert original in augmented
        assert "AGENTIC TOOLS MODE" in augmented
        assert "5" in augmented


class TestGetProtocolHandler:
    """Tests for get_protocol_handler factory."""

    def test_get_native_handler(self):
        """Test getting native tools handler."""
        from core.agentic.protocols import get_protocol_handler, NativeToolsHandler
        from core.agentic.types import SearchProtocol

        handler = get_protocol_handler(SearchProtocol.NATIVE_TOOLS)
        assert isinstance(handler, NativeToolsHandler)

    def test_get_xml_handler(self):
        """Test getting XML marker handler."""
        from core.agentic.protocols import get_protocol_handler, XMLMarkerHandler
        from core.agentic.types import SearchProtocol

        handler = get_protocol_handler(SearchProtocol.XML_MARKERS)
        assert isinstance(handler, XMLMarkerHandler)


# =============================================================================
# Test Controller Module
# =============================================================================

class TestAgenticSearchController:
    """Tests for AgenticSearchController."""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock model manager."""
        manager = MagicMock()
        manager.api_models = {
            "gpt-4o": "openai/gpt-4o",
            "local-model": "local/llama"
        }
        manager.generate_once = AsyncMock(return_value="Test response")
        manager.generate_async = self._create_async_generator(["Chunk 1 ", "Chunk 2"])
        return manager

    @pytest.fixture
    def mock_web_search_manager(self):
        """Create a mock web search manager."""
        manager = MagicMock()

        # Mock search result
        mock_result = MagicMock()
        mock_result.pages = [
            MagicMock(title="Result 1", content="Content 1", url="http://example.com/1"),
            MagicMock(title="Result 2", content="Content 2", url="http://example.com/2"),
        ]
        mock_result.get_formatted_content = MagicMock(return_value="Formatted search results...")

        manager.search = AsyncMock(return_value=mock_result)
        manager.multi_search = AsyncMock(return_value=mock_result)
        return manager

    def _create_async_generator(self, items):
        """Create an async generator mock."""
        async def gen(*args, **kwargs):
            for item in items:
                yield item
        return gen

    def test_detect_protocol(self, mock_model_manager, mock_web_search_manager):
        """Test protocol detection in controller."""
        from core.agentic.controller import AgenticSearchController
        from core.agentic.types import SearchProtocol

        controller = AgenticSearchController(
            model_manager=mock_model_manager,
            web_search_manager=mock_web_search_manager
        )

        assert controller.detect_protocol("gpt-4o") == SearchProtocol.NATIVE_TOOLS
        assert controller.detect_protocol("local-model") == SearchProtocol.XML_MARKERS

    def test_controller_initialization(self, mock_model_manager, mock_web_search_manager):
        """Test controller initialization with default values."""
        from core.agentic.controller import AgenticSearchController

        controller = AgenticSearchController(
            model_manager=mock_model_manager,
            web_search_manager=mock_web_search_manager
        )

        assert controller.max_rounds == 5
        assert controller.context_budget_tokens == 8000
        assert controller.compression_model == "gpt-4o-mini"

    def test_controller_custom_config(self, mock_model_manager, mock_web_search_manager):
        """Test controller with custom configuration."""
        from core.agentic.controller import AgenticSearchController

        controller = AgenticSearchController(
            model_manager=mock_model_manager,
            web_search_manager=mock_web_search_manager,
            max_rounds=3,
            context_budget_tokens=4000,
            compression_model="gpt-4o"
        )

        assert controller.max_rounds == 3
        assert controller.context_budget_tokens == 4000
        assert controller.compression_model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_compress_results_short(self, mock_model_manager, mock_web_search_manager):
        """Test result compression for short content."""
        from core.agentic.controller import AgenticSearchController

        controller = AgenticSearchController(
            model_manager=mock_model_manager,
            web_search_manager=mock_web_search_manager
        )

        # Mock result with short content (pages need title/url/content for WEB_N formatting)
        mock_page = MagicMock()
        mock_page.title = "Test Page"
        mock_page.url = "https://example.com"
        mock_page.content = "Short content"
        mock_page.score = 0.9
        mock_result = MagicMock()
        mock_result.pages = [mock_page]

        compressed = await controller._compress_results(mock_result)
        assert "Short content" in compressed
        assert "[WEB_1]" in compressed

    @pytest.mark.asyncio
    async def test_compress_results_empty(self, mock_model_manager, mock_web_search_manager):
        """Test result compression for empty results."""
        from core.agentic.controller import AgenticSearchController

        controller = AgenticSearchController(
            model_manager=mock_model_manager,
            web_search_manager=mock_web_search_manager
        )

        # Mock empty result
        mock_result = MagicMock()
        mock_result.pages = []

        compressed = await controller._compress_results(mock_result)
        assert compressed == "No results found."


# =============================================================================
# Integration Tests
# =============================================================================

class TestAgenticSearchIntegration:
    """Integration tests for the agentic search system."""

    def test_package_imports(self):
        """Test that all public interfaces can be imported."""
        from core.agentic import (
            AgenticSearchController,
            AgentState,
            SearchProtocol,
            SearchRequest,
            SearchRound,
            SearchDecision,
            AgenticSearchSession,
            ProgressEvent,
            SEARCH_TOOL_DEFINITION,
            DONE_TOOL_DEFINITION,
            detect_protocol,
            BaseProtocolHandler,
            NativeToolsHandler,
            XMLMarkerHandler,
        )

        # All imports should succeed
        assert AgenticSearchController is not None
        assert AgentState is not None
        assert SearchProtocol is not None

    def test_end_to_end_session_creation(self):
        """Test creating a complete session flow."""
        from core.agentic.types import (
            AgenticSearchSession,
            AgentState,
            SearchProtocol,
            SearchRound,
            SearchRequest,
            ProgressEvent,
        )

        # Create session
        session = AgenticSearchSession(
            query="What's the latest SpaceX news?",
            max_rounds=3,
            protocol=SearchProtocol.NATIVE_TOOLS
        )

        # Simulate round 1
        session.state = AgentState.SEARCHING
        round1 = SearchRound(
            round_number=1,
            request=SearchRequest(query="SpaceX news 2026"),
            summary="SpaceX launched Starship successfully...",
            duration_ms=1500.0
        )
        session.rounds.append(round1)
        session.accumulated_context = round1.summary

        # Check state
        assert session.current_round == 2
        assert session.can_continue is True

        # Simulate model signaling done
        session.model_signaled_done = True
        session.state = AgentState.DONE

        assert session.can_continue is False
        assert len(session.rounds) == 1


# =============================================================================
# Test Context Inventory
# =============================================================================

class TestContextInventory:
    """Tests for _compute_context_inventory() method."""

    @pytest.fixture
    def controller(self):
        """Create a minimal controller for testing."""
        from core.agentic.controller import AgenticSearchController
        manager = MagicMock()
        manager.api_models = {}
        web_mgr = MagicMock()
        return AgenticSearchController(
            model_manager=manager,
            web_search_manager=web_mgr,
        )

    def test_full_context_inventory(self, controller):
        """Test inventory with a fully populated context dict."""
        context = {
            'user_profile': "identity:\n- name: Luke\n- age: 33\nhealth:\n- has condition X",
            'recent_summaries': [{'content': 'sum1'}, {'content': 'sum2'}],
            'semantic_summaries': [{'content': 'sem1'}],
            'recent_reflections': [{'content': 'ref1'}, {'content': 'ref2'}, {'content': 'ref3'}],
            'personal_notes': [{'content': 'note1'}],
            'memories': [{'content': 'm1'}, {'content': 'm2'}, {'content': 'm3'}],
            'recent_conversations': [{'query': 'q1'}, {'query': 'q2'}],
            'reference_docs': [{'content': 'doc1'}],
            'dreams': [{'content': 'dream1'}],
        }

        inventory = controller._compute_context_inventory(context)

        assert "Context already gathered by retrieval pipeline:" in inventory
        assert "[USER PROFILE]" in inventory
        assert "[RECENT SUMMARIES]: 2" in inventory
        assert "[SEMANTIC SUMMARIES]: 1" in inventory
        assert "[RECENT REFLECTIONS]: 3" in inventory
        assert "[PERSONAL NOTES]: 1" in inventory
        assert "[RELEVANT MEMORIES]: 3" in inventory
        assert "[RECENT CONVERSATIONS]: 2" in inventory
        assert "[DAEMON DOCUMENTATION]: 1" in inventory
        assert "[RECENT DREAMS]: 1" in inventory
        assert "Do NOT re-search" in inventory

    def test_empty_context_returns_empty_string(self, controller):
        """Test that None or empty context returns empty string."""
        assert controller._compute_context_inventory(None) == ""
        assert controller._compute_context_inventory({}) == ""

    def test_partial_context(self, controller):
        """Test with only some sections populated."""
        context = {
            'user_profile': "name: Luke",
            'memories': [{'content': 'm1'}],
        }

        inventory = controller._compute_context_inventory(context)

        assert "[USER PROFILE]" in inventory
        assert "[RELEVANT MEMORIES]: 1" in inventory
        # Sections not present should not appear
        assert "[RECENT SUMMARIES]" not in inventory
        assert "[PERSONAL NOTES]" not in inventory

    def test_inventory_conciseness(self, controller):
        """Test that inventory is concise (doesn't dump full content)."""
        context = {
            'user_profile': "A " * 5000,  # Very long profile
            'memories': [{'content': 'x' * 1000} for _ in range(20)],
        }

        inventory = controller._compute_context_inventory(context)

        # Should be much shorter than the full content
        assert len(inventory) < 500

    def test_dict_format_summaries_fallback(self, controller):
        """Test handling of old dict-format summaries."""
        context = {
            'summaries': {
                'recent': [{'content': 'r1'}, {'content': 'r2'}],
                'semantic': [{'content': 's1'}],
            }
        }

        inventory = controller._compute_context_inventory(context)

        assert "[RECENT SUMMARIES]: 2" in inventory
        assert "[SEMANTIC SUMMARIES]: 1" in inventory


# =============================================================================
# Test Memory Search Tracking
# =============================================================================

class TestMemorySearchTracking:
    """Tests for memory_search_counts tracking and diversity hints."""

    def test_memory_search_counts_default(self):
        """Test that memory_search_counts starts empty."""
        from core.agentic.types import AgenticSearchSession

        session = AgenticSearchSession(query="test")
        assert session.memory_search_counts == {}

    def test_memory_search_counts_increment(self):
        """Test manual increment of memory_search_counts."""
        from core.agentic.types import AgenticSearchSession

        session = AgenticSearchSession(query="test")

        # Simulate tracking
        collection = "facts"
        session.memory_search_counts[collection] = (
            session.memory_search_counts.get(collection, 0) + 1
        )
        assert session.memory_search_counts["facts"] == 1

        session.memory_search_counts[collection] = (
            session.memory_search_counts.get(collection, 0) + 1
        )
        assert session.memory_search_counts["facts"] == 2

    def test_diversity_hint_in_iteration_prompt(self):
        """Test that diversity hint appears after 2 searches of same collection."""
        from core.agentic.controller import AgenticSearchController
        from core.agentic.types import AgenticSearchSession

        manager = MagicMock()
        manager.api_models = {}
        controller = AgenticSearchController(
            model_manager=manager,
            web_search_manager=MagicMock(),
        )

        session = AgenticSearchSession(query="tell me about myself")
        session.memory_search_counts = {"facts": 2}

        prompt = controller._build_iteration_prompt(
            query="tell me about myself",
            search_context="[some results]",
            round_number=3,
            session=session,
        )

        assert "already searched 'facts' 2 times" in prompt
        assert "different collection" in prompt

    def test_no_diversity_hint_below_threshold(self):
        """Test that no diversity hint when count < 2."""
        from core.agentic.controller import AgenticSearchController
        from core.agentic.types import AgenticSearchSession

        manager = MagicMock()
        manager.api_models = {}
        controller = AgenticSearchController(
            model_manager=manager,
            web_search_manager=MagicMock(),
        )

        session = AgenticSearchSession(query="test")
        session.memory_search_counts = {"facts": 1}

        prompt = controller._build_iteration_prompt(
            query="test",
            search_context="[some results]",
            round_number=2,
            session=session,
        )

        assert "already searched" not in prompt

    def test_multiple_collections_tracked(self):
        """Test tracking across multiple collections."""
        from core.agentic.types import AgenticSearchSession

        session = AgenticSearchSession(query="test")
        session.memory_search_counts["facts"] = 2
        session.memory_search_counts["summaries"] = 1
        session.memory_search_counts["conversations"] = 3

        assert session.memory_search_counts["facts"] == 2
        assert session.memory_search_counts["summaries"] == 1
        assert session.memory_search_counts["conversations"] == 3


# =============================================================================
# Test Iteration Prompt With Inventory
# =============================================================================

class TestIterationPromptWithInventory:
    """Tests for _build_iteration_prompt() with context inventory."""

    @pytest.fixture
    def controller(self):
        """Create a minimal controller."""
        from core.agentic.controller import AgenticSearchController
        manager = MagicMock()
        manager.api_models = {}
        return AgenticSearchController(
            model_manager=manager,
            web_search_manager=MagicMock(),
        )

    def test_inventory_included_in_prompt(self, controller):
        """Test that context inventory is included in iteration prompt."""
        from core.agentic.types import AgenticSearchSession

        session = AgenticSearchSession(query="tell me about myself")
        session.context_inventory = (
            "Context already gathered by retrieval pipeline:\n"
            "- [USER PROFILE]: 40 categorized facts\n"
            "- [RECENT SUMMARIES]: 7 session summaries\n"
            "Do NOT re-search for information already covered above."
        )

        prompt = controller._build_iteration_prompt(
            query="tell me about myself",
            search_context="",
            round_number=2,
            session=session,
        )

        assert "Context already gathered by retrieval pipeline:" in prompt
        assert "[USER PROFILE]: 40 categorized facts" in prompt
        assert "Do NOT re-search" in prompt

    def test_empty_inventory_no_issue(self, controller):
        """Test that empty inventory doesn't break the prompt."""
        from core.agentic.types import AgenticSearchSession

        session = AgenticSearchSession(query="test")
        session.context_inventory = ""

        prompt = controller._build_iteration_prompt(
            query="test",
            search_context="[some results]",
            round_number=2,
            session=session,
        )

        assert "User Question: test" in prompt
        assert "Context already gathered" not in prompt

    def test_prompt_without_session(self, controller):
        """Test iteration prompt still works without session."""
        prompt = controller._build_iteration_prompt(
            query="test",
            search_context="results here",
            round_number=2,
            session=None,
        )

        assert "User Question: test" in prompt
        assert "results here" in prompt

    def test_inventory_and_diversity_hint_combined(self, controller):
        """Test that both inventory and diversity hint appear together."""
        from core.agentic.types import AgenticSearchSession

        session = AgenticSearchSession(query="tell me about myself")
        session.context_inventory = (
            "Context already gathered by retrieval pipeline:\n"
            "- [USER PROFILE]: 40 facts\n"
            "Do NOT re-search for information already covered above."
        )
        session.memory_search_counts = {"facts": 3}

        prompt = controller._build_iteration_prompt(
            query="tell me about myself",
            search_context="[some facts results]",
            round_number=4,
            session=session,
        )

        assert "Context already gathered" in prompt
        assert "already searched 'facts' 3 times" in prompt
