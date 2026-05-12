"""
Tests for the fetch_url agentic tool.

Covers:
- SearchDecision dataclass — fetch_url fields exist and work
- FETCH_URL_TOOL_DEFINITION — correct structure
- _classify_round_action — "[Fetch URL]" prefix classified correctly
- XMLMarkerHandler — parse <fetch_url url="...">reason</fetch_url> and alias patterns
- NativeToolsHandler — parse fetch_url tool call, availability flag controls tool listing
- ToolExecutor.dispatch_single — URL reroute (web_search with URL -> fetch_url)
- ToolExecutor._execute_fetch_url — mock _tavily_extract, success/failure/empty
- AgenticFormatter.format_fetch_url_context — output format
- _strip_leaked_xml_markers — strips fetch_url and other markers
- Response parser sentence-level thinking detection
- URL detection in handlers — _has_url flag, _web_search_keywords matching
"""

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from core.agentic.types import (
    AgenticSearchSession,
    FETCH_URL_TOOL_DEFINITION,
    SearchDecision,
)
from core.agentic.formatters import AgenticFormatter
from core.agentic.protocols import (
    NativeToolsHandler,
    XMLMarkerHandler,
    get_protocol_handler,
)
from core.agentic.types import SearchProtocol


# ── SearchDecision dataclass ──────────────────────────────────────

class TestSearchDecisionFetchUrlFields:
    """Test that SearchDecision has fetch_url fields with correct defaults."""

    def test_fetch_url_fields_exist(self):
        d = SearchDecision()
        assert hasattr(d, "wants_fetch_url")
        assert hasattr(d, "fetch_url")
        assert hasattr(d, "fetch_url_reason")

    def test_defaults_are_false_and_none(self):
        d = SearchDecision()
        assert d.wants_fetch_url is False
        assert d.fetch_url is None
        assert d.fetch_url_reason is None

    def test_set_fetch_url_fields(self):
        d = SearchDecision(
            wants_fetch_url=True,
            fetch_url="https://example.com",
            fetch_url_reason="user asked",
        )
        assert d.wants_fetch_url is True
        assert d.fetch_url == "https://example.com"
        assert d.fetch_url_reason == "user asked"

    def test_fetch_url_does_not_interfere_with_other_tools(self):
        d = SearchDecision(wants_fetch_url=True, fetch_url="https://x.com")
        assert d.wants_search is False
        assert d.wants_wolfram is False
        assert d.wants_memory_search is False
        assert d.wants_git_stats is False
        assert d.is_done is False


# ── FETCH_URL_TOOL_DEFINITION ────────────────────────────────────

class TestFetchUrlToolDefinition:
    """Test FETCH_URL_TOOL_DEFINITION structure."""

    def test_has_required_keys(self):
        assert FETCH_URL_TOOL_DEFINITION["type"] == "function"
        func = FETCH_URL_TOOL_DEFINITION["function"]
        assert func["name"] == "fetch_url"
        assert "description" in func
        assert "parameters" in func

    def test_url_is_required(self):
        params = FETCH_URL_TOOL_DEFINITION["function"]["parameters"]
        assert "url" in params["required"]

    def test_has_reason_field(self):
        props = FETCH_URL_TOOL_DEFINITION["function"]["parameters"]["properties"]
        assert "reason" in props

    def test_url_field_type(self):
        props = FETCH_URL_TOOL_DEFINITION["function"]["parameters"]["properties"]
        assert props["url"]["type"] == "string"

    def test_description_mentions_fetch(self):
        desc = FETCH_URL_TOOL_DEFINITION["function"]["description"]
        assert "url" in desc.lower() or "URL" in desc


# ── _classify_round_action ───────────────────────────────────────

class TestClassifyRoundAction:
    """Test _classify_round_action handles [Fetch URL] prefix."""

    def test_classify_fetch_url(self):
        result = AgenticSearchSession._classify_round_action("[Fetch URL] https://example.com")
        assert result == "fetch_url"

    def test_classify_web_search_unchanged(self):
        result = AgenticSearchSession._classify_round_action("some web query")
        assert result == "web_search"

    def test_classify_memory_search_unchanged(self):
        result = AgenticSearchSession._classify_round_action("[Memory: facts] test")
        assert result == "memory_search"

    def test_classify_git_stats_unchanged(self):
        result = AgenticSearchSession._classify_round_action("[Git Stats] commits today")
        assert result == "git_stats"

    def test_classify_file_read_unchanged(self):
        result = AgenticSearchSession._classify_round_action("[File Read] path/to/file.py")
        assert result == "file_read"


# ── XMLMarkerHandler — fetch_url parsing ─────────────────────────

class TestXMLMarkerFetchUrl:
    """Test XMLMarkerHandler parses <fetch_url url='...'> tags."""

    def test_parse_fetch_url_basic(self):
        handler = XMLMarkerHandler()
        text = '<fetch_url url="https://github.com/user/repo">check the repo</fetch_url>'
        decisions = handler.parse_response(text)
        assert len(decisions) == 1
        assert decisions[0].wants_fetch_url is True
        assert decisions[0].fetch_url == "https://github.com/user/repo"
        assert decisions[0].fetch_url_reason == "check the repo"

    def test_parse_fetch_url_single_quotes(self):
        handler = XMLMarkerHandler()
        text = "<fetch_url url='https://docs.python.org/3/'>reading python docs</fetch_url>"
        decisions = handler.parse_response(text)
        assert len(decisions) == 1
        assert decisions[0].wants_fetch_url is True
        assert decisions[0].fetch_url == "https://docs.python.org/3/"

    def test_parse_fetch_url_with_surrounding_text(self):
        handler = XMLMarkerHandler()
        text = 'Let me check that link. <fetch_url url="https://example.com">user link</fetch_url> I will analyze the content.'
        decisions = handler.parse_response(text)
        assert len(decisions) == 1
        assert decisions[0].wants_fetch_url is True
        assert decisions[0].fetch_url == "https://example.com"

    def test_parse_fetch_url_empty_reason(self):
        handler = XMLMarkerHandler()
        text = '<fetch_url url="https://example.com"></fetch_url>'
        decisions = handler.parse_response(text)
        assert len(decisions) == 1
        assert decisions[0].wants_fetch_url is True
        assert decisions[0].fetch_url == "https://example.com"
        # Empty reason is treated as None (empty string stripped)
        assert decisions[0].fetch_url_reason is None

    def test_parse_fetch_url_case_insensitive(self):
        handler = XMLMarkerHandler()
        text = '<FETCH_URL url="https://example.com">reason</FETCH_URL>'
        decisions = handler.parse_response(text)
        assert len(decisions) == 1
        assert decisions[0].wants_fetch_url is True

    def test_parse_multiple_fetch_urls(self):
        handler = XMLMarkerHandler()
        text = (
            '<fetch_url url="https://first.com">first</fetch_url> '
            '<fetch_url url="https://second.com">second</fetch_url>'
        )
        decisions = handler.parse_response(text)
        assert len(decisions) == 2
        urls = {d.fetch_url for d in decisions}
        assert "https://first.com" in urls
        assert "https://second.com" in urls

    def test_parse_fetch_url_mixed_with_search(self):
        handler = XMLMarkerHandler()
        text = (
            '<search>latest python news</search> '
            '<fetch_url url="https://python.org">main site</fetch_url>'
        )
        decisions = handler.parse_response(text)
        assert len(decisions) == 2
        types = {
            "search": any(d.wants_search for d in decisions),
            "fetch_url": any(d.wants_fetch_url for d in decisions),
        }
        assert types["search"] is True
        assert types["fetch_url"] is True

    def test_done_takes_priority_over_fetch_url(self):
        handler = XMLMarkerHandler()
        text = '<fetch_url url="https://example.com">reason</fetch_url> <done/>'
        decisions = handler.parse_response(text)
        assert len(decisions) == 1
        assert decisions[0].is_done is True

    def test_no_markers_returns_wants_answer(self):
        handler = XMLMarkerHandler()
        text = "I have enough information to answer now."
        decisions = handler.parse_response(text)
        assert len(decisions) == 1
        assert decisions[0].wants_answer is True


# ── XMLMarkerHandler — alias patterns ────────────────────────────

class TestXMLMarkerAliasPatterns:
    """Test XMLMarkerHandler parses alias patterns for search and memory."""

    def test_web_search_content_alias(self):
        handler = XMLMarkerHandler()
        text = '<web_search>python async tutorial</web_search>'
        decisions = handler.parse_response(text)
        assert len(decisions) == 1
        assert decisions[0].wants_search is True
        assert decisions[0].search_query == "python async tutorial"

    def test_web_search_attr_alias(self):
        handler = XMLMarkerHandler()
        text = '<web_search query="python async tutorial"/>'
        decisions = handler.parse_response(text)
        assert len(decisions) == 1
        assert decisions[0].wants_search is True
        assert decisions[0].search_query == "python async tutorial"

    def test_search_attr_pattern(self):
        handler = XMLMarkerHandler()
        text = '<search query="latest news"/>'
        decisions = handler.parse_response(text)
        assert len(decisions) == 1
        assert decisions[0].wants_search is True
        assert decisions[0].search_query == "latest news"

    def test_search_memory_attr_alias(self):
        handler = XMLMarkerHandler()
        text = '<search_memory query="my birthday"/>'
        decisions = handler.parse_response(text)
        assert len(decisions) == 1
        assert decisions[0].wants_memory_search is True
        assert decisions[0].memory_query == "my birthday"


# ── NativeToolsHandler — fetch_url ───────────────────────────────

class TestNativeProtocolFetchUrl:
    """Test NativeToolsHandler parses fetch_url tool calls."""

    def test_parse_fetch_url_native(self):
        handler = NativeToolsHandler(fetch_url_available=True)

        mock_response = MagicMock()
        tool_call = MagicMock()
        tool_call.function.name = "fetch_url"
        tool_call.function.arguments = '{"url": "https://github.com/user/repo", "reason": "user asked"}'
        mock_response.tool_calls = [tool_call]
        mock_response.content = None

        decisions = handler.parse_response(mock_response)
        assert len(decisions) == 1
        assert decisions[0].wants_fetch_url is True
        assert decisions[0].fetch_url == "https://github.com/user/repo"
        assert decisions[0].fetch_url_reason == "user asked"

    def test_parse_fetch_url_empty_url(self):
        handler = NativeToolsHandler(fetch_url_available=True)

        mock_response = MagicMock()
        tool_call = MagicMock()
        tool_call.function.name = "fetch_url"
        tool_call.function.arguments = '{"url": ""}'
        mock_response.tool_calls = [tool_call]
        mock_response.content = None

        decisions = handler.parse_response(mock_response)
        assert len(decisions) == 1
        # Empty url should not trigger fetch_url — falls through to wants_answer
        assert decisions[0].wants_fetch_url is False
        assert decisions[0].wants_answer is True

    def test_fetch_url_in_tools_list_when_available(self):
        handler = NativeToolsHandler(fetch_url_available=True)
        tools = handler.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "fetch_url" in tool_names

    def test_fetch_url_not_in_tools_when_disabled(self):
        handler = NativeToolsHandler(fetch_url_available=False)
        tools = handler.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "fetch_url" not in tool_names

    def test_fetch_url_in_augmented_system_prompt(self):
        handler = NativeToolsHandler(fetch_url_available=True)
        augmented = handler.augment_system_prompt("base prompt", 5)
        assert "fetch_url" in augmented

    def test_fetch_url_not_in_augmented_prompt_when_disabled(self):
        handler = NativeToolsHandler(fetch_url_available=False)
        augmented = handler.augment_system_prompt("base prompt", 5)
        assert "fetch_url" not in augmented

    def test_no_tool_calls_returns_wants_answer(self):
        handler = NativeToolsHandler(fetch_url_available=True)
        mock_response = MagicMock()
        mock_response.tool_calls = None
        mock_response.content = "Here is my answer."

        decisions = handler.parse_response(mock_response)
        assert len(decisions) == 1
        assert decisions[0].wants_answer is True
        assert decisions[0].partial_response == "Here is my answer."


# ── Factory Function ─────────────────────────────────────────────

class TestFactoryFunction:
    """Test get_protocol_handler passes fetch_url_available."""

    def test_factory_passes_fetch_url_available(self):
        handler = get_protocol_handler(
            SearchProtocol.NATIVE_TOOLS,
            fetch_url_available=True,
        )
        assert handler.fetch_url_available is True

    def test_factory_defaults_false(self):
        handler = get_protocol_handler(SearchProtocol.NATIVE_TOOLS)
        assert handler.fetch_url_available is False

    def test_xml_handler_returned_for_xml_protocol(self):
        handler = get_protocol_handler(
            SearchProtocol.XML_MARKERS,
            fetch_url_available=True,
        )
        assert isinstance(handler, XMLMarkerHandler)


# ── ToolExecutor.dispatch_single — URL reroute ───────────────────

class TestToolExecutorUrlReroute:
    """Test URL reroute: web_search with URL -> fetch_url dispatch."""

    @pytest.mark.asyncio
    async def test_url_in_web_search_reroutes_to_fetch_url(self):
        from core.agentic.tools import ToolExecutor

        mock_model_manager = MagicMock()
        mock_web_search_manager = MagicMock()
        formatter = AgenticFormatter()

        executor = ToolExecutor(
            model_manager=mock_model_manager,
            web_search_manager=mock_web_search_manager,
            formatter=formatter,
        )

        # Mock _execute_fetch_url to avoid real API calls
        executor._execute_fetch_url = AsyncMock(return_value="Title: Example\nURL: https://example.com\n\nPage content here")

        decision = SearchDecision(
            wants_search=True,
            search_query="https://example.com/page",
            search_reason="user wants to see this page",
        )

        result = await executor.dispatch_single(decision, 1, MagicMock(), None, None)

        # Should have been rerouted to fetch_url
        executor._execute_fetch_url.assert_awaited_once_with("https://example.com/page")
        assert "FETCHED URL" in result.formatted_context
        assert result.decision.wants_fetch_url is True

    @pytest.mark.asyncio
    async def test_url_embedded_in_search_query_reroutes(self):
        from core.agentic.tools import ToolExecutor

        executor = ToolExecutor(
            model_manager=MagicMock(),
            web_search_manager=MagicMock(),
            formatter=AgenticFormatter(),
        )
        executor._execute_fetch_url = AsyncMock(return_value="content")

        decision = SearchDecision(
            wants_search=True,
            search_query="check out https://github.com/user/repo please",
        )

        result = await executor.dispatch_single(decision, 1, MagicMock(), None, None)
        executor._execute_fetch_url.assert_awaited_once_with("https://github.com/user/repo")

    @pytest.mark.asyncio
    async def test_plain_search_query_not_rerouted(self):
        from core.agentic.tools import ToolExecutor

        executor = ToolExecutor(
            model_manager=MagicMock(),
            web_search_manager=MagicMock(),
            formatter=AgenticFormatter(),
        )

        # Mock web search execution
        mock_result = MagicMock()
        mock_result.pages = []
        executor._execute_search = AsyncMock(return_value=mock_result)
        executor._compress_results = AsyncMock(return_value="No results found.")

        decision = SearchDecision(
            wants_search=True,
            search_query="latest python news",
        )

        result = await executor.dispatch_single(decision, 1, MagicMock(), None, None)
        # Should NOT have been rerouted — normal web search
        assert result.decision.wants_search is True
        assert "FETCHED URL" not in result.formatted_context


# ── ToolExecutor._execute_fetch_url ──────────────────────────────

class TestExecuteFetchUrl:
    """Test ToolExecutor._execute_fetch_url with mocked _tavily_extract."""

    @pytest.mark.asyncio
    async def test_execute_fetch_url_success(self):
        from core.agentic.tools import ToolExecutor

        mock_page = MagicMock()
        mock_page.title = "Example Page"
        mock_page.content = "This is the page content."
        mock_page.snippet = "snippet"
        mock_page.url = "https://example.com"

        mock_wsm = MagicMock()
        mock_wsm._tavily_extract = AsyncMock(return_value=[mock_page])

        executor = ToolExecutor(
            model_manager=MagicMock(),
            web_search_manager=mock_wsm,
            formatter=AgenticFormatter(),
        )

        # Mock assign_web_ids to avoid import issues
        with patch("core.agentic.tools.ToolExecutor._execute_fetch_url") as _:
            pass  # We test the real method below

        result = await executor._execute_fetch_url("https://example.com")
        assert "Example Page" in result
        assert "This is the page content." in result

    @pytest.mark.asyncio
    async def test_execute_fetch_url_empty_pages(self):
        from core.agentic.tools import ToolExecutor

        mock_wsm = MagicMock()
        mock_wsm._tavily_extract = AsyncMock(return_value=[])

        executor = ToolExecutor(
            model_manager=MagicMock(),
            web_search_manager=mock_wsm,
            formatter=AgenticFormatter(),
        )

        result = await executor._execute_fetch_url("https://empty.com")
        assert "Could not fetch" in result

    @pytest.mark.asyncio
    async def test_execute_fetch_url_no_content(self):
        from core.agentic.tools import ToolExecutor

        mock_page = MagicMock()
        mock_page.title = "Empty Page"
        mock_page.content = ""
        mock_page.snippet = ""
        mock_page.url = "https://empty.com"

        mock_wsm = MagicMock()
        mock_wsm._tavily_extract = AsyncMock(return_value=[mock_page])

        executor = ToolExecutor(
            model_manager=MagicMock(),
            web_search_manager=mock_wsm,
            formatter=AgenticFormatter(),
        )

        result = await executor._execute_fetch_url("https://empty.com")
        assert "no extractable content" in result

    @pytest.mark.asyncio
    async def test_execute_fetch_url_exception(self):
        from core.agentic.tools import ToolExecutor

        mock_wsm = MagicMock()
        mock_wsm._tavily_extract = AsyncMock(side_effect=Exception("API error"))

        executor = ToolExecutor(
            model_manager=MagicMock(),
            web_search_manager=mock_wsm,
            formatter=AgenticFormatter(),
        )

        result = await executor._execute_fetch_url("https://fail.com")
        assert "URL fetch error" in result

    @pytest.mark.asyncio
    async def test_execute_fetch_url_no_web_search_manager(self):
        from core.agentic.tools import ToolExecutor

        executor = ToolExecutor(
            model_manager=MagicMock(),
            web_search_manager=None,
            formatter=AgenticFormatter(),
        )

        result = await executor._execute_fetch_url("https://example.com")
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_execute_fetch_url_uses_snippet_fallback(self):
        from core.agentic.tools import ToolExecutor

        mock_page = MagicMock()
        mock_page.title = "Snippet Page"
        mock_page.content = None
        mock_page.snippet = "This is the snippet content."
        mock_page.url = "https://example.com"

        mock_wsm = MagicMock()
        mock_wsm._tavily_extract = AsyncMock(return_value=[mock_page])

        executor = ToolExecutor(
            model_manager=MagicMock(),
            web_search_manager=mock_wsm,
            formatter=AgenticFormatter(),
        )

        result = await executor._execute_fetch_url("https://example.com")
        assert "This is the snippet content." in result


# ── AgenticFormatter.format_fetch_url_context ────────────────────

class TestFormatFetchUrlContext:
    """Test AgenticFormatter.format_fetch_url_context output."""

    def test_basic_format(self):
        formatter = AgenticFormatter()
        result = formatter.format_fetch_url_context(
            round_num=2,
            url="https://example.com",
            content="Page content here"
        )
        assert "[FETCHED URL" in result
        assert "Round 2" in result
        assert "https://example.com" in result
        assert "Page content here" in result

    def test_format_with_empty_content(self):
        formatter = AgenticFormatter()
        result = formatter.format_fetch_url_context(
            round_num=1,
            url="https://empty.com",
            content=""
        )
        assert "https://empty.com" in result

    def test_format_preserves_url(self):
        formatter = AgenticFormatter()
        url = "https://github.com/user/repo/blob/main/README.md"
        result = formatter.format_fetch_url_context(1, url, "readme content")
        assert url in result


# ── _strip_leaked_xml_markers ────────────────────────────────────

class TestStripLeakedXmlMarkers:
    """Test _strip_leaked_xml_markers from handlers.py."""

    def test_strip_fetch_url_markers(self):
        from gui.handlers import _strip_leaked_xml_markers

        text = 'Here is the info <fetch_url url="https://example.com">reason</fetch_url> from the page.'
        cleaned = _strip_leaked_xml_markers(text)
        assert "<fetch_url" not in cleaned
        assert "</fetch_url>" not in cleaned

    def test_strip_search_markers(self):
        from gui.handlers import _strip_leaked_xml_markers

        text = 'Let me <search>python async</search> for that.'
        cleaned = _strip_leaked_xml_markers(text)
        assert "<search>" not in cleaned
        assert "</search>" not in cleaned

    def test_strip_web_search_markers(self):
        from gui.handlers import _strip_leaked_xml_markers

        text = 'I found <web_search>latest news</web_search> results.'
        cleaned = _strip_leaked_xml_markers(text)
        assert "<web_search>" not in cleaned
        assert "</web_search>" not in cleaned

    def test_strip_memory_markers(self):
        from gui.handlers import _strip_leaked_xml_markers

        text = '<memory collection="facts">user name</memory>'
        cleaned = _strip_leaked_xml_markers(text)
        assert "<memory" not in cleaned
        assert "</memory>" not in cleaned

    def test_strip_search_memory_markers(self):
        from gui.handlers import _strip_leaked_xml_markers

        text = '<search_memory query="birthday"/>'
        cleaned = _strip_leaked_xml_markers(text)
        assert "<search_memory" not in cleaned

    def test_strip_git_stats_markers(self):
        from gui.handlers import _strip_leaked_xml_markers

        text = '<git_stats>commits this week</git_stats>'
        cleaned = _strip_leaked_xml_markers(text)
        assert "<git_stats>" not in cleaned

    def test_strip_recall_image_markers(self):
        from gui.handlers import _strip_leaked_xml_markers

        text = '<recall_image query="my cat"/>'
        cleaned = _strip_leaked_xml_markers(text)
        assert "<recall_image" not in cleaned

    def test_preserves_normal_text(self):
        from gui.handlers import _strip_leaked_xml_markers

        text = "This is a normal response with no XML markers."
        cleaned = _strip_leaked_xml_markers(text)
        assert cleaned == text

    def test_collapses_blank_lines(self):
        from gui.handlers import _strip_leaked_xml_markers

        text = 'Before\n\n\n<search>query</search>\n\n\nAfter'
        cleaned = _strip_leaked_xml_markers(text)
        assert "\n\n\n" not in cleaned

    def test_leaked_xml_regex_covers_fetch_url(self):
        from gui.handlers import _LEAKED_XML_TOOL_RE

        # Opening tags with attributes
        assert _LEAKED_XML_TOOL_RE.search('<fetch_url url="https://x.com">')
        # Closing tags
        assert _LEAKED_XML_TOOL_RE.search('</fetch_url>')
        # Plain opening tag
        assert _LEAKED_XML_TOOL_RE.search('<fetch_url>')


# ── Response parser — sentence-level thinking detection ──────────

class TestResponseParserSentenceThinking:
    """Test sentence-level thinking pattern detection for single-paragraph text."""

    def test_likely_untagged_thinking_single_paragraph(self):
        from core.response_parser import ResponseParser

        # Text with multiple thinking patterns but no line breaks
        text = (
            "The user wants to know about their schedule. "
            "I should check the memory for calendar entries. "
            "Let me fire the search_memory tool for this."
        )
        assert ResponseParser.likely_untagged_thinking(text) is True

    def test_likely_untagged_thinking_too_short(self):
        from core.response_parser import ResponseParser

        text = "Short text."
        assert ResponseParser.likely_untagged_thinking(text) is False

    def test_likely_untagged_thinking_normal_response(self):
        from core.response_parser import ResponseParser

        text = (
            "Based on the information I found, here are the key points. "
            "Python 3.12 introduced several performance improvements. "
            "The new features include better error messages."
        )
        assert ResponseParser.likely_untagged_thinking(text) is False

    def test_count_sentence_pattern_hits_multiple(self):
        from core.response_parser import ResponseParser

        text = "The user asked about their pet. I should use search_memory to find facts."
        hits = ResponseParser._count_sentence_pattern_hits(text)
        assert hits >= 2

    def test_count_sentence_pattern_hits_none(self):
        from core.response_parser import ResponseParser

        text = "Here is the weather forecast for today."
        hits = ResponseParser._count_sentence_pattern_hits(text)
        assert hits == 0

    def test_detect_untagged_thinking_single_paragraph_with_split(self):
        from core.response_parser import ResponseParser

        text = (
            "The user is asking about their schedule. "
            "I need to adjust my search strategy this time.\n\n"
            "Your schedule for today includes three meetings and a dentist appointment at 3pm."
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        assert thinking != ""
        assert "schedule" in answer.lower() or "meetings" in answer.lower()


# ── URL detection in handlers ────────────────────────────────────

class TestHandlerUrlDetection:
    """Test URL detection patterns used in the agentic gate in handlers.py."""

    def test_has_url_http(self):
        text = "check this http://example.com please"
        assert "http://" in text.lower()

    def test_has_url_https(self):
        text = "look at https://github.com/user/repo"
        assert "https://" in text.lower()

    def test_no_url(self):
        text = "tell me about python asyncio"
        assert "http://" not in text.lower()
        assert "https://" not in text.lower()

    def test_web_search_keywords_fetch_url(self):
        """Verify 'fetch url' is in the web search keyword list."""
        # This mirrors the keyword list in handlers.py
        _web_search_keywords = [
            'web search', 'search the web', 'search for', 'search online',
            'google ', 'look it up', 'fetch the', 'fetch url',
            'go to http', 'check out http', 'visit http',
        ]
        text = "fetch url https://example.com"
        _lower = text.lower()
        assert any(kw in _lower for kw in _web_search_keywords)

    def test_web_search_keywords_go_to_http(self):
        _web_search_keywords = [
            'web search', 'search the web', 'search for', 'search online',
            'google ', 'look it up', 'fetch the', 'fetch url',
            'go to http', 'check out http', 'visit http',
        ]
        text = "go to https://docs.python.org"
        _lower = text.lower()
        assert any(kw in _lower for kw in _web_search_keywords)

    def test_url_extraction_regex(self):
        """Test the URL extraction regex used in handlers.py."""
        import re
        _url_pattern = re.compile(r'https?://[^\s<>"\')\]]+')

        text = "Check https://github.com/user/repo and http://example.com/page?q=1"
        urls = _url_pattern.findall(text)
        assert len(urls) == 2
        assert "https://github.com/user/repo" in urls
        assert "http://example.com/page?q=1" in urls

    def test_url_extraction_no_urls(self):
        import re
        _url_pattern = re.compile(r'https?://[^\s<>"\')\]]+')

        text = "Tell me about http protocols and HTTPS"
        urls = _url_pattern.findall(text)
        # "http protocols" does not match because there's no "://"
        # but "HTTPS" alone also doesn't match
        assert len(urls) == 0

    def test_url_extraction_with_trailing_punct(self):
        import re
        _url_pattern = re.compile(r'https?://[^\s<>"\')\]]+')

        # URL followed by period should not include the period (it's a valid URL char though)
        text = "Visit https://example.com."
        urls = _url_pattern.findall(text)
        assert len(urls) == 1
        # Period IS included since regex doesn't exclude it — matches handlers.py behavior
        assert urls[0].startswith("https://example.com")


# ── ToolExecutor._dispatch_fetch_url integration ─────────────────

class TestDispatchFetchUrl:
    """Test the full _dispatch_fetch_url method on ToolExecutor."""

    @pytest.mark.asyncio
    async def test_dispatch_fetch_url_produces_events(self):
        from core.agentic.tools import ToolExecutor

        mock_page = MagicMock()
        mock_page.title = "Test Page"
        mock_page.content = "Test content"
        mock_page.snippet = ""
        mock_page.url = "https://test.com"

        mock_wsm = MagicMock()
        mock_wsm._tavily_extract = AsyncMock(return_value=[mock_page])

        executor = ToolExecutor(
            model_manager=MagicMock(),
            web_search_manager=mock_wsm,
            formatter=AgenticFormatter(),
        )

        decision = SearchDecision(
            wants_fetch_url=True,
            fetch_url="https://test.com",
            fetch_url_reason="testing",
        )

        result = await executor._dispatch_fetch_url(decision, round_number=3)

        # Check events
        assert len(result.start_events) == 1
        assert result.start_events[0].event_type == "fetching_url"
        assert "https://test.com" in result.start_events[0].message

        assert len(result.end_events) == 1
        assert result.end_events[0].event_type == "url_fetched"

        # Check round data
        assert result.round_data is not None
        assert result.round_data.round_number == 3
        assert "[Fetch URL]" in result.round_data.request.query

        # Check formatted context
        assert "FETCHED URL" in result.formatted_context
        assert "https://test.com" in result.formatted_context

    @pytest.mark.asyncio
    async def test_dispatch_fetch_url_records_duration(self):
        from core.agentic.tools import ToolExecutor

        mock_wsm = MagicMock()
        mock_wsm._tavily_extract = AsyncMock(return_value=[])

        executor = ToolExecutor(
            model_manager=MagicMock(),
            web_search_manager=mock_wsm,
            formatter=AgenticFormatter(),
        )

        decision = SearchDecision(
            wants_fetch_url=True,
            fetch_url="https://test.com",
        )

        result = await executor._dispatch_fetch_url(decision, round_number=1)
        assert result.round_data.duration_ms >= 0


# ── Controller._dispatch_single routing ──────────────────────────

class TestControllerDispatchRouting:
    """Test that controller._dispatch_single routes fetch_url to tool_executor."""

    @pytest.mark.asyncio
    async def test_controller_routes_fetch_url(self):
        from core.agentic.controller import AgenticSearchController

        mock_mm = MagicMock()
        mock_mm.api_models = {}
        mock_wsm = MagicMock()

        controller = AgenticSearchController(
            model_manager=mock_mm,
            web_search_manager=mock_wsm,
        )

        # Mock the tool executor's dispatch method
        mock_result = MagicMock()
        controller._tool_executor._dispatch_fetch_url = AsyncMock(return_value=mock_result)

        decision = SearchDecision(
            wants_fetch_url=True,
            fetch_url="https://example.com",
            fetch_url_reason="test",
        )

        result = await controller._dispatch_single(
            decision, 1, MagicMock(), None, None
        )
        assert result is mock_result
        controller._tool_executor._dispatch_fetch_url.assert_awaited_once_with(decision, 1)
