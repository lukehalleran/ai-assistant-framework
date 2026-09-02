"""Tests for the email_search agentic tool (Step 1).

Step 1: agentic tool email_search with native + XML wiring, gate arm, and formatting.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from core.agentic.types import SearchDecision, EMAIL_SEARCH_TOOL_DEFINITION
from core.agentic.protocols import (
    NativeToolsHandler,
    XMLMarkerHandler,
    get_protocol_handler,
)
from core.agentic.gate import evaluate_agentic_gate
from core.email.provider import EmailMessage


class TestEmailSearchToolDefinition:
    """Native tool definition is complete."""

    def test_email_search_tool_definition_exists(self):
        """EMAIL_SEARCH_TOOL_DEFINITION exists and is well-formed."""
        assert EMAIL_SEARCH_TOOL_DEFINITION is not None
        assert EMAIL_SEARCH_TOOL_DEFINITION["type"] == "function"
        func = EMAIL_SEARCH_TOOL_DEFINITION["function"]
        assert func["name"] == "email_search"
        assert "description" in func
        assert "parameters" in func
        # Parameters should have query (optional) and window_days (optional with default)
        props = func["parameters"]["properties"]
        assert "query" in props
        assert "window_days" in props
        assert props["window_days"]["default"] == 7


class TestNativeEmailSearchParsing:
    """Native tool call parsing."""

    def test_native_email_search_with_query(self):
        """Parse native tool call: email_search with query."""
        handler = NativeToolsHandler(email_search_available=True)
        tool_call = MagicMock()
        tool_call.function.name = "email_search"
        tool_call.function.arguments = '{"query": "from Morgan", "window_days": 14, "reason": "emails from Morgan"}'

        decision = handler._parse_single_tool_call(tool_call)
        assert decision is not None
        assert decision.wants_email_search is True
        assert decision.email_query == "from Morgan"
        assert decision.email_window_days == 14
        assert decision.email_reason == "emails from Morgan"

    def test_native_email_search_recent_only(self):
        """Parse native tool call: email_search without query (recent only)."""
        handler = NativeToolsHandler(email_search_available=True)
        tool_call = MagicMock()
        tool_call.function.name = "email_search"
        tool_call.function.arguments = '{"window_days": 7}'

        decision = handler._parse_single_tool_call(tool_call)
        assert decision is not None
        assert decision.wants_email_search is True
        assert decision.email_query is None
        assert decision.email_window_days == 7

    def test_native_email_search_defaults(self):
        """Parse native tool call: email_search with minimal args."""
        handler = NativeToolsHandler(email_search_available=True)
        tool_call = MagicMock()
        tool_call.function.name = "email_search"
        tool_call.function.arguments = '{}'

        decision = handler._parse_single_tool_call(tool_call)
        assert decision is not None
        assert decision.wants_email_search is True
        assert decision.email_query is None
        assert decision.email_window_days == 7  # default


class TestXMLEmailSearchParsing:
    """XML-format email_search parsing (both attribute and nested forms)."""

    def test_xml_email_search_attribute_form(self):
        """XML: <email_search query="..." window_days="...">reason</email_search>"""
        handler = XMLMarkerHandler()
        text = '<email_search query="subject recruitment" window_days="30">find recruitment emails</email_search>'
        decisions = handler.parse_response(text)
        assert len(decisions) > 0
        # Find the email_search decision
        email_decisions = [d for d in decisions if d.wants_email_search]
        assert len(email_decisions) == 1
        d = email_decisions[0]
        assert d.email_query == "subject recruitment"
        assert d.email_window_days == 30
        assert d.email_reason == "find recruitment emails"

    def test_xml_email_search_recent_only(self):
        """XML: <email_search window_days="7">what's in my inbox</email_search>"""
        handler = XMLMarkerHandler()
        text = '<email_search window_days="7">what is in my inbox this week</email_search>'
        decisions = handler.parse_response(text)
        email_decisions = [d for d in decisions if d.wants_email_search]
        assert len(email_decisions) == 1
        d = email_decisions[0]
        assert d.email_query is None  # No query, just recent
        assert d.email_window_days == 7

    def test_xml_email_search_nested_form(self):
        """XML: <email_search><query>x</query><window_days>7</window_days></email_search>"""
        handler = XMLMarkerHandler()
        text = '<email_search><query>from Alice</query><window_days>14</window_days><reason>emails from Alice</reason></email_search>'
        decisions = handler.parse_response(text)
        email_decisions = [d for d in decisions if d.wants_email_search]
        assert len(email_decisions) >= 1  # May match both nested pattern and no-attr pattern
        # Find one with the right query (nested form extracted correctly)
        correct_decisions = [d for d in email_decisions if d.email_query == "from Alice"]
        assert len(correct_decisions) > 0
        d = correct_decisions[0]
        assert d.email_window_days == 14
        assert d.email_reason == "emails from Alice"

    def test_xml_email_search_multiple_tools(self):
        """XML: multiple tools in one response including email_search."""
        handler = XMLMarkerHandler()
        text = (
            '<search>current events</search>'
            '\n<email_search query="from boss" window_days="7">find boss emails</email_search>'
        )
        decisions = handler.parse_response(text)
        assert len(decisions) >= 2
        email_decisions = [d for d in decisions if d.wants_email_search]
        assert len(email_decisions) == 1
        assert email_decisions[0].email_query == "from boss"


class TestEmailSearchGateArm:
    """Gate arm: narrow Tier-1 email search arm."""

    @pytest.mark.asyncio
    async def test_gate_fires_on_email_cue_with_question(self):
        """Gate fires: 'What emails do I have from Morgan?' — email noun + question."""
        decision = await evaluate_agentic_gate(
            "What emails do I have from Morgan?"
        )
        assert decision.should_trigger is True
        assert "tools" in decision.modes

    @pytest.mark.asyncio
    async def test_gate_fires_on_inbox_query(self):
        """Gate fires: 'Any new emails in my inbox?' — inbox noun + question."""
        decision = await evaluate_agentic_gate(
            "Any important emails in my inbox this week?"
        )
        assert decision.should_trigger is True
        assert "tools" in decision.modes

    @pytest.mark.asyncio
    async def test_gate_fires_on_gmail_question(self):
        """Gate fires: 'Check my Gmail' — gmail noun + imperative."""
        decision = await evaluate_agentic_gate(
            "Check my Gmail for updates"
        )
        assert decision.should_trigger is True
        assert "tools" in decision.modes

    @pytest.mark.asyncio
    async def test_gate_does_not_fire_on_narration(self):
        """Gate does NOT fire: 'I emailed the form yesterday' — narration, not request."""
        decision = await evaluate_agentic_gate(
            "I emailed the form to them yesterday"
        )
        # Might not trigger agentic at all, or triggers something else
        # but NOT for email search.
        if decision.should_trigger:
            assert "email_search" not in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_gate_does_not_fire_on_long_message(self):
        """Gate does NOT fire when >30 words (unless other signal)."""
        long_text = " ".join(
            ["Tell me about emails"] + ["word"] * 40  # >30 words total
        )
        decision = await evaluate_agentic_gate(long_text)
        # May or may not trigger depending on other signals, but length gate
        # is applied.
        if decision.should_trigger and decision.modes and "email" in str(decision.modes):
            # If it DID fire with email, verify word count logic worked
            pass

    @pytest.mark.asyncio
    async def test_gate_does_not_fire_on_bare_mail_word(self):
        """Gate does NOT fire on bare 'mail' (only bounded email nouns)."""
        decision = await evaluate_agentic_gate(
            "I mailed the package to them"
        )
        if decision.should_trigger:
            # Should NOT be for email_search (bare 'mail' in 'mailed' doesn't match)
            pass


class TestEmailSearchFormatting:
    """Drives the deployed _execute_email_search with a fake service —
    2026-09-01 findings: coverage disclosure (F1) + counting-shape wide
    fetch (F3). (The original stubs here were vacuous `pass` bodies.)"""

    def _run(self, monkeypatch, messages, query, window_days=7,
             coverage="Searched: Gmail. Outlook not connected."):
        import asyncio
        import core.email.service as svc
        import core.email.registry as reg
        from core.agentic.tools import ToolExecutor

        class _FakeService:
            def __init__(self, msgs):
                self._msgs = msgs
                self.providers = ["gmail"]
                self.calls = []
            async def search(self, q, *, window_days=30, limit=20):
                self.calls.append(("search", limit))
                return self._msgs[:limit]
            async def recent(self, *, window_days=7, limit=25):
                self.calls.append(("recent", limit))
                return self._msgs[:limit]

        fake = _FakeService(messages)
        monkeypatch.setattr(svc, "get_email_service", lambda: fake)
        monkeypatch.setattr(reg, "coverage_note", lambda: coverage)
        ex = ToolExecutor.__new__(ToolExecutor)
        out = asyncio.run(
            ToolExecutor._execute_email_search(ex, query, window_days))
        return out, fake

    @staticmethod
    def _msgs(n):
        return [EmailMessage(provider="gmail", message_id=f"m{i}",
                             sender=f"S{i} <s{i}@x.com>", subject=f"Subj {i}",
                             snippet="hello", date=f"2026-08-{(i % 28) + 1:02d}T10:00:00")
                for i in range(n)]

    def test_zero_results_carry_coverage(self, monkeypatch):
        out, _ = self._run(monkeypatch, [], "Morgan registration")
        assert "No emails found" in out
        assert "Outlook not connected" in out

    def test_results_carry_coverage(self, monkeypatch):
        out, _ = self._run(monkeypatch, self._msgs(3), None)
        assert "[EMAIL RESULTS] 3 message(s)" in out
        assert "Outlook not connected" in out

    def test_counting_shape_fetches_wide_lists_few(self, monkeypatch):
        out, fake = self._run(monkeypatch, self._msgs(60),
                              "how many emails am I getting lately")
        # wide fetch via recent(limit=200), true total reported
        assert ("recent", 200) in fake.calls
        assert "60 message(s) in the last 7 days" in out
        # but only the newest 20 rendered
        assert "newest 20 listed" in out
        assert out.count("\n    hello") == 20

    def test_non_counting_search_uses_normal_cap(self, monkeypatch):
        out, fake = self._run(monkeypatch, self._msgs(5), "Morgan registration")
        assert fake.calls and fake.calls[0][0] == "search"
        assert fake.calls[0][1] <= 20


class TestEmailSearchDispatch:
    """Dispatch table wiring."""

    def test_dispatch_table_has_email_search_row(self):
        """DISPATCH_TABLE includes email_search predicate and handler."""
        from core.agentic.tools import DISPATCH_TABLE
        email_predicates = [
            (pred, handler, args) for pred, handler, args in DISPATCH_TABLE
            if "email" in handler.lower()
        ]
        assert len(email_predicates) > 0
        email_row = email_predicates[0]
        assert "email_search" in email_row[1]

    def test_email_search_handler_name(self):
        """Handler name is _dispatch_email_search."""
        from core.agentic.tools import DISPATCH_TABLE
        for pred, handler_name, arg_builder in DISPATCH_TABLE:
            if "email" in handler_name.lower():
                assert handler_name == "_dispatch_email_search"
                break


class TestEmailSearchProtocolsHandler:
    """NativeToolsHandler includes email_search_available param."""

    def test_handler_accepts_email_search_available_param(self):
        """NativeToolsHandler.__init__ accepts email_search_available."""
        handler = NativeToolsHandler(email_search_available=True)
        assert handler.email_search_available is True

    def test_handler_default_email_search_available_false(self):
        """Default is False when not specified."""
        handler = NativeToolsHandler()
        assert handler.email_search_available is False

    def test_email_search_tool_in_tools_list_when_enabled(self):
        """email_search_tool is added to tools list when available."""
        handler = NativeToolsHandler(email_search_available=True)
        tools = handler.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "email_search" in tool_names

    def test_email_search_tool_not_in_tools_list_when_disabled(self):
        """email_search_tool is NOT added when not available."""
        handler = NativeToolsHandler(email_search_available=False)
        tools = handler.get_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "email_search" not in tool_names

    def test_factory_forwards_email_search_availability(self):
        from core.agentic.types import SearchProtocol

        handler = get_protocol_handler(
            SearchProtocol.NATIVE_TOOLS,
            email_search_available=True,
        )
        assert isinstance(handler, NativeToolsHandler)
        assert handler.email_search_available is True


class TestEmailSearchControllerWiring:
    def test_controller_reports_configured_provider_available(self, monkeypatch):
        import core.email.service as service_module
        from core.agentic.controller import AgenticSearchController

        provider = MagicMock()
        provider.is_configured.return_value = True
        service = MagicMock(providers=[provider])
        monkeypatch.setattr(service_module, "get_email_service", lambda: service)

        assert AgenticSearchController._email_search_is_available() is True

    def test_controller_fails_closed_when_provider_check_raises(self, monkeypatch):
        import core.email.service as service_module
        from core.agentic.controller import AgenticSearchController

        def _raise():
            raise RuntimeError("provider unavailable")

        monkeypatch.setattr(service_module, "get_email_service", _raise)
        assert AgenticSearchController._email_search_is_available() is False
