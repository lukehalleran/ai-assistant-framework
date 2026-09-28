"""Regression tests for the 2026-09-27 (lane F, batch X2) fixes:

1. ``utils/test_envelope.py`` — the single structural parser for the
   ``[test]...[/test]`` operator-marker convention, recognising BOTH the
   whole-line block form and the inline form (live evidence,
   ~/daemon_exec/sep27_runs/F/PLAN_20260927_followup_tool_calls.md item 4:
   ``memory/fact_source.py`` used to recognise the tags only as whole
   lines, so an inline probe like "[test]Navient. Search that[/test]"
   was invisible to it).
2. ``memory/fact_source.contains_test_block`` / ``quoted_correspondence_lines``
   now delegate to that shared parser (inline form recognised, BC-58).
3. ``core.response_parser.ResponseParser.strip_tool_markers`` — a raw
   agentic tool-call marker (e.g. "<email_search>...</email_search>") that
   leaked into a delivered reply, built directly on
   ``core.agentic.protocols.XMLMarkerHandler``'s own compiled patterns.
4. ``gui.handlers._apply_delivery_revisions`` — the shared agentic/enhanced
   delivery chokepoint now strips a leaked raw tool marker FIRST, trying an
   optional ``regenerate_fn`` (agentic only) before falling back to the
   cleaned text plus an honest delivery notice (BC-91, BC-46, BC-44).
5. Ingress: ``gui.handlers._handle_submit_inner``'s A10 chokepoint now
   unwraps the ``[test]...[/test]`` envelope before text reaches
   gate/intent/tone/STM classification, while the STORED query/user_text
   keep the envelope so fact_source's extractors still skip the turn.

Live evidence (2026-09-27 21:56-21:58 probes): "[test]Navient. Search
that[/test]" → the email tool never ran, but the final synthesis shipped
"<email_search>Navient</email_search>" verbatim as the reply.

Uses synthetic content throughout (no real people/documents).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import utils.read_time_markers as read_time_markers
import utils.test_envelope as test_envelope
from core.response_parser import ResponseParser
from memory import fact_source

from tests.unit.test_handle_submit import (
    _debug_record,
    _final_content,
    _gate_decision,
    _make_orchestrator,
    _run_submit,
)
from tests.unit.test_sep08_agentic_answer_integrity import _force_agentic_tools_patch


# ---------------------------------------------------------------------------
# (1) utils.test_envelope — the shared structural parser
# ---------------------------------------------------------------------------

class TestTestEnvelopeParser:
    def test_has_envelope_inline_form(self):
        assert test_envelope.has_envelope("[test]Navient. Search that[/test]") is True

    def test_has_envelope_block_form(self):
        assert test_envelope.has_envelope("[test]\nfoo\nbar\n[/test]") is True

    def test_has_envelope_false_when_dangling_open_tag(self):
        assert test_envelope.has_envelope("[test]Navient, no close") is False

    def test_has_envelope_false_on_ordinary_text(self):
        assert test_envelope.has_envelope("just a normal message") is False

    def test_has_envelope_false_on_empty_or_none(self):
        assert test_envelope.has_envelope("") is False
        assert test_envelope.has_envelope(None) is False

    def test_inner_text_unwraps_inline_form(self):
        assert test_envelope.inner_text("[test]yes please[/test]") == "yes please"

    def test_inner_text_unwraps_block_form_content(self):
        assert test_envelope.inner_text("[test]\nfoo\nbar\n[/test]") == "\nfoo\nbar\n"

    def test_inner_text_is_identity_on_ordinary_text(self):
        text = "Don't search yet, just ask me whether I want you to search my email"
        assert test_envelope.inner_text(text) == text

    def test_inner_text_is_identity_on_dangling_tag(self):
        text = "[test]Navient, no close"
        assert test_envelope.inner_text(text) == text

    def test_envelope_line_indices_inline_marks_only_that_line(self):
        assert test_envelope.envelope_line_indices("hi\n[test]inline probe[/test]\nbye") == {1}

    def test_envelope_line_indices_block_marks_every_line_inclusive(self):
        assert test_envelope.envelope_line_indices("[test]\nfoo\nbar\n[/test]") == {0, 1, 2, 3}

    def test_envelope_line_indices_empty_when_no_envelope(self):
        assert test_envelope.envelope_line_indices("no markers here") == set()


# ---------------------------------------------------------------------------
# (2) memory.fact_source — now delegates to the shared parser
# ---------------------------------------------------------------------------

class TestFactSourceInlineTestBlock:
    def test_contains_test_block_recognises_inline_probe(self):
        # This is the exact live-evidence shape (item 4 of the plan): an
        # inline probe used to be INVISIBLE to contains_test_block, so a
        # fact could be minted from it.
        assert fact_source.contains_test_block("[test]Navient. Search that[/test]") is True

    def test_contains_test_block_still_recognises_whole_line_form(self):
        text = "[test]\nUser started Lorvatin today\n[/test]"
        assert fact_source.contains_test_block(text) is True

    def test_contains_test_block_false_on_ordinary_text(self):
        assert fact_source.contains_test_block("Started Lorvatin today") is False

    def test_quoted_correspondence_lines_covers_inline_probe(self):
        text = "hi\n[test]inline probe[/test]\nbye"
        assert fact_source.quoted_correspondence_lines(text) == {1}

    def test_strip_quoted_correspondence_drops_inline_probe_content(self):
        text = "hi\n[test]inline probe[/test]\nbye"
        assert fact_source.strip_quoted_correspondence(text) == "hi\nbye"


# ---------------------------------------------------------------------------
# (3) core.response_parser.ResponseParser.strip_tool_markers
# ---------------------------------------------------------------------------

class TestStripToolMarkers:
    def test_strips_email_search_marker_and_reports_it(self):
        text = "Sure, checking now.\n<email_search>Navient</email_search>\nRunning that now."
        clean, found = ResponseParser.strip_tool_markers(text)
        assert "<email_search>" not in clean
        assert "email_search" in found

    def test_strips_a_marker_the_old_handlers_taglist_never_covered(self):
        # gui/handlers.py's hand-written _AGENTIC_OUTER_TAGS never listed
        # lookup_contact/pattern_scan/propose_action/pubmed — proving this
        # helper is NOT that same list (built from protocols.py instead).
        text = '<lookup_contact name="Harper">why</lookup_contact> ok'
        clean, found = ResponseParser.strip_tool_markers(text)
        assert "<lookup_contact" not in clean
        assert "lookup_contact" in found

    def test_no_marker_returns_text_unchanged_and_empty_found(self):
        text = "Just an ordinary reply with no markers at all."
        clean, found = ResponseParser.strip_tool_markers(text)
        assert clean == text
        assert found == []

    def test_empty_and_none_input_never_raises(self):
        assert ResponseParser.strip_tool_markers("") == ("", [])
        assert ResponseParser.strip_tool_markers(None) == ("", [])


# ---------------------------------------------------------------------------
# (4) gui.handlers._apply_delivery_revisions — the shared chokepoint
# ---------------------------------------------------------------------------

def _delivery_revision_patches():
    """Neutralize grounding/personal-claim so only the tool-marker guard
    under test can change the body."""
    return [
        patch("gui.handlers._apply_grounding_check_for_delivery",
              new_callable=AsyncMock, return_value=(None, None)),
        patch("gui.handlers._apply_personal_claim_check_for_delivery",
              new_callable=AsyncMock, return_value=None),
    ]


_LEAKED_EMAIL_MARKER_BODY = (
    "Sure, let me check your email for that.\n"
    "<email_search>Navient</email_search>\n"
    "Running that now."
)


class TestApplyDeliveryRevisionsToolMarkerGuard:
    @pytest.mark.asyncio
    async def test_successful_regenerate_replaces_body_with_no_notice(self):
        from gui.handlers import _apply_delivery_revisions

        ctx = SimpleNamespace(telemetry={})
        patches = _delivery_revision_patches()
        for p in patches:
            p.start()
        try:
            result = await _apply_delivery_revisions(
                ctx, _LEAKED_EMAIL_MARKER_BODY,
                regenerate_fn=AsyncMock(return_value="Here's what I found in your email."),
            )
        finally:
            patch.stopall()

        assert "<email_search>" not in result
        assert result == "Here's what I found in your email."
        assert read_time_markers.NOTICE_TOOL_NOT_RUN not in result
        assert ctx.telemetry.get("tool_marker_recovered") is True

    @pytest.mark.asyncio
    async def test_no_regenerate_fn_delivers_cleaned_text_plus_notice(self):
        """The enhanced-mode shape: no regenerate_fn is ever passed."""
        from gui.handlers import _apply_delivery_revisions

        ctx = SimpleNamespace(telemetry={})
        patches = _delivery_revision_patches()
        for p in patches:
            p.start()
        try:
            result = await _apply_delivery_revisions(ctx, _LEAKED_EMAIL_MARKER_BODY)
        finally:
            patch.stopall()

        assert "<email_search>" not in result
        assert read_time_markers.NOTICE_TOOL_NOT_RUN in result
        assert ctx.telemetry.get("tool_marker_notice") is True

    @pytest.mark.asyncio
    async def test_regenerate_fn_raising_falls_back_to_notice(self):
        """regenerate_fn unavailable/erroring (the agentic path's real shape
        when the controller can't regenerate) — same honest fallback."""
        from gui.handlers import _apply_delivery_revisions

        ctx = SimpleNamespace(telemetry={})
        patches = _delivery_revision_patches()
        for p in patches:
            p.start()
        try:
            result = await _apply_delivery_revisions(
                ctx, _LEAKED_EMAIL_MARKER_BODY,
                regenerate_fn=AsyncMock(side_effect=RuntimeError("no controller state")),
            )
        finally:
            patch.stopall()

        assert "<email_search>" not in result
        assert read_time_markers.NOTICE_TOOL_NOT_RUN in result
        assert ctx.telemetry.get("tool_marker_notice") is True

    @pytest.mark.asyncio
    async def test_clean_body_untouched_no_telemetry_set(self):
        from gui.handlers import _apply_delivery_revisions

        ctx = SimpleNamespace(telemetry={})
        patches = _delivery_revision_patches()
        for p in patches:
            p.start()
        try:
            result = await _apply_delivery_revisions(ctx, "An ordinary, marker-free reply.")
        finally:
            patch.stopall()

        assert result == "An ordinary, marker-free reply."
        assert "tool_marker_notice" not in ctx.telemetry
        assert "tool_marker_recovered" not in ctx.telemetry


# ---------------------------------------------------------------------------
# (4b) End-to-end through handle_submit: agentic AND enhanced modes.
# ---------------------------------------------------------------------------

class TestLeakedMarkerNeverShipsAgentic:
    @pytest.mark.asyncio
    async def test_leaked_email_search_marker_stripped_with_notice(self):
        orch = _make_orchestrator(
            agentic_enabled=True, agentic_items=[_LEAKED_EMAIL_MARKER_BODY],
        )
        # regenerate_final_answer on the bare MagicMock controller is not
        # awaitable — the guard's own try/except falls back to the notice,
        # exactly the "recovery unavailable" shape.
        with patch("gui.handlers._dispatch_storage") as mock_dispatch, \
                _force_agentic_tools_patch():
            results = await _run_submit("do you remember my brother's name?", orch)

        content = _final_content(results)
        debug = _debug_record(results)
        assert debug is not None
        assert "<email_search>" not in content
        assert "<email_search>" not in debug["response"]
        assert read_time_markers.NOTICE_TOOL_NOT_RUN in content

        assert mock_dispatch.call_count == 1
        stored_text = mock_dispatch.call_args[0][2]
        assert "<email_search>" not in stored_text
        assert read_time_markers.NOTICE_TOOL_NOT_RUN in stored_text


class TestLeakedMarkerNeverShipsEnhanced:
    @pytest.mark.asyncio
    async def test_leaked_email_search_marker_stripped_with_notice(self):
        orch = _make_orchestrator(streaming_chunks=[_LEAKED_EMAIL_MARKER_BODY])

        with patch("gui.handlers._dispatch_storage") as mock_dispatch:
            results = await _run_submit("can you look into that account issue?", orch)

        content = _final_content(results)
        debug = _debug_record(results)
        assert debug is not None
        assert debug["mode"] == "enhanced"
        assert "<email_search>" not in content
        assert "<email_search>" not in debug["response"]
        assert read_time_markers.NOTICE_TOOL_NOT_RUN in content

        assert mock_dispatch.call_count == 1
        stored_text = mock_dispatch.call_args[0][2]
        assert "<email_search>" not in stored_text
        assert read_time_markers.NOTICE_TOOL_NOT_RUN in stored_text


# ---------------------------------------------------------------------------
# (5) Ingress: [test]...[/test] unwrapped for classification, kept for storage
# ---------------------------------------------------------------------------

class TestIngressEnvelopeUnwrapForClassification:
    @pytest.mark.asyncio
    async def test_gate_sees_inner_text_storage_keeps_envelope(self):
        orch = _make_orchestrator(agentic_enabled=True, agentic_items=["ok"])
        mock_gate = AsyncMock(return_value=_gate_decision())

        with patch("gui.handlers._dispatch_storage") as mock_dispatch, \
                patch("core.agentic.gate.evaluate_agentic_gate", mock_gate):
            await _run_submit("[test]yes please[/test]", orch)

        assert mock_gate.await_args is not None
        assert mock_gate.await_args.kwargs["user_text"] == "yes please"

        assert mock_dispatch.call_count == 1
        stored_args = mock_dispatch.call_args[0]
        merged_input_stored, user_text_stored = stored_args[1], stored_args[3]
        assert merged_input_stored == "[test]yes please[/test]"
        assert user_text_stored == "[test]yes please[/test]"
