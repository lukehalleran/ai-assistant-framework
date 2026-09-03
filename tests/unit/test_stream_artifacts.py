"""Tests for the 2026-08-03 stream-artifact fixes.

Two live incidents from box testing:

1. Stray trailing 'e' — the OpenRouter kimi-3 endpoint emitted a lone 'e'
   content token right before finish_reason=stop, producing "…landed?e"
   (streamed turn, stored verbatim) and "…impress them.e" (non-streaming
   summary). strip_trailing_stream_artifact() removes it at the storage
   boundaries (sanitize_for_storage + both add_summary paths).

2. Thinking-tag flash — between the synthetic </thinking> marker and the
   first real content token, the stream buffer is exactly
   "<thinking></thinking>"; parse_thinking_block returns ("", "") for it, so
   the display fallthrough showed the literal tags for ~1s.
   is_empty_thinking_shell() lets handlers keep the 💭 indicator up instead.
"""

import pytest

from core.response_parser import ResponseParser


class TestStripTrailingStreamArtifact:
    def test_strips_e_after_question_mark(self):
        # The live turn-1 incident (2026-08-03 12:06)
        text = "How are you doing today, after everything with your dad and the way it all landed?e"
        assert ResponseParser.strip_trailing_stream_artifact(text) == text[:-1]

    def test_strips_e_after_period(self):
        # The summary-path incident ("…DeepSeek not supporting image input.e")
        text = "related to DeepSeek not supporting image input.e"
        assert ResponseParser.strip_trailing_stream_artifact(text) == text[:-1]

    def test_strips_e_after_closing_quote(self):
        text = 'He said "done."e'
        assert ResponseParser.strip_trailing_stream_artifact(text) == 'He said "done."'

    def test_strips_e_after_curly_quote(self):
        text = "It’s over.”e"
        assert ResponseParser.strip_trailing_stream_artifact(text) == "It’s over.”"

    def test_trailing_whitespace_after_artifact(self):
        text = "Are you home now, or still out?e\n"
        assert ResponseParser.strip_trailing_stream_artifact(text) == "Are you home now, or still out?"

    def test_preserves_normal_word_ending_in_e(self):
        text = "I agree"
        assert ResponseParser.strip_trailing_stream_artifact(text) == text

    def test_preserves_sentence_ending_in_e_period(self):
        text = "Grab a coffee."
        assert ResponseParser.strip_trailing_stream_artifact(text) == text

    def test_preserves_ie_abbreviation(self):
        # "i.e" is the one legitimate letter-after-period ending
        text = "the first smoothing parameter, i.e"
        assert ResponseParser.strip_trailing_stream_artifact(text) == text

    def test_preserves_e_separated_by_space(self):
        # A detached final "e" is a different (unobserved) shape — leave it
        text = "the way it all landed? e"
        assert ResponseParser.strip_trailing_stream_artifact(text) == text

    def test_preserves_mid_text_pattern(self):
        # Pattern only fires at end-of-text
        text = "landed?e is what the bug looked like, now fixed"
        assert ResponseParser.strip_trailing_stream_artifact(text) == text

    def test_empty_and_none_safe(self):
        assert ResponseParser.strip_trailing_stream_artifact("") == ""
        assert ResponseParser.strip_trailing_stream_artifact(None) == ""

    def test_sanitize_for_storage_applies_strip(self):
        stored = ResponseParser.sanitize_for_storage(
            "Hey. Good to see you back — how did it all land?e"
        )
        assert not stored.endswith("?e")
        assert stored.endswith("?")

    def test_sanitize_for_storage_combined_with_empty_pair(self):
        stored = ResponseParser.sanitize_for_storage(
            "<thinking></thinking>A bit better is real progress.e"
        )
        assert stored == "A bit better is real progress."


class TestStripTrailingStreamError:
    """2026-08-14: kimi-3 upstream (DigitalOcean) closed the connection
    mid-stream; ResponseGenerator's "[Streaming Error: ...]" sentinel wasn't in
    API_ERROR_PREFIXES, so the turn persisted as a real reply (~20 historical
    docs in chroma, one in the corpus). The appended-after-partial-content case
    is handled here; the marker-only case by the storage-time error guard.
    """

    LIVE_MARKER = "[Streaming Error: Upstream error from DigitalOcean: Connection closed.]"

    def test_strips_bracketed_marker_after_partial_content(self):
        text = f"That trend is real, and it's the right direction.\n{self.LIVE_MARKER}"
        assert ResponseParser.strip_trailing_stream_error(text) == (
            "That trend is real, and it's the right direction."
        )

    def test_strips_unbracketed_marker_shape(self):
        # The second emit shape: "[Streaming Error] <msg>" (response_generator.py:388)
        text = "Partial answer here [Streaming Error] Error processing stream"
        assert ResponseParser.strip_trailing_stream_error(text) == "Partial answer here"

    def test_strips_truncated_bracketed_marker(self):
        # Stream can die mid-marker too
        text = "Partial answer here\n[Streaming Error: Provider returned"
        assert ResponseParser.strip_trailing_stream_error(text) == "Partial answer here"

    def test_marker_only_response_left_intact(self):
        # Whole-text markers are the storage-time API-error guard's job —
        # stripping to "" here would reroute them to the misleading
        # "entirely thinking content" skip path.
        assert ResponseParser.strip_trailing_stream_error(self.LIVE_MARKER) == self.LIVE_MARKER

    def test_quoted_marker_mid_text_untouched(self):
        # A conversation ABOUT the error must keep its content
        text = "The error [Streaming Error: x] means the provider dropped the stream. Retry it."
        assert ResponseParser.strip_trailing_stream_error(text) == text

    def test_empty_and_none_safe(self):
        assert ResponseParser.strip_trailing_stream_error("") == ""
        assert ResponseParser.strip_trailing_stream_error(None) == ""

    def test_sanitize_for_storage_strips_appended_marker(self):
        stored = ResponseParser.sanitize_for_storage(
            f"Glad today's landing better.\n\n{self.LIVE_MARKER}"
        )
        assert stored == "Glad today's landing better."

    def test_sanitize_for_storage_keeps_marker_only_response(self):
        # So memory_storage._is_api_error_response sees and skips it
        assert ResponseParser.sanitize_for_storage(self.LIVE_MARKER) == self.LIVE_MARKER

    def test_streaming_error_registered_as_api_error_prefix(self):
        from models.model_manager import API_ERROR_PREFIXES

        assert self.LIVE_MARKER.startswith(API_ERROR_PREFIXES)
        assert "[Streaming Error] Error processing stream".startswith(API_ERROR_PREFIXES)


class TestIsEmptyThinkingShell:
    def test_empty_pair_is_shell(self):
        # The exact live buffer between marker arrival and first content token
        assert ResponseParser.is_empty_thinking_shell("<thinking></thinking>")

    def test_pair_with_whitespace_is_shell(self):
        assert ResponseParser.is_empty_thinking_shell("<thinking> </thinking>\n")

    def test_lone_open_marker_is_shell(self):
        # handlers checks has_incomplete_thinking_block first, but shell
        # detection should agree that a bare marker has nothing displayable
        assert ResponseParser.is_empty_thinking_shell("<thinking>")

    def test_pair_plus_content_is_not_shell(self):
        assert not ResponseParser.is_empty_thinking_shell("<thinking></thinking>Hey.")

    def test_plain_text_is_not_shell(self):
        assert not ResponseParser.is_empty_thinking_shell("Hey. Good to see you back.")

    def test_empty_string_is_not_shell(self):
        assert not ResponseParser.is_empty_thinking_shell("")
        assert not ResponseParser.is_empty_thinking_shell("   ")

    def test_parse_thinking_block_empty_pair_returns_nothing(self):
        # Documents the fallthrough that caused the flash: the empty pair
        # yields ("", ""), so display code can't rely on parse alone.
        thinking, answer = ResponseParser.parse_thinking_block("<thinking></thinking>")
        assert thinking == ""
        assert answer == ""


class TestSummaryStoragePathsStripArtifact:
    def test_corpus_manager_add_summary_strips(self, tmp_path):
        from memory.corpus_manager import CorpusManager

        cm = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
        cm.add_summary("The user scored 87.86 on the exam, up six points.e")
        summaries = [e for e in cm.corpus if "@summary" in (e.get("tags") or [])]
        assert len(summaries) == 1
        assert summaries[0]["content"].endswith("six points.")
        assert not summaries[0]["content"].endswith(".e")


class TestDisplayPathsStripArtifact:
    """2026-08-14: the 08-03 artifact fix covered the STORAGE boundary and the
    enhanced display path (via _sanitize_response_text → sanitize_for_storage),
    but the agentic / raw / best-of-duel display paths yielded the raw text —
    a live agentic turn showed "…by the way?e" in the chat bubble and debug
    record while the stored copy was clean."""

    def test_sanitize_response_text_strips_artifact(self):
        from gui.handlers import _sanitize_response_text
        out = _sanitize_response_text("Did they ever respond, by the way?e")
        assert out.endswith("by the way?")

    def test_all_display_paths_call_artifact_strip(self):
        # Source-level guard: the agentic, raw, and duel branches must each
        # apply strip_trailing_stream_artifact to what they yield/record
        # (the enhanced path goes through _sanitize_response_text instead).
        from pathlib import Path
        src = Path("gui/handlers.py").read_text()
        calls = src.count("strip_trailing_stream_artifact")
        assert calls >= 4, (
            f"expected >=4 strip_trailing_stream_artifact call sites in "
            f"gui/handlers.py (agentic final_output + display_output, raw, "
            f"duel x2), found {calls}"
        )


class TestDebugRecordEmptyShellStrip:
    """2026-08-05: an agentic debug record's RESPONSE opened with a literal
    "<thinking></thinking>" shell — display and storage already strip it, so
    the record read as a leak that never actually reached the user. The
    record builder now drops a LEADING EMPTY shell only (real thinking
    blocks stay visible for diagnostics)."""

    def _record(self, response):
        from gui.handlers import _build_debug_record
        return _build_debug_record(
            mode="agentic-search", user_text="q", prompt="p", system_prompt="s",
            response=response, model="m", prompt_tokens=1, system_tokens=1,
            total_tokens=2, citations=[], orchestrator=None,
        )

    def test_leading_empty_shell_stripped(self):
        rec = self._record("<thinking></thinking>That's such a specific kind of torture.")
        assert rec["response"].startswith("That's such a specific")

    def test_reasoning_family_covered(self):
        rec = self._record("<reasoning> </reasoning>\nAnswer text.")
        assert rec["response"] == "Answer text."

    def test_real_thinking_block_kept_for_diagnostics(self):
        raw = "<thinking>actual reasoning here</thinking>Answer."
        assert self._record(raw)["response"] == raw

    def test_plain_response_untouched(self):
        assert self._record("Just an answer.")["response"] == "Just an answer."

    def test_exact_response_plan_attached_to_debug_and_provenance(self):
        from core.response_planner import ResponsePlan
        from gui.handlers import _build_debug_record

        orch = type("Orchestrator", (), {})()
        orch.enable_citations = False
        orch._current_response_plan = ResponsePlan(
            key_points=["address Fable directly"],
            tone="direct",
            avoid=["reverse speaker and audience"],
            strategy="hand over the floor",
            context_digest_sha256="c" * 64,
            context_sections=["recent_conversations"],
            directive_locked=True,
        )
        provenance = {"response_mode": "enhanced"}
        rec = _build_debug_record(
            mode="enhanced", user_text="q", prompt="p", system_prompt="s",
            response="answer", model="m", prompt_tokens=1, system_tokens=1,
            total_tokens=2, citations=[], orchestrator=orch,
            provenance=provenance,
        )
        assert rec["response_plan"]["key_points"] == ["address Fable directly"]
        assert rec["response_plan"]["directive_locked"] is True
        assert provenance["response_plan"] == rec["response_plan"]


class TestSpecialTokenStrip:
    """2026-08-21: kimi-3 intermittently emits <|sep|> as the FIRST content
    chunk — 11 stored corpus replies began "<|sep|>That's ...". Edge runs of
    <|token|> markers are stripped; mid-text mentions survive (a conversation
    ABOUT the token must not be mangled)."""

    def test_leading_sep_stripped(self):
        from core.response_parser import ResponseParser as R
        assert R.strip_stream_special_tokens("<|sep|>Hello there.") == "Hello there."

    def test_leading_run_and_whitespace(self):
        from core.response_parser import ResponseParser as R
        assert R.strip_stream_special_tokens(" <|sep|><|eos|>  Hi") == "Hi"

    def test_trailing_stripped(self):
        from core.response_parser import ResponseParser as R
        assert R.strip_stream_special_tokens("ends here <|sep|>") == "ends here"

    def test_mid_text_preserved(self):
        from core.response_parser import ResponseParser as R
        mid = "the <|sep|> token appears mid-text when we discuss it"
        assert R.strip_stream_special_tokens(mid) == mid

    def test_folded_into_trailing_artifact_strip(self):
        # every existing display/storage call site inherits via this fold
        from core.response_parser import ResponseParser as R
        assert R.strip_trailing_stream_artifact("<|sep|>Landed fine.") == "Landed fine."

    def test_sanitize_for_storage_strips_leading_sep(self):
        from core.response_parser import ResponseParser
        out = ResponseParser.sanitize_for_storage("<|sep|>A real answer here.")
        assert "<|sep|>" not in out
        assert "A real answer here." in out


class TestEmptyShellThenContent:
    """2026-08-22: kimi-3 emitted literal "<thinking>" + "</thinking>" as its
    first two CONTENT chunks, then "<|sep|>Nice —...". The 08-03 empty-shell
    hold covered the shell-only buffer, but once content followed,
    parse_thinking_block found no thinking BODY and the streaming display
    fell through to the RAW buffer — literal tags on screen until the
    end-of-stream recovery (15s later, live turn 13:06). The shell is now
    stripped from the buffer the moment content follows."""

    def test_strip_leading_empty_shell_with_content(self):
        from core.response_parser import ResponseParser as R
        out = R.strip_leading_empty_thinking_shell(
            "<thinking></thinking><|sep|>Nice — pushed and docs updated."
        )
        assert out == "<|sep|>Nice — pushed and docs updated."

    def test_whitespace_and_variant_tags(self):
        from core.response_parser import ResponseParser as R
        assert R.strip_leading_empty_thinking_shell(
            "<thinking>  </thinking>\nAnswer here."
        ) == "Answer here."
        assert R.strip_leading_empty_thinking_shell(
            "<reasoning></reasoning>Answer."
        ) == "Answer."

    def test_nonempty_thinking_untouched(self):
        # a REAL thinking block must go through parse_thinking_block, not this
        from core.response_parser import ResponseParser as R
        text = "<thinking>real chain of thought</thinking>The answer."
        assert R.strip_leading_empty_thinking_shell(text) == text

    def test_mid_text_shell_untouched(self):
        from core.response_parser import ResponseParser as R
        text = "Discussing the <thinking></thinking> marker pair itself."
        assert R.strip_leading_empty_thinking_shell(text) == text

    def test_live_chunk_sequence_ends_clean(self):
        # chunk-by-chunk reproduction of the 13:06 turn: after shell strip +
        # edge-token strip, what the display yields never contains tags or sep
        from core.response_parser import ResponseParser as R
        buf = ""
        for chunk in ["<thinking>", "</thinking>", "<|sep|>Nice", " — pushed."]:
            buf += chunk
            stripped = R.strip_leading_empty_thinking_shell(buf)
            if stripped != buf:
                buf = stripped
            visible = R.strip_trailing_stream_artifact(buf)
            if buf not in ("<thinking>", ""):  # shell-only states show 💭, not text
                assert "<thinking>" not in visible
                assert "<|sep|>" not in visible
        assert visible == "Nice — pushed."

    def test_handlers_wire_shell_strip_and_display_strip(self):
        import inspect
        import gui.handlers as h
        src = inspect.getsource(h)
        assert "strip_leading_empty_thinking_shell" in src
        assert src.count("strip_trailing_stream_artifact(display_output)") >= 4


class TestDegenerateStreamWatchdog:
    """2026-09-01: enhanced and agentic streaming loops now detect degenerate
    (repeating garbage) streams and abort them before storage, same as
    insight mode's watchdog."""

    def test_watchdog_shape_check_present_no_wall_clock(self):
        """Both loops run the degenerate-shape check; NEITHER has a wall-clock
        arm — the agentic loop legitimately spends minutes in rounds before
        the first response chunk (211s live turns on record), so a duration
        ceiling would discard real answers. Insight's 240s ceiling is
        insight-only by design."""
        import inspect
        from gui.handlers import _run_enhanced, _run_agentic_search

        enhanced_src = inspect.getsource(_run_enhanced)
        assert "looks_degenerate_stream" in enhanced_src, (
            "enhanced mode should call looks_degenerate_stream"
        )
        assert "_STREAM_MAX_S" not in enhanced_src, (
            "enhanced mode must NOT have a wall-clock watchdog arm"
        )

        agentic_src = inspect.getsource(_run_agentic_search)
        assert "looks_degenerate_stream" in agentic_src, (
            "agentic mode should call looks_degenerate_stream"
        )
        assert "_STREAM_MAX_S" not in agentic_src, (
            "agentic mode must NOT have a wall-clock watchdog arm"
        )

    def test_watchdog_returns_before_storage(self):
        """A tripped watchdog must return early so nothing is stored."""
        import inspect
        from gui.handlers import _run_enhanced, _run_agentic_search

        # Enhanced mode's watchdog yields an abort message and returns early
        enhanced_src = inspect.getsource(_run_enhanced)
        assert "Stream aborted by watchdog" in enhanced_src, (
            "enhanced mode watchdog should log stream abort"
        )

        # Agentic mode's watchdog sets ctx.handled and returns early
        agentic_src = inspect.getsource(_run_agentic_search)
        assert "Stream aborted by watchdog" in agentic_src, (
            "agentic watchdog should log stream abort"
        )
        assert "ctx.handled = True" in agentic_src, (
            "agentic watchdog paths should set ctx.handled before return"
        )


class TestDocGenAndSelfNoteArtifactStrip:
    """2026-09-01: doc-gen and self-note paths receive LLM output that can
    contain trailing artifacts (kimi-3 lone 'e', edge <|sep|> tokens).
    Document body and self-note summaries are now stripped before storage."""

    def test_document_generator_strips_trailing_artifact(self):
        """The document_generator.py now strips artifacts from markdown
        before validation and assembly."""
        # This is a unit test that the strip is called at the right place
        import inspect
        from knowledge.document_generator import DocumentGenerator

        src = inspect.getsource(DocumentGenerator.generate)
        # Check that the artifact strip is called on markdown before validation
        assert "ResponseParser.strip_trailing_stream_artifact(markdown" in src, (
            "DocumentGenerator.generate should strip artifacts from markdown "
            "before validation and assembly"
        )

    def test_daemon_notes_summary_strips_artifact(self):
        """The daemon_notes_manager._generate_summary now strips artifacts
        from LLM output before returning."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from knowledge.daemon_notes_manager import DaemonNotesManager

        dnm = DaemonNotesManager(
            model_manager=MagicMock(),
            chroma_store=None
        )

        # Mock the model_manager's generate_once to return artifact-laden text
        mm_mock = MagicMock()
        mm_mock.generate_once = AsyncMock(
            return_value="Working note summary.e"  # kimi-3 artifact
        )

        result = asyncio.run(dnm._generate_summary("topic", model_manager=mm_mock))
        # The artifact should have been stripped
        assert result == "Working note summary.", (
            f"Expected artifact stripped; got {result!r}"
        )
        assert not result.endswith(".e")


class TestShutdownSummaryPathSanitized:
    """2026-08-22: shutdown _store_summary writes chroma via raw
    add_to_collection, bypassing chroma_store.add_summary (where the strip +
    junk check live) — two summaries landed with leading <|sep|> AFTER the
    fix shipped. The third path now sanitizes + junk-rejects at entry."""

    def _proc(self):
        from memory.shutdown_processor import ShutdownProcessor
        p = object.__new__(ShutdownProcessor)

        class Corpus:
            def __init__(self): self.added = []
            def get_summaries(self, n): return []
            def add_summary(self, node): self.added.append(node)

        class Store:
            def __init__(self): self.added = []
            def add_to_collection(self, coll, text, md):
                self.added.append((coll, text)); return "id1"

        p.corpus_manager = Corpus()
        p.chroma_store = Store()
        p.claim_index = None
        p.memory_coordinator = None
        return p

    def test_leading_sep_stripped_before_storage(self):
        p = self._proc()
        p._store_summary(
            "<|sep|>- The user sent their ODS accommodations notice, which "
            "includes extra time on exams, and has a meeting with ODS on "
            "Wednesday to discuss next steps for the semester.", 20, 0, 0, 5, [])
        assert p.chroma_store.added, "summary should store"
        coll, text = p.chroma_store.added[0]
        assert "<|sep|>" not in text
        assert p.corpus_manager.added[0]["content"].startswith("- The user")

    def test_junk_summary_rejected(self):
        p = self._proc()
        p._store_summary("[API Error] request failed", 20, 0, 0, 5, [])
        assert not p.chroma_store.added
        assert not p.corpus_manager.added

    def test_thinking_block_stripped_before_storage(self):
        """Test that leading thinking blocks are removed via sanitize_for_storage."""
        p = self._proc()
        # Summary with a thinking block at the start
        summary_with_thinking = (
            "<thinking>Let me think about the user's patterns...</thinking>"
            "Based on the data, the user has been improving steadily over the past few weeks."
        )
        p._store_summary(summary_with_thinking, 20, 0, 0, 5, [])
        assert p.chroma_store.added, "summary should store"
        coll, text = p.chroma_store.added[0]
        assert "<thinking>" not in text
        assert "</thinking>" not in text
        assert "Based on the data" in text
        assert p.corpus_manager.added[0]["content"] == text
