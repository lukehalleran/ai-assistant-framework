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
