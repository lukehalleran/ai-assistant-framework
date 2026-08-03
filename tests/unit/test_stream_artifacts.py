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
