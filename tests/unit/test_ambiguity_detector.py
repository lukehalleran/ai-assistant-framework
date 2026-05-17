"""
Unit tests for cross-session ambiguity detection.
"""

import pytest
from datetime import datetime
from core.ambiguity_detector import (
    AmbiguityDetector,
    _extract_referential_phrases,
)


def _make_conv(query, timestamp, content_type=""):
    """Helper to create a mock conversation entry."""
    mem = {"query": query, "timestamp": timestamp}
    if content_type:
        mem["metadata"] = {"content_type": content_type}
    return mem


class TestAmbiguityDetector:

    def test_cross_session_match_triggers(self):
        """Phrase appearing in two different sessions should trigger."""
        convs = [
            _make_conv("Not entirely alone is beautiful", "2026-05-17T10:00:00"),
            _make_conv("The ending says not entirely alone", "2026-05-16T12:00:00"),
        ]
        result = AmbiguityDetector.detect(
            "Not entirely alone is my favorite",
            convs,
        )
        assert result.is_ambiguous is True
        assert "not entirely alone" in result.ambiguous_phrase.lower()
        assert len(result.matching_entries) >= 2

    def test_same_session_no_trigger(self):
        """Phrase in same session should NOT trigger."""
        convs = [
            _make_conv("Not entirely alone is beautiful", "2026-05-17T10:00:00"),
            _make_conv("The ending says not entirely alone", "2026-05-17T10:15:00"),
        ]
        result = AmbiguityDetector.detect(
            "Not entirely alone is my favorite",
            convs,
        )
        assert result.is_ambiguous is False

    def test_long_message_no_trigger(self):
        """Long messages self-disambiguate and should not trigger."""
        convs = [
            _make_conv("Not entirely alone", "2026-05-17T10:00:00"),
            _make_conv("Not entirely alone again", "2026-05-16T12:00:00"),
        ]
        long_msg = "Not entirely alone " + " ".join(["word"] * 60)
        result = AmbiguityDetector.detect(long_msg, convs)
        assert result.is_ambiguous is False

    def test_no_match_no_trigger(self):
        """Phrase not in any conversation should not trigger."""
        convs = [
            _make_conv("Talked about the weather", "2026-05-17T10:00:00"),
            _make_conv("Discussed benchmarks", "2026-05-16T12:00:00"),
        ]
        result = AmbiguityDetector.detect(
            "Ananke is my favorite",
            convs,
        )
        assert result.is_ambiguous is False

    def test_empty_conversations(self):
        result = AmbiguityDetector.detect("something", [])
        assert result.is_ambiguous is False

    def test_empty_message(self):
        convs = [_make_conv("hello", "2026-05-17T10:00:00")]
        result = AmbiguityDetector.detect("", convs)
        assert result.is_ambiguous is False

    def test_disambiguation_note_format(self):
        """The note should contain the phrase and session labels."""
        convs = [
            _make_conv("Not entirely alone poem", "2026-05-17T10:00:00"),
            _make_conv("The line not entirely alone", "2026-05-16T12:00:00"),
        ]
        result = AmbiguityDetector.detect(
            "Not entirely alone is my favorite",
            convs,
        )
        assert result.is_ambiguous is True
        assert "DISAMBIGUATION NOTE" in result.disambiguation_note
        assert "not entirely alone" in result.disambiguation_note.lower()

    def test_quoted_phrase_extraction(self):
        """Quoted phrases should be detected."""
        convs = [
            _make_conv('The song "Cosmic Disaster" was great', "2026-05-17T10:00:00"),
            _make_conv('I keep thinking about Cosmic Disaster', "2026-05-16T12:00:00"),
        ]
        result = AmbiguityDetector.detect(
            'I love "Cosmic Disaster"',
            convs,
        )
        assert result.is_ambiguous is True

    def test_content_type_in_note(self):
        """Disambiguation note should mention content type if available."""
        convs = [
            _make_conv("Not entirely alone lyrics", "2026-05-17T10:00:00", content_type="lyrics"),
            _make_conv("Not entirely alone discussion", "2026-05-16T12:00:00"),
        ]
        result = AmbiguityDetector.detect(
            "Not entirely alone is my favorite",
            convs,
        )
        if result.is_ambiguous:
            assert "lyrics" in result.disambiguation_note.lower()


class TestExtractReferentialPhrases:

    def test_quoted_phrases(self):
        phrases = _extract_referential_phrases('I love "Not Entirely Alone"')
        assert any("Not Entirely Alone" in p for p in phrases)

    def test_favorite_pattern(self):
        phrases = _extract_referential_phrases("Not Entirely Alone is my favorite")
        assert len(phrases) >= 1

    def test_title_case_sequence(self):
        phrases = _extract_referential_phrases("What about Not Entirely Alone?")
        assert any("Not Entirely Alone" in p for p in phrases)

    def test_normal_message_no_phrases(self):
        phrases = _extract_referential_phrases("how are the benchmarks looking")
        assert len(phrases) == 0

    def test_short_phrase_no_extract(self):
        phrases = _extract_referential_phrases("ok")
        assert len(phrases) == 0
