"""
Unit tests for content type detection.
"""

import pytest
from core.content_type_detector import detect_content_type, _looks_like_lyrics


class TestDetectContentType:

    def test_spoken_prefix_is_poem(self):
        text = "[Spoken]\nAnanke, god of compulsion\nfrom the Orphic pantheon"
        result = detect_content_type(text)
        assert result.content_type == "poem"
        assert result.confidence >= 0.90

    def test_spoken_prefix_case_insensitive(self):
        text = "[spoken]\nSome spoken word piece"
        result = detect_content_type(text)
        assert result.content_type == "poem"

    def test_code_fence(self):
        text = "Check this out:\n```python\ndef hello():\n    print('hi')\n```"
        result = detect_content_type(text)
        assert result.content_type == "code"
        assert result.confidence >= 0.85

    def test_dream_narrative(self):
        text = "I had a dream last night that I was flying over the city"
        result = detect_content_type(text)
        assert result.content_type == "dream"

    def test_dreamt_variation(self):
        text = "I dreamt about my old school and all the hallways were different"
        result = detect_content_type(text)
        assert result.content_type == "dream"

    def test_shared_message_mom(self):
        text = "my mom said: honey please come visit this weekend we miss you"
        result = detect_content_type(text)
        assert result.content_type == "message"

    def test_shared_message_friend(self):
        text = "my friend texted me this morning about the concert"
        result = detect_content_type(text)
        assert result.content_type == "message"

    def test_share_preamble_with_lyrics(self):
        text = "check this out:\nWake up older\nNot older or wiser but probably safe\nTake your time here\nThings that you'll want"
        result = detect_content_type(text)
        assert result.content_type == "lyrics"

    def test_multiline_lyrics_no_preamble(self):
        text = (
            "Smile for the camera flash\n"
            "Blinding\n"
            "You see shooting stars in the carpet\n"
            "The threads are unwinding\n"
            "The clocks are unticking\n"
            "The boxes are moving\n"
            "Cause something is kicking inside of them"
        )
        result = detect_content_type(text)
        assert result.content_type == "lyrics"

    def test_normal_conversation_no_detection(self):
        text = "How are the benchmark results looking?"
        result = detect_content_type(text)
        assert result.content_type == ""

    def test_short_message_no_detection(self):
        text = "yeah that makes sense"
        result = detect_content_type(text)
        assert result.content_type == ""

    def test_question_not_lyrics(self):
        text = "What do you think about this?\nIs it any good?\nShould I try it?\nLet me know?"
        result = detect_content_type(text)
        assert result.content_type == ""

    def test_empty_input(self):
        result = detect_content_type("")
        assert result.content_type == ""

    def test_none_input(self):
        result = detect_content_type(None)
        assert result.content_type == ""

    def test_long_prose_not_lyrics(self):
        text = (
            "So today I went to the store and picked up some groceries. "
            "Then I came home and started working on the project. "
            "After about two hours I took a break and went for a walk. "
            "The weather was nice so I stayed out for a while."
        )
        result = detect_content_type(text)
        assert result.content_type == ""


class TestLooksLikeLyrics:

    def test_short_lines_are_lyrics(self):
        text = "line one\nline two\nline three\nline four\nline five"
        assert _looks_like_lyrics(text) is True

    def test_too_few_lines(self):
        text = "line one\nline two"
        assert _looks_like_lyrics(text) is False

    def test_long_lines_not_lyrics(self):
        text = "\n".join(["x" * 100 for _ in range(5)])
        assert _looks_like_lyrics(text) is False

    def test_many_questions_not_lyrics(self):
        text = "Is this right?\nWhat about this?\nCan you help?\nWhy not?"
        assert _looks_like_lyrics(text) is False

    def test_one_rhetorical_question_ok(self):
        text = "Walking down the road\nWhere does it end?\nThe sun is setting\nTime moves on\nForever"
        assert _looks_like_lyrics(text) is True


class TestTitleAttribution:

    def test_spoken_with_by_attribution(self):
        text = "[Spoken] A poem by Walt Whitman\nI celebrate myself\nAnd sing myself"
        result = detect_content_type(text)
        assert result.content_type == "poem"
        assert "Whitman" in result.attribution_hint

    def test_no_attribution(self):
        text = "[Spoken]\nSome anonymous poem\nWith no author named"
        result = detect_content_type(text)
        assert result.attribution_hint == ""
