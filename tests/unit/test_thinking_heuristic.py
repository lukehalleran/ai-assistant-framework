"""
Tests for likely_untagged_thinking() streaming heuristic and the
streaming-loop integration behavior it drives in handlers.py.

Covers:
- True positives: multi-pattern thinking dumps are detected
- True negatives: normal responses, short text, single-pattern text
- Bail-out: responses >300 chars are never suppressed (false-positive guard)
- Edge cases: thinking at boundary, patterns only in later lines
- _detect_untagged_thinking consistency: if the full parser splits,
  the heuristic should also have fired
"""

import pytest
from core.response_parser import ResponseParser


# ── likely_untagged_thinking: true positives ──

class TestLikelyUntaggedThinkingPositives:
    """Cases that SHOULD be detected as untagged thinking."""

    def test_classic_two_pattern_thinking(self):
        text = (
            "I should think about what the user is asking.\n"
            "The user wants to know about Python.\n"
            "Let me consider the best way to explain this."
        )
        assert ResponseParser.likely_untagged_thinking(text) is True

    def test_meta_reasoning_plus_user_reference(self):
        text = (
            "Let me check what I know about this topic.\n"
            "The user is asking about machine learning.\n"
            "I need to be careful here."
        )
        assert ResponseParser.likely_untagged_thinking(text) is True

    def test_planning_plus_meta(self):
        text = (
            "How should I approach this question?\n"
            "I should mention the key differences.\n"
            "This is a technical question."
        )
        assert ResponseParser.likely_untagged_thinking(text) is True

    def test_user_reference_plus_strategy(self):
        text = (
            "The user asked about their project timeline.\n"
            "What would actually be useful here is a breakdown.\n"
            "I could mention the deadline they set."
        )
        assert ResponseParser.likely_untagged_thinking(text) is True

    def test_bullet_reasoning_plus_meta(self):
        text = (
            "I need to consider several things:\n"
            "- Explicitly the user wants dates\n"
            "- Temporal context matters here\n"
        )
        assert ResponseParser.likely_untagged_thinking(text) is True


# ── likely_untagged_thinking: true negatives ──

class TestLikelyUntaggedThinkingNegatives:
    """Cases that should NOT be detected as thinking."""

    def test_normal_response(self):
        text = (
            "Python is a great programming language for beginners.\n"
            "It has clean syntax and a large standard library.\n"
            "You can install it from python.org."
        )
        assert ResponseParser.likely_untagged_thinking(text) is False

    def test_short_text(self):
        assert ResponseParser.likely_untagged_thinking("I should check") is False

    def test_empty(self):
        assert ResponseParser.likely_untagged_thinking("") is False

    def test_none(self):
        assert ResponseParser.likely_untagged_thinking(None) is False

    def test_single_line(self):
        text = "I should think about this and the user wants an answer."
        assert ResponseParser.likely_untagged_thinking(text) is False

    def test_single_pattern_match(self):
        """One pattern hit is not enough."""
        text = (
            "I should mention that Python supports async/await.\n"
            "Here is how you use it:\n"
            "```python\nasync def main(): pass\n```"
        )
        assert ResponseParser.likely_untagged_thinking(text) is False

    def test_conversational_response_with_first_person(self):
        """Normal first-person language should not trigger."""
        text = (
            "I think that's a great idea!\n"
            "You could try using pandas for this.\n"
            "Here's an example of how to do it."
        )
        assert ResponseParser.likely_untagged_thinking(text) is False

    def test_response_with_let_me(self):
        """'Let me' in a normal response context."""
        text = (
            "Great question! Let me explain how this works.\n"
            "The HTTP protocol uses request-response pairs.\n"
            "Each request has a method like GET or POST."
        )
        # "Let me" alone matches pattern 0, but needs a second distinct pattern
        assert ResponseParser.likely_untagged_thinking(text) is False


# ── Bail-out behavior (streaming integration) ──

class TestStreamingBailout:
    """The 300-char bail-out prevents false-positive suppression."""

    def test_short_thinking_suppressed(self):
        """Under 300 chars with patterns: should suppress."""
        text = (
            "I should think about what the user is asking.\n"
            "The user wants to know about Python."
        )
        assert len(text) < 300
        assert ResponseParser.likely_untagged_thinking(text) is True

    def test_long_text_with_patterns_not_suppressed(self):
        """Over 300 chars: the streaming loop bails out regardless of heuristic.

        The heuristic itself doesn't enforce the 300-char limit (it's a pure
        pattern check), but handlers.py only applies it when len < 300.
        We test the condition handlers.py would check.
        """
        text = (
            "I should think about what the user is asking.\n"
            "The user wants to know about Python.\n"
            + "Here is a very long explanation. " * 20
        )
        assert len(text) > 300
        # Heuristic still fires (it doesn't know about length)
        assert ResponseParser.likely_untagged_thinking(text) is True
        # But the streaming guard would NOT suppress (len >= 300)
        would_suppress = len(text) < 300 and ResponseParser.likely_untagged_thinking(text)
        assert would_suppress is False

    def test_normal_long_response_never_suppressed(self):
        """Normal response that's long: never suppressed at any length."""
        text = (
            "Python is a great programming language.\n"
            "It was created by Guido van Rossum.\n"
            + "It supports many paradigms. " * 20
        )
        assert ResponseParser.likely_untagged_thinking(text) is False
        would_suppress = len(text) < 300 and ResponseParser.likely_untagged_thinking(text)
        assert would_suppress is False


# ── Consistency with _detect_untagged_thinking ──

class TestHeuristicConsistency:
    """If _detect_untagged_thinking splits text, likely_untagged_thinking
    should also have detected patterns (the heuristic is a superset check)."""

    def test_splittable_text_detected_by_both(self):
        """Text with a clean split should be caught by both methods."""
        text = (
            "I should think about this carefully.\n"
            "The user wants to know about their dog.\n"
            "Let me check my memory.\n"
            "\n"
            "Your dog Flapjack is a golden retriever! "
            "You've mentioned him several times before."
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        if thinking and answer:
            # If the full parser found a split, the fast check must also fire
            assert ResponseParser.likely_untagged_thinking(text) is True

    def test_no_split_no_detection(self):
        """Text with no thinking patterns: neither method fires."""
        text = (
            "Here is your answer about Python.\n"
            "It supports async/await since version 3.5.\n"
            "You can use asyncio.gather for parallelism."
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        assert thinking == ""
        assert ResponseParser.likely_untagged_thinking(text) is False


# ── parse_thinking_block integration ──

class TestParseThinkingBlockIntegration:
    """Ensure parse_thinking_block still works correctly for all modes."""

    def test_tagged_thinking(self):
        text = "<thinking>Let me analyze.</thinking>The answer is 42."
        thinking, answer = ResponseParser.parse_thinking_block(text)
        assert thinking == "Let me analyze."
        assert answer == "The answer is 42."

    def test_think_tag_variant(self):
        text = "<think>Planning my response.</think>Here you go."
        thinking, answer = ResponseParser.parse_thinking_block(text)
        assert thinking == "Planning my response."
        assert answer == "Here you go."

    def test_no_thinking(self):
        text = "Just a normal response with no thinking."
        thinking, answer = ResponseParser.parse_thinking_block(text)
        assert thinking == ""
        assert answer == text

    def test_output_wrapper(self):
        text = "Some reasoning here\n<output>The real answer.</output>"
        thinking, answer = ResponseParser.parse_thinking_block(text)
        assert answer == "The real answer."

    def test_has_incomplete_thinking_block(self):
        assert ResponseParser.has_incomplete_thinking_block("<thinking>partial") is True
        assert ResponseParser.has_incomplete_thinking_block("<thinking>done</thinking>answer") is False
        assert ResponseParser.has_incomplete_thinking_block("no tags here") is False
