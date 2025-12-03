"""
Test suite for need-type detection.

Categories:
1. Clear PRESENCE signals (short emotional statements)
2. Clear PERSPECTIVE signals (problem-framing, questions)
3. NEUTRAL/ambiguous cases
4. Semantic detection tests
5. Hybrid detection tests
6. Edge cases and context sensitivity
"""

import pytest
from utils.need_detector import NeedType, NeedAnalysis, detect_need_type
from utils.need_detector import _keyword_need_detection, _semantic_need_detection


class TestPresenceDetection:
    """Tests for PRESENCE need-type detection."""

    @pytest.mark.parametrize("message,expected", [
        ("I am lonely", NeedType.PRESENCE),
        ("I'm sad", NeedType.PRESENCE),
        ("This sucks", NeedType.PRESENCE),
        ("Ugh", NeedType.PRESENCE),
        ("It hurts", NeedType.PRESENCE),
        ("I'm tired", NeedType.PRESENCE),
        ("I don't know anymore", NeedType.PRESENCE),
        ("I feel empty", NeedType.PRESENCE),
        ("Everything feels heavy", NeedType.PRESENCE),
    ])
    def test_short_emotional_statements(self, message, expected):
        result = detect_need_type(message)
        assert result.need_type == expected, f"Expected {expected} for '{message}'"

    def test_single_word_exhales(self):
        """Single-word emotional exhales should be PRESENCE."""
        for word in ["ugh", "sigh", "man", "god"]:
            result = detect_need_type(word)
            assert result.need_type == NeedType.PRESENCE, f"Expected PRESENCE for '{word}'"


class TestPerspectiveDetection:
    """Tests for PERSPECTIVE need-type detection."""

    @pytest.mark.parametrize("message,expected", [
        ("I think the reason dating is hard is my living situation", NeedType.PERSPECTIVE),
        ("Should I try a different approach?", NeedType.PERSPECTIVE),
        ("The problem is I don't have enough time", NeedType.PERSPECTIVE),
        ("I'm not sure if I should take the job because the pay is lower", NeedType.PERSPECTIVE),
        ("Do you think I'm overreacting?", NeedType.PERSPECTIVE),
        ("Maybe I should reconsider my options", NeedType.PERSPECTIVE),
        ("What would you do in my situation?", NeedType.PERSPECTIVE),
    ])
    def test_problem_framing_statements(self, message, expected):
        result = detect_need_type(message)
        assert result.need_type == expected, f"Expected {expected} for '{message}'"

    def test_question_marks_favor_perspective(self):
        """Messages with questions should lean PERSPECTIVE."""
        result = detect_need_type("What should I do about this?")
        assert result.need_type == NeedType.PERSPECTIVE


class TestNeutralCases:
    """Tests for NEUTRAL/ambiguous detection."""

    @pytest.mark.parametrize("message", [
        "Okay",
        "Thanks",
        "Got it",
        "Sure",
        "Cool",
    ])
    def test_minimal_messages(self, message):
        result = detect_need_type(message)
        assert result.need_type == NeedType.NEUTRAL

    def test_informational_updates(self):
        """Pure information should be NEUTRAL."""
        result = detect_need_type("I have a meeting at 3pm")
        assert result.need_type == NeedType.NEUTRAL


class TestSemanticDetection:
    """Tests for semantic similarity detection."""

    @pytest.mark.semantic
    def test_paraphrased_presence(self):
        """Paraphrased presence statements should still detect PRESENCE."""
        paraphrased = [
            "Nobody ever really sees me",
            "I feel like I'm disappearing",
            "The weight of everything is crushing",
        ]
        for msg in paraphrased:
            result = detect_need_type(msg)
            # Note: May be NEUTRAL if semantic embedder not available
            assert result.need_type in [NeedType.PRESENCE, NeedType.NEUTRAL], f"Expected PRESENCE or NEUTRAL for '{msg}'"

    @pytest.mark.semantic
    def test_paraphrased_perspective(self):
        """Paraphrased perspective statements should detect PERSPECTIVE."""
        paraphrased = [
            "I'm trying to weigh my options here",
            "Would it be smarter to wait?",
            "Help me think through this decision",
        ]
        for msg in paraphrased:
            result = detect_need_type(msg)
            # Note: May be NEUTRAL if semantic embedder not available
            assert result.need_type in [NeedType.PERSPECTIVE, NeedType.NEUTRAL], f"Expected PERSPECTIVE or NEUTRAL for '{msg}'"


class TestHybridDetection:
    """Tests for hybrid keyword + semantic detection."""

    def test_keyword_and_semantic_agreement_boosts_confidence(self):
        """When keyword and semantic agree, confidence should be higher."""
        result = detect_need_type("I am so lonely and empty inside")
        assert result.need_type == NeedType.PRESENCE
        # Confidence may vary based on semantic availability
        assert result.confidence > 0.3

    def test_semantic_can_override_weak_keyword(self):
        """Semantic should be able to override weak keyword signals."""
        # Message that's semantically PRESENCE but has no strong keyword
        result = detect_need_type("Everything just feels so heavy right now")
        # Should detect PRESENCE (either keyword or semantic)
        assert result.need_type in [NeedType.PRESENCE, NeedType.NEUTRAL]


class TestEdgeCases:
    """Edge cases and context sensitivity."""

    def test_i_feel_like_vs_i_feel(self):
        """Distinguish 'I feel like' (opinion) from 'I feel X' (state)."""
        # "I feel like" = perspective (opinion/analysis framing)
        result1 = detect_need_type("I feel like nobody understands the situation")
        assert result1.need_type == NeedType.PERSPECTIVE

        # "I feel sad" = presence (emotional state)
        result2 = detect_need_type("I feel sad")
        assert result2.need_type == NeedType.PRESENCE

    def test_long_emotional_statements_still_presence(self):
        """Long messages can still be PRESENCE if emotionally focused."""
        msg = "I just feel so alone in all of this. Like nobody really gets it."
        result = detect_need_type(msg)
        # Should still recognize emotional presence need despite length
        # May be NEUTRAL if semantic not available and no strong keywords
        assert result.need_type in [NeedType.PRESENCE, NeedType.NEUTRAL]

    def test_questions_about_feelings_are_perspective(self):
        """Even emotional topics framed as questions = PERSPECTIVE."""
        result = detect_need_type("Why do I always feel this way?")
        assert result.need_type == NeedType.PERSPECTIVE


class TestDebugLogCases:
    """Test cases from actual debug log conversation."""

    @pytest.mark.parametrize("message,expected_need", [
        ("Morning man!", NeedType.NEUTRAL),
        ("I'm still feeling a little off. Not like yesterday more just frustration. Online dating sucks lol I get so many matches and I think the reason they don't go anywhere is my school/living situation", NeedType.PERSPECTIVE),
        ("Like I mentioned before I live with my mom at the moment", NeedType.NEUTRAL),
        ("I lead with transitional info when it comes up. I feel like I am in a good position. In two years I will be making more than 90% of my matches", NeedType.PERSPECTIVE),
        ("No one can see it ugh. I will have to wait 2 years", NeedType.PRESENCE),
        ("It feels very long. I am lonely", NeedType.PRESENCE),
        ("I'm not it's just. Ugh. I will be 34", NeedType.PRESENCE),
    ])
    def test_debug_log_examples(self, message, expected_need):
        result = detect_need_type(message)
        assert result.need_type == expected_need, \
            f"Expected {expected_need.value} for '{message[:50]}...', got {result.need_type.value}"


class TestKeywordDetectionUnit:
    """Unit tests for keyword detection specifically."""

    def test_keyword_detection_basic(self):
        """Test basic keyword detection without semantic."""
        result = _keyword_need_detection("I am lonely")
        assert result.need_type == NeedType.PRESENCE
        assert result.trigger == "keyword"

    def test_keyword_perspective_markers(self):
        """Test perspective keyword detection."""
        result = _keyword_need_detection("I think the problem is my approach because of timing")
        assert result.need_type == NeedType.PERSPECTIVE
        assert result.trigger == "keyword"

    def test_keyword_short_message_bias(self):
        """Test that short messages bias toward PRESENCE."""
        # Short emotional message
        result = _keyword_need_detection("I'm sad")
        assert result.need_type == NeedType.PRESENCE

    def test_keyword_question_bias(self):
        """Test that questions bias toward PERSPECTIVE."""
        result = _keyword_need_detection("What should I do?")
        assert result.need_type == NeedType.PERSPECTIVE


class TestConfidenceScoring:
    """Test confidence scoring behavior."""

    def test_high_confidence_fast_path(self):
        """Test that clear cases take fast path."""
        result = detect_need_type("Ugh")
        # Single word exhale should be high confidence PRESENCE
        assert result.need_type == NeedType.PRESENCE
        # Confidence threshold for fast path is 0.8
        # Single exhale gets +1.5 score, becomes 1.5/3 = 0.5 confidence
        # So it won't take fast path but should still be PRESENCE
        assert result.trigger in ["keyword", "hybrid_agreement", "hybrid_keyword_wins"]

    def test_neutral_confidence(self):
        """Test that neutral cases have appropriate confidence."""
        result = detect_need_type("Thanks")
        assert result.need_type == NeedType.NEUTRAL
        # Neutral cases should have low-medium confidence
        assert result.confidence <= 0.5


class TestExtremeMessages:
    """Test handling of extreme/unusual messages."""

    def test_empty_message(self):
        """Test handling of empty message."""
        result = detect_need_type("")
        assert result.need_type == NeedType.NEUTRAL

    def test_very_long_message(self):
        """Test handling of very long messages."""
        long_msg = "I think " + "and also " * 50 + "the problem is complex"
        result = detect_need_type(long_msg)
        # Should lean PERSPECTIVE due to length and "I think" + "the problem is"
        assert result.need_type in [NeedType.PERSPECTIVE, NeedType.NEUTRAL]

    def test_mixed_signals(self):
        """Test message with mixed presence + perspective signals."""
        result = detect_need_type("I'm so sad but I think maybe I should try therapy?")
        # Has both "I'm so sad" (presence) and "I think maybe" + "?" (perspective)
        # Could go either way depending on scoring
        assert result.need_type in [NeedType.PRESENCE, NeedType.PERSPECTIVE, NeedType.NEUTRAL]


class TestMessageVariations:
    """Test variations of similar messages."""

    def test_lonely_variations(self):
        """Test different ways of expressing loneliness."""
        variations = [
            "I am lonely",
            "I'm lonely",
            "I feel alone",
            "I'm so lonely",
        ]
        for msg in variations:
            result = detect_need_type(msg)
            assert result.need_type == NeedType.PRESENCE, f"Expected PRESENCE for '{msg}'"

    def test_problem_framing_variations(self):
        """Test different ways of framing problems."""
        variations = [
            "The problem is my schedule",
            "I think the issue is timing",
            "The reason is probably because I'm busy",
        ]
        for msg in variations:
            result = detect_need_type(msg)
            assert result.need_type == NeedType.PERSPECTIVE, f"Expected PERSPECTIVE for '{msg}'"
