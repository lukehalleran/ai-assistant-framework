"""
Tests for core/intent_classifier.py — Query Intent Classifier.

Covers:
  - All 9 intent types with representative queries
  - Regex pattern confidence levels
  - Tone-level emotional bias
  - STM refinement (low-confidence upgrade)
  - Edge cases (empty queries, very short, ambiguous)
  - Profile overrides populated correctly
  - IntentResult properties
"""

import pytest
from core.intent_classifier import (
    IntentClassifier,
    IntentResult,
    IntentType,
    _PROFILES,
)


@pytest.fixture
def classifier():
    return IntentClassifier()


# ═══════════════════════════════════════════════════════════════════════════
# Basic Classification — Happy Path
# ═══════════════════════════════════════════════════════════════════════════

class TestCasualSocial:
    def test_greeting_hello(self, classifier):
        r = classifier.classify("Hello")
        assert r.intent == IntentType.CASUAL_SOCIAL
        assert r.confidence >= 0.90

    def test_greeting_hey(self, classifier):
        r = classifier.classify("hey!")
        assert r.intent == IntentType.CASUAL_SOCIAL
        assert r.confidence >= 0.90

    def test_greeting_thanks(self, classifier):
        r = classifier.classify("thanks")
        assert r.intent == IntentType.CASUAL_SOCIAL
        assert r.confidence >= 0.90

    def test_greeting_good_morning(self, classifier):
        r = classifier.classify("Good morning!")
        assert r.intent == IntentType.CASUAL_SOCIAL
        assert r.confidence >= 0.90

    def test_short_ok(self, classifier):
        r = classifier.classify("okay")
        assert r.intent == IntentType.CASUAL_SOCIAL

    def test_bye(self, classifier):
        r = classifier.classify("bye")
        assert r.intent == IntentType.CASUAL_SOCIAL
        assert r.confidence >= 0.90

    def test_lol(self, classifier):
        r = classifier.classify("lol")
        assert r.intent == IntentType.CASUAL_SOCIAL

    def test_has_low_retrieval_counts(self, classifier):
        r = classifier.classify("hi")
        assert r.retrieval_overrides.get("max_mems", 99) <= 5
        assert r.retrieval_overrides.get("max_wiki", 99) == 0
        assert r.retrieval_overrides.get("max_proposals", 99) == 0

    def test_has_high_gate_threshold(self, classifier):
        r = classifier.classify("hey")
        assert r.gate_threshold_override is not None
        assert r.gate_threshold_override >= 0.60


class TestEmotionalSupport:
    def test_i_feel_sad(self, classifier):
        r = classifier.classify("I feel so sad today")
        assert r.intent == IntentType.EMOTIONAL_SUPPORT
        assert r.confidence >= 0.85

    def test_im_stressed(self, classifier):
        r = classifier.classify("I'm really stressed out")
        assert r.intent == IntentType.EMOTIONAL_SUPPORT
        assert r.confidence >= 0.85

    def test_cant_cope(self, classifier):
        r = classifier.classify("I can't cope with everything")
        assert r.intent == IntentType.EMOTIONAL_SUPPORT
        assert r.confidence >= 0.85

    def test_struggling(self, classifier):
        r = classifier.classify("I'm struggling right now")
        assert r.intent == IntentType.EMOTIONAL_SUPPORT

    def test_need_help_emotional(self, classifier):
        r = classifier.classify("Please help me, I'm scared")
        assert r.intent == IntentType.EMOTIONAL_SUPPORT

    def test_overwhelmed(self, classifier):
        r = classifier.classify("I'm feeling overwhelmed by everything")
        assert r.intent == IntentType.EMOTIONAL_SUPPORT
        assert r.confidence >= 0.85

    def test_truth_weight_low(self, classifier):
        """Emotional queries shouldn't prioritize truth over empathy."""
        r = classifier.classify("I feel broken inside")
        assert r.weight_overrides.get("truth", 1.0) <= 0.15

    def test_continuity_weight_high(self, classifier):
        """Emotional queries should value conversational continuity."""
        r = classifier.classify("I feel so lonely right now")
        assert r.intent == IntentType.EMOTIONAL_SUPPORT
        assert r.weight_overrides.get("continuity", 0.0) >= 0.20


class TestFactualRecall:
    def test_whats_my_name(self, classifier):
        r = classifier.classify("What's my sister's name?")
        assert r.intent == IntentType.FACTUAL_RECALL
        assert r.confidence >= 0.85

    def test_do_you_remember(self, classifier):
        r = classifier.classify("Do you remember my birthday?")
        assert r.intent == IntentType.FACTUAL_RECALL

    def test_i_told_you(self, classifier):
        r = classifier.classify("I told you about my project, remember?")
        assert r.intent == IntentType.FACTUAL_RECALL

    def test_what_did_i_tell(self, classifier):
        r = classifier.classify("What did I tell you about my diet?")
        assert r.intent == IntentType.FACTUAL_RECALL

    def test_truth_weight_high(self, classifier):
        """Factual queries should prioritize verified truth."""
        r = classifier.classify("What's my dog's name?")
        assert r.weight_overrides.get("truth", 0.0) >= 0.25


class TestTemporalRecall:
    def test_last_week(self, classifier):
        r = classifier.classify("What did we talk about last week?")
        assert r.intent == IntentType.TEMPORAL_RECALL

    def test_yesterday(self, classifier):
        r = classifier.classify("What happened yesterday?")
        assert r.intent == IntentType.TEMPORAL_RECALL

    def test_remember_when(self, classifier):
        r = classifier.classify("Remember when we discussed neural networks?")
        assert r.intent == IntentType.TEMPORAL_RECALL

    def test_over_time(self, classifier):
        r = classifier.classify("How has my mood changed over time?")
        assert r.intent == IntentType.TEMPORAL_RECALL

    def test_recency_weight_high(self, classifier):
        r = classifier.classify("What were we discussing last month?")
        assert r.weight_overrides.get("recency", 0.0) >= 0.35

    def test_more_summaries(self, classifier):
        r = classifier.classify("What happened last week?")
        assert r.retrieval_overrides.get("max_summaries", 0) >= 10


class TestTechnicalHelp:
    def test_how_do_i(self, classifier):
        r = classifier.classify("How do I fix this memory leak?")
        assert r.intent == IntentType.TECHNICAL_HELP

    def test_error_message(self, classifier):
        r = classifier.classify("I'm getting an error when running pytest")
        assert r.intent == IntentType.TECHNICAL_HELP

    def test_debug(self, classifier):
        r = classifier.classify("Help me debug this function")
        assert r.intent == IntentType.TECHNICAL_HELP

    def test_not_working(self, classifier):
        r = classifier.classify("The API endpoint isn't working")
        assert r.intent == IntentType.TECHNICAL_HELP

    def test_relevance_weight_high(self, classifier):
        r = classifier.classify("How do I fix this bug?")
        assert r.weight_overrides.get("relevance", 0.0) >= 0.40


class TestMetaConversational:
    def test_what_do_you_know(self, classifier):
        r = classifier.classify("What do you know about me?")
        assert r.intent == IntentType.META_CONVERSATIONAL

    def test_your_memory(self, classifier):
        r = classifier.classify("How is your memory system working?")
        assert r.intent == IntentType.META_CONVERSATIONAL

    def test_what_have_you_learned(self, classifier):
        r = classifier.classify("What have you learned about my preferences?")
        assert r.intent == IntentType.META_CONVERSATIONAL

    def test_show_profile(self, classifier):
        r = classifier.classify("Show me my profile data")
        assert r.intent == IntentType.META_CONVERSATIONAL


class TestProjectWork:
    def test_lets_build(self, classifier):
        r = classifier.classify("Let's build a new authentication module")
        assert r.intent == IntentType.PROJECT_WORK

    def test_add_feature(self, classifier):
        r = classifier.classify("Add a feature for user notifications")
        assert r.intent == IntentType.PROJECT_WORK

    def test_file_reference(self, classifier):
        r = classifier.classify("Update the logic in auth_handler.py")
        assert r.intent == IntentType.PROJECT_WORK

    def test_pull_request(self, classifier):
        r = classifier.classify("Create a PR for this change")
        assert r.intent == IntentType.PROJECT_WORK

    def test_git_commits_boosted(self, classifier):
        r = classifier.classify("Let's work on the codebase")
        assert r.retrieval_overrides.get("max_git_commits", 0) >= 10


class TestCreativeExploration:
    def test_brainstorm(self, classifier):
        r = classifier.classify("Let's brainstorm some ideas for the UI")
        assert r.intent == IntentType.CREATIVE_EXPLORATION

    def test_what_if(self, classifier):
        r = classifier.classify("What if we combined the two approaches?")
        assert r.intent == IntentType.CREATIVE_EXPLORATION

    def test_explore(self, classifier):
        r = classifier.classify("Help me explore different design patterns")
        assert r.intent == IntentType.CREATIVE_EXPLORATION


class TestGeneral:
    def test_ambiguous_query(self, classifier):
        """Queries that don't match any strong pattern → GENERAL."""
        r = classifier.classify("Tell me about the different species of parrots and their habitats around the world")
        assert r.intent == IntentType.GENERAL

    def test_general_has_empty_overrides(self, classifier):
        r = classifier.classify("Tell me about machine learning algorithms and their applications")
        if r.intent == IntentType.GENERAL:
            assert r.weight_overrides == {}
            assert r.retrieval_overrides == {}


# ═══════════════════════════════════════════════════════════════════════════
# Tone-Level Emotional Bias
# ═══════════════════════════════════════════════════════════════════════════

class TestToneBias:
    def test_high_tone_general_becomes_emotional(self, classifier):
        """HIGH tone with ambiguous query → EMOTIONAL_SUPPORT."""
        r = classifier.classify("I don't know anymore", tone_level="HIGH")
        assert r.intent == IntentType.EMOTIONAL_SUPPORT

    def test_medium_tone_general_becomes_emotional(self, classifier):
        """MEDIUM tone with ambiguous query → EMOTIONAL_SUPPORT."""
        r = classifier.classify("things are rough", tone_level="MEDIUM")
        assert r.intent == IntentType.EMOTIONAL_SUPPORT

    def test_high_tone_boosts_emotional_confidence(self, classifier):
        """HIGH tone with already-emotional query → confidence boost."""
        r_base = classifier.classify("I'm feeling stressed")
        r_toned = classifier.classify("I'm feeling stressed", tone_level="HIGH")
        assert r_toned.confidence >= r_base.confidence

    def test_conversational_tone_no_bias(self, classifier):
        """CONVERSATIONAL tone doesn't bias toward emotional."""
        r = classifier.classify("hi there", tone_level="CONVERSATIONAL")
        assert r.intent == IntentType.CASUAL_SOCIAL

    def test_tone_doesnt_override_strong_match(self, classifier):
        """HIGH tone shouldn't override a strong factual match."""
        r = classifier.classify("What's my sister's name?", tone_level="HIGH")
        assert r.intent == IntentType.FACTUAL_RECALL


# ═══════════════════════════════════════════════════════════════════════════
# STM Refinement
# ═══════════════════════════════════════════════════════════════════════════

class TestSTMRefinement:
    def test_low_confidence_refined_by_stm(self, classifier):
        """Low-confidence GENERAL can be refined by STM intent."""
        r = classifier.classify("Can you go over that again?")
        # This might be GENERAL or low-confidence something
        if r.confidence < 0.50:
            refined = classifier.refine_with_stm(r, "Get practical solution to technical problem")
            assert refined.intent == IntentType.TECHNICAL_HELP
            assert refined.source == "stm_refined"
            assert refined.confidence >= 0.50

    def test_high_confidence_not_refined(self, classifier):
        """High-confidence results should not be overridden by STM."""
        r = classifier.classify("Hello!")
        assert r.confidence >= 0.90
        refined = classifier.refine_with_stm(r, "Get emotional support")
        assert refined.intent == r.intent  # unchanged
        assert refined.source == "regex"  # unchanged

    def test_stm_none_returns_original(self, classifier):
        r = classifier.classify("hmm")
        refined = classifier.refine_with_stm(r, None)
        assert refined is r

    def test_stm_empty_string_returns_original(self, classifier):
        r = classifier.classify("hmm")
        refined = classifier.refine_with_stm(r, "")
        assert refined is r

    def test_stm_no_keyword_match_returns_original(self, classifier):
        r = classifier.classify("hmm")
        original_intent = r.intent
        refined = classifier.refine_with_stm(r, "zxcvbnm asdfghjkl")
        assert refined.intent == original_intent

    def test_stm_factual_recall(self, classifier):
        r = IntentResult(intent=IntentType.GENERAL, confidence=0.30)
        refined = classifier.refine_with_stm(r, "Recall information about user's pet")
        assert refined.intent == IntentType.FACTUAL_RECALL

    def test_stm_temporal(self, classifier):
        r = IntentResult(intent=IntentType.GENERAL, confidence=0.30)
        refined = classifier.refine_with_stm(r, "Review history of previous conversations")
        assert refined.intent == IntentType.TEMPORAL_RECALL

    def test_stm_emotional(self, classifier):
        r = IntentResult(intent=IntentType.GENERAL, confidence=0.30)
        refined = classifier.refine_with_stm(r, "Provide emotional support and comfort")
        assert refined.intent == IntentType.EMOTIONAL_SUPPORT


# ═══════════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_query(self, classifier):
        r = classifier.classify("")
        assert r.intent == IntentType.CASUAL_SOCIAL
        assert r.confidence >= 0.90

    def test_whitespace_only(self, classifier):
        r = classifier.classify("   ")
        assert r.intent == IntentType.CASUAL_SOCIAL

    def test_single_word_question(self, classifier):
        r = classifier.classify("What?")
        # Could be various — just verify it doesn't crash
        assert isinstance(r.intent, IntentType)

    def test_very_long_query(self, classifier):
        long_q = "How do I " + "fix " * 200 + "this bug?"
        r = classifier.classify(long_q)
        assert r.intent == IntentType.TECHNICAL_HELP

    def test_mixed_signals(self, classifier):
        """Query that matches multiple patterns → highest confidence wins."""
        r = classifier.classify("I'm feeling stressed about debugging this error")
        # Both emotional and technical patterns match
        assert r.intent in (IntentType.EMOTIONAL_SUPPORT, IntentType.TECHNICAL_HELP)

    def test_case_insensitive(self, classifier):
        r1 = classifier.classify("HELLO")
        r2 = classifier.classify("hello")
        assert r1.intent == r2.intent


# ═══════════════════════════════════════════════════════════════════════════
# Profile Overrides
# ═══════════════════════════════════════════════════════════════════════════

class TestProfiles:
    def test_all_intents_have_profiles(self):
        """Every IntentType must have a profile in _PROFILES."""
        for intent in IntentType:
            assert intent in _PROFILES, f"Missing profile for {intent.value}"

    def test_profiles_have_required_keys(self):
        for intent, profile in _PROFILES.items():
            assert "weights" in profile, f"Missing 'weights' in {intent.value}"
            assert "retrieval" in profile, f"Missing 'retrieval' in {intent.value}"
            assert "gate" in profile, f"Missing 'gate' in {intent.value}"

    def test_weight_sums_approximately_one(self):
        """Non-empty weight overrides should roughly sum to 1.0."""
        for intent, profile in _PROFILES.items():
            w = profile["weights"]
            if w:
                total = sum(w.values())
                assert 0.95 <= total <= 1.05, (
                    f"{intent.value} weights sum to {total:.2f}, expected ~1.0"
                )

    def test_result_carries_overrides(self, classifier):
        r = classifier.classify("What's my name?")
        assert isinstance(r.weight_overrides, dict)
        assert isinstance(r.retrieval_overrides, dict)

    def test_is_high_confidence_property(self):
        r = IntentResult(intent=IntentType.GENERAL, confidence=0.80)
        assert r.is_high_confidence is True

        r2 = IntentResult(intent=IntentType.GENERAL, confidence=0.50)
        assert r2.is_high_confidence is False


# ═══════════════════════════════════════════════════════════════════════════
# IntentType Enum
# ═══════════════════════════════════════════════════════════════════════════

class TestIntentTypeEnum:
    def test_string_values(self):
        assert IntentType.FACTUAL_RECALL.value == "factual_recall"
        assert IntentType.CASUAL_SOCIAL.value == "casual_social"
        assert IntentType.GENERAL.value == "general"

    def test_is_string_enum(self):
        assert isinstance(IntentType.FACTUAL_RECALL, str)
        assert IntentType.FACTUAL_RECALL == "factual_recall"

    def test_all_nine_types(self):
        assert len(IntentType) == 9
