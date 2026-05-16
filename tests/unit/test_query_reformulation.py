"""
Unit tests for MemoryRetriever._reformulate_for_embedding().

Validates that meta-framing is stripped for embedding lookup while
preserving entities, possessives, and falling back for deictic/empty cases.
"""

import pytest
from memory.memory_retriever import MemoryRetriever


reformulate = MemoryRetriever._reformulate_for_embedding


# -----------------------------------------------------------------------
# Positive reformulations — meta framing stripped, content preserved
# -----------------------------------------------------------------------


class TestPositiveReformulations:
    """Queries where meta framing should be stripped."""

    def test_what_do_you_know_about_health_fallback(self):
        """'my health' after stripping has only 1 content word — falls back."""
        q = "What do you know about my health?"
        assert reformulate(q) == q

    def test_what_do_you_know_about_health_situation_fallback(self):
        """'my health situation' has only 2 content words — falls back."""
        q = "What do you know about my health situation?"
        assert reformulate(q) == q

    def test_brainstorm_gift_ideas(self):
        result = reformulate("Help me brainstorm gift ideas for my sister Sarah")
        assert "sister sarah" in result.lower()
        assert "brainstorm" not in result.lower()

    def test_what_did_we_talk_about_running(self):
        result = reformulate("What did we talk about last week about running?")
        assert "running" in result.lower()
        assert "what did we talk about" not in result.lower()

    def test_what_if_redesigned_api_fallback(self):
        """'redesigned the API' has only 2 content words — falls back."""
        q = "What if we redesigned the API?"
        assert reformulate(q) == q

    def test_lets_explore_architecture(self):
        result = reformulate("Let's explore architecture for Kubernetes deploy")
        assert "kubernetes" in result.lower()
        assert "let's explore" not in result.lower()

    def test_what_do_you_know_about_pets(self):
        result = reformulate("What do you know about my dog Biscuit and cat Mochi?")
        assert "biscuit" in result.lower()
        assert "mochi" in result.lower()
        assert "what do you know" not in result.lower()

    def test_how_long_gym_fallback(self):
        """'going to the gym' has only 2 content words — falls back."""
        q = "How long have I been going to the gym?"
        assert reformulate(q) == q

    def test_what_have_i_been_doing_for_health(self):
        result = reformulate("What have I been doing for my health over time?")
        assert "health" in result.lower()
        assert "what have i been doing" not in result.lower()

    def test_tell_me_about_family(self):
        """'Tell me about my family' → 'my family' has only 1 content word → fallback."""
        q = "Tell me about my family"
        result = reformulate(q)
        # "family" is the only content word — too short, falls back
        assert result == q

    def test_do_you_remember_brother_fallback(self):
        """'my brother Auggie' has only 2 content words — falls back."""
        q = "Do you remember my brother Auggie?"
        assert reformulate(q) == q

    def test_temporal_phrases_preserved(self):
        """Temporal phrases stay — they provide useful embedding context."""
        result = reformulate("What did we discuss last month about the project?")
        assert "project" in result.lower()
        assert "last month" in result.lower()
        assert "what did we discuss" not in result.lower()

    def test_lets_brainstorm_dashboard(self):
        result = reformulate("Let's brainstorm creative ideas for improving the dashboard")
        assert "dashboard" in result.lower()
        assert "brainstorm" not in result.lower()


# -----------------------------------------------------------------------
# Fallback / no-change cases — should return original query
# -----------------------------------------------------------------------


class TestFallbackCases:
    """Queries where reformulation should fall back to original."""

    def test_deictic_what_about_that(self):
        q = "What about that?"
        assert reformulate(q) == q

    def test_deictic_explain_that(self):
        q = "Explain that"
        assert reformulate(q) == q

    def test_deictic_what_did_you_mean(self):
        q = "What did you mean?"
        assert reformulate(q) == q

    def test_deictic_tell_me_more(self):
        q = "Tell me more"
        assert reformulate(q) == q

    def test_generic_can_you_help(self):
        q = "Can you help me?"
        assert reformulate(q) == q

    def test_generic_do_you_remember(self):
        q = "Do you remember?"
        assert reformulate(q) == q

    def test_short_query(self):
        q = "Hi"
        assert reformulate(q) == q

    def test_empty_query(self):
        q = ""
        assert reformulate(q) == q

    def test_pure_temporal_no_topic(self):
        """'What did we discuss yesterday?' has no topic after stripping — fallback."""
        q = "What did we discuss yesterday?"
        result = reformulate(q)
        # Should fall back since stripping leaves nothing meaningful
        assert result == q

    def test_direct_content_query_unchanged(self):
        """Queries that are already content-focused should pass through."""
        q = "Tell me about machine learning and neural networks"
        result = reformulate(q)
        # "Tell me about" gets stripped but content remains
        assert "machine learning" in result.lower()
        assert "neural networks" in result.lower()


# -----------------------------------------------------------------------
# Entity / possessive preservation
# -----------------------------------------------------------------------


class TestEntityPreservation:
    """Named entities and possessives must survive reformulation."""

    def test_preserves_proper_names(self):
        result = reformulate("What do you know about my sister Sarah and brother Mike?")
        assert "sarah" in result.lower()
        assert "mike" in result.lower()

    def test_preserves_possessives(self):
        result = reformulate("What do you know about my dog Biscuit?")
        assert "my dog biscuit" in result.lower()

    def test_preserves_project_names(self):
        result = reformulate("What do you remember about Daemon?")
        assert "daemon" in result.lower()

    def test_preserves_place_names(self):
        result = reformulate("Tell me about Georgia Tech")
        assert "georgia tech" in result.lower()
