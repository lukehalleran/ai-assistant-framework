# tests/test_web_search_trigger.py
"""
Unit tests for WebSearchTrigger module.

Tests cover:
- Keyword matching for recency indicators
- Pattern matching for explicit search requests
- Fast-changing topic detection
- Crisis level suppression
- Confidence scoring
- Edge cases and boundary conditions
"""

import pytest
from unittest.mock import MagicMock, patch

# Import modules under test
from utils.web_search_trigger import (
    WebSearchDepth,
    WebSearchDecision,
    should_search_heuristic,
    analyze_for_web_search,
    get_search_decision_for_prompt,
    RECENCY_KEYWORDS_STRONG,
    RECENCY_KEYWORDS_MODERATE,
    NEWS_KEYWORDS,
    FAST_CHANGING_TOPICS,
    STATIC_TOPICS,
    EXPLICIT_SEARCH_PHRASES,
    SUPPRESSION_PATTERNS,
)


# ===== WebSearchDecision Tests =====

class TestWebSearchDecision:
    def test_decision_dataclass(self):
        """Test WebSearchDecision creation."""
        decision = WebSearchDecision(
            should_search=True,
            depth=WebSearchDepth.STANDARD,
            confidence=0.8,
            reason="Test reason",
            matched_keywords=["latest"],
            matched_patterns=["search for"]
        )
        assert decision.should_search is True
        assert decision.depth == WebSearchDepth.STANDARD
        assert decision.confidence == 0.8
        assert "latest" in decision.matched_keywords

    def test_depth_enum(self):
        """Test WebSearchDepth enum values."""
        assert WebSearchDepth.QUICK.value == "quick"
        assert WebSearchDepth.STANDARD.value == "standard"
        assert WebSearchDepth.DEEP.value == "deep"


# ===== Strong Recency Keywords Tests =====

class TestStrongRecencyKeywords:
    @pytest.mark.parametrize("query,expected_search", [
        ("What's the latest news on AI?", True),
        ("What are the newest features in Python?", True),
        ("What is happening right now in the market?", True),
        ("Tell me about breaking news", True),
        ("What is current bitcoin price?", True),
        ("Show me today's weather", True),
        ("What's live on TV right now?", True),
    ])
    def test_strong_recency_triggers_search(self, query, expected_search):
        """Test strong recency keywords trigger search."""
        decision = should_search_heuristic(query)
        assert decision.should_search == expected_search, f"Query: {query}"
        assert decision.confidence >= 0.4, f"Query: {query}, conf: {decision.confidence}"


# ===== Moderate Recency Keywords Tests =====

class TestModerateRecencyKeywords:
    @pytest.mark.parametrize("query", [
        "What are recent developments in AI?",
        "Show me new JavaScript frameworks",
        "What's updated in the API?",
        "Tell me about modern web development",
    ])
    def test_moderate_recency_increases_confidence(self, query):
        """Test moderate recency keywords increase confidence."""
        decision = should_search_heuristic(query)
        assert decision.confidence >= 0.2
        assert len(decision.matched_keywords) > 0


# ===== Explicit Search Request Tests =====

class TestExplicitSearchRequests:
    @pytest.mark.parametrize("query,expected_search", [
        ("Search for Python tutorials", True),
        ("Look up the weather forecast", True),
        ("Google the nearest restaurant", True),
        ("Search the web for AI news", True),
        ("Find information about climate change", True),
    ])
    def test_explicit_search_triggers(self, query, expected_search):
        """Test explicit search phrases trigger search."""
        decision = should_search_heuristic(query)
        assert decision.should_search == expected_search, f"Query: {query}"
        assert len(decision.matched_patterns) > 0


# ===== Fast-Changing Topics Tests =====

class TestFastChangingTopics:
    @pytest.mark.parametrize("query", [
        "What's the current stock price of Apple?",
        "Show me bitcoin price",
        "What's the weather forecast?",
        "What's the score of the game?",
        "When is the iPhone release date?",
        "What's the election poll status?",
    ])
    def test_fast_changing_topics_trigger(self, query):
        """Test fast-changing topics increase confidence."""
        decision = should_search_heuristic(query)
        assert decision.confidence >= 0.3, f"Query: {query}, conf: {decision.confidence}"
        assert len(decision.matched_keywords) > 0


# ===== Static Topics Tests =====

class TestStaticTopics:
    @pytest.mark.parametrize("query,should_not_search", [
        ("What is the definition of photosynthesis?", True),
        ("Explain the theory of relativity", True),
        ("How to make pasta?", True),
        ("What's the history of Rome?", True),
        ("What is the formula for velocity?", True),
    ])
    def test_static_topics_reduce_confidence(self, query, should_not_search):
        """Test static topics reduce search confidence."""
        decision = should_search_heuristic(query)
        # Static topics should reduce confidence
        if should_not_search:
            assert decision.confidence < 0.5, f"Query: {query}, conf: {decision.confidence}"


# ===== Suppression Pattern Tests =====

class TestSuppressionPatterns:
    @pytest.mark.parametrize("query", [
        "How are you doing today?",
        "How do you feel about this?",
        "Tell me about yourself",
        "Do you remember when we talked about this?",
        "I'm feeling stressed today",
        "Can we talk about my feelings?",
    ])
    def test_suppression_patterns_block_search(self, query):
        """Test suppression patterns prevent search."""
        decision = should_search_heuristic(query)
        assert decision.should_search is False, f"Query should be suppressed: {query}"
        assert len(decision.matched_patterns) > 0


# ===== Year Pattern Tests =====

class TestYearPatterns:
    @pytest.mark.parametrize("query", [
        "What happened in 2025?",
        "Show me 2026 predictions",
        "Events in 2025 related to AI",
    ])
    def test_recent_year_increases_confidence(self, query):
        """Test current/recent year mention increases confidence."""
        decision = should_search_heuristic(query)
        assert decision.confidence >= 0.3, f"Query: {query}, conf: {decision.confidence}"


# ===== Empty and Edge Cases =====

class TestEdgeCases:
    def test_empty_query(self):
        """Test empty query returns no search."""
        decision = should_search_heuristic("")
        assert decision.should_search is False
        assert decision.confidence == 0.0
        assert decision.reason == "Empty query"

    def test_none_query(self):
        """Test None-like query handling."""
        decision = should_search_heuristic("")
        assert decision.should_search is False

    def test_very_short_query(self):
        """Test very short query."""
        decision = should_search_heuristic("Hi")
        # Should have low confidence without keywords
        assert decision.confidence < 0.5

    def test_mixed_signals(self):
        """Test query with both recency and static indicators."""
        decision = should_search_heuristic("What's the latest theory about quantum physics?")
        # Has "latest" (strong) but "theory" (static) - should still lean toward search
        assert decision.confidence >= 0.1


# ===== Search Depth Tests =====

class TestSearchDepth:
    def test_high_confidence_standard_depth(self):
        """Test high confidence results in STANDARD depth."""
        decision = should_search_heuristic("search for the latest breaking news on AI developments")
        if decision.confidence >= 0.8:
            assert decision.depth == WebSearchDepth.STANDARD

    def test_moderate_confidence_quick_depth(self):
        """Test moderate confidence results in QUICK depth."""
        decision = should_search_heuristic("recent news")
        if 0.5 <= decision.confidence < 0.8:
            assert decision.depth in [WebSearchDepth.QUICK, WebSearchDepth.STANDARD]


# ===== Integration Helper Tests =====

class TestIntegrationHelpers:
    def test_analyze_for_web_search_function(self):
        """Test convenience function."""
        decision = analyze_for_web_search("What's the latest AI news?")
        assert isinstance(decision, WebSearchDecision)
        assert decision.should_search is True

    def test_get_search_decision_disabled(self):
        """Test decision with web search disabled."""
        decision = get_search_decision_for_prompt(
            "What's the latest news?",
            web_search_enabled=False
        )
        assert decision.should_search is False
        assert "disabled" in decision.reason.lower()

    def test_get_search_decision_high_crisis(self):
        """Test decision with HIGH crisis level."""
        decision = get_search_decision_for_prompt(
            "What's the latest news?",
            crisis_level="HIGH",
            web_search_enabled=True
        )
        assert decision.should_search is False
        assert "crisis" in decision.reason.lower()

    def test_get_search_decision_medium_crisis(self):
        """Test decision with MEDIUM crisis level."""
        decision = get_search_decision_for_prompt(
            "What's the latest news?",
            crisis_level="MEDIUM",
            web_search_enabled=True
        )
        assert decision.should_search is False
        assert "crisis" in decision.reason.lower()

    def test_get_search_decision_conversational(self):
        """Test decision with CONVERSATIONAL crisis level."""
        decision = get_search_decision_for_prompt(
            "What's the latest news?",
            crisis_level="CONVERSATIONAL",
            web_search_enabled=True
        )
        # Should allow search during conversational mode
        assert decision.should_search is True


# ===== Confidence Threshold Tests =====

class TestConfidenceThreshold:
    def test_below_threshold_no_search(self):
        """Test queries below confidence threshold don't search."""
        # Generic query without recency or search indicators
        decision = should_search_heuristic("Tell me a joke")
        assert decision.confidence < 0.5
        # Note: should_search depends on SEARCH_CONFIDENCE_THRESHOLD (default 0.5)

    def test_above_threshold_search(self):
        """Test queries above confidence threshold do search."""
        decision = should_search_heuristic("What's the latest breaking news today?")
        assert decision.confidence >= 0.5
        assert decision.should_search is True


# ===== Keyword Set Integrity Tests =====

class TestKeywordSets:
    def test_no_overlap_strong_static(self):
        """Test no overlap between strong recency and static keywords."""
        overlap = RECENCY_KEYWORDS_STRONG & STATIC_TOPICS
        assert len(overlap) == 0, f"Unexpected overlap: {overlap}"

    def test_no_overlap_suppression_search(self):
        """Test suppression patterns don't accidentally match search phrases."""
        for supp in SUPPRESSION_PATTERNS:
            for search in EXPLICIT_SEARCH_PHRASES:
                # They can contain common words but shouldn't be identical
                assert supp != search

    def test_keyword_sets_not_empty(self):
        """Test all keyword sets have content."""
        assert len(RECENCY_KEYWORDS_STRONG) > 0
        assert len(RECENCY_KEYWORDS_MODERATE) > 0
        assert len(NEWS_KEYWORDS) > 0
        assert len(FAST_CHANGING_TOPICS) > 0
        assert len(STATIC_TOPICS) > 0


# ===== Real-World Query Tests =====

class TestRealWorldQueries:
    @pytest.mark.parametrize("query,expect_search", [
        # Should trigger search
        ("What's happening in the stock market today?", True),
        ("Latest iPhone announcement", True),
        ("Current weather in New York", True),
        ("Search for Python 3.12 new features", True),
        ("Breaking news about climate change", True),

        # Should NOT trigger search
        ("How do I feel about this?", False),
        ("Remember our conversation yesterday?", False),
        ("I'm feeling anxious", False),
        ("Tell me about yourself", False),
    ])
    def test_real_world_queries(self, query, expect_search):
        """Test real-world query classification."""
        decision = should_search_heuristic(query)
        assert decision.should_search == expect_search, \
            f"Query: '{query}', expected search={expect_search}, got {decision.should_search}, " \
            f"conf={decision.confidence}, reason={decision.reason}"


# ===== LLM-First Trigger Tests =====

# Import LLM-related components
from utils.web_search_trigger import (
    LLMSearchTriggerResponse,
    quick_prefilter_should_skip,
    analyze_for_web_search_llm,
)


class TestLLMSearchTriggerResponse:
    """Tests for LLM response parsing."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        json_str = '''{"should_search": true, "confidence": 0.85, "reason": "Current news query", "search_terms": ["flu variant 2026"], "search_depth": "standard", "num_searches": 1}'''
        result = LLMSearchTriggerResponse.parse(json_str)
        assert result is not None
        assert result.should_search is True
        assert result.confidence == 0.85
        assert result.reason == "Current news query"
        assert result.search_terms == ["flu variant 2026"]
        assert result.search_depth == "standard"
        assert result.num_searches == 1

    def test_parse_json_with_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        json_str = '''```json
{"should_search": true, "confidence": 0.9, "reason": "Test", "search_terms": [], "search_depth": "quick", "num_searches": 1}
```'''
        result = LLMSearchTriggerResponse.parse(json_str)
        assert result is not None
        assert result.should_search is True
        assert result.confidence == 0.9

    def test_parse_json_missing_fields(self):
        """Test parsing JSON with missing fields uses defaults."""
        json_str = '{"should_search": false}'
        result = LLMSearchTriggerResponse.parse(json_str)
        assert result is not None
        assert result.should_search is False
        assert result.confidence == 0.0
        assert result.search_terms == []
        assert result.search_depth == "quick"
        assert result.num_searches == 1

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns None."""
        result = LLMSearchTriggerResponse.parse("not valid json")
        assert result is None

    def test_parse_empty_string(self):
        """Test parsing empty string returns None."""
        result = LLMSearchTriggerResponse.parse("")
        assert result is None

    def test_parse_clamps_confidence(self):
        """Test confidence is clamped to 0.0-1.0."""
        # High confidence clamped
        json_str = '{"should_search": true, "confidence": 1.5}'
        result = LLMSearchTriggerResponse.parse(json_str)
        assert result.confidence == 1.0

        # Negative confidence clamped
        json_str = '{"should_search": false, "confidence": -0.5}'
        result = LLMSearchTriggerResponse.parse(json_str)
        assert result.confidence == 0.0

    def test_parse_clamps_num_searches(self):
        """Test num_searches is clamped to 1-4."""
        # High num_searches clamped
        json_str = '{"should_search": true, "num_searches": 10}'
        result = LLMSearchTriggerResponse.parse(json_str)
        assert result.num_searches == 4

        # Zero num_searches clamped
        json_str = '{"should_search": true, "num_searches": 0}'
        result = LLMSearchTriggerResponse.parse(json_str)
        assert result.num_searches == 1

    def test_parse_normalizes_depth(self):
        """Test invalid depth normalized to quick."""
        json_str = '{"should_search": true, "search_depth": "INVALID"}'
        result = LLMSearchTriggerResponse.parse(json_str)
        assert result.search_depth == "quick"


class TestQuickPrefilter:
    """Tests for quick pre-filter function."""

    def test_empty_query_skips(self):
        """Test empty query is skipped."""
        assert quick_prefilter_should_skip("") is True
        assert quick_prefilter_should_skip("   ") is True

    def test_short_query_skips(self):
        """Test very short query is skipped."""
        assert quick_prefilter_should_skip("hi") is True
        assert quick_prefilter_should_skip("ok") is True

    def test_greeting_skips(self):
        """Test short greetings are skipped."""
        assert quick_prefilter_should_skip("hello") is True
        assert quick_prefilter_should_skip("hey") is True
        assert quick_prefilter_should_skip("thanks") is True

    def test_suppression_pattern_skips(self):
        """Test suppression patterns are skipped."""
        assert quick_prefilter_should_skip("how are you doing today?") is True
        assert quick_prefilter_should_skip("I'm feeling stressed") is True

    def test_normal_query_not_skipped(self):
        """Test normal search-worthy queries are not skipped."""
        assert quick_prefilter_should_skip("What is the weather in New York?") is False
        assert quick_prefilter_should_skip("Tell me about Python programming") is False
        assert quick_prefilter_should_skip("What's the latest news on AI?") is False


class TestWebSearchDecisionNewFields:
    """Tests for new fields in WebSearchDecision."""

    def test_new_fields_defaults(self):
        """Test new fields have correct defaults."""
        decision = WebSearchDecision(
            should_search=True,
            depth=WebSearchDepth.QUICK,
            confidence=0.5,
            reason="Test",
            matched_keywords=[],
            matched_patterns=[]
        )
        # New fields should have defaults
        assert decision.search_terms == []
        assert decision.num_searches == 1
        assert decision.source == "heuristic"

    def test_new_fields_can_be_set(self):
        """Test new fields can be explicitly set."""
        decision = WebSearchDecision(
            should_search=True,
            depth=WebSearchDepth.STANDARD,
            confidence=0.8,
            reason="LLM decision",
            matched_keywords=["latest"],
            matched_patterns=[],
            search_terms=["optimized query 1", "optimized query 2"],
            num_searches=2,
            source="llm"
        )
        assert decision.search_terms == ["optimized query 1", "optimized query 2"]
        assert decision.num_searches == 2
        assert decision.source == "llm"


class TestAnalyzeForWebSearchLLM:
    """Tests for async LLM-first trigger function."""

    @pytest.mark.asyncio
    async def test_disabled_config_returns_no_search(self):
        """Test web search disabled returns no search."""
        decision = await analyze_for_web_search_llm(
            query="What's the latest news?",
            web_search_enabled=False
        )
        assert decision.should_search is False
        assert "disabled" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_crisis_suppression(self):
        """Test crisis levels suppress search."""
        for level in ["HIGH", "MEDIUM"]:
            decision = await analyze_for_web_search_llm(
                query="What's the latest news?",
                crisis_level=level
            )
            assert decision.should_search is False
            assert "crisis" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_prefilter_skips_llm(self):
        """Test pre-filter skips LLM for obvious non-search queries."""
        decision = await analyze_for_web_search_llm(
            query="hello",  # Too short
            model_manager=None
        )
        assert decision.should_search is False
        assert "pre-filter" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_no_model_manager_uses_heuristics(self):
        """Test without model manager falls back to heuristics."""
        decision = await analyze_for_web_search_llm(
            query="What's the latest Bitcoin price?",
            model_manager=None
        )
        # Should use heuristics (bitcoin + latest should trigger)
        assert decision.source == "heuristic"

    @pytest.mark.asyncio
    async def test_with_mock_model_manager(self):
        """Test with mocked model manager returns LLM decision."""
        from unittest.mock import AsyncMock

        mock_manager = MagicMock()
        mock_response = '''{"should_search": true, "confidence": 0.9, "reason": "Time-sensitive", "search_terms": ["test query"], "search_depth": "standard", "num_searches": 1}'''

        # Use AsyncMock for the coroutine
        mock_manager.generate_once = AsyncMock(return_value=mock_response)

        # Patch LLM_FIRST_ENABLED to True (must be at module level)
        import utils.web_search_trigger as trigger_module
        original_value = trigger_module.LLM_FIRST_ENABLED
        try:
            trigger_module.LLM_FIRST_ENABLED = True

            decision = await analyze_for_web_search_llm(
                query="What's the current flu variant spreading in 2026?",
                model_manager=mock_manager
            )

            # Should have LLM-influenced decision
            assert decision.source == "llm"
            assert decision.search_terms == ["test query"]
        finally:
            trigger_module.LLM_FIRST_ENABLED = original_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
