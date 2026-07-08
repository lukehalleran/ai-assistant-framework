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
        # "newest" (strong recency, +0.4) + "version" (fast-changing, +0.3).
        # Was previously "newest features in Python?", which only crossed the
        # threshold because "new" matched as a substring inside "newest"
        # (double-counting bug); word-boundary matching removed that, so this
        # now needs a genuine second signal like its sibling cases.
        ("What's the newest version of Python?", True),
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


# ===== Substring / Word-Boundary Regression =====

class TestWordBoundaryMatching:
    """Keyword sets must match whole words, not substrings.

    Regression for a casual life-update ("...now walking to get some ice cream
    or something") that scored 0.70 and triggered a web search because "eth"
    (Ethereum ticker) matched inside "som-eth-ing" (+0.3 fast-changing) on top
    of "today" (+0.4 strong recency).
    """

    def test_ticker_does_not_match_inside_word(self):
        # "eth" must not match "something"; "btc"/"eth" are only whole-word hits.
        decision = should_search_heuristic(
            "Went to my dad's to swim, had a beer, now walking to get ice cream or something"
        )
        assert decision.should_search is False, f"conf={decision.confidence}"
        assert "eth" not in decision.matched_keywords

    def test_today_alone_below_threshold(self):
        # A bare temporal word is not enough to trigger a search on its own.
        decision = should_search_heuristic("Yeah I did not shit today.")
        assert decision.should_search is False
        assert decision.confidence < 0.5

    def test_new_does_not_match_inside_newest(self):
        # "new" (moderate) must not double-count inside "newest" (strong).
        decision = should_search_heuristic("What are the newest features in Python?")
        assert "new" not in decision.matched_keywords

    def test_newest_features_triggers_via_topic_corroboration(self):
        # "newest" (strong, +0.4) + "features" (fast-changing topic, +0.3) —
        # crosses the threshold on genuine signals, not the old "new"-inside-
        # "newest" substring accident.
        decision = should_search_heuristic("What are the newest features in Python?")
        assert decision.should_search is True

    def test_features_alone_does_not_trigger(self):
        # Topic word without any recency signal stays below threshold.
        decision = should_search_heuristic("Explain the features of dataclasses")
        assert decision.should_search is False

    def test_whole_word_keywords_still_match(self):
        # Word-boundary matching must not break legitimate whole-word hits.
        decision = should_search_heuristic("What's the current bitcoin price today?")
        assert decision.should_search is True
        assert "bitcoin" in decision.matched_keywords


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
        """Test that _classify_with_llm_unified correctly parses LLM JSON."""
        from unittest.mock import AsyncMock

        mock_manager = MagicMock()
        mock_response = '''{"should_search": true, "confidence": 0.9, "reason": "Time-sensitive", "search_terms": ["test query"], "search_depth": "standard", "num_searches": 1}'''

        mock_manager.generate_once = AsyncMock(return_value=mock_response)

        # Test the internal LLM classification directly to avoid
        # heuristic short-circuits and veto logic in the outer function.
        from utils.web_search_trigger import _classify_with_llm_unified

        result = await _classify_with_llm_unified(
            query="test query for LLM classification",
            model_manager=mock_manager,
        )

        assert result is not None
        assert result.should_search is True
        assert result.confidence == 0.9
        assert result.search_terms == ["test query"]
        assert result.search_depth == "standard"
        assert mock_manager.generate_once.called


class TestConversationContextResolution:
    """Elliptical follow-ups ("check the news") must resolve against the prior
    topic instead of producing generic terms. Covers the conversation_context
    threading added to the trigger + the gate's recent-context digest."""

    def test_prompt_includes_context_block_and_followup_guideline(self):
        """When context is provided, the prompt carries it + the follow-up rule."""
        from utils.web_search_trigger import _build_llm_trigger_prompt

        ctx = "User: They are requiring IDs and biometrics to use Claude\nAssistant: That's a real privacy escalation."
        prompt = _build_llm_trigger_prompt("check the news", "2026-06-21", conversation_context=ctx)

        # Distinctive data-block header (the guideline text also says "RECENT
        # CONVERSATION", so match the header phrase, not the bare token).
        assert "turns immediately before this query" in prompt
        assert "biometrics to use Claude" in prompt
        assert "FOLLOW-UPS" in prompt  # the deictic-resolution guideline

    def test_prompt_omits_context_block_when_none(self):
        """No context → no RECENT CONVERSATION block (unchanged legacy behavior)."""
        from utils.web_search_trigger import _build_llm_trigger_prompt

        prompt = _build_llm_trigger_prompt("check the news", "2026-06-21")
        assert "turns immediately before this query" not in prompt
        # Guideline text is always present; only the data block is conditional.
        assert "FOLLOW-UPS" in prompt

    @pytest.mark.asyncio
    async def test_classify_threads_context_into_prompt(self):
        """_classify_with_llm_unified forwards conversation_context to the prompt."""
        from unittest.mock import AsyncMock
        from utils.web_search_trigger import _classify_with_llm_unified

        mock_manager = MagicMock()
        mock_manager.generate_once = AsyncMock(return_value=(
            '{"should_search": true, "confidence": 0.9, "reason": "follow-up", '
            '"search_terms": ["Claude biometric ID requirement 2026"], '
            '"search_depth": "standard", "num_searches": 1}'
        ))

        ctx = "User: They are requiring IDs and biometrics to use Claude"
        result = await _classify_with_llm_unified(
            query="check the news just the other day",
            model_manager=mock_manager,
            conversation_context=ctx,
        )

        assert result is not None
        # The prior topic reached the LLM prompt (first positional arg).
        sent_prompt = mock_manager.generate_once.call_args.args[0]
        assert "biometrics to use Claude" in sent_prompt
        assert "turns immediately before this query" in sent_prompt

    @pytest.mark.asyncio
    async def test_cache_key_separates_by_context(self):
        """Same query under different prior topics must NOT collide in the cache,
        but the identical (query, context) pair must be served from cache."""
        from unittest.mock import AsyncMock, patch
        import utils.web_search_trigger as wst

        wst._llm_trigger_cache.clear()

        # Non-decisive heuristic so the LLM path is actually exercised
        # (avoids the conf<=0 and conf>=0.7 short-circuits).
        neutral = WebSearchDecision(
            should_search=True, depth=WebSearchDepth.STANDARD, confidence=0.5,
            reason="neutral", matched_keywords=["news"], matched_patterns=[],
        )

        mock_manager = MagicMock()
        mock_manager.generate_once = AsyncMock(side_effect=[
            '{"should_search": true, "confidence": 0.9, "reason": "a", "search_terms": ["topic A terms"], "search_depth": "standard", "num_searches": 1}',
            '{"should_search": true, "confidence": 0.9, "reason": "b", "search_terms": ["topic B terms"], "search_depth": "standard", "num_searches": 1}',
        ])

        query = "what's the latest on that"
        with patch.object(wst, "should_search_heuristic", return_value=neutral), \
             patch.object(wst, "LLM_FIRST_ENABLED", True):
            res_a = await wst.analyze_for_web_search_llm(
                query=query, model_manager=mock_manager,
                conversation_context="topic A context",
            )
            res_b = await wst.analyze_for_web_search_llm(
                query=query, model_manager=mock_manager,
                conversation_context="topic B context",
            )
            # Same query, same context as A → cache hit, no new LLM call.
            res_a2 = await wst.analyze_for_web_search_llm(
                query=query, model_manager=mock_manager,
                conversation_context="topic A context",
            )

        # Different context → different terms (no collision on the bare query).
        assert res_a.search_terms == ["topic A terms"]
        assert res_b.search_terms == ["topic B terms"]
        # Only two LLM calls: the third (A repeated) was served from cache.
        assert mock_manager.generate_once.call_count == 2
        assert res_a2.search_terms == res_a.search_terms

    def test_gate_build_recent_context_digest(self):
        """Gate helper renders prior turns oldest→newest; tolerates empty/None."""
        from core.agentic.gate import _build_recent_context

        class FakeCorpus:
            def get_recent_memories(self, count=2):
                # newest-first, as the real CorpusManager returns
                return [
                    {"query": "check the news", "response": "..."},
                    {"query": "they want biometrics for Claude", "response": "privacy concern"},
                ]

        ctx = _build_recent_context(FakeCorpus())
        assert ctx is not None
        # reversed to chronological: the older biometrics turn precedes the newest
        assert ctx.index("biometrics for Claude") < ctx.index("check the news")
        assert "User: they want biometrics for Claude" in ctx

        assert _build_recent_context(None) is None

        class EmptyCorpus:
            def get_recent_memories(self, count=2):
                return []

        assert _build_recent_context(EmptyCorpus()) is None


class TestReferentialFollowupBypass:
    """A conf=0.0/no-keyword query is normally short-circuited before the LLM. A
    referential follow-up ("they're only letting us use it for 7 days") must
    instead reach the LLM WHEN conversation context is available to resolve it —
    the heuristic can't see the prior topic ("it" = the model just discussed), the
    LLM (with context) can. Regression for enhanced-mode never searching a
    pronoun-driven follow-up on a just-searched topic."""

    def test_query_depends_on_context_detects_referential(self):
        from utils.web_search_trigger import query_depends_on_context
        assert query_depends_on_context("They're only letting us use it for 7 days") is True
        assert query_depends_on_context("is that still happening") is True
        assert query_depends_on_context("did they push it back") is True
        assert query_depends_on_context("those got cancelled") is True
        # Token adjacent to punctuation or at end-of-query (the shapes the old
        # padded-substring tuple missed).
        assert query_depends_on_context("what about that?") is True
        assert query_depends_on_context("did you see them?") is True
        assert query_depends_on_context("tell me more about it: the pricing part") is True
        assert query_depends_on_context("did you check it") is True
        # Standalone queries carrying their own subject noun are not referential.
        assert query_depends_on_context("what is the capital of France") is False
        assert query_depends_on_context("bitcoin price today") is False
        assert query_depends_on_context("") is False
        assert query_depends_on_context(None) is False

    def _zero_heuristic(self):
        return WebSearchDecision(
            should_search=False, depth=WebSearchDepth.QUICK, confidence=0.0,
            reason="No strong indicators", matched_keywords=[], matched_patterns=[],
        )

    @pytest.mark.asyncio
    async def test_referential_followup_with_context_reaches_llm(self):
        from unittest.mock import AsyncMock, patch
        import utils.web_search_trigger as wst
        wst._llm_trigger_cache.clear()

        mock_manager = MagicMock()
        mock_manager.generate_once = AsyncMock(return_value=(
            '{"should_search": true, "confidence": 0.85, "reason": "time-limited access claim", '
            '"search_terms": ["Fable 5 limited 7 day access"], "search_depth": "standard", "num_searches": 1}'
        ))

        with patch.object(wst, "should_search_heuristic", return_value=self._zero_heuristic()), \
             patch.object(wst, "LLM_FIRST_ENABLED", True):
            decision = await wst.analyze_for_web_search_llm(
                query="They're only letting us use it for 7 days",
                model_manager=mock_manager,
                conversation_context=(
                    "User: fable is supposed to be available today\n"
                    "Assistant: the freeze may have lifted"
                ),
            )

        # LLM was consulted (not short-circuited) and its verdict carried through.
        assert mock_manager.generate_once.called
        assert decision.should_search is True
        assert decision.source == "llm"

    @pytest.mark.asyncio
    async def test_conf_zero_without_context_still_short_circuits(self):
        from unittest.mock import AsyncMock, patch
        import utils.web_search_trigger as wst
        wst._llm_trigger_cache.clear()

        mock_manager = MagicMock()
        mock_manager.generate_once = AsyncMock(return_value='{"should_search": true, "confidence": 0.9}')

        with patch.object(wst, "should_search_heuristic", return_value=self._zero_heuristic()), \
             patch.object(wst, "LLM_FIRST_ENABLED", True):
            decision = await wst.analyze_for_web_search_llm(
                query="They're only letting us use it for 7 days",
                model_manager=mock_manager,
                conversation_context=None,  # nothing to resolve "it"/"us" against → skip
            )

        assert not mock_manager.generate_once.called  # short-circuited, no LLM call
        assert decision.should_search is False

    @pytest.mark.asyncio
    async def test_conf_zero_nonreferential_with_context_still_short_circuits(self):
        """Context present but the query is standalone (no pronoun referent) → still
        skip the LLM. We only bypass for referential follow-ups, to bound cost."""
        from unittest.mock import AsyncMock, patch
        import utils.web_search_trigger as wst
        wst._llm_trigger_cache.clear()

        mock_manager = MagicMock()
        mock_manager.generate_once = AsyncMock(return_value='{"should_search": true, "confidence": 0.9}')

        with patch.object(wst, "should_search_heuristic", return_value=self._zero_heuristic()), \
             patch.object(wst, "LLM_FIRST_ENABLED", True):
            decision = await wst.analyze_for_web_search_llm(
                query="my grandmother baked fresh bread",
                model_manager=mock_manager,
                conversation_context="User: tell me about your day",
            )

        assert not mock_manager.generate_once.called
        assert decision.should_search is False


class TestUserLocationInjection:
    """Location-dependent queries must carry the user's location in
    search_terms instead of leaking literal "my area"/"near me" to the search
    engine (which returns arbitrary big-market results — the DC-weather bug)."""

    def test_prompt_includes_location_line_and_guideline(self):
        from utils.web_search_trigger import _build_llm_trigger_prompt

        prompt = _build_llm_trigger_prompt(
            "how hot is it in my area", "2026-07-02",
            user_location="Saint Charles, IL",
        )
        assert "User location: Saint Charles, IL" in prompt
        assert "LOCAL QUERIES" in prompt
        assert 'NEVER emit "my area"' in prompt

    def test_prompt_omits_location_when_unknown(self):
        from utils.web_search_trigger import _build_llm_trigger_prompt

        prompt = _build_llm_trigger_prompt("how hot is it in my area", "2026-07-02")
        assert "User location:" not in prompt
        assert "LOCAL QUERIES" not in prompt

    @pytest.mark.asyncio
    async def test_classify_threads_location_into_prompt(self):
        """_classify_with_llm_unified resolves and forwards the location."""
        from unittest.mock import AsyncMock
        import utils.web_search_trigger as wst

        mock_manager = MagicMock()
        mock_manager.generate_once = AsyncMock(return_value=(
            '{"should_search": true, "confidence": 0.9, "reason": "weather", '
            '"search_terms": ["Saint Charles IL weather"], "search_depth": "quick", '
            '"num_searches": 1}'
        ))

        with patch("utils.location_resolver.get_user_location",
                   return_value="Saint Charles, IL"):
            await wst._classify_with_llm_unified(
                "how hot is it in my area", mock_manager
            )

        prompt = mock_manager.generate_once.call_args[0][0]
        assert "User location: Saint Charles, IL" in prompt

    @pytest.mark.asyncio
    async def test_classify_survives_location_failure(self):
        """A broken resolver must not break trigger classification."""
        from unittest.mock import AsyncMock
        import utils.web_search_trigger as wst

        mock_manager = MagicMock()
        mock_manager.generate_once = AsyncMock(return_value=(
            '{"should_search": false, "confidence": 0.2, "reason": "chat", '
            '"search_terms": [], "search_depth": "quick", "num_searches": 0}'
        ))

        with patch("utils.location_resolver.get_user_location",
                   side_effect=RuntimeError("network down")):
            parsed = await wst._classify_with_llm_unified("hey there", mock_manager)

        assert parsed is not None
        prompt = mock_manager.generate_once.call_args[0][0]
        assert "User location:" not in prompt


class TestUnjustifiedLocationStrip:
    """The trigger LLM sometimes localizes queries the prompt forbids
    localizing (institution/account queries — the 2026-07-08 wrong-college
    incident). The parse path must strip location the query never justified."""

    def test_prompt_forbids_institution_localization(self):
        from utils.web_search_trigger import _build_llm_trigger_prompt

        prompt = _build_llm_trigger_prompt(
            "my college login keeps failing", "2026-07-08",
            user_location="Saint Charles, IL",
        )
        assert "LOCATION IS ONLY FOR PHYSICAL SURROUNDINGS" in prompt
        assert "not their college unless they named it" in prompt

    @pytest.mark.asyncio
    async def test_parsed_terms_lose_unjustified_location(self):
        from unittest.mock import AsyncMock
        import utils.web_search_trigger as wst

        mock_manager = MagicMock()
        mock_manager.generate_once = AsyncMock(return_value=(
            '{"should_search": true, "confidence": 0.8, "reason": "login issue", '
            '"search_terms": ["login attempt failed account archived 2026", '
            '"how to resolve login issues account archived Saint Charles IL"], '
            '"search_depth": "quick", "num_searches": 2}'
        ))

        with patch("utils.location_resolver.get_user_location",
                   return_value="Saint Charles, IL"):
            parsed = await wst._classify_with_llm_unified(
                "my school account says archived and login failed", mock_manager
            )

        assert parsed is not None
        assert parsed.search_terms == [
            "login attempt failed account archived 2026",
            "how to resolve login issues account archived",
        ]

    @pytest.mark.asyncio
    async def test_justified_location_terms_survive(self):
        from unittest.mock import AsyncMock
        import utils.web_search_trigger as wst

        mock_manager = MagicMock()
        mock_manager.generate_once = AsyncMock(return_value=(
            '{"should_search": true, "confidence": 0.9, "reason": "weather", '
            '"search_terms": ["weather forecast Saint Charles IL today"], '
            '"search_depth": "quick", "num_searches": 1}'
        ))

        with patch("utils.location_resolver.get_user_location",
                   return_value="Saint Charles, IL"):
            parsed = await wst._classify_with_llm_unified(
                "how hot is it in my area today", mock_manager
            )

        assert parsed is not None
        assert parsed.search_terms == ["weather forecast Saint Charles IL today"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
