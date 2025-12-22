# tests/test_web_search_manager.py
"""
Unit tests for WebSearchManager module.

Tests cover:
- WebSearchResult data class functionality
- WebSearchRateLimiter credit tracking
- WebSearchCache operations
- WebSearchManager search functionality
- Crisis level suppression
- Error handling and graceful degradation
"""

import asyncio
import json
import os
import pytest
import tempfile
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

# Import modules under test
from knowledge.web_search_manager import (
    WebSearchDepth,
    WebPage,
    WebSearchResult,
    WebSearchSession,
    WebSearchRateLimiter,
    WebSearchCache,
    WebSearchManager,
    quick_web_search,
)


# ===== Fixtures =====

@pytest.fixture
def temp_state_file():
    """Create a temporary state file for rate limiter tests."""
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def mock_tavily_client():
    """Create a mock Tavily client."""
    client = MagicMock()
    client.search.return_value = {
        "results": [
            {
                "url": "https://example.com/1",
                "title": "Example Result 1",
                "content": "This is example content 1",
                "score": 0.95,
                "published_date": "2024-01-15"
            },
            {
                "url": "https://example.com/2",
                "title": "Example Result 2",
                "content": "This is example content 2",
                "score": 0.85,
            }
        ]
    }
    client.extract.return_value = {
        "results": [
            {
                "url": "https://example.com/1",
                "title": "Example Page 1",
                "raw_content": "Full extracted content from page 1"
            }
        ]
    }
    return client


# ===== WebPage Tests =====

class TestWebPage:
    def test_create_web_page(self):
        """Test WebPage creation with all fields."""
        page = WebPage(
            url="https://example.com",
            title="Example",
            content="Content here",
            snippet="Snippet...",
            score=0.9,
            published_date="2024-01-15",
            source="tavily_search"
        )
        assert page.url == "https://example.com"
        assert page.title == "Example"
        assert page.content == "Content here"
        assert page.score == 0.9

    def test_web_page_defaults(self):
        """Test WebPage with only required fields."""
        page = WebPage(url="https://test.com", title="Test", content="Test content")
        assert page.snippet == ""
        assert page.score == 0.0
        assert page.source == "tavily"


# ===== WebSearchResult Tests =====

class TestWebSearchResult:
    def test_has_results_true(self):
        """Test has_results property with results."""
        result = WebSearchResult(
            query="test",
            pages=[WebPage(url="https://test.com", title="Test", content="Content")]
        )
        assert result.has_results is True

    def test_has_results_false_empty(self):
        """Test has_results property with no pages."""
        result = WebSearchResult(query="test", pages=[])
        assert result.has_results is False

    def test_has_results_false_error(self):
        """Test has_results property with error."""
        result = WebSearchResult(
            query="test",
            pages=[WebPage(url="https://test.com", title="Test", content="Content")],
            error="Search failed"
        )
        assert result.has_results is False

    def test_get_formatted_content(self):
        """Test content formatting."""
        pages = [
            WebPage(url="https://a.com", title="Page A", content="Content A"),
            WebPage(url="https://b.com", title="Page B", content="Content B"),
        ]
        result = WebSearchResult(query="test", pages=pages)
        formatted = result.get_formatted_content(max_chars=1000)
        assert "**Page A**" in formatted
        assert "https://a.com" in formatted
        assert "Content A" in formatted

    def test_get_formatted_content_truncation(self):
        """Test content truncation."""
        pages = [
            WebPage(url="https://a.com", title="Page A", content="X" * 500),
            WebPage(url="https://b.com", title="Page B", content="Y" * 500),
        ]
        result = WebSearchResult(query="test", pages=pages)
        formatted = result.get_formatted_content(max_chars=200)
        assert len(formatted) <= 210  # Allow some margin for truncation marker


# ===== WebSearchSession Tests =====

class TestWebSearchSession:
    def test_all_pages_deduplication(self):
        """Test that all_pages deduplicates results."""
        page1 = WebPage(url="https://a.com", title="A", content="A")
        page2 = WebPage(url="https://b.com", title="B", content="B")
        page1_dup = WebPage(url="https://a.com", title="A2", content="A2")

        session = WebSearchSession(
            initial_query="test",
            depth=WebSearchDepth.STANDARD,
            search_results=[page1, page2],
            extracted_pages=[page1_dup]  # Duplicate URL
        )

        all_pages = session.all_pages
        assert len(all_pages) == 2
        urls = [p.url for p in all_pages]
        assert "https://a.com" in urls
        assert "https://b.com" in urls


# ===== WebSearchRateLimiter Tests =====

class TestWebSearchRateLimiter:
    def test_initial_state(self, temp_state_file):
        """Test initial rate limiter state."""
        limiter = WebSearchRateLimiter(
            daily_limit=100,
            per_query_limit=5,
            state_file=temp_state_file
        )
        assert limiter.can_search(1.0) is True
        assert limiter.get_remaining_credits() == 100.0

    def test_record_usage(self, temp_state_file):
        """Test recording credit usage."""
        limiter = WebSearchRateLimiter(
            daily_limit=100,
            state_file=temp_state_file
        )
        limiter.record_usage(10.0)
        assert limiter.get_remaining_credits() == 90.0

    def test_can_search_within_limit(self, temp_state_file):
        """Test can_search within daily limit."""
        limiter = WebSearchRateLimiter(
            daily_limit=100,
            state_file=temp_state_file
        )
        limiter.record_usage(95.0)
        assert limiter.can_search(5.0) is True
        assert limiter.can_search(6.0) is False

    def test_estimate_credits(self, temp_state_file):
        """Test credit estimation for different depths."""
        limiter = WebSearchRateLimiter(state_file=temp_state_file)

        assert limiter.estimate_credits(WebSearchDepth.QUICK) == 1.0
        assert limiter.estimate_credits(WebSearchDepth.STANDARD) == 2.0
        assert limiter.estimate_credits(WebSearchDepth.DEEP) == 3.0
        assert limiter.estimate_credits(WebSearchDepth.STANDARD, num_extracts=2) == 4.0

    def test_persistence(self, temp_state_file):
        """Test that state persists across instances."""
        limiter1 = WebSearchRateLimiter(
            daily_limit=100,
            state_file=temp_state_file
        )
        limiter1.record_usage(30.0)

        # Create new instance with same state file
        limiter2 = WebSearchRateLimiter(
            daily_limit=100,
            state_file=temp_state_file
        )
        assert limiter2.get_remaining_credits() == 70.0


# ===== WebSearchCache Tests =====

class TestWebSearchCache:
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = WebSearchCache()
        cache._initialized = True
        cache._collection = None
        result = cache.get("test query", WebSearchDepth.QUICK)
        assert result is None

    def test_generate_cache_key(self):
        """Test cache key generation is deterministic."""
        cache = WebSearchCache()
        key1 = cache._generate_cache_key("test query", WebSearchDepth.QUICK)
        key2 = cache._generate_cache_key("test query", WebSearchDepth.QUICK)
        key3 = cache._generate_cache_key("Test Query", WebSearchDepth.QUICK)  # Case insensitive
        key4 = cache._generate_cache_key("test query", WebSearchDepth.STANDARD)

        assert key1 == key2
        assert key1 == key3  # Same after normalization
        assert key1 != key4  # Different depth


# ===== WebSearchManager Tests =====

class TestWebSearchManager:
    def test_initialization_no_api_key(self):
        """Test manager initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            manager = WebSearchManager(api_key="")
            assert manager.is_available() is False

    def test_initialization_with_api_key(self, mock_tavily_client):
        """Test manager initialization with API key."""
        with patch('tavily.TavilyClient', return_value=mock_tavily_client):
            manager = WebSearchManager(api_key="test_key")
            assert manager.api_key == "test_key"

    def test_get_status_no_key(self):
        """Test status without API key."""
        with patch.dict(os.environ, {}, clear=True):
            manager = WebSearchManager(api_key="")
            status = manager.get_status()
            assert status["api_key_configured"] is False
            assert status["available"] is False

    @pytest.mark.asyncio
    async def test_search_crisis_suppression_high(self, mock_tavily_client):
        """Test search is suppressed during HIGH crisis level."""
        with patch('tavily.TavilyClient', return_value=mock_tavily_client):
            manager = WebSearchManager(api_key="test_key")
            result = await manager.search("test query", crisis_level="HIGH")
            assert result.has_results is False
            assert "suppressed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_search_crisis_suppression_medium(self, mock_tavily_client):
        """Test search is suppressed during MEDIUM crisis level."""
        with patch('tavily.TavilyClient', return_value=mock_tavily_client):
            manager = WebSearchManager(api_key="test_key")
            result = await manager.search("test query", crisis_level="MEDIUM")
            assert result.has_results is False
            assert "suppressed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_search_allowed_conversational(self, mock_tavily_client):
        """Test search is allowed during CONVERSATIONAL crisis level."""
        with patch('tavily.TavilyClient', return_value=mock_tavily_client):
            manager = WebSearchManager(api_key="test_key")
            manager._tavily_client = mock_tavily_client
            manager._initialized = True

            result = await manager.search("test query", crisis_level="CONVERSATIONAL")
            # Should attempt search (may return results or empty depending on mock)
            assert result.error is None or "suppressed" not in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_search_rate_limit_exceeded(self, mock_tavily_client, temp_state_file):
        """Test search fails when rate limit exceeded."""
        rate_limiter = WebSearchRateLimiter(
            daily_limit=0,  # No credits available
            state_file=temp_state_file
        )

        with patch('tavily.TavilyClient', return_value=mock_tavily_client):
            manager = WebSearchManager(api_key="test_key", rate_limiter=rate_limiter)
            # Use unique query to avoid cache hits
            result = await manager.search("unique rate limit test query 12345")
            assert result.has_results is False
            assert "limit" in result.error.lower()

    @pytest.mark.asyncio
    async def test_search_timeout(self, mock_tavily_client):
        """Test search handles timeout gracefully."""
        with patch('tavily.TavilyClient', return_value=mock_tavily_client):
            manager = WebSearchManager(api_key="test_key", default_timeout=0.001)
            manager._tavily_client = mock_tavily_client
            manager._initialized = True

            # Make search take too long
            async def slow_search(*args, **kwargs):
                await asyncio.sleep(1)
                return {"results": []}

            with patch.object(manager, '_execute_search', slow_search):
                # Use unique query to avoid cache hits
                result = await manager.search("unique timeout test query 67890", timeout=0.001)
                assert result.has_results is False
                assert "timed out" in result.error.lower()


# ===== Integration Tests =====

class TestWebSearchIntegration:
    @pytest.mark.asyncio
    async def test_quick_web_search_convenience_function(self):
        """Test the convenience function."""
        with patch.dict(os.environ, {}, clear=True):
            result = await quick_web_search("test query")
            # Without API key, should return error result
            assert result.has_results is False

    def test_depth_enum_values(self):
        """Test WebSearchDepth enum values."""
        assert WebSearchDepth.QUICK.value == "quick"
        assert WebSearchDepth.STANDARD.value == "standard"
        assert WebSearchDepth.DEEP.value == "deep"


# ===== Edge Case Tests =====

class TestEdgeCases:
    def test_empty_query_handling(self):
        """Test handling of empty query."""
        result = WebSearchResult(query="", pages=[])
        assert result.has_results is False

    def test_web_page_empty_content(self):
        """Test WebPage with empty content."""
        page = WebPage(url="https://test.com", title="Test", content="")
        assert page.content == ""

    def test_formatted_content_empty_pages(self):
        """Test formatted content with no pages."""
        result = WebSearchResult(query="test", pages=[])
        formatted = result.get_formatted_content()
        assert formatted == ""

    def test_rate_limiter_negative_remaining(self, temp_state_file):
        """Test rate limiter doesn't go negative."""
        limiter = WebSearchRateLimiter(
            daily_limit=10,
            state_file=temp_state_file
        )
        limiter.record_usage(15.0)  # Over the limit
        assert limiter.get_remaining_credits() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
