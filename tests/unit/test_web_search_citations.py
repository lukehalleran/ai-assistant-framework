"""Tests for web search citation system: WEB_N IDs, broad news detection, citation extraction.

Covers:
- assign_web_ids() deduplication and stable ID assignment
- _is_broad_news_query() positive and negative indicators
- dedupe_search_terms() Jaccard overlap collapsing
- Citation pattern matching for [WEB_N]
- Invalid citation detection and stripping
- format_web_sources_with_ids() output format
"""
import pytest
import re
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from knowledge.web_search_manager import (
    WebPage,
    WebSearchResult,
    NumberedWebSource,
    assign_web_ids,
    format_web_sources_with_ids,
    WebSearchManager,
)


# ---------------------------------------------------------------------------
# assign_web_ids — stable ID assignment after merge/dedupe
# ---------------------------------------------------------------------------

class TestAssignWebIds:

    def test_basic_assignment(self):
        pages = [
            WebPage(url="https://reuters.com/article1", title="Reuters Article", content="Content 1", score=0.9),
            WebPage(url="https://bbc.com/news/2", title="BBC News", content="Content 2", score=0.8),
        ]
        numbered, source_map = assign_web_ids(pages)
        assert len(numbered) == 2
        assert numbered[0].source_id == "WEB_1"
        assert numbered[1].source_id == "WEB_2"
        assert "WEB_1" in source_map
        assert source_map["WEB_1"]["title"] == "Reuters Article"

    def test_dedup_by_url(self):
        """Same URL appearing twice should deduplicate."""
        pages = [
            WebPage(url="https://reuters.com/article1", title="Reuters", content="Content A", score=0.9),
            WebPage(url="https://reuters.com/article1", title="Reuters", content="Content B", score=0.7),
        ]
        numbered, source_map = assign_web_ids(pages)
        assert len(numbered) == 1
        assert numbered[0].score == 0.9  # keep higher score

    def test_dedup_strips_trailing_slash(self):
        pages = [
            WebPage(url="https://reuters.com/article1/", title="A", content="C1", score=0.9),
            WebPage(url="https://reuters.com/article1", title="A", content="C2", score=0.8),
        ]
        numbered, source_map = assign_web_ids(pages)
        assert len(numbered) == 1

    def test_dedup_strips_query_params(self):
        pages = [
            WebPage(url="https://reuters.com/article1?ref=homepage", title="A", content="C1", score=0.9),
            WebPage(url="https://reuters.com/article1", title="A", content="C2", score=0.8),
        ]
        numbered, source_map = assign_web_ids(pages)
        assert len(numbered) == 1

    def test_ranked_by_score(self):
        pages = [
            WebPage(url="https://low.com", title="Low", content="Low", score=0.3),
            WebPage(url="https://high.com", title="High", content="High", score=0.9),
        ]
        numbered, source_map = assign_web_ids(pages)
        assert numbered[0].source_id == "WEB_1"
        assert numbered[0].title == "High"  # highest score first

    def test_empty_pages(self):
        numbered, source_map = assign_web_ids([])
        assert numbered == []
        assert source_map == {}

    def test_domain_extraction(self):
        pages = [WebPage(url="https://www.reuters.com/article", title="R", content="C", score=0.5)]
        numbered, source_map = assign_web_ids(pages)
        assert numbered[0].domain == "reuters.com"

    def test_no_id_collision_across_searches(self):
        """Multiple searches' pages merged then assigned — IDs must be unique."""
        search_a = [
            WebPage(url="https://a.com/1", title="A1", content="C", score=0.9),
            WebPage(url="https://a.com/2", title="A2", content="C", score=0.8),
        ]
        search_b = [
            WebPage(url="https://b.com/1", title="B1", content="C", score=0.7),
            WebPage(url="https://b.com/2", title="B2", content="C", score=0.6),
        ]
        merged = search_a + search_b
        numbered, source_map = assign_web_ids(merged)
        ids = [n.source_id for n in numbered]
        assert len(ids) == len(set(ids))  # all unique
        assert ids == ["WEB_1", "WEB_2", "WEB_3", "WEB_4"]


# ---------------------------------------------------------------------------
# _is_broad_news_query — positive and negative indicators
# ---------------------------------------------------------------------------

class TestIsBroadNewsQuery:

    def test_broad_news_positive(self):
        assert WebSearchManager._is_broad_news_query("What's going on in the news?")
        assert WebSearchManager._is_broad_news_query("Catch me up on current events")
        assert WebSearchManager._is_broad_news_query("What did I miss?")
        assert WebSearchManager._is_broad_news_query("latest headlines?")
        assert WebSearchManager._is_broad_news_query("I haven't been up on the news. What's going on there?")

    def test_specific_topic_not_broad(self):
        """Named entities or specific topics should NOT trigger broad decomposition."""
        assert not WebSearchManager._is_broad_news_query("Latest news on Artemis II")
        assert not WebSearchManager._is_broad_news_query("What's happening with Iran?")
        assert not WebSearchManager._is_broad_news_query("Any OpenAI news today?")
        assert not WebSearchManager._is_broad_news_query("News about the Supreme Court")

    def test_specific_domain_not_broad(self):
        assert not WebSearchManager._is_broad_news_query("Latest on the election")
        assert not WebSearchManager._is_broad_news_query("Update on the war")
        assert not WebSearchManager._is_broad_news_query("News about Nvidia earnings")

    def test_casual_not_broad(self):
        assert not WebSearchManager._is_broad_news_query("Hey how are you")
        assert not WebSearchManager._is_broad_news_query("Thanks a lot")


# ---------------------------------------------------------------------------
# dedupe_search_terms — Jaccard overlap
# ---------------------------------------------------------------------------

class TestDedupeSearchTerms:

    def test_near_duplicates_collapsed(self):
        terms = [
            "current news April 2026",
            "latest headlines April 2026",
            "news updates April 2026",
        ]
        result = WebSearchManager.dedupe_search_terms(terms)
        assert len(result) < len(terms)

    def test_distinct_terms_preserved(self):
        terms = [
            "Iran Supreme Leader succession Reuters",
            "Nvidia Samsung chip earnings CNBC",
            "Artemis II NASA moon mission",
        ]
        result = WebSearchManager.dedupe_search_terms(terms)
        assert len(result) == 3

    def test_single_term(self):
        assert WebSearchManager.dedupe_search_terms(["hello"]) == ["hello"]

    def test_empty(self):
        assert WebSearchManager.dedupe_search_terms([]) == []


# ---------------------------------------------------------------------------
# format_web_sources_with_ids
# ---------------------------------------------------------------------------

class TestFormatWebSourcesWithIds:

    def test_basic_format(self):
        sources = [
            NumberedWebSource("WEB_1", "Reuters Article", "https://reuters.com/a", "reuters.com", "Content here", 0.9),
            NumberedWebSource("WEB_2", "BBC News", "https://bbc.com/b", "bbc.com", "More content", 0.8),
        ]
        result = format_web_sources_with_ids(sources)
        assert "[WEB_1]" in result
        assert "[WEB_2]" in result
        assert "Reuters Article" in result
        assert "reuters.com/a" in result

    def test_empty_sources(self):
        assert format_web_sources_with_ids([]) == ""

    def test_max_chars_truncation(self):
        sources = [
            NumberedWebSource("WEB_1", "Title", "https://a.com", "a.com", "x" * 5000, 0.9),
            NumberedWebSource("WEB_2", "Title2", "https://b.com", "b.com", "y" * 5000, 0.8),
        ]
        result = format_web_sources_with_ids(sources, max_chars=200)
        assert len(result) <= 250  # some overhead for markers


# ---------------------------------------------------------------------------
# Citation pattern matching for [WEB_N]
# ---------------------------------------------------------------------------

class TestCitationPatternWEB:

    def _get_pattern(self):
        """Get the citation regex from orchestrator."""
        return re.compile(
            r'\[('
            r'WEB_\d+|'
            r'MEM_\w+_\d+(?:-\d+)?|'
            r'SUM_\w+_\d+(?:-\d+)?|'
            r'REFL_\w+_\d+(?:-\d+)?|'
            r'FACT_\d+(?:-\d+)?|'
            r'PROFILE_\w+'
            r')\]'
        )

    def test_matches_web_citations(self):
        pattern = self._get_pattern()
        text = "According to Reuters [WEB_1], markets rose. NASA confirmed [WEB_3] the mission."
        matches = pattern.findall(text)
        assert "WEB_1" in matches
        assert "WEB_3" in matches

    def test_matches_mixed_citations(self):
        pattern = self._get_pattern()
        text = "You mentioned [MEM_RECENT_3] that. Also [WEB_2] reported this."
        matches = pattern.findall(text)
        assert "MEM_RECENT_3" in matches
        assert "WEB_2" in matches

    def test_invalid_web_id_not_matched(self):
        pattern = self._get_pattern()
        text = "This is [WEB] without a number"
        matches = pattern.findall(text)
        assert len(matches) == 0

    def test_strips_cleanly(self):
        pattern = self._get_pattern()
        text = "According to Reuters [WEB_1], markets rose."
        cleaned = pattern.sub('', text)
        assert "[WEB_1]" not in cleaned
        assert "According to Reuters" in cleaned
