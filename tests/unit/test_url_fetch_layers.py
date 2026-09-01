"""Layered URL fetch (2026-08-29): local direct fetch + SPA embedded-JSON
salvage first, Tavily extract fallback.

Live failure: a chatgpt.com/share link — the whole conversation embedded as a
react-router turbo-stream payload — came back "[blank page]" through the
Tavily-only fetch path, and the raw URL became the stored topic label.
"""

import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge.web_search_manager import WebSearchManager, WebPage
from utils.page_extract import (
    MIN_VISIBLE_CHARS,
    _looks_like_prose,
    extract_page_text,
)


PROSE = (
    "Yes, in theory that is a coherent way to represent a person over time. "
    "I would treat you as a trajectory through a changing attractor landscape."
)
PROSE2 = (
    "Could I be expressed as a specific set of events, a starting attractor "
    "state and a series of rules covering how both of these evolve over time?"
)


def _spa_shell(script_body: str) -> str:
    return (
        "<!DOCTYPE html><html><head><title>ChatGPT - Shared</title></head>"
        "<body><div id='root'></div>"
        f"<script>{script_body}</script></body></html>"
    )


class TestPageExtract:

    def test_visible_text_path(self):
        body = "<p>" + "Real paragraph content here. " * 40 + "</p>"
        html = f"<html><head><title>Doc</title></head><body>{body}</body></html>"
        title, text, method = extract_page_text(html, "https://example.com")
        assert method == "visible"
        assert title == "Doc"
        assert "Real paragraph content" in text

    def test_turbo_stream_enqueue_salvage(self):
        # Flat reference array, chatgpt.com/share shape: content lives as
        # long strings among ids/urls/machine tokens.
        payload = json.dumps([
            {"_1": 2}, "loaderData", PROSE,
            "https://example.com/some/asset.png",
            "a" * 80,  # base64ish blob
            PROSE2,
        ])
        script = f'window.__sc.streamController.enqueue({json.dumps(payload)});'
        title, text, method = extract_page_text(_spa_shell(script), "https://example.com")
        assert method == "embedded_json"
        assert PROSE in text and PROSE2 in text
        assert "asset.png" not in text
        assert "aaaa" not in text

    def test_next_data_salvage(self):
        payload = json.dumps({"props": {"pageProps": {"post": {"body": PROSE}}}})
        html = (
            "<html><head><title>T</title></head><body>"
            f"<script id=\"__NEXT_DATA__\" type=\"application/json\">{payload}</script>"
            "</body></html>"
        )
        _, text, method = extract_page_text(html, "https://example.com")
        assert method == "embedded_json"
        assert PROSE in text

    def test_ld_json_salvage(self):
        payload = json.dumps({"@type": "Article", "articleBody": PROSE})
        html = (
            "<html><body>"
            f"<script type=\"application/ld+json\">{payload}</script>"
            "</body></html>"
        )
        _, text, method = extract_page_text(html, "https://example.com")
        assert method == "embedded_json"
        assert PROSE in text

    def test_visible_text_wins_over_salvage_when_substantive(self):
        body = "<p>" + "Visible article text. " * 60 + "</p>"
        payload = json.dumps({"k": PROSE})
        html = (
            f"<html><body>{body}"
            f"<script type=\"application/json\">{payload}</script></body></html>"
        )
        _, text, method = extract_page_text(html, "https://example.com")
        assert method == "visible"
        assert "Visible article text" in text

    def test_salvage_dedupes_and_keeps_order(self):
        payload = json.dumps([PROSE, PROSE2, PROSE])
        script = f'x.enqueue({json.dumps(payload)});'
        _, text, _ = extract_page_text(_spa_shell(script), "https://example.com")
        assert text.count(PROSE) == 1
        assert text.index(PROSE) < text.index(PROSE2)

    def test_empty_and_unparseable(self):
        assert extract_page_text("", "u") == ("", "", "none")
        _, text, method = extract_page_text("<html><body></body></html>", "u")
        assert text == "" and method == "none"

    def test_prose_filter(self):
        assert _looks_like_prose(PROSE)
        assert not _looks_like_prose("https://example.com/page?a=1&b=2%20c=3&d=44")
        assert not _looks_like_prose("QWxhZGRpbjpvcGVuIHNlc2FtZQ" * 4)
        assert not _looks_like_prose(".cls{color:red;margin:0;padding:0;border:0;outline:0;background:#fff}")
        assert not _looks_like_prose("short string")

    def test_output_capped(self):
        from utils.page_extract import MAX_EXTRACT_CHARS
        body = "<p>" + "words and more words here. " * 5000 + "</p>"
        _, text, _ = extract_page_text(f"<html><body>{body}</body></html>", "u")
        assert len(text) <= MAX_EXTRACT_CHARS


def _bare_manager():
    mgr = object.__new__(WebSearchManager)
    mgr.max_content_chars = 10000
    mgr._tavily_client = None
    return mgr


def _page(content: str, source: str = "direct_fetch") -> WebPage:
    return WebPage(url="https://example.com", title="T", content=content,
                   snippet=content[:500], source=source)


class TestFetchUrlContentLayering:

    @pytest.mark.asyncio
    async def test_direct_substantive_skips_tavily(self):
        mgr = _bare_manager()
        mgr._direct_fetch = AsyncMock(return_value=[_page("x" * 5000)])
        mgr._tavily_extract = AsyncMock()
        pages = await mgr.fetch_url_content("https://example.com")
        assert pages[0].source == "direct_fetch"
        mgr._tavily_extract.assert_not_awaited()  # no credits spent

    @pytest.mark.asyncio
    async def test_direct_thin_falls_to_tavily(self):
        mgr = _bare_manager()
        mgr._direct_fetch = AsyncMock(return_value=[_page("thin")])
        mgr._tavily_extract = AsyncMock(
            return_value=[_page("y" * 3000, source="tavily_extract")])
        pages = await mgr.fetch_url_content("https://example.com")
        assert pages[0].source == "tavily_extract"

    @pytest.mark.asyncio
    async def test_direct_thin_kept_when_tavily_empty(self):
        mgr = _bare_manager()
        mgr._direct_fetch = AsyncMock(return_value=[_page("thin but real")])
        mgr._tavily_extract = AsyncMock(return_value=[])
        pages = await mgr.fetch_url_content("https://example.com")
        assert pages and pages[0].content == "thin but real"

    @pytest.mark.asyncio
    async def test_bigger_direct_beats_smaller_tavily(self):
        mgr = _bare_manager()
        mgr._direct_fetch = AsyncMock(return_value=[_page("d" * 399)])
        mgr._tavily_extract = AsyncMock(return_value=[_page("t" * 50, source="tavily_extract")])
        pages = await mgr.fetch_url_content("https://example.com")
        assert pages[0].source == "direct_fetch"

    @pytest.mark.asyncio
    async def test_both_fail(self):
        mgr = _bare_manager()
        mgr._direct_fetch = AsyncMock(return_value=[])
        mgr._tavily_extract = AsyncMock(return_value=[])
        assert await mgr.fetch_url_content("https://example.com") == []

    @pytest.mark.asyncio
    async def test_env_kill_switch(self, monkeypatch):
        monkeypatch.setenv("WEB_FETCH_DIRECT_ENABLED", "0")
        mgr = _bare_manager()
        mgr._direct_fetch = AsyncMock()
        mgr._tavily_extract = AsyncMock(
            return_value=[_page("t" * 1000, source="tavily_extract")])
        pages = await mgr.fetch_url_content("https://example.com")
        assert pages[0].source == "tavily_extract"
        mgr._direct_fetch.assert_not_awaited()


def _mock_httpx_response(status=200, ctype="text/html", text=""):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": ctype}
    resp.text = text
    return resp


def _mock_httpx_client(resp):
    client = AsyncMock()
    client.get.return_value = resp
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class TestDirectFetch:

    @pytest.fixture(autouse=True)
    def _public_dns(self):
        # Direct-fetch unit tests replace the HTTP client; avoid depending on
        # the sandbox DNS resolver while exercising extraction behavior.
        with patch("knowledge.web_search_manager._validate_fetch_url_dns", new=AsyncMock()):
            yield

    @pytest.mark.asyncio
    async def test_html_page_extracted(self):
        mgr = _bare_manager()
        html = ("<html><head><title>Doc</title></head><body><p>"
                + "Body text of the page. " * 60 + "</p></body></html>")
        client = _mock_httpx_client(_mock_httpx_response(text=html))
        with patch("httpx.AsyncClient", return_value=client):
            pages = await mgr._direct_fetch("http://example.com/a")
        assert pages and pages[0].title == "Doc"
        assert "Body text of the page" in pages[0].content
        assert pages[0].source == "direct_fetch"

    @pytest.mark.asyncio
    async def test_http_error_returns_empty(self):
        mgr = _bare_manager()
        client = _mock_httpx_client(_mock_httpx_response(status=403))
        with patch("httpx.AsyncClient", return_value=client):
            assert await mgr._direct_fetch("http://example.com/a") == []

    @pytest.mark.asyncio
    async def test_binary_content_type_defers_to_tavily(self):
        mgr = _bare_manager()
        client = _mock_httpx_client(
            _mock_httpx_response(ctype="application/pdf", text="%PDF"))
        with patch("httpx.AsyncClient", return_value=client):
            assert await mgr._direct_fetch("http://example.com/a.pdf") == []

    @pytest.mark.asyncio
    async def test_network_error_returns_empty(self):
        mgr = _bare_manager()
        client = AsyncMock()
        client.__aenter__ = AsyncMock(side_effect=OSError("refused"))
        with patch("httpx.AsyncClient", return_value=client):
            assert await mgr._direct_fetch("http://example.com/a") == []

    @pytest.mark.asyncio
    async def test_json_content_passed_through(self):
        mgr = _bare_manager()
        payload = json.dumps({"data": PROSE})
        client = _mock_httpx_client(
            _mock_httpx_response(ctype="application/json", text=payload))
        with patch("httpx.AsyncClient", return_value=client):
            pages = await mgr._direct_fetch("http://api.example.com/x")
        assert pages and PROSE in pages[0].content


class TestToolWiring:

    def test_execute_fetch_url_uses_layered_method(self):
        from core.agentic.tools import ToolExecutor
        src = inspect.getsource(ToolExecutor._execute_fetch_url)
        assert "fetch_url_content" in src
        assert "_tavily_extract" not in src

    @pytest.mark.asyncio
    async def test_execute_fetch_url_end_to_end(self):
        from core.agentic.tools import ToolExecutor
        executor = object.__new__(ToolExecutor)
        executor.web_search_manager = MagicMock()
        executor.web_search_manager.fetch_url_content = AsyncMock(
            return_value=[_page(PROSE * 10)])
        executor._current_web_source_map = {}
        executor._merge_web_ids = MagicMock(return_value=[])
        result = await executor._execute_fetch_url("https://example.com")
        assert PROSE in result


class TestTopicUrlStrip:
    """The share link had become the stored topic label verbatim."""

    def _manager(self):
        from utils.topic_manager import TopicManager
        return TopicManager(model_manager=None, enable_llm_fallback=False)

    def test_live_share_message_topic_has_no_url(self):
        tm = self._manager()
        topic = tm._extract_primary_from_text(
            "Thank you. I want to share these thoughts I had using chatgpt "
            "https://chatgpt.com/share/6a93702d-b828-83ea-9c91-83d6c3b317b7?ogimg=plain"
        )
        assert not topic or "http" not in topic.lower()

    def test_url_only_message_yields_no_topic(self):
        tm = self._manager()
        topic = tm._extract_primary_from_text(
            "https://chatgpt.com/share/6a93702d-b828-83ea-9c91-83d6c3b317b7"
        )
        assert not topic or "http" not in (topic or "").lower()

    def test_normal_topics_unaffected(self):
        tm = self._manager()
        topic = tm._extract_primary_from_text(
            "Tell me about the President of the United States"
        )
        assert topic and "http" not in topic.lower()
