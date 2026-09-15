# /knowledge/web_search_manager.py
"""
WebSearchManager - Tavily-based web search with caching, rate limiting, and LLM-driven link following.

Module Contract:
- Purpose: Provide real-time web search capabilities for queries requiring current information
- Inputs:
  - Query text and optional search parameters (depth, timeout)
  - Crisis level to suppress search during therapeutic moments
- Outputs:
  - WebSearchResult containing relevant web content with sources
  - NumberedWebSource + web_source_map for stable [WEB_N] citation IDs (assigned centrally after merge/dedupe).
    assign_web_ids() accepts existing_url_to_id + start_index so multi-round agentic search
    CONTINUES numbering across rounds instead of restarting at WEB_1 (avoids two distinct
    sources colliding on [WEB_1]); the per-round merge state lives in ToolExecutor._merge_web_ids.
- Side effects:
  - Network requests to Tavily API
  - ChromaDB cache writes (72-hour TTL)
  - Credit tracking for rate limiting

Enhanced Features (2026-01):
- Query Decomposition: Complex queries are split into sub-queries for parallel search
  - LLM-based detection of multi-entity/multi-facet queries
  - Parallel Tavily searches with result merging and deduplication
  - Credit-aware: respects daily limits, caps sub-queries at 4
- News Detection: Automatically detects news/current events queries
  - Uses Tavily's news-optimized agent (topic="news") for better results
  - Limits to recent results (days=1) for "today" queries
  - Keyword/phrase detection for news intent
  - Semantic similarity fallback: _semantic_broad_news_check() compares query
    embedding against 5 news anchors vs 4 non-news anchors (threshold 0.45)
  - Named entity check uses mid-sentence capitals only (sentence-initial words skipped)
- Entry point: multi_search() for decomposed queries, search() for single queries
  - sub_queries param: callers can pass pre-computed sub-queries for parallel search
- Query Localization (2026-07-02, narrowed 2026-07-04): every query entering
  search() passes _localize_query() — literal deictic phrases ("my area",
  "near me") are replaced with the user's location (utils/location_resolver.py:
  override → IP geo → profile), and placeless CURRENT-CONDITIONS weather queries
  get the location appended (bare "temperature"/"humidity" require a cue like
  "outside"/"right now"; named-place detection is case-insensitive so
  "weather in tokyo" is never rewritten). search(localize=False) bypasses
  localization entirely — instrument callers (literature oracle) need queries
  to reach Tavily verbatim. Runs BEFORE the cache check so localized/unlocalized
  text never share a cache entry. decompose_query() also carries the location in
  its LLM prompt. Regression guard for the "'my area' weather query returned DC
  news" bug.
- Localization scope guard (2026-07-08): location is for PHYSICAL-SURROUNDINGS
  queries only — the decompose prompt forbids attaching it to institution/
  account/login sub-queries, and parsed sub_queries pass through
  location_resolver.strip_unjustified_location() (deterministic backstop).
  Regression guard for the wrong-college incident: a school-login query
  localized to "Springfield IL" retrieved Springfield Community College and
  the response presented its IT desk number as the user's school's.
"""

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import threading
import time
import urllib.parse
import weakref
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import re as _re

from utils.retrieval_outcome import OutcomeList, RetrievalError

log = logging.getLogger(__name__)


class UnsafeFetchURLError(ValueError):
    """Raised when a requested URL could access a local or private resource."""


_BLOCKED_HOST_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".lan",
    ".home",
    ".corp",
    ".home.arpa",
)


def _ensure_public_ip(address: ipaddress._BaseAddress) -> None:
    """Reject every address that is not globally routable."""
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    if not address.is_global:
        raise UnsafeFetchURLError(f"non-public network address is not allowed: {address}")


def _validate_fetch_url_syntax(url: str) -> urllib.parse.SplitResult:
    """Validate URL structure before either local or third-party extraction.

    DNS is checked separately immediately before each direct request. Keeping the
    structural check here also prevents an unsafe URL from being handed to the
    Tavily extract fallback when direct fetching is disabled.
    """
    if not isinstance(url, str) or not url.strip():
        raise UnsafeFetchURLError("URL is empty")
    try:
        parsed = urllib.parse.urlsplit(url.strip())
        port = parsed.port  # Force validation of malformed/out-of-range ports.
    except ValueError as exc:
        raise UnsafeFetchURLError(f"invalid URL: {exc}") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise UnsafeFetchURLError("only http:// and https:// URLs are allowed")
    if not parsed.hostname:
        raise UnsafeFetchURLError("URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeFetchURLError("URLs containing credentials are not allowed")
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(_BLOCKED_HOST_SUFFIXES):
        raise UnsafeFetchURLError(f"local hostname is not allowed: {host}")
    if ":" in host and "%" in host:
        raise UnsafeFetchURLError("scoped IPv6 addresses are not allowed")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # Single-label names commonly resolve only inside a private network.
        if "." not in host:
            raise UnsafeFetchURLError(f"non-public hostname is not allowed: {host}")
    else:
        _ensure_public_ip(address)
    _ = port
    return parsed


async def _validate_fetch_url_dns(url: str) -> None:
    """Resolve a URL hostname and reject private/special DNS answers."""
    parsed = _validate_fetch_url_syntax(url)
    host = parsed.hostname or ""
    try:
        ipaddress.ip_address(host)
        return  # Literal addresses were validated by the syntax pass.
    except ValueError:
        pass

    def _resolve() -> list[tuple]:
        return socket.getaddrinfo(
            host,
            parsed.port or (443 if parsed.scheme.lower() == "https" else 80),
            type=socket.SOCK_STREAM,
        )

    answers = await asyncio.to_thread(_resolve)
    if not answers:
        raise OSError(f"hostname did not resolve: {host}")
    for answer in answers:
        _ensure_public_ip(ipaddress.ip_address(answer[4][0].split("%", 1)[0]))

# News detection patterns
NEWS_KEYWORDS = {
    'news', 'today', 'yesterday', 'latest', 'recent', 'breaking',
    'happened', 'happening', 'current events', 'headlines', 'update',
    'this week', 'this month', 'announced', 'reports', 'reported'
}
NEWS_PHRASES = [
    'what happened', 'what\'s happening', 'whats happening',
    'news in', 'news from', 'news about', 'news today',
    'current events', 'latest news', 'breaking news',
    'today in', 'yesterday in', 'this week in'
]


def _is_news_query(query: str) -> bool:
    """
    Detect if a query is asking for news/current events.

    Returns True if query contains news-related keywords or phrases.
    """
    query_lower = query.lower()

    # Check for news phrases first (more specific)
    for phrase in NEWS_PHRASES:
        if phrase in query_lower:
            return True

    # Check for keyword combinations (need at least one strong indicator)
    words = set(query_lower.split())
    news_word_matches = words & NEWS_KEYWORDS

    # "today"/"yesterday" is a weak signal on its own — a casual life-update
    # ("I did not shit today. Went to my dad's to swim...") must NOT be treated
    # as a news query (that mislabel forced topic='news', days=1 on Tavily). They
    # also don't count toward the "2 news keywords" rule below, or "feeling
    # better today than yesterday" would qualify. Only genuine news words
    # (news/headlines/breaking/happened/...) count; a temporal word is news only
    # when it co-occurs with one of those. The old len>20 rule fired on any
    # longish sentence containing "today".
    temporal_words = {'today', 'yesterday'}
    strong_news_matches = news_word_matches - temporal_words
    if (words & temporal_words) and strong_news_matches:
        return True

    # Multiple genuine news keywords suggest news intent
    if len(strong_news_matches) >= 2:
        return True

    # Single strong indicators
    if news_word_matches & {'news', 'headlines', 'breaking', 'current events'}:
        return True

    return False


class WebSearchDepth(Enum):
    """Search depth levels with associated credit costs."""
    QUICK = "quick"      # ~1 credit: snippets only
    STANDARD = "standard"  # ~2 credits: search + extract top 2
    DEEP = "deep"        # ~3-5 credits: search + extract + LLM link following


@dataclass
class WebPage:
    """Individual web page result from search or extraction."""
    url: str
    title: str
    content: str
    snippet: str = ""
    score: float = 0.0
    published_date: Optional[str] = None
    source: str = "tavily"  # "tavily_search" or "tavily_extract"


class FetchedPages(list):
    """``List[WebPage]`` that also records why a fetch came back empty.

    ``blocked`` is "budget" when nothing free was fetched and the billed
    fallback could not be paid for; otherwise None. Plain ``list`` equality
    still holds (an empty ``FetchedPages`` compares equal to ``[]``), so
    existing "not pages"/"pages == []" checks keep working unchanged."""

    def __init__(self, pages=(), *, blocked: Optional[str] = None):
        super().__init__(pages)
        self.blocked = blocked


@dataclass
class WebSearchResult:
    """Complete result from a web search operation."""
    query: str
    pages: List[WebPage] = field(default_factory=list)
    total_credits_used: float = 0.0
    search_depth: WebSearchDepth = WebSearchDepth.QUICK
    from_cache: bool = False
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None
    # Typed block reason (2026-09-12, adversarial-review follow-up findings
    # 2/3): "budget" when the daily search-credit budget refused this call
    # (or, for a merged MultiSearchResult, refused at least one sub-query).
    # None otherwise — a disabled toggle or any other failure stays in
    # ``error`` only, this field is reserved for the one case a receipt must
    # never lose track of even when other pages/results came back fine.
    blocked: Optional[str] = None
    # CGR-20260913-008/F2: set when a STANDARD/DEEP extract call raised
    # RetrievalError AFTER the base search already returned pages — the
    # search pages are kept (``error`` stays None so ``has_results``/caching
    # treat this as a partial success) and this field carries
    # ``f"tavily_extract:{reason}"`` instead. Additive, default None; no
    # consumer reads it in F2 (F6 surfaces it in a receipt).
    extract_error: Optional[str] = None

    @property
    def has_results(self) -> bool:
        return len(self.pages) > 0 and not self.error

    def get_formatted_content(self, max_chars: int = 10000) -> str:
        """Get formatted content for prompt injection."""
        if not self.pages:
            return ""

        parts = []
        total_chars = 0
        for page in self.pages:
            if total_chars >= max_chars:
                break
            content = page.content or page.snippet
            if not content:
                continue
            entry = f"**{page.title}** ({page.url})\n{content}"
            if total_chars + len(entry) > max_chars:
                # Truncate this entry to fit
                remaining = max_chars - total_chars
                if remaining > 100:  # Only add if we have reasonable space
                    entry = entry[:remaining] + "..."
                    parts.append(entry)
                break
            parts.append(entry)
            total_chars += len(entry) + 2  # +2 for newlines

        return "\n\n".join(parts)


@dataclass
class WebSearchSession:
    """Tracks a complete search session including link following."""
    initial_query: str
    depth: WebSearchDepth
    search_results: List[WebPage] = field(default_factory=list)
    extracted_pages: List[WebPage] = field(default_factory=list)
    followed_links: List[str] = field(default_factory=list)
    credits_used: float = 0.0
    start_time: float = field(default_factory=time.time)

    @property
    def all_pages(self) -> List[WebPage]:
        """Combined search and extracted pages, deduplicated."""
        seen = set()
        pages = []
        for p in self.search_results + self.extracted_pages:
            if p.url not in seen:
                seen.add(p.url)
                pages.append(p)
        return pages


@dataclass
class QueryDecomposition:
    """
    Result of query decomposition analysis.

    Used to determine if a complex query should be split into multiple
    sub-queries for parallel search.
    """
    original_query: str
    should_decompose: bool
    sub_queries: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""

    @property
    def query_count(self) -> int:
        """Number of queries to execute (1 if no decomposition)."""
        return len(self.sub_queries) if self.should_decompose else 1


@dataclass
class MultiSearchResult:
    """
    Result from a multi-query search operation.

    Extends WebSearchResult with decomposition metadata.
    """
    original_query: str
    sub_queries: List[str] = field(default_factory=list)
    pages: List[WebPage] = field(default_factory=list)
    total_credits_used: float = 0.0
    search_depth: WebSearchDepth = WebSearchDepth.QUICK
    from_cache: bool = False
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None
    decomposition_used: bool = False
    # See WebSearchResult.blocked — "budget" when at least one sub-query was
    # refused by the daily budget, even when other sub-queries returned
    # pages and ``error`` was therefore cleared.
    blocked: Optional[str] = None

    @property
    def has_results(self) -> bool:
        return len(self.pages) > 0 and not self.error

    def get_formatted_content(self, max_chars: int = 10000) -> str:
        """Get formatted content for prompt injection."""
        if not self.pages:
            return ""

        parts = []
        total_chars = 0
        for page in self.pages:
            if total_chars >= max_chars:
                break
            content = page.content or page.snippet
            if not content:
                continue
            entry = f"**{page.title}** ({page.url})\n{content}"
            if total_chars + len(entry) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    entry = entry[:remaining] + "..."
                    parts.append(entry)
                break
            parts.append(entry)
            total_chars += len(entry) + 2

        return "\n\n".join(parts)

    def to_web_search_result(self) -> 'WebSearchResult':
        """Convert to standard WebSearchResult for backward compatibility."""
        return WebSearchResult(
            query=self.original_query,
            pages=self.pages,
            total_credits_used=self.total_credits_used,
            search_depth=self.search_depth,
            from_cache=self.from_cache,
            timestamp=self.timestamp,
            error=self.error,
            blocked=self.blocked,
        )


@dataclass
class NumberedWebSource:
    """Web page with stable [WEB_N] source ID assigned after merge/dedupe."""
    source_id: str      # "WEB_1", "WEB_2", etc.
    title: str
    url: str
    domain: str
    content: str        # snippet or extracted text
    score: float = 0.0


def _canonical_url(url: str) -> str:
    """Canonical form for dedupe: strip trailing slash, fragment, query."""
    return (url or "").rstrip("/").split("#")[0].split("?")[0]


def assign_web_ids(
    pages: List[WebPage],
    existing_url_to_id: Optional[Dict[str, str]] = None,
    start_index: int = 0,
) -> Tuple[List[NumberedWebSource], Dict[str, Dict[str, str]]]:
    """
    Assign stable WEB_N IDs after merge/dedupe.

    Deduplicates by canonical URL, ranks by score, assigns sequential IDs.
    Returns (numbered_sources, web_source_map) for the NEWLY-numbered sources.

    For multi-round agentic search, pass ``existing_url_to_id`` (canonical URL →
    already-assigned "WEB_k") and ``start_index`` (count of IDs already assigned)
    so numbering CONTINUES across rounds instead of restarting at WEB_1 —
    otherwise a later round's first source collides with an earlier round's
    WEB_1 (two different sources cited as [WEB_1]). Pages whose canonical URL is
    already known are skipped (they keep their prior ID). Default args reproduce
    the legacy single-shot behavior.

    web_source_map: {"WEB_1": {"title": ..., "url": ..., "domain": ...}, ...}
    """
    if not pages:
        return [], {}

    existing_url_to_id = existing_url_to_id or {}

    # Dedupe by canonical URL (strip trailing slash, fragments); skip URLs that
    # were already numbered in a previous round.
    seen_urls: Dict[str, WebPage] = {}
    for page in pages:
        canonical = _canonical_url(page.url)
        if canonical in existing_url_to_id:
            continue
        if canonical not in seen_urls or page.score > seen_urls[canonical].score:
            seen_urls[canonical] = page

    # Rank by score descending
    ranked = sorted(seen_urls.values(), key=lambda p: p.score, reverse=True)

    numbered = []
    source_map = {}
    for idx, page in enumerate(ranked):
        source_id = f"WEB_{start_index + idx + 1}"
        domain = ""
        try:
            from urllib.parse import urlparse
            domain = urlparse(page.url).netloc.replace("www.", "")
        except Exception:
            pass
        content = page.content or page.snippet or ""
        numbered.append(NumberedWebSource(
            source_id=source_id,
            title=page.title,
            url=page.url,
            domain=domain,
            content=content,
            score=page.score,
        ))
        source_map[source_id] = {
            "title": page.title,
            "url": page.url,
            "domain": domain,
        }

    return numbered, source_map


def render_prenumbered_web_sources(
    numbered_sources: List[NumberedWebSource],
    *,
    max_sources: int = 8,
    max_chars_per_source: int = 2000,
) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """
    Format ALREADY-numbered sources (ids already assigned by a prior
    ``assign_web_ids``/``_merge_web_ids`` call) — factored out of
    ``render_numbered_web_sources`` so a caller that must assign ids itself
    first (e.g. the agentic controller's ``ToolExecutor._merge_web_ids``,
    which threads the session-wide id map so numbering stays unique across
    rounds) can reuse the exact same per-source formatting without calling
    ``assign_web_ids`` a second time — that would mint a SECOND, colliding
    set of ids for the same pages.

    Formats each kept source as
    ``f"[{sid}] **{title}** ({url})\\n{content}"`` with content clipped to
    max_chars_per_source ("..." suffix when clipped). Pages with empty
    content are skipped entirely. Returns (lines, source_map) where
    source_map contains ONLY the ids that were actually rendered — a source
    cut off by max_sources, or skipped for empty content, never gets an id
    in the map (a clipped-away source can't be cited as though it were shown).
    """
    lines: List[str] = []
    source_map: Dict[str, Dict[str, str]] = {}
    for src in numbered_sources[:max_sources]:
        content = src.content
        if not content:
            continue
        if len(content) > max_chars_per_source:
            content = content[:max_chars_per_source] + "..."
        lines.append(f"[{src.source_id}] **{src.title}** ({src.url})\n{content}")
        source_map[src.source_id] = {
            "title": src.title,
            "url": src.url,
            "domain": src.domain,
        }
    return lines, source_map


def render_numbered_web_sources(
    pages: List[WebPage],
    *,
    max_sources: int = 8,
    max_chars_per_source: int = 2000,
    existing_url_to_id: Optional[Dict[str, str]] = None,
    start_index: int = 0,
) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """
    Render numbered [WEB_N] source blocks EXACTLY as the formatter's inline
    web-search block does (2026-09-06 evidence-transport fix) — the single
    shared renderer so the base prompt path, the token-budget shrink ladder,
    and the agentic pre-gathered-web path can never drift from each other.

    Calls assign_web_ids for ranking/dedupe/numbering (pass-through
    existing_url_to_id/start_index for continuous numbering across rounds),
    then delegates the per-source formatting to
    render_prenumbered_web_sources.
    """
    numbered_sources, _ = assign_web_ids(
        pages, existing_url_to_id=existing_url_to_id, start_index=start_index
    )
    return render_prenumbered_web_sources(
        numbered_sources, max_sources=max_sources, max_chars_per_source=max_chars_per_source
    )


def trim_web_search_result(
    result: "WebSearchResult",
    *,
    max_sources: int,
    max_chars_per_source: int,
) -> "WebSearchResult":
    """
    Return a NEW WebSearchResult (dataclasses.replace) whose pages are the
    top max_sources after assign_web_ids ranking (dedupe by canonical URL,
    score descending), each page a new WebPage with content clipped to
    max_chars_per_source. Never mutates the input — cached WebSearchResult
    objects are shared across turns/callers. query/from_cache/timestamp/
    error/search_depth/total_credits_used are preserved unchanged.
    """
    numbered_sources, _ = assign_web_ids(result.pages)
    original_by_canonical: Dict[str, WebPage] = {}
    for page in result.pages:
        canonical = _canonical_url(page.url)
        if (
            canonical not in original_by_canonical
            or page.score > original_by_canonical[canonical].score
        ):
            original_by_canonical[canonical] = page

    trimmed_pages: List[WebPage] = []
    for src in numbered_sources[:max_sources]:
        base = original_by_canonical.get(_canonical_url(src.url))
        if base is None:
            # Defensive only — every numbered source is derived from
            # result.pages, so this should be unreachable. Never fabricate
            # a page silently if it somehow is.
            base = WebPage(url=src.url, title=src.title, content=src.content, score=src.score)
        content = base.content or ""
        if content and len(content) > max_chars_per_source:
            content = content[:max_chars_per_source] + "..."
        trimmed_pages.append(replace(base, content=content))

    return replace(result, pages=trimmed_pages)


def format_web_sources_with_ids(
    numbered_sources: List[NumberedWebSource],
    max_chars: int = 10000,
) -> str:
    """Format web sources with [WEB_N] markers for prompt injection."""
    if not numbered_sources:
        return ""
    parts = []
    total_chars = 0
    for src in numbered_sources:
        if total_chars >= max_chars:
            break
        content = src.content
        if not content:
            continue
        entry = f"[{src.source_id}] **{src.title}** ({src.url})\n{content}"
        if total_chars + len(entry) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                entry = entry[:remaining] + "..."
                parts.append(entry)
            break
        parts.append(entry)
        total_chars += len(entry) + 2
    return "\n\n".join(parts)


DISABLED_ERROR = "Web search is disabled in Settings"


# Live rate limiters, so any module can ask what the REAL remaining search
# budget is without holding a manager reference (2026-09-12). The agentic
# gate's Tier-4 trigger call passed no credit count and therefore assumed the
# default 100 while the prompt builder's call passed the live number: on
# 2026-09-11 the true budget was 0 from 19:11 onward, so the two calls landed
# in different cache buckets ("ok" vs "none"), paid gpt-4o-mini TWICE per turn
# (87 calls in one session, 12 cache hits) and returned CONTRADICTORY verdicts
# for the same message — the gate's True routed 5 turns into 24-28 s agentic
# loops the gatherer's False had declined, and 83 searches ran into
# "Daily limit reached". WeakSet: a limiter is registered for as long as its
# manager lives, and never keeps one alive.
_LIVE_RATE_LIMITERS: "weakref.WeakSet" = weakref.WeakSet()

# Cheapest search Tavily will bill (QUICK) — the floor under which a search
# cannot be funded at all. Keep in step with estimate_credits().
MIN_SEARCH_CREDITS = 1.0


def live_remaining_credits() -> Optional[float]:
    """Smallest remaining daily budget across live rate limiters, or None when
    no limiter exists in this process (tests, scripts, search disabled)."""
    values = []
    for limiter in list(_LIVE_RATE_LIMITERS):
        try:
            values.append(float(limiter.get_remaining_credits()))
        except Exception:
            continue
    return min(values) if values else None


@dataclass
class SearchReservation:
    """A provisional credit hold created by ``WebSearchRateLimiter.reserve()``.

    Charge-on-dispatch: ``spend(cost)`` must be called immediately BEFORE a
    billable provider call is dispatched, and commits that cost into
    ``used`` right away — a dispatched call is billed even if the overall
    search is later cancelled or times out, so ``used`` only ever grows.
    A charge lands on the day its call is actually DISPATCHED, not the day
    the reservation was first taken (2026-09-12): ``day_used`` tracks the
    portion of ``used`` charged against ``date`` — the current dispatch
    day — and a ``spend()`` that finds the reservation's day has rolled
    over re-books it on today's budget (empty hold) before charging this
    dispatch, so a search that starts before midnight and keeps dispatching
    provider calls after it competes for TODAY's budget like anything else,
    instead of spending against a day the limiter has already closed the
    books on.
    ``settle()`` is idempotent and MUST run in a ``finally``: it folds
    ``day_used`` into the limiter's real daily counter and releases the
    reservation's outstanding hold (``amount``, including any extensions),
    but only against the day the reservation is CURRENTLY booked on — a
    midnight rollover between the last ``spend()`` and ``settle()`` neither
    debits nor releases against the (already-reset) new day.
    """
    limiter: "WebSearchRateLimiter"
    amount: float
    date: str
    used: float = 0.0
    # Part of ``used`` charged against ``date`` (today's dispatch day). See
    # the class docstring — this is what actually gets folded into
    # ``_credits_today`` on settle, never the full lifetime ``used``.
    day_used: float = 0.0
    _settled: bool = False

    def spend(self, cost: float) -> bool:
        """Commit ``cost`` against this reservation before dispatching a
        billable call. Extends the reservation (against the limiter's daily
        cap) when the hold is too small; returns False — dispatch nothing —
        when even an extension can't afford it."""
        return self.limiter._reservation_spend(self, cost)

    def settle(self) -> None:
        """Fold ``used`` into the limiter's real counter and release the
        rest of the hold. Safe to call more than once; only the first call
        does anything."""
        self.limiter._reservation_settle(self)


class WebSearchRateLimiter:
    """
    Credit-aware rate limiter for Tavily API.

    Tracks daily credit usage and enforces limits to prevent overuse.
    Designed for Tavily free tier (1000 credits/month).

    Reservation API (2026-09-12): two concurrent callers each checking
    ``can_search()`` before ever recording usage could both be admitted for
    the same last credit — ``can_search``/``record_usage`` are a
    check-then-act race with no hold in between. ``reserve()`` closes it by
    debiting the budget UP FRONT (into ``_reserved_today``, alongside
    ``_credits_today``) so a second concurrent caller sees the first
    reservation's amount and can be correctly refused; ``record_usage()``
    still works unchanged for any direct caller that never reserves. A
    plain ``threading.Lock`` guards the shared counters — the event loop
    itself is single-threaded, but Tavily calls run in executor threads.

    Cross-midnight dispatch (2026-09-12, adversarial-review follow-up
    finding 2): a charge lands on the day its provider call is actually
    DISPATCHED. ``_reservation_spend`` re-books a reservation whose ``date``
    predates today onto today's budget (with an empty hold) before charging
    it, so a search that reserved before midnight and is still dispatching
    calls after it competes for TODAY's budget instead of silently spending
    against a day the limiter has already reset and closed the books on.
    """

    def __init__(
        self,
        daily_limit: int = 100,
        per_query_limit: int = 5,
        state_file: Optional[str] = None
    ):
        self.daily_limit = daily_limit
        self.per_query_limit = per_query_limit
        if state_file:
            self.state_file = state_file
        else:
            try:
                from config.app_config import WEB_SEARCH_CREDITS_PATH
                self.state_file = WEB_SEARCH_CREDITS_PATH
            except (ImportError, AttributeError):
                self.state_file = os.path.join("data", "web_search_credits.json")
        self._credits_today = 0.0
        self._current_date = ""
        # In-flight reservations not yet settled (2026-09-12) — see
        # `reserve()`. Always zero outside an active reserve/spend/settle
        # cycle; never persisted (a crash mid-search should not permanently
        # shrink tomorrow's budget).
        self._reserved_today = 0.0
        self._lock = threading.Lock()
        self._load_state()
        _LIVE_RATE_LIMITERS.add(self)

    def _load_state(self) -> None:
        """Load credit state from disk."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    self._credits_today = state.get("credits_today", 0.0)
                    self._current_date = state.get("date", "")
        except Exception as e:
            log.debug(f"[WebSearch] Failed to load rate limit state: {e}")

    def _save_state(self) -> None:
        """Persist credit state to disk."""
        try:
            from utils.safe_json import atomic_write_json
            atomic_write_json(self.state_file, {
                "credits_today": self._credits_today,
                "date": self._current_date
            })
        except Exception as e:
            log.debug(f"[WebSearch] Failed to save rate limit state: {e}")

    def _check_date_reset_locked(self) -> None:
        """Reset credits (and any outstanding reservation hold) on a new
        day. Caller must already hold ``self._lock``."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._current_date:
            self._credits_today = 0.0
            self._reserved_today = 0.0
            self._current_date = today
            self._save_state()

    def _check_date_reset(self) -> None:
        """Reset credits if we're on a new day."""
        with self._lock:
            self._check_date_reset_locked()

    def can_search(self, estimated_credits: float = 1.0) -> bool:
        """Check if we have budget for a search, net of any in-flight
        (not yet settled) reservations."""
        with self._lock:
            self._check_date_reset_locked()
            return (self._credits_today + self._reserved_today + estimated_credits) <= self.daily_limit

    def record_usage(self, credits: float) -> None:
        """Record credit usage directly (callers that never reserved)."""
        with self._lock:
            self._check_date_reset_locked()
            self._credits_today += credits
            self._save_state()
        log.debug(f"[WebSearch] Credits used: {credits}, total today: {self._credits_today}/{self.daily_limit}")

    def get_remaining_credits(self) -> float:
        """Get remaining daily credits, net of any in-flight reservations."""
        with self._lock:
            self._check_date_reset_locked()
            return max(0.0, self.daily_limit - self._credits_today - self._reserved_today)

    def reserve(self, amount: float) -> Optional[SearchReservation]:
        """Provisionally hold ``amount`` credits against today's budget.

        Returns None when the budget (net of every other outstanding
        reservation) can't afford it. On success the caller owns the
        returned reservation for the lifetime of one search: it must call
        ``.spend(cost)`` before each billable provider call and ``.settle()``
        exactly once, in a ``finally``, no matter how the search ends.
        """
        with self._lock:
            self._check_date_reset_locked()
            if self._credits_today + self._reserved_today + amount > self.daily_limit:
                return None
            self._reserved_today += amount
            return SearchReservation(limiter=self, amount=amount, date=self._current_date)

    def _reservation_spend(self, reservation: SearchReservation, cost: float) -> bool:
        """Commit ``cost`` into ``reservation.day_used``/``reservation.used``,
        extending the hold (against today's remaining budget) when it
        doesn't already fit.

        Cross-midnight dispatch (2026-09-12): a reservation whose ``date``
        predates today has already had its hold zeroed by the date reset,
        and the old-day usage it carries belongs to a day this limiter no
        longer tracks. Re-book it on today's budget with an empty hold
        first — the extension below then competes for exactly this
        dispatch's cost, like any other reservation today, instead of the
        charge silently landing nowhere.
        """
        with self._lock:
            if reservation._settled:
                return False
            self._check_date_reset_locked()
            if reservation.date != self._current_date:
                reservation.date = self._current_date
                reservation.amount = 0.0
                reservation.day_used = 0.0
            if reservation.day_used + cost <= reservation.amount:
                reservation.day_used += cost
                reservation.used += cost
                return True
            needed = (reservation.day_used + cost) - reservation.amount
            if self._credits_today + self._reserved_today + needed <= self.daily_limit:
                self._reserved_today += needed
                reservation.amount += needed
                reservation.day_used += cost
                reservation.used += cost
                return True
            return False

    def _reservation_settle(self, reservation: SearchReservation) -> None:
        """Fold ``reservation.day_used`` (the part of ``used`` charged
        against ``reservation.date``, today's dispatch day) into today's
        real counter and release the rest of the hold — idempotent, and a
        no-op for a reservation whose day has rolled over since its last
        spend: the date reset already zeroed its hold, and usage charged
        against a closed day is dropped with that day, never moved onto the
        new one."""
        with self._lock:
            if reservation._settled:
                return
            reservation._settled = True
            self._check_date_reset_locked()
            if reservation.date == self._current_date:
                self._credits_today += reservation.day_used
                self._reserved_today = max(0.0, self._reserved_today - reservation.amount)
                self._save_state()

    def estimate_credits(self, depth: WebSearchDepth, num_extracts: int = 0) -> float:
        """Estimate credits for a search operation."""
        base_costs = {
            WebSearchDepth.QUICK: 1.0,
            WebSearchDepth.STANDARD: 2.0,
            WebSearchDepth.DEEP: 3.0,
        }
        base = base_costs.get(depth, 1.0)
        # Each extract costs ~1 credit
        return min(base + num_extracts, self.per_query_limit)


# 2026-09-05 13:07 daemon_debug.log incident: the enhanced-path gatherer searched
# "Were any UK or US politicians charged with crimes this week? What did the police
# or courts announce?" at QUICK depth (1 credit, cached at 13:07:27), and 11 seconds
# later the agentic loop's Round-1 seeded search re-ran the IDENTICAL string at
# STANDARD depth (2 more credits, "Credits used: 2.0, total today: 3.0/100") because
# WebSearchCache._generate_cache_key keys on (query, depth) and the exact-key lookup
# missed at the different depth. Within one turn, a same-string re-search at a
# different depth is waste, not a quality upgrade.
WEB_SEARCH_SAME_QUERY_REUSE_S = int(os.getenv("WEB_SEARCH_SAME_QUERY_REUSE_S", "180"))


class WebSearchCache:
    """
    ChromaDB-backed cache for web search results.

    Uses an exact normalized query + search-depth key. This intentionally avoids
    returning stale or mismatched facts for merely similar current-events queries.
    """

    COLLECTION_NAME = "web_search_cache"
    TTL_HOURS = 72

    def __init__(
        self,
        chroma_store: Optional[Any] = None,
        ttl_hours: Optional[float] = None,
    ):
        self._store = chroma_store
        self._collection = None
        self._initialized = False
        # In-process same-query (across depths) reuse map — see
        # WEB_SEARCH_SAME_QUERY_REUSE_S above for the incident this guards against.
        self._recent_by_query: Dict[str, Tuple[float, WebSearchResult]] = {}
        self._recent_max = 64
        if ttl_hours is None:
            try:
                from config.app_config import WEB_SEARCH_CACHE_TTL_HOURS
                ttl_hours = WEB_SEARCH_CACHE_TTL_HOURS
            except (ImportError, AttributeError):
                ttl_hours = self.TTL_HOURS
        self.ttl_hours = max(0.0, float(ttl_hours))

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of ChromaDB collection."""
        if self._initialized:
            return self._collection is not None

        self._initialized = True

        if self._store is None:
            try:
                from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
                self._store = MultiCollectionChromaStore()
            except Exception as e:
                log.warning(f"[WebSearchCache] Failed to initialize store: {e}")
                return False

        try:
            # Try to get or create collection
            if hasattr(self._store, 'client'):
                self._collection = self._store.client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
            return self._collection is not None
        except Exception as e:
            log.warning(f"[WebSearchCache] Failed to create collection: {e}")
            return False

    def _generate_cache_key(self, query: str, depth: WebSearchDepth) -> str:
        """Generate a deterministic cache key."""
        content = f"{query.lower().strip()}:{depth.value}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def _norm(query: str) -> str:
        """Normalize a query string for the in-process same-query reuse map."""
        return (query or "").lower().strip()

    def _get_same_query_recent(self, query: str, depth: WebSearchDepth) -> Optional[WebSearchResult]:
        """
        Same-normalized-query reuse across search depths within one turn.

        See WEB_SEARCH_SAME_QUERY_REUSE_S above for the incident this closes:
        a same-string re-search at a different depth within the reuse window
        is waste, not a quality upgrade. Never serves across different
        normalized queries; expired entries are dropped from the map.
        """
        if WEB_SEARCH_SAME_QUERY_REUSE_S <= 0:
            return None

        norm = self._norm(query)
        entry = self._recent_by_query.get(norm)
        if entry is None:
            return None

        ts, stored = entry
        age = time.time() - ts
        if age > WEB_SEARCH_SAME_QUERY_REUSE_S:
            self._recent_by_query.pop(norm, None)
            return None

        log.debug(
            f"[WebSearchCache] Same-query reuse within {age:.0f}s "
            f"(cached depth {stored.search_depth.value} → requested {depth.value})"
        )
        return replace(stored, from_cache=True)

    def get(self, query: str, depth: WebSearchDepth) -> Optional[WebSearchResult]:
        """Retrieve cached result if available and not expired."""
        if self._ensure_initialized():
            try:
                cache_key = self._generate_cache_key(query, depth)

                # Query by ID first (exact match)
                result = self._collection.get(ids=[cache_key], include=["metadatas", "documents"])

                if result and result.get("ids") and len(result["ids"]) > 0:
                    metadata = result["metadatas"][0] if result.get("metadatas") else {}
                    cached_time = metadata.get("timestamp", 0)

                    # Check TTL
                    if time.time() - cached_time > (self.ttl_hours * 3600):
                        log.debug(f"[WebSearchCache] Cache expired for query: {query[:50]}...")
                        return None

                    # Reconstruct result from cache
                    pages_json = metadata.get("pages_json", "[]")
                    pages_data = json.loads(pages_json)
                    pages = [WebPage(**p) for p in pages_data]

                    return WebSearchResult(
                        query=query,
                        pages=pages,
                        total_credits_used=0,  # No credits used for cache hit
                        search_depth=depth,
                        from_cache=True,
                        timestamp=cached_time
                    )
            except Exception as e:
                log.debug(f"[WebSearchCache] Cache lookup failed: {e}")

        # Exact (query, depth) miss (or store unavailable) — fall back to the
        # in-process same-query-different-depth reuse map.
        return self._get_same_query_recent(query, depth)

    def put(self, result: WebSearchResult) -> None:
        """Cache a search result."""
        if result.has_results:
            norm = self._norm(result.query)
            self._recent_by_query[norm] = (time.time(), result)
            if len(self._recent_by_query) > self._recent_max:
                oldest_key = min(
                    self._recent_by_query,
                    key=lambda k: self._recent_by_query[k][0]
                )
                self._recent_by_query.pop(oldest_key, None)

        if not self._ensure_initialized() or not result.has_results:
            return

        try:
            cache_key = self._generate_cache_key(result.query, result.search_depth)

            # Serialize pages for storage
            pages_data = [
                {
                    "url": p.url,
                    "title": p.title,
                    "content": p.content[:5000] if p.content else "",  # Limit content size
                    "snippet": p.snippet[:500] if p.snippet else "",
                    "score": p.score,
                    "published_date": p.published_date,
                    "source": p.source
                }
                for p in result.pages[:10]  # Max 10 pages
            ]

            # Flatten metadata for ChromaDB
            metadata = {
                "query": result.query[:500],
                "depth": result.search_depth.value,
                "timestamp": result.timestamp,
                "pages_json": json.dumps(pages_data),
                "num_pages": len(result.pages)
            }

            # The document aids inspection; lookups use the deterministic ID.
            self._collection.upsert(
                ids=[cache_key],
                documents=[result.query],
                metadatas=[metadata]
            )

            log.debug(f"[WebSearchCache] Cached result for: {result.query[:50]}...")
        except Exception as e:
            log.debug(f"[WebSearchCache] Failed to cache result: {e}")

    def clear_expired(self) -> int:
        """Remove expired cache entries. Returns count of removed entries."""
        if not self._ensure_initialized():
            return 0

        try:
            # Get all entries
            all_entries = self._collection.get(include=["metadatas"])
            if not all_entries or not all_entries.get("ids"):
                return 0

            expired_ids = []
            current_time = time.time()
            ttl_seconds = self.ttl_hours * 3600

            for i, entry_id in enumerate(all_entries["ids"]):
                metadata = all_entries["metadatas"][i] if all_entries.get("metadatas") else {}
                cached_time = metadata.get("timestamp", 0)
                if current_time - cached_time > ttl_seconds:
                    expired_ids.append(entry_id)

            if expired_ids:
                self._collection.delete(ids=expired_ids)
                log.info(f"[WebSearchCache] Cleared {len(expired_ids)} expired entries")

            return len(expired_ids)
        except Exception as e:
            log.debug(f"[WebSearchCache] Failed to clear expired entries: {e}")
            return 0


class WebSearchManager:
    """
    Main interface for web search operations.

    Features:
    - Tavily Search + Extract APIs
    - Three search depths (QUICK, STANDARD, DEEP)
    - LLM-driven link following for DEEP searches
    - ChromaDB caching with 72-hour TTL
    - Rate limiting with daily credit cap
    - Crisis suppression (no search during HIGH/MEDIUM tone)

    Usage:
        manager = WebSearchManager(api_key="...")
        result = await manager.search("latest AI news", depth=WebSearchDepth.STANDARD)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limiter: Optional[WebSearchRateLimiter] = None,
        cache: Optional[WebSearchCache] = None,
        default_timeout: float = 30.0,
        max_content_chars: int = 10000,
        link_selector_model: str = "gpt-4o-mini",
    ):
        """
        Initialize WebSearchManager.

        Args:
            api_key: Tavily API key (falls back to TAVILY_API_KEY env var)
            rate_limiter: Optional custom rate limiter
            cache: Optional custom cache instance
            default_timeout: Default timeout for search operations
            max_content_chars: Maximum chars per extracted page
            link_selector_model: Model for DEEP mode link selection
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self.rate_limiter = rate_limiter or WebSearchRateLimiter()
        self.cache = cache or WebSearchCache()
        self.default_timeout = default_timeout
        self.max_content_chars = max_content_chars
        self.link_selector_model = link_selector_model

        self._tavily_client = None
        self._initialized = False
        self._api_key_invalid = False  # Set True on 401 to stop retrying

    def _ensure_tavily(self) -> bool:
        """Lazy initialization of Tavily client."""
        if self._api_key_invalid:
            return False

        if self._initialized:
            return self._tavily_client is not None

        self._initialized = True

        if not self.api_key:
            log.warning("[WebSearch] No Tavily API key configured")
            return False

        try:
            from tavily import TavilyClient
            self._tavily_client = TavilyClient(api_key=self.api_key)
            log.debug("[WebSearch] Tavily client initialized")
            return True
        except ImportError:
            log.warning("[WebSearch] tavily-python not installed. Run: pip install tavily-python")
            return False
        except Exception as e:
            log.warning(f"[WebSearch] Failed to initialize Tavily: {e}")
            return False

    def is_available(self) -> bool:
        """Check if web search is available.

        Also honours the LIVE Settings toggle (2026-09-09, audit F04): the
        gatherer, the agentic tool-health block and every direct caller
        consult this one predicate, so a disabled search can't reach the
        provider through a path that never read the config constant.
        """
        if not self.is_enabled():
            return False
        if self._api_key_invalid:
            return False
        return bool(self.api_key) and self._ensure_tavily()

    def budget_exhausted(self) -> bool:
        """True when the remaining daily credit budget cannot fund even the
        cheapest search. Separate from `is_available()` on purpose: the API
        key and the Settings toggle are configuration, this is a budget that
        refills at midnight, and every surface that reports capability has to
        say which one is missing (2026-09-12)."""
        limiter = getattr(self, "rate_limiter", None)
        if limiter is None:
            return False
        try:
            return float(limiter.get_remaining_credits()) < MIN_SEARCH_CREDITS
        except Exception:
            return False

    @staticmethod
    def is_enabled() -> bool:
        """Live value of ``config.app_config.WEB_SEARCH_ENABLED``."""
        try:
            import config.app_config as _cfg  # lazy import: live-config read
            return bool(getattr(_cfg, "WEB_SEARCH_ENABLED", True))
        except ImportError:
            return True

    async def search(
        self,
        query: str,
        depth: WebSearchDepth = WebSearchDepth.STANDARD,
        crisis_level: Optional[str] = None,
        timeout: Optional[float] = None,
        use_cache: bool = True,
        max_results: int = 5,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        localize: bool = True,
    ) -> WebSearchResult:
        """
        Perform a web search with the specified depth.

        Args:
            query: Search query
            depth: Search depth level
            crisis_level: Current tone/crisis level (HIGH/MEDIUM suppresses search)
            timeout: Operation timeout in seconds
            use_cache: Whether to check/update cache
            max_results: Maximum number of results to return
            localize: Apply the _localize_query backstop (disable for
                non-conversational callers whose queries must go verbatim)

        Returns:
            WebSearchResult with pages or error
        """
        # Settings toggle (2026-09-09, audit F04): every public entry point
        # honours the LIVE flag — the gatherer's own check is not enough,
        # the agentic loop and instrument callers reach search() directly.
        # Checked before the cache so a disabled search is never served
        # from cache either; distinct from provider readiness below.
        if not self.is_enabled():
            log.debug("[WebSearch] Suppressed: web search disabled in Settings")
            return WebSearchResult(
                query=query,
                search_depth=depth,
                error=DISABLED_ERROR,
            )

        # Crisis suppression
        if crisis_level and crisis_level.upper() in ("HIGH", "MEDIUM"):
            log.debug(f"[WebSearch] Suppressed during {crisis_level} crisis level")
            return WebSearchResult(
                query=query,
                search_depth=depth,
                error=f"Search suppressed during {crisis_level} crisis level"
            )

        # Localization backstop — before the cache check so the cache keys on
        # the localized text ("weather my area" and "weather Springfield, IL"
        # must not share an entry). localize=False for non-conversational
        # callers (literature oracle) whose queries must reach Tavily verbatim.
        if localize:
            query = self._localize_query(query)

        # Check cache first
        if use_cache:
            cached = self.cache.get(query, depth)
            if cached:
                log.debug(f"[WebSearch] Cache hit for: {query[:50]}...")
                return cached

        # Rate limit check — RESERVE the estimate up front (2026-09-12).
        # can_search() + record_usage()-at-the-end was a check-then-act race:
        # two concurrent callers could each pass can_search() against the
        # same last credit before either recorded usage. reserve() debits
        # the hold immediately so a second concurrent caller sees it.
        estimated_credits = self.rate_limiter.estimate_credits(depth)
        reservation = self.rate_limiter.reserve(estimated_credits)
        if reservation is None:
            remaining = self.rate_limiter.get_remaining_credits()
            log.warning(f"[WebSearch] Daily limit reached. Remaining: {remaining}")
            return WebSearchResult(
                query=query,
                search_depth=depth,
                error=f"Daily credit limit reached. Remaining: {remaining}",
                blocked="budget",
            )

        # Ensure Tavily is ready
        if not self._ensure_tavily():
            reservation.settle()  # never dispatched anything — releases the hold
            return WebSearchResult(
                query=query,
                search_depth=depth,
                error="Tavily client not available"
            )

        timeout = timeout or self.default_timeout

        try:
            result = await asyncio.wait_for(
                self._execute_search(query, depth, max_results,
                                     include_domains=include_domains,
                                     exclude_domains=exclude_domains,
                                     reservation=reservation),
                timeout=timeout
            )

            # Cache successful results
            if use_cache and result.has_results:
                self.cache.put(result)

            return result

        except asyncio.TimeoutError:
            log.warning(f"[WebSearch] Timeout after {timeout}s for: {query[:50]}...")
            return WebSearchResult(
                query=query,
                search_depth=depth,
                error=f"Search timed out after {timeout}s"
            )
        except Exception as e:
            log.error(f"[WebSearch] Search failed: {e}")
            return WebSearchResult(
                query=query,
                search_depth=depth,
                error=str(e)
            )
        finally:
            # Settles on every exit — success, exception, timeout, or this
            # coroutine being cancelled out from under `wait_for` — so a
            # reservation can never leak as a phantom hold against tomorrow.
            reservation.settle()

    async def _execute_search(
        self,
        query: str,
        depth: WebSearchDepth,
        max_results: int,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        reservation: Optional[SearchReservation] = None,
    ) -> WebSearchResult:
        """Execute the actual search based on depth.

        Charge-on-dispatch (2026-09-12): with a ``reservation``, every
        billable provider call is paid for via ``reservation.spend(cost)``
        immediately BEFORE it is dispatched — a call already in flight is
        never refunded even if the search later times out or is cancelled.
        Without a reservation (a direct caller that bypassed ``search()``),
        credit usage is recorded the old way, once, at the end.
        """
        session = WebSearchSession(initial_query=query, depth=depth)

        # Detect if this is a news query - use news topic for better results
        is_news = _is_news_query(query)
        topic = "news" if is_news else "general"
        days = 1 if is_news else None  # Limit to today for news queries

        if is_news:
            log.info(f"[WebSearch] News query detected, using topic='news', days=1")

        # Step 1: Basic search (all depths)
        if reservation is not None and not reservation.spend(1.0):
            remaining = self.rate_limiter.get_remaining_credits()
            return WebSearchResult(
                query=query,
                search_depth=depth,
                error=f"Daily credit limit reached. Remaining: {remaining}",
                blocked="budget",
            )
        try:
            search_pages = await self._tavily_search(
                query, max_results, topic=topic, days=days,
                include_domains=include_domains, exclude_domains=exclude_domains,
            )
        except RetrievalError as e:
            # The base call was already dispatched (charged via the
            # reservation's spend() above, folded in by search()'s finally
            # settle()) — this return skips record_usage exactly like
            # today's invalid-key branch immediately below.
            if self._api_key_invalid:
                return WebSearchResult(
                    query=query,
                    search_depth=depth,
                    error="Tavily API key is invalid",
                )
            return WebSearchResult(
                query=query,
                search_depth=depth,
                error=f"Web search provider failed ({e.reason})",
            )
        session.search_results = search_pages
        session.credits_used += 1.0  # Base search cost

        # If the API key was just flagged invalid, bail out with error
        if self._api_key_invalid:
            return WebSearchResult(
                query=query,
                search_depth=depth,
                error="Tavily API key is invalid"
            )

        # Step 2: Extract content for STANDARD and DEEP. A RetrievalError here
        # keeps the already-fetched search pages — the extract call was
        # already dispatched (and billed) before it failed — and records the
        # failure on `extract_error` instead of `error`, so `has_results`/
        # caching still treat this as a partial success (CGR-008/F2, #110/#111).
        extract_error: Optional[str] = None
        if depth in (WebSearchDepth.STANDARD, WebSearchDepth.DEEP) and search_pages:
            urls_to_extract = [p.url for p in search_pages[:2]]  # Top 2 results
            extract_cost = len(urls_to_extract) * 0.5
            if reservation is None or reservation.spend(extract_cost):
                try:
                    extracted = await self._tavily_extract(urls_to_extract)
                    session.extracted_pages = extracted
                except RetrievalError as e:
                    log.warning("[WebSearch] Extract failed after search succeeded; keeping search pages")
                    extract_error = f"tavily_extract:{e.reason}"
                session.credits_used += extract_cost  # Extract costs (dispatched either way)
            else:
                log.debug("[WebSearch] Skipping extract: reservation budget exhausted")

        # Step 3: LLM-driven link following for DEEP
        if depth == WebSearchDepth.DEEP and search_pages:
            additional_urls = await self._select_links_for_following(
                query,
                session.all_pages
            )
            if reservation is not None and additional_urls:
                # Charge on dispatch per URL: keep only as many as the
                # reservation can afford (with extension), in order, so the
                # extract call below never dispatches an unpaid-for URL.
                affordable_urls = []
                for candidate_url in additional_urls:
                    if not reservation.spend(0.5):
                        break
                    affordable_urls.append(candidate_url)
                additional_urls = affordable_urls
            if additional_urls:
                session.followed_links = additional_urls
                try:
                    more_extracted = await self._tavily_extract(additional_urls)
                    session.extracted_pages.extend(more_extracted)
                except RetrievalError as e:
                    log.warning("[WebSearch] Extract failed after search succeeded; keeping search pages")
                    extract_error = extract_error or f"tavily_extract:{e.reason}"
                session.credits_used += len(additional_urls) * 0.5

        # Record credit usage — only for a direct caller with no reservation;
        # a reservation already tracks `used` and is settled by the caller.
        if reservation is None:
            self.rate_limiter.record_usage(session.credits_used)

        return WebSearchResult(
            query=query,
            pages=session.all_pages,
            total_credits_used=session.credits_used,
            search_depth=depth,
            from_cache=False,
            timestamp=time.time(),
            extract_error=extract_error,
        )

    async def _tavily_search(
        self,
        query: str,
        max_results: int,
        topic: str = "general",
        days: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[WebPage]:
        """
        Execute Tavily search API call.

        Args:
            query: Search query
            max_results: Maximum results to return
            topic: "general" or "news" - news uses Tavily's news-optimized agent
            days: For news topic, limit to N days back (default None = 3 days)
            include_domains: Restrict results to these domains (e.g. ["reddit.com", "stackoverflow.com"])
            exclude_domains: Exclude results from these domains
        """
        if not self._tavily_client:
            raise RetrievalError(source="tavily_search", reason="client_unavailable")

        try:
            # Tavily rejects queries over ~400 chars with 400 Bad Request
            if len(query) > 400:
                log.debug(f"[WebSearch] Truncating long query from {len(query)} to 400 chars")
                query = query[:400]

            # Build search kwargs
            search_kwargs = {
                "query": query,
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
                "topic": topic,
            }

            # days parameter only works with news topic
            if topic == "news" and days is not None:
                search_kwargs["days"] = days

            if include_domains:
                search_kwargs["include_domains"] = list(include_domains)
            if exclude_domains:
                search_kwargs["exclude_domains"] = list(exclude_domains)

            log.debug(f"[WebSearch] Tavily search: topic={topic}, days={days}, query={query[:50]}...")

            # Run in executor since tavily-python is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._tavily_client.search(**search_kwargs)
            )

            pages = []
            for result in response.get("results", []):
                pages.append(WebPage(
                    url=result.get("url", ""),
                    title=result.get("title", ""),
                    content=result.get("content", ""),
                    snippet=result.get("content", "")[:500],
                    score=result.get("score", 0.0),
                    published_date=result.get("published_date"),
                    source="tavily_search"
                ))

            log.debug(f"[WebSearch] Search returned {len(pages)} results")
            return pages

        except Exception as e:
            error_str = str(e).lower()
            if "invalid" in error_str and ("api" in error_str or "key" in error_str) or "401" in error_str:
                log.error(f"[WebSearch] Tavily API key is invalid — disabling web search for this session: {e}")
                self._api_key_invalid = True
                reason = "invalid_api_key"
            else:
                log.error(f"[WebSearch] Tavily search failed: {e}")
                reason = type(e).__name__
            raise RetrievalError(source="tavily_search", reason=reason) from e

    # Direct local fetch below this many extracted chars is a shell/failure —
    # fall through to Tavily extract.
    _DIRECT_FETCH_MIN_CHARS = 400
    _DIRECT_FETCH_MAX_BYTES = 2_000_000
    _DIRECT_FETCH_MAX_REDIRECTS = 5

    _DIRECT_FETCH_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) "
            "Gecko/20100101 Firefox/130.0"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "application/json;q=0.8,*/*;q=0.7"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    async def fetch_url_content(self, url: str) -> List[WebPage]:
        """Layered single-URL fetch: local direct fetch + extraction first,
        Tavily extract as fallback.

        The direct layer handles JS-rendered SPA pages via embedded-JSON
        salvage (utils/page_extract — chatgpt.com/share links returned
        "blank page" through Tavily, 2026-08-29) and costs no API credits.
        Env WEB_FETCH_DIRECT_ENABLED (default on) disables the local layer.

        The Tavily fallback is billed (2026-09-12): it reserves 0.5 credits
        before dispatching and settles in a `finally`, same as `search()`.
        A manager built without a `rate_limiter` (some direct-fetch-only
        test doubles) skips budgeting entirely rather than erroring. Both
        the reservation AND the actual `spend()` are checked before the
        billed provider call ever dispatches (adversarial-review follow-up
        finding 3) — a reservation can still be refused at spend time (a
        cross-midnight re-book that can no longer afford it), and the old
        code dispatched Tavily regardless because it never looked at
        `spend()`'s return value. A refusal at either point returns
        whatever the free direct layer already produced, or a
        ``FetchedPages(blocked="budget")`` when it produced nothing —
        callers can then tell "genuinely nothing there" from "budget
        stopped us from checking".
        """
        # This check applies even when direct fetching is disabled: otherwise a
        # private URL could still be forwarded to a third-party extractor.
        _validate_fetch_url_syntax(url)
        direct: List[WebPage] = []
        if os.getenv("WEB_FETCH_DIRECT_ENABLED", "1").lower() not in ("0", "false"):
            direct = await self._direct_fetch(url)
            if direct and len(direct[0].content) >= self._DIRECT_FETCH_MIN_CHARS:
                return direct

        limiter = getattr(self, "rate_limiter", None)
        reservation = limiter.reserve(0.5) if limiter is not None else None
        if limiter is not None and reservation is None:
            # Unaffordable: skip the billed Tavily fallback, keep whatever
            # the free direct layer already produced (possibly nothing).
            return direct if direct else FetchedPages(blocked="budget")
        tavily: List[WebPage] = []
        extract_failed_reason: Optional[str] = None
        try:
            if reservation is not None and not reservation.spend(0.5):
                # Refused at spend time: never dispatch the billed call.
                return direct if direct else FetchedPages(blocked="budget")
            try:
                tavily = await self._tavily_extract([url])
            except RetrievalError as e:
                log.warning("[WebSearch] Extract failed fetching URL content")
                extract_failed_reason = f"tavily_extract:{e.reason}"
        finally:
            if reservation is not None:
                reservation.settle()
        if extract_failed_reason is not None:
            # Keep whatever the free direct layer already produced; otherwise
            # a typed failed outcome (still falsy, still no `blocked`) instead
            # of a bare `[]` (CGR-008/F2, #111).
            return direct if direct else OutcomeList.failed(extract_failed_reason)
        if tavily and (tavily[0].content or "").strip():
            # Prefer whichever layer extracted more actual content.
            if direct and len(direct[0].content) > len(tavily[0].content):
                return direct
            return tavily
        return direct

    async def _direct_fetch(self, url: str) -> List[WebPage]:
        """Fetch a public URL locally and extract its content.

        Every redirect target is validated before it is requested. Unsafe URLs
        raise ``UnsafeFetchURLError`` so callers cannot silently fall through to
        another fetch backend; ordinary network/extraction failures return [].
        """
        try:
            import httpx  # lazy import: startup cost
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=15.0,
                headers=self._DIRECT_FETCH_HEADERS,
                trust_env=False,
            ) as client:
                current_url = url
                body_text = None
                ctype = ""
                for redirect_count in range(self._DIRECT_FETCH_MAX_REDIRECTS + 1):
                    await _validate_fetch_url_dns(current_url)
                    async with client.stream("GET", current_url) as resp:
                        if resp.status_code in {301, 302, 303, 307, 308}:
                            location = resp.headers.get("location")
                            if not location:
                                return []
                            if redirect_count >= self._DIRECT_FETCH_MAX_REDIRECTS:
                                log.debug(f"[WebSearch] Too many redirects for {url}")
                                return []
                            current_url = urllib.parse.urljoin(current_url, location)
                            _validate_fetch_url_syntax(current_url)
                            continue
                        if resp.status_code >= 400:
                            log.debug(f"[WebSearch] Direct fetch HTTP {resp.status_code} for {url}")
                            return []
                        content_length = resp.headers.get("content-length")
                        if content_length:
                            try:
                                if int(content_length) > self._DIRECT_FETCH_MAX_BYTES:
                                    log.debug(f"[WebSearch] Direct fetch too large for {url}")
                                    return []
                            except ValueError:
                                pass
                        ctype = (resp.headers.get("content-type") or "").lower()
                        if not ("json" in ctype or "html" in ctype or "xml" in ctype
                                or ctype.startswith("text/") or not ctype):
                            # Binary (pdf, images, …) — let Tavily extract handle it.
                            return []
                        # Audit F11 (2026-08-31): stream with a byte cap — the
                        # header check alone let a chunked response (no
                        # Content-Length) buffer an unbounded body into memory
                        # before the old post-hoc text slice ran.
                        _chunks = []
                        _received = 0
                        async for _chunk in resp.aiter_bytes():
                            _chunks.append(_chunk)
                            _received += len(_chunk)
                            if _received >= self._DIRECT_FETCH_MAX_BYTES:
                                break
                        _raw = b"".join(_chunks)[:self._DIRECT_FETCH_MAX_BYTES]
                        try:
                            body_text = _raw.decode(
                                resp.charset_encoding or "utf-8", errors="replace")
                        except LookupError:
                            body_text = _raw.decode("utf-8", errors="replace")
                        break
            if body_text is None:
                return []
            title = current_url
            if "json" in ctype:
                text = body_text
            else:
                from utils.page_extract import extract_page_text
                title_extracted, text, method = extract_page_text(
                    body_text, current_url
                )
                title = title_extracted or current_url
                log.info(
                    f"[WebSearch] Direct fetch {url}: method={method}, "
                    f"{len(text)} chars"
                )
            content = (text or "")[: self.max_content_chars]
            if not content.strip():
                return []
            return [WebPage(
                url=current_url,
                title=title,
                content=content,
                snippet=content[:500],
                source="direct_fetch",
            )]
        except UnsafeFetchURLError:
            raise
        except Exception as e:
            log.debug(f"[WebSearch] Direct fetch failed for {url}: {e}")
            return []

    async def _tavily_extract(self, urls: List[str]) -> List[WebPage]:
        """Extract full content from URLs using Tavily Extract API."""
        if not urls:
            return []
        if not self._tavily_client:
            raise RetrievalError(source="tavily_extract", reason="client_unavailable")

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._tavily_client.extract(urls=urls)
            )

            pages = []
            for result in response.get("results", []):
                raw_content = result.get("raw_content", "")
                # Apply head truncation (v1 simple approach)
                content = raw_content[:self.max_content_chars] if raw_content else ""

                pages.append(WebPage(
                    url=result.get("url", ""),
                    title=result.get("title", "") or result.get("url", ""),
                    content=content,
                    snippet=content[:500] if content else "",
                    source="tavily_extract"
                ))

            log.debug(f"[WebSearch] Extract returned {len(pages)} pages")
            return pages

        except Exception as e:
            error_str = str(e).lower()
            if "invalid" in error_str and ("api" in error_str or "key" in error_str) or "401" in error_str:
                log.error(f"[WebSearch] Tavily API key is invalid — disabling web search for this session: {e}")
                self._api_key_invalid = True
                reason = "invalid_api_key"
            else:
                log.error(f"[WebSearch] Tavily extract failed: {e}")
                reason = type(e).__name__
            raise RetrievalError(source="tavily_extract", reason=reason) from e

    async def _select_links_for_following(
        self,
        query: str,
        current_pages: List[WebPage],
        max_links: int = 2
    ) -> List[str]:
        """
        Use LLM to select additional links to follow for DEEP searches.

        This analyzes current results and identifies URLs that likely contain
        more relevant information worth extracting.
        """
        if not current_pages:
            return []

        try:
            from models.model_manager import ModelManager
            model_manager = ModelManager()

            # Build prompt for link selection
            links_info = []
            for i, page in enumerate(current_pages[:10]):
                links_info.append(f"{i+1}. {page.title}\n   URL: {page.url}\n   Snippet: {page.snippet[:200]}...")

            prompt = f"""Given this search query: "{query}"

And these search results:
{chr(10).join(links_info)}

Which 1-2 URLs would provide the most valuable additional information for answering this query?
Consider: primary sources, official documentation, authoritative references.

Return ONLY the URLs (one per line), nothing else. If none are worth following, return "NONE"."""

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model_manager.generate_response_sync(
                    prompt=prompt,
                    system_prompt="You are a research assistant. Select the most informative URLs.",
                    model=self.link_selector_model,
                    max_tokens=200,
                    temperature=0.0
                )
            )

            if not response or "NONE" in response.upper():
                return []

            # Parse URLs from response
            selected_urls = []
            already_have = {p.url for p in current_pages}

            for line in response.strip().split("\n"):
                line = line.strip()
                if line.startswith("http") and line not in already_have:
                    selected_urls.append(line)
                    if len(selected_urls) >= max_links:
                        break

            log.debug(f"[WebSearch] LLM selected {len(selected_urls)} links to follow")
            return selected_urls

        except Exception as e:
            log.debug(f"[WebSearch] Link selection failed: {e}")
            return []

    # =========================================================================
    # Query Localization
    # =========================================================================

    # Literal deictic-location phrases that must never reach the search engine —
    # they make the ranker pick an arbitrary big-market result (a "my area"
    # weather query once returned DC news to an Illinois user).
    _DEICTIC_LOCATION_SUBS = [
        (re.compile(r"\b(?:in|for|around|near)\s+my\s+(?:area|city|town|location|neighborhood)\b", re.I),
         "in {loc}"),
        (re.compile(r"\bnear\s+me\b", re.I), "in {loc}"),
        (re.compile(r"\bmy\s+(?:area|city|town|location|neighborhood)\b", re.I), "{loc}"),
    ]

    # Query shapes that are meaningless without a place. Kept narrow on purpose:
    # over-matching would staple the user's city onto queries about global topics.
    _LOCATION_DEPENDENT_RE = re.compile(
        r"\b(weather|forecast|heat\s+(?:advisory|warning|index|wave)|"
        r"air\s+quality|uv\s+index|wind\s+chill|excessive\s+heat)\b",
        re.I,
    )
    # "temperature"/"humidity" also mean oven settings, fermentation, hardware —
    # location-dependent only alongside a current-conditions cue ("outside",
    # "right now"), never on their own ("what temperature to bake salmon").
    _AMBIGUOUS_WEATHER_RE = re.compile(r"\b(temperature|humidity)\b", re.I)
    _CURRENT_CONDITIONS_RE = re.compile(
        r"\b(outside|outdoors|today|tonight|tomorrow|right\s+now|currently|"
        r"this\s+(?:week|weekend|morning|afternoon|evening))\b",
        re.I,
    )
    # "in <word>"/"at <word>" marks a query that already targets a place.
    # Case-insensitive ("weather in tokyo" is typical chat), with function
    # words excluded so "weather in the morning" doesn't read as a place.
    _NAMES_A_PLACE_RE = re.compile(
        r"\b(?:in|at)\s+(?!my\b|the\b|a\b|an\b|this\b|that\b|what\b|which\b)[\w']",
        re.I,
    )

    def _get_user_location(self) -> Optional[str]:
        try:
            from utils.location_resolver import get_user_location
            return get_user_location()
        except Exception as e:
            log.debug(f"[WebSearch] Location resolution failed: {e}")
            return None

    def _get_user_institution(self) -> Optional[str]:
        try:
            from utils.institution_resolver import get_user_institution
            return get_user_institution()
        except Exception as e:
            log.debug(f"[WebSearch] Institution resolution failed: {e}")
            return None

    def _localize_query(self, query: str) -> str:
        """Deterministic backstop behind the LLM prompts: substitute literal
        "my area"/"near me" with the user's location, and append the location
        to weather-type queries that name no place at all. Runs on every query
        entering search(), so LLM-generated sub-queries are covered too."""
        loc = self._get_user_location()
        if not loc or not query:
            return query

        original = query
        for pattern, template in self._DEICTIC_LOCATION_SUBS:
            query = pattern.sub(template.format(loc=loc), query)

        wants_location = bool(self._LOCATION_DEPENDENT_RE.search(query)) or (
            self._AMBIGUOUS_WEATHER_RE.search(query)
            and self._CURRENT_CONDITIONS_RE.search(query)
        )
        if wants_location:
            city = loc.split(",")[0].strip().lower()
            names_a_place = (
                city in query.lower()
                # "in Chicago", "weather in tokyo" — already targets somewhere
                or self._NAMES_A_PLACE_RE.search(query)
            )
            if not names_a_place:
                query = f"{query} {loc}"

        if query != original:
            log.info(f"[WebSearch] Localized query: '{original}' -> '{query}'")
        return query

    # =========================================================================
    # Query Decomposition and Multi-Search
    # =========================================================================

    async def decompose_query(
        self,
        query: str,
        max_sub_queries: int = 4,
        min_confidence: float = 0.6
    ) -> QueryDecomposition:
        """
        Analyze a query and decompose into sub-queries if beneficial.

        Uses LLM to detect multi-entity or multi-facet queries that would
        benefit from parallel searches.

        Args:
            query: Original user query
            max_sub_queries: Maximum number of sub-queries to generate
            min_confidence: Minimum confidence to trigger decomposition

        Returns:
            QueryDecomposition with sub-queries if applicable
        """
        if not query or len(query) < 20:
            return QueryDecomposition(
                original_query=query,
                should_decompose=False,
                reason="Query too short for decomposition"
            )

        # Check credit budget - don't decompose if we can't afford it
        remaining_credits = self.rate_limiter.get_remaining_credits()
        if remaining_credits < 2:
            return QueryDecomposition(
                original_query=query,
                should_decompose=False,
                reason="Insufficient credits for multi-search"
            )

        try:
            from models.model_manager import ModelManager
            model_manager = ModelManager()

            location_block = ""
            user_location = self._get_user_location()
            if user_location:
                location_block = (
                    f"\nUser location: {user_location}\n"
                    f"If the query is location-dependent (weather, temperature, forecasts, "
                    f"local news, nearby places), include \"{user_location}\" explicitly in every "
                    f"relevant sub-query. Never emit \"my area\", \"near me\", \"local\", or "
                    f"\"nearby\" as literal search text.\n"
                    f"Location is ONLY for physical-surroundings queries: never add it to "
                    f"sub-queries about the user's accounts, logins, school/college, employer, "
                    f"bank, or any service they use — a nearby institution is not theirs unless "
                    f"they named it. If unnamed, keep those sub-queries place-free.\n"
                )

            institution_block = ""
            user_institution = self._get_user_institution()
            if user_institution:
                # 2026-09-12: only inject the school when the QUERY itself
                # gives a reason to name it — mirrors the same gate in
                # utils.web_search_trigger._build_llm_trigger_prompt. The
                # post-parse backstop below still scrubs any slip-through.
                from utils.institution_resolver import query_justifies_institution
                if query_justifies_institution(query, user_institution):
                    institution_block = (
                        f"\nUser's school: {user_institution}\n"
                        f"If the query is about the user's OWN school logistics (drop/withdrawal "
                        f"deadlines, registration, registrar, tuition, academic calendar), use "
                        f"\"{user_institution}\" in those sub-queries instead of generic "
                        f"\"college\"/\"school\". Never apply it when the user names a different "
                        f"school, and never for general coursework/concept questions.\n"
                    )

            prompt = f"""Analyze this search query and determine if it should be split into multiple focused sub-queries for better search results.

Query: "{query}"
{location_block}{institution_block}
Criteria for splitting:
1. Multiple distinct entities (e.g., "Tesla vs Rivian" → search each separately)
2. Multiple facets/aspects (e.g., "iPhone 16 price and reviews" → search price, search reviews)
3. Comparison queries (e.g., "Python vs JavaScript for web dev" → search each language)
4. Time-spanning queries (e.g., "AI progress 2024 to 2025" → search each year)

DO NOT split if:
- Query is already focused on one topic
- Splitting would lose important context
- Query is a simple factual question

Respond in this exact format:
SHOULD_SPLIT: yes/no
CONFIDENCE: 0.0-1.0
REASON: brief explanation
SUB_QUERIES:
- first sub-query (if splitting)
- second sub-query (if splitting)
- etc (max {max_sub_queries})

If not splitting, leave SUB_QUERIES empty."""

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model_manager.generate(
                    prompt=prompt,
                    system_prompt="You are a search query analyst. Be concise and precise.",
                    model_name=self.link_selector_model,
                    max_tokens=300,
                    temperature=0.0
                )
            )

            if not response:
                return QueryDecomposition(
                    original_query=query,
                    should_decompose=False,
                    reason="LLM response empty"
                )

            # Parse response
            lines = response.strip().split("\n")
            should_split = False
            confidence = 0.0
            reason = ""
            sub_queries = []

            in_sub_queries = False
            for line in lines:
                line = line.strip()
                if line.startswith("SHOULD_SPLIT:"):
                    should_split = "yes" in line.lower()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":")[1].strip())
                    except (ValueError, IndexError):
                        confidence = 0.5
                elif line.startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip() if ":" in line else ""
                elif line.startswith("SUB_QUERIES:"):
                    in_sub_queries = True
                elif in_sub_queries and line.startswith("-"):
                    sub_query = line[1:].strip()
                    if sub_query and len(sub_queries) < max_sub_queries:
                        sub_queries.append(sub_query)

            # Apply confidence threshold
            if confidence < min_confidence:
                should_split = False
                reason = f"Confidence {confidence:.2f} below threshold {min_confidence}"

            # Backstop: strip an unjustified location (institution/account
            # sub-queries must stay place-free — 2026-07-08 wrong-college
            # incident) and an unjustified institution, then apply the
            # deterministic institution backstop for academic-logistics
            # sub-queries the LLM left generic (2026-08-27: "class
            # withdrawal deadline 2026" et al. burned 6 credits on generic
            # pages while the profile knew the school). Same policy the
            # trigger classifier uses (BC-58) — one scoping fix reaches both
            # producers instead of two independently-drifting blocks.
            if sub_queries:
                from utils.institution_resolver import scope_identity_terms
                sub_queries = scope_identity_terms(
                    sub_queries, query, user_location, user_institution
                )

            # Validate we have enough sub-queries
            if should_split and len(sub_queries) < 2:
                should_split = False
                reason = "Not enough sub-queries generated"

            log.info(
                f"[WebSearch] Query decomposition: split={should_split}, "
                f"confidence={confidence:.2f}, sub_queries={len(sub_queries)}"
            )

            return QueryDecomposition(
                original_query=query,
                should_decompose=should_split,
                sub_queries=sub_queries,
                confidence=confidence,
                reason=reason
            )

        except Exception as e:
            log.warning(f"[WebSearch] Query decomposition failed: {e}")
            return QueryDecomposition(
                original_query=query,
                should_decompose=False,
                reason=f"Decomposition error: {str(e)}"
            )

    # Semantic anchors for broad news detection (lazy-initialized)
    _broad_news_anchors = None
    _not_news_anchors = None

    _BROAD_NEWS_PHRASES = [
        "what is happening in the news today",
        "give me a news briefing current events update",
        "catch me up on what's going on in the world",
        "latest headlines breaking news today",
        "what did I miss in the news recently",
    ]
    _NOT_NEWS_PHRASES = [
        "tell me about my dog my family my notes",
        "explain how a concept works scientific theory",
        "help me write code fix this bug programming",
        "how are you feeling emotions conversation casual",
    ]

    @classmethod
    def _get_news_anchors(cls):
        """Lazily embed broad-news anchor phrases."""
        if cls._broad_news_anchors is not None:
            return cls._broad_news_anchors, cls._not_news_anchors
        try:
            from models.model_manager import ModelManager
            embedder = ModelManager._get_cached_embedder()
            if embedder is None:
                return None, None
            import numpy as np
            cls._broad_news_anchors = embedder.encode(
                cls._BROAD_NEWS_PHRASES, convert_to_numpy=True, normalize_embeddings=True
            )
            cls._not_news_anchors = embedder.encode(
                cls._NOT_NEWS_PHRASES, convert_to_numpy=True, normalize_embeddings=True
            )
            return cls._broad_news_anchors, cls._not_news_anchors
        except Exception:
            return None, None

    @staticmethod
    def _is_broad_news_query(query: str) -> bool:
        """Detect broad news/current-events briefing requests (not topic-specific)."""
        q = query.lower().strip()
        positives = [
            "what's going on", "what's happening", "whats going on", "whats happening",
            "catch me up", "current events", "latest headlines", "what did i miss",
            "news update", "fill me in", "what's new in the world", "what have i missed",
            "up on the news", "what's the news", "what's in the news",
        ]
        has_positive = any(p in q for p in positives)
        # Also catch: "news" + question mark + no specific topic
        if not has_positive:
            if "news" in q and "?" in query:
                has_positive = True
            else:
                # No keyword match — try semantic similarity as fallback
                has_positive = WebSearchManager._semantic_broad_news_check(query)

        if not has_positive:
            return False

        # Negative: specific topic modifier → focused search, not briefing
        specifics = ["news about", "latest on ", "update on ", "happened with",
                     "what about ", "regarding ", "tell me about"]
        if any(s in q for s in specifics):
            return False
        # Named entities (capitalized mid-sentence words) → specific query
        # Skip first word of each sentence (always capitalized, not an entity signal)
        _sentences = _re.split(r'[.!?]+\s*', query)
        _NON_ENTITY_MID = {
            "i", "i'll", "i'm", "i've", "i'd",
        }
        has_named_entity = False
        for _sent in _sentences:
            _words = _sent.split()
            for w in _words[1:]:  # skip sentence-initial word
                if len(w) > 1 and w[0].isupper() and w.lower().rstrip("?.,!") not in _NON_ENTITY_MID:
                    has_named_entity = True
                    break
            if has_named_entity:
                break
        if has_named_entity:
            return False
        return True

    @classmethod
    def _semantic_broad_news_check(cls, query: str, threshold: float = 0.45) -> bool:
        """Semantic similarity fallback for broad news detection."""
        news_embs, not_news_embs = cls._get_news_anchors()
        if news_embs is None:
            return False
        try:
            from models.model_manager import ModelManager
            embedder = ModelManager._get_cached_embedder()
            if embedder is None:
                return False
            import numpy as np
            q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            max_news_sim = float(np.max(news_embs @ q_emb))
            max_not_news_sim = float(np.max(not_news_embs @ q_emb))
            margin = max_news_sim - max_not_news_sim
            if max_news_sim > threshold and margin > 0.05:
                log.debug(
                    f"[WebSearch] Semantic broad news match: sim={max_news_sim:.3f}, "
                    f"margin={margin:.3f}"
                )
                return True
        except Exception:
            pass
        return False

    async def _decompose_news_query(self, query: str, user_interests: List[str] = None) -> List[str]:
        """Generate 3-5 targeted sub-queries for a news briefing."""
        date_str = datetime.now().strftime("%B %Y")
        interest_hint = ""
        if user_interests:
            interest_hint = (
                f"\nThe user has known interests in: {', '.join(user_interests[:5])}. "
                "Bias one query toward these if relevant."
            )
        prompt = (
            f"Generate 3-5 specific web search queries for a concise news briefing as of {date_str}.\n"
            "Prefer authoritative sources:\n"
            "- geopolitics/world: Reuters, AP, BBC\n"
            "- economy/markets: Reuters, CNBC, Financial Times\n"
            "- science/space/tech: official agency sources, plus Reuters/AP\n"
            "- wildcard: major developing story from reputable outlets\n"
            f"{interest_hint}\n"
            f"Return one query per line. No numbering. Avoid generic queries like 'current news {date_str}'."
        )
        try:
            from models.model_manager import ModelManager
            mm = ModelManager()
            response = await mm.generate_once(
                prompt=prompt,
                model_name=self.link_selector_model,
                system_prompt="Generate search queries. One per line. No numbering.",
                max_tokens=150,
                temperature=0.3,
            )
            queries = [
                line.strip() for line in (response or "").strip().split("\n")
                if line.strip() and len(line.strip()) > 10
            ]
            if queries:
                log.info(f"[WebSearch] News decomposition: {len(queries)} facet queries generated")
            return queries[:5]
        except Exception as e:
            log.warning(f"[WebSearch] News decomposition failed: {e}")
            return []

    @staticmethod
    def dedupe_search_terms(terms: List[str]) -> List[str]:
        """Collapse near-duplicate search terms by word overlap ratio."""
        if len(terms) <= 1:
            return terms
        # Remove common filler words that inflate overlap
        _FILLER = {"the", "a", "an", "in", "on", "of", "for", "and", "to", "is", "are", "was"}
        def _content_words(text):
            return {w for w in text.lower().split() if w not in _FILLER and len(w) > 2}

        deduped = [terms[0]]
        for term in terms[1:]:
            words_new = _content_words(term)
            is_dupe = False
            for existing in deduped:
                words_existing = _content_words(existing)
                # Use overlap ratio: shared / min(len_a, len_b)
                # This catches cases where short generic terms overlap heavily
                if not words_new or not words_existing:
                    continue
                overlap = len(words_new & words_existing)
                min_len = min(len(words_new), len(words_existing))
                ratio = overlap / min_len if min_len > 0 else 0
                if ratio > 0.5:
                    is_dupe = True
                    break
            if not is_dupe:
                deduped.append(term)
        return deduped

    async def multi_search(
        self,
        query: str,
        depth: WebSearchDepth = WebSearchDepth.STANDARD,
        crisis_level: Optional[str] = None,
        timeout: Optional[float] = None,
        use_cache: bool = True,
        max_results_per_query: int = 3,
        auto_decompose: bool = True,
        sub_queries: Optional[List[str]] = None
    ) -> MultiSearchResult:
        """
        Perform a multi-query search with automatic decomposition.

        This is the enhanced entry point that:
        1. Analyzes the query for decomposition potential
        2. If beneficial, splits into sub-queries
        3. Executes parallel searches
        4. Merges and deduplicates results

        Args:
            query: User search query
            depth: Search depth level
            crisis_level: Current tone/crisis level
            timeout: Per-search timeout in seconds
            use_cache: Whether to use cache
            max_results_per_query: Max results per sub-query
            auto_decompose: Whether to auto-analyze for decomposition
            sub_queries: Pre-computed sub-queries (skips decomposition analysis)

        Returns:
            MultiSearchResult with merged pages from all sub-queries
        """
        # Settings toggle (2026-09-09, audit F04) — see search().
        if not self.is_enabled():
            log.debug("[WebSearch] Multi-search suppressed: disabled in Settings")
            return MultiSearchResult(
                original_query=query,
                search_depth=depth,
                error=DISABLED_ERROR,
            )

        # Crisis suppression
        if crisis_level and crisis_level.upper() in ("HIGH", "MEDIUM"):
            log.debug(f"[WebSearch] Multi-search suppressed during {crisis_level}")
            return MultiSearchResult(
                original_query=query,
                search_depth=depth,
                error=f"Search suppressed during {crisis_level} crisis level"
            )

        # Pre-computed sub-queries: skip all decomposition analysis
        if sub_queries and len(sub_queries) > 1:
            log.info(f"[WebSearch] Using {len(sub_queries)} pre-computed sub-queries")
            decomposition = QueryDecomposition(
                original_query=query,
                should_decompose=True,
                sub_queries=sub_queries[:4],
                confidence=0.9,
                reason="Pre-computed sub-queries from caller",
            )
            # Fall through to parallel execution below

        # Broad news detection: bypass general decomposition, use facet queries
        elif self._is_broad_news_query(query):
            news_queries = await self._decompose_news_query(query)
            if news_queries:
                log.info(f"[WebSearch] Broad news detected, using {len(news_queries)} facet queries")
                decomposition = QueryDecomposition(
                    original_query=query,
                    should_decompose=True,
                    sub_queries=news_queries,
                    confidence=0.9,
                    reason="Broad news briefing request",
                )
                # Fall through to parallel execution below
            else:
                decomposition = None
        else:
            # Attempt general query decomposition if enabled
            decomposition = None
            if auto_decompose:
                decomposition = await self.decompose_query(query)

        # If decomposition not beneficial, fall back to single search
        if not decomposition or not decomposition.should_decompose:
            log.debug("[WebSearch] No decomposition, using single search")
            single_result = await self.search(
                query=query,
                depth=depth,
                crisis_level=crisis_level,
                timeout=timeout,
                use_cache=use_cache,
                max_results=max_results_per_query * 2  # More results for single query
            )
            return MultiSearchResult(
                original_query=query,
                sub_queries=[query],
                pages=single_result.pages,
                total_credits_used=single_result.total_credits_used,
                search_depth=depth,
                from_cache=single_result.from_cache,
                timestamp=single_result.timestamp,
                error=single_result.error,
                decomposition_used=False,
                blocked=single_result.blocked,
            )

        # Execute parallel searches for sub-queries
        log.info(
            f"[WebSearch] Executing multi-search with {len(decomposition.sub_queries)} sub-queries"
        )

        sub_queries = decomposition.sub_queries
        timeout = timeout or self.default_timeout

        # Create search tasks
        search_tasks = [
            self.search(
                query=sub_q,
                depth=depth,
                crisis_level=crisis_level,
                timeout=timeout,
                use_cache=use_cache,
                max_results=max_results_per_query
            )
            for sub_q in sub_queries
        ]

        # Execute in parallel
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*search_tasks, return_exceptions=True),
                timeout=timeout * 2  # Allow extra time for parallel execution
            )
        except asyncio.TimeoutError:
            log.warning("[WebSearch] Multi-search timed out")
            return MultiSearchResult(
                original_query=query,
                sub_queries=sub_queries,
                search_depth=depth,
                error="Multi-search timed out",
                decomposition_used=True
            )

        # Merge and deduplicate results
        all_pages: List[WebPage] = []
        seen_urls: Dict[str, WebPage] = {}
        total_credits = 0.0
        any_from_cache = False
        errors = []
        # Adversarial-review follow-up finding 3: a budget refusal on ONE
        # sub-query used to disappear whenever another sub-query still
        # returned pages (the joined `errors` string below is dropped
        # entirely once `all_pages` is non-empty). Checked BEFORE the
        # `result.error` skip so a refused-but-erroring sub-result is still
        # counted here even though its pages/credits are not merged in.
        budget_blocked = False

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"Sub-query {i+1} error: {str(result)}")
                continue

            if getattr(result, "blocked", None) == "budget":
                budget_blocked = True

            if result.error:
                errors.append(f"Sub-query '{sub_queries[i][:30]}...': {result.error}")
                continue

            total_credits += result.total_credits_used
            if result.from_cache:
                any_from_cache = True

            # Deduplicate by URL, keeping highest-scoring version
            for page in result.pages:
                if page.url in seen_urls:
                    # Keep the one with higher score
                    if page.score > seen_urls[page.url].score:
                        seen_urls[page.url] = page
                else:
                    seen_urls[page.url] = page

        # Sort by score (descending)
        all_pages = sorted(seen_urls.values(), key=lambda p: p.score, reverse=True)

        # Log summary
        log.info(
            f"[WebSearch] Multi-search complete: {len(all_pages)} unique pages, "
            f"{total_credits:.1f} credits, {len(errors)} errors"
        )

        return MultiSearchResult(
            original_query=query,
            sub_queries=sub_queries,
            pages=all_pages,
            total_credits_used=total_credits,
            search_depth=depth,
            from_cache=any_from_cache,
            timestamp=time.time(),
            error="; ".join(errors) if errors and not all_pages else None,
            decomposition_used=True,
            blocked="budget" if budget_blocked else None,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the web search system."""
        return {
            "available": self.is_available(),
            "api_key_configured": bool(self.api_key),
            "api_key_valid": not self._api_key_invalid,
            "remaining_credits": self.rate_limiter.get_remaining_credits(),
            "daily_limit": self.rate_limiter.daily_limit,
            "cache_initialized": self.cache._initialized
        }


# Convenience function for one-shot searches
async def quick_web_search(
    query: str,
    depth: WebSearchDepth = WebSearchDepth.QUICK,
    api_key: Optional[str] = None
) -> WebSearchResult:
    """
    Convenience function for quick web searches.

    Usage:
        result = await quick_web_search("latest news on AI")
    """
    manager = WebSearchManager(api_key=api_key)
    return await manager.search(query, depth=depth)


if __name__ == "__main__":
    # Quick test
    import asyncio

    logging.basicConfig(level=logging.DEBUG)

    async def test():
        manager = WebSearchManager()
        print(f"Status: {manager.get_status()}")

        if manager.is_available():
            result = await manager.search(
                "Python 3.12 new features",
                depth=WebSearchDepth.STANDARD
            )
            print(f"Results: {len(result.pages)} pages")
            print(f"Credits used: {result.total_credits_used}")
            if result.pages:
                print(f"First result: {result.pages[0].title}")
        else:
            print("Web search not available - check TAVILY_API_KEY")

    asyncio.run(test())
