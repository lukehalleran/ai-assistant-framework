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
- Side effects:
  - Network requests to Tavily API
  - ChromaDB cache writes (72-hour TTL)
  - Credit tracking for rate limiting
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


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


class WebSearchRateLimiter:
    """
    Credit-aware rate limiter for Tavily API.

    Tracks daily credit usage and enforces limits to prevent overuse.
    Designed for Tavily free tier (1000 credits/month).
    """

    def __init__(
        self,
        daily_limit: int = 100,
        per_query_limit: int = 5,
        state_file: Optional[str] = None
    ):
        self.daily_limit = daily_limit
        self.per_query_limit = per_query_limit
        self.state_file = state_file or os.path.join(
            os.path.dirname(__file__), "..", "data", "web_search_credits.json"
        )
        self._credits_today = 0.0
        self._current_date = ""
        self._load_state()

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
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump({
                    "credits_today": self._credits_today,
                    "date": self._current_date
                }, f)
        except Exception as e:
            log.debug(f"[WebSearch] Failed to save rate limit state: {e}")

    def _check_date_reset(self) -> None:
        """Reset credits if we're on a new day."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._current_date:
            self._credits_today = 0.0
            self._current_date = today
            self._save_state()

    def can_search(self, estimated_credits: float = 1.0) -> bool:
        """Check if we have budget for a search."""
        self._check_date_reset()
        return (self._credits_today + estimated_credits) <= self.daily_limit

    def record_usage(self, credits: float) -> None:
        """Record credit usage."""
        self._check_date_reset()
        self._credits_today += credits
        self._save_state()
        log.debug(f"[WebSearch] Credits used: {credits}, total today: {self._credits_today}/{self.daily_limit}")

    def get_remaining_credits(self) -> float:
        """Get remaining daily credits."""
        self._check_date_reset()
        return max(0.0, self.daily_limit - self._credits_today)

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


class WebSearchCache:
    """
    ChromaDB-backed cache for web search results.

    Uses semantic similarity for cache hits, with 72-hour TTL.
    """

    COLLECTION_NAME = "web_search_cache"
    TTL_HOURS = 72

    def __init__(self, chroma_store: Optional[Any] = None):
        self._store = chroma_store
        self._collection = None
        self._initialized = False

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

    def get(self, query: str, depth: WebSearchDepth) -> Optional[WebSearchResult]:
        """Retrieve cached result if available and not expired."""
        if not self._ensure_initialized():
            return None

        try:
            cache_key = self._generate_cache_key(query, depth)

            # Query by ID first (exact match)
            result = self._collection.get(ids=[cache_key], include=["metadatas", "documents"])

            if result and result.get("ids") and len(result["ids"]) > 0:
                metadata = result["metadatas"][0] if result.get("metadatas") else {}
                cached_time = metadata.get("timestamp", 0)

                # Check TTL
                if time.time() - cached_time > (self.TTL_HOURS * 3600):
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

        return None

    def put(self, result: WebSearchResult) -> None:
        """Cache a search result."""
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

            # Use query as document for semantic matching
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
            ttl_seconds = self.TTL_HOURS * 3600

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

    def _ensure_tavily(self) -> bool:
        """Lazy initialization of Tavily client."""
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
        """Check if web search is available."""
        return bool(self.api_key) and self._ensure_tavily()

    async def search(
        self,
        query: str,
        depth: WebSearchDepth = WebSearchDepth.STANDARD,
        crisis_level: Optional[str] = None,
        timeout: Optional[float] = None,
        use_cache: bool = True,
        max_results: int = 5,
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

        Returns:
            WebSearchResult with pages or error
        """
        # Crisis suppression
        if crisis_level and crisis_level.upper() in ("HIGH", "MEDIUM"):
            log.debug(f"[WebSearch] Suppressed during {crisis_level} crisis level")
            return WebSearchResult(
                query=query,
                search_depth=depth,
                error=f"Search suppressed during {crisis_level} crisis level"
            )

        # Check cache first
        if use_cache:
            cached = self.cache.get(query, depth)
            if cached:
                log.debug(f"[WebSearch] Cache hit for: {query[:50]}...")
                return cached

        # Rate limit check
        estimated_credits = self.rate_limiter.estimate_credits(depth)
        if not self.rate_limiter.can_search(estimated_credits):
            remaining = self.rate_limiter.get_remaining_credits()
            log.warning(f"[WebSearch] Daily limit reached. Remaining: {remaining}")
            return WebSearchResult(
                query=query,
                search_depth=depth,
                error=f"Daily credit limit reached. Remaining: {remaining}"
            )

        # Ensure Tavily is ready
        if not self._ensure_tavily():
            return WebSearchResult(
                query=query,
                search_depth=depth,
                error="Tavily client not available"
            )

        timeout = timeout or self.default_timeout

        try:
            result = await asyncio.wait_for(
                self._execute_search(query, depth, max_results),
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

    async def _execute_search(
        self,
        query: str,
        depth: WebSearchDepth,
        max_results: int
    ) -> WebSearchResult:
        """Execute the actual search based on depth."""
        session = WebSearchSession(initial_query=query, depth=depth)

        # Step 1: Basic search (all depths)
        search_pages = await self._tavily_search(query, max_results)
        session.search_results = search_pages
        session.credits_used += 1.0  # Base search cost

        # Step 2: Extract content for STANDARD and DEEP
        if depth in (WebSearchDepth.STANDARD, WebSearchDepth.DEEP) and search_pages:
            urls_to_extract = [p.url for p in search_pages[:2]]  # Top 2 results
            extracted = await self._tavily_extract(urls_to_extract)
            session.extracted_pages = extracted
            session.credits_used += len(urls_to_extract) * 0.5  # Extract costs

        # Step 3: LLM-driven link following for DEEP
        if depth == WebSearchDepth.DEEP and search_pages:
            additional_urls = await self._select_links_for_following(
                query,
                session.all_pages
            )
            if additional_urls:
                session.followed_links = additional_urls
                more_extracted = await self._tavily_extract(additional_urls)
                session.extracted_pages.extend(more_extracted)
                session.credits_used += len(additional_urls) * 0.5

        # Record credit usage
        self.rate_limiter.record_usage(session.credits_used)

        return WebSearchResult(
            query=query,
            pages=session.all_pages,
            total_credits_used=session.credits_used,
            search_depth=depth,
            from_cache=False,
            timestamp=time.time()
        )

    async def _tavily_search(self, query: str, max_results: int) -> List[WebPage]:
        """Execute Tavily search API call."""
        if not self._tavily_client:
            return []

        try:
            # Run in executor since tavily-python is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._tavily_client.search(
                    query=query,
                    max_results=max_results,
                    include_answer=False,
                    include_raw_content=False
                )
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
            log.error(f"[WebSearch] Tavily search failed: {e}")
            return []

    async def _tavily_extract(self, urls: List[str]) -> List[WebPage]:
        """Extract full content from URLs using Tavily Extract API."""
        if not self._tavily_client or not urls:
            return []

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
            log.error(f"[WebSearch] Tavily extract failed: {e}")
            return []

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

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the web search system."""
        return {
            "available": self.is_available(),
            "api_key_configured": bool(self.api_key),
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
