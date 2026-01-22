"""
Wolfram Alpha LLM API integration for computational queries.

Module Contract:
- Purpose: Handle mathematical computations, scientific data, unit conversions,
           and any query requiring numerical processing via Wolfram Alpha LLM API
- Inputs:
  - query(input_text: str) -> WolframResult: Execute computational query
  - is_available() -> bool: Check if Wolfram Alpha is configured
  - format_for_prompt(result: WolframResult) -> str: Format result for LLM context
- Outputs:
  - WolframResult with success status, result text, assumptions, execution time
- Side effects:
  - HTTP requests to Wolfram Alpha API
  - In-memory caching of results
  - Rate limiting tracking
- Error handling:
  - Graceful degradation on API errors
  - Timeout handling
  - Rate limit enforcement

Dependencies:
    - config.app_config (WOLFRAM_* constants)
    - httpx (async HTTP client)
"""

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class WolframResult:
    """Result from Wolfram Alpha query."""
    query: str
    success: bool
    result: str = ""
    input_interpreted: str = ""
    assumptions: List[str] = field(default_factory=list)
    related_queries: List[str] = field(default_factory=list)
    error: Optional[str] = None
    execution_time: float = 0.0
    cached: bool = False


class WolframRateLimiter:
    """Simple token bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens = float(requests_per_minute)
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire a token. Returns False if rate limited."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill

            # Refill tokens based on elapsed time
            self.tokens = min(
                self.requests_per_minute,
                self.tokens + elapsed * (self.requests_per_minute / 60.0)
            )
            self.last_refill = now

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

    def get_remaining(self) -> int:
        """Get approximate remaining tokens."""
        return int(self.tokens)


class WolframManager:
    """Manages interactions with Wolfram Alpha LLM API."""

    def __init__(
        self,
        app_id: str = None,
        api_url: str = None,
        timeout: float = None,
        cache_ttl: int = None,
        rate_limit: int = None,
    ):
        """
        Initialize WolframManager.

        Args:
            app_id: Wolfram Alpha App ID (uses config default if None)
            api_url: API endpoint URL (uses config default if None)
            timeout: Request timeout in seconds (uses config default if None)
            cache_ttl: Cache TTL in seconds (uses config default if None)
            rate_limit: Requests per minute limit (uses config default if None)
        """
        # Load config values with fallbacks
        try:
            from config.app_config import (
                WOLFRAM_APP_ID,
                WOLFRAM_API_URL,
                WOLFRAM_TIMEOUT,
                WOLFRAM_CACHE_TTL_SECONDS,
                WOLFRAM_RATE_LIMIT_PER_MINUTE,
                WOLFRAM_MAX_OUTPUT_CHARS,
            )
            self.app_id = app_id if app_id is not None else WOLFRAM_APP_ID
            self.api_url = api_url if api_url is not None else WOLFRAM_API_URL
            self.timeout = timeout if timeout is not None else WOLFRAM_TIMEOUT
            self.cache_ttl = cache_ttl if cache_ttl is not None else WOLFRAM_CACHE_TTL_SECONDS
            self.max_output_chars = WOLFRAM_MAX_OUTPUT_CHARS
            rate_limit_val = rate_limit if rate_limit is not None else WOLFRAM_RATE_LIMIT_PER_MINUTE
        except ImportError:
            # Fallback defaults if config not available
            self.app_id = app_id or ""
            self.api_url = api_url or "https://www.wolframalpha.com/api/v1/llm-api"
            self.timeout = timeout or 30.0
            self.cache_ttl = cache_ttl or 3600
            self.max_output_chars = 10000
            rate_limit_val = rate_limit or 60

        self.rate_limiter = WolframRateLimiter(rate_limit_val)

        # Simple in-memory cache: {hash: {"result": WolframResult, "timestamp": float}}
        self._cache: Dict[str, Dict[str, Any]] = {}

        logger.debug(f"[Wolfram] Initialized with app_id={'*' * 8 if self.app_id else 'NOT SET'}")

    def is_available(self) -> bool:
        """Check if Wolfram Alpha is configured and available."""
        return bool(self.app_id)

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        normalized = query.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _get_cached(self, query: str) -> Optional[WolframResult]:
        """Get cached result if valid."""
        key = self._get_cache_key(query)
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self.cache_ttl:
                result = entry["result"]
                # Create a copy with cached=True
                return WolframResult(
                    query=result.query,
                    success=result.success,
                    result=result.result,
                    input_interpreted=result.input_interpreted,
                    assumptions=result.assumptions.copy(),
                    related_queries=result.related_queries.copy(),
                    error=result.error,
                    execution_time=result.execution_time,
                    cached=True
                )
            else:
                # Expired, remove from cache
                del self._cache[key]
        return None

    def _set_cached(self, query: str, result: WolframResult) -> None:
        """Cache a result."""
        key = self._get_cache_key(query)
        self._cache[key] = {
            "result": result,
            "timestamp": time.time()
        }

    async def query(self, input_text: str) -> WolframResult:
        """
        Query Wolfram Alpha with natural language input.

        Args:
            input_text: The query to send to Wolfram Alpha

        Returns:
            WolframResult with success status and result or error
        """
        start_time = time.time()

        # Check configuration
        if not self.app_id:
            return WolframResult(
                query=input_text,
                success=False,
                error="Wolfram Alpha not configured (missing WOLFRAM_APP_ID)",
                execution_time=time.time() - start_time
            )

        # Check cache first
        cached = self._get_cached(input_text)
        if cached:
            logger.debug(f"[Wolfram] Cache hit for: {input_text[:50]}...")
            return cached

        # Check rate limit
        if not await self.rate_limiter.acquire():
            return WolframResult(
                query=input_text,
                success=False,
                error="Rate limit exceeded - please try again in a moment",
                execution_time=time.time() - start_time
            )

        # Make API request
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "appid": self.app_id,
                    "input": input_text,
                    "maxchars": str(self.max_output_chars),
                }

                response = await client.get(self.api_url, params=params)
                execution_time = time.time() - start_time

                if response.status_code == 200:
                    text = response.text
                    result = self._parse_response(input_text, text, execution_time)

                    # Cache successful results
                    if result.success:
                        self._set_cached(input_text, result)

                    logger.info(
                        f"[Wolfram] Query '{input_text[:40]}...' "
                        f"{'succeeded' if result.success else 'failed'} in {execution_time:.2f}s"
                    )

                    return result

                elif response.status_code == 403:
                    return WolframResult(
                        query=input_text,
                        success=False,
                        error="Invalid Wolfram Alpha API key",
                        execution_time=execution_time
                    )

                elif response.status_code == 501:
                    # Wolfram Alpha returns 501 for queries it can't handle
                    return WolframResult(
                        query=input_text,
                        success=False,
                        error=f"Wolfram Alpha could not process this query: {response.text[:200]}",
                        execution_time=execution_time
                    )

                else:
                    return WolframResult(
                        query=input_text,
                        success=False,
                        error=f"API error {response.status_code}: {response.text[:200]}",
                        execution_time=execution_time
                    )

        except httpx.TimeoutException:
            return WolframResult(
                query=input_text,
                success=False,
                error=f"Request timed out after {self.timeout}s",
                execution_time=time.time() - start_time
            )

        except httpx.RequestError as e:
            logger.error(f"[Wolfram] Connection error: {e}")
            return WolframResult(
                query=input_text,
                success=False,
                error=f"Connection error: {str(e)}",
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"[Wolfram] Unexpected error: {e}", exc_info=True)
            return WolframResult(
                query=input_text,
                success=False,
                error=f"Unexpected error: {str(e)}",
                execution_time=time.time() - start_time
            )

    def _parse_response(
        self,
        query: str,
        raw_text: str,
        execution_time: float
    ) -> WolframResult:
        """
        Parse Wolfram Alpha LLM API response.

        The LLM API returns plain text, not structured data.
        We extract any assumption indicators for user clarity.
        """
        if not raw_text or raw_text.strip() == "":
            return WolframResult(
                query=query,
                success=False,
                error="Empty response from Wolfram Alpha",
                execution_time=execution_time
            )

        # Check for error indicators in response
        lower_text = raw_text.lower()
        if "wolfram|alpha did not understand" in lower_text:
            return WolframResult(
                query=query,
                success=False,
                error="Wolfram Alpha could not interpret the query",
                result=raw_text,
                execution_time=execution_time
            )

        if "no result available" in lower_text:
            return WolframResult(
                query=query,
                success=False,
                error="No result available for this query",
                result=raw_text,
                execution_time=execution_time
            )

        # Extract assumptions if present (usually in parentheses or marked)
        assumptions = []
        if "assuming" in lower_text or "assumption" in lower_text:
            # Simple extraction - look for assumption patterns
            assumption_patterns = [
                r'\(assuming ([^)]+)\)',
                r'Assuming ([^.]+)\.',
            ]
            for pattern in assumption_patterns:
                matches = re.findall(pattern, raw_text, re.IGNORECASE)
                assumptions.extend(matches)

        return WolframResult(
            query=query,
            success=True,
            result=raw_text.strip(),
            input_interpreted=query,  # LLM API doesn't return separate interpretation
            assumptions=assumptions,
            execution_time=execution_time
        )

    def format_for_prompt(self, result: WolframResult) -> str:
        """
        Format Wolfram Alpha result for inclusion in LLM prompt.

        Used by the agentic controller to add results to context.
        """
        if not result.success:
            return f"Wolfram Alpha query failed: {result.error}"

        output = f"**Wolfram Alpha Result:**\n{result.result}"

        if result.assumptions:
            output += f"\n\n*Assumptions made: {', '.join(result.assumptions)}*"

        # Truncate if needed
        if len(output) > self.max_output_chars:
            output = output[:self.max_output_chars - 50] + "\n\n... [truncated]"

        return output

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status for UI display."""
        return {
            "remaining": self.rate_limiter.get_remaining(),
            "limit": self.rate_limiter.requests_per_minute,
            "cache_size": len(self._cache)
        }

    def clear_cache(self) -> int:
        """Clear the cache and return number of entries cleared."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"[Wolfram] Cleared {count} cached entries")
        return count
