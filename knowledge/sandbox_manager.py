"""
E2B Code Sandbox Manager for Daemon.

Provides secure Python code execution in ephemeral Firecracker microVMs.
Supports both stateless (single execution) and stateful (persistent session) modes.

Module Contract:
- Purpose: Handle multi-step computations, data analysis, visualizations
           via E2B's secure code interpreter API
- Inputs:
  - execute_code(code: str) -> SandboxResult: Execute in ephemeral sandbox
  - create_session() -> PersistentSession: Create stateful session
  - is_available() -> bool: Check if E2B is configured
  - format_for_prompt(result: SandboxResult) -> str: Format for LLM context
- Outputs:
  - SandboxResult with success status, stdout, stderr, error, results, execution time
- Side effects:
  - HTTP requests to E2B API
  - In-memory caching of results (ephemeral mode only)
  - Rate limiting tracking
- Error handling:
  - Graceful degradation on API errors
  - Timeout handling
  - Rate limit enforcement

Usage:
    manager = SandboxManager()

    # Ephemeral (stateless) - sandbox destroyed after execution
    result = await manager.execute_code("print('hello')")

    # Persistent session - variables survive across calls
    session = await manager.create_session()
    try:
        await session.run("x = 42")
        await session.run("print(x)")  # x is still available!
    finally:
        await session.close()

Dependencies:
    - config.app_config (SANDBOX_* constants)
    - e2b_code_interpreter (E2B SDK)
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SandboxResult:
    """Result from code execution."""
    code: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    results: List[Dict[str, Any]] = field(default_factory=list)  # Rich outputs (charts, etc.)
    execution_time: float = 0.0
    cached: bool = False
    truncated: bool = False

    def has_output(self) -> bool:
        """Check if execution produced any output."""
        return bool(self.stdout or self.stderr or self.results)

    def get_display_output(self) -> str:
        """Get human-readable output for display."""
        parts = []
        if self.stdout:
            parts.append(f"Output:\n{self.stdout}")
        if self.stderr and self.success:  # Only show stderr if not an error
            parts.append(f"Stderr:\n{self.stderr}")
        if self.error:
            parts.append(f"Exception:\n{self.error}")
        if self.results:
            for i, r in enumerate(self.results):
                if r.get("type") == "image":
                    parts.append(f"[Generated visualization #{i+1}]")
                elif r.get("type") == "text":
                    parts.append(f"Output #{i+1}:\n{r.get('text', '[data]')}")
        return "\n\n".join(parts) if parts else "(No output)"


# ============================================================================
# Rate Limiter (Token Bucket)
# ============================================================================

class SandboxRateLimiter:
    """Simple token bucket rate limiter for API calls."""

    def __init__(self, rate_per_minute: int):
        self.rate = rate_per_minute
        self.tokens = float(rate_per_minute)
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Try to acquire a token. Returns False if rate limited."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + elapsed * (self.rate / 60))
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def get_remaining(self) -> int:
        """Get approximate remaining tokens."""
        return int(self.tokens)


# ============================================================================
# Cache (for ephemeral mode only)
# ============================================================================

class ExecutionCache:
    """Simple TTL cache for code execution results."""

    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self._cache: Dict[str, tuple[SandboxResult, float]] = {}

    def _hash_code(self, code: str) -> str:
        """Generate cache key from code."""
        return hashlib.md5(code.strip().encode()).hexdigest()

    def get(self, code: str) -> Optional[SandboxResult]:
        """Get cached result if exists and not expired."""
        key = self._hash_code(code)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                # Return a copy with cached flag set
                cached_result = SandboxResult(
                    code=result.code,
                    success=result.success,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    error=result.error,
                    results=result.results.copy(),
                    execution_time=result.execution_time,
                    cached=True,
                    truncated=result.truncated
                )
                return cached_result
            del self._cache[key]
        return None

    def set(self, code: str, result: SandboxResult):
        """Cache a successful result."""
        if result.success:
            key = self._hash_code(code)
            self._cache[key] = (result, time.time())

    def clear(self):
        """Clear all cached results."""
        self._cache.clear()


# ============================================================================
# Main Manager Class
# ============================================================================

class SandboxManager:
    """
    Manages E2B code sandbox execution.

    Features:
    - Ephemeral execution (stateless, sandbox destroyed after use)
    - Persistent sessions (state preserved across multiple executions)
    - Result caching for identical code (ephemeral mode only)
    - Rate limiting
    - Output truncation for large results
    - Rich output handling (matplotlib figures, dataframes)
    """

    def __init__(
        self,
        api_key: str = None,
        timeout: int = None,
        max_output_chars: int = None,
        cache_ttl: int = None,
        rate_limit: int = None,
    ):
        """
        Initialize SandboxManager.

        Args:
            api_key: E2B API key (uses config default if None)
            timeout: Execution timeout in seconds (uses config default if None)
            max_output_chars: Max output chars (uses config default if None)
            cache_ttl: Cache TTL in seconds (uses config default if None)
            rate_limit: Requests per minute limit (uses config default if None)
        """
        # Load config values with fallbacks
        try:
            from config.app_config import (
                SANDBOX_ENABLED,
                SANDBOX_API_KEY,
                SANDBOX_TIMEOUT_SECONDS,
                SANDBOX_MAX_OUTPUT_CHARS,
                SANDBOX_CACHE_TTL_SECONDS,
                SANDBOX_RATE_LIMIT_PER_MINUTE,
            )
            self._config_enabled = SANDBOX_ENABLED
            self.api_key = api_key if api_key is not None else SANDBOX_API_KEY
            self.timeout = timeout if timeout is not None else SANDBOX_TIMEOUT_SECONDS
            self.max_output_chars = max_output_chars if max_output_chars is not None else SANDBOX_MAX_OUTPUT_CHARS
            self.cache_ttl = cache_ttl if cache_ttl is not None else SANDBOX_CACHE_TTL_SECONDS
            rate_limit_val = rate_limit if rate_limit is not None else SANDBOX_RATE_LIMIT_PER_MINUTE
        except ImportError:
            # Fallback defaults if config not available
            self._config_enabled = True
            self.api_key = api_key or ""
            self.timeout = timeout or 60
            self.max_output_chars = max_output_chars or 4000
            self.cache_ttl = cache_ttl or 3600
            rate_limit_val = rate_limit or 30

        self.enabled = self._config_enabled and bool(self.api_key)
        self.rate_limiter = SandboxRateLimiter(rate_limit_val)
        self.cache = ExecutionCache(self.cache_ttl)

        if not self.enabled:
            if not self._config_enabled:
                logger.warning("SandboxManager disabled: SANDBOX_ENABLED is False")
            elif not self.api_key:
                logger.warning("SandboxManager disabled: E2B_API_KEY not configured")

    def is_available(self) -> bool:
        """Check if sandbox execution is available."""
        return self.enabled

    async def execute_code(
        self,
        code: str,
        use_cache: bool = True,
        timeout: Optional[int] = None
    ) -> SandboxResult:
        """
        Execute Python code in an ephemeral sandbox.

        The sandbox is created and destroyed for this single execution.
        For multi-step workflows where variables need to persist, use create_session() instead.

        Args:
            code: Python code to execute
            use_cache: Whether to use cached results for identical code
            timeout: Execution timeout in seconds (default from config)

        Returns:
            SandboxResult with stdout, stderr, errors, and rich outputs
        """
        if not self.enabled:
            return SandboxResult(
                code=code,
                success=False,
                error="Sandbox execution not available (E2B_API_KEY not configured)"
            )

        # Check cache
        if use_cache:
            cached = self.cache.get(code)
            if cached:
                logger.debug(f"[Sandbox] Cache hit for code hash {hashlib.md5(code.encode()).hexdigest()[:8]}")
                return cached

        # Rate limiting
        if not await self.rate_limiter.acquire():
            return SandboxResult(
                code=code,
                success=False,
                error="Rate limit exceeded. Please wait before executing more code."
            )

        timeout = timeout or self.timeout
        start_time = time.time()

        try:
            # Import here to avoid startup cost if not used
            from e2b_code_interpreter import Sandbox

            # Execute in ephemeral sandbox (use create() for newer SDK versions)
            sandbox = Sandbox.create(api_key=self.api_key)
            try:
                execution = sandbox.run_code(code, timeout=timeout)
                result = self._parse_execution(code, execution, start_time)

                # Cache successful results
                if result.success and use_cache:
                    self.cache.set(code, result)

                return result
            finally:
                sandbox.kill()

        except Exception as e:
            logger.error(f"[Sandbox] Execution error: {e}")
            return SandboxResult(
                code=code,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    async def create_session(self) -> "PersistentSession":
        """
        Create a persistent sandbox session.

        Variables and state persist across multiple run() calls within the session.
        The caller is responsible for calling close() when done.

        This is the preferred method for the ReAct loop where multi-step
        computations need to share state.

        Usage:
            session = await manager.create_session()
            try:
                await session.run("x = 42")
                await session.run("print(x)")  # x is still available!
            finally:
                await session.close()

        Returns:
            PersistentSession wrapper for stateful execution

        Raises:
            RuntimeError: If sandbox is not available
        """
        if not self.enabled:
            raise RuntimeError("Sandbox execution not available (E2B_API_KEY not configured)")

        try:
            from e2b_code_interpreter import Sandbox

            # Create sandbox - will be kept alive until close() is called
            # Use create() for newer SDK versions
            sandbox = Sandbox.create(api_key=self.api_key)
            logger.info("[Sandbox] Created persistent sandbox session")

            return PersistentSession(sandbox, self)

        except Exception as e:
            logger.error(f"[Sandbox] Failed to create sandbox session: {e}")
            raise RuntimeError(f"Failed to create sandbox session: {e}")

    def _parse_execution(self, code: str, execution: Any, start_time: float) -> SandboxResult:
        """Parse E2B execution result into SandboxResult."""
        stdout = ""
        stderr = ""

        # Handle stdout/stderr
        if hasattr(execution, 'logs'):
            if hasattr(execution.logs, 'stdout') and execution.logs.stdout:
                stdout = "".join(execution.logs.stdout)
            if hasattr(execution.logs, 'stderr') and execution.logs.stderr:
                stderr = "".join(execution.logs.stderr)

        # Truncate large outputs
        truncated = False
        if len(stdout) > self.max_output_chars:
            stdout = self._truncate_text(stdout, self.max_output_chars)
            truncated = True
        if len(stderr) > self.max_output_chars:
            stderr = self._truncate_text(stderr, self.max_output_chars)
            truncated = True

        # Parse rich results (charts, dataframes, etc.)
        results = []
        if hasattr(execution, 'results') and execution.results:
            for r in execution.results:
                if hasattr(r, 'png') and r.png:
                    # Matplotlib figure - store as base64
                    results.append({
                        "type": "image",
                        "format": "png",
                        "data": r.png  # Already base64
                    })
                elif hasattr(r, 'text') and r.text:
                    # Text output (dataframe repr, etc.)
                    text = r.text
                    if len(text) > self.max_output_chars:
                        text = self._truncate_text(text, self.max_output_chars)
                        truncated = True
                    results.append({
                        "type": "text",
                        "text": text
                    })

        # Check for errors
        error = None
        success = True
        if hasattr(execution, 'error') and execution.error:
            error = str(execution.error)
            success = False
        elif stderr and any(err in stderr.lower() for err in ['error', 'traceback', 'exception']):
            # Some errors go to stderr without setting error flag
            error = stderr
            success = False

        return SandboxResult(
            code=code,
            success=success,
            stdout=stdout,
            stderr=stderr,
            error=error,
            results=results,
            execution_time=time.time() - start_time,
            truncated=truncated
        )

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Truncate text, keeping beginning and end for context."""
        if len(text) <= max_chars:
            return text

        # Keep first 80%, last 20% (useful for seeing error traces at end)
        head_size = int(max_chars * 0.8)
        tail_size = max_chars - head_size - 60  # Room for truncation notice

        head = text[:head_size]
        tail = text[-tail_size:] if tail_size > 0 else ""

        truncated_chars = len(text) - max_chars
        return f"{head}\n\n... [truncated {truncated_chars:,} chars] ...\n\n{tail}"

    def format_for_prompt(self, result: SandboxResult, purpose: Optional[str] = None) -> str:
        """Format sandbox result for injection into LLM context."""
        lines = []

        if purpose:
            lines.append(f"[Code Execution: {purpose}]")
        else:
            lines.append("[Code Execution Result]")

        lines.append(f"Success: {result.success}")
        lines.append(f"Execution time: {result.execution_time:.2f}s")

        if result.stdout:
            lines.append(f"\nOutput:\n```\n{result.stdout}\n```")

        if result.stderr and result.success:  # Only show stderr if not an error
            lines.append(f"\nStderr:\n```\n{result.stderr}\n```")

        if result.error:
            lines.append(f"\nError:\n```\n{result.error}\n```")

        if result.results:
            for i, r in enumerate(result.results):
                if r["type"] == "image":
                    lines.append(f"\n[Generated visualization #{i+1}]")
                elif r["type"] == "text":
                    lines.append(f"\nRich output #{i+1}:\n```\n{r['text']}\n```")

        if result.truncated:
            lines.append("\n(Note: Output was truncated due to length)")

        return "\n".join(lines)

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status for UI display."""
        return {
            "remaining": self.rate_limiter.get_remaining(),
            "limit": self.rate_limiter.rate,
            "cache_size": len(self.cache._cache)
        }

    def clear_cache(self) -> int:
        """Clear the cache and return number of entries cleared."""
        count = len(self.cache._cache)
        self.cache.clear()
        logger.info(f"[Sandbox] Cleared {count} cached entries")
        return count

    async def cleanup(self):
        """Clean up resources (clear cache)."""
        self.cache.clear()
        logger.info("[Sandbox] SandboxManager cleanup complete")


# ============================================================================
# Persistent Session Class
# ============================================================================

class PersistentSession:
    """
    Wrapper for persistent sandbox execution with state.

    Variables defined in one run() call are available in subsequent calls.
    This enables multi-step computations like:

        await session.run("df = pd.read_csv('data.csv')")
        await session.run("print(df.describe())")  # df is still available

    Always call close() when done to release the sandbox resources.
    """

    def __init__(self, sandbox: Any, manager: SandboxManager):
        self._sandbox = sandbox
        self._manager = manager
        self._closed = False
        self._execution_count = 0
        self._created_at = datetime.now()

    @property
    def is_closed(self) -> bool:
        """Check if session has been closed."""
        return self._closed

    @property
    def execution_count(self) -> int:
        """Number of code executions in this session."""
        return self._execution_count

    @property
    def age_seconds(self) -> float:
        """How long this session has been alive."""
        return (datetime.now() - self._created_at).total_seconds()

    async def run(self, code: str, timeout: Optional[int] = None) -> SandboxResult:
        """
        Execute code in the persistent session.

        Variables from previous run() calls are still available.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds

        Returns:
            SandboxResult with execution results

        Raises:
            RuntimeError: If session has been closed
        """
        if self._closed:
            raise RuntimeError("Cannot run code: session has been closed")

        # Rate limiting still applies
        if not await self._manager.rate_limiter.acquire():
            return SandboxResult(
                code=code,
                success=False,
                error="Rate limit exceeded. Please wait before executing more code."
            )

        timeout = timeout or self._manager.timeout
        start_time = time.time()

        try:
            execution = self._sandbox.run_code(code, timeout=timeout)
            self._execution_count += 1
            return self._manager._parse_execution(code, execution, start_time)

        except Exception as e:
            logger.error(f"[Sandbox] Session execution error: {e}")
            return SandboxResult(
                code=code,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    async def close(self):
        """
        Close the session and release sandbox resources.

        After calling close(), the session cannot be used again.
        """
        if not self._closed:
            try:
                self._sandbox.kill()
                logger.info(
                    f"[Sandbox] Closed sandbox session after {self._execution_count} executions "
                    f"({self.age_seconds:.1f}s lifetime)"
                )
            except Exception as e:
                logger.warning(f"[Sandbox] Error closing sandbox session: {e}")
            finally:
                self._closed = True


# ============================================================================
# Module-level singleton (optional, for easy import)
# ============================================================================

_manager: Optional[SandboxManager] = None


def get_sandbox_manager() -> SandboxManager:
    """Get or create the singleton SandboxManager instance."""
    global _manager
    if _manager is None:
        _manager = SandboxManager()
    return _manager
