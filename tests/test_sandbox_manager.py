# tests/test_sandbox_manager.py
"""
Unit tests for SandboxManager module.

Tests cover:
- SandboxResult data class functionality
- SandboxRateLimiter token bucket
- ExecutionCache operations
- SandboxManager execution functionality
- PersistentSession variable persistence (CRITICAL)
- Error handling and graceful degradation
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

# Import modules under test
from knowledge.sandbox_manager import (
    SandboxResult,
    SandboxRateLimiter,
    ExecutionCache,
    SandboxManager,
    PersistentSession,
    get_sandbox_manager,
)


# ===== SandboxResult Tests =====

class TestSandboxResult:
    def test_create_result_success(self):
        """Test SandboxResult creation for successful execution."""
        result = SandboxResult(
            code="print('hello')",
            success=True,
            stdout="hello\n",
            execution_time=0.5
        )
        assert result.success
        assert result.stdout == "hello\n"
        assert result.code == "print('hello')"
        assert result.execution_time == 0.5
        assert result.error is None

    def test_create_result_failure(self):
        """Test SandboxResult creation for failed execution."""
        result = SandboxResult(
            code="1/0",
            success=False,
            error="ZeroDivisionError: division by zero",
            execution_time=0.1
        )
        assert not result.success
        assert "ZeroDivisionError" in result.error
        assert result.stdout == ""

    def test_has_output(self):
        """Test has_output method."""
        # With stdout
        result1 = SandboxResult(code="", success=True, stdout="output")
        assert result1.has_output()

        # With stderr
        result2 = SandboxResult(code="", success=True, stderr="warning")
        assert result2.has_output()

        # With results
        result3 = SandboxResult(code="", success=True, results=[{"type": "text"}])
        assert result3.has_output()

        # Empty
        result4 = SandboxResult(code="", success=True)
        assert not result4.has_output()

    def test_get_display_output_success(self):
        """Test display output for successful execution."""
        result = SandboxResult(
            code="print(42)",
            success=True,
            stdout="42\n"
        )
        display = result.get_display_output()
        assert "42" in display
        assert "Output" in display

    def test_get_display_output_error(self):
        """Test display output for failed execution."""
        result = SandboxResult(
            code="raise ValueError",
            success=False,
            error="ValueError"
        )
        display = result.get_display_output()
        assert "ValueError" in display
        assert "Exception" in display

    def test_get_display_output_with_image(self):
        """Test display output with image result."""
        result = SandboxResult(
            code="plt.plot([1,2,3])",
            success=True,
            results=[{"type": "image", "format": "png", "data": "base64data"}]
        )
        display = result.get_display_output()
        assert "visualization" in display.lower()

    def test_get_display_output_empty(self):
        """Test display output when no output exists."""
        result = SandboxResult(code="x = 1", success=True)
        display = result.get_display_output()
        assert "No output" in display


# ===== SandboxRateLimiter Tests =====

class TestSandboxRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_success(self):
        """Test successful token acquisition."""
        limiter = SandboxRateLimiter(rate_per_minute=60)
        assert await limiter.acquire()

    @pytest.mark.asyncio
    async def test_acquire_rate_limited(self):
        """Test rate limiting when tokens exhausted."""
        limiter = SandboxRateLimiter(rate_per_minute=2)

        # Exhaust tokens
        assert await limiter.acquire()
        assert await limiter.acquire()

        # Should be rate limited
        assert not await limiter.acquire()

    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test that tokens refill over time."""
        limiter = SandboxRateLimiter(rate_per_minute=60)  # 1 per second

        # Exhaust most tokens
        for _ in range(55):
            await limiter.acquire()

        remaining_before = limiter.get_remaining()

        # Wait a bit for refill
        await asyncio.sleep(0.1)

        # Should have some tokens refilled
        remaining_after = limiter.get_remaining()
        assert remaining_after >= remaining_before

    def test_get_remaining(self):
        """Test get_remaining returns expected value."""
        limiter = SandboxRateLimiter(rate_per_minute=30)
        assert limiter.get_remaining() == 30


# ===== ExecutionCache Tests =====

class TestExecutionCache:
    def test_cache_miss(self):
        """Test cache miss for uncached code."""
        cache = ExecutionCache(ttl_seconds=3600)
        result = cache.get("print('hello')")
        assert result is None

    def test_cache_hit(self):
        """Test cache hit for cached code."""
        cache = ExecutionCache(ttl_seconds=3600)
        original = SandboxResult(
            code="print('hello')",
            success=True,
            stdout="hello\n",
            execution_time=0.5
        )
        cache.set("print('hello')", original)

        cached = cache.get("print('hello')")
        assert cached is not None
        assert cached.cached
        assert cached.stdout == original.stdout

    def test_cache_expiry(self):
        """Test cache expiry after TTL."""
        cache = ExecutionCache(ttl_seconds=0)  # Immediate expiry

        original = SandboxResult(code="x=1", success=True)
        cache.set("x=1", original)

        # Should be expired
        time.sleep(0.01)
        result = cache.get("x=1")
        assert result is None

    def test_cache_clear(self):
        """Test cache clear operation."""
        cache = ExecutionCache(ttl_seconds=3600)
        cache.set("code1", SandboxResult(code="1", success=True))
        cache.set("code2", SandboxResult(code="2", success=True))

        assert len(cache._cache) == 2
        cache.clear()
        assert len(cache._cache) == 0

    def test_cache_only_success(self):
        """Test that only successful results are cached."""
        cache = ExecutionCache(ttl_seconds=3600)

        failed = SandboxResult(code="1/0", success=False, error="Error")
        cache.set("1/0", failed)

        # Should not be cached
        assert cache.get("1/0") is None

    def test_cache_whitespace_normalization(self):
        """Test that code is normalized before caching."""
        cache = ExecutionCache(ttl_seconds=3600)
        original = SandboxResult(code="x=1", success=True)

        # Set with trailing whitespace
        cache.set("  x=1  ", original)

        # Get with different whitespace
        cached = cache.get("x=1")
        assert cached is not None


# ===== SandboxManager Tests (with mocking) =====

class TestSandboxManager:
    def test_is_available_no_key(self):
        """Test is_available returns False when no API key."""
        with patch.dict('os.environ', {'E2B_API_KEY': ''}):
            # Create new manager to pick up env change
            manager = SandboxManager(api_key="")
            assert not manager.is_available()

    def test_is_available_with_key(self):
        """Test is_available returns True when API key is set."""
        manager = SandboxManager(api_key="test_key")
        assert manager.is_available()

    @pytest.mark.asyncio
    async def test_execute_code_not_available(self):
        """Test execute_code when sandbox not available."""
        manager = SandboxManager(api_key="")

        result = await manager.execute_code("print('test')")

        assert not result.success
        assert "not configured" in result.error

    @pytest.mark.asyncio
    async def test_execute_code_rate_limited(self):
        """Test execute_code when rate limited."""
        manager = SandboxManager(api_key="test_key", rate_limit=1)

        # Mock the rate limiter to deny
        manager.rate_limiter.tokens = 0

        result = await manager.execute_code("print('test')", use_cache=False)

        assert not result.success
        assert "Rate limit" in result.error

    @pytest.mark.asyncio
    async def test_execute_code_cached(self):
        """Test execute_code returns cached result."""
        manager = SandboxManager(api_key="test_key")

        # Pre-populate cache
        cached_result = SandboxResult(
            code="print('cached')",
            success=True,
            stdout="cached\n",
            execution_time=0.1
        )
        manager.cache.set("print('cached')", cached_result)

        result = await manager.execute_code("print('cached')", use_cache=True)

        assert result.success
        assert result.cached
        assert result.stdout == "cached\n"

    def test_format_for_prompt_success(self):
        """Test format_for_prompt for successful execution."""
        manager = SandboxManager(api_key="test")
        result = SandboxResult(
            code="print(42)",
            success=True,
            stdout="42\n",
            execution_time=0.5
        )

        formatted = manager.format_for_prompt(result)

        assert "[Code Execution Result]" in formatted
        assert "Success: True" in formatted
        assert "42" in formatted
        assert "0.50s" in formatted

    def test_format_for_prompt_with_purpose(self):
        """Test format_for_prompt with purpose."""
        manager = SandboxManager(api_key="test")
        result = SandboxResult(code="x=1", success=True)

        formatted = manager.format_for_prompt(result, purpose="test calculation")

        assert "test calculation" in formatted

    def test_format_for_prompt_error(self):
        """Test format_for_prompt for failed execution."""
        manager = SandboxManager(api_key="test")
        result = SandboxResult(
            code="1/0",
            success=False,
            error="ZeroDivisionError"
        )

        formatted = manager.format_for_prompt(result)

        assert "Success: False" in formatted
        assert "Error" in formatted
        assert "ZeroDivisionError" in formatted

    def test_format_for_prompt_truncated(self):
        """Test format_for_prompt includes truncation note."""
        manager = SandboxManager(api_key="test")
        result = SandboxResult(
            code="print('x' * 10000)",
            success=True,
            stdout="x" * 100,
            truncated=True
        )

        formatted = manager.format_for_prompt(result)

        assert "truncated" in formatted.lower()

    def test_truncate_text(self):
        """Test text truncation preserves head and tail."""
        manager = SandboxManager(api_key="test")

        long_text = "A" * 1000 + "MIDDLE" + "Z" * 1000
        truncated = manager._truncate_text(long_text, max_chars=500)

        assert len(truncated) <= 600  # Some allowance for notice
        assert "A" in truncated  # Head preserved
        assert "Z" in truncated  # Tail preserved
        assert "truncated" in truncated

    def test_truncate_text_short(self):
        """Test short text is not truncated."""
        manager = SandboxManager(api_key="test")

        short_text = "Hello world"
        result = manager._truncate_text(short_text, max_chars=500)

        assert result == short_text

    def test_get_rate_limit_status(self):
        """Test rate limit status reporting."""
        manager = SandboxManager(api_key="test", rate_limit=30)

        status = manager.get_rate_limit_status()

        assert "remaining" in status
        assert "limit" in status
        assert "cache_size" in status
        assert status["limit"] == 30

    def test_clear_cache(self):
        """Test cache clearing."""
        manager = SandboxManager(api_key="test")

        # Add some entries
        manager.cache.set("code1", SandboxResult(code="1", success=True))
        manager.cache.set("code2", SandboxResult(code="2", success=True))

        count = manager.clear_cache()

        assert count == 2
        assert len(manager.cache._cache) == 0


# ===== PersistentSession Tests (CRITICAL) =====

class TestPersistentSession:
    """Tests for variable persistence across executions - the key feature of PersistentSession."""

    def test_session_properties(self):
        """Test session property accessors."""
        mock_sandbox = MagicMock()
        manager = SandboxManager(api_key="test")
        session = PersistentSession(mock_sandbox, manager)

        assert not session.is_closed
        assert session.execution_count == 0
        assert session.age_seconds >= 0

    @pytest.mark.asyncio
    async def test_session_close(self):
        """Test session close operation."""
        mock_sandbox = MagicMock()
        manager = SandboxManager(api_key="test")
        session = PersistentSession(mock_sandbox, manager)

        await session.close()

        assert session.is_closed
        mock_sandbox.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_run_after_close(self):
        """Test that running code on closed session raises error."""
        mock_sandbox = MagicMock()
        manager = SandboxManager(api_key="test")
        session = PersistentSession(mock_sandbox, manager)

        await session.close()

        with pytest.raises(RuntimeError, match="closed"):
            await session.run("print('test')")

    @pytest.mark.asyncio
    async def test_session_execution_count(self):
        """Test execution count increments."""
        mock_sandbox = MagicMock()
        mock_execution = MagicMock()
        mock_execution.logs.stdout = ["output"]
        mock_execution.logs.stderr = []
        mock_execution.results = []
        mock_execution.error = None
        mock_sandbox.run_code.return_value = mock_execution

        manager = SandboxManager(api_key="test")
        session = PersistentSession(mock_sandbox, manager)

        assert session.execution_count == 0

        await session.run("x = 1")
        assert session.execution_count == 1

        await session.run("x = 2")
        assert session.execution_count == 2

        await session.close()

    @pytest.mark.asyncio
    async def test_session_rate_limited(self):
        """Test session respects rate limiting."""
        mock_sandbox = MagicMock()
        manager = SandboxManager(api_key="test", rate_limit=1)
        session = PersistentSession(mock_sandbox, manager)

        # Exhaust rate limit
        manager.rate_limiter.tokens = 0

        result = await session.run("print('test')")

        assert not result.success
        assert "Rate limit" in result.error

        await session.close()


# ===== Module Singleton Tests =====

class TestModuleSingleton:
    def test_get_sandbox_manager_returns_same_instance(self):
        """Test singleton pattern for get_sandbox_manager."""
        manager1 = get_sandbox_manager()
        manager2 = get_sandbox_manager()

        assert manager1 is manager2


# ===== Integration-style Tests (require E2B key) =====

@pytest.mark.skipif(
    not SandboxManager().is_available(),
    reason="E2B not configured"
)
class TestSandboxManagerIntegration:
    """Integration tests that require actual E2B API key."""

    @pytest.mark.asyncio
    async def test_simple_execution(self):
        """Test actual code execution."""
        manager = get_sandbox_manager()

        result = await manager.execute_code("print('hello world')")

        assert result.success
        assert "hello world" in result.stdout

    @pytest.mark.asyncio
    async def test_math_computation(self):
        """Test numerical computation with numpy."""
        manager = get_sandbox_manager()

        code = """
import numpy as np
x = np.array([1, 2, 3, 4, 5])
print(f"Mean: {np.mean(x)}")
"""
        result = await manager.execute_code(code)

        assert result.success
        assert "Mean: 3.0" in result.stdout

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling for invalid code."""
        manager = get_sandbox_manager()

        result = await manager.execute_code("1/0")

        assert not result.success
        assert "ZeroDivisionError" in (result.error or result.stderr)

    @pytest.mark.asyncio
    async def test_session_variable_persistence(self):
        """Test that variables persist across session runs - CRITICAL TEST."""
        manager = get_sandbox_manager()
        session = await manager.create_session()

        try:
            # Define variable in first execution
            r1 = await session.run("x = 42")
            assert r1.success

            # Use variable in second execution
            r2 = await session.run("print(x * 2)")
            assert r2.success
            assert "84" in r2.stdout

            # Define another variable using the first
            r3 = await session.run("y = x + 8; print(y)")
            assert r3.success
            assert "50" in r3.stdout

        finally:
            await session.close()

    @pytest.mark.asyncio
    async def test_session_dataframe_persistence(self):
        """Test pandas DataFrame persistence across calls."""
        manager = get_sandbox_manager()
        session = await manager.create_session()

        try:
            # Create DataFrame
            r1 = await session.run("""
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print("Created DataFrame")
""")
            assert r1.success

            # Use DataFrame
            r2 = await session.run("print(df.sum())")
            assert r2.success
            assert "6" in r2.stdout   # sum of column 'a'
            assert "15" in r2.stdout  # sum of column 'b'

        finally:
            await session.close()

    @pytest.mark.asyncio
    async def test_session_function_persistence(self):
        """Test that defined functions persist."""
        manager = get_sandbox_manager()
        session = await manager.create_session()

        try:
            # Define function
            r1 = await session.run("""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
print("Function defined")
""")
            assert r1.success

            # Use function
            r2 = await session.run("print(factorial(5))")
            assert r2.success
            assert "120" in r2.stdout

        finally:
            await session.close()
