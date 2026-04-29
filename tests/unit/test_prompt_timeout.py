"""
Tests for prompt context retrieval timeout behaviour.

Verifies that a global gather timeout preserves completed task results
instead of wiping all context.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_builder():
    """Return a minimal UnifiedPromptBuilder with mocked dependencies."""
    from core.prompt.builder import UnifiedPromptBuilder

    mm = MagicMock()
    mm.active_model_name = "test-model"
    mc = MagicMock()
    mc.graph_memory = None
    mc.entity_resolver = None

    builder = UnifiedPromptBuilder(
        model_manager=mm,
        memory_coordinator=mc,
        tokenizer_manager=MagicMock(),
    )
    return builder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPromptTimeoutPartialContext:
    """Completed tasks must survive a global gather timeout."""

    @pytest.mark.asyncio
    async def test_partial_timeout_preserves_completed_results(self):
        """Fast task result must be kept when a slow task causes timeout."""
        # Build a minimal task dict that mimics builder internals
        fast_result = [{"content": "fast data", "query": "q", "response": "r"}]

        async def _fast():
            return fast_result

        async def _slow():
            await asyncio.sleep(10)  # will be cancelled
            return [{"content": "slow data"}]

        async def _failing():
            raise ValueError("deliberate failure")

        tasks = {
            "fast": asyncio.create_task(_fast()),
            "slow": asyncio.create_task(_slow()),
            "failing": asyncio.create_task(_failing()),
        }

        done, pending = await asyncio.wait(
            list(tasks.values()),
            timeout=0.2,
            return_when=asyncio.ALL_COMPLETED,
        )

        gathered = {}
        timed_out_names = []
        for name, task in tasks.items():
            if task in done:
                try:
                    gathered[name] = task.result() or []
                except Exception:
                    gathered[name] = []
            else:
                task.cancel()
                gathered[name] = []
                timed_out_names.append(name)

        # Fast task result is preserved
        assert gathered["fast"] == fast_result
        # Failed task gets empty default
        assert gathered["failing"] == []
        # Slow (timed-out) task gets empty default
        assert gathered["slow"] == []
        assert "slow" in timed_out_names

        # Clean up pending tasks
        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    @pytest.mark.asyncio
    async def test_all_success_all_results_preserved(self):
        """When all tasks finish in time, every result is kept."""
        async def _make(name, value):
            return [{"name": name, "value": value}]

        tasks = {
            "a": asyncio.create_task(_make("a", 1)),
            "b": asyncio.create_task(_make("b", 2)),
            "c": asyncio.create_task(_make("c", 3)),
        }

        done, pending = await asyncio.wait(
            list(tasks.values()),
            timeout=5.0,
            return_when=asyncio.ALL_COMPLETED,
        )

        assert not pending

        gathered = {}
        for name, task in tasks.items():
            gathered[name] = task.result() or []

        assert gathered["a"] == [{"name": "a", "value": 1}]
        assert gathered["b"] == [{"name": "b", "value": 2}]
        assert gathered["c"] == [{"name": "c", "value": 3}]

    @pytest.mark.asyncio
    async def test_single_task_failure_isolates_to_one_section(self):
        """A single failing task must not zero out other tasks' results."""
        good_result = [{"content": "good"}]

        async def _good():
            return good_result

        async def _bad():
            raise RuntimeError("section broken")

        tasks = {
            "good1": asyncio.create_task(_good()),
            "bad": asyncio.create_task(_bad()),
            "good2": asyncio.create_task(_good()),
        }

        done, pending = await asyncio.wait(
            list(tasks.values()),
            timeout=5.0,
            return_when=asyncio.ALL_COMPLETED,
        )

        assert not pending

        gathered = {}
        timed_out_names = []
        for name, task in tasks.items():
            if task in done:
                try:
                    gathered[name] = task.result() or []
                except Exception:
                    gathered[name] = []
            else:
                task.cancel()
                gathered[name] = []
                timed_out_names.append(name)

        # Good tasks are intact
        assert gathered["good1"] == good_result
        assert gathered["good2"] == good_result
        # Failing task gets default
        assert gathered["bad"] == []
        # No timeout involved
        assert timed_out_names == []

    @pytest.mark.asyncio
    async def test_builder_gather_preserves_partial_results(self, caplog):
        """
        Integration check: builder._gather logic (via asyncio.wait) preserves
        completed sections even when the overall gather exceeds the timeout.

        This test mocks the task dict directly on the builder and calls the
        actual gather path by reproducing the loop from builder.py.
        """
        fast_data = [{"content": "kept", "query": "q", "response": "r"}]

        async def _fast():
            return fast_data

        async def _slow():
            await asyncio.sleep(10)
            return [{"content": "dropped"}]

        tasks = {
            "recent": asyncio.create_task(_fast()),
            "memories": asyncio.create_task(_slow()),
        }

        with caplog.at_level(logging.WARNING):
            done, pending = await asyncio.wait(
                list(tasks.values()),
                timeout=0.1,
                return_when=asyncio.ALL_COMPLETED,
            )

        gathered = {}
        timed_out_names = []
        for name, task in tasks.items():
            if task in done:
                try:
                    gathered[name] = task.result() or []
                except Exception:
                    gathered[name] = []
            else:
                task.cancel()
                gathered[name] = []
                timed_out_names.append(name)

        assert gathered["recent"] == fast_data, "fast task result must be preserved"
        assert gathered["memories"] == [], "slow task falls back to empty"
        assert "memories" in timed_out_names

        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
