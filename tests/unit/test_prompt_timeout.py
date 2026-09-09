"""Timeout/cancellation assertions through UnifiedPromptBuilder.build_prompt.

Gatherers are event-controlled I/O boundaries. Only the wait deadline is
shortened; production creates/drains the tasks and retains their results.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

from tests.unit.test_independent_prompt_audit import full_builder, retrieval_limits


RECENT = [{"query": "Synthetic question", "response": "Synthetic retained answer."}]


def _make_builder(monkeypatch):
    return full_builder(monkeypatch, RECENT, budget=10000)


class TestPromptTimeoutPartialContext:
    @pytest.mark.asyncio
    async def test_partial_timeout_preserves_completed_results(self, monkeypatch, caplog):
        builder = _make_builder(monkeypatch)
        fast_done, slow_entered, slow_closed = (asyncio.Event() for _ in range(3))

        async def recent(*args):
            fast_done.set()
            return RECENT

        async def profile(*args, **kwargs):
            slow_entered.set()
            try:
                await asyncio.Future()
            finally:
                slow_closed.set()

        builder.context_gatherer._get_recent_conversations = recent
        builder.context_gatherer.get_user_profile_context = profile
        real_wait = asyncio.wait

        async def expired_deadline(tasks, *, timeout, return_when):
            assert timeout == 30.0  # exercise the deployed global gather wait
            await asyncio.wait_for(asyncio.gather(fast_done.wait(), slow_entered.wait()), 2)
            return await real_wait(tasks, timeout=0, return_when=return_when)

        monkeypatch.setattr("core.prompt.builder.asyncio.wait", expired_deadline)
        result = await builder.build_prompt("Synthetic question", retrieval_overrides=retrieval_limits())
        assert result.get("recent_conversations") == RECENT
        assert not result.get("user_profile") and "_build_time" in result
        assert slow_closed.is_set(), "timed-out gatherer survived its request"
        assert "partial context used" in caplog.text
        assert builder.context_gatherer._distress_active is False

    @pytest.mark.asyncio
    async def test_all_success_all_results_preserved(self, monkeypatch):
        builder = _make_builder(monkeypatch)
        builder.context_gatherer.get_user_profile_context = AsyncMock(return_value="Synthetic profile.")
        result = await builder.build_prompt("Synthetic question", retrieval_overrides=retrieval_limits())
        assert result.get("recent_conversations") == RECENT
        assert result.get("user_profile") == "Synthetic profile."
        assert "_build_time" in result

    @pytest.mark.asyncio
    async def test_one_section_exception_preserves_other_sections(self, monkeypatch, caplog):
        builder = _make_builder(monkeypatch)
        builder.context_gatherer.get_user_profile_context = AsyncMock(side_effect=ValueError("synthetic section failure"))
        result = await builder.build_prompt("Synthetic question", retrieval_overrides=retrieval_limits())
        assert result.get("recent_conversations") == RECENT
        assert not result.get("user_profile") and "_build_time" in result
        assert "synthetic section failure" in caplog.text

    @pytest.mark.asyncio
    async def test_cancelled_request_drains_gatherers_before_reset(self, monkeypatch):
        builder = _make_builder(monkeypatch)
        entered, closed = asyncio.Event(), asyncio.Event()
        state_at_close = []

        async def slow_recent(*args):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                state_at_close.append(builder.context_gatherer._current_turn_upload_filenames[:])
                closed.set()

        builder.context_gatherer._get_recent_conversations = slow_recent
        # build_prompt's real kwarg for this is `_uploaded_filenames` (see
        # core/prompt/builder.py:1153-1154 and its use at :2182 from
        # build_prompt_from_context) -- NOT `current_turn_upload_filenames`,
        # which build_prompt's **kwargs silently absorbs and never reads.
        task = asyncio.create_task(builder.build_prompt(
            "Synthetic question", retrieval_overrides=retrieval_limits(),
            _uploaded_filenames=["synthetic.txt"]))
        try:
            await asyncio.wait_for(entered.wait(), 2)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert closed.is_set(), "retrieval outlived its cancelled request"
        assert state_at_close == [["synthetic.txt"]]
        assert builder.context_gatherer._current_turn_upload_filenames == []
