"""Tests for utils/async_results.py — the shared classifier for
``asyncio.gather(..., return_exceptions=True)`` result lists.

``asyncio.CancelledError`` is a ``BaseException``, not an ``Exception``, so it
slips through every ``isinstance(r, Exception)`` / ``r is not None`` filter in
the code base when a child task of a gather is cancelled (client disconnect,
an outer timeout, an explicit ``task.cancel()``). These tests drive the
deployed ``classify_gather_results``/``partition_gather_results`` directly.
"""
import asyncio

import pytest

from utils.async_results import (
    GatherOutcome,
    classify_gather_results,
    partition_gather_results,
)


def test_mixed_list_classifies_positionally():
    """[1, ValueError(), None, CancelledError()] → errors at 1 and 3, values
    1 and None at 0 and 2; partition separates values from errors, the
    CancelledError included among the errors."""
    err_v = ValueError("boom")
    err_c = asyncio.CancelledError()
    results = [1, err_v, None, err_c]

    outcomes = classify_gather_results(results)
    assert outcomes == [
        GatherOutcome(0, 1, None),
        GatherOutcome(1, None, err_v),
        GatherOutcome(2, None, None),
        GatherOutcome(3, None, err_c),
    ]

    values, errors = partition_gather_results(results)
    assert values == [1, None]
    assert errors == [err_v, err_c]
    assert any(isinstance(e, asyncio.CancelledError) for e in errors)


def test_no_running_loop_cancelled_error_is_error_not_raised():
    """A plain sync caller (no event loop) — CancelledError in results is an
    ordinary per-item error; nothing is raised."""
    outcomes = classify_gather_results([asyncio.CancelledError()])
    assert len(outcomes) == 1
    assert isinstance(outcomes[0].error, asyncio.CancelledError)
    assert outcomes[0].value is None


@pytest.mark.asyncio
async def test_healthy_task_cancelled_error_is_error_not_raised():
    """Inside a healthy (non-cancelling) task, a CancelledError result is
    still just an ordinary per-item error — nothing is raised."""
    task = asyncio.current_task()
    assert task is not None and task.cancelling() == 0

    outcomes = classify_gather_results([asyncio.CancelledError(), "ok"])
    assert isinstance(outcomes[0].error, asyncio.CancelledError)
    assert outcomes[1] == GatherOutcome(1, "ok", None)


async def _cancelled_from_outside_then_classify(results):
    """Wait until cancelled from OUTSIDE, swallow the delivered CancelledError
    (so the task is not unwound by asyncio itself), then call the helper while
    ``Task.cancelling()`` is still 1. Returns normally only if the helper
    does not re-raise. (A task that calls ``self.cancel()`` would end
    cancelled on ``await`` regardless of the helper — that recipe proves
    nothing, referee note 2026-09-16.)"""
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    assert asyncio.current_task().cancelling() == 1
    partition_gather_results(results)
    return "completed"


@pytest.mark.asyncio
async def test_caller_being_cancelled_reraises_cancelled_error():
    """When the CURRENT task has a pending cancellation (Task.cancelling() >
    0) and a CancelledError is among the results, the helper re-raises it so
    the caller unwinds as asyncio expects."""
    task = asyncio.create_task(_cancelled_from_outside_then_classify([asyncio.CancelledError()]))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_caller_being_cancelled_without_cancelled_result_returns_normally():
    """Control for the test above, same task recipe: the re-raise fires ONLY
    when a CancelledError is actually among the results — an ordinary
    Exception while the caller is mid-cancellation is an ordinary per-item
    error and the helper returns normally."""
    task = asyncio.create_task(_cancelled_from_outside_then_classify([ValueError("boom")]))
    await asyncio.sleep(0)
    task.cancel()
    assert await task == "completed"


@pytest.mark.asyncio
async def test_real_gather_with_cancelled_child_and_survivor():
    """A real asyncio.gather(..., return_exceptions=True) where one child
    raises CancelledError and the other returns normally — classify sees one
    error (the CancelledError) and one value ("ok")."""

    async def cancels():
        raise asyncio.CancelledError()

    async def ok():
        return "ok"

    results = await asyncio.gather(cancels(), ok(), return_exceptions=True)
    outcomes = classify_gather_results(results)

    assert isinstance(outcomes[0].error, asyncio.CancelledError)
    assert outcomes[0].value is None
    assert outcomes[1] == GatherOutcome(1, "ok", None)
