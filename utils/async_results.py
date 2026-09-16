"""Classify the result list of ``asyncio.gather(..., return_exceptions=True)``.

One decision, made once: a ``BaseException`` in the result list is an ERROR
for that position — including ``asyncio.CancelledError``, which is NOT an
``Exception`` and therefore slipped through every ``isinstance(r, Exception)``
/ ``r is not None`` filter in the code base (2026-09-16 typecheck triage:
the agentic parallel tool dispatch, best-of/duel, Gmail fetch and synthesis
articulation all read attributes off a cancelled child's CancelledError).

Caller cancellation is honoured, never swallowed: when the CURRENT task has a
pending cancellation request (``Task.cancelling()``) and a CancelledError is
among the results, that CancelledError is re-raised so the caller unwinds as
asyncio expects. A child's cancellation while the caller is healthy is an
ordinary per-item error.

Leaf module: stdlib only, imported by core/, knowledge/ and gui/.
"""
from __future__ import annotations

import asyncio
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple


class GatherOutcome(NamedTuple):
    """One position of a gather result list."""

    index: int
    value: Any
    error: Optional[BaseException]


def _caller_is_cancelling() -> bool:
    try:
        task = asyncio.current_task()
    except RuntimeError:  # no running loop (sync caller in a test)
        return False
    return task is not None and task.cancelling() > 0


def classify_gather_results(results: Sequence[Any]) -> List[GatherOutcome]:
    """Positional classification: ``(index, value, error)`` per result.

    ``error`` is set for EVERY ``BaseException`` result (``value`` is then
    ``None``); otherwise ``value`` is the result as returned (which may itself
    be ``None`` — a legitimate "no result" from the child).

    Raises the results' ``CancelledError`` when the calling task is being
    cancelled (see module docstring).
    """
    outcomes: List[GatherOutcome] = []
    cancelled: List[BaseException] = []
    for index, result in enumerate(results):
        if isinstance(result, BaseException):
            outcomes.append(GatherOutcome(index, None, result))
            if isinstance(result, asyncio.CancelledError):
                cancelled.append(result)
        else:
            outcomes.append(GatherOutcome(index, result, None))
    if cancelled and _caller_is_cancelling():
        raise cancelled[0]
    return outcomes


def partition_gather_results(results: Sequence[Any]) -> Tuple[List[Any], List[BaseException]]:
    """``(values, errors)`` — values in order, every BaseException in errors.

    Positions are not preserved; use :func:`classify_gather_results` when a
    site must align results with the task list it built.
    """
    outcomes = classify_gather_results(results)
    values = [o.value for o in outcomes if o.error is None]
    errors = [o.error for o in outcomes if o.error is not None]
    return values, errors
