"""First-token deadline for streamed LLM answers (oct27 wave 2, lane L1).

Incident: AGT turn 10-24 spent 27 s on 321 tokens (7 s for the same shape)
because the provider stalled before the first chunk; the only guard was the
120 s per-chunk STREAM_TIMEOUT.  A stall BEFORE any chunk is safe to retry
(nothing reached the user); a stall after a chunk is not.

``guard_first_chunk`` waits for the first chunk of an already-opened stream
under ``timeout_s``.  On a stall it closes the stream, re-opens ONCE through
``reopen`` and waits again; a second stall raises ``FirstTokenTimeout`` (the
caller fails exactly as it does for any other stream error).  Once a chunk has
arrived the rest of the stream is passed through untouched (no retry).

class: BC-41 (a provider stall holding the turn), BC-69 (unbounded wait).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Awaitable, Callable

logger = logging.getLogger(__name__)


class FirstTokenTimeout(asyncio.TimeoutError):
    """No chunk arrived within the deadline, on the first try and on the retry."""


_DONE = object()


async def _close_quietly(stream: Any) -> None:
    for name in ("aclose", "close"):
        fn = getattr(stream, name, None)
        if fn is None:
            continue
        try:
            res = fn()
            if asyncio.iscoroutine(res):
                await asyncio.wait_for(res, timeout=2.0)
        except Exception:  # degrades: abandoned stalled stream is left to the GC
            pass
        return


async def _first(it: Any, timeout_s: float):
    try:
        return await asyncio.wait_for(it.__anext__(), timeout=timeout_s)
    except StopAsyncIteration:
        return _DONE


async def guard_first_chunk(
    stream: Any,
    reopen: Callable[[], Awaitable[Any]],
    timeout_s: float,
    *,
    label: str = "stream",
) -> AsyncIterator[Any]:
    """Return an async iterator over ``stream`` guarded by a first-chunk deadline.

    ``timeout_s <= 0`` disables the guard (the stream is returned as an iterator
    unchanged).  The returned iterator yields the buffered first chunk, then the
    remainder of the (possibly re-opened) stream.
    """
    it = stream.__aiter__()
    if not timeout_s or timeout_s <= 0:
        return it
    try:
        first = await _first(it, timeout_s)
    except asyncio.TimeoutError:
        logger.warning(
            "[FirstToken] %s: no chunk within %.1fs - closing and retrying once", label, timeout_s
        )
        await _close_quietly(stream)
        stream = await reopen()
        if not hasattr(stream, "__aiter__"):
            raise FirstTokenTimeout(f"{label}: retry did not return a stream")
        it = stream.__aiter__()
        try:
            first = await _first(it, timeout_s)
        except asyncio.TimeoutError:
            await _close_quietly(stream)
            logger.error("[FirstToken] %s: second stall within %.1fs - failing", label, timeout_s)
            raise FirstTokenTimeout(f"{label}: no first chunk after retry ({timeout_s:.1f}s)") from None

    async def _chain():
        if first is _DONE:
            return
        yield first
        async for item in it:
            yield item

    return _chain().__aiter__()
