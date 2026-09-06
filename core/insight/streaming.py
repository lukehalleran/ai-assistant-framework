"""Bound insight synthesis even when the provider stops sending chunks."""

import asyncio
import time

from core.response_parser import ResponseParser


async def synthesis_events(stream, *, heartbeat_s=8.0, max_seconds=240.0):
    """Yield text/progress/runaway events; always cancel outstanding reads."""
    iterator = stream.__aiter__()
    started = time.monotonic()
    deadline = started + max_seconds
    accumulated = ""
    checked_at = 0
    pending = None
    try:
        while True:
            pending = asyncio.ensure_future(iterator.__anext__())
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    yield {"kind": "runaway", "reason": "duration"}
                    return
                done, _ = await asyncio.wait({pending}, timeout=min(heartbeat_s, remaining))
                if done:
                    break
                if time.monotonic() >= deadline:
                    yield {"kind": "runaway", "reason": "duration"}
                    return
                yield {"kind": "progress", "seconds": int(time.monotonic() - started)}
            try:
                piece = pending.result()
            except StopAsyncIteration:
                return
            if time.monotonic() >= deadline:
                yield {"kind": "runaway", "reason": "duration"}
                return
            accumulated += piece
            if len(accumulated) - checked_at > 2000:
                checked_at = len(accumulated)
                if ResponseParser.looks_degenerate_stream(accumulated):
                    yield {"kind": "runaway", "reason": "degenerate"}
                    return
            yield {"kind": "text", "value": piece}
    finally:
        if pending is not None:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()
