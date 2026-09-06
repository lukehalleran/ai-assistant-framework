"""The actual streaming watchdog must bound a silent provider, too."""

import asyncio

import pytest

from core.insight.streaming import synthesis_events


@pytest.mark.asyncio
async def test_silent_provider_is_cancelled_at_deadline():
    closed = asyncio.Event()

    async def silent():
        try:
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            closed.set()

    events = [e async for e in synthesis_events(silent(), heartbeat_s=0.01, max_seconds=0.035)]
    assert events[-1] == {"kind": "runaway", "reason": "duration"}
    assert any(e["kind"] == "progress" for e in events)
    assert closed.is_set()


@pytest.mark.asyncio
async def test_normal_stream_preserves_chunks_and_closes():
    closed = []

    async def normal():
        try:
            yield "Useful "
            yield "answer."
        finally:
            closed.append(True)

    events = [e async for e in synthesis_events(normal())]
    assert "".join(e["value"] for e in events) == "Useful answer."
    assert closed == [True]


@pytest.mark.asyncio
async def test_consumer_cancel_closes_pending_provider_read():
    closed = asyncio.Event()

    async def silent():
        try:
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            closed.set()

    events = synthesis_events(silent(), heartbeat_s=0.005)
    assert (await anext(events))["kind"] == "progress"
    await events.aclose()
    assert closed.is_set()
