"""Chat routes: POST /api/chat (SSE stream), GET/DELETE /api/session."""

import asyncio
from contextlib import aclosing

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.chat_service import submit_stream
from api.schemas import ChatRequest, SessionState
from api.sse import SSE_HEADERS, format_sse
from utils.logging_utils import get_logger

logger = get_logger("api_routes")

router = APIRouter(prefix="/api", tags=["chat"])


def _state(request: Request):
    return request.app.state.daemon


class _TurnReservation:
    """Idempotently release a stream lock reserved by the route handler."""

    def __init__(self, lock: asyncio.Lock):
        self._lock = lock
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._lock.release()


class _ReservedStreamingResponse(StreamingResponse):
    """Release admission even when response streaming is cancelled early."""

    def __init__(self, *args, reservation: _TurnReservation, **kwargs):
        super().__init__(*args, **kwargs)
        self._reservation = reservation

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                # A send failure can leave the iterator suspended at yield.
                # Finish pipeline cancellation before releasing admission.
                await self.body_iterator.aclose()
            finally:
                self._reservation.release()


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    state = _state(request)
    session = state.session

    if session.stream_lock.locked():
        raise HTTPException(status_code=409, detail="A chat turn is already streaming.")

    # asyncio.Lock.acquire() completes without yielding when the lock is free,
    # making the check-and-reserve sequence atomic within this event loop.  The
    # reservation happens before the StreamingResponse is returned, so another
    # POST cannot pass admission while this response body is still unconsumed.
    await session.stream_lock.acquire()
    reservation = _TurnReservation(session.stream_lock)

    # The response wrapper covers normal completion and cancellation after its
    # ASGI call starts.  The request-task callback also covers cancellation in
    # the narrow gap between returning the response and starting that call.
    request_task = asyncio.current_task()
    if request_task is not None:
        request_task.add_done_callback(lambda _task: reservation.release())

    async def event_stream():
        async with aclosing(submit_stream(req, state)) as events:
            async for event in events:
                yield format_sse(event)

    try:
        return _ReservedStreamingResponse(
            event_stream(),
            reservation=reservation,
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )
    except BaseException:
        reservation.release()
        raise


@router.get("/session", response_model=SessionState)
async def get_session(request: Request):
    session = _state(request).session
    return SessionState(
        history=session.history,
        pending_action_id=session.pending_action_id,
        personality=session.personality,
    )


@router.delete("/session", status_code=204)
async def clear_session(request: Request):
    # Clears in-memory UI state ONLY — stored memory (ChromaDB/corpus) is never
    # touched from this endpoint (repo rule: never auto-delete user data).
    session = _state(request).session
    if session.stream_lock.locked():
        raise HTTPException(status_code=409, detail="Cannot clear while a chat turn is streaming.")
    session.clear()
    return None
