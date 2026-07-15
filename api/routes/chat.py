"""Chat routes: POST /api/chat (SSE stream), GET/DELETE /api/session."""

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


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    state = _state(request)
    session = state.session

    if session.stream_lock.locked():
        raise HTTPException(status_code=409, detail="A chat turn is already streaming.")

    async def event_stream():
        async with session.stream_lock:
            async for event in submit_stream(req, state):
                yield format_sse(event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


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
    _state(request).session.clear()
    return None
