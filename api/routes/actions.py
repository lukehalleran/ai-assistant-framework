"""Pending internet-action decisions: approve / reject.

Thin wrappers over gui.handlers.execute_pending_action_core /
reject_pending_action_core — the same audited path the Gradio buttons use.
"""

from fastapi import APIRouter, Request

from api.schemas import ActionDecisionResponse, ChatMessage
from utils.logging_utils import get_logger

logger = get_logger("api_routes")

router = APIRouter(prefix="/api/actions", tags=["actions"])


def _state(request: Request):
    return request.app.state.daemon


async def _decide(request: Request, action_id: str, approve: bool) -> ActionDecisionResponse:
    from gui.handlers import execute_pending_action_core, reject_pending_action_core

    state = _state(request)
    core = execute_pending_action_core if approve else reject_pending_action_core
    outcome = await core(action_id, state.orchestrator)

    message = ChatMessage(role="assistant", content=outcome.message)
    state.session.history.append(message.model_dump())
    if state.session.pending_action_id == action_id:
        state.session.pending_action_id = None

    return ActionDecisionResponse(outcome=outcome, message=message)


@router.post("/{action_id}/approve", response_model=ActionDecisionResponse)
async def approve_action(action_id: str, request: Request):
    return await _decide(request, action_id, approve=True)


@router.post("/{action_id}/reject", response_model=ActionDecisionResponse)
async def reject_action(action_id: str, request: Request):
    return await _decide(request, action_id, approve=False)
