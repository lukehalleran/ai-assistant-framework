"""POST /api/shutdown (Quit) and POST /api/heartbeat. See api/lifecycle.py.

Both sit under /api/, so LaunchAuthMiddleware already requires the launch
token + same-origin Origin. /api/shutdown additionally refuses any
non-loopback PEER (a trusted tailnet Host may use the SPA, not kill the host).
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from api.lifecycle import is_loopback_client, request_clean_shutdown

router = APIRouter(prefix="/api", tags=["lifecycle"])


@router.post("/heartbeat")
async def heartbeat(request: Request):
    request.app.state.lifecycle.heartbeat()
    # quit_allowed lets the SPA hide Quit for a client /api/shutdown would 403
    # (same peer test as the shutdown route below).
    peer = request.client.host if request.client else None
    return {"ok": True, "quit_allowed": is_loopback_client(peer)}


@router.post("/shutdown")
async def shutdown(request: Request):
    peer = request.client.host if request.client else None
    if not is_loopback_client(peer):
        raise HTTPException(status_code=403, detail="shutdown is loopback-only")
    state = request.app.state.lifecycle
    busy = state.inflight()
    started = request_clean_shutdown(state, "POST /api/shutdown")
    if not started:
        return JSONResponse({"status": "already_shutting_down"}, status_code=202)
    body = {"status": "shutting_down"}
    if busy:
        body["turn_in_flight"] = True
    return JSONResponse(body, status_code=202)
