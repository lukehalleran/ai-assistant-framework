"""First-run Setup routes (2026-10-12, B2).

Thin HTTP layer over gui/setup_core.py. Behind LaunchAuthMiddleware like every
/api route. Keys are never echoed or logged: responses carry only a masked
last-4 hint. Blocking work (the OpenRouter key check) runs in a worker thread.
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from gui import setup_core
from utils.logging_utils import get_logger

logger = get_logger("api_routes")

router = APIRouter(prefix="/api/setup", tags=["setup"])


class OpenRouterKeyBody(BaseModel):
    key: str
    save_unverified: bool = False


class KeyBody(BaseModel):
    key: str


class ProfileBody(BaseModel):
    name: str
    pronouns: Optional[str] = None
    style: Optional[str] = None


class ObsidianBody(BaseModel):
    path: Optional[str] = None


def _orchestrator(request: Request):
    return request.app.state.daemon.orchestrator


@router.get("/status")
async def get_status(request: Request):
    return setup_core.get_status(_orchestrator(request))


@router.post("/openrouter")
async def post_openrouter(request: Request, body: OpenRouterKeyBody):
    # default_http_get is looked up at call time (tests replace it with a fake).
    return await asyncio.to_thread(
        setup_core.save_openrouter_key, _orchestrator(request), body.key,
        http_get=setup_core.default_http_get, save_unverified=body.save_unverified,
    )


@router.post("/key/{name}")
async def post_optional_key(request: Request, name: str, body: KeyBody):
    if name not in setup_core.OPTIONAL_KEYS:
        raise HTTPException(status_code=404, detail="Unknown key.")
    return await asyncio.to_thread(
        setup_core.save_optional_key, _orchestrator(request), name, body.key)


@router.post("/profile")
async def post_profile(request: Request, body: ProfileBody):
    return await asyncio.to_thread(
        setup_core.save_profile, _orchestrator(request), body.name, body.pronouns, body.style)


@router.post("/obsidian")
async def post_obsidian(body: ObsidianBody):
    return await asyncio.to_thread(setup_core.save_obsidian_path, body.path)


@router.post("/complete")
async def post_complete(request: Request):
    return await asyncio.to_thread(setup_core.complete_setup, _orchestrator(request))
