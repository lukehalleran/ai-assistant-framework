"""Curation Center API (docs/AUTONOMOUS_CURATION_DESIGN.md).

Queue review + activity/undo over the in-process CurationEngine. Every
mutation here is human-initiated (the one-click surface that replaces the
terminal candidate-file + --apply workflow). The engine itself enforces the
safety rules (reversible instruments, pre-images, journal).
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from utils.logging_utils import get_logger

logger = get_logger("api_routes")

router = APIRouter(prefix="/api/curation", tags=["curation"])


def _engine(request: Request):
    from memory.curation.service import get_engine, init_engine

    engine = get_engine()
    if engine is None:
        state = request.app.state.daemon
        ms = getattr(state.orchestrator, "memory_system", None)
        if ms is not None:
            engine = init_engine(
                chroma_store=getattr(ms, "chroma_store", None),
                user_profile=getattr(ms, "user_profile", None),
                corpus_manager=getattr(ms, "corpus_manager", None),
                graph_memory=getattr(ms, "graph_memory", None),
            )
    if engine is None:
        raise HTTPException(status_code=503, detail="curation disabled")
    return engine


class DismissBody(BaseModel):
    reason: str = ""


async def _run_operation(method, *args, timeout=None, **kwargs):
    """The worker owns the lock; cancelling its HTTP wait cannot release it."""
    try:
        worker = asyncio.to_thread(method, *args, **kwargs)
        if timeout is None:
            return await worker
        return await asyncio.wait_for(worker, timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Scan wait timed out; scan is still running")
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown proposal")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logger.error(f"[Curation API] operation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/queue")
async def queue(request: Request):
    engine = _engine(request)
    return {
        "proposals": [p.model_dump(mode="json") for p in await _run_operation(engine.pending)],
        "max_mode": engine.max_mode.value,
    }


@router.post("/scan")
async def scan_now(request: Request):
    """On-demand scan (same code path as the shutdown phase)."""
    from config.app_config import CURATION_SCAN_TIMEOUT_S

    engine = _engine(request)
    report = await _run_operation(engine.run_scan, timeout=CURATION_SCAN_TIMEOUT_S)
    return report.model_dump(mode="json")


@router.post("/{proposal_id}/apply")
async def apply(proposal_id: str, request: Request):
    engine = _engine(request)
    p = await _run_operation(engine.apply, proposal_id, actor="human")
    return p.model_dump(mode="json")


@router.post("/{proposal_id}/dismiss")
async def dismiss(proposal_id: str, request: Request, body: Optional[DismissBody] = None):
    engine = _engine(request)
    p = await _run_operation(engine.dismiss, proposal_id, reason=(body.reason if body else ""))
    return p.model_dump(mode="json")


@router.post("/{proposal_id}/undo")
async def undo(proposal_id: str, request: Request):
    engine = _engine(request)
    p = await _run_operation(engine.undo, proposal_id)
    return p.model_dump(mode="json")


@router.get("/activity")
async def activity(request: Request, limit: int = 100):
    engine = _engine(request)
    # A count is never meaningful when negative; normalize it before passing
    # it to journal implementations, whose negative slicing semantics differ.
    return {"events": await _run_operation(engine.journal.tail, limit=min(max(limit, 0), 500))}
