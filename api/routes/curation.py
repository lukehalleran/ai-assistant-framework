"""Curation Center API (docs/AUTONOMOUS_CURATION_DESIGN.md).

Queue review + activity/undo over the in-process CurationEngine. Every
mutation here is human-initiated (the one-click surface that replaces the
terminal candidate-file + --apply workflow). The engine itself enforces the
safety rules (reversible instruments, pre-images, journal).
"""

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
            )
    if engine is None:
        raise HTTPException(status_code=503, detail="curation disabled")
    return engine


class DismissBody(BaseModel):
    reason: str = ""


@router.get("/queue")
async def queue(request: Request):
    engine = _engine(request)
    return {
        "proposals": [p.model_dump(mode="json") for p in engine.pending()],
        "max_mode": engine.max_mode.value,
    }


@router.post("/scan")
async def scan_now(request: Request):
    """On-demand scan (same code path as the shutdown phase)."""
    import asyncio

    from config.app_config import CURATION_SCAN_TIMEOUT_S

    engine = _engine(request)
    try:
        report = await asyncio.wait_for(
            asyncio.to_thread(engine.run_scan), timeout=CURATION_SCAN_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="scan timed out")
    return report.model_dump(mode="json")


@router.post("/{proposal_id}/apply")
async def apply(proposal_id: str, request: Request):
    engine = _engine(request)
    try:
        p = engine.apply(proposal_id, actor="human")
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown proposal")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"[Curation API] apply failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return p.model_dump(mode="json")


@router.post("/{proposal_id}/dismiss")
async def dismiss(proposal_id: str, request: Request, body: Optional[DismissBody] = None):
    engine = _engine(request)
    try:
        p = engine.dismiss(proposal_id, reason=(body.reason if body else ""))
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown proposal")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return p.model_dump(mode="json")


@router.post("/{proposal_id}/undo")
async def undo(proposal_id: str, request: Request):
    engine = _engine(request)
    try:
        p = engine.undo(proposal_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown proposal")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"[Curation API] undo failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return p.model_dump(mode="json")


@router.get("/activity")
async def activity(request: Request, limit: int = 100):
    engine = _engine(request)
    return {"events": engine.journal.tail(limit=min(limit, 500))}
