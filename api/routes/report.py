"""Report-a-problem routes (R1, 2026-10-12): local, consented diagnostic bundle.

Thin layer over utils.problem_report.  The server builds the bundle from the
in-process debug records (+ the bounded tail of turn telemetry) and RETURNS it;
it writes nothing and sends nothing — the browser saves/copies it only when
the user clicks.  Behind LaunchAuthMiddleware like every /api route.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from config import app_config
from utils import problem_report
from utils.logging_utils import get_logger

logger = get_logger("api_routes")

router = APIRouter(prefix="/api/report", tags=["report"])


class BuildReportRequest(BaseModel):
    index: int = -1
    description: str = Field(default="", max_length=problem_report.MAX_DESCRIPTION_CHARS)
    include_full: bool = False


def _records(request: Request) -> list:
    return request.app.state.daemon.session.debug_records


def _telemetry_rows() -> list:
    # Live config read (BC-82): the path can differ per launch/profile.
    return problem_report.read_telemetry_tail(app_config.TURN_TELEMETRY_PATH)


def _active_model(request: Request):
    mm = getattr(request.app.state.daemon.orchestrator, "model_manager", None)
    getter = getattr(mm, "get_active_model_name", None)
    return getter() if callable(getter) else None


@router.get("/turns")
async def report_turns(request: Request):
    records = _records(request)
    return {"turns": problem_report.list_turns(records, _telemetry_rows() if records else [])}


@router.post("/build")
async def build_report(body: BuildReportRequest, request: Request):
    records = _records(request)
    if not records:
        raise HTTPException(status_code=404, detail="Send a message first.")
    try:
        return problem_report.build_bundle(
            records,
            body.index,
            description=body.description,
            include_full=body.include_full,
            telemetry_rows=_telemetry_rows(),
            telemetry_path=app_config.TURN_TELEMETRY_PATH,
            info=problem_report.app_info(
                model=_active_model(request), mode=app_config.DAEMON_MODE),
            include_system=app_config.DAEMON_MODE == "dev",
        )
    except IndexError:
        raise HTTPException(status_code=404, detail=f"No debug record at index {body.index}.")
