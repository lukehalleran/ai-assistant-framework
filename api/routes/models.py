"""Model listing + active-model switch (mirrors the Gradio selector in gui/launch.py)."""

from fastapi import APIRouter, HTTPException, Request

from gui import settings_core
from api.schemas import ActiveModelRequest, ModelListResponse
from utils.logging_utils import get_logger

logger = get_logger("api_routes")

router = APIRouter(prefix="/api/models", tags=["models"])


def _model_manager(request: Request):
    return request.app.state.daemon.orchestrator.model_manager


@router.get("", response_model=ModelListResponse)
async def list_models(request: Request):
    mm = _model_manager(request)
    api_aliases = list(getattr(mm, "api_models", {}).keys())
    local_models = list(getattr(mm, "models", {}).keys())
    active = None
    try:
        active = mm.get_active_model_name()
    except Exception:
        pass
    choices = sorted(set(api_aliases + local_models)) or ([active] if active else [])
    return ModelListResponse(models=choices, active=active)


@router.put("/active", response_model=ModelListResponse)
async def set_active_model(req: ActiveModelRequest, request: Request):
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="No model name given.")
    mm = _model_manager(request)
    try:
        mm.switch_model(name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to switch model: {e}")

    # 2026-09-30 (BC-26/BC-37): persist to config.local.yaml via the same writer
    # Settings uses — never rewrite the committed config.yaml.
    try:
        ok, err = settings_core.save_settings(
            lambda d: d.setdefault("models", {}).update({"active": name}))
        if not ok:
            logger.warning(f"[API] Model switch persist failed (runtime switch OK): {err}")
    except Exception as e:
        logger.warning(f"[API] Model switch persisted-to-yaml failed (runtime switch OK): {e}")

    return await list_models(request)
