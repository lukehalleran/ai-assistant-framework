"""Model listing + active-model switch (mirrors the Gradio selector in gui/launch.py)."""

from fastapi import APIRouter, HTTPException, Request

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

    # Persist to config.yaml (same best-effort pattern as the Gradio selector)
    try:
        import yaml
        from pathlib import Path
        cfg_path = Path("config") / "config.yaml"
        data = {}
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        data.setdefault("models", {})["active"] = name
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    except Exception as e:
        logger.warning(f"[API] Model switch persisted-to-yaml failed (runtime switch OK): {e}")

    return await list_models(request)
