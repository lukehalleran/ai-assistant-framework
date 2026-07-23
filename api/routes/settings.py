"""Settings routes (2026-07-14: Gradio Settings tab → SPA).

Thin HTTP layer over gui/settings_core.py — the SAME deployed apply functions
the Gradio tab calls (no forked settings logic). Each PUT mutates the live
orchestrator and persists to config/config.yaml.

Semantics:
- 400: validation error (nothing changed) — settings_core returned ok=False.
- 200 with persisted=False: runtime change applied, YAML write failed.
"""

from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    DuelSettings,
    ProposalsSettings,
    SettingsApplyResult,
    SettingsSnapshot,
    StreamingSettings,
    SummaryCadenceSettings,
    SynthesisSettings,
    TemperatureSettings,
    TokenSettings,
    WebSearchSettings,
)
from gui import settings_core
from utils.logging_utils import get_logger

logger = get_logger("api_routes")

router = APIRouter(prefix="/api/settings", tags=["settings"])


def _orchestrator(request: Request):
    return request.app.state.daemon.orchestrator


def _to_response(result: dict) -> SettingsApplyResult:
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("message", "Failed to apply."))
    return SettingsApplyResult(**result)


@router.get("", response_model=SettingsSnapshot)
async def get_settings(request: Request):
    return SettingsSnapshot(**settings_core.get_settings_snapshot(_orchestrator(request)))


@router.put("/streaming", response_model=SettingsApplyResult)
async def put_streaming(request: Request, body: StreamingSettings):
    return _to_response(settings_core.apply_streaming(
        _orchestrator(request),
        disable_best_of=body.disable_best_of,
        disable_query_rewrite=body.disable_query_rewrite,
        disable_llm_summaries=body.disable_llm_summaries,
        best_of_latency_budget_s=body.best_of_latency_budget_s,
    ))


@router.put("/web-search", response_model=SettingsApplyResult)
async def put_web_search(request: Request, body: WebSearchSettings):
    return _to_response(settings_core.apply_web_search(
        _orchestrator(request),
        enabled=body.enabled,
        daily_credit_limit=body.daily_credit_limit,
    ))


@router.put("/duel", response_model=SettingsApplyResult)
async def put_duel(request: Request, body: DuelSettings):
    return _to_response(settings_core.apply_duel(
        _orchestrator(request),
        enabled=body.enabled,
        model_1=body.model_1 or "",
        model_2=body.model_2 or "",
    ))


@router.put("/tokens", response_model=SettingsApplyResult)
async def put_tokens(request: Request, body: TokenSettings):
    return _to_response(settings_core.apply_tokens(
        _orchestrator(request),
        best_of_max_tokens=body.best_of_max_tokens,
        judge_max_tokens=body.judge_max_tokens,
        streaming_max_tokens=body.streaming_max_tokens,
    ))


@router.put("/temperature", response_model=SettingsApplyResult)
async def put_temperature(request: Request, body: TemperatureSettings):
    return _to_response(settings_core.apply_temperature(
        _orchestrator(request), temperature=body.temperature,
    ))


@router.put("/summary-cadence", response_model=SettingsApplyResult)
async def put_summary_cadence(request: Request, body: SummaryCadenceSettings):
    return _to_response(settings_core.apply_summary_cadence(
        _orchestrator(request), every_n=body.every_n,
    ))


@router.put("/synthesis", response_model=SettingsApplyResult)
async def put_synthesis(request: Request, body: SynthesisSettings):
    return _to_response(settings_core.apply_synthesis(
        _orchestrator(request),
        enabled=body.enabled,
        candidates_per_session=body.candidates_per_session,
    ))


@router.put("/proposals", response_model=SettingsApplyResult)
async def put_proposals(request: Request, body: ProposalsSettings):
    return _to_response(settings_core.apply_proposals(
        _orchestrator(request),
        enabled=body.enabled,
        max_per_session=body.max_per_session,
    ))
