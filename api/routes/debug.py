"""Debug/Provenance routes (2026-07-14: Gradio dev tabs → SPA).

Serves the server-held per-turn debug records that chat_service appends to
session.debug_records — the same dicts gui/handlers._build_debug_record()
produces (query, full prompt, response, tokens, timings, provenance).
Read-only; records are UI state and clear with DELETE /api/session.

System prompts are dev-mode-gated at the SOURCE (handlers only include
system_prompt when DAEMON_MODE == "dev"), and the prompt export re-checks
DAEMON_MODE to mirror the Gradio download button exactly.

Privacy boundary (2026-09-02): everything these routes return — records,
provenance, the prompt export — passes through utils.privacy_redaction, because
exports get pasted into other models. The server-held records stay raw; only
the share surface is redacted.
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

from api.schemas import DebugRecordsResponse
from utils.logging_utils import get_logger
from utils.privacy_redaction import build_redacted_prompt_export, redact_data

logger = get_logger("api_routes")

router = APIRouter(prefix="/api", tags=["debug"])

# Same display cap as the Gradio Provenance tab (full text is in ChromaDB).
_THINKING_DISPLAY_CAP = 500


def _records(request: Request) -> list:
    return request.app.state.daemon.session.debug_records


def _record_at(request: Request, index: int) -> dict:
    records = _records(request)
    if not records:
        raise HTTPException(status_code=404, detail="No debug records yet. Send a message first.")
    try:
        record = records[index]
    except IndexError:
        raise HTTPException(status_code=404, detail=f"No debug record at index {index}.")
    if not isinstance(record, dict):
        raise HTTPException(status_code=404, detail=f"No debug record at index {index}.")
    return record


@router.get("/debug", response_model=DebugRecordsResponse)
async def debug_records(request: Request):
    # Debug records can contain the complete recent-conversation context.  The
    # browser never needs an unredacted copy, so sanitize at the server boundary
    # rather than relying solely on clipboard UI code.
    records = [redact_data(r) for r in _records(request) if isinstance(r, dict)]
    return DebugRecordsResponse(records=records, count=len(records))


@router.get("/debug/prompt", response_class=PlainTextResponse)
async def export_prompt(request: Request, index: int = -1):
    """Full prompt of one turn as a downloadable TXT (Gradio download button)."""
    from config.app_config import DAEMON_MODE

    record = _record_at(request, index)
    content = build_redacted_prompt_export(
        record,
        include_system=DAEMON_MODE == "dev",
    )

    filename = f"daemon_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    return PlainTextResponse(
        content,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/provenance")
async def provenance(request: Request, index: int = -1):
    """Provenance view of one turn (default: latest).

    Same merge the Gradio Provenance tab renders: the record's provenance dict
    plus mode/model/citations/token counts, thinking blocks display-capped.
    """
    record = _record_at(request, index)
    prov = dict(record.get("provenance", {}) or {})
    prov["mode"] = record.get("mode", "")
    prov["model"] = record.get("model", "")
    prov["citations_enabled"] = record.get("citations_enabled", False)
    citations = record.get("citations", [])
    if citations:
        prov["citations"] = citations
    if record.get("prompt_tokens"):
        prov["prompt_tokens"] = record["prompt_tokens"]
    if record.get("total_tokens"):
        prov["total_tokens"] = record["total_tokens"]
    for key in ("thinking_block", "thinking_a", "thinking_b"):
        if key in prov and isinstance(prov[key], str) and len(prov[key]) > _THINKING_DISPLAY_CAP:
            prov[key] = prov[key][:_THINKING_DISPLAY_CAP] + " [truncated]"
    return redact_data(prov)
