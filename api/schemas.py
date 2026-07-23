"""Pydantic request/response models for the Daemon API.

Contracts:
- ChatRequest: POST /api/chat body. `file_ids` reference prior /api/uploads results.
- ChatEvent: one SSE message on the /api/chat stream. `event` selects the SSE
  event name; `data` is the JSON payload. Event names and payloads:
    progress      {"text": str}                       transient status line
    thinking      {"text": str}                       thinking indicator/content
    duel_thinking {"thinking_a","thinking_b","model_a","model_b","winner","scores"}
    message       {"content": str}                    CUMULATIVE assistant text (replace-render)
    complete      {"content","pending_action_id","debug","turn_index"}
    error         {"message": str}
- ActionDecisionResponse: approve/reject result + the chat line that was appended.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from core.actions.types import ActionOutcome


class ChatRequest(BaseModel):
    text: str
    raw_mode: bool = False
    fast_mode: bool = False
    enable_citations: bool = False
    file_ids: List[str] = Field(default_factory=list)


class ChatEvent(BaseModel):
    event: Literal["progress", "thinking", "duel_thinking", "message", "complete", "error"]
    data: Dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionState(BaseModel):
    history: List[ChatMessage] = Field(default_factory=list)
    pending_action_id: Optional[str] = None
    personality: str = "default"


class UploadedFileInfo(BaseModel):
    file_id: str
    name: str
    size: int


class UploadResult(BaseModel):
    files: List[UploadedFileInfo] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    models: List[str] = Field(default_factory=list)
    active: Optional[str] = None


class ActiveModelRequest(BaseModel):
    name: str


class ActionDecisionResponse(BaseModel):
    outcome: ActionOutcome
    message: ChatMessage


# ---- Debug / Provenance (2026-07-14: Gradio dev tabs → SPA) ----------------

class DebugRecordsResponse(BaseModel):
    """Server-held per-turn debug records (survive SPA reloads)."""
    records: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0


# ---- Settings (2026-07-14: Gradio Settings tab → SPA) ----------------------

class StreamingSettings(BaseModel):
    disable_best_of: bool = False
    disable_query_rewrite: bool = False
    disable_llm_summaries: bool = False
    best_of_latency_budget_s: float = Field(default=0.0, ge=0, le=120)


class WebSearchSettings(BaseModel):
    enabled: bool = True
    daily_credit_limit: int = Field(default=100, ge=10, le=500)


class DuelSettings(BaseModel):
    enabled: bool = False
    model_1: Optional[str] = None
    model_2: Optional[str] = None


class TokenSettings(BaseModel):
    best_of_max_tokens: int = Field(default=128, ge=16, le=10000)
    judge_max_tokens: int = Field(default=64, ge=16, le=2000)
    streaming_max_tokens: int = Field(default=2048, ge=256, le=10000)


class TemperatureSettings(BaseModel):
    temperature: float = Field(ge=0.0, le=1.5)


class SummaryCadenceSettings(BaseModel):
    every_n: int = Field(ge=1, le=200)


class SynthesisSettings(BaseModel):
    """Synthesis dreaming — LLM candidate generation at shutdown."""
    enabled: bool = False
    candidates_per_session: int = Field(default=8, ge=1, le=20)


class ProposalsSettings(BaseModel):
    """Goal-directed code proposals — LLM generation at shutdown."""
    enabled: bool = True
    max_per_session: int = Field(default=5, ge=1, le=10)


class SettingsSnapshot(BaseModel):
    streaming: StreamingSettings
    web_search: WebSearchSettings
    duel: DuelSettings
    tokens: TokenSettings
    temperature: float
    summary_every_n: int
    synthesis: SynthesisSettings
    proposals: ProposalsSettings
    model_choices: List[str] = Field(default_factory=list)


class SettingsApplyResult(BaseModel):
    """Mirror of gui.settings_core result dicts. ok=True/persisted=False means
    the runtime change applied but the config.yaml write failed."""
    ok: bool
    persisted: bool
    message: str
