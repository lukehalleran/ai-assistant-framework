"""
utils/emotional_context.py

Combines tone detection (severity) and need detection (type) into unified context.
Used by orchestrator to determine response strategy.
"""

from dataclasses import dataclass
import math
import re
from typing import Optional, List, Dict, Any
from config import app_config
from utils.tone_detector import CrisisLevel, ToneAnalysis, detect_crisis_level
from utils.need_detector import NeedType, NeedAnalysis, detect_need_type

@dataclass
class EmotionalContext:
    """Combined emotional analysis for response calibration."""
    crisis_level: CrisisLevel
    need_type: NeedType
    tone_confidence: float
    need_confidence: float
    tone_trigger: str
    need_trigger: str
    explanation: str
    # Content-free decision-model receipt from ToneAnalysis (labels and numbers only);
    # None when the detector returned before the arbiter stage. Plan 2026-10-08 D13.
    arbiter_receipt: Optional[Dict[str, Any]] = None


# receipt key -> record field; scalar labels/numbers only (D13, BC-72: no message text, ever)
_TONE_DM_SCALARS = (
    ("mode", "tone_dm_mode"), ("status", "tone_dm_status"), ("reason", "tone_dm_reason"),
    ("level", "tone_dm_level"), ("decision_confidence", "tone_dm_decision_confidence"),
    ("policy", "tone_dm_policy"), ("retried", "tone_dm_retried"),
    ("latency_ms", "tone_dm_latency_ms"), ("served_model", "tone_dm_served_model"),
    ("provider", "tone_dm_provider"), ("cost_usd", "tone_dm_cost_usd"),
    ("agrees", "tone_dm_agrees"),
)


_TONE_DM_NUMERIC = frozenset({"decision_confidence", "latency_ms", "cost_usd"})
_DM_LABEL_RE = re.compile(r"[A-Za-z0-9_.\-/~]{1,60}")


def _dm_scalar(value: Any) -> Any:
    """A label (identifier-shaped str, no spaces), bool or finite number; anything else is
    dropped or replaced by ``invalid_label`` so free text can never ride a receipt field."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        return value if _DM_LABEL_RE.fullmatch(value) else "invalid_label"
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def tone_receipt_fields(ctx: Optional[EmotionalContext]) -> Dict[str, Any]:
    """Flatten the tone arbiter receipt into ``tone_dm_*`` + ``tone_arbiter_backend`` fields.

    Always returns the mode, a non-empty status and the backend: a turn whose arbiter never
    ran (no receipt, off mode, early return) records the resolved mode, status ``not_run`` and
    backend ``none`` rather than an empty field (BC-47). Never raises.
    """
    try:
        receipt = getattr(ctx, "arbiter_receipt", None)
        receipt = receipt if isinstance(receipt, dict) else {}
        out: Dict[str, Any] = {}
        for src, dst in _TONE_DM_SCALARS:
            if src in receipt:
                value = _dm_scalar(receipt[src])
                if src in _TONE_DM_NUMERIC and isinstance(value, str):
                    value = None  # a number field never carries a label
                out[dst] = value
        probs = receipt.get("probs")
        if isinstance(probs, (list, tuple)) and len(probs) == 4:
            for i, p in enumerate(probs):
                if isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p):
                    out[f"tone_dm_p{i}"] = round(float(p), 3)
        out["tone_dm_deciding_level"] = _dm_scalar(
            receipt.get("llm_level") or (receipt.get("level") if receipt.get("backend") == "jev" else None)
        )
        if not out.get("tone_dm_mode"):
            out["tone_dm_mode"] = app_config.decision_model_mode("tone_arbiter")
        if not out.get("tone_dm_status"):
            out["tone_dm_status"] = "not_run"
        out["tone_arbiter_backend"] = _dm_scalar(receipt.get("backend")) or "none"
        return out
    except Exception:  # degrades: the tone receipt fields are absent from this turn's record
        return {}


async def analyze_emotional_context(
    message: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    model_manager=None,
    previous_tone: Optional[object] = None,
    allow_sticky_floor: bool = True,
) -> EmotionalContext:
    """
    Unified emotional analysis combining severity and need-type.

    Args:
        message: User message to analyze
        conversation_history: Recent conversation turns (optional)
        model_manager: Optional model manager for embedder/LLM access
        previous_tone: Prior turn's tone (CrisisLevel/str); makes distress sticky
            across short turns (see tone_detector.detect_crisis_level).
        allow_sticky_floor: False when the floor-chain budget is exhausted —
            disables the distress-sticky floor stage only (organic signals
            unaffected); see tone_detector.detect_crisis_level.

    Returns:
        EmotionalContext with both crisis level and need type
    """
    # Get tone analysis (async)
    tone = await detect_crisis_level(
        message, conversation_history, model_manager, previous_tone=previous_tone,
        allow_sticky_floor=allow_sticky_floor,
    )

    # Get need analysis (sync, but fast)
    need = detect_need_type(message, model_manager)

    return EmotionalContext(
        crisis_level=tone.level,
        need_type=need.need_type,
        tone_confidence=tone.confidence,
        need_confidence=need.confidence,
        tone_trigger=tone.trigger,
        need_trigger=need.trigger,
        explanation=f"{tone.explanation} | {need.explanation}",
        arbiter_receipt=getattr(tone, "arbiter_receipt", None),
    )


def format_emotional_context_log(ctx: EmotionalContext, message: str) -> str:
    """
    Format emotional context for backend logging.

    Args:
        ctx: EmotionalContext result
        message: Original user message (truncated for privacy)

    Returns:
        Formatted log string
    """
    msg_preview = message[:50] + "..." if len(message) > 50 else message
    msg_preview = msg_preview.replace("\n", " ")

    return (
        f"EMOTIONAL_CONTEXT: Crisis={ctx.crisis_level.value} (conf={ctx.tone_confidence:.2f}, trigger={ctx.tone_trigger}), "
        f"Need={ctx.need_type.value} (conf={ctx.need_confidence:.2f}, trigger={ctx.need_trigger}) "
        f"| Message: \"{msg_preview}\""
    )
