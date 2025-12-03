"""
utils/emotional_context.py

Combines tone detection (severity) and need detection (type) into unified context.
Used by orchestrator to determine response strategy.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
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


async def analyze_emotional_context(
    message: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    model_manager=None
) -> EmotionalContext:
    """
    Unified emotional analysis combining severity and need-type.

    Args:
        message: User message to analyze
        conversation_history: Recent conversation turns (optional)
        model_manager: Optional model manager for embedder/LLM access

    Returns:
        EmotionalContext with both crisis level and need type
    """
    # Get tone analysis (async)
    tone = await detect_crisis_level(message, conversation_history, model_manager)

    # Get need analysis (sync, but fast)
    need = detect_need_type(message, model_manager)

    return EmotionalContext(
        crisis_level=tone.level,
        need_type=need.need_type,
        tone_confidence=tone.confidence,
        need_confidence=need.confidence,
        tone_trigger=tone.trigger,
        need_trigger=need.trigger,
        explanation=f"{tone.explanation} | {need.explanation}"
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
