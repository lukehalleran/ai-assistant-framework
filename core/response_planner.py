"""
Structured Response Planning + Post-Answer Review Gate.

Pre-answer: lightweight LLM call produces a ResponsePlan (key points,
tone, strategy, avoid) from query + context signals.  The plan is
injected into the system prompt so the main LLM follows it.

Post-answer: lightweight LLM call checks whether the response
adequately followed the plan.  If it didn't with high confidence, the
caller (gui/handlers.py) retries via agentic search.

Both calls are advisory — failures return None and never block.

Inputs:
    - model_manager (generate_once)
    - ContextResult from context_pipeline

Outputs:
    - ResponsePlan (Pydantic, or None)
    - ReviewResult (Pydantic, or None)

Side effects:
    - Two LLM calls (~200 tokens each) per non-small-talk, non-crisis query

Config (config/app_config.py):
    RESPONSE_PLANNING_ENABLED, RESPONSE_PLANNING_MODEL,
    RESPONSE_PLANNING_MAX_TOKENS, RESPONSE_PLANNING_TIMEOUT,
    RESPONSE_REVIEW_ENABLED, RESPONSE_REVIEW_MODEL,
    RESPONSE_REVIEW_MAX_TOKENS, RESPONSE_REVIEW_CONFIDENCE_THRESHOLD,
    RESPONSE_REVIEW_TIMEOUT
"""

import asyncio
import json
from typing import List, Optional

from pydantic import BaseModel, Field

from utils.logging_utils import get_logger

logger = get_logger("response_planner")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ResponsePlan(BaseModel):
    """Pre-answer response plan produced by the planner LLM call."""
    key_points: List[str] = Field(default_factory=list, description="2-4 things the response must cover")
    tone: str = Field(default="neutral", description="Single word: warm, analytical, empathetic, casual, etc.")
    avoid: List[str] = Field(default_factory=list, description="1-2 things to avoid")
    strategy: str = Field(default="", description="One sentence approach description")
    raw_llm_output: str = Field(default="", description="Raw LLM output for debugging")


class ReviewResult(BaseModel):
    """Post-answer review result from the review gate LLM call."""
    passes: bool = Field(default=True, description="Whether the response passes review")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Review confidence")
    issues: List[str] = Field(default_factory=list, description="Specific problems found")
    suggestion: str = Field(default="", description="How to improve")


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class ResponsePlanner:
    """
    Lightweight pre-answer planning and post-answer review.

    create_plan() runs in parallel with build_prompt_from_context() in
    the orchestrator.  review_answer() runs after streaming completes
    in gui/handlers.py.
    """

    def __init__(self, model_manager):
        self.model_manager = model_manager

    # ------------------------------------------------------------------
    # Bypass logic
    # ------------------------------------------------------------------

    @staticmethod
    def should_plan(context) -> bool:
        """Return False for small-talk, crisis, or when disabled by config."""
        try:
            from config.app_config import RESPONSE_PLANNING_ENABLED
            if not RESPONSE_PLANNING_ENABLED:
                return False
        except ImportError:
            return False

        # Skip small-talk (set by IntentClassifier CASUAL_SOCIAL)
        if getattr(context, "is_small_talk", False):
            return False
        qa = getattr(context, "query_analysis", None)
        if qa and getattr(qa, "is_small_talk", False):
            return False

        # Skip crisis / elevated tone
        tone = getattr(context, "tone_level", None)
        if tone is not None:
            from core.context_pipeline import ToneLevel
            if tone in (ToneLevel.CRISIS, ToneLevel.ELEVATED):
                return False

        # Skip for casual social intent
        intent = getattr(context, "intent", None)
        if intent and hasattr(intent, "intent"):
            from core.intent_classifier import IntentType
            if intent.intent == IntentType.CASUAL_SOCIAL:
                return False

        # Skip for very short queries
        query = getattr(context, "original_query", "") or ""
        if len(query.split()) < 8:
            return False

        return True

    # ------------------------------------------------------------------
    # Pre-answer planning
    # ------------------------------------------------------------------

    async def create_plan(self, query: str, context) -> Optional[ResponsePlan]:
        """
        Generate a response plan from query + context signals.

        Returns None on any failure (LLM error, timeout, bad JSON).
        """
        try:
            from config.app_config import (
                RESPONSE_PLANNING_MODEL,
                RESPONSE_PLANNING_MAX_TOKENS,
                RESPONSE_PLANNING_TIMEOUT,
            )
        except ImportError:
            RESPONSE_PLANNING_MODEL = None
            RESPONSE_PLANNING_MAX_TOKENS = 200
            RESPONSE_PLANNING_TIMEOUT = 5.0

        # Extract context signals
        intent_type = "unknown"
        intent_obj = getattr(context, "intent", None)
        if intent_obj and hasattr(intent_obj, "intent_type"):
            intent_type = str(intent_obj.intent_type.value) if hasattr(intent_obj.intent_type, "value") else str(intent_obj.intent_type)

        tone_level = "CONVERSATIONAL"
        tone = getattr(context, "tone_level", None)
        if tone is not None:
            tone_level = tone.value if hasattr(tone, "value") else str(tone)

        topics = getattr(context, "topics", []) or []
        topics_str = ", ".join(topics[:5]) if topics else "none"

        thread_ctx = getattr(context, "thread_context", None)
        thread_depth = thread_ctx.get("thread_depth", 0) if thread_ctx else 0

        prompt = (
            "You are a response planner. Given the query and context signals below, "
            "produce a JSON response plan.\n\n"
            f"Query: {query}\n"
            f"Intent: {intent_type}\n"
            f"Tone level: {tone_level}\n"
            f"Topics: {topics_str}\n"
            f"Thread depth: {thread_depth}\n\n"
            "Output ONLY valid JSON with these fields:\n"
            '- "key_points": list of 2-4 strings (what the response must cover)\n'
            '- "tone": single word (warm, analytical, empathetic, casual, direct, etc.)\n'
            '- "avoid": list of 1-2 strings (things to avoid)\n'
            '- "strategy": one sentence describing the approach\n\n'
            "JSON:"
        )

        try:
            raw = await asyncio.wait_for(
                self.model_manager.generate_once(
                    prompt,
                    model_name=RESPONSE_PLANNING_MODEL,
                    system_prompt="You are a concise response planner. Output only valid JSON.",
                    max_tokens=RESPONSE_PLANNING_MAX_TOKENS,
                    temperature=0.3,
                ),
                timeout=RESPONSE_PLANNING_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.debug("[RESPONSE PLANNER] Timed out, skipping plan")
            return None
        except Exception as e:
            logger.debug(f"[RESPONSE PLANNER] LLM call failed: {e}")
            return None

        if not raw or not raw.strip():
            return None

        return self._parse_plan(raw)

    # ------------------------------------------------------------------
    # Post-answer review
    # ------------------------------------------------------------------

    async def review_answer(
        self,
        plan: ResponsePlan,
        response: str,
        query: str,
    ) -> Optional[ReviewResult]:
        """
        Review a response against its plan.

        Returns None on any failure.
        """
        try:
            from config.app_config import (
                RESPONSE_REVIEW_MODEL,
                RESPONSE_REVIEW_MAX_TOKENS,
                RESPONSE_REVIEW_TIMEOUT,
            )
        except ImportError:
            RESPONSE_REVIEW_MODEL = None
            RESPONSE_REVIEW_MAX_TOKENS = 200
            RESPONSE_REVIEW_TIMEOUT = 5.0

        plan_summary = (
            f"Key points: {'; '.join(plan.key_points)}\n"
            f"Tone: {plan.tone}\n"
            f"Avoid: {'; '.join(plan.avoid)}\n"
            f"Strategy: {plan.strategy}"
        )

        # Truncate response for review (first 500 chars)
        response_excerpt = response[:500]
        if len(response) > 500:
            response_excerpt += "..."

        prompt = (
            "You are a response reviewer. Check if the response adequately addresses "
            "the plan.\n\n"
            f"Original query: {query}\n\n"
            f"Plan:\n{plan_summary}\n\n"
            f"Response (excerpt):\n{response_excerpt}\n\n"
            "Output ONLY valid JSON with these fields:\n"
            '- "passes": true if the response adequately addresses the plan, false otherwise\n'
            '- "confidence": 0.0 to 1.0 (how confident you are in this judgment)\n'
            '- "issues": list of strings (specific problems, empty if passes)\n'
            '- "suggestion": string (how to improve, empty if passes)\n\n'
            "JSON:"
        )

        try:
            raw = await asyncio.wait_for(
                self.model_manager.generate_once(
                    prompt,
                    model_name=RESPONSE_REVIEW_MODEL,
                    system_prompt="You are a strict response reviewer. Output only valid JSON.",
                    max_tokens=RESPONSE_REVIEW_MAX_TOKENS,
                    temperature=0.1,
                ),
                timeout=RESPONSE_REVIEW_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.debug("[REVIEW GATE] Timed out, skipping review")
            return None
        except Exception as e:
            logger.debug(f"[REVIEW GATE] LLM call failed: {e}")
            return None

        if not raw or not raw.strip():
            return None

        return self._parse_review(raw)

    # ------------------------------------------------------------------
    # System prompt injection
    # ------------------------------------------------------------------

    @staticmethod
    def format_plan_injection(plan: ResponsePlan) -> str:
        """Format plan as a system prompt section string."""
        points = "\n".join(f"  - {p}" for p in plan.key_points) if plan.key_points else "  - (none)"
        avoids = "\n".join(f"  - {a}" for a in plan.avoid) if plan.avoid else "  - (none)"
        return (
            "\n\n[RESPONSE PLAN]\n"
            "Based on query analysis, your response should:\n"
            f"Cover:\n{points}\n"
            f"Tone: {plan.tone}\n"
            f"Avoid:\n{avoids}\n"
            f"Strategy: {plan.strategy}\n"
            "Follow this plan while remaining natural. "
            "Do not mention this plan in your response."
        )

    # ------------------------------------------------------------------
    # JSON parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_plan(raw: str) -> Optional[ResponsePlan]:
        """Parse LLM output into ResponsePlan, returning None on failure."""
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
            plan = ResponsePlan(
                key_points=data.get("key_points", []),
                tone=data.get("tone", "neutral"),
                avoid=data.get("avoid", []),
                strategy=data.get("strategy", ""),
                raw_llm_output=raw,
            )
            logger.debug(f"[RESPONSE PLANNER] Plan created: {len(plan.key_points)} points, tone={plan.tone}")
            return plan
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"[RESPONSE PLANNER] Failed to parse plan JSON: {e}")
            return None

    @staticmethod
    def _parse_review(raw: str) -> Optional[ReviewResult]:
        """Parse LLM output into ReviewResult, returning None on failure."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
            return ReviewResult(
                passes=bool(data.get("passes", True)),
                confidence=float(data.get("confidence", 0.0)),
                issues=data.get("issues", []),
                suggestion=data.get("suggestion", ""),
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"[REVIEW GATE] Failed to parse review JSON: {e}")
            return None
