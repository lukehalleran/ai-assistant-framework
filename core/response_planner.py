"""
Structured Response Planning + Post-Answer Review Gate.

Pre-answer: after retrieval finishes, a lightweight LLM call produces a
ResponsePlan (key points, tone, strategy, avoid) from the query, context
signals, and a bounded digest of the same gathered context the main model
will receive.  The plan is injected into the system prompt so the main LLM
follows it.

Post-answer: lightweight LLM call checks whether the response
adequately followed the plan.  If it didn't with high confidence, the
caller (gui/handlers.py) retries via agentic search.

Both calls are advisory — failures return None and never block.

Inputs:
    - model_manager (generate_once)
    - ContextResult from context_pipeline
    - a bounded, retrieval-ranked digest derived from PromptBuilder's final
      prompt-context dictionary (contents are not copied into telemetry)

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
import hashlib
import json
import re
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
    dropped_points: List[str] = Field(default_factory=list, description="Planner key points removed by the embellishment guard (2026-09-03)")
    tone: str = Field(default="neutral", description="Single word: warm, analytical, empathetic, casual, etc.")
    avoid: List[str] = Field(default_factory=list, description="1-2 things to avoid")
    strategy: str = Field(default="", description="One sentence approach description")
    raw_llm_output: str = Field(default="", description="Raw LLM output for debugging")
    planner_source: str = Field(default="llm", description="llm or deterministic safeguard")
    planner_model: str = Field(default="", description="Model used to create the plan")
    context_digest_sha256: str = Field(
        default="", description="Hash of the exact context digest shown to the planner"
    )
    context_sections: List[str] = Field(
        default_factory=list, description="Prompt-context sections represented in the digest"
    )
    directive_locked: bool = Field(
        default=False, description="True when an explicit user speech act was deterministically preserved"
    )

    def audit_record(self) -> dict:
        """Return the exact operative plan plus non-content alignment metadata.

        Deliberately omit ``raw_llm_output``: the parsed fields below are the
        instructions actually injected, while retaining arbitrary raw model
        chatter would expand the PII surface.  The hash and section names let a
        trace establish which gathered-context digest the planner saw without
        persisting that digest a second time.
        """
        operative = {
            "key_points": list(self.key_points),
            "tone": self.tone,
            "avoid": list(self.avoid),
            "strategy": self.strategy,
        }
        canonical = json.dumps(
            operative, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return {
            **operative,
            "plan_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "planner_source": self.planner_source,
            "planner_model": self.planner_model,
            "context_digest_sha256": self.context_digest_sha256,
            "context_sections": list(self.context_sections),
            "directive_locked": self.directive_locked,
        }


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

    create_plan() runs after build_prompt_from_context() in the orchestrator,
    so it cannot plan from classifier labels while blind to retrieved context.
    review_answer() runs after streaming completes in gui/handlers.py.
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

        # Skip crisis / elevated / concern tone. CONCERN was added 2026-08-05:
        # it selects the LIGHT SUPPORT response mode ("2-4 sentences, don't
        # offer unsolicited advice"), while the planner injected a [RESPONSE
        # PLAN] with "Cover: <advice topics> / Strategy: ..." into the SAME
        # prompt — two contradictory instruction blocks the model had to
        # paper over. LIGHT SUPPORT is an instruction not to plan.
        tone = getattr(context, "tone_level", None)
        if tone is not None:
            from core.context_pipeline import ToneLevel
            if tone in (ToneLevel.CRISIS, ToneLevel.ELEVATED, ToneLevel.CONCERN):
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

    # Retrieval-ranked sections that materially affect what the response says.
    # Each gets a fair bounded excerpt so one large conversation cannot crowd
    # every other evidence class out of the planner view.
    _CONTEXT_DIGEST_KEYS = (
        "stm_summary",
        "recent_conversations",
        "memories",
        "relevant_emails",
        "web_search_results",
        "user_profile",
        "narrative_state",
        "graph_context",
        "recent_summaries",
        "semantic_summaries",
        "personal_notes",
        "reference_docs",
        "unresolved_threads",
        "upcoming_schedule",
        "google_calendar",
    )

    # Sections that are Daemon-synthesized (graph edges, narrative, summaries):
    # labelled in the digest so the planner weighs them below user-authored
    # memories (2026-09-03: "User has dog Mochi" + a "Dog Behavior" topic
    # label out-voted twenty memories calling the cat a cat).
    _DERIVED_DIGEST_KEYS = frozenset({
        "graph_context", "narrative_state", "recent_summaries", "semantic_summaries",
    })

    @staticmethod
    def build_context_digest(
        prompt_context: Optional[dict],
        *,
        max_chars: int = 6000,
        max_section_chars: int = 1200,
    ) -> tuple[str, List[str]]:
        """Build a bounded digest from the final gathered prompt context.

        The digest is ephemeral planner input.  Audit records retain only its
        SHA-256 and represented section names, avoiding a second persisted copy
        of potentially personal context.
        """
        if not isinstance(prompt_context, dict) or max_chars <= 0:
            return "", []

        chunks: List[str] = []
        included: List[str] = []
        remaining = max_chars
        for key in ResponsePlanner._CONTEXT_DIGEST_KEYS:
            value = prompt_context.get(key)
            if value in (None, "", [], {}, ()):
                continue

            # Context lists are already relevance/recency ranked.  Keep the
            # leading items and bound serialization before composing sections.
            compact_value = value[:3] if isinstance(value, (list, tuple)) else value
            try:
                rendered = json.dumps(
                    compact_value,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                rendered = str(compact_value)
            rendered = rendered[:max_section_chars]
            label = (
                f"{key} — derived, not the user's words"
                if key in ResponsePlanner._DERIVED_DIGEST_KEYS else key
            )
            chunk = f"[{label}]\n{rendered}"
            if len(chunk) > remaining:
                if remaining < len(key) + 8:
                    break
                chunk = chunk[:remaining]
            chunks.append(chunk)
            included.append(key)
            remaining -= len(chunk) + 2
            if remaining <= 0:
                break

        return "\n\n".join(chunks), included

    @staticmethod
    def _is_direct_communication_command(query: str) -> bool:
        """Detect an explicit command to address another audience.

        This is intentionally narrow: questions such as "how should I
        communicate with X?" must not be converted into role-play.  A direct
        imperative, or the unambiguous "the floor is yours" handoff, is locked
        so an LLM planner cannot reverse speaker and audience.
        """
        text = (query or "").strip()
        if not text or text.endswith("?"):
            return False
        verb_target = re.search(
            r"\b(?:communicate|speak|talk|address|respond|reply|write)\b"
            r"[^.!?\n]{0,40}\b(?:to|with)\b",
            text,
            flags=re.IGNORECASE,
        )
        if not verb_target:
            return False
        if re.search(r"\bthe\s+floor\s+is\s+yours\b", text, flags=re.IGNORECASE):
            return True
        return bool(
            re.match(
                r"^(?:please\s+)?(?:communicate|speak|talk|address|respond|reply|write)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _direct_communication_plan(
        *,
        planner_model: str,
        context_digest_sha256: str,
        context_sections: List[str],
    ) -> ResponsePlan:
        """Deterministically preserve the speaker/addressee requested by user."""
        return ResponsePlan(
            key_points=[
                "Carry out the requested communication and address the named recipient directly.",
                "Treat the user as handing over the floor, not as the person being interviewed about the recipient.",
            ],
            tone="direct",
            avoid=[
                "Do not ask the user to share or explain the interaction instead.",
                "Do not reverse the requested speaker and audience.",
            ],
            strategy="Speak directly to the requested recipient while using relevant gathered context.",
            planner_source="deterministic_direct_communication",
            planner_model=planner_model,
            context_digest_sha256=context_digest_sha256,
            context_sections=context_sections,
            directive_locked=True,
        )

    async def create_plan(
        self,
        query: str,
        context,
        *,
        context_digest: str = "",
        context_sections: Optional[List[str]] = None,
    ) -> Optional[ResponsePlan]:
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

        digest_hash = (
            hashlib.sha256(context_digest.encode("utf-8")).hexdigest()
            if context_digest else ""
        )
        represented_sections = list(context_sections or [])
        planner_model = str(RESPONSE_PLANNING_MODEL or "")

        # Preserve explicit speaker/addressee commands without asking a second
        # model to reinterpret them.  This prevents the exact class of reversal
        # where "communicate with Fable" became "ask the user about Fable."
        if self._is_direct_communication_command(query):
            return self._direct_communication_plan(
                planner_model=planner_model,
                context_digest_sha256=digest_hash,
                context_sections=represented_sections,
            )

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

        # Previous exchange — without it, a pronoun-anchored fragment
        # ("It was maybe 3 years of...") or a referent correction ("No I
        # mean...") is planned blind and the plan confidently reinforces
        # whatever the topic classifier guessed (2026-07-28 incident: a
        # long-covid frequency fragment got an "exercise routine" plan).
        exchange_block = ""
        last_ex = getattr(context, "last_exchange", None)
        if isinstance(last_ex, dict):
            last_user = str(last_ex.get("query") or "").strip()
            last_asst = str(last_ex.get("response") or "").strip()
            if last_user or last_asst:
                exchange_block = (
                    "Previous exchange (use it to resolve pronouns and fragments):\n"
                    f"User: {last_user[:400]}\n"
                    f"Assistant: {last_asst[:400]}\n\n"
                )

        prompt = (
            "You are a response planner. Given the query and context signals below, "
            "produce a JSON response plan.\n\n"
            f"{exchange_block}"
            f"Query: {query}\n"
            f"Intent: {intent_type}\n"
            f"Tone level: {tone_level}\n"
            f"Topics: {topics_str}\n"
            f"Thread depth: {thread_depth}\n\n"
            "The user's literal query is authoritative. Inferred Intent, Topics, "
            "and short-term summaries are fallible context signals: they must never "
            "change the requested speech act, speaker, or addressee. If the user asks "
            "you to communicate with someone, plan direct communication to that "
            "recipient; do not plan to interview the user about them.\n\n"
            "Named entities keep the attributes the user's own words give them. "
            "Never assign or infer a species, gender, role, relationship, or "
            "location for a person, pet, or place from the Topics label, the "
            "Intent, or general knowledge — only from the query, the previous "
            "exchange, or user-authored memories in the digest; when those "
            "describe the entity, that description wins over the Topics label. "
            "If an attribute is unknown, plan around the name alone.\n\n"
            "Key points restate what the user said and what the context holds. Never add an "
            "event, activity, feeling, or detail the user did not state (\"turned 2\" is an age, "
            "not a celebration; \"sent the email\" is not a reply).\n\n"
            "Compact digest of the SAME gathered context available to the main "
            "response model (retrieval-ranked excerpts; may be empty):\n"
            f"{context_digest or '(no gathered context)'}\n\n"
            "If the query is a fragment, opens with a pronoun (\"It was...\", "
            "\"That's...\"), or corrects an interpretation (\"No I mean...\"), "
            "resolve what it refers to from the previous exchange and plan for "
            "THAT — do not treat it as a standalone statement, and do not trust "
            "the Topics label over the previous exchange.\n\n"
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

        plan = self._parse_plan(raw)
        if plan is not None:
            # Embellishment guard (2026-09-03): the planner turned "turned 2
            # last week" into "recent birthday celebration". A key point that
            # introduces an event noun or a name absent from the query, the
            # previous exchange and the digest is dropped (never rewritten).
            sources = "\n".join(filter(None, [query or "", exchange_block or "", context_digest or ""]))
            kept, dropped = self.unsupported_key_points(plan.key_points, sources)
            if dropped:
                logger.info(f"[RESPONSE PLANNER] Dropped {len(dropped)} unsupported key point(s): {dropped}")
                plan.key_points = kept
                plan.dropped_points = dropped
            plan.planner_source = "llm"
            plan.planner_model = planner_model
            plan.context_digest_sha256 = digest_hash
            plan.context_sections = represented_sections
        return plan

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

    # Event nouns the planner tends to invent around a bare fact. Each entry is
    # a stem regex applied to BOTH the key point and the sources, so "celebrated"
    # in the query licenses "celebration" in the plan.
    _EVENT_STEMS = (
        r"celebrat\w*", r"part(?:y|ies)", r"cake", r"gifts?", r"trips?", r"vacations?",
        r"wedding", r"funeral", r"ceremon\w*", r"surger\w*", r"interview\w*",
        r"anniversar\w*", r"graduat\w*", r"concert", r"holiday", r"dinner", r"lunch",
        r"brunch", r"meeting", r"appointment", r"visit\w*", r"reunion",
    )
    _POINT_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
    _POINT_NAME_STOP = frozenset({
        "The", "This", "That", "These", "Those", "Their", "They", "User", "Daemon", "Also",
        "Share", "Cover", "Note", "Ask", "Offer", "Keep", "Avoid", "Acknowledge", "Mention",
    })

    @classmethod
    def unsupported_key_points(cls, points: List[str], sources: str) -> tuple:
        """Split ``points`` into (kept, dropped). A point is dropped when it
        carries an event stem or a TitleCase name (past its first word) that
        the sources never mention. Under-fires: only the listed event stems
        and capitalised names are checked; if every point would be dropped the
        original list is kept (an empty plan is worse than an embellished one)."""
        src = (sources or "").lower()
        kept: List[str] = []
        dropped: List[str] = []
        for pt in points or []:
            text = str(pt or "")
            low = text.lower()
            bad = False
            for stem in cls._EVENT_STEMS:
                if re.search(r"\b" + stem + r"\b", low) and not re.search(r"\b" + stem + r"\b", src):
                    bad = True
                    break
            if not bad:
                words = text.split()
                for tok in cls._POINT_NAME_RE.findall(" ".join(words[1:])):
                    if tok in cls._POINT_NAME_STOP:
                        continue
                    if tok.lower() not in src and tok.lower().rstrip("s") not in src:
                        bad = True
                        break
            (dropped if bad else kept).append(text)
        if points and not kept:
            return list(points), []
        return kept, dropped

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
