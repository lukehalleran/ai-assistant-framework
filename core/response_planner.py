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

        # A task directive ("jot down a note for this session: TA sessions
        # are Saturdays at 11 CT") needs an ACTION routed through the
        # agentic gate, not an answer plan (2026-09-11, round 5, B14): the
        # live turn's [RESPONSE PLAN] restated a false calendar claim
        # pulled straight from the digest ("TA sessions are scheduled for
        # Saturdays at 11 AM CT... recurring weekly events through
        # December 12") in place of actually saving the user's one-line
        # note. QUESTIONS are never task directives (see
        # ``is_task_directive``'s own docstring), so an info-seeking
        # request is unaffected and keeps planning normally.
        try:
            from utils.query_checker import is_task_directive
            if is_task_directive(query):
                return False
        except Exception:
            pass

        # A bare self-report ("Cool. Managed to push today and there is a
        # new doc I think will be helpful") requests nothing — planning for
        # it invites the same embellishment the digest guard exists for
        # (2026-09-10 probe T3: STM misread "doc" as "doctor" and the
        # planner confidently planned three points about a new doctor). A
        # self-report that is ALSO request-shaped (asks a question, wants
        # something) still plans normally. Round 2 (same probe, retest):
        # ``is_self_report`` alone returned False on this EXACT text — its
        # subject is elided after the "Cool." ack rather than restated as a
        # pronoun, so the skip never fired and the planner ran anyway.
        # ``is_status_report`` is the same shape's separate, narrow cousin
        # (see its docstring for why it's not folded into is_self_report).
        try:
            from utils.query_checker import is_self_report, is_status_report, is_request_shaped
            if ((is_self_report(query) or is_status_report(query))
                    and not is_request_shaped(query)):
                return False
        except Exception:
            pass

        return True

    # ------------------------------------------------------------------
    # Pre-answer planning
    # ------------------------------------------------------------------

    # Retrieval-ranked sections that materially affect what the response says.
    # Each gets a fair bounded excerpt so one large conversation cannot crowd
    # every other evidence class out of the planner view. "user_uploads" is
    # FIRST and rendered with its own reserved allowance BEFORE this
    # sequential fill (see build_context_digest) — F3 (2026-09-08
    # homework-session audit): the planner had NO evidence channel for the
    # attachment the answer is supposed to be using at all (direct
    # invocation with only a user_uploads context yielded an EMPTY digest),
    # so a large STM/history/profile digest exhausting the shared budget
    # ahead of it must never crowd out the current attachment either.
    _CONTEXT_DIGEST_KEYS = (
        "user_uploads",
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

    # Reserved allowance for the user_uploads digest chunk, rendered before
    # the fair sequential loop below (2026-09-08, F3).
    _USER_UPLOADS_DIGEST_MAX_CHARS = 1500

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

        # user_uploads gets a reserved allowance rendered BEFORE the fair
        # sequential loop (2026-09-08, F3) — a large STM/history/profile
        # digest must never be able to exhaust the shared budget ahead of
        # the current attachment the answer is supposed to be using.
        # Roster-only marker items (metadata type "upload_roster" with no
        # content) carry no evidence for the planner and are dropped.
        uploads_value = prompt_context.get("user_uploads")
        if uploads_value not in (None, "", [], {}, ()):
            upload_items = uploads_value
            if isinstance(uploads_value, (list, tuple)):
                upload_items = [
                    item for item in uploads_value
                    if not (
                        isinstance(item, dict)
                        and item.get("metadata", {}).get("type") == "upload_roster"
                        and not item.get("content")
                    )
                ]
            if upload_items:
                compact_uploads = (
                    upload_items[:3] if isinstance(upload_items, (list, tuple)) else upload_items
                )
                try:
                    rendered_uploads = json.dumps(
                        compact_uploads,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                        separators=(",", ":"),
                    )
                except (TypeError, ValueError):
                    rendered_uploads = str(compact_uploads)
                rendered_uploads = rendered_uploads[:ResponsePlanner._USER_UPLOADS_DIGEST_MAX_CHARS]
                upload_chunk = f"[user_uploads — current attachments]\n{rendered_uploads}"
                if len(upload_chunk) > remaining:
                    upload_chunk = upload_chunk[:remaining] if remaining > 0 else ""
                if upload_chunk:
                    chunks.append(upload_chunk)
                    included.append("user_uploads")
                    remaining -= len(upload_chunk) + 2

        for key in ResponsePlanner._CONTEXT_DIGEST_KEYS:
            if key == "user_uploads":
                continue  # rendered above with its own reserved allowance
            if remaining <= 0:
                break
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
            "Strategy should engage with why the observation matters in this "
            "conversation, not just repeat or validate it. A clarification "
            "updates the earlier interpretation; don't turn it into a lesson "
            "about communicating clearly. A current self-report is not automatically "
            "temporal recall. Previous assistant advice is not evidence: do not "
            "instruct the responder to endorse or reject a change the user "
            "mentioned (a treatment, habit, routine, or plan) just because they "
            "mentioned it. Preserve a balanced discussion of options and "
            "uncertainty when the user seeks one.\n\n"
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
            strict_sources = "\n".join(filter(None, [query or "", exchange_block or ""]))
            original_points = list(plan.key_points)
            kept, dropped = self.unsupported_key_points(
                original_points, sources, query=query, strict_sources=strict_sources,
            )
            # "Every key point dropped" (as opposed to "there were none to
            # begin with") is only true when unsupported_key_points's own
            # never-fully-empty safeguard did NOT fall back to keeping
            # everything — i.e. dropped is non-empty and kept came back
            # empty from a genuine strict-mode wipe.
            all_points_dropped = bool(original_points) and bool(dropped) and not kept
            if dropped:
                logger.info(f"[RESPONSE PLANNER] Dropped {len(dropped)} unsupported key point(s): {dropped}")
                plan.key_points = kept
                plan.dropped_points = dropped

            # B12 (2026-09-10, round 3): the same prefix-expansion + head-noun
            # checks apply to `strategy` and each `avoid` line — the live
            # round-3 plan had key_points=[] from the LLM itself (nothing to
            # drop above) but strategy "...express support for their new
            # doctor" carried the exact same unsupported "doc"->"doctor"
            # expansion, uncaught because only key_points was ever checked.
            if self._statement_unsupported(plan.strategy, sources, query=query, strict_sources=strict_sources):
                logger.info(f"[RESPONSE PLANNER] Blanked unsupported strategy: {plan.strategy!r}")
                plan.strategy = ""
            plan.avoid = [
                a for a in plan.avoid
                if not self._statement_unsupported(a, sources, query=query, strict_sources=strict_sources)
            ]

            # An all-points-dropped plan, or a plan left with nothing at all
            # (no key points, no strategy, no avoid — T5's exact shape once
            # its lone strategy sentence is blanked), is discarded outright
            # rather than injecting an empty "[RESPONSE PLAN] Cover: (none)"
            # block that adds nothing but plan-shaped noise to the prompt.
            if all_points_dropped or (not plan.key_points and not plan.strategy and not plan.avoid):
                logger.info("[RESPONSE PLANNER] Plan emptied by embellishment guards — discarding")
                return None

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

    # Closed-class function words skipped when hunting for a key point's head
    # noun below — grammar, not topic vocabulary. 2026-09-10 round 2 (probe
    # T5): "user"/"they"/"assistant"/"daemon" are the plan's own subject
    # labels (and, via the rendered "User: ..." exchange block, trivially
    # "present in the sources" no matter what the point actually claims), and
    # "believes"/"thinks"/"feels"/"wants"/"found"/"mentioned"/"shared" are
    # reporting verbs, not content — "The user believes this new doctor will
    # be helpful" head-nouned to "user" and was kept because the exchange
    # label literally contains the word "User". ("has"/"they" were already
    # covered by "have/has/had" and the pronoun row above.)
    _HEAD_NOUN_STOP = frozenset({
        "a", "an", "the", "this", "that", "these", "those",
        "is", "are", "was", "were", "be", "been", "being",
        "will", "would", "can", "could", "should", "may", "might", "must",
        "have", "has", "had", "do", "does", "did",
        "and", "or", "but", "nor", "so", "yet",
        "of", "to", "in", "on", "at", "by", "for", "with", "about", "as",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "its", "our", "their",
        "not", "no",
        "user", "assistant", "daemon",
        "believes", "thinks", "feels", "wants", "found", "mentioned", "shared",
    })

    @classmethod
    def _head_noun_stop_words(cls) -> frozenset:
        """``_HEAD_NOUN_STOP`` plus the current user's own display name
        (2026-09-10 round 2) — resolved dynamically via
        ``utils.user_identity`` rather than hardcoded, per the project's
        no-personal-vocabulary-in-source doctrine. Falls back to the base
        set on any failure (the resolver's own fallback, "the user", is
        already covered by the "the"/"user" stop tokens)."""
        try:
            from utils.user_identity import get_user_display_name
            name = get_user_display_name() or ""
        except Exception:
            return cls._HEAD_NOUN_STOP
        extra = {t.lower() for t in re.findall(r"[A-Za-z']+", name)}
        return cls._HEAD_NOUN_STOP | extra if extra else cls._HEAD_NOUN_STOP

    @classmethod
    def _head_noun(cls, point: str) -> str:
        """Coarse content-word anchor: the first token after a key point's
        leading subject/verb word that isn't a closed-class function word.
        Deliberately a heuristic guard, not a parser — see
        ``unsupported_key_points``'s statement-mode check."""
        tokens = re.findall(r"[A-Za-z']+", point or "")
        if len(tokens) < 2:
            return ""
        stop = cls._head_noun_stop_words()
        for tok in tokens[1:]:
            low = tok.lower().strip("'")
            if not low or low in stop:
                continue
            return low
        return ""

    # Statement-mode companion to the head-noun check above (2026-09-10
    # round 2, probe T5): the head-noun check alone can miss an embellishment
    # when the word right after the subject is itself grammar/reporting
    # vocabulary now in ``_HEAD_NOUN_STOP`` — the next content word it lands
    # on ("new" in "The user believes this new doctor...") can coincidentally
    # appear in strict_sources even though the point's actual (unsupported)
    # claim is a different word entirely ("doctor"). Independently: any QUERY
    # token of <= 4 letters that is a STRICT PREFIX of a LONGER word in the
    # point ("doc" -> "doctor") is an unsupported abbreviation expansion
    # unless that longer word itself appears in strict_sources.
    @classmethod
    def _has_unsupported_prefix_expansion(cls, point_text: str, query: str, strict_src: str) -> bool:
        q_tokens = {
            t.lower() for t in re.findall(r"[A-Za-z']+", query or "")
            if 1 < len(t) <= 4 and t.lower() not in cls._HEAD_NOUN_STOP
        }
        if not q_tokens:
            return False
        strict_src = strict_src or ""
        for tok in re.findall(r"[A-Za-z']+", point_text or ""):
            low = tok.lower().strip("'")
            if len(low) <= 4:
                continue
            for qt in q_tokens:
                if low != qt and low.startswith(qt):
                    if low not in strict_src and low.rstrip("s") not in strict_src:
                        return True
        return False

    @classmethod
    def _strict_mode_context(cls, query: Optional[str], strict_sources: Optional[str]) -> tuple:
        """Resolve (strict_mode, strict_src) from a query the same way for
        every caller (unsupported_key_points, _statement_unsupported):
        strict mode applies only to a non-request-shaped (bare statement)
        query, and ``strict_src`` defaults to the query itself when
        ``strict_sources`` is omitted. ``query=None`` always yields
        (False, "") — prior no-query behavior is preserved exactly."""
        if query is None:
            return False, ""
        try:
            from utils.query_checker import is_request_shaped
            if is_request_shaped(query):
                return False, ""
        except Exception:
            return False, ""
        return True, (strict_sources if strict_sources is not None else query).lower()

    @classmethod
    def _basic_unsupported(cls, text: str, src: str) -> bool:
        """Event-stem / TitleCase-name check — the ORIGINAL (pre-2026-09-10)
        unsupported-point test, factored out so both the list-mode
        (``unsupported_key_points``) and single-string (``_statement_unsupported``)
        callers share one implementation."""
        low = (text or "").lower()
        for stem in cls._EVENT_STEMS:
            if re.search(r"\b" + stem + r"\b", low) and not re.search(r"\b" + stem + r"\b", src):
                return True
        words = (text or "").split()
        for tok in cls._POINT_NAME_RE.findall(" ".join(words[1:])):
            if tok in cls._POINT_NAME_STOP:
                continue
            if tok.lower() not in src and tok.lower().rstrip("s") not in src:
                return True
        return False

    @classmethod
    def _strict_unsupported(cls, text: str, query: str, strict_src: str) -> bool:
        """Statement-mode head-noun + prefix-expansion checks (2026-09-10
        rounds 1-2), factored out for reuse by ``_statement_unsupported``."""
        head = cls._head_noun(text)
        if head and head not in strict_src and head.rstrip("s") not in strict_src:
            return True
        return cls._has_unsupported_prefix_expansion(text, query, strict_src)

    @classmethod
    def unsupported_key_points(cls, points: List[str], sources: str, *,
                                query: str = None, strict_sources: str = None) -> tuple:
        """Split ``points`` into (kept, dropped). A point is dropped when it
        carries an event stem or a TitleCase name (past its first word) that
        the sources never mention. Under-fires: only the listed event stems
        and capitalised names are checked; if every point would be dropped the
        original list is kept (an empty plan is worse than an embellished one).

        Statement-mode head-noun check (2026-09-10, probe T3): when ``query``
        is given and is NOT request-shaped (a bare statement, not a question
        or a request), a point is additionally dropped when its head noun —
        the first content word after its subject — never appears in
        ``strict_sources`` (query + last exchange; the retrieval DIGEST is
        deliberately excluded). A statement turn's digest alone is not
        support: "Cool. Managed to push today and there is a new doc I
        think will be helpful" (a repo-doc share) had its STM-expanded
        "doc"→"doctor" misread validated by a digest full of doctor/
        psychiatrist memories, and the planner confidently planned three
        points about a new doctor. ``strict_sources`` defaults to ``query``
        when omitted. Passing no ``query`` (None) preserves prior behavior
        exactly — a request-shaped query still gets full digest support.

        Round 2 (2026-09-10, same probe, retest): the head-noun alone still
        missed it — "The user believes this new doctor will be helpful"
        head-nouned to "user" (present in the exchange's own "User: ..."
        label) — so a second, independent statement-mode check
        (``_has_unsupported_prefix_expansion``) drops a point whose text
        contains a word that is a strict, longer expansion of a short (<=4
        letter) query token ("doc" -> "doctor") and that expanded word is
        itself absent from ``strict_sources``.
        """
        src = (sources or "").lower()
        strict_mode, strict_src = cls._strict_mode_context(query, strict_sources)
        kept: List[str] = []
        dropped: List[str] = []
        # Tracked separately from the final bad/kept split so the "never
        # fully empty" safeguard below can tell WHY everything was dropped:
        # the original event/name checks emptying the plan still falls back
        # to keeping everything (unchanged prior behavior), but the new
        # strict-mode check emptying an otherwise-fine plan is allowed to
        # stand — a statement with every point built on an unsupported
        # referent is better left unplanned than embellished.
        any_kept_pre_strict = False
        for pt in points or []:
            text = str(pt or "")
            bad = cls._basic_unsupported(text, src)
            if not bad:
                any_kept_pre_strict = True
            if not bad and strict_mode and cls._strict_unsupported(text, query, strict_src):
                bad = True
            (dropped if bad else kept).append(text)
        if points and not kept:
            if strict_mode and any_kept_pre_strict:
                return kept, dropped
            return list(points), []
        return kept, dropped

    @classmethod
    def _statement_unsupported(cls, text: str, sources: str, *,
                                query: str = None, strict_sources: str = None) -> bool:
        """Single-string counterpart to ``unsupported_key_points`` for the
        plan's ``strategy`` sentence and each ``avoid`` line (2026-09-10,
        round 3): the SAME event-stem/name + statement-mode head-noun/
        prefix-expansion checks, but with NO "never fully empty" list
        safeguard — an embellished strategy sentence is blanked outright
        rather than kept because it happened to be the plan's only content.

        Live round-3 finding: T5 ("Cool. Managed to push today and there is
        a new doc I think will be helpful") produced key_points=[] from the
        LLM itself (nothing to drop) plus strategy "Acknowledge the user's
        progress and express support for their new doctor." — the same
        unsupported "doc"->"doctor" expansion ``unsupported_key_points``
        already catches for key points, un-checked for strategy/avoid.
        """
        text = str(text or "")
        if not text.strip():
            return False
        src = (sources or "").lower()
        if cls._basic_unsupported(text, src):
            return True
        strict_mode, strict_src = cls._strict_mode_context(query, strict_sources)
        if strict_mode and cls._strict_unsupported(text, query, strict_src):
            return True
        return False

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
