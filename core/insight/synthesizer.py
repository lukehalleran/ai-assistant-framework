"""
core/insight/synthesizer.py

Module Contract
- Purpose: Stream the final insight-mode response (and produce the optional
  document body). The framing invariants are HARD-CODED in the system prompt —
  they are doctrine, not style preferences:
    * evidence-assembly, never verdict-rendering (theme sweeps): quote + date,
      motivational-interviewing shape, the user draws conclusions;
    * never assert thick evaluative claims in system voice — appraisals render
      as "you described X as … (date)";
    * MANDATORY denominator caveat: this corpus over-samples days the user
      chose to talk, often hard days — record-frequency ≠ life-frequency;
    * assessment runs present the verdict HONESTLY either way, showing both
      supporting and refuting evidence (fail-honest doctrine);
    * elevated tone → third-party/therapist framing, user's own words only,
      no excavating questions.
- Inputs: InsightIntent, labeled evidence, optional Assessment, model_manager
  + model_name, tone_level.
- Outputs: async generator of visible text chunks (reasoning-filtered via
  InterleavedReasoningFilter, same discipline as the agentic final response).
- Side effects: one streaming LLM call; no storage writes.
"""

from __future__ import annotations

from typing import AsyncGenerator, Optional

from core.insight.provenance import render_evidence_block
from core.insight.types import Assessment, EvidenceItem, InsightIntent
from utils.logging_utils import get_logger

logger = get_logger("insight_synthesizer")

_BASE_SYSTEM = """You are assembling evidence from a user's own recorded history \
(conversations, notes, extracted facts) about a personal theme. Non-negotiable rules:

1. QUOTE AND DATE. Every substantive point cites its evidence reference \
([E3]) with its date. Never paraphrase the user's words into stronger claims.
2. NEVER assert a value judgment in your own voice. Evidence marked as an \
appraisal is the author's take at the time: render it as "you described X as \
'…' (date)", never as "X is …".
3. Evidence marked "assistant's interpretation, not your words" is the \
system's own prior inference. If you use it at all, attribute it explicitly \
("I suggested at the time that…") and give it less weight than the user's words.
4. MANDATORY DENOMINATOR CAVEAT: include, verbatim in spirit, a short \
paragraph noting that this record over-samples the days the user chose to \
talk — often the hard days — so how often something appears here is NOT how \
often it happens in their life, and absences prove little.
5. Organize by pattern/time, not by store. Contradictory or counter-evidence \
gets its own visible place — never buried.
6. Close in a motivational-interviewing shape: reflect what the record shows, \
then hand interpretation back to the user with at most ONE open question. \
Do not diagnose, do not prescribe, do not moralize."""

_SWEEP_TAIL = """
7. You are ASSEMBLING evidence, not rendering a verdict. Do not conclude \
whether the user's read on themselves is right — lay out what the record \
holds and let them look at it."""

_ASSESS_TAIL = """
7. The user explicitly asked you to assess their stated insight. Present the \
verdict HONESTLY — including "the record disagrees" or "the record can't \
settle this" — with the supporting AND refuting evidence shown side by side. \
Do not soften a disagreement into agreement, and do not manufacture \
confidence the evidence lacks."""

_ELEVATED_TAIL = """
8. TONE GUARD: the user is currently in a distressed state. Keep the framing \
third-party and practical (something they can hand to a therapist), use ONLY \
their own recorded words for anything emotionally heavy, ask NO excavating \
questions, and keep the close brief and steady."""

_VERDICT_DISPLAY = {
    "agree": "the record broadly supports this",
    "partial": "the record partially supports this, with real caveats",
    "insufficient": "the record cannot settle this either way",
    "disagree": "the record cuts against this",
}


def build_synthesis_prompts(
    intent: InsightIntent,
    evidence: list[EvidenceItem],
    assessment: Optional[Assessment],
    *,
    tone_elevated: bool = False,
    max_evidence_chars: int = 12000,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the synthesis call. Pure."""
    system = _BASE_SYSTEM
    system += _ASSESS_TAIL if intent.kind == "insight_assessment" else _SWEEP_TAIL
    if tone_elevated:
        system += _ELEVATED_TAIL

    evidence_block = render_evidence_block(evidence, max_chars=max_evidence_chars)

    parts = [f"Theme / stated insight: {intent.theme}", ""]
    if intent.raw_query and intent.raw_query != intent.theme:
        parts.append(f"The user's request, verbatim: {intent.raw_query}")
        parts.append("")
    if assessment is not None and intent.kind == "insight_assessment":
        parts.append(
            "Independent assessment verdict (present this honestly): "
            f"{assessment.overall} — {_VERDICT_DISPLAY.get(assessment.overall, assessment.overall)}"
        )
        for c in assessment.claims:
            support = ", ".join(c.support) or "none"
            refute = ", ".join(c.refute) or "none"
            parts.append(
                f"  - claim: {c.claim} → {c.verdict} "
                f"(support: {support}; refute: {refute}) {c.notes}".rstrip()
            )
        parts.append("")
    parts.append(f"Assembled evidence ({len(evidence)} items):")
    parts.append(evidence_block or "(the sweep found no evidence on this theme)")
    return system, "\n".join(parts)


async def synthesize_stream(
    intent: InsightIntent,
    evidence: list[EvidenceItem],
    assessment: Optional[Assessment],
    *,
    model_manager,
    model_name: Optional[str] = None,
    tone_elevated: bool = False,
) -> AsyncGenerator[str, None]:
    """Stream visible synthesis chunks (reasoning-channel filtered)."""
    from config.app_config import INSIGHT_SYNTHESIS_MAX_TOKENS
    from core.reasoning_stream_filter import InterleavedReasoningFilter

    system_prompt, prompt = build_synthesis_prompts(
        intent, evidence, assessment, tone_elevated=tone_elevated
    )

    stream = await model_manager.generate_async(
        prompt=prompt,
        model_name=model_name,
        system_prompt=system_prompt,
        max_tokens=INSIGHT_SYNTHESIS_MAX_TOKENS,
    )

    # Yield ONLY confirmed visible content — synthetic <thinking> markers are
    # dropped (insight mode shows sweep progress instead of a thinking shell).
    from core.reasoning_stream_filter import CONTENT

    rfilter = InterleavedReasoningFilter()
    if hasattr(stream, "__aiter__"):
        async for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                delta_reasoning = (
                    getattr(delta, "reasoning_content", "")
                    or getattr(delta, "reasoning", "")
                    or ""
                )
                delta_content = getattr(delta, "content", "") or ""
                for kind, text in rfilter.feed(delta_reasoning, delta_content):
                    if kind == CONTENT:
                        yield text
            elif isinstance(chunk, str) and chunk:
                for kind, text in rfilter.feed("", chunk):
                    if kind == CONTENT:
                        yield text
        for kind, text in rfilter.finish():
            if kind == CONTENT:
                yield text
    elif isinstance(stream, str):
        yield stream
