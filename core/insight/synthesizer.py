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

import json
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
Calibrated causal and forward-looking assessment is allowed when tied to the
retrieved evidence and uncertainty. Do not diagnose, prescribe a medication or
dose, or claim clinical review is unnecessary; keep claims A–E separate."""

# A user's explanation of their own symptoms is relevant context, but it is
# not an observed outcome. This distinction is essential for assessment mode:
# "I think", "maybe", "could", "feels like", "strongest theory", and similar
# uncertainty markers must be reported as hypotheses to test, never counted as
# support for the hypothesis. Only dated observations/behaviors and independent
# research can support or refute a claim.
_HYPOTHESIS_RULE = """
HYPOTHESIS VERSUS OBSERVATION: Treat the user's causal interpretations,
predictions, and uncertainty (for example "I think", "maybe", "could be",
"feels like", "strongest theory", "I don't know", or "likely") as hypotheses
or context to evaluate—not as evidence that the hypothesis is true. Do not use
such statements to support a claim. Extract observable dated events, symptoms,
behaviors, medication timing, and independently sourced research separately;
when an item mixes them, quote only the observation and label the interpretation
as the user's hypothesis.
"""

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

_PATTERN_TAIL = """
7. COMPUTED AGGREGATE: the "Deterministic aggregate" block below was COUNTED \
by code over the full record — restate its numbers faithfully; never recount, \
extrapolate, or invent counts of your own. Organize the story earliest → \
latest by time bucket, stating counts WITH their dates.
8. TREND CONFIDENCE: calibrate language to n — "3 mentions in 6 months" is \
not a pattern the way "daily for two weeks" is. If the aggregate says \
trend=insufficient, say plainly that the record is too thin to call a trend. \
Do not infer causation from a trend alone; causal language requires the \
explicit longitudinal contract and alternative-explanation checks.
9. CAUSAL BOUNDARY: calibrated causal or forward-looking assessment is \
allowed only when phases, alternatives, and uncertainty support it; do not \
prescribe medication/dose or claim professional review is unnecessary.
10. DENOMINATOR, WITH NUMBERS: use the turns-per-bucket denominators in the \
aggregate — if mentions rose in a bucket where total talking also rose, say \
so explicitly. Record-frequency is not life-frequency."""

_DELIBERATION_TAIL = """
11. FROZEN DELIBERATION CONTRACT: the supplied longitudinal manifest was
planned and validated before retrieval. State its outcome definition, phase
bounds, analytical assumptions, missing requested channels, counterevidence,
and sensitivity limitations before giving the bottom-line assessment.
12. CLAIM CHAIN: report every claim's own status/confidence/coverage and honor
dependencies. Never collapse the chain to the weakest downstream claim. An
outside-authority prescription/diagnosis claim does not erase supported
historical or research claims.
13. SOURCE HONESTY: distinguish the user's dated statements/notes from
external research. Do not treat a search result as proof of efficacy merely
because it exists, and do not describe a requested-but-unavailable channel as
searched.
14. DECISION SUPPORT: a calibrated causal or forward-looking assessment is
allowed when the evidence contract supports it. Do not prescribe a treatment
or dose, diagnose, or imply that independent professional assessment is
unnecessary."""

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


def _render_deliberation_manifest(manifest: dict, max_chars: int = 14000) -> str:
    """Serialize a structurally valid bounded manifest, never raw-truncate JSON."""
    encoded = json.dumps(
        manifest, ensure_ascii=False, default=str, separators=(",", ":"),
    )
    if len(encoded) <= max_chars:
        return encoded

    spec = manifest.get("spec") or {}
    compact = {
        "status": manifest.get("status"),
        "planner_provenance": manifest.get("planner_provenance"),
        "manifest_compacted": True,
        "spec": {
            key: spec.get(key) for key in (
                "analysis_kind", "claims", "outcome_terms",
                "series_terms",
                "concept_synonyms",
                "behavioral_indicators", "directional_indicators", "phases",
                "requested_channels", "research_queries", "assumptions",
                "decision_context",
            ) if key in spec
        },
        "anchor_candidates": (manifest.get("anchor_candidates") or [])[:12],
        # 32-row cap: weekly co_occurrence runs have 26+ buckets; the old
        # 12-row cap silently dropped the LAST weeks of a windowed table
        # (live 2026-08-31: Aug 24-31 rows vanished from a 14-bucket report).
        "phase_summary": [
            {**row, "source_ids": (row.get("source_ids") or [])[:6]}
            for row in (manifest.get("phase_summary") or [])[:32]
        ],
        "external_sources": [
            {key: row.get(key) for key in (
                "source_id", "source_class", "title", "url", "date", "snippet"
            )}
            for row in (manifest.get("external_sources") or [])[:16]
        ],
        "channels": [
            {**row, "source_ids": list(row.get("source_ids") or [])[:6]}
            for row in (manifest.get("channels") or [])
        ],
        "claim_chain": [
            {**row, "rationale": str(row.get("rationale") or "")[:350]}
            for row in (manifest.get("claim_chain") or [])[:8]
        ],
        "limitations": (manifest.get("limitations") or [])[:16],
        "sensitivity": (manifest.get("sensitivity") or [])[:16],
        "co_occurrence": manifest.get("co_occurrence") or {},
        "doctrine": manifest.get("doctrine") or {},
    }
    encoded = json.dumps(
        compact, ensure_ascii=False, default=str, separators=(",", ":"),
    )
    if len(encoded) <= max_chars:
        return encoded

    # Last-resort structural summary retains the contract and status of every
    # claim/channel, rather than returning malformed or misleading partial JSON.
    minimal = {
        "status": compact["status"],
        "manifest_compacted": True,
        "spec": compact["spec"],
        "phase_summary": [
            {key: row.get(key) for key in (
                "label", "start", "end", "outcome_events", "observations",
                "observed_days", "calendar_coverage", "decreased",
                "increased", "unclear_direction", "covariate_counts",
                "series_counts", "joint_event_count",
            )}
            for row in compact["phase_summary"]
        ],
        "channels": [
            {key: row.get(key) for key in (
                "channel", "status", "attempted", "reason", "count",
            )}
            for row in compact["channels"]
        ],
        "claim_chain": [
            {key: row.get(key) for key in (
                "claim_id", "proposition", "status", "confidence", "coverage",
                "directness", "dependencies", "authority",
            )}
            for row in compact["claim_chain"]
        ],
        "limitations": compact["limitations"],
        "co_occurrence": compact["co_occurrence"],
        "doctrine": compact["doctrine"],
    }
    return json.dumps(
        minimal, ensure_ascii=False, default=str, separators=(",", ":"),
    )


def build_synthesis_prompts(
    intent: InsightIntent,
    evidence: list[EvidenceItem],
    assessment: Optional[Assessment],
    *,
    tone_elevated: bool = False,
    max_evidence_chars: int = 12000,
    patterns: Optional[list] = None,
    deliberation_manifest: Optional[dict] = None,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the synthesis call. Pure.
    ``patterns`` is a list of memory.pattern_engine.PatternResult for
    pattern_temporal runs — rendered as a computed-numbers block."""
    system = _BASE_SYSTEM
    if intent.kind == "insight_assessment":
        system += _ASSESS_TAIL
    elif intent.kind == "pattern_temporal":
        system += _PATTERN_TAIL
    else:
        system += _SWEEP_TAIL
    if tone_elevated:
        system += _ELEVATED_TAIL
    if deliberation_manifest is not None:
        system += _DELIBERATION_TAIL
    if intent.kind in {"pattern_temporal", "insight_assessment"}:
        system += _HYPOTHESIS_RULE

    evidence_block = render_evidence_block(evidence, max_chars=max_evidence_chars)

    parts = [f"Theme / stated insight: {intent.theme}", ""]
    if intent.raw_query and intent.raw_query != intent.theme:
        parts.append(f"The user's request, verbatim: {intent.raw_query}")
        parts.append("")
    if patterns:
        parts.append(
            "Deterministic aggregate (COMPUTED over the full record — restate, "
            "never recount):")
        for p in patterns:
            parts.append(p.render_table())
        parts.append("")
    if deliberation_manifest is not None:
        parts.append(
            "Frozen longitudinal deliberation manifest (COMPUTED/VALIDATED; "
            "do not silently change its evidence set):"
        )
        parts.append(_render_deliberation_manifest(deliberation_manifest))
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
    patterns: Optional[list] = None,
    deliberation_manifest: Optional[dict] = None,
    disable_reasoning: bool = False,
) -> AsyncGenerator[str, None]:
    """Stream visible synthesis chunks (reasoning-channel filtered)."""
    from config.app_config import INSIGHT_SYNTHESIS_MAX_TOKENS
    from core.reasoning_stream_filter import InterleavedReasoningFilter

    system_prompt, prompt = build_synthesis_prompts(
        intent, evidence, assessment, tone_elevated=tone_elevated,
        patterns=patterns, deliberation_manifest=deliberation_manifest,
    )

    stream = await model_manager.generate_async(
        prompt=prompt,
        model_name=model_name,
        system_prompt=system_prompt,
        max_tokens=INSIGHT_SYNTHESIS_MAX_TOKENS,
        disable_reasoning=disable_reasoning,
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
