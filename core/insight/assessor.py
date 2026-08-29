"""
core/insight/assessor.py

Module Contract
- Purpose: Independent adversarial assessment of the user's stated insight
  against the assembled evidence. One strict-JSON LLM call that MUST seek
  refuting evidence per claim; the overall verdict is WORST-OF across claims.
- Inputs: claims (from FacetPlan), labeled EvidenceItems, model_manager.
- Outputs: Assessment (overall ∈ disagree/insufficient/partial/agree).
- Key behaviors: parse failure or empty response → overall "insufficient"
  (FAIL HONEST, never fail-agree — a broken assessor must not greenlight a
  document). Verdicts outside the vocabulary coerce to "insufficient".
- Side effects: one LLM call; no storage writes.
"""

from __future__ import annotations

import json

from core.insight.provenance import render_evidence_block
from core.insight.types import Assessment, ClaimAssessment, EvidenceItem
from utils.logging_utils import get_logger

logger = get_logger("insight_assessor")

_ASSESS_SYSTEM = (
    "You are an adversarial evidence auditor for a personal-memory system. "
    "Your job is to test the user's stated insight against their OWN recorded "
    "history — you are NOT here to validate it. For every claim you MUST "
    "actively look for refuting evidence before crediting supporting evidence. "
    "Evidence labeled \"assistant's interpretation\" is the system's own prior "
    "inference and carries LESS weight than the user's recorded words. "
    "Evidence marked as an appraisal is the author's value judgment at the "
    "time, not an objective fact. Respond with STRICT JSON only."
)

_ASSESS_PROMPT = """Claims to test:
{claims}

Evidence (dated, source-attributed):
{evidence}

CAVEAT you must weigh: this corpus over-samples days the user chose to talk —
often difficult days. Absence of counter-evidence is weak evidence at best,
and frequency in this record is NOT frequency in life.

For each claim return a verdict:
- "agree": multiple independent, user-sourced evidence items support it and
  refuting evidence is absent or clearly weaker
- "partial": genuinely mixed, or supported only with substantial caveats
- "insufficient": the record cannot settle it either way
- "disagree": the record actively cuts against the claim

JSON:
{{"claims": [{{"claim": "<restated>", "verdict": "agree|partial|insufficient|disagree",
  "support": ["E3", ...], "refute": ["E7", ...], "notes": "<one or two sentences>"}}]}}"""


async def assess(
    claims: list[str],
    evidence: list[EvidenceItem],
    model_manager,
) -> Assessment:
    """Adversarially assess claims against evidence. Never raises; any failure
    path returns overall='insufficient'."""
    if not claims:
        return Assessment(overall="insufficient", claims=[])

    evidence_block = render_evidence_block(evidence, max_chars=11000)
    prompt = _ASSESS_PROMPT.format(
        claims="\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1)),
        evidence=evidence_block or "(no evidence found)",
    )

    try:
        text = await model_manager.generate_once(
            prompt,
            system_prompt=_ASSESS_SYSTEM,
            max_tokens=900,
            temperature=0.0,
        )
    except Exception as e:
        logger.warning(f"[Insight] Assessor LLM call failed: {e}")
        return Assessment(overall="insufficient", claims=[])

    if not isinstance(text, str) or not text.strip():
        return Assessment(overall="insufficient", claims=[])

    raw = text.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        logger.warning(f"[Insight] No JSON in assessor response: {raw[:200]}")
        return Assessment(overall="insufficient", claims=[])
    try:
        data = json.loads(raw[start:end + 1])
    except Exception as e:
        logger.warning(f"[Insight] Assessor JSON parse failed: {e}")
        return Assessment(overall="insufficient", claims=[])

    parsed: list[ClaimAssessment] = []
    for item in data.get("claims") or []:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict") or "").strip().lower()
        if verdict not in ("agree", "partial", "insufficient", "disagree"):
            verdict = "insufficient"
        parsed.append(ClaimAssessment(
            claim=str(item.get("claim") or "").strip() or "(unstated)",
            verdict=verdict,
            support=[str(r) for r in (item.get("support") or [])],
            refute=[str(r) for r in (item.get("refute") or [])],
            notes=str(item.get("notes") or "").strip(),
        ))

    if not parsed:
        return Assessment(overall="insufficient", claims=[])

    result = Assessment.from_claims(parsed)
    logger.info(
        f"[Insight] Assessment: overall={result.overall} "
        f"({', '.join(c.verdict for c in parsed)})"
    )
    return result
