"""
core/insight/facets.py

Module Contract
- Purpose: Decompose an insight theme into 4-6 facet queries (distinct search
  angles, ALWAYS including a counter-evidence facet) via one strict-JSON LLM
  call. Assessment runs additionally yield 2-5 checkable claims. This is the
  step that converts the low-pairwise-similarity / high-collective-signal
  retrieval problem into N per-doc-tractable searches.
- Inputs: InsightIntent, model_manager (generate_once), optional entity_resolver.
- Outputs: FacetPlan (facets + claims; ``fallback=True`` when the LLM call or
  parse failed and a single-facet deterministic plan was substituted).
- Key behaviors: temperature 0.0; robust JSON slicing (find('{')/rfind('}'),
  same discipline as llm_fact_extractor); caps enforced from config
  (INSIGHT_MAX_FACETS); deterministic fallback never raises.
- Side effects: one LLM call; no storage writes.
"""

from __future__ import annotations

import json
import re

from core.insight.types import FacetPlan, FacetQuery, InsightIntent
from utils.logging_utils import get_logger

logger = get_logger("insight_facets")

_DECOMPOSE_SYSTEM = (
    "You decompose a personal-memory research theme into search facets for a "
    "memory-retrieval sweep over a user's own conversation history, notes, and "
    "extracted facts. Respond with STRICT JSON only — no prose, no markdown."
)

_DECOMPOSE_PROMPT = """Theme: {theme}
Mode: {mode}

Produce a JSON object:
{{
  "facets": [
    {{"name": "<short label>",
      "query": "<semantic search sentence>",
      "keywords": ["<exact word or short phrase>", ...],
      "entities": ["<named person/place/thing mentioned in the theme>", ...]}}
  ]{claims_field}
}}

Rules:
- {max_facets} facets maximum, at least 3. Each facet is a DISTINCT angle on
  the theme (different vocabulary, different time frames, adjacent behaviors,
  causes, consequences).
- EXACTLY ONE facet must target COUNTER-EVIDENCE: material that would cut
  AGAINST the theme (times the pattern did not hold, contrary statements,
  exceptions). Name it "counter-evidence".
- keywords are exact words the user might have typed — plain, lowercase,
  no operators. 2-6 per facet.
- entities: only names actually present in the theme text; never invent.
{claims_rules}"""

_CLAIMS_FIELD = ',\n  "claims": ["<checkable claim>", ...]'
_CLAIMS_RULES = (
    "- claims: restate the user's insight as 2-5 separately checkable claims "
    "about their history (specific enough that stored evidence could support "
    "or refute each one)."
)

_STOPWORDS = frozenset(
    "a an the my our your i me is are was were of about on for with and or to in it this that".split()
)


def _fallback_plan(intent: InsightIntent) -> FacetPlan:
    """Deterministic single-facet plan when decomposition fails."""
    words = [
        w for w in re.findall(r"[a-zA-Z][a-zA-Z'-]+", intent.theme.lower())
        if w not in _STOPWORDS
    ]
    facet = FacetQuery(
        name="theme",
        query_text=intent.theme,
        keywords=words[:6],
        entities=[],
    )
    claims = [intent.theme] if intent.kind == "insight_assessment" else []
    return FacetPlan(facets=[facet], claims=claims, fallback=True)


async def decompose(
    intent: InsightIntent,
    model_manager,
    entity_resolver=None,
) -> FacetPlan:
    """Decompose the theme into a FacetPlan. Never raises — falls back to a
    deterministic single-facet plan on any LLM/parse failure."""
    from config.app_config import INSIGHT_DECOMPOSE_MAX_TOKENS, INSIGHT_MAX_FACETS

    is_assessment = intent.kind == "insight_assessment"
    prompt = _DECOMPOSE_PROMPT.format(
        theme=intent.theme,
        mode=("assess a stated insight" if is_assessment else "gather a theme"),
        max_facets=INSIGHT_MAX_FACETS,
        claims_field=_CLAIMS_FIELD if is_assessment else "",
        claims_rules=_CLAIMS_RULES if is_assessment else "",
    )

    try:
        text = await model_manager.generate_once(
            prompt,
            system_prompt=_DECOMPOSE_SYSTEM,
            max_tokens=INSIGHT_DECOMPOSE_MAX_TOKENS,
            temperature=0.0,
        )
    except Exception as e:
        logger.warning(f"[Insight] Decompose LLM call failed: {e}")
        return _fallback_plan(intent)

    if not isinstance(text, str) or not text.strip():
        return _fallback_plan(intent)

    raw = text.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        logger.warning(f"[Insight] No JSON object in decompose response: {raw[:200]}")
        return _fallback_plan(intent)
    try:
        data = json.loads(raw[start:end + 1])
    except Exception as e:
        logger.warning(f"[Insight] Decompose JSON parse failed: {e}")
        return _fallback_plan(intent)

    facets: list[FacetQuery] = []
    for item in (data.get("facets") or [])[:INSIGHT_MAX_FACETS]:
        if not isinstance(item, dict):
            continue
        query_text = str(item.get("query") or "").strip()
        if not query_text:
            continue
        facets.append(FacetQuery(
            name=str(item.get("name") or f"facet-{len(facets) + 1}").strip()[:60],
            query_text=query_text,
            keywords=[str(k).strip().lower() for k in (item.get("keywords") or []) if str(k).strip()][:6],
            entities=[str(e).strip() for e in (item.get("entities") or []) if str(e).strip()][:4],
        ))

    if not facets:
        return _fallback_plan(intent)

    claims = []
    if is_assessment:
        claims = [str(c).strip() for c in (data.get("claims") or []) if str(c).strip()][:5]
        if not claims:
            claims = [intent.theme]

    logger.info(
        f"[Insight] Decomposed theme into {len(facets)} facets"
        + (f", {len(claims)} claims" if claims else "")
    )
    return FacetPlan(facets=facets, claims=claims, fallback=False)
