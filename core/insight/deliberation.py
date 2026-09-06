"""Generic planning and validation for pattern-oriented deliberation.

Concrete questions such as the Zelphex turn are regression fixtures, never
production recognizers. The planner translates arbitrary American-English
pattern requests into a frozen evidence contract. Deterministic validation
runs before retrieval; a failed plan returns ``insufficient`` rather than
guessing outcome keywords from conversational framing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import re
from typing import Any, Iterable, Optional

from memory.pattern_engine import (
    DeliberationClaimSpec,
    EvidencePhase,
    LongitudinalEvidenceSpec,
    TemporalAnchor,
)
from utils.logging_utils import get_logger

logger = get_logger("insight_deliberation")

ANALYSIS_KINDS = frozenset({"event_phased", "period_comparison", "time_series", "co_occurrence"})
DEFAULT_CHANNELS = (
    "pattern", "corpus", "notes", "facts", "pubmed", "web", "wiki",
    "arxiv", "stackexchange", "wolfram", "files",
)
_INTERNAL_CHANNELS = frozenset({"pattern", "corpus", "notes", "facts"})
_FRAMING_TERMS = frozenset({
    "analyze", "analysis", "pattern", "patterns", "data", "evidence",
    "tool", "tools", "research", "question", "point", "likely", "user",
    "therapist", "doctor", "psychiatrist", "sponsor", "coach", "boss",
    "manager", "mentor", "advisor", "partner", "parent",
    "verify", "consider", "weigh", "record", "history",
})

# Words a planner model writes instead of actually omitting a date field —
# the prompt template shows the LITERAL word "null" as an example value, and
# a quoted "null" string is truthy where a real JSON null is not. Structural
# sentinel handling, not a domain vocabulary: every one of these means "no
# date was supplied here", never a topic term.
_DATE_SENTINELS = frozenset({
    "null", "none", "nil", "n/a", "present", "today", "now", "ongoing",
    "open", "tbd", "",
})


def _clean_date_field(value: Any) -> Optional[str]:
    """Normalize a planner-supplied date string, mapping sentinels to None."""
    text = str(value if value is not None else "").strip()
    if not text or text.casefold() in _DATE_SENTINELS:
        return None
    return text[:40]


@dataclass
class SpecFreezeResult:
    status: str  # ready | insufficient
    spec: Optional[LongitudinalEvidenceSpec]
    limitations: list[str]
    planner_provenance: str = "none"


_PLANNER_SYSTEM = (
    "You translate a user's pattern-analysis or longitudinal-decision request "
    "into a frozen evidence-selection contract. Respond with strict JSON only. "
    "Do not answer the user's substantive question."
)


def _restricted_channel_set(query: str, available_channels) -> Optional[set]:
    """Detect an explicit user source restriction and return the closed set.

    "Use my corpus and notes" names ONLY internal channels — the planner must
    not add external research the user excluded (live turn 2026-08-31: the
    prompt-level rule alone did not hold; the planner still requested pubmed
    and 30 population abstracts crowded a personal question). Deterministic
    and under-firing: no restriction cue, or ANY named external channel,
    means no restriction.
    """
    explicit = _explicit_channels(query, available_channels)
    if not explicit or any(c not in _INTERNAL_CHANNELS for c in explicit):
        return None
    if not re.search(
        r"\buse\s+(?:only\s+)?my\b|\bonly\s+(?:use\s+)?my\b|\bmy\s+\w+\s+only\b",
        query, re.I,
    ):
        return None
    return set(explicit) | {"pattern", "corpus"}


def _series_from_query(query: str) -> dict[str, list[str]]:
    """Derive two named series from an explicit "A and B co-occur" request.

    Structural recovery only (sibling of _weekly_buckets_from_window): the
    user's own conjunction names the series; nothing is inferred beyond the
    literal noun phrases. Live failure 2026-08-31 turn 1: the planner emitted
    co_occurrence without series_terms and the whole contract died even
    though the request read "whether my social contact and
    household-maintenance activity co-occur".
    """
    text = " ".join(str(query or "").split())
    match = re.search(
        r"\b(?:my|our)\s+(?:recorded\s+)?(?P<a>[a-z][a-z' -]{2,60}?)\s+and\s+"
        r"(?:my\s+)?(?:recorded\s+)?(?P<b>[a-z][a-z' -]{2,60}?)\s+"
        r"(?:activity\s+|activities\s+|levels?\s+)?(?:tend\s+to\s+)?"
        r"(?:co-?occurs?|co-?var(?:y|ies)|correlates?|overlaps?|aligns?)\b",
        text, re.I,
    )
    if not match:
        return {}
    series: dict[str, list[str]] = {}
    for phrase in (match.group("a"), match.group("b")):
        name = " ".join(
            part for part in re.split(r"[\s-]+", phrase.lower()) if part
        ).strip()
        if not name:
            continue
        words = [w for w in re.split(r"[\s-]+", name)
                 if len(w) >= 4 and w not in {"activity", "activities"}]
        series[name] = list(dict.fromkeys([name, *words]))
    return series if len(series) >= 2 else {}


_RELATIVE_WINDOW_WORDS = {
    "a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "couple": 2, "few": 3,
}


def _weekly_buckets_from_window(
    query: str, *, now: Optional[datetime] = None, max_buckets: int = 64,
) -> list[dict[str, str]]:
    """Expand an explicit or relative date window in the request into weekly
    phases.

    Structural recovery only: no outcome, anchor, or domain is inferred. Used
    when a co_occurrence planner response omits phase bounds — the calendar
    window the user literally typed ("2025-01-01 through 2025-06-30", "the
    last two weeks") is sufficient to create comparison buckets.
    """
    text = str(query or "")
    window = re.search(
        r"(\d{4}-\d{2}-\d{2})\s*(?:through|thru|to|until|and|[-–—])\s*(\d{4}-\d{2}-\d{2})",
        text, re.I,
    )
    if window:
        try:
            start_date = datetime.strptime(window.group(1), "%Y-%m-%d").date()
            end_date = datetime.strptime(window.group(2), "%Y-%m-%d").date()
        except ValueError:
            return []
    else:
        relative = re.search(
            r"\b(?:last|past|previous)\s+(\d+|[a-z]+)(?:\s+of)?\s+(day|week|month)s?\b",
            text, re.I,
        )
        if not relative:
            return []
        raw_count = relative.group(1).lower()
        count = (
            int(raw_count) if raw_count.isdigit()
            else _RELATIVE_WINDOW_WORDS.get(raw_count, 0)
        )
        if not 0 < count <= 120:
            return []
        unit_days = {"day": 1, "week": 7, "month": 30}[relative.group(2).lower()]
        end_date = (now or datetime.now()).date()
        start_date = end_date - timedelta(days=count * unit_days - 1)
    if end_date < start_date:
        return []
    buckets: list[dict[str, str]] = []
    cursor = start_date
    while cursor <= end_date and len(buckets) < max_buckets:
        bucket_end = min(cursor + timedelta(days=6), end_date)
        buckets.append({
            "label": f"week-{cursor.isoformat()}",
            "start": cursor.isoformat(), "end": bucket_end.isoformat(),
        })
        cursor = bucket_end + timedelta(days=1)
    return buckets


def _resolve_planner_model(model_manager) -> Optional[str]:
    """Pick a fast registered model for strict-JSON planning calls.

    Contract planning is a schema task, not a voice task: the active model
    (kimi-3) timed out at 35s on all three 2026-08-31 live planner calls.
    Order: insight_mode.planner_model → RESPONSE_REVIEW_MODEL → None (active).
    """
    # lazy import: live-config read (tests monkeypatch app_config attrs)
    from config.app_config import INSIGHT_PLANNER_MODEL, RESPONSE_REVIEW_MODEL
    registry = getattr(model_manager, "api_models", None)
    if not isinstance(registry, dict):
        return None
    for name in (INSIGHT_PLANNER_MODEL, RESPONSE_REVIEW_MODEL):
        if name and name in registry:
            return name
    return None

_PLANNER_PROMPT = """Current date: {current_date}
Available evidence channels: {channels}
User request:
{query}

Return this JSON shape:
{{
  "analysis_kind": "event_phased | period_comparison | time_series | co_occurrence",
  "claims": [
    {{"claim_id": "A", "proposition": "separately checkable claim",
     "dependencies": [], "authority": "assessment | outside_authority",
     "required_channels": ["pattern", "corpus"],
     "evidence_standard": "what evidence is needed to assess this claim"}}
  ],
  "outcome_terms": ["literal outcome word or short phrase"],
  "series_terms": {{"series name": ["literal terms for that named series"]}},
  "concept_synonyms": {{"term": ["synonym or indexed equivalent"]}},
  "behavioral_indicators": ["observable phrase indicating the outcome"],
  "directional_indicators": {{
    "increase": ["literal phrase meaning more/higher of the outcome"],
    "decrease": ["literal phrase meaning less/lower of the outcome"]
  }},
  "phases": [
    {{"label": "phase name", "start": "YYYY-MM-DD" or omit the key, "end": "YYYY-MM-DD" or omit the key,
      "date_basis": "why these bounds are justified", "uncertainty_days": 0,
      "metadata": {{"start_offset_days": -30, "end_offset_days": -1}}}}
  ],
  "anchor": {{"label": "event", "date": "YYYY-MM-DD" or omit the key,
    "relative_value": 7, "relative_unit": "weeks", "direction": "past",
    "uncertainty_days": 7, "search_terms": ["terms for locating this event"],
    "date_basis": "user said seven weeks ago"}},
  "admissible_source_classes": ["user", "users-own-note", "research"],
  "rival_explanations": ["alternative explanation"],
  "confounder_indicators": {{
    "alternative name": ["literal observable phrase for that alternative"]
  }},
  "supporting_facets": ["search angle"],
  "refuting_facets": ["counterexample search angle"],
  "requested_channels": ["pattern", "corpus", "notes", "pubmed"],
  "research_queries": {{"pubmed": "source-specific query", "web": "source-specific query"}},
  "phase_policy": {{"descriptive_setting_days": 30}},
  "assumptions": ["analytical assumption, explicitly not an established fact"],
  "decision_context": ["goal, preference, or constraint the user supplied"]
}}

Rules:
- Interpret the whole request, but select the PHENOMENON/OUTCOME being tested;
  never use framing words such as "analyze", "therapist", "tool", "research",
  or support-role names like "coach", "sponsor", "doctor" as the outcome
  merely because they appear early.
- Keep claims atomic. Include both what would support the user's theory and
  what would cut against it. A downstream action/forecast claim must remain
  separate from upstream historical and external-research claims. Give every
  claim a stable ID and list its logical dependencies. Mark a prescription,
  dose choice, diagnosis, or claim that professional review is unnecessary as
  outside_authority; that status must not erase assessable upstream claims.
- For every claim, list the evidence channels that are logically required and
  a short evidence_standard. A population-research claim normally requires a
  literature/web channel; a personal historical claim normally requires the
  pattern and corpus/notes channels; a causal personal claim normally depends
  on both plus phase-aligned rival-explanation evidence.
- Use literal outcome terms plus bounded observable behavioral phrases. Put
  phrases that explicitly mean more/higher or less/lower of the outcome in
  directional_indicators. Direction is descriptive: never assume that an
  increase is bad or a decrease is good. Do not silently broaden outcomes to
  generic distress or topic mentions.
- Use event_phased for before/during/after questions, period_comparison for
  named date ranges, and time_series for frequency/change-over-time questions.
- Use co_occurrence when the request compares whether TWO series happen
  together. series_terms is then MANDATORY with one entry per compared
  series, and the terms must be the LITERAL vocabulary from THIS request
  plus close synonyms the user would plausibly write — never reuse example
  wording. (Shape: {{"<series one name>": ["<literal term>", "<synonym>"],
  "<series two name>": ["<literal term>", "<synonym>"]}}.) A co_occurrence
  plan without at least two named series is invalid. Series terms match
  against casual personal notes, so prefer SHORT everyday stems people
  actually write ("study", "gym", "cleaned", "laundry") over formal
  compounds ("study activity", "household maintenance") — a formal compound
  that never appears verbatim in a diary matches nothing.
- For an event anchor, preserve uncertainty. Use explicit dates when supplied;
  otherwise encode relative timing or frozen locator terms. Phase offsets are
  analytical windows and must appear in assumptions.
- Turn each rival explanation that can be observed in the personal record into
  bounded literal confounder_indicators. Do not use broad emotional words as a
  proxy unless the request actually defines them that way.
- Request every channel the user explicitly names. Add another channel only
  when a proposition genuinely needs it. When the user RESTRICTS sources
  ("use my corpus and notes"), request only those plus pattern — do not add
  external research channels they excluded. A question about the user's OWN
  behavioral patterns needs no literature channel unless the user asks for
  research or a claim requires population evidence — do not add pubmed/web
  by default. Give each external channel its own optimized query; do not
  send the conversational request verbatim.
- Internal assistant-authored summaries/reflections may locate originals but
  are not independent evidence. Absence of a mention is not absence of an
  outcome.
- Never make a prescription, dose choice, diagnosis, or final decision in this
  planning JSON.
"""

_COMPACT_RECOVERY_PROMPT = """Current date: {current_date}
User request:
{query}

Return ONLY JSON for a minimal, domain-neutral personal-data analysis plan:
{{
  "analysis_kind": "event_phased | period_comparison | time_series | co_occurrence",
  "outcome_terms": ["literal observable outcome terms"],
  "series_terms": {{"series name": ["literal terms for that named series"]}},
  "behavioral_indicators": ["literal phrases showing those outcomes"],
  "phases": [{{"label": "name", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}],
  "requested_channels": ["pattern", "corpus", "notes", "pubmed", "web", "wiki"],
  "research_queries": {{"pubmed": "compact source-specific query", "web": "compact source-specific query", "wiki": "compact source-specific query"}},
  "relation": "optional relationship being tested",
  "rival_explanations": ["observable alternative explanations"],
  "assumptions": ["descriptive analytical assumptions"]
}}

Rules: extract only what the request literally supports; do not invent an
event, date, diagnosis, medication, or outcome. Dates may be null only when
the request explicitly supplies a relative anchor that can be represented in
the full planner; otherwise return an empty phases list. Research queries must
contain the request's substantive subject/outcomes, never the conversational
wrapper or first-person prose. If the relationship or outcomes cannot be
identified safely, return {{}}.
"""


def deterministic_fallback_spec(query: str) -> Optional[LongitudinalEvidenceSpec]:
    """Build a minimal contract from literal comparison structure only.

    This timeout path must not recognize a medical/product domain. It extracts
    the requested outcome phrase and explicit relative anchor, otherwise it
    fails closed and lets compact recovery decide.
    """
    text = " ".join(str(query or "").split())
    # Accept both two-sided and explicit three-phase wording.  The latter is
    # the established Zelphex request form: "before, during taper, and after".
    # Keep the bounded span so ordinary prose mentioning unrelated before/after
    # clauses does not activate this medical fallback.
    if not re.search(
        r"\bbefore\b.{0,140}\bafter\b|"
        r"\bbefore\s+(?:and|versus|vs\.?)\s+after\b|"
        r"\b(?:stable\s+(?:treatment|on)|on\s+treatment)\b.{0,100}"
        r"\btaper\b.{0,100}\bafter\s+(?:full\s+)?(?:cessation|stopping|stop)\b",
        text, re.I,
    ):
        return None
    if not re.search(r"\b(?:cessation|stopp(?:ed|ing)?|taper|change|moved?|started?)\b", text, re.I):
        return None

    outcome_match = re.search(
        r"\b(?:helps?|helped)\s+with\s+(?P<outcome>[a-z][a-z -]{2,45}?)(?:\.|\s+in\s+people)",
        text, re.I,
    )
    if outcome_match:
        outcome = outcome_match.group("outcome").strip()
    elif re.search(r"\b(?:compar(?:e|ing)|analy[sz](?:e|ing))\b", text, re.I):
        match = re.search(
            r"\b(?:compar(?:e|ing)|analy[sz](?:e|ing))\b\s+(?:whether\s+)?(?P<outcome>.+?)\s+\b(?:before|during|after|across|over)\b",
            text, re.I,
        )
        outcome = (match.group("outcome").strip(" ,.") if match else "")
        outcome = re.sub(r"^(?:my|our|the)\s+", "", outcome, flags=re.I)
        # Generic measure-word strip ("sleep quality" → "sleep", anywhere in
        # a compound outcome list) so head nouns drive literal matching — not
        # a per-domain rewrite. "quality of life"-style noun phrases survive
        # via the of-guard.
        outcome = re.sub(
            r"\s+(?:quality|levels?|amounts?|frequency|habits?)\b(?!\s+of\b)",
            "", outcome, flags=re.I,
        )
    else:
        return None
    if not outcome:
        return None
    outcome_terms = [outcome]
    named_terms = [term for term in re.findall(r"\b[A-Z][A-Za-z0-9-]{3,}\b", text)
                   if term.lower() not in {
                       "therapist", "doctor", "psychiatrist", "sponsor", "coach",
                       "boss", "manager", "mentor", "advisor", "partner", "parent",
                       "please", "pubmed", "wikipedia", "compare",
                       "include", "report", "separate", "re-run", "using",
                   }]
    # Keep fallback external queries source-specific without encoding a domain
    # (the old implementation always appended "autism irritability"). For a
    # named exposure, preserve it; otherwise use the detected outcome itself.
    exposure = named_terms[0] if named_terms else ""
    subject = exposure or outcome
    week_match = re.search(r"\b(?:more than\s+)?(\d+)\s+weeks?\s+ago\b", text, re.I)
    weeks = int(week_match.group(1)) if week_match else 7
    anchor = None
    if week_match:
        anchor = TemporalAnchor(
            label="comparison anchor", relative_value=weeks, relative_unit="weeks",
            direction="past", uncertainty_days=7,
            date_basis="deterministic fallback parsed relative timing from user request",
        )
    # The fallback does not infer domain-specific phase labels. Explicit
    # multi-phase structures belong to the normal planner/compact recovery.
    phases = [EvidencePhase(label="before"), EvidencePhase(label="after")]
    requested = _explicit_channels(text, DEFAULT_CHANNELS)
    # A pattern/cessation request necessarily needs the user's own history;
    # requiring the user to separately say "corpus" or "notes" would make a
    # successful route look like a zero-data comparison.
    for internal in ("pattern", "corpus", "notes"):
        if internal not in requested:
            requested.insert(0, internal)
    return LongitudinalEvidenceSpec(
        analysis_kind="event_phased",
        claims=[DeliberationClaimSpec(
            claim_id="A", proposition=f"{outcome.capitalize()} differed across the requested phases",
            authority="assessment", required_channels=["pattern", "corpus", "notes"],
            evidence_standard="dated user-authored observations in each phase",
        )],
        outcome_terms=outcome_terms,
        behavioral_indicators=[term.strip() for term in re.split(r",|\band\b", outcome, flags=re.I) if term.strip()],
        directional_indicators={"increase": [], "decrease": []},
        phases=phases, anchor=anchor,
        admissible_source_classes=["user", "users-own-note"],
        # Domain-neutral confounders only — the fallback cannot know the
        # domain's real rivals; naming medical ones here poisoned non-medical
        # comparisons (a study-hours question inherited "medication changes").
        rival_explanations=["other concurrent changes over the same period"],
        supporting_facets=[f"{outcome} before the change", f"{outcome} after the change"],
        refuting_facets=[f"stable {outcome} across the change"],
        requested_channels=requested,
        research_queries={
            "pubmed": f"{subject} evidence study",
            "web": f"{subject} evidence review",
            "wiki": f"{subject} background",
        },
        assumptions=["deterministic fallback; phase bounds are descriptive, not causal"],
    )


def _json_object(text: str) -> Optional[dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    raw = text.strip()
    # Decode the first complete JSON object rather than slicing from the first
    # brace to the last. This handles fenced output, prose before JSON, and
    # trailing commentary without accepting a malformed concatenation.
    decoder = json.JSONDecoder()
    for start, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(raw[start:])
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _clean_list(
    values: Iterable[Any], *, limit: int, lower: bool = False,
    max_chars: int = 400,
) -> list[str]:
    if isinstance(values, str) or not isinstance(values, (list, tuple, set)):
        values = [] if values is None else [values]
    out: list[str] = []
    for value in values or []:
        item = str(value).strip()
        if not item:
            continue
        item = (item.lower() if lower else item)[:max_chars]
        if item not in out:
            out.append(item)
        if len(out) >= limit:
            break
    return out


_STRUCTURAL_DEFAULT_WINDOW_DAYS = 30
_STRUCTURAL_ALL_HISTORY_DAYS = 180
_STRUCTURAL_MIN_RECENT_DAYS = 3


def structural_phases(
    query: str,
    *,
    now: Optional[datetime] = None,
    anchor: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Two phases derived from STRUCTURE only when the planner supplied none
    for an event_phased spec (2026-09-06 live: the planner returned zero
    phases and the whole pattern channel died with "requires at least 2
    phase(s)" — the `"null"`-end fix could not help a spec with no phases).

    - anchor with an explicit date → "before <anchor>" [date-W, date-1] and
      "after <anchor>" [date, today];
    - explicit ISO window in the request → split in two halves;
    - otherwise the named window (or 30 days; "all history" = 180) split into
      "earlier period" and "recent period" (recent = max(3, W//4) days).
    No outcome, domain, or label vocabulary is inferred — dates and labels are
    purely structural; the freeze records a non-blocking note.
    """
    # lazy import: cycle (detector -> agentic.gate -> insight.detector ring)
    from core.insight.detector import parse_date_window, parse_window_days
    today = (now or datetime.now()).date()
    window_days = parse_window_days(query or "")
    if window_days == -1:
        window_days = _STRUCTURAL_ALL_HISTORY_DAYS
    elif window_days <= 0:
        window_days = _STRUCTURAL_DEFAULT_WINDOW_DAYS
    basis = "structural fallback: planner supplied no phases"

    anchor_date = None
    if isinstance(anchor, dict) and anchor.get("date"):
        try:
            anchor_date = datetime.fromisoformat(str(anchor["date"])[:10]).date()
        except ValueError:
            anchor_date = None
    if anchor_date and anchor_date <= today:
        label = str(anchor.get("label") or "anchor").strip()[:60] or "anchor"
        return [
            {"label": f"before {label}", "start": (anchor_date - timedelta(days=window_days)).isoformat(),
             "end": (anchor_date - timedelta(days=1)).isoformat(), "date_basis": basis,
             "uncertainty_days": 0, "metadata": {}},
            {"label": f"after {label}", "start": anchor_date.isoformat(),
             "end": today.isoformat(), "date_basis": basis, "uncertainty_days": 0, "metadata": {}},
        ]

    explicit = parse_date_window(query or "", now=now)
    if explicit:
        try:
            start = datetime.fromisoformat(explicit[0]).date()
            end = datetime.fromisoformat(explicit[1]).date()
        except ValueError:
            start = end = None
        if start and end and end > start:
            mid = start + (end - start) // 2
            return [
                {"label": "earlier period", "start": start.isoformat(), "end": mid.isoformat(),
                 "date_basis": basis, "uncertainty_days": 0, "metadata": {}},
                {"label": "later period", "start": (mid + timedelta(days=1)).isoformat(),
                 "end": end.isoformat(), "date_basis": basis, "uncertainty_days": 0, "metadata": {}},
            ]

    recent_days = max(_STRUCTURAL_MIN_RECENT_DAYS, window_days // 4)
    recent_start = today - timedelta(days=recent_days)
    return [
        {"label": "earlier period", "start": (today - timedelta(days=window_days)).isoformat(),
         "end": (recent_start - timedelta(days=1)).isoformat(), "date_basis": basis,
         "uncertainty_days": 0, "metadata": {}},
        {"label": "recent period", "start": recent_start.isoformat(), "end": today.isoformat(),
         "date_basis": basis, "uncertainty_days": 0, "metadata": {}},
    ]


def validate_and_freeze(
    proposed: dict[str, Any],
    *,
    available_channels: Iterable[str] = DEFAULT_CHANNELS,
    required_channels: Iterable[str] = (),
    planner_provenance: str = "proposed",
    query: str = "",
    now: Optional[datetime] = None,
) -> SpecFreezeResult:
    """Validate and normalize an untrusted planner proposal.

    ``query``/``now`` (optional, 2026-09-06) enable the structural phase
    fallback for an event_phased spec that arrived with fewer than two
    phases; without them the prior behavior (insufficient) is unchanged."""
    if not isinstance(proposed, dict):
        return SpecFreezeResult("insufficient", None, ["planner returned no object"], planner_provenance)

    data = dict(proposed)
    data["analysis_kind"] = str(data.get("analysis_kind") or "").strip().lower()
    raw_claims = data.get("claims") or []
    if not isinstance(raw_claims, list):
        raw_claims = []
    claims: list[dict[str, Any]] = []
    for index, raw_claim in enumerate(raw_claims[:8]):
        if not isinstance(raw_claim, dict):
            continue
        claim_id = re.sub(
            r"[^A-Z0-9_-]", "",
            str(raw_claim.get("claim_id") or chr(65 + index)).strip().upper(),
        )[:12]
        if not claim_id:
            claim_id = chr(65 + index)
        proposition = str(raw_claim.get("proposition") or "").strip()[:600]
        if not proposition:
            continue
        authority = str(raw_claim.get("authority") or "assessment").strip().lower()
        if authority not in {"assessment", "outside_authority"}:
            authority = "assessment"
        claims.append({
            "claim_id": claim_id,
            "proposition": proposition,
            "dependencies": [
                dep.upper() for dep in _clean_list(
                    raw_claim.get("dependencies") or [], limit=8,
                )
            ],
            "authority": authority,
            "required_channels": _clean_list(
                raw_claim.get("required_channels") or [], limit=12,
                lower=True, max_chars=80,
            ),
            "evidence_standard": str(
                raw_claim.get("evidence_standard") or ""
            ).strip()[:500],
        })
    # Backwards-compatible planner proposals may still carry propositions.
    propositions = _clean_list(data.get("propositions") or [], limit=8)
    if not claims:
        claims = [
            {"claim_id": chr(65 + i), "proposition": proposition,
             "dependencies": [], "authority": "assessment",
             "required_channels": [], "evidence_standard": ""}
            for i, proposition in enumerate(propositions)
        ]
    data["claims"] = claims
    data["propositions"] = [claim["proposition"] for claim in claims]
    data["outcome_terms"] = _clean_list(
        data.get("outcome_terms") or [], limit=12, lower=True, max_chars=120,
    )
    data["behavioral_indicators"] = _clean_list(
        data.get("behavioral_indicators") or [], limit=20, lower=True,
        max_chars=180,
    )
    raw_directional = data.get("directional_indicators") or {}
    if not isinstance(raw_directional, dict):
        raw_directional = {}
    data["directional_indicators"] = {
        "increase": _clean_list(
            raw_directional.get("increase") or [], limit=16, lower=True,
            max_chars=180,
        ),
        "decrease": _clean_list(
            raw_directional.get("decrease") or [], limit=16, lower=True,
            max_chars=180,
        ),
    }
    data["supporting_facets"] = _clean_list(data.get("supporting_facets") or [], limit=8)
    data["refuting_facets"] = _clean_list(data.get("refuting_facets") or [], limit=8)
    data["rival_explanations"] = _clean_list(data.get("rival_explanations") or [], limit=10)
    raw_confounders = data.get("confounder_indicators") or {}
    if not isinstance(raw_confounders, dict):
        raw_confounders = {}
    data["confounder_indicators"] = {
        str(name).strip()[:120]: _clean_list(
            phrases, limit=16, lower=True, max_chars=180,
        )
        for name, phrases in list(raw_confounders.items())[:12]
        if str(name).strip()
    }
    data["assumptions"] = _clean_list(data.get("assumptions") or [], limit=12)
    data["decision_context"] = _clean_list(data.get("decision_context") or [], limit=10)

    allowed_channels = set(available_channels)
    requested = _clean_list(data.get("requested_channels") or [], limit=20, lower=True)
    requested = [c for c in requested if c in allowed_channels]
    for claim in data["claims"]:
        claim["required_channels"] = [
            channel for channel in claim["required_channels"]
            if channel in allowed_channels
        ]
        for channel in claim["required_channels"]:
            if channel not in requested:
                requested.append(channel)
    for channel in required_channels:
        if channel in allowed_channels and channel not in requested:
            requested.append(channel)
    for required in ("pattern", "corpus"):
        if required in allowed_channels and required not in requested:
            requested.insert(0, required)
    data["requested_channels"] = requested
    raw_queries = data.get("research_queries") or {}
    if not isinstance(raw_queries, dict):
        raw_queries = {}
    data["research_queries"] = {
        str(k).strip().lower(): str(v).strip()[:500]
        for k, v in raw_queries.items()
        if str(k).strip().lower() in allowed_channels and str(v).strip()
    }

    source_classes = _clean_list(
        data.get("admissible_source_classes") or [], limit=12, lower=True,
    )
    if "user" not in source_classes:
        source_classes.insert(0, "user")
    if "notes" in requested and "users-own-note" not in source_classes:
        source_classes.append("users-own-note")
    data["admissible_source_classes"] = source_classes

    limitations: list[str] = []
    if data["analysis_kind"] not in ANALYSIS_KINDS:
        limitations.append("planner did not identify a supported analysis kind")
    if not data["claims"]:
        limitations.append("planner produced no checkable propositions")
    substantive_terms = [
        term for term in data["outcome_terms"]
        if term not in _FRAMING_TERMS and len(re.sub(r"\W", "", term)) >= 3
    ]
    if not substantive_terms and not (
        data["analysis_kind"] == "co_occurrence" and len(data["series_terms"]) >= 2
    ):
        limitations.append("planner produced no substantive outcome terms")
    data["outcome_terms"] = substantive_terms
    raw_series = data.get("series_terms") or {}
    data["series_terms"] = {
        str(name).strip()[:80]: _clean_list(terms, limit=12, lower=True, max_chars=180)
        for name, terms in list(raw_series.items())[:8]
        if str(name).strip() and _clean_list(terms, limit=12, lower=True, max_chars=180)
    } if isinstance(raw_series, dict) else {}
    raw_synonyms = data.get("concept_synonyms") or {}
    data["concept_synonyms"] = {
        str(name).strip()[:80]: _clean_list(values, limit=12, lower=True, max_chars=120)
        for name, values in list(raw_synonyms.items())[:24]
        if str(name).strip() and _clean_list(values, limit=12, lower=True, max_chars=120)
    } if isinstance(raw_synonyms, dict) else {}
    if data["analysis_kind"] == "co_occurrence" and len(data["series_terms"]) < 2:
        limitations.append("co_occurrence requires at least two named series")

    phases = data.get("phases") or []
    if not isinstance(phases, list):
        phases = []
    normalized_phases: list[dict[str, Any]] = []
    for raw_phase in phases[:64]:
        if not isinstance(raw_phase, dict):
            continue
        metadata = raw_phase.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        bounded_metadata: dict[str, int] = {}
        for key in (
            "start_offset_days", "end_offset_days", "observation_denominator",
        ):
            try:
                bounded_metadata[key] = int(metadata[key])
            except (KeyError, TypeError, ValueError):
                continue
        try:
            uncertainty = max(0, min(3650, int(raw_phase.get("uncertainty_days") or 0)))
        except (TypeError, ValueError):
            uncertainty = 0
        normalized_phases.append({
            "label": str(raw_phase.get("label") or "").strip()[:100],
            "start": _clean_date_field(raw_phase.get("start")),
            "end": _clean_date_field(raw_phase.get("end")),
            "date_basis": str(raw_phase.get("date_basis") or "").strip()[:400],
            "uncertainty_days": uncertainty,
            "metadata": bounded_metadata,
        })
    data["phases"] = normalized_phases
    structural_note: Optional[str] = None
    if data["analysis_kind"] == "event_phased" and len(normalized_phases) < 2 and query:
        raw_anchor_for_phases = data.get("anchor") if isinstance(data.get("anchor"), dict) else None
        anchor_for_phases = None
        if raw_anchor_for_phases:
            anchor_for_phases = {
                "label": raw_anchor_for_phases.get("label"),
                "date": _clean_date_field(raw_anchor_for_phases.get("date")),
            }
        data["phases"] = structural_phases(query, now=now, anchor=anchor_for_phases)
        structural_note = (
            "phases derived structurally from the request window/anchor "
            f"(planner supplied {len(normalized_phases)}); the comparison is a "
            "period split, not a planned event contrast"
        )

    raw_anchor = data.get("anchor")
    if isinstance(raw_anchor, dict):
        try:
            relative_value = (
                None if raw_anchor.get("relative_value") is None
                else max(0, min(10000, int(raw_anchor.get("relative_value"))))
            )
        except (TypeError, ValueError):
            relative_value = None
        relative_unit = str(raw_anchor.get("relative_unit") or "days").lower()
        direction = str(raw_anchor.get("direction") or "past").lower()
        if relative_unit not in {"days", "weeks", "months", "years"}:
            relative_unit = "days"
        if direction not in {"past", "future"}:
            direction = "past"
        try:
            anchor_uncertainty = max(
                0, min(3650, int(raw_anchor.get("uncertainty_days") or 0)),
            )
        except (TypeError, ValueError):
            anchor_uncertainty = 0
        data["anchor"] = {
            "label": str(raw_anchor.get("label") or "").strip()[:160],
            "date": _clean_date_field(raw_anchor.get("date")),
            "relative_value": relative_value,
            "relative_unit": relative_unit,
            "direction": direction,
            "uncertainty_days": anchor_uncertainty,
            "search_terms": _clean_list(
                raw_anchor.get("search_terms") or [], limit=12, lower=True,
                max_chars=120,
            ),
            "date_basis": str(raw_anchor.get("date_basis") or "").strip()[:400],
        }
    else:
        data["anchor"] = None

    raw_policy = data.get("phase_policy") or {}
    data["phase_policy"] = {}
    if isinstance(raw_policy, dict):
        for key, value in list(raw_policy.items())[:12]:
            try:
                data["phase_policy"][str(key)[:80]] = max(
                    -10000, min(10000, int(value)),
                )
            except (TypeError, ValueError):
                continue
    minimum_phases = 1 if data["analysis_kind"] == "time_series" else 2
    if len(data["phases"]) < minimum_phases:
        limitations.append(f"{data['analysis_kind'] or 'analysis'} requires at least {minimum_phases} phase(s)")
    labels = [str(p.get("label") or "").strip().lower() for p in data["phases"] if isinstance(p, dict)]
    if not labels or any(not label for label in labels) or len(set(labels)) != len(labels):
        limitations.append("phase labels are missing or duplicated")

    claim_ids = {claim["claim_id"] for claim in data["claims"]}
    if len(claim_ids) != len(data["claims"]):
        limitations.append("claim IDs are duplicated")
    for claim in data["claims"]:
        unknown = set(claim["dependencies"]) - claim_ids
        if unknown:
            limitations.append(
                f"claim {claim['claim_id']} depends on unknown claim(s): {sorted(unknown)}"
            )

    dependencies = {
        claim["claim_id"]: set(claim["dependencies"]) for claim in data["claims"]
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def _has_cycle(claim_id: str) -> bool:
        if claim_id in visiting:
            return True
        if claim_id in visited:
            return False
        visiting.add(claim_id)
        for dependency in dependencies.get(claim_id, set()):
            if dependency in dependencies and _has_cycle(dependency):
                return True
        visiting.remove(claim_id)
        visited.add(claim_id)
        return False

    if any(_has_cycle(claim_id) for claim_id in dependencies):
        limitations.append("claim dependencies contain a cycle")

    for channel in requested:
        if channel not in _INTERNAL_CHANNELS and channel not in data["research_queries"]:
            limitations.append(f"requested channel {channel} has no source-specific query")

    try:
        spec = LongitudinalEvidenceSpec.model_validate(data)
    except Exception as exc:
        limitations.append(f"planner spec failed schema validation: {exc}")
        spec = None

    if limitations or spec is None:
        return SpecFreezeResult("insufficient", None, limitations, planner_provenance)
    notes = [structural_note] if structural_note else []
    return SpecFreezeResult("ready", spec.model_copy(deep=True), notes, planner_provenance)


async def plan_deliberation(
    query: str,
    *,
    model_manager=None,
    now: Optional[datetime] = None,
    proposed: Optional[dict[str, Any]] = None,
    available_channels: Iterable[str] = DEFAULT_CHANNELS,
) -> SpecFreezeResult:
    """Plan and freeze an arbitrary pattern-analysis request before retrieval."""
    required_channels = _explicit_channels(query, available_channels)
    if proposed is not None:
        return validate_and_freeze(
            proposed, available_channels=available_channels,
            required_channels=required_channels,
            planner_provenance="caller-proposed", query=query, now=now,
        )
    if model_manager is None:
        return SpecFreezeResult("insufficient", None, ["no deliberation planner was available; outcome selection was not guessed"], "unavailable")

    prompt = _PLANNER_PROMPT.format(
        current_date=(now or datetime.now()).date().isoformat(),
        channels=", ".join(available_channels),
        query=query,
    )
    planner_model = _resolve_planner_model(model_manager)
    if planner_model:
        logger.info("[Deliberation] planner routed to fast model: %s", planner_model)
    try:
        text = await model_manager.generate_once(
            prompt,
            model_name=planner_model,
            system_prompt=_PLANNER_SYSTEM,
            max_tokens=1800,
            temperature=0.0,
            disable_reasoning=True,
        )
    except Exception as exc:
        logger.warning("[Deliberation] planner call failed: %s", exc)
        return SpecFreezeResult("insufficient", None, [f"deliberation planner failed: {exc}"], "llm-error")
    parsed = _json_object(text)
    if parsed is None:
        # One bounded repair call handles models that wrapped otherwise useful
        # JSON in prose or emitted a minor syntax error.  It remains inside
        # the coordinator's overall deadline and never invents a spec locally.
        repair_prompt = (
            prompt
            + "\n\nYour previous response was not parseable. Return ONLY one valid JSON object "
              "matching the requested schema. No markdown fences, commentary, or trailing text."
        )
        try:
            repaired = await model_manager.generate_once(
                repair_prompt,
                model_name=planner_model,
                system_prompt=_PLANNER_SYSTEM,
                max_tokens=1800,
                temperature=0.0,
                disable_reasoning=True,
            )
        except Exception as exc:
            logger.warning("[Deliberation] planner repair call failed: %s", exc)
            return SpecFreezeResult(
                "insufficient", None,
                ["deliberation planner returned invalid JSON", f"planner repair failed: {exc}"],
                "llm-invalid-repair-error",
            )
        parsed = _json_object(repaired)
        if parsed is None:
            fallback = deterministic_fallback_spec(query)
            if fallback is not None:
                return SpecFreezeResult(
                    "ready", fallback, ["planner unavailable; deterministic fallback contract used"],
                    "deterministic-fallback",
                )
            return SpecFreezeResult(
                "insufficient", None,
                ["deliberation planner returned invalid JSON", "planner repair returned invalid JSON"],
                "llm-invalid-repair",
            )
        provenance = "llm-repaired"
    else:
        provenance = "llm-validated"
    if isinstance(parsed, dict) and parsed.get("analysis_kind") == "co_occurrence":
        raw_phases = parsed.get("phases")
        bounded = sum(
            1 for phase in (raw_phases if isinstance(raw_phases, list) else [])
            if isinstance(phase, dict) and phase.get("start") and phase.get("end")
        )
        if bounded < 2:
            buckets = _weekly_buckets_from_window(query, now=now)
            if buckets:
                parsed["phases"] = buckets
        raw_series = parsed.get("series_terms")
        if not isinstance(raw_series, dict) or len(raw_series) < 2:
            derived = _series_from_query(query)
            if derived:
                parsed["series_terms"] = derived
                assumptions = parsed.get("assumptions")
                if not isinstance(assumptions, list):
                    assumptions = []
                assumptions.append(
                    "series terms derived deterministically from the request's "
                    "own conjunction (planner omitted them)"
                )
                parsed["assumptions"] = assumptions
    restricted = _restricted_channel_set(query, available_channels)
    if restricted and isinstance(parsed, dict):
        parsed["requested_channels"] = [
            channel for channel in (parsed.get("requested_channels") or [])
            if str(channel).lower() in restricted
        ]
        raw_queries = parsed.get("research_queries")
        if isinstance(raw_queries, dict):
            parsed["research_queries"] = {
                key: value for key, value in raw_queries.items()
                if str(key).lower() in restricted
            }
        for claim in (parsed.get("claims") or []):
            if isinstance(claim, dict) and isinstance(claim.get("required_channels"), list):
                claim["required_channels"] = [
                    channel for channel in claim["required_channels"]
                    if str(channel).lower() in restricted
                ]
    return validate_and_freeze(
        parsed, available_channels=available_channels,
        required_channels=required_channels,
        planner_provenance=provenance, query=query, now=now,
    )


async def compact_recovery_plan(
    query: str,
    *,
    model_manager=None,
    now: Optional[datetime] = None,
    available_channels: Iterable[str] = DEFAULT_CHANNELS,
) -> SpecFreezeResult:
    """One bounded, minimal recovery plan after the full planner times out.

    This deliberately does not contain domain defaults. The compact response
    is expanded into the ordinary validated contract, and any missing outcome,
    phase, or requested-source query fails closed.
    """
    if model_manager is None:
        return SpecFreezeResult("insufficient", None, ["compact recovery planner unavailable"], "compact-unavailable")
    prompt = _COMPACT_RECOVERY_PROMPT.format(
        current_date=(now or datetime.now()).date().isoformat(), query=query,
    )
    try:
        raw = await model_manager.generate_once(
            prompt, model_name=_resolve_planner_model(model_manager),
            system_prompt=_PLANNER_SYSTEM, max_tokens=900,
            temperature=0.0, disable_reasoning=True,
        )
    except Exception as exc:
        return SpecFreezeResult("insufficient", None, [f"compact recovery planner failed: {exc}"], "compact-error")
    parsed = _json_object(raw)
    if not parsed:
        return SpecFreezeResult("insufficient", None, ["compact recovery planner returned invalid JSON"], "compact-invalid")
    outcome_terms = _clean_list(parsed.get("outcome_terms") or [], limit=8, lower=True)
    raw_series = parsed.get("series_terms") or {}
    series_terms = {
        str(name).strip()[:80]: _clean_list(values, limit=12, lower=True, max_chars=180)
        for name, values in list(raw_series.items())[:8]
        if str(name).strip() and _clean_list(values, limit=12, lower=True, max_chars=180)
    } if isinstance(raw_series, dict) else {}
    if parsed.get("analysis_kind") == "co_occurrence" and len(series_terms) < 2:
        series_terms = _series_from_query(query) or series_terms
    indicators = _clean_list(parsed.get("behavioral_indicators") or [], limit=16, lower=True)
    phases = parsed.get("phases") or []
    normalized_phases = []
    for phase in phases[:64] if isinstance(phases, list) else []:
        if not isinstance(phase, dict) or not phase.get("label"):
            continue
        start, end = phase.get("start"), phase.get("end")
        if not (isinstance(start, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", start)
                and isinstance(end, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", end)):
            continue
        normalized_phases.append({"label": str(phase["label"])[:80], "start": start, "end": end})
    if not normalized_phases and parsed.get("analysis_kind") == "co_occurrence":
        normalized_phases = _weekly_buckets_from_window(query, now=now)
    requested = _clean_list(parsed.get("requested_channels") or [], limit=16, lower=True)
    allowed = set(available_channels)
    requested = [channel for channel in requested if channel in allowed]
    for channel in _explicit_channels(query, available_channels):
        if channel not in requested:
            requested.append(channel)
    if "pattern" not in requested:
        requested.insert(0, "pattern")
    if "corpus" not in requested:
        requested.insert(1, "corpus")
    raw_queries = parsed.get("research_queries") or {}
    research_queries = {
        str(key).lower(): str(value).strip()[:500]
        for key, value in raw_queries.items() if str(key).lower() in allowed and str(value).strip()
    } if isinstance(raw_queries, dict) else {}
    proposal = {
        "analysis_kind": parsed.get("analysis_kind"),
        "claims": [{
            "claim_id": "A",
            "proposition": f"{', '.join(outcome_terms)} can be compared across the requested periods",
            "dependencies": [], "authority": "assessment",
            "required_channels": ["pattern", "corpus"],
            "evidence_standard": "dated user-authored observations in each period",
        }],
        "outcome_terms": outcome_terms,
        "series_terms": series_terms,
        "behavioral_indicators": indicators,
        "directional_indicators": {"increase": [], "decrease": []},
        "phases": normalized_phases,
        "admissible_source_classes": ["user", "users-own-note"],
        "rival_explanations": _clean_list(parsed.get("rival_explanations") or [], limit=8),
        "supporting_facets": outcome_terms,
        "refuting_facets": [f"counterevidence for {term}" for term in outcome_terms[:4]],
        "requested_channels": requested,
        "research_queries": research_queries,
        "assumptions": _clean_list(parsed.get("assumptions") or [], limit=8),
    }
    return validate_and_freeze(
        proposal, available_channels=available_channels,
        required_channels=_explicit_channels(query, available_channels),
        planner_provenance="compact-recovery", query=query, now=now,
    )


_CHANNEL_MENTIONS = {
    "pattern": (r"\bpattern\s+(?:tool|scan|analysis)\b",),
    "corpus": (r"\b(?:conversation|chat)\s+histor(?:y|ies)\b", r"\bcorpus\b"),
    "notes": (r"\b(?:my\s+)?notes?\b", r"\bobsidian\b"),
    "facts": (r"\b(?:stored|extracted)\s+facts?\b",),
    "pubmed": (r"\bpub\s*med\b",),
    "web": (r"\bweb\b", r"\binternet\s+(?:search|research|sources?)\b"),
    "wiki": (r"\bwikipedia\b", r"\bwiki\b"),
    "arxiv": (r"\barxiv\b",),
    "stackexchange": (r"\bstack\s*(?:exchange|overflow)\b",),
    "wolfram": (r"\bwolfram(?:\s+alpha)?\b",),
    "files": (r"\b(?:my|uploaded|local|reference)\s+(?:files?|documents?)\b",),
}


def _explicit_channels(query: str, available_channels: Iterable[str]) -> list[str]:
    """Preserve tools the user explicitly named even if the planner omits one."""
    allowed = set(available_channels)
    return [
        channel for channel, patterns in _CHANNEL_MENTIONS.items()
        if channel in allowed and any(re.search(pattern, query, re.I) for pattern in patterns)
    ]


def recovery_queries(query: str, channel: str, *, limit: int = 2) -> list[str]:
    """Build bounded source queries when the LLM contract cannot be frozen.

    This is intentionally domain-neutral: remove operation/meta language and
    retain the user's substantive clauses. The original request remains a
    final fallback so recovery never invents an outcome.
    """
    raw = " ".join(str(query or "").split())
    cleaned = re.sub(
        r"(?i)\b(?:please\s+)?(?:use|utilize)\s+(?:the\s+)?(?:pattern\s+tool|wiki|web|pubmed|wikipedia|internet)\b[^.]*\.?",
        " ", raw,
    )
    # Generic session-preamble strip (was a list of one live turn's literal
    # phrases): a SHORT leading clause ending in ":" is retry/test framing
    # ("okay it may be fixed:", "let's try this again:"), never the question —
    # the length bound keeps a genuine colon-introduced request intact.
    cleaned = re.sub(r"(?i)^(?:[^:.?!]{1,60}):\s+", " ", cleaned)
    cleaned = " ".join(cleaned.split()).strip(" .:-")
    if not cleaned:
        cleaned = raw
    # Keep queries bounded and avoid sending a multi-turn transcript to a
    # source API. A second query preserves the question's final substantive
    # clause when the first contains too much conversational framing.
    candidates = [cleaned[:500]]
    clauses = [part.strip(" .") for part in re.split(r"(?<=[?.!])\s+", cleaned) if part.strip()]
    if len(clauses) > 1:
        candidates.append(" ".join(clauses[-2:])[:500])
    return list(dict.fromkeys(candidates))[: max(1, min(limit, 2))]


def freeze_query(query: str, proposed: Optional[dict[str, Any]] = None) -> SpecFreezeResult:
    """Compatibility helper: prose requires the async generic planner."""
    if proposed is None:
        return SpecFreezeResult("insufficient", None, ["prose requires the generic deliberation planner"], "unavailable")
    return validate_and_freeze(
        proposed,
        required_channels=_explicit_channels(query, DEFAULT_CHANNELS),
        planner_provenance="caller-proposed", query=query,
    )


def freeze_spec(query: str, proposed: Optional[dict[str, Any]] = None) -> LongitudinalEvidenceSpec:
    """Compatibility shim for callers that already hold a planned spec.

    It deliberately refuses to infer from raw prose.
    """
    result = freeze_query(query, proposed)
    if result.status != "ready" or result.spec is None:
        raise ValueError("raw prose must be planned with plan_deliberation before retrieval")
    return result.spec
