"""
memory/pattern_engine.py

Module Contract
- Purpose: DETERMINISTIC pattern aggregation over the timestamped stores —
  counts, streaks, gaps, and trends by dimension x time bucket. No LLM calls,
  no writes, no fabrication: every exemplar reference carries a real store
  timestamp and (where available) a real doc reference. This is the shared
  core the insight-mode ``pattern_temporal`` facet consumes (v1); future
  surfaces (agentic tool, SPA view, shutdown notes) are thin adapters over
  the same ``run_pattern_query()``.
- Why it exists: the celebrated "two days in a row of heavy songs" flag was
  pure model synthesis over [RECENT CONVERSATION] — structurally blind beyond
  the ~20-turn window. The memory gate scores documents one at a time and
  cannot assemble "N low-similarity events that matter collectively"
  (docs/PATTERN_ANALYSIS.md; Daemon proposals d0db9576/b85bc622/d17bf634/
  ed7d47df all requested this capability).
- Dimensions (v1):
    topic_keyword   corpus_manager.search_keyword (word-boundary, time-bounded,
                    speaker-attributed). Default counts USER-side hits only —
                    the assistant echoes the user's vocabulary constantly, so
                    counting both sides inflates every pattern.
    tone            logs/turn_records.jsonl stream-parse (tone_level per turn;
                    ``test_env`` rows excluded). Counts non-CONVERSATIONAL
                    turns by default, or the levels passed in ``terms``.
    relation        user_profile.get_fact_history — value timeline for one
                    relation. Appraisal-stance values are labeled, mirroring
                    the stance-consumer doctrine.
    session_rhythm  corpus timestamps → messages + sessions per bucket
                    (>30 min gap = new session) + median first-message hour.
    content_type    THE deployed content_type_detector run over corpus entries
                    in-window (regex, sub-second at corpus scale) — covers
                    history without a backfill; new turns also persist the tag
                    in chroma metadata (memory_storage, 2026-08-23).
- Denominator doctrine: every result carries turns-per-bucket over the SAME
  window so consumers can distinguish "mentioned X more" from "talked more".
  The corpus over-samples days the user chose to talk — often hard days —
  so record-frequency is NOT life-frequency; synthesis must say so.
- Determinism: ``now`` is injectable (tests); bucket boundaries are computed
  from dates, never wall-clock drift.
- Side effects: none (read-only against every source).
- On-demand only (owner decision 2026-08-29): NOTHING here is injected into
  prompts uninvited — callers are explicit user requests.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import re
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from utils.logging_utils import get_logger

logger = get_logger("pattern_engine")

DIMENSIONS = ("topic_keyword", "tone", "relation", "session_rhythm",
              "content_type", "daily_notes")

_NON_CONVERSATIONAL = {"concern", "medium", "elevated", "high", "crisis"}

_SESSION_GAP_MINUTES = 30


class ExemplarRef(BaseModel):
    """A REAL store reference — the engine never fabricates these."""

    date: str
    text: str = ""
    source: str = ""            # corpus | facts | telemetry | profile
    doc_id: Optional[str] = None
    speaker: str = ""
    is_appraisal: bool = False


class PatternBucket(BaseModel):
    start: str                  # ISO date (bucket start)
    label: str = ""
    count: int = 0
    denominator: int = 0        # user turns in this bucket (same window)
    values: list[str] = Field(default_factory=list)
    exemplars: list[ExemplarRef] = Field(default_factory=list)


class PatternQuery(BaseModel):
    dimension: str = "topic_keyword"
    terms: list[str] = Field(default_factory=list)   # keywords / tone levels / content types
    relation: str = ""                               # relation dimension only
    window_days: int = 0                             # 0 = config default; -1 = all history
    bucket: str = "auto"                             # auto | day | week | month
    speaker: str = "user"                            # topic_keyword: user | both
    now: Optional[datetime] = None                   # injectable for tests
    vault_path: Optional[str] = None                 # daily_notes: test override


class PatternResult(BaseModel):
    dimension: str
    terms: list[str] = Field(default_factory=list)
    since: str = ""
    until: str = ""
    bucket_size: str = "week"
    buckets: list[PatternBucket] = Field(default_factory=list)
    total: int = 0
    active_days: int = 0                # distinct days with >=1 hit
    window_days: int = 0
    longest_streak_days: int = 0        # consecutive days with hits
    longest_gap_days: int = 0           # longest hitless run between first/last
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    trend: str = "insufficient"         # increasing | decreasing | stable | insufficient
    denominator_total: int = 0          # user turns in window
    notes: list[str] = Field(default_factory=list)

    def render_table(self) -> str:
        """Compact text table for LLM consumption. Counts are COMPUTED —
        the synthesizer must restate, never recount."""
        lines = [
            f"dimension={self.dimension} terms={','.join(self.terms) or self.dimension} "
            f"window={self.since}..{self.until} bucket={self.bucket_size}",
            f"total={self.total} hits on {self.active_days} distinct days "
            f"(user turns in window: {self.denominator_total})",
            f"first={self.first_seen or '-'} last={self.last_seen or '-'} "
            f"longest_streak={self.longest_streak_days}d longest_gap={self.longest_gap_days}d "
            f"trend={self.trend}",
        ]
        for b in self.buckets:
            extra = f" [{'; '.join(b.values)}]" if b.values else ""
            lines.append(
                f"  {b.label or b.start}: {b.count} hits / {b.denominator} turns{extra}"
            )
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Longitudinal evidence substrate
# ---------------------------------------------------------------------------

class EvidencePhase(BaseModel):
    """A bounded exposure/observation interval.

    Bounds are intentionally strings at the API boundary: callers may retain
    dates with a source basis and uncertainty rather than silently resolving a
    fuzzy statement to one day.
    """
    label: str
    start: Optional[str] = None
    end: Optional[str] = None
    date_basis: str = ""
    uncertainty_days: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TemporalAnchor(BaseModel):
    """Event/date used to construct relative phases without inventing a date."""

    label: str = ""
    date: Optional[str] = None
    relative_value: Optional[int] = None
    relative_unit: str = "days"  # days | weeks | months | years
    direction: str = "past"      # past | future
    uncertainty_days: int = 0
    search_terms: list[str] = Field(default_factory=list)
    date_basis: str = ""


class DeliberationClaimSpec(BaseModel):
    """One proposition in a dependency-aware deliberation chain."""

    claim_id: str
    proposition: str
    dependencies: list[str] = Field(default_factory=list)
    authority: str = "assessment"  # assessment | outside_authority
    required_channels: list[str] = Field(default_factory=list)
    evidence_standard: str = ""


class LongitudinalEvidenceSpec(BaseModel):
    """Frozen contract for a longitudinal assessment before retrieval."""
    analysis_kind: str = "event_phased"  # event_phased | period_comparison | time_series | co_occurrence
    propositions: list[str] = Field(default_factory=list)
    claims: list[DeliberationClaimSpec] = Field(default_factory=list)
    outcome_terms: list[str] = Field(default_factory=list)
    series_terms: dict[str, list[str]] = Field(default_factory=dict)
    concept_synonyms: dict[str, list[str]] = Field(default_factory=dict)
    behavioral_indicators: list[str] = Field(default_factory=list)
    directional_indicators: dict[str, list[str]] = Field(default_factory=lambda: {
        "increase": [], "decrease": [],
    })
    phases: list[EvidencePhase] = Field(default_factory=list)
    admissible_source_classes: list[str] = Field(
        default_factory=lambda: ["user", "therapist_report", "research"])
    rival_explanations: list[str] = Field(default_factory=list)
    confounder_indicators: dict[str, list[str]] = Field(default_factory=dict)
    supporting_facets: list[str] = Field(default_factory=list)
    refuting_facets: list[str] = Field(default_factory=list)
    sensitivity_axes: list[str] = Field(default_factory=lambda: [
        "date_bounds", "phase_boundaries", "proxy_inclusion", "confounders"])
    requested_channels: list[str] = Field(default_factory=lambda: [
        "pattern", "corpus", "notes"])
    research_queries: dict[str, str] = Field(default_factory=dict)
    phase_policy: dict[str, int] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    anchor: Optional[TemporalAnchor] = None
    decision_context: list[str] = Field(default_factory=list)


class LongitudinalEvent(BaseModel):
    """One observable item; absence of an item is never an event."""
    timestamp: str
    text: str = ""
    source_class: str = "user"
    source_id: Optional[str] = None
    speaker: str = "user"
    indicators: list[str] = Field(default_factory=list)
    polarity: int = 0  # -1 decreased, 0 unclear, +1 increased (not good/bad)
    polarity_basis: str = ""
    evidence_role: str = "observation"  # observation | timing | hypothesis | secondhand | unusable
    evidence_class: str = "direct_observation"
    provenance: dict[str, Any] = Field(default_factory=dict)


class PhaseComparison(BaseModel):
    phase: EvidencePhase
    event_count: int = 0
    observed_days: int = 0
    observation_denominator: int = 0
    coverage: float = 0.0
    decreased_count: int = 0
    increased_count: int = 0
    unclear_direction_count: int = 0
    events: list[LongitudinalEvent] = Field(default_factory=list)
    proxy_events: list[LongitudinalEvent] = Field(default_factory=list)
    proxy_event_count: int = 0
    covariate_counts: dict[str, int] = Field(default_factory=dict)
    covariate_events: list[LongitudinalEvent] = Field(default_factory=list)
    evidence_class_counts: dict[str, int] = Field(default_factory=dict)
    series_counts: dict[str, int] = Field(default_factory=dict)
    joint_event_count: int = 0
    notes: list[str] = Field(default_factory=list)


class LongitudinalScanResult(BaseModel):
    spec: LongitudinalEvidenceSpec
    comparisons: list[PhaseComparison] = Field(default_factory=list)
    excluded_events: int = 0
    exclusion_reasons: dict[str, int] = Field(default_factory=dict)
    sensitivity: list[dict[str, Any]] = Field(default_factory=list)
    # Deterministic bucket-level co-occurrence classification (co_occurrence
    # kind only): the model narrates these numbers, it never re-counts the
    # per-phase rows itself.
    co_occurrence: dict[str, Any] = Field(default_factory=dict)
    manifest: str = ""

    def render_manifest(self) -> str:
        if self.manifest:
            return self.manifest
        lines = ["LONGITUDINAL EVIDENCE MANIFEST"]
        for c in self.comparisons:
            lines.append(f"- {c.phase.label}: {c.event_count} observable events; "
                         f"{c.observation_denominator} user observations; "
                         f"coverage={c.coverage:.2f}")
            if c.proxy_event_count:
                lines.append(f"  proxy/context events={c.proxy_event_count} (not user-authored observations)")
            if c.evidence_class_counts:
                lines.append("  evidence classes=" + ", ".join(
                    f"{name}:{count}" for name, count in sorted(c.evidence_class_counts.items())
                ))
            lines.extend(f"  note: {n}" for n in c.notes)
        if self.excluded_events:
            lines.append(f"- excluded={self.excluded_events}: "
                         + ", ".join(f"{k}={v}" for k, v in self.exclusion_reasons.items()))
        return "\n".join(lines)


def _date_bound(
    value: Optional[str], fallback: datetime, *, is_end: bool = False,
) -> datetime:
    parsed = _parse_ts(value) if value else None
    if parsed is not None and value and len(value) == 10 and is_end:
        # Calendar-date bounds are inclusive through the end of that date.
        return parsed + timedelta(days=1) - timedelta(microseconds=1)
    return parsed or fallback


def _event_polarity(
    text: str, directional_indicators: dict[str, list[str]],
) -> tuple[int, str]:
    """Apply only the frozen domain-neutral direction lexicon.

    The planner may freeze literal phrases meaning an increase or decrease of
    the selected outcome. The deterministic scan does not invent domain
    semantics and does not equate increase/decrease with good/bad. If both
    directions occur in one item, it remains unclear.
    """
    folded = text.casefold()
    increases = [
        phrase for phrase in directional_indicators.get("increase", [])
        if phrase and _literal_phrase_in_text(phrase, folded)
    ]
    decreases = [
        phrase for phrase in directional_indicators.get("decrease", [])
        if phrase and _literal_phrase_in_text(phrase, folded)
    ]
    if increases and not decreases:
        return 1, f"frozen increase indicator: {increases[0]}"
    if decreases and not increases:
        return -1, f"frozen decrease indicator: {decreases[0]}"
    if increases and decreases:
        return 0, "conflicting frozen direction indicators"
    return 0, "outcome mentioned without a frozen direction indicator"


def _literal_phrase_in_text(phrase: str, folded_text: str) -> bool:
    """Match a frozen phrase on token boundaries, not arbitrary substrings."""
    normalized = " ".join(str(phrase).casefold().split())
    if not normalized:
        return False
    pattern = r"(?<!\w)" + r"\s+".join(
        re.escape(part) for part in normalized.split(" ")
    ) + r"(?!\w)"
    return re.search(pattern, folded_text) is not None


def _evidence_role(event: LongitudinalEvent) -> str:
    """Conservative pre-count role gate; explicit metadata wins."""
    role = str(event.provenance.get("evidence_role") or event.evidence_role or "observation").lower()
    if role in {"timing", "observation", "hypothesis", "secondhand", "unusable"}:
        if role != "observation":
            return role
    text = event.text.casefold()
    # A repeated request to analyze a symptom is not itself a symptom event.
    # This catches pasted/retried prompts (often containing the target word)
    # without discarding ordinary user questions that include a dated report.
    if ("pattern tool" in text or "before and after" in text) and "?" in text:
        return "unusable"
    if re.search(r"\b(?:my\s+therapist|therapist|doctor|psychiatrist)\s+(?:said|thinks?|suggested|believes?)\b", text):
        return "secondhand"
    if re.search(r"\b(?:could|might|maybe|perhaps|possibly|strongest\s+theory|is\s+it\s+likely|i\s+wonder|i\s+think)\b", text):
        # Keep mixed statements that contain a concrete past-tense observation;
        # reject pure explanatory/speculative excerpts from outcome counts.
        if not re.search(r"\b(?:was|were|felt|had|did|took|stopped|started|slept|said)\b", text):
            return "hypothesis"
    if "?" in event.text and not re.search(r"\b(?:yesterday|last\s+night|this\s+morning|ago)\b", text):
        return "unusable"
    return "observation"


def _event_identity(event: LongitudinalEvent, ts: datetime) -> tuple[str, str]:
    """Deduplicate one underlying source without merging unrelated items.

    A canonical lineage ID wins when an adapter supplies one. Derived-store
    artifacts can point at the same canonical source; otherwise source IDs are
    scoped by source class. The text hash fallback prevents two different
    same-timestamp records from collapsing.
    """
    provenance = event.provenance or {}
    canonical = (
        provenance.get("canonical_source_id")
        or provenance.get("derived_from")
        or provenance.get("source_record_id")
    )
    if canonical:
        return ("canonical", str(canonical))
    if event.source_id:
        return (event.source_class, str(event.source_id))
    digest = hashlib.sha256(
        f"{event.source_class}\0{ts.isoformat()}\0{event.text}".encode("utf-8")
    ).hexdigest()
    return (event.source_class, digest)


def run_longitudinal_scan(
    spec: LongitudinalEvidenceSpec,
    events: Iterable[LongitudinalEvent | dict[str, Any]],
) -> LongitudinalScanResult:
    """Assign explicit events to phases with source-independent deduplication.

    This deliberately performs bounded literal indicator matching only. It is
    transparent and safe to extend with behavioral coding later; it does not
    treat medication mentions or missing mentions as outcome evidence.
    """
    normalized: list[LongitudinalEvent] = []
    proxy_normalized: list[LongitudinalEvent] = []
    observations: list[LongitudinalEvent] = []
    proxy_observations: list[LongitudinalEvent] = []
    excluded: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, str]] = set()
    terms = [t.casefold() for t in spec.outcome_terms if t.strip()]
    indicators = [t.casefold() for t in spec.behavioral_indicators if t.strip()]
    directional_terms = [
        str(term).casefold()
        for direction in ("increase", "decrease")
        for term in spec.directional_indicators.get(direction, [])
        if str(term).strip()
    ]
    confounder_terms = {
        str(name): [str(term).casefold() for term in values if str(term).strip()]
        for name, values in spec.confounder_indicators.items()
        if isinstance(values, list)
    }
    allowed = set(spec.admissible_source_classes)
    for raw in events:
        try:
            event = raw if isinstance(raw, LongitudinalEvent) else LongitudinalEvent.model_validate(raw)
            is_proxy = event.evidence_class in {"assistant_summary", "proxy"}
            if event.speaker.casefold() in {"assistant", "daemon", "system"} and not is_proxy:
                excluded["assistant_derived"] += 1; continue
            if allowed and event.source_class not in allowed and not is_proxy:
                excluded["source_class"] += 1; continue
            role = _evidence_role(event)
            event.evidence_role = role
            if event.provenance.get("date_basis") == "index-time":
                excluded["index_time_date"] += 1
                continue
            if role in {"hypothesis", "secondhand", "unusable"}:
                excluded[f"evidence_role:{role}"] += 1
                continue
            ts = _parse_ts(event.timestamp)
            if ts is None:
                excluded["invalid_date"] += 1; continue
            key = _event_identity(event, ts)
            if key in seen:
                excluded["duplicate"] += 1; continue
            seen.add(key)
            (proxy_observations if is_proxy else observations).append(event)
            text = event.text.casefold()
            matched = [
                x for x in terms + indicators + directional_terms
                if _literal_phrase_in_text(x, text)
            ]
            if not matched:
                # No match is simply unobserved, never evidence of absence.
                continue
            event.indicators = sorted(set(event.indicators + matched))
            if not event.polarity_basis:
                event.polarity, event.polarity_basis = _event_polarity(
                    event.text, spec.directional_indicators,
                )
            (proxy_normalized if is_proxy else normalized).append(event)
        except Exception:
            excluded["malformed"] += 1
    comparisons = []
    for phase in spec.phases:
        start = _date_bound(phase.start, datetime.min, is_end=False)
        end = _date_bound(phase.end, datetime.max, is_end=True)
        selected = [e for e in normalized if start <= _parse_ts(e.timestamp) <= end]
        selected_proxy = [e for e in proxy_normalized if start <= _parse_ts(e.timestamp) <= end]
        phase_observations = [
            e for e in observations if start <= _parse_ts(e.timestamp) <= end
        ]
        covariate_counts: dict[str, int] = {}
        covariate_events: list[LongitudinalEvent] = []
        evidence_class_counts: dict[str, int] = defaultdict(int)
        seen_covariates: set[tuple[str, str]] = set()
        for name, phrases in confounder_terms.items():
            matches = [
                event for event in phase_observations
                if any(_literal_phrase_in_text(phrase, event.text.casefold()) for phrase in phrases)
            ]
            covariate_counts[name] = len(matches)
            for event in matches:
                identity = _event_identity(event, _parse_ts(event.timestamp))
                if identity not in seen_covariates:
                    seen_covariates.add(identity)
                    covariate_events.append(event)
        for event in selected:
            evidence_class_counts[event.evidence_class] += 1
        series_counts: dict[str, int] = {}
        joint_event_count = 0
        if spec.analysis_kind == "co_occurrence" and spec.series_terms:
            series_ids: dict[str, set[tuple[str, str]]] = {}
            for name, phrases in spec.series_terms.items():
                ids = {
                    _event_identity(event, _parse_ts(event.timestamp))
                    for event in phase_observations
                    if any(
                        _literal_phrase_in_text(str(phrase), event.text.casefold())
                        for phrase in phrases
                    )
                }
                series_ids[name] = ids
                series_counts[name] = len(ids)
            if len(series_ids) >= 2:
                joint_event_count = len(set.intersection(*series_ids.values()))
        # Count every admissible user observation in the phase, not only
        # outcome hits. An explicit metadata denominator may supplement an
        # adapter that can enumerate more observations than it returned.
        denominator = max(
            len(phase_observations),
            int(phase.metadata.get("observation_denominator", 0)),
        )
        days = len({_parse_ts(e.timestamp).date() for e in phase_observations})
        span = max(1, (end.date() - start.date()).days + 1) if start != datetime.min and end != datetime.max else 0
        comparisons.append(PhaseComparison(phase=phase, event_count=len(selected),
            observed_days=days, observation_denominator=denominator,
            coverage=(days / span if span else 0.0),
            decreased_count=sum(e.polarity < 0 for e in selected),
            increased_count=sum(e.polarity > 0 for e in selected),
            unclear_direction_count=sum(e.polarity == 0 for e in selected),
            events=selected,
            proxy_events=selected_proxy,
            proxy_event_count=len(selected_proxy),
            covariate_counts=covariate_counts,
            covariate_events=covariate_events,
            evidence_class_counts=dict(evidence_class_counts),
            series_counts=series_counts,
            joint_event_count=joint_event_count,
            notes=["absence of mention is not absence of symptom"] + (["date bounds unresolved"] if not phase.start or not phase.end else [])))
    result = LongitudinalScanResult(spec=spec, comparisons=comparisons,
        excluded_events=sum(excluded.values()), exclusion_reasons=dict(excluded))
    if spec.analysis_kind == "co_occurrence" and spec.series_terms:
        series_names = list(spec.series_terms)
        presence = {name: 0 for name in series_names}
        all_series = none_series = 0
        for comp in comparisons:
            present = [name for name in series_names
                       if comp.series_counts.get(name, 0) > 0]
            for name in present:
                presence[name] += 1
            if len(present) == len(series_names):
                all_series += 1
            elif not present:
                none_series += 1
        result.co_occurrence = {
            "series": series_names,
            "bucket_count": len(comparisons),
            "buckets_with_series": presence,
            "buckets_with_all_series": all_series,
            "buckets_with_no_series": none_series,
            "buckets_with_partial_series": max(
                0, len(comparisons) - all_series - none_series,
            ),
            "same_event_joint_mentions": sum(
                comp.joint_event_count for comp in comparisons
            ),
            "counting_basis": (
                "a series is present in a bucket when >=1 admissible dated "
                "observation matches its frozen terms; record absence is not "
                "behavior absence"
            ),
        }
    # Make date uncertainty operational: report bounded ± shifts and whether
    # the selected event count changes. This is intentionally a sensitivity
    # result, not a silently resolved date.
    for phase, comp in zip(spec.phases, comparisons):
        if phase.uncertainty_days and phase.start and phase.end:
            start = _date_bound(phase.start, datetime.min, is_end=False)
            end = _date_bound(phase.end, datetime.max, is_end=True)
            for direction in (-1, 1):
                lo = start + timedelta(days=direction * phase.uncertainty_days)
                hi = end + timedelta(days=direction * phase.uncertainty_days)
                n = sum(1 for e in normalized if lo <= _parse_ts(e.timestamp) <= hi)
                result.sensitivity.append({"axis": "date_bounds", "phase": phase.label,
                    "variant": f"shift_{direction * phase.uncertainty_days:+d}d", "event_count": n,
                    "changed": n != comp.event_count})
    for axis in spec.sensitivity_axes:
        if axis not in {"date_bounds"}:
            result.sensitivity.append({"axis": axis, "status": "not_available",
                "reason": "requires additional coded evidence or confounder data"})
    result.manifest = result.render_manifest()
    return result


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_ts(value) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.replace(tzinfo=None) if dt.tzinfo else dt
        except (ValueError, TypeError):
            return None
    return None


def _auto_bucket(window_days: int) -> str:
    if window_days <= 14:
        return "day"
    if window_days <= 120:
        return "week"
    return "month"


def _bucket_start(dt: datetime, bucket: str) -> datetime:
    d = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if bucket == "day":
        return d
    if bucket == "week":
        return d - timedelta(days=d.weekday())
    return d.replace(day=1)


def _bucket_starts(since: datetime, until: datetime, bucket: str) -> list[datetime]:
    """All bucket starts covering [since, until] — empty buckets INCLUDED
    (a gap is information)."""
    starts = []
    cur = _bucket_start(since, bucket)
    while cur <= until:
        starts.append(cur)
        if bucket == "day":
            cur += timedelta(days=1)
        elif bucket == "week":
            cur += timedelta(days=7)
        else:
            cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
    return starts


def _streak_and_gap(hit_days: set) -> tuple[int, int]:
    if not hit_days:
        return 0, 0
    days = sorted(hit_days)
    longest_streak = streak = 1
    longest_gap = 0
    for prev, cur in zip(days, days[1:]):
        delta = (cur - prev).days
        if delta == 1:
            streak += 1
            longest_streak = max(longest_streak, streak)
        else:
            streak = 1
            longest_gap = max(longest_gap, delta - 1)
    return longest_streak, longest_gap


def _trend(buckets: list[PatternBucket]) -> str:
    """First-half vs second-half comparison. Deliberately coarse — the
    synthesizer's job is calibrated language, not statistics theater."""
    counts = [b.count for b in buckets]
    total = sum(counts)
    if total < 4 or len(counts) < 2:
        return "insufficient"
    mid = len(counts) // 2
    first, second = sum(counts[:mid]), sum(counts[mid:])
    if second >= max(first * 1.5, first + 2):
        return "increasing"
    if first >= max(second * 1.5, second + 2):
        return "decreasing"
    return "stable"


def _corpus_entries(corpus_manager, since: datetime, until: datetime) -> list[dict]:
    """Episodic corpus entries in-window (junk filtered), parsed timestamps
    attached as '_ts'. Read-only."""
    from memory.utils import is_junk_conversation_doc
    out = []
    try:
        entries = corpus_manager._get_episodic_sorted()
    except Exception:
        entries = [e for e in getattr(corpus_manager, "corpus", [])
                   if isinstance(e, dict)]
    for e in entries:
        ts = _parse_ts(e.get("timestamp"))
        if ts is None or not (since <= ts <= until):
            continue
        try:
            if is_junk_conversation_doc(e.get("query", ""), e.get("response", "")):
                continue
        except Exception:
            pass
        e = dict(e)
        e["_ts"] = ts
        out.append(e)
    return out


def _denominators(corpus_manager, starts: list[datetime], bucket: str,
                  since: datetime, until: datetime) -> tuple[dict, int]:
    """User turns per bucket over the window — the denominator that keeps
    'mentioned more' distinguishable from 'talked more'."""
    per: dict[datetime, int] = defaultdict(int)
    total = 0
    for e in _corpus_entries(corpus_manager, since, until):
        per[_bucket_start(e["_ts"], bucket)] += 1
        total += 1
    return per, total


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------

def run_pattern_query(
    query: PatternQuery,
    *,
    corpus_manager=None,
    user_profile=None,
    telemetry_path: Optional[str] = None,
) -> PatternResult:
    """Run one deterministic pattern query. Never raises — failures degrade
    to an empty result with a note. Synchronous (callers may to_thread)."""
    from config.app_config import (
        PATTERN_DEFAULT_WINDOW_DAYS,
        PATTERN_EXEMPLARS_PER_BUCKET,
        PATTERN_KEYWORD_HIT_CAP,
        PATTERN_MAX_EXEMPLARS,
    )

    now = query.now or datetime.now()
    window_days = query.window_days or PATTERN_DEFAULT_WINDOW_DAYS
    if window_days < 0:
        window_days = 3650  # "all history" — bounded for bucket math
    until = now
    since = now - timedelta(days=window_days)
    bucket = query.bucket if query.bucket in ("day", "week", "month") \
        else _auto_bucket(window_days)

    result = PatternResult(
        dimension=query.dimension,
        terms=list(query.terms),
        since=since.date().isoformat(),
        until=until.date().isoformat(),
        bucket_size=bucket,
        window_days=window_days,
    )

    starts = _bucket_starts(since, until, bucket)
    buckets: dict[datetime, PatternBucket] = {
        s: PatternBucket(start=s.date().isoformat(),
                         label=f"{s.date().isoformat()} ({bucket})")
        for s in starts
    }

    # events: list of (ts, ExemplarRef|None, value|None)
    try:
        if query.dimension == "topic_keyword":
            events = _events_topic_keyword(
                query, corpus_manager, since, until, PATTERN_KEYWORD_HIT_CAP, result)
        elif query.dimension == "tone":
            events = _events_tone(query, telemetry_path, since, until, result)
        elif query.dimension == "relation":
            events = _events_relation(query, user_profile, since, until, result)
        elif query.dimension == "session_rhythm":
            events = _events_session_rhythm(
                query, corpus_manager, since, until, bucket, buckets, result)
        elif query.dimension == "content_type":
            events = _events_content_type(query, corpus_manager, since, until, result)
        elif query.dimension == "daily_notes":
            events = _events_daily_notes(query, since, until, result)
        else:
            result.notes.append(f"unknown dimension: {query.dimension}")
            events = []
    except Exception as e:
        logger.warning(f"[PatternEngine] {query.dimension} query failed: {e}")
        result.notes.append(f"{query.dimension} source unavailable: {e}")
        events = []

    hit_days: set = set()
    for ts, exemplar, value in events:
        b = buckets.get(_bucket_start(ts, bucket))
        if b is None:
            continue
        b.count += 1
        hit_days.add(ts.date())
        if value and value not in b.values:
            b.values.append(value)
        if exemplar is not None and len(b.exemplars) < PATTERN_EXEMPLARS_PER_BUCKET:
            b.exemplars.append(exemplar)

    # Denominator (corpus user turns) for every dimension that has a corpus.
    if corpus_manager is not None:
        try:
            per, total = _denominators(corpus_manager, starts, bucket, since, until)
            for s, b in buckets.items():
                b.denominator = per.get(s, 0)
            result.denominator_total = total
        except Exception as e:
            result.notes.append(f"denominator unavailable: {e}")

    ordered = [buckets[s] for s in starts]
    # Trim leading/trailing all-empty buckets ONLY when the whole window is
    # empty-edged AND long — keep interior gaps (a gap is information).
    result.buckets = ordered
    result.total = sum(b.count for b in ordered)
    result.active_days = len(hit_days)
    result.longest_streak_days, result.longest_gap_days = _streak_and_gap(hit_days)
    if hit_days:
        result.first_seen = min(hit_days).isoformat()
        result.last_seen = max(hit_days).isoformat()
    result.trend = _trend(ordered)

    # Global exemplar cap (buckets already cap locally).
    kept = 0
    for b in result.buckets:
        room = max(0, PATTERN_MAX_EXEMPLARS - kept)
        b.exemplars = b.exemplars[:room]
        kept += len(b.exemplars)

    return result


# ---------------------------------------------------------------------------
# per-dimension event extraction — each returns [(ts, ExemplarRef|None, value|None)]
# ---------------------------------------------------------------------------

def _events_topic_keyword(query, corpus_manager, since, until, hit_cap, result):
    if corpus_manager is None:
        result.notes.append("no corpus available")
        return []
    if not query.terms:
        result.notes.append("topic_keyword requires terms")
        return []
    hits = corpus_manager.search_keyword(
        query.terms, start=since, end=until, max_results=hit_cap,
        context_chars=160,
    )
    if len(hits) >= hit_cap:
        result.notes.append(f"hit cap {hit_cap} reached — counts are a floor")
    events = []
    for h in hits:
        if query.speaker == "user" and h.get("speaker") != "user":
            continue
        ts = _parse_ts(h.get("timestamp"))
        if ts is None:
            continue
        events.append((ts, ExemplarRef(
            date=ts.isoformat(),
            text=(h.get("excerpt") or "")[:160],
            source="corpus",
            speaker=h.get("speaker", ""),
        ), None))
    if query.speaker == "user":
        result.notes.append("user-side mentions only (assistant echoes excluded)")
    return events


def _events_tone(query, telemetry_path, since, until, result):
    import json
    import os
    if telemetry_path is None:
        from config.app_config import TURN_TELEMETRY_PATH as telemetry_path  # noqa: F811
    if not telemetry_path or not os.path.exists(telemetry_path):
        result.notes.append("no telemetry available")
        return []
    wanted = {t.lower() for t in query.terms} or _NON_CONVERSATIONAL
    events = []
    earliest = None
    floored = 0
    with open(telemetry_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("test_env"):
                continue
            ts = _parse_ts(row.get("ts") or row.get("time"))
            if ts is None:
                continue
            earliest = ts if earliest is None else min(earliest, ts)
            if not (since <= ts <= until):
                continue
            level = str(row.get("tone_level") or "").split(".")[-1].lower()
            if level not in wanted:
                continue
            # Floored turns are CARRIED tone, not fresh evidence — counting
            # them as mood data would report the sticky-floor mechanism (and
            # its 07-21→08-27 latch bug era) as the user's emotional state.
            # Same doctrine as the escalation tracker's distress counter.
            if str(row.get("tone_trigger") or "") == "distress_sticky_floor":
                floored += 1
                continue
            events.append((ts, ExemplarRef(
                date=ts.isoformat(),
                text=f"tone={level} ({row.get('tone_trigger') or 'n/a'}): "
                     f"{(row.get('query') or '')[:100]}",
                source="telemetry",
            ), level))
    if floored:
        result.notes.append(
            f"{floored} floored (carried-tone) turns excluded — organic tone "
            f"signals only")
    if earliest is not None and earliest > since:
        result.notes.append(
            f"telemetry coverage starts {earliest.date().isoformat()} — "
            f"earlier part of the window has no tone data")
    return events


def _events_relation(query, user_profile, since, until, result):
    if user_profile is None:
        result.notes.append("no profile available")
        return []
    relation = query.relation or (query.terms[0] if query.terms else "")
    if not relation:
        result.notes.append("relation dimension requires a relation name")
        return []
    events = []
    for fact in user_profile.get_fact_history(relation):
        ts = _parse_ts(fact.get("timestamp"))
        if ts is None or not (since <= ts <= until):
            continue
        value = str(fact.get("value") or "")[:120]
        is_appraisal = str(fact.get("stance") or "") == "appraisal"
        events.append((ts, ExemplarRef(
            date=ts.isoformat(),
            text=f"{relation} = {value}"
                 + (" [user's appraisal at the time]" if is_appraisal else ""),
            source="profile",
            doc_id=fact.get("fact_id"),
            is_appraisal=is_appraisal,
        ), value))
    return events


def _events_session_rhythm(query, corpus_manager, since, until, bucket, buckets, result):
    if corpus_manager is None:
        result.notes.append("no corpus available")
        return []
    entries = _corpus_entries(corpus_manager, since, until)
    entries.sort(key=lambda e: e["_ts"])
    # sessions + first-message hours per bucket
    sessions_per: dict = defaultdict(int)
    first_hours: dict = defaultdict(list)
    prev_ts = None
    prev_day = None
    for e in entries:
        ts = e["_ts"]
        if prev_ts is None or (ts - prev_ts) > timedelta(minutes=_SESSION_GAP_MINUTES):
            sessions_per[_bucket_start(ts, bucket)] += 1
        if ts.date() != prev_day:
            first_hours[_bucket_start(ts, bucket)].append(ts.hour + ts.minute / 60)
            prev_day = ts.date()
        prev_ts = ts
    for s, b in buckets.items():
        n_sessions = sessions_per.get(s, 0)
        hours = sorted(first_hours.get(s, []))
        if n_sessions:
            b.values.append(f"sessions: {n_sessions}")
        if hours:
            med = hours[len(hours) // 2]
            b.values.append(f"median first-message hour: {int(med):02d}:{int((med % 1) * 60):02d}")
    # events = every message (count == message volume)
    return [(e["_ts"], None, None) for e in entries]


def _note_frontmatter_and_emotion(text: str) -> tuple[dict, str]:
    """Parse a Daemon daily note's YAML frontmatter + the first line of its
    '## Emotional State' section. Lenient: failures return ({}, '')."""
    import re as _re
    import yaml
    fm: dict = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end > 0:
            try:
                parsed = yaml.safe_load(text[3:end])
                if isinstance(parsed, dict):
                    fm = parsed
            except Exception:
                pass
    emotion = ""
    m = _re.search(r"^##\s+Emotional\s+State\s*\n(.+)$", text,
                   _re.MULTILINE)
    if m:
        emotion = m.group(1).strip()[:160]
    return fm, emotion


def _events_daily_notes(query, since, until, result):
    """The Daemon-generated daily notes are an INDEPENDENT per-day series
    (usage_intensity, active_hours, Emotional State) — not a compression of
    counted conversations, so no double-counting. Summaries/reflections stay
    deliberately UNCOUNTED for exactly that reason (they compress the same
    turns the corpus dimensions already count)."""
    from utils.daily_notes_generator import read_daily_note
    span = (until.date() - since.date()).days
    if span > 400:
        result.notes.append("daily_notes window capped at 400 days")
        since = until - timedelta(days=400)
    events = []
    missing = 0
    cur = since.date()
    vault = query.vault_path
    while cur <= until.date():
        try:
            text = read_daily_note(cur, vault)
        except Exception:
            text = None
        if text:
            fm, emotion = _note_frontmatter_and_emotion(text)
            intensity = fm.get("usage_intensity", fm.get("intensity"))
            active = fm.get("active_hours", fm.get("total_active_hours"))
            ts = datetime(cur.year, cur.month, cur.day, 12)
            desc = f"[daily note] intensity={intensity}"
            if active is not None:
                desc += f" active={active}h"
            if emotion:
                desc += f" — {emotion}"
            events.append((ts, ExemplarRef(
                date=ts.date().isoformat(),
                text=desc[:240],
                source="daily_note",
            ), f"intensity {intensity}" if intensity is not None else None))
        else:
            missing += 1
        cur += timedelta(days=1)
    if missing:
        result.notes.append(
            f"{missing} days in the window have no daily note "
            f"(absence ≠ nothing happened)")
    if not events and missing:
        result.notes.append("no daily notes found — vault may be unavailable")
    return events


def _events_content_type(query, corpus_manager, since, until, result):
    from core.content_type_detector import detect_content_type
    if corpus_manager is None:
        result.notes.append("no corpus available")
        return []
    wanted = {t.lower() for t in query.terms} or {"lyrics"}
    events = []
    for e in _corpus_entries(corpus_manager, since, until):
        try:
            ct = detect_content_type(e.get("query", ""))
        except Exception:
            continue
        if ct.content_type and ct.content_type.lower() in wanted:
            events.append((e["_ts"], ExemplarRef(
                date=e["_ts"].isoformat(),
                text=f"[{ct.content_type}] {(e.get('query') or '')[:120]}",
                source="corpus",
                speaker="user",
            ), ct.content_type))
    result.notes.append(
        "content types detected by the deployed detector at query time "
        f"(matching: {', '.join(sorted(wanted))})")
    return events
