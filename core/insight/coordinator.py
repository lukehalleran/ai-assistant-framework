"""Single-owner coordinator for generic pattern-oriented deliberation.

The coordinator freezes an LLM-proposed evidence contract before retrieval,
then executes a registry of source adapters, resolves temporal anchors, runs
the deterministic phase scan, and independently assesses a dependency-aware
claim chain. It never prescribes or converts an unavailable source into
imagined evidence.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib
import inspect
import json
from typing import Any, Awaitable, Callable, Optional

from core.insight.deliberation import (
    DEFAULT_CHANNELS,
    SpecFreezeResult,
    deterministic_fallback_spec,
    compact_recovery_plan,
    _explicit_channels,
    recovery_queries,
    plan_deliberation,
)
from memory.pattern_engine import (
    EvidencePhase,
    LongitudinalEvent,
    LongitudinalEvidenceSpec,
    LongitudinalScanResult,
    TemporalAnchor,
    run_longitudinal_scan,
)

CHANNEL_STATES = frozenset({
    "succeeded", "no_results", "no_relevant_results", "partial", "unavailable", "failed", "skipped", "insufficient",
})
_INTERNAL_CHANNELS = frozenset({"pattern", "corpus", "notes", "facts"})
# External research also benefits from bounded ladders: direct, broadened,
# and adjacent-endpoint queries can find literature that uses different
# terminology. The planner still caps the total at eight queries/channel.
_MULTI_QUERY_CHANNELS = frozenset({
    "notes", "facts", "files", "pubmed", "web", "wiki",
})
_OUTSIDE_AUTHORITY = "outside_authority"
_ASSESSMENT_STATUSES = frozenset({
    "supported", "partially_supported", "mixed", "counterevidence",
    "insufficient", "outside_authority",
})


@dataclass
class ChannelStatus:
    channel: str
    status: str
    requested: bool = True
    attempted: bool = False
    reason: str = ""
    count: int = 0
    source_ids: tuple[str, ...] = ()
    queries_attempted: int = 0
    query_failures: int = 0


@dataclass
class DeliberationClaim:
    claim_id: str
    proposition: str
    status: str
    confidence: float
    coverage: str
    directness: str
    dependencies: tuple[str, ...] = ()
    authority: str = "assessment"
    support_source_ids: tuple[str, ...] = ()
    refute_source_ids: tuple[str, ...] = ()
    rationale: str = ""


@dataclass
class DeliberationResult:
    freeze: SpecFreezeResult
    scan: Optional[LongitudinalScanResult]
    internal_events: list[LongitudinalEvent]
    external_evidence: list[dict[str, Any]]
    channels: list[ChannelStatus]
    claim_chain: list[DeliberationClaim]
    manifest: dict[str, Any]


Adapter = Callable[[str], Any | Awaitable[Any]]
ClaimAssessor = Callable[..., Any | Awaitable[Any]]


def _stratified_rows(
    rows: list[dict[str, Any]], *, group_key: str, total: int, per_group: int,
) -> list[dict[str, Any]]:
    """Bound evidence without letting the first source crowd out the rest."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(group_key) or "unknown")
        bucket = groups.setdefault(key, [])
        if len(bucket) < per_group:
            bucket.append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < total and any(groups.values()):
        for key in list(groups):
            if groups[key] and len(selected) < total:
                selected.append(groups[key].pop(0))
    return selected


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


def _relative_days(value: int, unit: str) -> int:
    multipliers = {"days": 1, "weeks": 7, "months": 30, "years": 365}
    return int(value) * multipliers.get((unit or "days").lower(), 1)


def _row_text(row: dict[str, Any]) -> str:
    return str(
        row.get("text") or row.get("content") or row.get("document")
        or row.get("abstract") or row.get("snippet") or ""
    ).strip()


def _row_id(row: dict[str, Any], channel: str) -> str:
    if row.get("pmid") and not row.get("source_id"):
        return f"pmid:{row['pmid']}"
    explicit = row.get("source_id") or row.get("id") or row.get("url")
    if explicit:
        value = str(explicit)
        if value.startswith(("http://", "https://")) or ":" in value:
            return value
        return f"{channel}:{value}"
    digest = hashlib.sha256(
        f"{channel}\0{_row_text(row)}\0{row.get('timestamp') or row.get('date') or ''}".encode("utf-8")
    ).hexdigest()[:20]
    return f"{channel}:{digest}"


def normalize_chroma_rows(rows: Any, *, channel: str) -> list[dict[str, Any]]:
    """Normalize Chroma wrapper rows without assigning evidentiary weight."""
    normalized: list[dict[str, Any]] = []
    for raw in rows or []:
        if not isinstance(raw, dict):
            continue
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        text = _row_text(raw)
        if not text:
            continue
        merged = {
            "id": raw.get("id") or metadata.get("id"),
            "source_id": raw.get("id") or metadata.get("id"),
            "text": text,
            "title": metadata.get("title") or metadata.get("name") or "",
            "url": metadata.get("url") or metadata.get("source_url") or "",
            "timestamp": metadata.get("timestamp") or metadata.get("date") or raw.get("timestamp"),
            "date": metadata.get("date") or metadata.get("timestamp") or raw.get("date"),
            "content_date": metadata.get("note_date") or metadata.get("date") or raw.get("content_date"),
            "record_date": metadata.get("generated_at") or raw.get("record_date") or raw.get("timestamp"),
            "date_basis": metadata.get("date_basis") or (
                "content" if metadata.get("note_date") or metadata.get("date")
                else "index-time" if channel == "notes" else "record"
            ),
            "source_class": channel,
            "canonical_source_id": (
                metadata.get("canonical_source_id")
                or metadata.get("source_record_id")
            ),
            "derived_from": metadata.get("derived_from"),
            "metadata": metadata,
        }
        merged["source_id"] = _row_id(merged, channel)
        normalized.append(merged)
    return normalized


def _event_from_row(row: dict[str, Any], channel: str) -> Optional[LongitudinalEvent]:
    text = _row_text(row)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    content_date = row.get("content_date") or row.get("note_date") or metadata.get("note_date")
    timestamp = content_date or row.get("date") or row.get("timestamp")
    if not text or not timestamp:
        return None
    # Daily notes generated by Daemon are derived summaries.  They remain
    # useful contextual/proxy evidence, but must never be presented as the
    # user's direct words or counted as primary user observations.
    generated_summary = (
        channel == "notes"
        and (metadata.get("source_type") == "daemon_daily_summary"
             or metadata.get("author") == "daemon"
             or "daily note" in str(row.get("title", "")).lower())
    )
    source_class = {
        "notes": "assistant-summary" if generated_summary else "users-own-note",
        "corpus": "user",
        "facts": "extracted-fact",
    }.get(channel, channel)
    source_id = _row_id(row, channel)
    evidence_class = {
        "corpus": "direct_observation",
        "notes": "assistant_summary" if generated_summary else "user_note",
        "facts": "assistant_inference",
    }.get(channel, "proxy")
    return LongitudinalEvent(
        timestamp=str(timestamp),
        text=text,
        source_class=source_class,
        source_id=source_id,
        # Extracted facts are assistant-derived locators, not independent
        # outcome observations. They may help resolve an anchor, but the scan
        # excludes them through its speaker rule.
        speaker="assistant" if generated_summary or channel == "facts" else ("user" if channel in {"corpus", "notes"} else "assistant"),
        evidence_class=evidence_class,
        provenance={
            "lineage": f"{channel}.source",
            "canonical_source_id": row.get("canonical_source_id") or source_id,
            "derived_from": row.get("derived_from") or metadata.get("derived_from"),
            "author": metadata.get("author"),
            "source_type": metadata.get("source_type"),
            "content_date": content_date,
            "record_date": row.get("record_date") or metadata.get("generated_at") or row.get("timestamp"),
            "date_basis": row.get("date_basis") or ("content" if content_date else "record"),
            "date_confidence": "high" if content_date else "low",
        },
    )


def _corpus_events(corpus_manager: Any) -> list[LongitudinalEvent]:
    events: list[LongitudinalEvent] = []
    for entry in (getattr(corpus_manager, "corpus", []) or []):
        if not isinstance(entry, dict) or not entry.get("query"):
            continue
        # Insight-analysis turns contain the user's research prompt and the
        # generated report. They are artifacts of this subsystem, not
        # observations about the user's life, and must not feed later scans.
        if str(entry.get("response_mode") or "").casefold() in {
            "insight-assembly", "insight_analysis", "pattern-analysis",
        }:
            continue
        source_id = str(entry.get("id") or "")
        event = _event_from_row({
            "timestamp": entry.get("timestamp"),
            "text": entry.get("query"),
            "source_id": source_id or None,
            "canonical_source_id": source_id or None,
        }, "corpus")
        if event is not None:
            events.append(event)
    return events


def _anchor_from_locator(
    anchor: TemporalAnchor,
    events: list[LongitudinalEvent],
) -> tuple[Optional[datetime], list[str], list[str]]:
    terms = [term.casefold() for term in anchor.search_terms if term.strip()]
    if not terms:
        return None, [], ["anchor has no date, relative timing, or frozen locator terms"]
    candidates: dict[str, list[str]] = {}
    for event in events:
        if event.speaker.casefold() not in {"user", "human"}:
            continue
        text = event.text.casefold()
        if not any(term in text for term in terms):
            continue
        ts = _parse_date(event.timestamp)
        if ts is None:
            continue
        candidates.setdefault(ts.date().isoformat(), []).append(event.source_id or "")
    if len(candidates) == 1:
        date_text = next(iter(candidates))
        return _parse_date(date_text), list(candidates), [
            f"anchor resolved from frozen locator terms to {date_text}"
        ]
    if not candidates:
        return None, [], ["anchor locator found no dated user-authored event"]
    return None, sorted(candidates), [
        "anchor locator found multiple distinct candidate dates; no date was chosen silently"
    ]


def _resolve_anchor(
    anchor: Optional[TemporalAnchor],
    *,
    events: list[LongitudinalEvent],
    now: datetime,
) -> tuple[Optional[datetime], list[str], list[str]]:
    if anchor is None:
        return None, [], ["no temporal anchor was specified"]
    explicit = _parse_date(anchor.date)
    if explicit is not None:
        return explicit, [explicit.date().isoformat()], [anchor.date_basis or "explicit anchor date"]
    if anchor.relative_value is not None:
        days = _relative_days(anchor.relative_value, anchor.relative_unit)
        direction = 1 if anchor.direction == "future" else -1
        resolved = now + timedelta(days=direction * days)
        return resolved, [resolved.date().isoformat()], [
            anchor.date_basis or f"relative anchor: {anchor.relative_value} {anchor.relative_unit} {anchor.direction}"
        ]
    return _anchor_from_locator(anchor, events)


def _resolve_phases(
    spec: LongitudinalEvidenceSpec,
    *,
    events: list[LongitudinalEvent],
    now: datetime,
) -> tuple[Optional[list[EvidencePhase]], list[str], list[str]]:
    """Resolve explicit or anchor-relative phases without domain defaults."""
    if spec.phases and all(phase.start for phase in spec.phases):
        candidates: list[str] = []
        notes = ["planner supplied explicit phase bounds"]
        resolved: list[EvidencePhase] = []
        for phase in spec.phases:
            start_dt = _parse_date(phase.start)
            if start_dt is None:
                notes.append(f"phase {phase.label} dropped: invalid bounds")
                continue
            end_dt = _parse_date(phase.end)
            if end_dt is None:
                # A start with no parseable end is open-ended (still running)
                # rather than malformed — extend it to "now" instead of
                # discarding the whole comparison over one missing field.
                end_date = now.date().isoformat()
                phase = phase.model_copy(update={"end": end_date}, deep=True)
                end_dt = _parse_date(end_date)
                notes.append(f"phase {phase.label}: open end coerced to {end_date}")
            if start_dt > end_dt:
                notes.append(f"phase {phase.label} dropped: invalid bounds")
                continue
            resolved.append(phase)
        if not resolved:
            return None, candidates, notes + ["no valid phases remained after bounds resolution"]
        if len(spec.phases) >= 2 and len(resolved) < 2:
            return None, candidates, notes + ["fewer than 2 valid phases"]
        phases = resolved
    else:
        anchor_date, candidates, notes = _resolve_anchor(spec.anchor, events=events, now=now)
        if anchor_date is None:
            return None, candidates, notes
        phases = []
        for phase in spec.phases:
            metadata = phase.metadata or {}
            try:
                start_offset = int(metadata["start_offset_days"])
                end_offset = int(metadata["end_offset_days"])
            except (KeyError, TypeError, ValueError):
                return None, candidates, notes + [
                    f"phase {phase.label} lacks explicit bounds or anchor-relative offsets"
                ]
            start = (anchor_date + timedelta(days=start_offset)).date().isoformat()
            end = (anchor_date + timedelta(days=end_offset)).date().isoformat()
            phases.append(phase.model_copy(update={
                "start": start,
                "end": end,
                "date_basis": phase.date_basis or f"offsets from {spec.anchor.label if spec.anchor else 'anchor'}",
                "uncertainty_days": max(
                    phase.uncertainty_days,
                    spec.anchor.uncertainty_days if spec.anchor else 0,
                ),
            }, deep=True))

    ordered = sorted(phases, key=lambda phase: _parse_date(phase.start) or datetime.min)
    for phase in ordered:
        start, end = _parse_date(phase.start), _parse_date(phase.end)
        if start is None or end is None or start > end:
            return None, candidates, notes + [f"phase {phase.label} has invalid bounds"]
    for left, right in zip(ordered, ordered[1:]):
        left_end = _parse_date(left.end)
        right_start = _parse_date(right.start)
        if left_end is not None and right_start is not None and right_start <= left_end:
            return None, candidates, notes + [
                f"phases {left.label} and {right.label} overlap; comparison was not run"
            ]
    return phases, candidates, notes


def _phase_manifest(scan: Optional[LongitudinalScanResult]) -> list[dict[str, Any]]:
    if scan is None:
        return []
    rows: list[dict[str, Any]] = []
    for comparison in scan.comparisons:
        rows.append({
            "label": comparison.phase.label,
            "start": comparison.phase.start,
            "end": comparison.phase.end,
            "date_basis": comparison.phase.date_basis,
            "uncertainty_days": comparison.phase.uncertainty_days,
            "outcome_events": comparison.event_count,
            "proxy_events": comparison.proxy_event_count,
            "proxy_source_ids": [event.source_id for event in comparison.proxy_events[:12]],
            "observations": comparison.observation_denominator,
            "observed_days": comparison.observed_days,
            "calendar_coverage": comparison.coverage,
            "decreased": comparison.decreased_count,
            "increased": comparison.increased_count,
            "unclear_direction": comparison.unclear_direction_count,
            "series_counts": comparison.series_counts,
            "joint_event_count": comparison.joint_event_count,
            "covariate_counts": comparison.covariate_counts,
            "source_ids": [event.source_id for event in comparison.events[:12]],
            "covariate_source_ids": [
                event.source_id for event in comparison.covariate_events[:12]
            ],
        })
    return rows


def _scan_fingerprint(
    scan: Optional[LongitudinalScanResult], events: list[LongitudinalEvent],
) -> str:
    """Identify the exact evidence and resolved phases used for counting."""
    if scan is None:
        return ""
    payload = {
        "phases": [
            {"label": c.phase.label, "start": c.phase.start, "end": c.phase.end}
            for c in scan.comparisons
        ],
        "events": sorted(
            [
                {"id": e.source_id or "", "date": e.timestamp, "text": e.text,
                 "class": e.evidence_class, "role": e.evidence_role}
                for e in events
            ],
            key=lambda row: (row["date"] or "", row["id"], row["text"]),
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()




def _default_claim_chain(spec: LongitudinalEvidenceSpec, reason: str) -> list[DeliberationClaim]:
    return [
        DeliberationClaim(
            claim_id=claim.claim_id,
            proposition=claim.proposition,
            status=_OUTSIDE_AUTHORITY if claim.authority == _OUTSIDE_AUTHORITY else "insufficient",
            confidence=0.0,
            coverage="not assessed" if claim.authority != _OUTSIDE_AUTHORITY else "authority boundary",
            directness="none",
            dependencies=tuple(claim.dependencies),
            authority=claim.authority,
            rationale=(
                "This claim requires an independently qualified professional."
                if claim.authority == _OUTSIDE_AUTHORITY else reason
            ),
        )
        for claim in spec.claims
    ]


_ASSESS_SYSTEM = (
    "You assess a frozen dependency-aware claim chain against a bounded evidence "
    "manifest. Return strict JSON only. Assessment is allowed; prescription, "
    "diagnosis, dose selection, and claims that professional review is unnecessary "
    "are outside authority."
)


async def assess_claim_chain(
    spec: LongitudinalEvidenceSpec,
    scan: Optional[LongitudinalScanResult],
    external_evidence: list[dict[str, Any]],
    channels: list[ChannelStatus],
    limitations: list[str],
    *,
    model_manager=None,
) -> list[DeliberationClaim]:
    """Independently assess each claim without flattening downstream uncertainty."""
    if not spec.claims:
        return []
    if model_manager is None:
        return _default_claim_chain(spec, "claim assessor unavailable")

    internal_rows: list[dict[str, Any]] = []
    if scan is not None:
        for comparison in scan.comparisons:
            for event in comparison.events:
                internal_rows.append({
                    "source_id": event.source_id,
                    "phase": comparison.phase.label,
                    "date": event.timestamp,
                    "text": event.text[:500],
                    "direction": (
                        "increase" if event.polarity > 0
                        else "decrease" if event.polarity < 0
                        else "unclear"
                    ),
                    "direction_basis": event.polarity_basis,
                    "source_class": event.source_class,
                    "evidence_role": "frozen outcome observation",
                })
            for event in comparison.proxy_events:
                internal_rows.append({
                    "source_id": event.source_id,
                    "phase": comparison.phase.label,
                    "date": event.timestamp,
                    "text": event.text[:500],
                    "evidence_role": "exploratory proxy/context; not direct user observation",
                    "source_class": event.source_class,
                })
            for event in comparison.covariate_events:
                internal_rows.append({
                    "source_id": event.source_id,
                    "phase": comparison.phase.label,
                    "date": event.timestamp,
                    "text": event.text[:500],
                    "evidence_role": "phase-aligned covariate/confounder",
                    "source_class": event.source_class,
                })
    merged_internal: dict[tuple[str, str], dict[str, Any]] = {}
    for row in internal_rows:
        key = (
            str(row.get("phase") or ""),
            str(row.get("source_id") or row.get("text") or ""),
        )
        if key in merged_internal:
            merged_internal[key]["evidence_role"] = "outcome and phase-aligned covariate/confounder"
        else:
            merged_internal[key] = row
    internal_rows = list(merged_internal.values())
    internal_rows = _stratified_rows(
        internal_rows, group_key="phase", total=48, per_group=12,
    )
    external_rows = [{
            "source_id": _row_id(row, str(row.get("source_class") or "external")),
            "source_class": row.get("source_class"),
            "title": str(row.get("title") or "")[:300],
            "text": _row_text(row)[:800],
            "url": row.get("url"),
            "date": row.get("date") or row.get("published_date"),
        } for row in external_evidence]
    external_rows = _stratified_rows(
        external_rows, group_key="source_class", total=32, per_group=8,
    )
    evidence_rows = internal_rows + external_rows

    prompt = json.dumps({
        "claims": [claim.model_dump() for claim in spec.claims],
        "phase_summary": _phase_manifest(scan),
        "evidence": evidence_rows,
        "channels": [asdict(channel) for channel in channels],
        "limitations": limitations,
        "required_output": {
            "claims": [{
                "claim_id": "A",
                "status": "supported | partially_supported | mixed | counterevidence | insufficient | outside_authority",
                "confidence": "0..1",
                "coverage": "brief evidence coverage description",
                "directness": "direct | indirect | mixed | none",
                "support_source_ids": ["source id"],
                "refute_source_ids": ["source id"],
                "rationale": "brief reasoning including dependency effects",
            }],
        },
        "rules": [
            "Assess each claim separately and honor its dependencies.",
            "Do not infer symptom absence from missing mentions.",
            "Assistant-derived material is not independent evidence.",
            "An unavailable requested channel lowers coverage.",
            "Outside-authority claims remain outside_authority and do not lower upstream statuses.",
        ],
    }, ensure_ascii=False)
    try:
        text = await model_manager.generate_once(
            prompt,
            system_prompt=_ASSESS_SYSTEM,
            max_tokens=1800,
            temperature=0.0,
            # Strict-JSON verdict call: a reasoning channel (kimi-3 medium
            # effort) consumed the entire 35s budget on the 2026-08-31 run and
            # the verdict fell to the 0.0-confidence default chain — same
            # class as the 08-02 tone-arbiter fix.
            disable_reasoning=True,
        )
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
    except Exception as exc:
        limitations.append(f"claim assessor failed: {exc}")
        return _default_claim_chain(spec, "claim assessor failed")

    proposed = {
        str(item.get("claim_id") or "").upper(): item
        for item in (data.get("claims") or []) if isinstance(item, dict)
    }
    valid_source_ids = {
        str(row.get("source_id")) for row in evidence_rows if row.get("source_id")
    }
    result: list[DeliberationClaim] = []
    for claim in spec.claims:
        if claim.authority == _OUTSIDE_AUTHORITY:
            result.extend(_default_claim_chain(
                spec.model_copy(update={"claims": [claim]}), "outside authority"
            ))
            continue
        item = proposed.get(claim.claim_id)
        if item is None:
            result.append(_default_claim_chain(
                spec.model_copy(update={"claims": [claim]}), "assessor omitted this claim"
            )[0])
            continue
        status = str(item.get("status") or "insufficient").lower()
        if status not in _ASSESSMENT_STATUSES - {_OUTSIDE_AUTHORITY}:
            status = "insufficient"
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        support = tuple(
            source_id for source_id in map(str, item.get("support_source_ids") or [])
            if source_id in valid_source_ids
        )
        refute = tuple(
            source_id for source_id in map(str, item.get("refute_source_ids") or [])
            if source_id in valid_source_ids
        )
        result.append(DeliberationClaim(
            claim_id=claim.claim_id,
            proposition=claim.proposition,
            status=status,
            confidence=confidence,
            coverage=str(item.get("coverage") or "unknown")[:300],
            directness=str(item.get("directness") or "unknown")[:80],
            dependencies=tuple(claim.dependencies),
            authority=claim.authority,
            support_source_ids=support,
            refute_source_ids=refute,
            rationale=str(item.get("rationale") or "")[:800],
        ))
    by_id = {claim.claim_id: claim for claim in result}
    channel_statuses = {channel.channel: channel.status for channel in channels}
    channel_counts = {channel.channel: channel.count for channel in channels}
    supported_statuses = {"supported", "partially_supported"}
    for claim in result:
        if claim.authority == _OUTSIDE_AUTHORITY:
            continue
        has_required_citations = {
            "supported": bool(claim.support_source_ids),
            "partially_supported": bool(claim.support_source_ids),
            "mixed": bool(claim.support_source_ids and claim.refute_source_ids),
            "counterevidence": bool(claim.refute_source_ids),
        }.get(claim.status, True)
        blocked_dependencies = [
            dependency for dependency in claim.dependencies
            if dependency not in by_id
            or by_id[dependency].status not in supported_statuses
        ]
        spec_claim = next(
            item for item in spec.claims if item.claim_id == claim.claim_id
        )
        # An internal channel that "succeeded" with ZERO items is not
        # evidence support — a personal claim must not float on population
        # research alone (2026-08-31 live turn: claims went partially_
        # supported at 0.34/0.45 on one external abstract while the pattern
        # channel held 0 in-window personal observations).
        missing_channels = [
            channel for channel in spec_claim.required_channels
            if channel_statuses.get(channel) != "succeeded"
            or (channel in _INTERNAL_CHANNELS and not channel_counts.get(channel))
        ]
        if not has_required_citations:
            claim.status = "insufficient"
            claim.confidence = min(claim.confidence, 0.25)
            claim.rationale = (
                f"{claim.rationale} Assessment did not cite admissible evidence "
                "for this status."
            ).strip()
        elif claim.status in supported_statuses and blocked_dependencies:
            claim.status = "insufficient"
            claim.confidence = min(claim.confidence, 0.25)
            claim.rationale = (
                f"{claim.rationale} Required dependencies were not supported: "
                f"{', '.join(blocked_dependencies)}."
            ).strip()
        elif claim.status in supported_statuses and missing_channels:
            claim.status = "insufficient"
            claim.confidence = min(claim.confidence, 0.25)
            claim.rationale = (
                f"{claim.rationale} Required evidence channels were not "
                f"successfully available: {', '.join(missing_channels)}."
            ).strip()
    return result


class LongitudinalDeliberationCoordinator:
    """Coordinate one frozen plan across every requested evidence channel."""

    def __init__(
        self,
        *,
        corpus_manager=None,
        adapters: Optional[dict[str, Adapter]] = None,
        note_collector: Optional[Adapter] = None,
        pubmed: Optional[Adapter] = None,
        web: Optional[Adapter] = None,
        wiki: Optional[Adapter] = None,
        model_manager=None,
        claim_assessor: Optional[ClaimAssessor] = None,
        available_channels: tuple[str, ...] = DEFAULT_CHANNELS,
        now: Optional[datetime] = None,
        planner_timeout_s: float = 35.0,
        adapter_timeout_s: float = 25.0,
        # The assessor has NO deterministic fallback (a timeout collapses the
        # whole verdict to insufficient/0.0), unlike the planner. Insight mode
        # streams keepalives, so the longer wait is visible, not a hang.
        assessor_timeout_s: float = 75.0,
    ):
        self.corpus_manager = corpus_manager
        self.adapters = dict(adapters or {})
        for name, adapter in {
            "notes": note_collector,
            "pubmed": pubmed,
            "web": web,
            "wiki": wiki,
        }.items():
            if adapter is not None:
                self.adapters[name] = adapter
        self.model_manager = model_manager
        self.claim_assessor = claim_assessor
        self.available_channels = available_channels
        self.now = now or datetime.now()
        self.planner_timeout_s = planner_timeout_s
        self.adapter_timeout_s = adapter_timeout_s
        self.assessor_timeout_s = assessor_timeout_s

    @staticmethod
    def _adapter_accepts_window(adapter) -> bool:
        try:
            return "window" in inspect.signature(adapter).parameters
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _adapter_accepts(adapter, name: str) -> bool:
        try:
            return name in inspect.signature(adapter).parameters
        except (TypeError, ValueError):
            return False

    async def _call_adapter(
        self, channel: str, query: str,
        window: Optional[tuple[str, str]] = None,
    ) -> tuple[list[dict[str, Any]], ChannelStatus]:
        adapter = self.adapters.get(channel)
        if adapter is None:
            return [], ChannelStatus(channel, "unavailable", attempted=False, reason="adapter not configured")
        try:
            kwargs: dict[str, Any] = {}
            if window is not None and self._adapter_accepts_window(adapter):
                kwargs["window"] = window
            # Research anchors (2026-09-06): the frozen spec's series/outcome
            # axes + synonyms, for adapters that rank rows (pubmed). Set per
            # run in ``_research_anchor`` before the adapter jobs fan out.
            anchor = getattr(self, "_research_anchor", None) or {}
            for name in ("anchor_terms", "concept_synonyms"):
                if anchor.get(name) and self._adapter_accepts(adapter, name):
                    kwargs[name] = anchor[name]
            value = adapter(query, **kwargs) if kwargs else adapter(query)
            if hasattr(value, "__await__"):
                value = await asyncio.wait_for(value, timeout=self.adapter_timeout_s)
            source_status = str(getattr(value, "status", "succeeded") or "succeeded")
            rows = value if isinstance(value, list) else ([value] if value else [])
            normalized = []
            for row in rows:
                item = dict(row) if isinstance(row, dict) else {"text": str(row)}
                item["source_class"] = channel
                item["source_id"] = _row_id(item, channel)
                normalized.append(item)
            return normalized, ChannelStatus(
                channel, source_status if source_status in CHANNEL_STATES else "succeeded",
                attempted=True, count=len(normalized),
                source_ids=tuple(item["source_id"] for item in normalized[:50]),
                queries_attempted=1,
            )
        except Exception as exc:
            return [], ChannelStatus(
                channel, "failed", attempted=True, reason=str(exc),
                queries_attempted=1, query_failures=1,
            )

    async def _call_channel_queries(
        self, channel: str, queries: list[str],
        window: Optional[tuple[str, str]] = None,
    ) -> tuple[list[dict[str, Any]], ChannelStatus]:
        """Run a bounded query set and expose partial source failures."""
        if self.adapters.get(channel) is None:
            return [], ChannelStatus(
                channel, "unavailable", attempted=False,
                reason="adapter not configured",
            )
        unique_queries = list(dict.fromkeys(
            query.strip() for query in queries if query and query.strip()
        ))[:8]
        if not unique_queries:
            return [], ChannelStatus(
                channel, "insufficient", attempted=False,
                reason="no source query was frozen",
            )
        # Keep external fan-out below unkeyed API rate limits (notably PubMed)
        # while preserving partial results when one query stalls or fails.
        concurrency = 2 if channel == "pubmed" else 4
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_call(query: str):
            async with semaphore:
                return await self._call_adapter(channel, query, window=window)

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*(bounded_call(query) for query in unique_queries)),
                timeout=max(10.0, self.adapter_timeout_s * 1.5),
            )
        except asyncio.TimeoutError:
            results = []
            # A channel-level timeout is partial/failed, never an empty
            # success; completed query tasks are cancelled by wait_for.
            return [], ChannelStatus(
                channel, "partial", attempted=True,
                reason="channel query deadline exceeded",
                queries_attempted=len(unique_queries),
                query_failures=len(unique_queries),
            )
        merged: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        empty_statuses: list[str] = []
        successes = 0
        for rows, status in results:
            if status.status == "succeeded":
                successes += 1
                for row in rows:
                    merged.setdefault(str(row["source_id"]), row)
            elif status.status in {"no_results", "no_relevant_results"}:
                empty_statuses.append(status.status)
            else:
                failures.append(status.reason or status.status)
        if not successes:
            if empty_statuses and not failures:
                status_name = (
                    "no_relevant_results"
                    if "no_relevant_results" in empty_statuses else "no_results"
                )
            else:
                status_name = "failed"
        elif not merged:
            # A reachable source returning zero rows is materially different
            # from an adapter failure or a missing credential/configuration.
            status_name = "no_results"
        elif failures or empty_statuses:
            status_name = "partial"
        else:
            status_name = "succeeded"
        rows = list(merged.values())
        return rows, ChannelStatus(
            channel, status_name, attempted=True,
            reason="; ".join(dict.fromkeys(failures + empty_statuses))[:600],
            count=len(rows),
            source_ids=tuple(row["source_id"] for row in rows[:50]),
            queries_attempted=len(unique_queries),
            query_failures=len(failures),
        )

    async def run(
        self,
        query: str,
        proposed_spec: Optional[dict[str, Any]] = None,
    ) -> DeliberationResult:
        try:
            frozen = await asyncio.wait_for(
                plan_deliberation(
                    query,
                    model_manager=self.model_manager,
                    now=self.now,
                    proposed=proposed_spec,
                    available_channels=self.available_channels,
                ),
                timeout=self.planner_timeout_s,
            )
        except asyncio.TimeoutError:
            # B → A: one compact, domain-neutral recovery plan; if it cannot
            # validate, terminal fail-closed behavior prevents raw prose from
            # becoming a computed comparison or external research query.
            try:
                compact = await asyncio.wait_for(
                    compact_recovery_plan(
                        query, model_manager=self.model_manager, now=self.now,
                        available_channels=self.available_channels,
                    ), timeout=min(12.0, max(1.0, self.planner_timeout_s / 3)),
                )
            except asyncio.TimeoutError:
                compact = SpecFreezeResult(
                    "insufficient", None, ["compact recovery planner timed out"],
                    "compact-timeout",
                )
            if compact.status == "ready" and compact.spec is not None:
                compact.limitations.insert(0, f"deliberation planner timed out after {self.planner_timeout_s:g}s")
                compact.limitations.append("compact recovery contract used")
                frozen = compact
            else:
                frozen = SpecFreezeResult(
                    "insufficient", None,
                    [f"deliberation planner timed out after {self.planner_timeout_s:g}s"]
                    + list(compact.limitations),
                    compact.planner_provenance or "timeout",
                )
        if frozen.status != "ready" or frozen.spec is None:
            channels = [
                ChannelStatus("pattern", "insufficient", attempted=False, reason="evidence specification could not be frozen")
            ]
            # A planner outage must not erase explicitly requested external
            # research. Run a bounded, clearly-labelled recovery query for
            # named channels; these rows are evidence about the topic, never
            # a substitute for the missing phase/causal contract.
            fallback_external: list[dict[str, Any]] = []
            for channel in _explicit_channels(query, self.available_channels):
                if channel in {"pattern", "corpus"}:
                    continue
                fallback_queries = recovery_queries(query, channel)
                if channel == "pubmed":
                    from knowledge.pubmed_search import build_pubmed_query_ladder
                    # Recovery has no frozen facets, but it can still use the
                    # same structured ladder rather than sending prose to
                    # E-utilities.
                    fallback_queries = build_pubmed_query_ladder(
                        fallback_queries[0] if fallback_queries else query,
                        limit=8,
                    )
                rows, status = await self._call_channel_queries(
                    channel, fallback_queries
                )
                channels.append(status)
                fallback_external.extend(rows)
            return DeliberationResult(
                frozen, None, [], fallback_external, channels, [],
                {"status": "insufficient", "limitations": frozen.limitations,
                 "planner_provenance": frozen.planner_provenance,
                 "fallback_external": bool(fallback_external),
                 "external_recovery": "bounded explicit-channel query",
                 "external_sources": [
                     {"source_id": _row_id(row, str(row.get("source_class") or "external")),
                      "source_class": row.get("source_class"),
                      "title": str(row.get("title") or "")[:300],
                      "url": row.get("url"),
                      "date": row.get("date") or row.get("published_date"),
                      "snippet": _row_text(row)[:400]}
                     for row in fallback_external[:24]
                 ],
                 "channels": [asdict(channel) for channel in channels]},
            )

        spec = frozen.spec
        limitations = list(frozen.limitations) + list(spec.assumptions)
        internal_events: list[LongitudinalEvent] = []
        channels: list[ChannelStatus] = []

        if "corpus" in spec.requested_channels:
            if self.corpus_manager is None:
                channels.append(ChannelStatus("corpus", "unavailable", reason="corpus manager not configured"))
            else:
                corpus_events = _corpus_events(self.corpus_manager)
                internal_events.extend(corpus_events)
                channels.append(ChannelStatus(
                    "corpus", "succeeded", attempted=True, count=len(corpus_events),
                    source_ids=tuple(event.source_id or "" for event in corpus_events[:50]),
                ))

        # A spec whose phases all carry explicit bounds defines a calendar
        # window BEFORE retrieval — offer it to window-aware internal adapters
        # so dated in-window rows are fetched by DATE, not just similarity
        # (live 2026-08-31: 0 of 70 semantically-retrieved notes fell inside a
        # six-month window while 200+ dated chunks sat in the store).
        explicit_window: Optional[tuple[str, str]] = None
        bounded_phases = [p for p in spec.phases if p.start and p.end]
        if bounded_phases and len(bounded_phases) == len(spec.phases):
            starts = [_parse_date(p.start) for p in bounded_phases]
            ends = [_parse_date(p.end) for p in bounded_phases]
            if all(starts) and all(ends):
                explicit_window = (
                    min(starts).date().isoformat(),
                    max(ends).date().isoformat(),
                )

        external_evidence: list[dict[str, Any]] = []
        adapter_jobs: list[tuple[str, list[str]]] = []
        for channel in spec.requested_channels:
            if channel in {"pattern", "corpus"}:
                continue
            channel_query = spec.research_queries.get(channel) or " ".join(
                spec.outcome_terms + spec.supporting_facets[:2]
            )
            if channel == "pubmed":
                # PubMed needs source syntax and concept broadening, not the
                # conversational recovery prose. The ladder is generic and
                # driven entirely by the frozen spec/query.
                from knowledge.pubmed_search import build_pubmed_query_ladder
                # The frozen spec's own named subjects are a better anchor
                # signal than positional word order — a crude term extractor
                # can lose the actual subject noun to stopwording (2026-09-06:
                # "rest days off medication effects" anchored on "rest").
                # Named series come first (co_occurrence's own compared
                # subjects), then outcome terms.
                anchor_terms = [*spec.series_terms.keys(), *spec.outcome_terms]
                queries = build_pubmed_query_ladder(
                    channel_query,
                    supporting_facets=spec.supporting_facets,
                    refuting_facets=spec.refuting_facets,
                    rival_explanations=spec.rival_explanations,
                    concept_synonyms=spec.concept_synonyms,
                    anchor_terms=anchor_terms,
                    limit=8,
                )
            else:
                queries = [channel_query]
            if channel in _MULTI_QUERY_CHANNELS and channel != "pubmed":
                queries.extend(spec.supporting_facets)
                queries.extend(spec.refuting_facets)
                queries.extend(spec.rival_explanations)
                queries.extend(spec.confounder_indicators)
            adapter_jobs.append((channel, queries))
        self._research_anchor = {
            "anchor_terms": [*spec.series_terms.keys(), *spec.outcome_terms],
            "concept_synonyms": dict(spec.concept_synonyms or {}),
        }
        adapter_results = await asyncio.gather(*(
            self._call_channel_queries(
                channel, queries,
                window=explicit_window if channel in {"notes", "facts"} else None,
            )
            for channel, queries in adapter_jobs
        ))
        internal_channel_dates: dict[str, list[datetime]] = {}
        for (channel, _queries), (rows, status) in zip(
            adapter_jobs, adapter_results,
        ):
            channels.append(status)
            if channel in {"notes", "facts"}:
                for row in rows:
                    event = _event_from_row(row, channel)
                    if event is not None:
                        internal_events.append(event)
                        ts = _parse_date(event.timestamp)
                        if ts is not None:
                            internal_channel_dates.setdefault(channel, []).append(ts)
            else:
                external_evidence.extend(rows)

        phases, anchor_candidates, phase_notes = _resolve_phases(
            spec, events=internal_events, now=self.now,
        )
        limitations.extend(phase_notes)
        scan: Optional[LongitudinalScanResult] = None
        in_window_counts: dict[str, dict[str, int]] = {}
        if phases is not None:
            # Semantic retrieval is date-blind; a windowed question can fill a
            # channel's cap with rows from the wrong years (2026 notes answered
            # a 2025-H1 question on 2026-08-31). Full history stays available
            # for anchor location — this is honest accounting, not filtering.
            bounds = [(_parse_date(p.start), _parse_date(p.end)) for p in phases]
            window_start = min(b[0] for b in bounds if b[0] is not None)
            window_end = max(b[1] for b in bounds if b[1] is not None) + timedelta(days=1)
            for channel_name, dates in internal_channel_dates.items():
                in_window = sum(1 for ts in dates if window_start <= ts < window_end)
                in_window_counts[channel_name] = {
                    "dated_rows": len(dates), "in_window": in_window,
                }
                if dates and in_window * 2 < len(dates):
                    limitations.append(
                        f"{channel_name} retrieval is date-blind: only {in_window} of "
                        f"{len(dates)} dated rows fall inside the requested window "
                        f"{window_start.date()}..{(window_end - timedelta(days=1)).date()}"
                    )
        if phases is not None:
            spec = spec.model_copy(update={"phases": phases}, deep=True)
            frozen.spec = spec
            scan = run_longitudinal_scan(spec, internal_events)
            matched = sum(comparison.event_count for comparison in scan.comparisons)
            channels.append(ChannelStatus(
                "pattern", "succeeded", attempted=True, count=matched,
                source_ids=tuple(
                    [event.source_id or "" for comparison in scan.comparisons
                     for event in comparison.events][:50]
                ),
            ))
        else:
            channels.append(ChannelStatus(
                "pattern", "insufficient", attempted=True,
                reason="; ".join(phase_notes) or "phase bounds unresolved",
            ))

        if self.claim_assessor is not None:
            try:
                assessed = self.claim_assessor(
                    spec, scan, external_evidence, channels, limitations,
                )
                claim_chain = (
                    await asyncio.wait_for(assessed, timeout=self.assessor_timeout_s)
                    if hasattr(assessed, "__await__") else assessed
                )
                if not isinstance(claim_chain, list):
                    raise TypeError("claim assessor returned a non-list result")
            except Exception as exc:
                limitations.append(f"injected claim assessor failed: {exc}")
                claim_chain = _default_claim_chain(
                    spec, "injected claim assessor failed",
                )
        else:
            try:
                claim_chain = await asyncio.wait_for(
                    assess_claim_chain(
                        spec, scan, external_evidence, channels, limitations,
                        model_manager=self.model_manager,
                    ),
                    timeout=self.assessor_timeout_s,
                )
            except asyncio.TimeoutError:
                limitations.append(
                    f"claim assessor timed out after {self.assessor_timeout_s:g}s"
                )
                claim_chain = _default_claim_chain(spec, "claim assessor timed out")

        external_sources = [{
            "source_id": _row_id(row, str(row.get("source_class") or "external")),
            "source_class": row.get("source_class"),
            "title": str(row.get("title") or "")[:300],
            "url": row.get("url"),
            "date": row.get("date") or row.get("published_date"),
            "snippet": _row_text(row)[:400],
        } for row in _stratified_rows(
            # Keep a real literature set intact: ten relevant PubMed records
            # should survive manifest construction when they are available.
            external_evidence, group_key="source_class", total=40, per_group=20,
        )]
        phase_summary = _phase_manifest(scan)
        spec_fingerprint = _json_fingerprint(spec.model_dump())
        evidence_fingerprint = _scan_fingerprint(scan, internal_events)
        external_fingerprint = _json_fingerprint(sorted(
            (str(row.get("source_id") or _row_id(row, str(row.get("source_class") or "external"))),
             str(row.get("date") or row.get("published_date") or ""))
            for row in external_evidence
        ))
        counted_output_fingerprint = _json_fingerprint(phase_summary)
        manifest = {
            "status": "ready" if scan is not None else "partial",
            "planner_provenance": frozen.planner_provenance,
            "spec": spec.model_dump(),
            "anchor_candidates": anchor_candidates,
            "phase_summary": phase_summary,
            "reproducibility": {
                "spec_fingerprint": spec_fingerprint,
                "evidence_fingerprint": evidence_fingerprint,
                "external_fingerprint": external_fingerprint,
                "counted_output_fingerprint": counted_output_fingerprint,
                "event_input_count": len(internal_events),
                "status": "captured",
                "counting_contract": "canonical spec/events/external IDs sorted with resolved phase bounds",
            },
            "internal_source_counts": {
                channel.channel: channel.count
                for channel in channels if channel.channel in _INTERNAL_CHANNELS
            },
            "internal_in_window_counts": in_window_counts,
            "external_sources": external_sources,
            "channels": [asdict(channel) for channel in channels],
            "claim_chain": [asdict(claim) for claim in claim_chain],
            "limitations": list(dict.fromkeys(limitations)),
            "sensitivity": scan.sensitivity if scan is not None else [],
            # Bucket-level co-occurrence headline numbers survive even when
            # downstream compaction truncates the per-week phase rows.
            "co_occurrence": scan.co_occurrence if scan is not None else {},
            "doctrine": {
                "absence_of_mention": "not evidence of symptom absence",
                "assistant_material": "locator only, not independent evidence",
                "authority": "assessment permitted; prescription and diagnosis require a qualified professional",
            },
        }
        return DeliberationResult(
            frozen, scan, internal_events, external_evidence, channels,
            claim_chain, manifest,
        )
