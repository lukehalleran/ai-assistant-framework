"""Supersede profile facts whose own content proves they expired.

The drop-deadline class: `deadline = "drop deadline is Fri 2026-08-28 at
3 PM Central"` is stale on Aug 30 by ARITHMETIC, not judgment — the fact's
value carries an explicit absolute date that has passed. This curator:

- fires only when BOTH hold: the relation names a dated event (allowlist)
  AND the value contains an explicit, fully-specified date (4-digit year
  required — "Aug 31" alone is ambiguous and produces NO action);
- waits a grace period past the date (the fact may still be referenced in
  the immediate aftermath: "did I make the deadline?");
- proposes SUPERSESSION (is_current=False + reason), never deletion — the
  fact stays queryable as history ("what was the deadline?");
- deliberately does NOT use temporal_resolver.resolve_date_expression,
  which is future-biased (a passed date resolves to NEXT year — exactly
  wrong for staleness).

Prime directive: anything ambiguous → no action.
"""

import re
from datetime import datetime, timedelta
from typing import List, Optional

from memory.curation.engine import StoreBundle, new_proposal_id
from memory.curation.types import (
    Confidence,
    CurationProposal,
    Instrument,
    ItemChange,
    SentinelResult,
)

# Relations that describe a dated event. Substring match on the canonical
# relation — under-fires by design (a relation not on this list is never
# touched, whatever its value says).
_DATED_EVENT_RELATION_CUES = (
    "deadline", "due", "appointment", "exam", "meeting", "interview",
    "event", "scheduled", "session_date",
)

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_ISO_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
# "Aug 31, 2026" / "August 31 2026" / "31 Aug 2026" — 4-digit year REQUIRED.
_MDY_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+"
    r"(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d{2})\b", re.IGNORECASE)
_DMY_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+"
    r"(20\d{2})\b", re.IGNORECASE)

GRACE_HOURS_DEFAULT = 48


def extract_explicit_dates(text: str) -> List[datetime]:
    """Every fully-specified (year-bearing) calendar date in the text.
    Invalid calendar combinations are skipped, never guessed."""
    out: List[datetime] = []
    for m in _ISO_RE.finditer(text or ""):
        try:
            out.append(datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            continue
    for m in _MDY_RE.finditer(text or ""):
        try:
            out.append(datetime(int(m.group(3)),
                                _MONTHS[m.group(1).lower()[:3]],
                                int(m.group(2))))
        except (ValueError, KeyError):
            continue
    for m in _DMY_RE.finditer(text or ""):
        try:
            out.append(datetime(int(m.group(3)),
                                _MONTHS[m.group(2).lower()[:3]],
                                int(m.group(1))))
        except (ValueError, KeyError):
            continue
    return out


def fact_expired(relation: str, value: str, *, now: Optional[datetime] = None,
                 grace_hours: int = GRACE_HOURS_DEFAULT) -> Optional[datetime]:
    """The latest explicit date in an event-relation fact, IF every explicit
    date in it has passed the grace window. Any future (or in-grace) date →
    None: a value like "opens 2026-08-01, closes 2026-12-01" is not stale
    until the LAST date passes."""
    rel = (relation or "").lower()
    if not any(cue in rel for cue in _DATED_EVENT_RELATION_CUES):
        return None
    dates = extract_explicit_dates(value or "")
    if not dates:
        return None
    now = now or datetime.now()
    cutoff = now - timedelta(hours=grace_hours)
    latest = max(dates)
    # End-of-day semantics: a date is "passed" only once the whole day is
    # over (plus grace) — never stale on the deadline day itself.
    latest_end = latest + timedelta(days=1)
    return latest if latest_end < cutoff else None


class TemporalStalenessCurator:
    name = "temporal_staleness"

    def __init__(self, grace_hours: int = GRACE_HOURS_DEFAULT):
        self.grace_hours = grace_hours

    def sentinels(self, stores: StoreBundle) -> List[SentinelResult]:
        now = datetime(2026, 9, 15)
        past = fact_expired(
            "deadline", "drop deadline is Fri 2026-08-28 at 3 PM Central",
            now=now, grace_hours=self.grace_hours)
        future = fact_expired(
            "deadline", "self-service W closes Sat Oct 31, 2026",
            now=now, grace_hours=self.grace_hours)
        yearless = fact_expired(
            "deadline", "homework due Aug 31", now=now,
            grace_hours=self.grace_hours)
        wrong_relation = fact_expired(
            "birthday", "born 1993-02-09", now=now,
            grace_hours=self.grace_hours)
        return [
            SentinelResult(name="past_deadline_expires", passed=past is not None),
            SentinelResult(name="future_deadline_kept", passed=future is None),
            SentinelResult(name="yearless_date_no_action", passed=yearless is None),
            SentinelResult(name="non_event_relation_no_action",
                           passed=wrong_relation is None),
        ]

    def scan(self, stores: StoreBundle) -> List[CurationProposal]:
        profile = stores.user_profile
        if profile is None:
            return []
        proposals: List[CurationProposal] = []
        cats = (getattr(profile, "profile", None) or {}).get("categories", {})
        for facts_list in cats.values():
            if not isinstance(facts_list, list):
                continue
            for fact in facts_list:
                if not isinstance(fact, dict):
                    continue
                if not fact.get("is_current", True):
                    continue
                if not fact.get("fact_id"):
                    continue  # can't address it reversibly — leave alone
                rel = fact.get("relation", "")
                val = str(fact.get("value", ""))
                expired = fact_expired(rel, val, grace_hours=self.grace_hours)
                if expired is None:
                    continue
                proposals.append(CurationProposal(
                    proposal_id=new_proposal_id(),
                    curator=self.name,
                    instrument=Instrument.METADATA,
                    confidence=Confidence.DETERMINISTIC,
                    title=f"Expired: {rel} = {val[:60]}",
                    evidence=(
                        f"The fact's own value names {expired.date().isoformat()}, "
                        f"which is more than {self.grace_hours}h past. Supersede "
                        "(is_current=False, kept as history) — the event date "
                        "has arithmetically passed."
                    ),
                    items=[ItemChange(
                        store="profile", doc_id=fact["fact_id"],
                        change_type="supersede_profile_fact",
                        after={"reason": f"date_passed:{expired.date().isoformat()}"},
                    )],
                ))
        return proposals
