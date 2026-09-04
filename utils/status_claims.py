# utils/status_claims.py
"""
Module Contract
- Purpose: deterministic, word-bounded detection of narrative/status claims
  (enrollment, employment, residence) that CONTRADICT a CURRENT profile
  fact. 2026-09-04 class-closing fix — the "Current Life State" narrative
  (memory/memory_consolidator.py, generate_narrative_context) wrote "he'd
  withdrawn from the fall semester" while the profile held a CURRENT
  `enrolled_in` fact plus a `dropped=<one course>` fact; the user's own
  words said only one course was dropped, not the whole semester.
- Inputs:
  - status_claim_conflicts(text, profile_facts) -> List[StatusClaimConflict]
  - remove_conflicting_claims(text, profile_facts) -> (revised_text, conflicts)
  - authoritative_facts_block(profile_facts) -> str  (prompt-injection block)
- Behavior: purely mechanical pattern matching + a profile-fact lookup the
  caller supplies. Never rewrites or infers a replacement claim — only
  detects a contradiction, removes the offending sentence, and appends one
  visible caution line so the discrepancy is never silently swallowed.
  NEVER touches stored data (profile/graph/corpus untouched) — this module
  operates on generated TEXT only.
- Dependencies: none beyond stdlib. Leaf module — imported by
  memory/memory_consolidator.py (narrative path) and
  utils/daily_notes_generator.py (daily-note path); callers are responsible
  for fetching `profile_facts` from UserProfile (this module never imports
  UserProfile, keeping it dependency-free and independently testable).
- profile_facts shape: an iterable of dicts, each with at least a
  "relation" key and a "value" key (the shape UserProfile fact dicts
  already carry). Only facts the caller considers CURRENT should be passed
  in — this module does not re-check is_current/TTL itself.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

# --- Relation families ------------------------------------------------
# Word-bounded, mechanical — deliberately narrow (under-fire by design,
# same doctrine as build_temporal_claim_audit above it in the daily-note
# path: never guess, never silently pick a side).

ENROLLMENT_RELATIONS = {"enrolled_in", "program", "dropped"}
EMPLOYMENT_RELATIONS = {"works_at", "occupation", "employer"}
RESIDENCE_RELATIONS = {"lives_in"}

STATUS_RELATIONS = ENROLLMENT_RELATIONS | EMPLOYMENT_RELATIONS | RESIDENCE_RELATIONS

_FAMILY_RELATIONS: Dict[str, set] = {
    "enrollment": ENROLLMENT_RELATIONS,
    "employment": EMPLOYMENT_RELATIONS,
    "residence": RESIDENCE_RELATIONS,
}

# --- Claim-shape patterns ----------------------------------------------
# Deliberately the NARROW "big" claim shapes that a current family fact
# would contradict — a specific course drop ("dropped CSE 6200") does not
# match the enrollment pattern (no "out"/semester-shaped tail), so it is
# never flagged; it is exactly what a current `dropped=<course>` fact
# supports.

_ENROLLMENT_WITHDRAW_RE = re.compile(
    r"\b(?:withdrew|withdrawn|dropped\s+out|left|quit)\b[^.?!\n]{0,60}?"
    r"\b(?:the\s+)?(?:fall|spring|summer|winter)?\s*"
    r"(?:semester|term|school|program|university|college)\b",
    re.IGNORECASE,
)

_EMPLOYMENT_LOSS_RE = re.compile(
    r"\b(?:quit(?:ting)?|was\s+fired|got\s+fired|lost\s+(?:his|her|their)\s+job|"
    r"(?:is|was)\s+unemployed|no\s+longer\s+work(?:s|ing)?)\b",
    re.IGNORECASE,
)

_RESIDENCE_MOVE_RE = re.compile(
    r"\bmoved\s+(?:back\s+)?to\s+[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,3}\b|"
    r"\bnow\s+lives\s+in\s+[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,3}\b",
    re.IGNORECASE,
)

_FAMILY_PATTERNS: Dict[str, re.Pattern] = {
    "enrollment": _ENROLLMENT_WITHDRAW_RE,
    "employment": _EMPLOYMENT_LOSS_RE,
    "residence": _RESIDENCE_MOVE_RE,
}


@dataclass
class StatusClaimConflict:
    family: str
    claim_text: str
    relation: str
    value: str
    start: int
    end: int


def _sentence_span(text: str, start: int, end: int) -> Tuple[int, int]:
    """Return the (start, end) offsets of the sentence-like window
    containing a matched claim, mirroring
    utils.daily_notes_generator._claim_sentence but returning offsets so
    the caller can excise the span."""
    left_candidates = [text.rfind(mark, 0, start) for mark in ("\n", ".", "?", "!")]
    left = max(left_candidates) + 1
    right_candidates = [
        pos for mark in ("\n", ".", "?", "!")
        if (pos := text.find(mark, end)) >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return left, right


def _current_facts_by_family(profile_facts: Iterable[Dict[str, Any]], family: str) -> List[Dict[str, Any]]:
    relations = _FAMILY_RELATIONS[family]
    out = []
    for f in profile_facts or []:
        if not isinstance(f, dict):
            continue
        if f.get("relation") in relations and f.get("value"):
            out.append(f)
    return out


def status_claim_conflicts(
    text: str, profile_facts: Iterable[Dict[str, Any]]
) -> List[StatusClaimConflict]:
    """Find generated-text claims that contradict a current profile fact.

    Deliberately mechanical: a claim only counts as a conflict when (a) it
    matches one of the narrow family regexes AND (b) the caller-supplied
    `profile_facts` contains a current fact in the SAME family. No
    profile_facts -> no conflicts, regardless of what the text claims.
    """
    if not text or not profile_facts:
        return []

    conflicts: List[StatusClaimConflict] = []
    for family, pattern in _FAMILY_PATTERNS.items():
        contradicting = _current_facts_by_family(profile_facts, family)
        if not contradicting:
            continue
        fact = contradicting[0]
        seen_spans = set()
        for match in pattern.finditer(text):
            start, end = _sentence_span(text, match.start(), match.end())
            if (start, end) in seen_spans:
                continue
            seen_spans.add((start, end))
            claim_text = " ".join(text[start:end].split())
            conflicts.append(
                StatusClaimConflict(
                    family=family,
                    claim_text=claim_text,
                    relation=str(fact.get("relation", "")),
                    value=str(fact.get("value", "")),
                    start=start,
                    end=end,
                )
            )

    conflicts.sort(key=lambda c: c.start)
    return conflicts


def _merge_spans(conflicts: List[StatusClaimConflict]) -> List[Tuple[int, int]]:
    spans = sorted({(c.start, c.end) for c in conflicts})
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def remove_conflicting_claims(
    text: str, profile_facts: Iterable[Dict[str, Any]]
) -> Tuple[str, List[StatusClaimConflict]]:
    """Excise contradicted claim sentences from generated text and append a
    visible caution line per removed claim. Never touches stored data —
    operates only on the in-memory generated string handed back to the
    caller (who then persists the returned text, not the original)."""
    conflicts = status_claim_conflicts(text, profile_facts)
    if not conflicts:
        return text, []

    revised = text
    for start, end in sorted(_merge_spans(conflicts), reverse=True):
        revised = revised[:start] + revised[end:]
    revised = re.sub(r"[ \t]{2,}", " ", revised)
    revised = re.sub(r"\n{3,}", "\n\n", revised).strip()

    caution_lines = [
        f'[CAUTION: a draft sentence claiming "{c.claim_text}" was removed — '
        f"current profile: {c.relation}={c.value}]"
        for c in conflicts
    ]
    if revised:
        revised = revised.rstrip() + "\n\n" + "\n".join(caution_lines)
    else:
        revised = "\n".join(caution_lines)

    return revised, conflicts


def authoritative_facts_block(profile_facts: Iterable[Dict[str, Any]]) -> str:
    """Render current enrollment/employment/residence facts as a short
    'do not contradict' block for injection into a generation prompt.
    Returns "" when there is nothing to inject."""
    lines = []
    for f in profile_facts or []:
        if not isinstance(f, dict):
            continue
        rel = f.get("relation", "")
        val = f.get("value", "")
        if rel in STATUS_RELATIONS and val:
            lines.append(f"- {rel} = {val}")
    if not lines:
        return ""
    return "AUTHORITATIVE CURRENT FACTS — do not contradict:\n" + "\n".join(lines)
