# utils/completed_plan_claims.py
"""
Module Contract
- Purpose: deterministic detection of a [TEMPORAL GROUNDING] narrative
  sentence that still frames a social/personal plan as forward-looking
  ("hanging out with a friend Saturday is pending") after the user has
  ALREADY reported it done in a later message. Codex audit (2026-09-06):
  completed plans render as pending in the narrative because nothing
  compared the narrative's forward-looking sentences against the user's
  later past-tense reports — the two existing post-checks
  (utils.streak_claims.remove_stale_streak_claims, utils.status_claims.
  remove_conflicting_claims) only catch a stale COUNT or a CONTRADICTED
  profile fact, not a stale PLAN.
- Inputs:
  - plan_sentences(narrative) -> List[str]
  - completed_by_user(plan_sentence, user_statements, as_of=None) -> Optional[dict]
  - remove_completed_plan_claims(narrative, user_statements, as_of=None)
        -> (revised_text, removed_sentences)
- Behavior: purely mechanical, mirroring memory.thread_store.
  check_quick_resolutions' shape (past-action completion cue + keyword
  overlap + digit-conflict guard) and utils.streak_claims' excise-and-
  caution-line output shape. A plan sentence needs a CLOSED forward-looking
  cue ("plan(s|ning) to", "going to", "will", "want(s) to", "intend(s) to",
  "scheduled", "upcoming", "pending", "hasn't … yet", "need(s) to"). A user
  statement resolves it only when it (a) carries a past-tense/completion
  cue, (b) shares >=2 content tokens with the plan sentence, (c) has no
  digit-token conflict (numbered items like "HW6" vs "hw7" never cross-
  resolve — same guard as check_quick_resolutions), and (d) is dated ON OR
  AFTER ``as_of`` (the narrative's own generation day — a statement from
  BEFORE the plan text was written cannot be reporting on it; use
  timestamps, never guess). ``user_statements`` is the SAME corpus-shaped
  list (dicts with ``user_text``/``query`` + ``timestamp``) the streak
  ledger already receives; assistant text is never a source. Never touches
  stored data — generated text only. Idempotent: a sentence already
  rewritten into a "[CAUTION: ...]" line is never re-scanned as a plan
  sentence.
- Dependencies: stdlib only (leaf module).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

# --- Forward-looking plan cues (closed grammatical set) -----------------
_PLAN_CUE_RE = re.compile(
    r"\b(?:plan(?:s|ning)?\s+to|going\s+to|will\b|wants?\s+to|intends?\s+to|"
    r"scheduled|upcoming|pending|needs?\s+to)\b",
    re.IGNORECASE,
)
_HASNT_YET_RE = re.compile(r"\bhas(?:n'?t|\s+not)\b[^.?!\n]{0,60}?\byet\b", re.IGNORECASE)


def _is_plan_sentence(sentence: str) -> bool:
    return bool(_PLAN_CUE_RE.search(sentence) or _HASNT_YET_RE.search(sentence))


# --- Completion cues (past-tense/perfective) -----------------------------
_STRONG_COMPLETION_RE = re.compile(
    r"\b(?:did|done|finished|sent|went|saw|met|hung\s+out|hanging\s+out\s+with|made\s+it|"
    r"got\s+\w+(?:\s+\w+){0,3}\s+done|attended|completed|showed\s+up|caught\s+up|"
    r"had\s+(?:a|the)\b)",
    re.IGNORECASE,
)
# A bare "-ed" word is a weak signal on its own (plenty of forward-planning
# verbs are past-tense in form — "wanted", "planned") — excluded so a plan
# restated in reported speech doesn't masquerade as a completion report.
_GENERIC_EDWORD_RE = re.compile(r"\b\w+ed\b", re.IGNORECASE)
_PLANNING_EDWORD_STOP = frozenset({
    "wanted", "planned", "scheduled", "intended", "needed", "hoped",
    "expected", "hasnt", "hadnt",
})


def _has_completion_cue(text: str) -> bool:
    if _STRONG_COMPLETION_RE.search(text):
        return True
    for m in _GENERIC_EDWORD_RE.finditer(text):
        word = re.sub(r"[^a-z]", "", m.group(0).lower())
        if word not in _PLANNING_EDWORD_STOP:
            return True
    return False


# --- Content-token overlap + digit-conflict guard (mirrors
# memory.thread_store.check_quick_resolutions' shape; duplicated rather
# than imported to keep this a dependency-free leaf module) --------------
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in",
    "on", "for", "my", "our", "his", "her", "their", "with", "at", "by",
    "this", "that", "it", "be", "as", "still", "not", "but",
})


def _norm_tokens(text: str) -> set:
    words = re.findall(r"[a-z]*\d[a-z\d]*|[a-z]{3,}", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS}


def _digit_tokens(tokens: set) -> set:
    return {w for w in tokens if any(c.isdigit() for c in w)}


def _coerce_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def _sentence_span(text: str, start: int, end: int) -> Tuple[int, int]:
    """(start, end) offsets of the sentence-like window containing a match,
    matching utils.status_claims._sentence_span / utils.streak_claims'
    excision windows."""
    left_candidates = [text.rfind(mark, 0, start) for mark in ("\n", ".", "?", "!")]
    left = max(left_candidates) + 1
    right_candidates = [pos for mark in ("\n", ".", "?", "!") if (pos := text.find(mark, end)) >= 0]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return left, right


def _plan_claim_spans(narrative: str) -> List[Tuple[str, int, int]]:
    """(sentence_text, start, end) for every forward-looking plan sentence
    in ``narrative``. Skips sentences already inside a prior "[CAUTION: ...]"
    line so a second pass over already-processed text is idempotent."""
    if not narrative:
        return []
    spans: List[Tuple[str, int, int]] = []
    seen = set()
    for pattern in (_PLAN_CUE_RE, _HASNT_YET_RE):
        for m in pattern.finditer(narrative):
            start, end = _sentence_span(narrative, m.start(), m.end())
            if (start, end) in seen:
                continue
            sentence = narrative[start:end]
            if sentence.strip().startswith("[CAUTION"):
                continue
            seen.add((start, end))
            spans.append((sentence, start, end))
    spans.sort(key=lambda s: s[1])
    return spans


def plan_sentences(narrative: str) -> List[str]:
    """Sentences in ``narrative`` carrying a forward-looking plan cue plus
    an object (the rest of the sentence)."""
    return [" ".join(s.split()) for s, _start, _end in _plan_claim_spans(narrative)]


def _statement_text(item: Dict[str, Any]) -> str:
    return str(item.get("user_text") or item.get("query") or item.get("user") or "")


def completed_by_user(
    plan_sentence: str,
    user_statements: List[Dict[str, Any]],
    as_of: Any = None,
) -> Optional[Dict[str, Any]]:
    """The earliest user statement that reports the plan's object done.

    A statement qualifies when it (a) is dated on or after ``as_of`` when a
    floor is given (the caller passes the narrative's own lookback horizon —
    the statements it receives are COUNT-windowed, so "today" was the wrong
    floor: a plan reported done yesterday stayed pending), (b) carries a
    past-tense/completion cue, (c) shares >=2 content tokens with
    ``plan_sentence``, and (d) has no digit-token conflict with it (numbered
    items never cross-resolve). Newest statement wins: a LATER user statement
    that re-asserts the plan (forward-looking cue + same overlap) cancels the
    completion — the user made the plan again. Returns the matching
    corpus-shaped statement dict, or None.
    """
    if not isinstance(plan_sentence, str) or not plan_sentence.strip():
        return None
    plan_tokens = _norm_tokens(plan_sentence)
    if not plan_tokens:
        return None
    plan_digits = _digit_tokens(plan_tokens)
    floor = _coerce_date(as_of) if as_of is not None else None

    candidates: List[Tuple[date, Dict[str, Any]]] = []
    replans: List[date] = []
    for item in user_statements or []:
        if not isinstance(item, dict):
            continue
        text = _statement_text(item)
        if not text.strip():
            continue
        day = _coerce_date(item.get("timestamp") or item.get("turn_id"))
        if day is None or (floor is not None and day < floor):
            continue
        stmt_tokens = _norm_tokens(text)
        if len(plan_tokens & stmt_tokens) < 2:
            continue
        stmt_digits = _digit_tokens(stmt_tokens)
        if plan_digits and stmt_digits and not (plan_digits & stmt_digits):
            continue
        if _has_completion_cue(text):
            candidates.append((day, item))
        elif plan_sentences(text):
            replans.append(day)

    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    latest_day, latest_item = candidates[-1]
    if any(day > latest_day for day in replans):
        return None
    return latest_item


@dataclass
class _CompletedPlanClaim:
    sentence: str
    start: int
    end: int
    statement: Dict[str, Any]


def remove_completed_plan_claims(
    narrative: str,
    user_statements: List[Dict[str, Any]],
    as_of: Any = None,
) -> Tuple[str, List[str]]:
    """Excise each completed-plan sentence and append one CAUTION line per
    removal (mirrors utils.streak_claims.remove_stale_streak_claims' output
    shape). Never removes a sentence lacking a plan cue; never touches the
    streak-ledger or status-claim caution lines. Returns (revised_text,
    removed_sentence_texts)."""
    spans = _plan_claim_spans(narrative)
    if not spans:
        return narrative, []

    today = _coerce_date(as_of) or date.today()
    removed: List[_CompletedPlanClaim] = []
    for sentence, start, end in spans:
        match = completed_by_user(sentence, user_statements, as_of=today)
        if match is not None:
            removed.append(_CompletedPlanClaim(sentence=sentence, start=start, end=end, statement=match))

    if not removed:
        return narrative, []

    revised = narrative
    for claim in sorted(removed, key=lambda c: c.start, reverse=True):
        revised = revised[:claim.start] + revised[claim.end:]
    revised = re.sub(r"[ \t]{2,}", " ", revised)
    revised = re.sub(r"\n[ \t]+", "\n", revised)
    revised = re.sub(r"\n{3,}", "\n\n", revised).strip()

    caution_lines = []
    removed_texts = []
    for claim in sorted(removed, key=lambda c: c.start):
        sentence_text = " ".join(claim.sentence.split())
        removed_texts.append(sentence_text)
        stmt_text = " ".join(_statement_text(claim.statement).split())
        if len(stmt_text) > 160:
            stmt_text = stmt_text[:160].rstrip() + "…"
        stmt_day = _coerce_date(claim.statement.get("timestamp") or claim.statement.get("turn_id"))
        date_str = stmt_day.isoformat() if stmt_day else "an unspecified date"
        caution_lines.append(
            f'[CAUTION: a draft sentence claiming "{sentence_text}" was removed — '
            f'the user reported it done on {date_str}: "{stmt_text}"]'
        )

    revised = (revised.rstrip() + "\n\n" if revised else "") + "\n".join(caution_lines)
    return revised, removed_texts
