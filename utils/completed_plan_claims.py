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
  check_quick_resolutions' shape (past-action completion cue + digit-
  conflict guard) and utils.streak_claims' excise-and-caution-line output
  shape. A plan sentence needs a CLOSED forward-looking cue ("plan(s|ning)
  to", "going to", "will", "want(s) to", "intend(s) to", "scheduled",
  "upcoming", "pending", "hasn't … yet", "need(s) to") and is NOT a
  recurring commitment (a habitual cue — "every", "each week", "daily" —
  disqualifies it; a completed occurrence never closes a recurring plan,
  2026-09-06 B6). A user statement resolves it via IDENTITY + STATUS
  (2026-09-06 B6/B7, replacing generic token overlap): a qualifying CLAUSE
  (memory.fact_source._split_clauses) must (a) carry a past-tense/
  completion cue not governed by a hedge/negation marker in that same
  clause ("haven't … yet", "still need to", "if", "would", "planning to"),
  (b) not report a DIFFERENT status ("cancelled", "rescheduled", "tried
  to", "couldn't"), (c) name the plan's object HEAD noun — directly, via a
  determiner-anchored reference ("that/the/my/this <head>"), or via a bare
  "it" only when the immediately previous same-day statement names the
  head, (d) not be a document-sharing verb over a document noun unless the
  plan itself is about a document, and (e) have no digit-token conflict
  with the plan (numbered items like "HW6" vs "hw7" never cross-resolve).
  A statement is only considered when dated ON OR AFTER ``as_of`` (the
  narrative's own generation day — a statement from BEFORE the plan text
  was written cannot be reporting on it; use timestamps, never guess).
  ``user_statements`` is the SAME corpus-shaped list (dicts with
  ``user_text``/``query`` + ``timestamp``) the streak ledger already
  receives; assistant text is never a source. Never touches stored data —
  generated text only. Idempotent: a sentence already rewritten into a
  "[CAUTION: ...]" line is never re-scanned as a plan sentence.
- Dependencies: stdlib only, plus a leaf import of
  memory.fact_source._split_clauses (grammar-level clause splitting,
  shared with that module's own negation-scoping — no other coupling).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

# Leaf import (2026-09-06, B6/B7): clause splitting is grammar-level, shared
# with memory.fact_source's own negation-scoping. Explicitly allowed by the
# Phase B contract rather than duplicating the regex here.
from memory.fact_source import _split_clauses
from utils.date_coerce import coerce_date as _coerce_date

# --- Forward-looking plan cues (closed grammatical set) -----------------
_PLAN_CUE_RE = re.compile(
    r"\b(?:plan(?:s|ning)?\s+to|going\s+to|will\b|wants?\s+to|intends?\s+to|"
    r"scheduled|upcoming|pending|needs?\s+to)\b",
    re.IGNORECASE,
)
_HASNT_YET_RE = re.compile(r"\bhas(?:n'?t|\s+not)\b[^.?!\n]{0,60}?\byet\b", re.IGNORECASE)

# A recurring commitment is never "closed" by one completed occurrence
# (2026-09-06, B6): "he plans to go to the gym every Tuesday" stays a
# standing plan even after one Tuesday's gym visit is reported.
_HABITUAL_CUE_RE = re.compile(
    r"\bevery\b|\beach\s+(?:day|night|morning|evening|week|month|year|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|"
    r"\b(?:daily|nightly|weekly|monthly|yearly|biweekly|always|usually|"
    r"typically|routinely|regularly)\b|"
    r"\bmost\s+(?:days?|nights?|mornings?|evenings?|weeks?)\b",
    re.IGNORECASE,
)


def _is_plan_sentence(sentence: str) -> bool:
    if _HABITUAL_CUE_RE.search(sentence):
        return False
    return bool(_PLAN_CUE_RE.search(sentence) or _HASNT_YET_RE.search(sentence))


# --- Completion cues (past-tense/perfective) -----------------------------
_STRONG_COMPLETION_RE = re.compile(
    r"\b(?:did|done|finished|sent|went|saw|met|hung\s+out|hanging\s+out\s+with|made\s+it|"
    r"got\s+\w+(?:\s+\w+){0,3}\s+done|attended|completed|showed\s+up|caught\s+up|"
    r"paid|had\s+(?:a|the)\b)",
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


# --- Status guards (2026-09-06, B6) ---------------------------------------
# A clause explicitly reporting a DIFFERENT status than "completed" is never
# a match, even when a bare completion word appears elsewhere in the same
# message ("cancelled the repair" must not close a car-repair plan).
_STATUS_NOT_COMPLETED_RE = re.compile(
    r"\b(?:cancell?ed|rescheduled|moved|tried\s+to|attempted|couldn'?t|"
    r"could\s+not)\b",
    re.IGNORECASE,
)
# The completion cue in a clause must not be governed by a hedge/negation
# marker IN THAT SAME CLAUSE — "haven't gotten the car fixed yet" must never
# resolve a plan just because "gotten"/"fixed" appear.
_COMPLETION_NEGATION_RE = re.compile(
    r"\bnot\b|\bnever\b|n't\b|\bstill\s+(?:need|needs|have|has)\s+to\b|"
    r"\bif\b|\bwould\b|\bplanning\s+to\b",
    re.IGNORECASE,
)
# Document-sharing verbs whose object is a document noun never complete a
# NON-document plan (verified defect #3): "attached the exercise about
# completed tasks" must never close a real-world plan just because
# "completed" appears in the pasted document's own title.
_DOCUMENT_SHARE_RE = re.compile(
    r"\battached\b|\buploaded\b|\bhere'?s\s+the\s+file\b|\bsent\s+you\b",
    re.IGNORECASE,
)
_DOCUMENT_NOUN_RE = re.compile(
    r"\b(?:file|document|doc|pdf|exercise|worksheet|spreadsheet|attachment|"
    r"slides?|assignment)\b",
    re.IGNORECASE,
)
# A bare pronoun reference ("got it done") only resolves via the immediately
# previous same-day statement naming the plan's head.
_BARE_IT_RE = re.compile(r"\bit\b", re.IGNORECASE)


# --- Object-head identity (2026-09-06, B6/B7) -----------------------------
# A plan's IDENTITY is its object head noun, not generic token overlap with
# the whole sentence — "paid the electric bill" must not be resolved by
# "paid the water bill" just because both share "the"/"paid"/"bill" and a
# completion cue.
_PLAN_TEMPORAL_WORDS = frozenset({
    "today", "tomorrow", "tonight", "yesterday", "soon", "later",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "week", "weekend", "month", "year", "morning", "afternoon",
    "evening", "night",
})
# Cue verbs/particles/pronouns that INTRODUCE — but are not themselves — the
# plan's object. A closed grammatical set, generic across social/task plans
# (never wording tied to one incident).
_PLAN_CUE_WORDS = frozenset({
    "plan", "plans", "planning", "going", "go", "will", "want", "wants",
    "intend", "intends", "scheduled", "upcoming", "pending", "needs", "need",
    "has", "hasnt", "hadnt", "yet", "still", "he", "she", "they", "i", "we",
    "you",
})
_OBJECT_LEAD_VERBS = frozenset({
    "hang", "hanging", "hangout", "meet", "meeting", "see", "seeing",
    "visit", "visiting", "call", "calling", "text", "texting", "finish",
    "finishing", "schedule", "scheduling", "attend", "attending", "do",
    "doing", "take", "taking", "have", "having", "send", "sending", "fix",
    "fixing", "get", "getting", "pay", "paying", "book", "booking", "set",
    "setting", "handle", "handling", "close", "closing", "out", "up",
    "over", "back", "along", "catch", "catching",
})


def _plan_object_clause(plan_sentence: str) -> str:
    """The specific clause (of a possibly multi-clause narrative sentence)
    that actually carries the forward-looking plan cue — a preceding
    unrelated clause ("Sent weekend hangout invites; hanging out with a
    friend Saturday is pending.") must not leak its own words into the
    object-head extraction."""
    for clause in _split_clauses(plan_sentence or ""):
        if _PLAN_CUE_RE.search(clause) or _HASNT_YET_RE.search(clause):
            return clause
    return plan_sentence or ""


def _plan_object_head(plan_sentence: str) -> Tuple[str, str]:
    """(head, modifier): the plan's object head noun — the last content
    token of the plan's own clause once cue verbs/particles/pronouns,
    stopwords, and temporal/adverbial filler are stripped — and the content
    word immediately preceding it, if any ("electric" before "bill").
    Empty strings when nothing plan-specific survives — a plan never
    matches without an identifiable object (under-fires by design)."""
    clause = _plan_object_clause(plan_sentence)
    tokens = re.findall(r"[A-Za-z0-9]+", clause)
    content = [
        t for t in tokens
        if t.lower() not in _STOPWORDS
        and t.lower() not in _PLAN_CUE_WORDS
        and t.lower() not in _OBJECT_LEAD_VERBS
        and t.lower() not in _PLAN_TEMPORAL_WORDS
    ]
    if not content:
        return "", ""
    head = content[-1].lower()
    modifier = content[-2].lower() if len(content) >= 2 else ""
    return head, modifier


def _clause_matches_head(clause: str, head: str, modifier: str) -> bool:
    """Identity match: the bare head noun when the plan named no modifier,
    or — when it did (a compound object like "electric bill") — either the
    exact modifier+head phrase or a determiner-anchored bare reference
    ("that/the/my/this bill"). A bare unqualified head is deliberately NOT
    accepted when a modifier was named — too weak to distinguish "the water
    bill" from "the electric bill" (object-head mismatch)."""
    if not head:
        return False
    low = clause.lower()
    if modifier:
        if re.search(rf"\b{re.escape(modifier)}\s+{re.escape(head)}\b", low):
            return True
        return bool(re.search(rf"\b(?:that|the|my|this)\s+{re.escape(head)}\b", low))
    return bool(re.search(rf"\b{re.escape(head)}\b", low))


def _coerce_datetime(value: Any) -> Optional[datetime]:
    """Finer-grained sibling of ``_coerce_date`` — used only to order
    same-day statements chronologically (the "immediately previous
    statement" lookback for a bare "it" reference needs real ordering, not
    just a shared date)."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
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
            if _HABITUAL_CUE_RE.search(sentence):
                # A recurring commitment is never "closed" (B6) — skip so
                # plan_sentences() never reports it as an open forward-
                # looking plan in the first place.
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
    """The latest-dated user statement whose CLAUSE reports the plan's
    object done, matched by IDENTITY (object head noun) rather than generic
    token overlap (2026-09-06, B6/B7).

    A clause qualifies when it:
      (a) belongs to a statement dated on or after ``as_of`` when a floor is
          given (the caller passes its own lookback horizon — statements
          are COUNT-windowed, so "today" was the wrong floor: a plan
          reported done yesterday stayed pending);
      (b) carries a past-tense/completion cue NOT governed by a hedge/
          negation marker in that SAME clause ("haven't ... yet", "still
          need to", "if", "would", "planning to");
      (c) does not report a DIFFERENT status ("cancelled", "rescheduled",
          "moved", "tried to", "attempted", "couldn't") — a distinct
          status, never a completion match;
      (d) names the plan's object HEAD — either directly, via a
          determiner-anchored reference ("that/the/my/this <head>"), or via
          a bare "it" ONLY when the immediately previous user statement on
          the same day names the head (memory.fact_source._split_clauses
          isolates the qualifying clause within a longer statement);
      (e) is not a document-sharing verb ("attached"/"uploaded"/"sent
          you"/"here's the file") whose object is a document noun, unless
          the PLAN itself is about a document;
      (f) has no digit-token conflict with the plan's head/modifier
          (numbered items never cross-resolve, e.g. HW6 vs hw7).

    Newest statement wins: a LATER user statement that re-asserts the plan
    (a forward-looking cue over the same object) cancels the completion —
    the user made the plan again. Returns a shallow copy of the matching
    corpus-shaped statement dict with the qualifying clause text under
    ``"_matched_clause"``, or ``None``.
    """
    if not isinstance(plan_sentence, str) or not plan_sentence.strip():
        return None
    head, modifier = _plan_object_head(plan_sentence)
    if not head:
        return None
    plan_digits = _digit_tokens({t for t in (head, modifier) if t})
    is_document_plan = bool(_DOCUMENT_NOUN_RE.search(plan_sentence))
    floor = _coerce_date(as_of) if as_of is not None else None

    # Chronological ordering by day, then by real timestamp when available
    # (falls back to input order) — needed for both "latest wins" and the
    # bare-"it" same-day lookback.
    entries: List[Tuple[date, datetime, Dict[str, Any], str]] = []
    for idx, item in enumerate(user_statements or []):
        if not isinstance(item, dict):
            continue
        text = _statement_text(item)
        if not text.strip():
            continue
        raw_ts = item.get("timestamp") or item.get("turn_id")
        day = _coerce_date(raw_ts)
        if day is None:
            continue
        order_dt = _coerce_datetime(raw_ts) or datetime.min.replace(microsecond=idx % 1000000)
        entries.append((day, order_dt, item, text))
    entries.sort(key=lambda e: (e[0], e[1]))

    candidates: List[Tuple[date, datetime, Dict[str, Any], str]] = []
    replans: List[date] = []
    for pos, (day, order_dt, item, text) in enumerate(entries):
        if floor is not None and day < floor:
            continue
        matched_clause: Optional[str] = None
        for clause in (_split_clauses(text) or [text]):
            if _STATUS_NOT_COMPLETED_RE.search(clause):
                continue
            if _COMPLETION_NEGATION_RE.search(clause):
                continue
            if not _has_completion_cue(clause):
                continue
            clause_digits = _digit_tokens(_norm_tokens(clause))
            if plan_digits and clause_digits and not (plan_digits & clause_digits):
                continue
            if (_DOCUMENT_SHARE_RE.search(clause) and _DOCUMENT_NOUN_RE.search(clause)
                    and not is_document_plan):
                continue
            if _clause_matches_head(clause, head, modifier):
                matched_clause = clause
                break
            if _BARE_IT_RE.search(clause) and pos > 0:
                prev_day, _prev_dt, _prev_item, prev_text = entries[pos - 1]
                if prev_day == day and _clause_matches_head(prev_text, head, modifier):
                    matched_clause = clause
                    break
        if matched_clause is not None:
            candidates.append((day, order_dt, item, matched_clause))
        elif plan_sentences(text):
            replans.append(day)

    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1]))
    latest_day, _latest_dt, latest_item, latest_clause = candidates[-1]
    if any(day > latest_day for day in replans):
        return None
    result = dict(latest_item)
    result["_matched_clause"] = " ".join(latest_clause.split())
    return result


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
        # B7 (2026-09-06): quote the exact qualifying CLAUSE that resolved
        # the plan, not the statement's first 160 chars (a long statement's
        # unrelated opening text used to be what the caution line quoted).
        matched_clause = claim.statement.get("_matched_clause")
        stmt_text = " ".join((matched_clause or _statement_text(claim.statement)).split())
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
