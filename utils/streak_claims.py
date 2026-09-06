# utils/streak_claims.py
"""
Module Contract
- Purpose: deterministic handling of the user's own STREAK / DAY-COUNT
  statements ("today is day 6 in a row that I'm normal", "8 days straight")
  across the daily-note and narrative synthesis paths. 2026-09-05 live
  defect: the user said "day 6" on Sep 2; the Sep 3 daily note copied
  Daemon's "six days" reply, the Sep 5 narrative ([TEMPORAL GROUNDING])
  restated "six consecutive days (August 31–September 5)" — an invented
  range — and the reply told the user "six days into a stable streak" on
  what the user then corrected to day 8. A count is only true on the day it
  was said; every later day it must be projected or re-dated.
- Inputs:
  - extract_streak_claims(text, stated_on) -> List[StreakClaim]
  - streak_ledger(statements, as_of, max_age_days) -> List[StreakClaim]
  - projected_count(claim, as_of) -> int
  - streak_ledger_block(claims, as_of) -> str        (prompt-injection block)
  - remove_stale_streak_claims(text, claims, as_of) -> (revised, removed)
- Behavior: purely mechanical. Extraction requires a streak CUE next to the
  number ("in a row", "straight", "consecutive", "streak", "day N of/that…")
  so ordinary "6 days" durations are left to build_temporal_claim_audit.
  Projection is arithmetic only (count + calendar days elapsed); the NEWEST
  user statement always wins over an older projection. The post-generation
  check removes a generated sentence that asserts a streak count different
  from the ledger's current count unless the sentence dates the count to
  the day it was said. Never touches stored data — generated text only.
- Dependencies: stdlib only (leaf module). Callers supply the user-authored
  statements (corpus `user_text`/`query` + timestamp); assistant text is
  never a source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30,
}
_NUM = r"(?P<n>\d{1,3}|" + "|".join(_NUMBER_WORDS) + r")"
_STREAK_TAIL = (
    r"(?:in\s+a\s+row|straight|consecutive(?:ly)?|running|streak|"
    r"(?:of|that|where)\s+(?:being|feeling|i\s+am|i'm|i’m|i\s+feel|i\s+have\s+been|i've\s+been|i’ve\s+been))"
)

# "day 6 in a row", "day 8 of feeling normal", "day 6 that I am 'normal'"
_DAY_N_RE = re.compile(
    r"\bday\s+" + _NUM + r"(?:\s*(?:st|nd|rd|th))?\s+(?:[^\w\n]{0,3}\w+\s+){0,2}?" + _STREAK_TAIL,
    re.IGNORECASE,
)
# "6 days in a row", "8 days straight", "six consecutive days", "8-day streak"
_N_DAYS_RE = re.compile(
    r"\b" + _NUM + r"(?:\s*-\s*|\s+)(?:consecutive\s+|straight\s+)?days?\b"
    r"(?:\s+(?:[^\w\n]{0,3}\w+\s+){0,2}?" + _STREAK_TAIL + r")?",
    re.IGNORECASE,
)
_N_DAYS_STRICT_CUE_RE = re.compile(r"\b(?:consecutive|straight|in\s+a\s+row|streak|running)\b", re.IGNORECASE)
# "the Nth day in a row", "my 8th straight day"
_NTH_DAY_RE = re.compile(
    r"\b" + _NUM + r"(?:st|nd|rd|th)\s+(?:consecutive\s+|straight\s+)?day\b"
    r"(?:\s+(?:[^\w\n]{0,3}\w+\s+){0,2}?" + _STREAK_TAIL + r")?",
    re.IGNORECASE,
)
# "today is day 8", "day 8 today", "this is day 8", "makes it day 8" — a bare
# self-count with a today-anchor is a streak statement even without a tail.
_TODAY_IS_DAY_RE = re.compile(
    r"\b(?:today\s+is|today's|today’s|this\s+is|that\s+makes\s+it|makes\s+it|it'?s|it’s)\s+day\s+" + _NUM + r"\b",
    re.IGNORECASE,
)
_DAY_N_TODAY_RE = re.compile(r"\bday\s+" + _NUM + r"\s+today\b", re.IGNORECASE)
# A count quoted about someone else / a draft is not the user's own count.
_QUOTED_OR_DRAFT_RE = re.compile(
    r"\b(?:draft|subject:|dear\b|she\s+said|he\s+said|they\s+said|according\s+to)\b",
    re.IGNORECASE,
)
# An ACTIVITY streak ("worked out 3 days in a row") is a separate count from
# the user's state streak ("day 8 of being productive") — the live corpus
# carried both on the same day. Activity claims are listed in the ledger but
# never drive the current count or the stale-sentence check.
_ACTIVITY_CUE_RE = re.compile(
    r"\b(?:work(?:ed|ing)?\s*out|workouts?|gym|exercis\w*|lift(?:ed|ing|s)?|cardio|ran|run(?:ning)?|"
    r"walk(?:ed|ing|s)?|stud(?:y|ied|ying)|lecture\w*|coding|coded|commits?|meditat\w*|journal\w*|"
    r"stretch\w*|yoga|swim\w*|bike\w*|cycling|practice\w*|showered|cooked)\b",
    re.IGNORECASE,
)
_DATE_TOKEN_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
    re.IGNORECASE,
)


@dataclass
class StreakClaim:
    count: int
    stated_on: date
    snippet: str
    source: str = "user"
    kind: str = "state"  # "state" (functional/normal/productive …) or "activity" (workouts …)

    def projected(self, as_of: date) -> int:
        return projected_count(self, as_of)


def _to_int(token: str) -> Optional[int]:
    t = (token or "").strip().lower()
    if t.isdigit():
        v = int(t)
        return v if 0 < v <= 400 else None
    return _NUMBER_WORDS.get(t)


def _sentence_window(text: str, start: int, end: int, max_chars: int = 200) -> str:
    left = max(text.rfind(m, 0, start) for m in ("\n", ".", "?", "!")) + 1
    rights = [p for m in ("\n", ".", "?", "!") if (p := text.find(m, end)) >= 0]
    right = min(rights) + 1 if rights else len(text)
    snippet = " ".join(text[left:right].split())
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "…"
    return snippet


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


def extract_streak_claims(text: str, stated_on: Any) -> List[StreakClaim]:
    """Streak/day-count statements in ONE user-authored text, dated to the day
    it was written. Requires a streak cue; plain durations ("for 6 days") are
    not streak counts. Returns claims in order of appearance, de-duplicated by
    (count, snippet)."""
    if not isinstance(text, str) or not text.strip():
        return []
    day = _coerce_date(stated_on)
    if day is None:
        return []
    out: List[StreakClaim] = []
    seen = set()

    def _add(m: re.Match, require_cue: bool = False) -> None:
        n = _to_int(m.group("n"))
        if n is None:
            return
        if require_cue and not _N_DAYS_STRICT_CUE_RE.search(m.group(0)):
            return
        snippet = _sentence_window(text, m.start(), m.end())
        if _QUOTED_OR_DRAFT_RE.search(snippet):
            return
        key = (n, snippet.casefold())
        if key in seen:
            return
        seen.add(key)
        kind = "activity" if _ACTIVITY_CUE_RE.search(snippet) else "state"
        out.append(StreakClaim(count=n, stated_on=day, snippet=snippet, kind=kind))

    for m in _DAY_N_RE.finditer(text):
        _add(m)
    for m in _TODAY_IS_DAY_RE.finditer(text):
        _add(m)
    for m in _DAY_N_TODAY_RE.finditer(text):
        _add(m)
    for m in _NTH_DAY_RE.finditer(text):
        _add(m, require_cue=True)
    for m in _N_DAYS_RE.finditer(text):
        _add(m, require_cue=True)
    return out


def projected_count(claim: StreakClaim, as_of: Any) -> int:
    """count + calendar days elapsed since it was stated (never negative)."""
    day = _coerce_date(as_of) or claim.stated_on
    return claim.count + max(0, (day - claim.stated_on).days)


def streak_ledger(
    statements: Iterable[Any],
    as_of: Any = None,
    max_age_days: int = 14,
) -> List[StreakClaim]:
    """Collect the user's streak claims from corpus-shaped statements.

    Each statement is a dict with a timestamp and ``user_text``/``query``
    (assistant ``response`` is ignored — generated text is not a source), or
    a ``(text, timestamp)`` tuple. Claims older than ``max_age_days`` before
    ``as_of`` are dropped. Sorted oldest → newest.
    """
    today = _coerce_date(as_of) or date.today()
    claims: List[StreakClaim] = []
    for item in statements or []:
        if isinstance(item, dict):
            text = item.get("user_text") or item.get("query") or item.get("user") or ""
            ts = item.get("timestamp") or item.get("turn_id")
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            text, ts = item
        else:
            continue
        day = _coerce_date(ts)
        if day is None or (today - day).days > max_age_days or day > today:
            continue
        claims.extend(extract_streak_claims(str(text), day))
    claims.sort(key=lambda c: c.stated_on)
    return claims


def current_streak_count(claims: List[StreakClaim], as_of: Any) -> Optional[Tuple[int, StreakClaim]]:
    """The newest STATE statement projected to ``as_of`` — the newest wins.
    Activity streaks never drive the current count."""
    state = [c for c in claims if c.kind == "state"]
    if not state:
        return None
    newest = max(state, key=lambda c: c.stated_on)
    return projected_count(newest, as_of), newest


def streak_ledger_block(claims: List[StreakClaim], as_of: Any) -> str:
    """Prompt block: every user count with its date, the projection to
    ``as_of``, and the rule. "" when there are no claims."""
    if not claims:
        return ""
    today = _coerce_date(as_of) or date.today()
    lines = []
    for c in claims:
        if c.kind != "state":
            lines.append(
                f'- {c.stated_on.isoformat()}: activity streak (separate count, not the state streak) — '
                f'the user counted {c.count} — "{c.snippet}"'
            )
            continue
        proj = projected_count(c, today)
        if c.stated_on == today:
            lines.append(f'- {c.stated_on.isoformat()} (today): the user counted day {c.count} — "{c.snippet}"')
        else:
            lines.append(
                f'- {c.stated_on.isoformat()}: the user counted day {c.count} — "{c.snippet}" '
                f"→ if unbroken, {today.isoformat()} is day {proj}"
            )
    cur = current_streak_count(claims, today)
    tail = ""
    if cur:
        count, newest = cur
        tail = (
            f"\nCURRENT COUNT as of {today.isoformat()}: day {count} "
            f"(from the user's newest statement on {newest.stated_on.isoformat()})."
        )
    return (
        "STREAK LEDGER (the user's OWN day counts; a count is only true on the day it was said):\n"
        + "\n".join(lines)
        + tail
        + "\nRule: never restate an older count as the current one — use the CURRENT COUNT, "
        "or give a count together with the date it was said. A newer user statement "
        "overrides an older projection. Never invent a start date or a date range for a streak."
    )


# --- Post-generation check on generated text ---------------------------
_GEN_N_DAYS_RE = re.compile(
    r"\b" + _NUM + r"(?:\s*-\s*|\s+)(?:consecutive|straight|stable|solid|functional|good|normal|clean|sober)?\s*"
    r"days?\b(?:\s+(?:into|of|in\s+a\s+row|straight|streak))?",
    re.IGNORECASE,
)
_GEN_DAY_N_RE = re.compile(r"\bday\s+" + _NUM + r"\b", re.IGNORECASE)
_GEN_STREAK_CUE_RE = re.compile(
    r"\b(?:streak|in\s+a\s+row|consecutive|straight|stable|functional|normal|solid|"
    r"sober|clean|functioning|day\s+\d)\b",
    re.IGNORECASE,
)


@dataclass
class StaleStreakClaim:
    claim_text: str
    stated_count: int
    current_count: int
    start: int
    end: int


def stale_streak_claims(text: str, claims: List[StreakClaim], as_of: Any) -> List[StaleStreakClaim]:
    """Sentences in generated text asserting a streak count that is not the
    ledger's current count. A sentence that dates its count to the day the
    user said it (contains that date) is not stale — it is correctly
    historical. No ledger → nothing is ever flagged."""
    if not text or not claims:
        return []
    today = _coerce_date(as_of) or date.today()
    cur = current_streak_count(claims, today)
    if cur is None:
        return []
    current, _ = cur
    counts_with_dates = {c.count: c.stated_on for c in claims if c.kind == "state"}
    stale: List[StaleStreakClaim] = []
    seen = set()
    for pattern in (_GEN_N_DAYS_RE, _GEN_DAY_N_RE):
        for m in pattern.finditer(text):
            n = _to_int(m.group("n"))
            if n is None or n == current:
                continue
            left = max(text.rfind(mk, 0, m.start()) for mk in ("\n", ".", "?", "!")) + 1
            rights = [p for mk in ("\n", ".", "?", "!") if (p := text.find(mk, m.end())) >= 0]
            right = min(rights) + 1 if rights else len(text)
            sentence = text[left:right]
            if (left, right) in seen:
                continue
            if not _GEN_STREAK_CUE_RE.search(sentence):
                continue  # an ordinary duration ("sick for 3 days"), not a streak count
            if _ACTIVITY_CUE_RE.search(sentence):
                continue  # an activity streak ("three workouts in a row") is a separate count
            # Historical mention dated to the day it was said → keep.
            said_on = counts_with_dates.get(n)
            if said_on is not None:
                month = said_on.strftime("%B")
                abbr = said_on.strftime("%b")
                if re.search(
                    rf"\b(?:{month}|{abbr}\.?)\s+{said_on.day}\b|\b{said_on.isoformat()}\b|"
                    rf"\b{said_on.month}/{said_on.day}\b",
                    sentence,
                    re.IGNORECASE,
                ):
                    continue
            seen.add((left, right))
            stale.append(
                StaleStreakClaim(
                    claim_text=" ".join(sentence.split()),
                    stated_count=n,
                    current_count=current,
                    start=left,
                    end=right,
                )
            )
    stale.sort(key=lambda s: s.start)
    return stale


def remove_stale_streak_claims(
    text: str, claims: List[StreakClaim], as_of: Any
) -> Tuple[str, List[StaleStreakClaim]]:
    """Excise stale streak-count sentences and append one caution line per
    removal (mirrors utils.status_claims.remove_conflicting_claims). Operates
    on the generated string only."""
    stale = stale_streak_claims(text, claims, as_of)
    if not stale:
        return text, []
    today = _coerce_date(as_of) or date.today()
    revised = text
    for s in sorted(stale, key=lambda s: s.start, reverse=True):
        revised = revised[:s.start] + revised[s.end:]
    revised = re.sub(r"[ \t]{2,}", " ", revised)
    revised = re.sub(r"\n[ \t]+", "\n", revised)
    revised = re.sub(r"\n{3,}", "\n\n", revised).strip()
    cur = current_streak_count(claims, today)
    newest_note = ""
    if cur:
        _, newest = cur
        newest_note = f" (user said day {newest.count} on {newest.stated_on.isoformat()})"
    caution = [
        f'[CAUTION: a draft sentence claiming "{s.claim_text}" was removed — '
        f"the user's own count as of {today.isoformat()} is day {s.current_count}{newest_note}]"
        for s in stale
    ]
    revised = (revised.rstrip() + "\n\n" if revised else "") + "\n".join(caution)
    return revised, stale
