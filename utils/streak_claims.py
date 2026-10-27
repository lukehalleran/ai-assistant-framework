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
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils.date_coerce import coerce_date as _coerce_date

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


# ======================================================================
# Dated day counts (2026-10-27, class: BC-103 supply / BC-100 / BC-75 / BC-52)
# ----------------------------------------------------------------------
# "23 days into <activity>", "day 23 of <activity>", a "(Day 23)" note title:
# a count that was true on the day it was written. 2026-09-11..20: a note's
# "Day 23" was read as TODAY's count and echoed forward for nine days because
# no input ever carried the start date. Every count becomes a START DATE
# (stated_on - (N - 1)); the elapsed arithmetic is done here, once, and the
# result is handed to the model (see memory/day_count_anchors.py for the store
# and the prompt supply). Extraction is structural: a number, a day unit, a
# relation word from the closed class of time-span prepositions, and the noun
# phrase it governs. No activity vocabulary is listed anywhere.
# Sources are the USER's own text only; the caller never passes assistant text.
# ======================================================================

# Closed-class relation words that link a day count to the activity it counts
# (function words, not topic words).
_DAY_COUNT_RELATIONS = ("into", "of", "since", "on", "off", "without", "after")
_REL_ALT = "|".join(_DAY_COUNT_RELATIONS)
# Noun phrase after the relation: up to 5 words, stops at clause punctuation.
_NP = r"(?P<np>[^\W\d_][\w'’\-]*(?:\s+[\w'’\-]+){0,4})"
_DAY_COUNT_A_RE = re.compile(
    r"\b" + _NUM + r"(?:\s*-\s*|\s+)days?\s+(?P<rel>" + _REL_ALT + r")\s+" + _NP,
    re.IGNORECASE,
)
_DAY_COUNT_B_RE = re.compile(
    r"\bday\s+" + _NUM + r"\s+(?P<rel>" + _REL_ALT + r")\s+" + _NP,
    re.IGNORECASE,
)
_BARE_DAY_RE = re.compile(r"\bday\s+" + _NUM + r"\b(?!\s*(?:/|of\s+\d))", re.IGNORECASE)
# Function words that make "day N" a plan/hypothetical/other-time reference.
_IRREALIS_LEAD = frozenset(
    {"if", "when", "by", "until", "till", "before", "once", "unless", "around", "maybe", "perhaps", "hopefully"}
)
_INTERROGATIVE_LEAD = frozenset(
    {"is", "are", "was", "were", "do", "does", "did", "what", "which", "when", "how", "why", "who", "can", "should"}
)
# Words that end a noun phrase (function words: determiners, conjunctions, auxiliaries, pronouns).
_NP_STOP = frozenset(
    {"and", "but", "so", "because", "which", "that", "where", "while", "though", "although", "i", "i'm", "im",
     "we", "he", "she", "it", "they", "is", "are", "was", "were", "has", "have", "had", "now", "today",
     "yesterday", "then", "or", "with", "to", "for", "in", "at", "by", "from"}
)
_NP_LEAD_SKIP = frozenset({"the", "a", "an", "my", "our", "this", "that", "these", "those", "your", "his", "her", "their"})


def _lemma_token(word: str) -> str:
    """Crude singular/gerund stem used only when no spaCy pipeline is supplied."""
    w = re.sub(r"[^\w\-]", "", word.lower())
    for suffix in ("ing", "es", "s", "e"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def subject_lemmas(np_text: str, nlp: Any = None) -> frozenset:
    """Content lemmas of an activity noun phrase (determiners/function words dropped).
    With a spaCy pipeline: lemmas of NOUN/PROPN/VERB/ADJ tokens; without: stemmed words."""
    text = (np_text or "").strip()
    if not text:
        return frozenset()
    if nlp is not None:
        try:
            doc = nlp(text)
            lem = {_lemma_token(t.lemma_) for t in doc if t.pos_ in ("NOUN", "PROPN", "VERB", "ADJ") and not t.is_stop}
            if lem:
                return frozenset(lem)
        except Exception:  # degrades: falls back to the stem heuristic
            pass
    words = [w for w in re.split(r"[\s\-]+", text) if w]
    out = {_lemma_token(w) for w in words if w.lower() not in _NP_LEAD_SKIP and w.lower() not in _NP_STOP}
    return frozenset(x for x in out if len(x) > 1)


def _clean_np(raw: str) -> str:
    words: List[str] = []
    for w in raw.split():
        if w.lower().strip(",;:") in _NP_STOP and words:
            break
        words.append(w.strip(",;:"))
    while words and words[0].lower() in _NP_LEAD_SKIP:
        words.pop(0)
    return " ".join(words)


@dataclass
class DayCountClaim:
    count: int
    stated_on: date
    subject: str            # activity noun phrase as written ("" = unattributed)
    lemmas: frozenset       # content lemmas of ``subject``
    snippet: str
    source: str = "message"  # "message" (turn) | "note" (user's own dated note)

    @property
    def start(self) -> date:
        return self.stated_on - timedelta(days=self.count - 1)

    def projected(self, as_of: Any) -> int:
        day = _coerce_date(as_of) or self.stated_on
        return self.count + max(0, (day - self.stated_on).days)

    def to_dict(self) -> Dict[str, Any]:
        return {"count": self.count, "stated_on": self.stated_on.isoformat(), "subject": self.subject,
                "lemmas": sorted(self.lemmas), "snippet": self.snippet, "source": self.source}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Optional["DayCountClaim"]:
        try:
            stated = _coerce_date(d.get("stated_on"))
            n = int(d["count"])
            if stated is None or not 0 < n <= 400:
                return None
            return cls(count=n, stated_on=stated, subject=str(d.get("subject") or ""),
                       lemmas=frozenset(str(x) for x in d.get("lemmas") or []),
                       snippet=str(d.get("snippet") or ""), source=str(d.get("source") or "message"))
        except (KeyError, TypeError, ValueError):
            return None


def _sentence_prefix_words(text: str, pos: int) -> List[str]:
    left = max(text.rfind(m, 0, pos) for m in ("\n", ".", "?", "!")) + 1
    return re.findall(r"[a-z']+", text[left:pos].lower())


def _sentence_lead(text: str, pos: int) -> List[str]:
    left = max(text.rfind(m, 0, pos) for m in ("\n", ".", "?", "!")) + 1
    return re.findall(r"[a-z']+", text[left:pos].lower())[:1]


def extract_day_count_claims(
    text: str, stated_on: Any, *, source: str = "message", allow_bare: bool = True, nlp: Any = None,
) -> List[DayCountClaim]:
    """Day-count claims in ONE user-authored text, dated to the day it was written.

    Attributed: "<N> days <rel> <activity>" / "day <N> <rel> <activity>". Unattributed:
    a bare "day <N>" (message: any user sentence that is not a plan/hypothetical/
    question; note: ``allow_bare=True`` for titles/headings only). Rejected: quoted or
    drafted text, irrealis lead words (if/when/by/until…), a question that opens with an
    interrogative/auxiliary, N outside 1..400. Never raises."""
    if not isinstance(text, str) or not text.strip():
        return []
    day = _coerce_date(stated_on)
    if day is None:
        return []
    out: List[DayCountClaim] = []
    seen = set()
    attributed_spans: List[Tuple[int, int]] = []

    def _ok_context(m: re.Match) -> bool:
        snippet = _sentence_window(text, m.start(), m.end())
        if _QUOTED_OR_DRAFT_RE.search(snippet):
            return False
        if any(w in _IRREALIS_LEAD for w in _sentence_prefix_words(text, m.start())):
            return False
        lead = _sentence_lead(text, m.start())
        if snippet.rstrip().endswith("?") and lead and lead[0] in _INTERROGATIVE_LEAD:
            return False
        return True

    def _add(n_tok: str, m: re.Match, np_raw: str) -> None:
        n = _to_int(n_tok)
        if n is None:
            return
        snippet = _sentence_window(text, m.start(), m.end())
        subject = _clean_np(np_raw)
        lem = subject_lemmas(subject, nlp)
        if not lem:
            return
        key = (n, tuple(sorted(lem)))
        if key in seen:
            return
        seen.add(key)
        out.append(DayCountClaim(count=n, stated_on=day, subject=subject, lemmas=lem, snippet=snippet, source=source))

    for rx in (_DAY_COUNT_A_RE, _DAY_COUNT_B_RE):
        for m in rx.finditer(text):
            if _ok_context(m):
                attributed_spans.append((m.start(), m.end()))
                _add(m.group("n"), m, m.group("np"))
    if allow_bare:
        for m in _BARE_DAY_RE.finditer(text):
            if any(s <= m.start() < e for s, e in attributed_spans):
                continue
            n = _to_int(m.group("n"))
            if n is None or not _ok_context(m):
                continue
            key = (n, ())
            if key in seen:
                continue
            seen.add(key)
            out.append(DayCountClaim(count=n, stated_on=day, subject="", lemmas=frozenset(),
                                     snippet=_sentence_window(text, m.start(), m.end()), source=source))
    return out


@dataclass
class EpisodeAnchor:
    lemmas: frozenset
    subject: str
    start: date
    newest: DayCountClaim
    claims: List[DayCountClaim]

    @property
    def spread_days(self) -> int:
        starts = [c.start for c in self.claims]
        return (max(starts) - min(starts)).days

    def projected(self, as_of: Any) -> int:
        return self.newest.projected(as_of)


START_AGREEMENT_DAYS = 3


def _claim_rank(c: DayCountClaim) -> Tuple[date, int]:
    # Newest statement wins; on the same day the user's message outranks a (retrospective) note.
    return (c.stated_on, 1 if c.source == "message" else 0)


def episode_anchors(claims: Iterable[DayCountClaim]) -> List[EpisodeAnchor]:
    """Merge claims into episodes. Attributed claims group by shared activity lemma; an
    unattributed claim joins the episode whose start date it agrees with (within
    START_AGREEMENT_DAYS) and is otherwise dropped (never guessed onto a subject)."""
    groups: List[List[DayCountClaim]] = []
    attributed = [c for c in claims if c.lemmas]
    for c in sorted(attributed, key=_claim_rank):
        for g in groups:
            if any(c.lemmas & m.lemmas for m in g):
                g.append(c)
                break
        else:
            groups.append([c])
    # merge groups that became linked through a later claim
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if any(a.lemmas & b.lemmas for a in groups[i] for b in groups[j]):
                    groups[i] += groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    for c in sorted((c for c in claims if not c.lemmas), key=_claim_rank):
        best, gap = None, None
        for g in groups:
            d = abs((c.start - max(g, key=_claim_rank).start).days)
            if d <= START_AGREEMENT_DAYS and (gap is None or d < gap):
                best, gap = g, d
        if best is not None:
            best.append(c)
    out: List[EpisodeAnchor] = []
    for g in groups:
        newest = max(g, key=_claim_rank)
        lem = frozenset().union(*(m.lemmas for m in g))
        out.append(EpisodeAnchor(lemmas=lem, subject=newest.subject or next((m.subject for m in g if m.subject), ""),
                                 start=newest.start, newest=newest, claims=sorted(g, key=_claim_rank)))
    out.sort(key=lambda a: a.newest.stated_on, reverse=True)
    return out


def _tokens_lemma_set(text: str, nlp: Any = None) -> frozenset:
    return subject_lemmas(text, nlp) if text else frozenset()


def day_count_lines(
    anchors: Iterable[EpisodeAnchor], as_of: Any, relevant_text: str = "", *,
    max_lines: int = 3, max_age_days: int = 400, nlp: Any = None,
) -> List[str]:
    """Prompt lines "<activity>: day N today (<date>) — from …". An anchor is supplied only when
    one of its activity lemmas occurs in ``relevant_text`` (no always-on block)."""
    today = _coerce_date(as_of) or date.today()
    if not relevant_text or max_lines <= 0:
        return []
    rel = _tokens_lemma_set(relevant_text, nlp) | frozenset(_lemma_token(w) for w in re.findall(r"[\w'\-]+", relevant_text))
    lines: List[str] = []
    for a in anchors:
        if (today - a.newest.stated_on).days > max_age_days or a.newest.stated_on > today:
            continue
        if not (a.lemmas & rel):
            continue
        n = a.projected(today)
        c = a.newest
        what = "your note" if c.source == "note" else "your message"
        line = (f'- {a.subject or "the episode"}: day {n} today ({today.isoformat()}) — from {what} dated '
                f'{c.stated_on.isoformat()} ("{c.snippet[:110]}"; day 1 ≈ {a.start.isoformat()})')
        if a.spread_days > START_AGREEMENT_DAYS - 1:
            first = a.claims[0]
            line += (f" (your statements differ by {a.spread_days} days: {first.stated_on.isoformat()} "
                     f"day {first.count} vs {c.stated_on.isoformat()} day {c.count})")
        lines.append(line)
        if len(lines) >= max_lines:
            break
    return lines


@dataclass
class StaleDayCount:
    claim_text: str
    stated_count: int
    current_count: int
    start: int
    end: int


def stale_day_count_claims(
    text: str, anchors: Iterable[EpisodeAnchor], as_of: Any, *, nlp: Any = None,
) -> List[StaleDayCount]:
    """Sentences of GENERATED text asserting "N days into <activity>" / "day N of <activity>" whose N
    differs from the anchor's count projected to ``as_of``, unless the sentence dates the count to the
    day it was stated. Pure; wired by nobody yet (verify-side hand-off for figure_check / handlers)."""
    if not text:
        return []
    today = _coerce_date(as_of) or date.today()
    anchors = list(anchors)
    found: List[StaleDayCount] = []
    seen = set()
    for rx in (_DAY_COUNT_A_RE, _DAY_COUNT_B_RE):
        for m in rx.finditer(text):
            n = _to_int(m.group("n"))
            lem = subject_lemmas(_clean_np(m.group("np")), nlp)
            if n is None or not lem:
                continue
            for a in anchors:
                if not (a.lemmas & lem):
                    continue
                current = a.projected(today)
                if n == current:
                    continue
                left = max(text.rfind(mk, 0, m.start()) for mk in ("\n", ".", "?", "!")) + 1
                rights = [p for mk in ("\n", ".", "?", "!") if (p := text.find(mk, m.end())) >= 0]
                right = min(rights) + 1 if rights else len(text)
                sentence = text[left:right]
                said = a.newest.stated_on
                if re.search(
                    rf"\b(?:{said.strftime('%B')}|{said.strftime('%b')}\.?)\s+{said.day}\b|\b{said.isoformat()}\b|"
                    rf"\b{said.month}/{said.day}\b", sentence, re.IGNORECASE):
                    continue
                if (left, right) in seen:
                    continue
                seen.add((left, right))
                found.append(StaleDayCount(" ".join(sentence.split()), n, current, left, right))
                break
    found.sort(key=lambda s: s.start)
    return found
