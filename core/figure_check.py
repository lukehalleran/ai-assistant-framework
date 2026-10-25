"""Module Contract -- core/figure_check.py

Purpose:
    Every number or time the reply asserts about the user, the conversation, the
    user's data or Daemon itself must be COPIED from a source in context, DERIVED
    by a fixed deterministic operation, or it is UNSUPPORTED (2026-10-25, lane
    FIGB; incident 16:07 -- "how wrecked you were feeling an hour ago" said one
    minute after the reply that used the word; class: BC-103 the model is left to
    compute a quantity the code already knows, BC-46, BC-58).
    World facts ("withdrawal resolves within a week") are OUT OF SCOPE: they belong
    to the grounding check. Figures copied from tool/web results are COPIED (the
    source pool includes them).

Pure: no model call, no I/O, no network. spaCy (the codebase's loaded
    ``en_core_web_sm``) is used for clause parsing and lemmas; without it the
    check falls back to stop-word + crude-stem binding and number scope checks
    degrade (receipt ``nlp`` False).

Public surface:
    extract_figures(reply)                      -> list[Figure]
    build_source_pool(raw_context, history, source_material, now) -> list[SourceItem]
    check_reply(reply, pool, now, ...)          -> FigureReport
    apply_corrections(reply, report, correct_classes) -> (revised, n_corrected)

DERIVED operators (the complete, fixed set): elapsed(now - item.ts);
    elapsed(item_a.ts - item_b.ts); weekday(date); sum / difference of two pool
    values with the same unit; count of items in one rendered list section; unit
    conversion within a dimension (mg/g, second..week).

Elapsed expression -> interval table (age of the referent item, ``_interval``):
    just now / a minute ago ........ [0, 5 min]
    a few minutes ago .............. [1, 20 min]
    N minutes ago .................. [0.5N, 1.5N + 2 min]
    an hour ago .................... [30 min, 1 h 45 min]
    N hours ago .................... [N - 0.75 h, N + 0.75 h]
    yesterday ...................... the previous calendar day
    N days ago ..................... N calendar days +- 1
    earlier today .................. the same calendar day
    this morning / afternoon / evening  04-12 / 12-18 / 18-24 the same day
    last night ..................... 18:00 yesterday to 04:00 today
    a few hours / days, N weeks / months, last week: coarse bands (vague class)
An elapsed expression is CONSISTENT if ANY referent's timestamp falls inside it.
    ``apply_corrections`` edits ONLY precise elapsed phrases that EVERY referent
    contradicts when the referents agree with each other, and impossible
    weekday/date pairs. Nothing else is ever edited.

Classes (FigureFinding.cls): copied, derived:<op>, unsupported, elapsed_unconfirmed, copied_mismatch,
    out_of_scope, unbound (no referent / no timing source), elapsed_precise,
    elapsed_vague, weekday_date (the last three = a contradicted time).
    An ambiguous or weakly bound precise elapsed is class ``elapsed_unconfirmed``
    with reason ``ambiguous_referents`` (logged, not edited).
"""
from __future__ import annotations

import calendar
import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Optional, Sequence

from core.grounding_check import weekday_date_mismatches
from memory.claim_ledger import normalize_value, norm_unit, numbers_in_text
from utils.date_coerce import to_naive_local
from utils.logging_utils import get_logger

logger = get_logger("figure_check")

# --------------------------------------------------------------------------- #
# Lexicons: ONE table per category (never incident phrases)
# --------------------------------------------------------------------------- #

# time-unit lexicon (seconds)
TIME_UNITS: dict[str, float] = {
    "second": 1.0, "minute": 60.0, "hour": 3600.0, "day": 86400.0,
    "week": 604800.0, "month": 2592000.0,
}
# mass-unit lexicon (milligrams)
MASS_UNITS: dict[str, float] = {"mcg": 0.001, "µg": 0.001, "mg": 1.0, "g": 1000.0, "kg": 1_000_000.0}
# unit lemma -> (dimension, factor to the dimension's base unit)
DIMENSIONS: dict[str, tuple[str, float]] = {
    **{u: ("time", f) for u, f in TIME_UNITS.items()},
    **{u: ("mass", f) for u, f in MASS_UNITS.items()},
}
# vague quantifier words -> approximate value (single table)
VAGUE_COUNT_WORDS: dict[str, float] = {"couple": 2.0}
# multiplicative adverbs -> count of a "time" unit
MULTIPLICATIVE_WORDS: dict[str, float] = {"twice": 2.0, "thrice": 3.0}

_HEDGE = r"(?:about|around|roughly|approximately|almost|nearly)\s+"
_TIME_UNIT_ALT = "|".join(TIME_UNITS)
_ELAPSED_N_RE = re.compile(
    rf"(?<!\w)(?P<hedge>{_HEDGE})?"
    rf"(?P<n>a\s+few|a\s+couple(?:\s+of)?|several|an?|\d+(?:\.\d+)?|[a-z]+(?:-[a-z]+)?)"
    rf"\s+(?P<unit>{_TIME_UNIT_ALT})s?\s+ago\b",
    re.IGNORECASE,
)
# deictic elapsed expressions: id -> pattern (categories: now / day-part / relative day / relative week)
_ELAPSED_DEICTIC: dict[str, str] = {
    "just_now": r"just\s+now",
    "a_moment": r"a\s+moment\s+ago",
    "earlier_today": r"earlier\s+today",
    "this_morning": r"this\s+morning",
    "this_afternoon": r"this\s+afternoon",
    "this_evening": r"this\s+evening",
    "last_night": r"last\s+night",
    "yesterday": r"yesterday",
    "last_week": r"last\s+week",
}
_ELAPSED_DEICTIC_RE = re.compile(
    r"(?<!\w)(?P<hedge>" + _HEDGE + r")?(?P<d>" +
    "|".join(f"(?P<{k}>{v})" for k, v in _ELAPSED_DEICTIC.items()) + r")(?!\w)",
    re.IGNORECASE,
)
_PRECISE_DEICTIC = {"just_now", "yesterday"}

_MONTH_ALT = "|".join(sorted(
    {m for m in list(calendar.month_name)[1:] + list(calendar.month_abbr)[1:]} | {"Sept"},
    key=len, reverse=True))
_FIRST_SECOND_PERSON_LEMMAS = frozenset({"i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
                                         "ourselves", "you", "your", "yours", "yourself"})
_NUMWORDS_SMALL = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
                   9: "nine", 10: "ten"}

# bound on one check (pure CPU): remaining figures become ``unbound`` past it
DEFAULT_DEADLINE_S = 3.0
_MAX_ITEMS = 600
_ITEM_CHARS = 6000
_BIND_ITEMS = 80          # newest timestamped / overlapping items lemmatized for binding
_PAIR_TS = 40             # newest timestamps used for elapsed(a - b)
_PAIR_VALUES = 150        # values per unit used for sum / difference


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class SourceItem:
    id: str
    text: str
    role: str                         # user | assistant | tool | document | profile | clock | meta
    timestamp: Optional[datetime]
    section: str
    count: Optional[int] = None       # meta items: number of entries in a rendered list section


@dataclass
class Figure:
    span: tuple[int, int]
    text: str
    kind: str                         # number | elapsed | weekday_date
    value: Optional[float] = None
    unit: str = ""
    approx: bool = False
    precise: bool = False             # elapsed only
    expr: Optional[tuple] = None      # elapsed: (family, n, unit_or_id)
    clause: str = ""
    lemmas: frozenset = frozenset()
    in_scope: bool = False
    anchor: str = "unknown"          # elapsed: predicate | duration | noun | unknown (spaCy attachment)
    person: bool = False              # clause subject/possessor is the user, user+Daemon or Daemon


@dataclass
class FigureFinding:
    figure: Figure
    cls: str
    reason: str = ""
    referent_ids: list[str] = field(default_factory=list)
    overlap: int = 0
    actual_age_s: Optional[float] = None
    replacement: Optional[str] = None
    evidence_excerpt: str = ""


@dataclass
class FigureReport:
    findings: list[FigureFinding] = field(default_factory=list)
    status: str = "ok"                # ok | partial
    nlp: bool = True
    figures_total: int = 0

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.cls] = out.get(f.cls, 0) + 1
        return out


# --------------------------------------------------------------------------- #
# spaCy access (lazy, optional)
# --------------------------------------------------------------------------- #

def _get_nlp():
    try:
        import memory.fact_extractor as fact_extractor  # lazy import: startup-cost (spaCy model load)
        return fact_extractor._get_nlp()
    except Exception:  # degrades: no spaCy -> regex/stem binding, elapsed + weekday checks only
        return None


def _stop_words() -> frozenset:
    try:
        from spacy.lang.en.stop_words import STOP_WORDS  # lazy import: startup-cost (spaCy)
        return frozenset(STOP_WORDS)
    except Exception:  # degrades: no stop-word list -> a small closed function-word set
        return frozenset("a an the and or but of to in on at for with by from is are was were be been am it this that "
                         "these those as if so than then there here you your i my we our me us do did does have has "
                         "had not no yes how what when where which who".split())


_STOP = _stop_words()


def _crude_lemma(word: str) -> str:
    w = word.lower()
    for suf in ("ing", "ed", "es", "s", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def _is_time_or_number_word(word: str) -> bool:
    """Time-unit lexicon (singular/plural) or a spelled number: never a binding lemma."""
    w = (word or "").lower()
    return w in TIME_UNITS or (w.endswith("s") and w[:-1] in TIME_UNITS) or _n_value(w) is not None


def _content_lemmas_nlp(doc, exclude: Sequence[tuple[int, int]] = ()) -> frozenset:
    out = set()
    for t in doc:
        if any(s <= t.idx < e for s, e in exclude):
            continue
        if t.lower_ in MASS_UNITS and not t.like_num:   # a dose unit binds; a time unit never does
            out.add(t.lower_)
            continue
        if t.is_punct or t.is_space or t.is_stop or t.like_num or len(t.text) < 3:
            continue
        lem = t.lemma_.lower()
        if _is_time_or_number_word(lem) or _is_time_or_number_word(t.lower_) or lem in _STOP or len(lem) < 3:
            continue
        # crude stem on top of the lemma: "wrecked" (ADJ) and "wreck" (VERB) lemmatise apart
        out.add(_crude_lemma(lem))
    return frozenset(out)


def _content_lemmas_crude(text: str, exclude: Sequence[tuple[int, int]] = ()) -> frozenset:
    out = set()
    for m in re.finditer(r"[A-Za-zµ]{2,}", text or ""):
        if any(s <= m.start() < e for s, e in exclude):
            continue
        w = m.group(0).lower()
        if w in MASS_UNITS:
            out.add(w)
            continue
        if len(w) < 3 or w in _STOP:
            continue
        lem = _crude_lemma(w)
        if _is_time_or_number_word(w) or _is_time_or_number_word(lem) or lem in _STOP:
            continue
        out.add(lem)
    return frozenset(out)


# --------------------------------------------------------------------------- #
# Timestamps and unit helpers
# --------------------------------------------------------------------------- #

def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, datetime):
            return to_naive_local(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return datetime.fromtimestamp(float(value)) if value > 1e9 else None
        text = str(value).strip().replace("Z", "+00:00")
        return to_naive_local(datetime.fromisoformat(text))
    except (ValueError, OverflowError, OSError):
        return None


def _unit_key(word: str) -> str:
    w = (word or "").strip().lower()
    if not w:
        return ""
    if w == "%":
        return "%"
    if w in DIMENSIONS:
        return w
    if w.endswith("s") and w[:-1] in DIMENSIONS:
        return w[:-1]
    try:
        return norm_unit(w) or w
    except Exception:  # degrades: unit stays the lowercased surface
        return w


def _base(value: float, unit: str) -> tuple[str, float]:
    """(dimension-or-unit key, value in the base unit)."""
    dim = DIMENSIONS.get(unit)
    if dim:
        return dim[0], value * dim[1]
    return unit, value


def _n_value(token: str) -> Optional[float]:
    t = (token or "").strip().lower()
    if t in ("a", "an"):
        return 1.0
    try:
        return float(t)
    except ValueError:
        pass
    norm = normalize_value("count", t.replace("-", " "))
    return None if norm is None else float(norm)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def _elapsed_figures(text: str) -> list[Figure]:
    out: list[Figure] = []
    for m in _ELAPSED_N_RE.finditer(text):
        n_raw = m.group("n").lower()
        unit = m.group("unit").lower()
        start = m.start()
        prior = re.search(r"([A-Za-z]+)[\s-]+$", text[: m.start("n")])
        if prior and _n_value(n_raw) is not None and 1 <= _n_value(n_raw) <= 9:
            tens = _n_value(prior.group(1))  # "twenty" + "five": the tens word is the previous token
            if tens and tens >= 20 and tens % 10 == 0:
                n_raw, start = f"{prior.group(1)} {n_raw}".lower(), prior.start(1)
        vague = False
        if n_raw.startswith("a few") or n_raw.startswith("a couple") or n_raw == "several":
            vague, n = True, None
        else:
            n = _n_value(n_raw)
            if n is None:
                continue
        precise = (not vague) and unit in ("minute", "hour", "day")
        out.append(Figure(
            span=(start, m.end()), text=text[start:m.end()], kind="elapsed", value=n, unit=unit,
            precise=precise, expr=("rel", n, unit if not vague else f"~{unit}"),
        ))
    for m in _ELAPSED_DEICTIC_RE.finditer(text):
        key = next(k for k in _ELAPSED_DEICTIC if m.group(k))
        out.append(Figure(
            span=(m.start(), m.end()), text=m.group(0), kind="elapsed",
            precise=key in _PRECISE_DEICTIC, expr=("deictic", None, key),
        ))
    return out


def _weekday_date_figures(text: str) -> list[Figure]:
    """One Figure per impossible weekday/date pair, located via the shared checker."""
    out: list[Figure] = []
    for item in weekday_date_mismatches(text):
        m = re.match(r"(?P<wd>\w+) (?P<iso>\d{4}-\d{2}-\d{2}) \(actual (?P<actual>\w+)\)", item)
        if not m:
            continue
        iso = m.group("iso")
        year, month, day = (int(p) for p in iso.split("-"))
        month_names = {calendar.month_name[month].lower(), calendar.month_abbr[month].lower()}
        if month == 9:
            month_names.add("sept")
        for w in re.finditer(rf"\b{m.group('wd')}\b", text, flags=re.IGNORECASE):
            window = text[max(0, w.start() - 50): w.end() + 50].lower()
            near_date = iso in window or (
                any(mn in window for mn in month_names)
                and re.search(rf"\b0?{day}(?:st|nd|rd|th)?\b", window))
            if near_date:
                out.append(Figure(
                    span=(w.start(), w.end()), text=w.group(0), kind="weekday_date",
                    expr=("weekday", iso, m.group("actual")),
                ))
    return out


def _number_figures(text: str, taken: Sequence[tuple[int, int]]) -> list[Figure]:
    out: list[Figure] = []

    def overlaps(s: int, e: int) -> bool:
        return any(s < te and e > ts for ts, te in taken)

    for canon, off, surface in numbers_in_text(text):
        end = off + len(surface)
        if overlaps(off, end):
            continue
        before = text[max(0, off - 1): off]
        after = text[end: end + 12]
        if before in (":", "/", "#", "$", "[", "_") or re.match(r"[:/\-]\d|\.\d|(?:st|nd|rd|th)\b", after):
            continue
        value = float(canon)
        if surface.isdigit() and len(surface) == 4 and 1900 <= value <= 2100:
            continue
        m = re.match(r"\s*(?:-\s*)?(%|[A-Za-zµ]+)", after if after.startswith("%") else text[end:end + 30])
        word = m.group(1) if m else ""
        if word.lower() in ("x", "times", "time"):
            unit = "time"
        elif word == "%":
            unit = "%"
        elif not word or word.lower() in _STOP:
            unit = ""
        else:
            unit = _unit_key(word)
        out.append(Figure(span=(off, end), text=surface, kind="number", value=value, unit=unit))
    for m in re.finditer(r"(?<!\w)(" + "|".join(MULTIPLICATIVE_WORDS) + r")(?!\w)", text, flags=re.IGNORECASE):
        if not overlaps(m.start(), m.end()):
            out.append(Figure(span=(m.start(), m.end()), text=m.group(0), kind="number",
                              value=MULTIPLICATIVE_WORDS[m.group(1).lower()], unit="time"))
    for m in re.finditer(r"(?<!\w)a\s+(" + "|".join(VAGUE_COUNT_WORDS) + r")(?:\s+of)?\s+([A-Za-z]+)", text,
                         flags=re.IGNORECASE):
        if not overlaps(m.start(), m.end()):
            out.append(Figure(span=(m.start(), m.end()), text=m.group(0), kind="number",
                              value=VAGUE_COUNT_WORDS[m.group(1).lower()], unit=_unit_key(m.group(2)),
                              approx=True))
    return out


def extract_figures(reply: str) -> list[Figure]:
    """Every figure the reply asserts, in text order (scope is decided by ``check_reply``)."""
    text = reply or ""
    if not text.strip():
        return []
    elapsed = _elapsed_figures(text)
    weekday = _weekday_date_figures(text)
    taken = [f.span for f in elapsed + weekday]
    # citation brackets like [WEB_3] are markers, not figures
    taken += [m.span() for m in re.finditer(r"\[[A-Za-z_]+\d*\]", text)]
    # date numbers ("October 26", "26 Oct") are not quantities
    for m in re.finditer(rf"\b(?:{_MONTH_ALT})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTH_ALT})\b",
                         text, flags=re.IGNORECASE):
        taken.append(m.span())
    figures = elapsed + weekday + _number_figures(text, taken)
    figures.sort(key=lambda f: f.span)
    # an elapsed span nested in another elapsed span ("just now" inside "a moment ...") keeps the longest
    kept: list[Figure] = []
    for f in figures:
        if kept and f.kind == "elapsed" and kept[-1].kind == "elapsed" and f.span[0] < kept[-1].span[1]:
            continue
        kept.append(f)
    return kept


# --------------------------------------------------------------------------- #
# Source pool
# --------------------------------------------------------------------------- #

_CONVERSATION_KEYS = ("recent_conversations", "recent", "relevant_conversations", "relevant_memories",
                      "memories", "conversation_history", "history")
_TEXT_FIELDS = ("text", "content", "summary", "fact", "document", "description", "title")
_TS_FIELDS = ("timestamp", "created_at", "date", "last_seen")


def _role_for_key(key: str) -> str:
    k = key.lower()
    if "web" in k or "tool" in k:
        return "tool"
    if any(p in k for p in ("profile", "fact", "summar", "reflect")):
        return "profile"
    return "document"


def _leaf_text(value: Any, depth: int = 0) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if depth >= 2:
        return ""
    if isinstance(value, Mapping):
        return " ".join(t for t in (_leaf_text(v, depth + 1) for v in value.values()) if t)
    if isinstance(value, (list, tuple)):
        return " ".join(t for t in (_leaf_text(v, depth + 1) for v in list(value)[:50]) if t)
    return ""


def _entry_text_ts(entry: Any) -> tuple[str, Optional[datetime]]:
    if isinstance(entry, str):
        return entry, None
    if isinstance(entry, Mapping):
        meta = entry.get("metadata") if isinstance(entry.get("metadata"), Mapping) else {}
        ts = next((entry.get(k) for k in _TS_FIELDS if entry.get(k) not in (None, "")), None)
        if ts is None:
            ts = next((meta.get(k) for k in _TS_FIELDS if meta.get(k) not in (None, "")), None)
        for f in _TEXT_FIELDS:
            if isinstance(entry.get(f), str) and entry.get(f).strip():
                return entry[f], _parse_ts(ts)
        return _leaf_text(entry), _parse_ts(ts)
    pages = getattr(entry, "pages", None)
    if pages:
        return " ".join(_leaf_text(getattr(p, "title", "")) + " " + _leaf_text(getattr(p, "content", ""))
                        for p in list(pages)[:20]), None
    return "", None


def build_source_pool(raw_context: Any, history: Iterable[Any], source_material: str, now: datetime,
                      *, user_text: str = "") -> list[SourceItem]:
    """Everything in context a reply figure may legitimately come from."""
    import core.personal_claim_check as personal_claim_check  # lazy import: layering (keeps this module's import graph one-way)

    items: list[SourceItem] = []
    ctx = raw_context if isinstance(raw_context, Mapping) else {}
    try:
        entries = personal_claim_check.build_personal_evidence(
            user_text or "", dict(ctx), history=list(history or ()), max_chars=400_000)
    except Exception as exc:  # degrades: conversation rows unavailable to the pool this turn
        logger.debug(f"[FigureCheck] conversation evidence skipped: {exc}")
        entries = []
    for e in entries:
        role = e.get("role") or "unknown"
        if role == "attachment":
            role = "document"
        items.append(SourceItem(
            id=str(e.get("source_id") or f"conv:{len(items)}"), text=str(e.get("text") or "")[:_ITEM_CHARS],
            role=role, timestamp=_parse_ts(e.get("timestamp")), section="conversation"))
    for key, value in ctx.items():
        if key in _CONVERSATION_KEYS or len(items) >= _MAX_ITEMS:
            continue
        role = _role_for_key(str(key))
        seq: list[Any]
        if isinstance(value, (list, tuple)):
            seq = list(value)
            if seq:
                items.append(SourceItem(id=f"count:{key}", text="", role="meta", timestamp=None,
                                        section=str(key), count=len(seq)))
        elif value is None or isinstance(value, (bool, int, float)):
            continue
        else:
            seq = [value]
        for entry in seq[:100]:
            if len(items) >= _MAX_ITEMS:
                break
            text, ts = _entry_text_ts(entry)
            if text and text.strip():
                # Content-derived id (BC-95): a list position is not identity, and ids reach the findings log.
                item_id = hashlib.sha1(f"{text}|{ts}".encode("utf-8")).hexdigest()[:10]
                items.append(SourceItem(id=f"{key}:{item_id}", text=text[:_ITEM_CHARS], role=role, timestamp=ts,
                                        section=str(key)))
    if source_material and str(source_material).strip():
        items.append(SourceItem(id="source_material", text=str(source_material)[:_ITEM_CHARS * 4], role="tool",
                                timestamp=None, section="source_material"))
    items.append(SourceItem(id="clock", text="", role="clock", timestamp=now, section="clock"))
    return items


# --------------------------------------------------------------------------- #
# Elapsed intervals
# --------------------------------------------------------------------------- #

def _day_delta(now: datetime, ts: datetime) -> int:
    return (now.date() - ts.date()).days


def _interval_ok(fig: Figure, ts: datetime, now: datetime) -> bool:
    """True when the item stamped ``ts`` falls inside the figure's elapsed interval."""
    age = (now - ts).total_seconds()
    if age < -60:
        return False
    family, n, key = fig.expr
    if family == "deictic":
        if key == "just_now":
            return age <= 300
        if key == "a_moment":
            return age <= 600
        if key == "yesterday":
            return _day_delta(now, ts) == 1
        if key == "earlier_today":
            return _day_delta(now, ts) == 0
        if key in ("this_morning", "this_afternoon", "this_evening"):
            lo, hi = {"this_morning": (4, 12), "this_afternoon": (12, 18), "this_evening": (18, 24)}[key]
            return _day_delta(now, ts) == 0 and lo <= ts.hour < hi
        if key == "last_night":
            start = datetime.combine(now.date() - timedelta(days=1), datetime.min.time()) + timedelta(hours=18)
            end = datetime.combine(now.date(), datetime.min.time()) + timedelta(hours=4)
            return start <= ts < end
        if key == "last_week":
            monday = now.date() - timedelta(days=now.weekday())
            return monday - timedelta(days=7) <= ts.date() < monday
        return False
    unit = key.lstrip("~")
    if key.startswith("~"):  # a few / couple of / several
        lo_hi = {"minute": (60, 1200), "hour": (3600, 21600), "day": (None, None), "week": (None, None)}.get(unit)
        if unit == "day":
            return 2 <= _day_delta(now, ts) <= 5
        if unit in ("week", "month"):
            days = _day_delta(now, ts)
            return (7 if unit == "week" else 30) <= days <= (28 if unit == "week" else 120)
        return bool(lo_hi) and lo_hi[0] <= age <= lo_hi[1]
    if unit == "minute":
        return age <= 300 if n <= 1 else 0.5 * n * 60 <= age <= (1.5 * n + 2) * 60
    if unit == "hour":
        return 1800 <= age <= 6300 if n <= 1 else (n - 0.75) * 3600 <= age <= (n + 0.75) * 3600
    if unit == "day":
        return abs(_day_delta(now, ts) - n) <= 1
    if unit == "week":
        return abs(_day_delta(now, ts) - 7 * n) <= 4
    if unit == "month":
        return abs(_day_delta(now, ts) - 30 * n) <= 15
    return False


def _humanize_age(ts: datetime, now: datetime, capitalize: bool) -> str:
    age = max((now - ts).total_seconds(), 0.0)
    days = _day_delta(now, ts)
    if age < 90:
        out = "a minute ago"
    elif age < 3600:
        out = f"{max(2, round(age / 60))} minutes ago"
    elif age < 6300:
        out = "an hour ago"
    elif age < 86400 and days <= 1 and not (days == 1 and age >= 64800):
        out = f"about {round(age / 3600)} hours ago"
    elif days <= 1:
        out = "yesterday"
    else:
        out = f"{_NUMWORDS_SMALL.get(days, days)} days ago"
    if capitalize and out and out[0].isdigit():  # a sentence never starts on a numeral
        head, _, rest = out.partition(" ")
        out = f"{_NUMWORDS_SMALL.get(int(head), head)} {rest}"
    return out[0].upper() + out[1:] if capitalize and out else out


# --------------------------------------------------------------------------- #
# The check
# --------------------------------------------------------------------------- #

def _figure_clause(doc, fig: Figure):
    """(clause text, token indices) of the nearest verbal head's subtree around the figure."""
    start, end = fig.span
    toks = [t for t in doc if start <= t.idx < end] or [t for t in doc if t.idx >= start][:1]
    if not toks:
        return fig.text, []
    tok = toks[-1]
    head = tok
    for anc in [tok] + list(tok.ancestors):
        if anc.pos_ in ("VERB", "AUX"):
            head = anc
            break
    else:
        head = tok.sent.root
    sub = list(head.subtree)
    return doc[sub[0].i: sub[-1].i + 1].text, sub


class _Binder:
    """Lemma-overlap binding of a clause to pool items (lazy, cached)."""

    def __init__(self, pool: Sequence[SourceItem], nlp):
        self.pool = pool
        self.nlp = nlp
        self._lem: dict[str, frozenset] = {}

    def lemmas(self, item: SourceItem) -> frozenset:
        got = self._lem.get(item.id)
        if got is None:
            if self.nlp is not None:
                doc = next(iter(self.nlp.pipe([item.text[:2500]], disable=["parser", "ner"])))
                got = _content_lemmas_nlp(doc)
            else:
                got = _content_lemmas_crude(item.text)
            self._lem[item.id] = got
        return got

    def referents(self, fig_lemmas: frozenset, candidates: Sequence[SourceItem], min_overlap: int,
                  by_turn: bool = False):
        """(top overlap, best items). ``by_turn`` unions the lemmas of items stamped the same
        instant (the user row and the assistant row of one stored turn)."""
        scored = []
        if by_turn:
            turns: dict = {}
            for it in candidates:
                turns.setdefault(it.timestamp, []).append(it)
            for group in turns.values():
                lem = frozenset().union(*(self.lemmas(i) for i in group))
                ov = len(fig_lemmas & lem)
                if ov >= min_overlap:
                    scored.append((ov, group))
            if not scored:
                return 0, []
            top = max(s for s, _ in scored)
            return top, [i for s, g in scored if s == top for i in g]
        for it in candidates:
            ov = len(fig_lemmas & self.lemmas(it))
            if ov >= min_overlap:
                scored.append((ov, it))
        if not scored:
            return 0, []
        top = max(s for s, _ in scored)
        return top, [it for s, it in scored if s == top]


def _first_second_person(sub_tokens) -> bool:
    for t in sub_tokens:
        if t.lower_ in _FIRST_SECOND_PERSON_LEMMAS:
            return True
        person = t.morph.get("Person")
        if person and person[0] in ("1", "2") and t.pos_ == "PRON":
            return True
    return False


def _ranked_ts_items(pool: Sequence[SourceItem]) -> list[SourceItem]:
    return sorted((i for i in pool if i.timestamp is not None and i.role in ("user", "assistant")),
                  key=lambda i: i.timestamp, reverse=True)


def _pool_numbers(pool: Sequence[SourceItem]) -> dict[str, list[tuple[float, str, str]]]:
    """key (dimension or unit) -> [(base value, unit, item id)] for every unit-bearing number in the pool."""
    out: dict[str, list[tuple[float, str, str]]] = {}
    for it in pool:
        if not it.text:
            continue
        for fig in _number_figures(it.text, [m.span() for m in re.finditer(r"\[[A-Za-z_]+\d*\]", it.text)]):
            if not fig.unit:
                continue
            key, base = _base(fig.value, fig.unit)
            out.setdefault(key, []).append((base, fig.unit, it.id))
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def _derive_number(fig: Figure, key: str, base: float, pool: Sequence[SourceItem], now: datetime,
                   same_key: list[tuple[float, str, str]]) -> Optional[str]:
    """Name of the fixed operation that reproduces the figure, or None."""
    if key == "time":
        unit_s = TIME_UNITS.get(fig.unit, 1.0)
        stamps = [now] + [i.timestamp for i in _ranked_ts_items(pool)[:_PAIR_TS]]
        stamps = [s for s in stamps if s is not None]
        for i, a in enumerate(stamps):
            for b in stamps[i + 1:]:
                gap = abs((a - b).total_seconds()) / unit_s
                if abs(gap - fig.value) < 0.5 or (fig.unit == "day" and abs(abs((a.date() - b.date()).days) - fig.value) < 0.01):
                    return "elapsed"
    vals = [v for v, _, _ in same_key][:_PAIR_VALUES]
    for i, a in enumerate(vals):
        for b in vals[i:]:
            if _close(a + b, base) or _close(abs(a - b), base):
                return "sum_difference"
    if not fig.unit or key not in ("time", "mass"):
        for it in pool:
            if it.count is not None and float(it.count) == fig.value and fig.unit not in ("%",):
                return "list_count"
    return None


def check_reply(reply: str, pool: Sequence[SourceItem], now: datetime, *, min_overlap: int = 1,
                deadline_s: float = DEFAULT_DEADLINE_S, nlp: Any = "auto") -> FigureReport:
    """Classify every figure of ``reply`` (read-only; never edits)."""
    nlp_model = _get_nlp() if nlp == "auto" else nlp
    report = FigureReport(nlp=nlp_model is not None)
    figures = extract_figures(reply)
    report.figures_total = len(figures)
    if not figures:
        return report
    started = time.monotonic()
    doc = nlp_model(reply) if nlp_model is not None else None
    binder = _Binder(pool, nlp_model)
    conv_ts = _ranked_ts_items(pool)[:_BIND_ITEMS]
    user_items = [i for i in pool if i.role in ("user", "profile", "assistant") and i.text]
    pool_numbers: Optional[dict] = None

    for fig in figures:
        if time.monotonic() - started > deadline_s:
            report.status = "partial"
            report.findings.append(FigureFinding(fig, "unbound", reason="deadline"))
            continue
        # ---- clause + lemmas ----
        excl = [fig.span]
        if doc is not None:
            fig.clause, sub = _figure_clause(doc, fig)
            first = next((t for t in doc if fig.span[0] <= t.idx < fig.span[1]), None)
            sent = first.sent if first is not None else doc[:]
            fig.lemmas = _content_lemmas_nlp(sent, excl)   # binding = the whole SENTENCE; scope stays clause-based
            person_in_clause = _first_second_person(sub)
            fig.person = person_in_clause
            if fig.kind == "elapsed":
                fig.anchor = _span_anchor(doc, fig)
        else:
            lo = max(0, reply.rfind(".", 0, fig.span[0]) + 1)
            hi = reply.find(".", fig.span[1])
            fig.clause = reply[lo: hi if hi >= 0 else len(reply)]
            fig.lemmas = _content_lemmas_crude(fig.clause, [(fig.span[0] - lo, fig.span[1] - lo)])  # sentence-bounded
            person_in_clause = bool(re.search(r"\b(?:you|your|we|our|i|my)\b", fig.clause, flags=re.IGNORECASE))
            fig.person = person_in_clause
        if fig.kind == "number" and fig.unit:
            fig.lemmas = fig.lemmas | {fig.unit}
        try:
            if fig.kind == "weekday_date":
                fig.in_scope = True
                _, iso, actual = fig.expr
                report.findings.append(FigureFinding(
                    fig, "weekday_date", reason=f"actual:{actual}", replacement=actual))
            elif fig.kind == "elapsed":
                fig.in_scope = True
                report.findings.append(_check_elapsed(fig, conv_ts, binder, now, min_overlap, pool))
            else:
                if pool_numbers is None:
                    pool_numbers = _pool_numbers(pool)
                report.findings.append(_check_number(
                    fig, person_in_clause, user_items, binder, now, min_overlap, pool, pool_numbers))
        except Exception as exc:  # degrades: this figure is unchecked; the rest of the reply still is
            logger.debug(f"[FigureCheck] figure skipped: {exc}")
            report.findings.append(FigureFinding(fig, "unbound", reason="check_error"))
    return report


def _elapsed_expr_key(fig: Figure) -> tuple:
    family, n, key = fig.expr
    return (family, n, key.lstrip("~"))


# duration-boundary prepositions: "since/until/till/from..to/for/during" open or close a span, not a point
_DURATION_PREPS = frozenset({"since", "until", "till", "from", "for", "during", "throughout", "to", "through"})
_CLAUSE_DEPS = frozenset({"ROOT", "advcl", "ccomp", "xcomp", "relcl", "conj"})


def _span_anchor(doc, fig: Figure) -> str:
    """How the time expression attaches: ``predicate`` (an adverbial of a VERB/AUX/ADJ head),
    ``duration`` (object of since/until/for/...), ``noun`` (describes a noun, "notes from yesterday")
    or ``unknown``."""
    toks = [t for t in doc if fig.span[0] <= t.idx < fig.span[1]]
    if not toks:
        return "unknown"
    ids = {t.i for t in toks}
    root = next((t for t in toks if t.head.i not in ids), toks[-1])
    head = root.head
    if root.dep_ == "pobj" and head.dep_ in ("prep", "agent"):
        if head.head.pos_ in ("NOUN", "PROPN"):
            return "noun"
        if head.lemma_.lower() in _DURATION_PREPS:
            return "duration"
        if head.head.pos_ in ("VERB", "AUX", "ADJ"):
            return "predicate"
    if root.dep_ in ("npadvmod", "compound", "nmod", "poss") and head.pos_ in ("NOUN", "PROPN"):
        return "noun"
    if head.pos_ in ("VERB", "AUX", "ADJ"):
        return "predicate"
    return "unknown"


def _clause_content_lemmas(tok) -> frozenset:
    """Content lemmas of the SMALLEST clause holding ``tok``: the subtree of the nearest
    VERB/AUX ancestor whose dep is ROOT/advcl/ccomp/xcomp/relcl/conj, minus the subtrees of
    clauses nested inside it. Sentence co-occurrence alone never counts."""
    head = tok
    for anc in [tok] + list(tok.ancestors):
        if anc.pos_ in ("VERB", "AUX") and anc.dep_ in _CLAUSE_DEPS:
            head = anc
            break
    else:
        head = tok.sent.root
    nested = [d for d in head.subtree if d is not head and d.pos_ in ("VERB", "AUX") and d.dep_ in _CLAUSE_DEPS]
    skip = {t.i for d in nested for t in d.subtree}
    out = set()
    for t in head.subtree:
        if t.i in skip or t.is_punct or t.is_space or t.is_stop or t.like_num or len(t.text) < 3:
            continue
        lem = t.lemma_.lower()
        if _is_time_or_number_word(lem) or _is_time_or_number_word(t.lower_) or lem in _STOP:
            continue
        out.add(_crude_lemma(lem))
    return frozenset(out)


def _referent_clause_narrates(nlp, text: str, bound: frozenset) -> bool:
    """True when an event-anchored time expression sits in the SAME clause as a bound lemma."""
    spans = _event_anchored_spans(text)
    if not spans:
        return False
    doc = next(iter(nlp.pipe([text[:2500]], disable=["ner"])))
    for s_, e_ in spans:
        tok = next((t for t in doc if s_ <= t.idx < e_), None)
        if tok is not None and bound & _clause_content_lemmas(tok):
            return True
    return False


def _event_anchored_spans(text: str) -> list:
    spans = [f.span for f in _elapsed_figures(text or "")]
    spans += [m.span() for m in re.finditer(r"\b(?:" + "|".join(calendar.day_name) + r")\b", text or "", flags=re.IGNORECASE)]
    spans += [m.span() for m in re.finditer(
        rf"\b(?:{_MONTH_ALT})\.?\s+\d{{1,2}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b", text or "", flags=re.IGNORECASE)]
    return spans


def _event_anchored_times(text: str) -> list:
    """Event-anchored time expressions only ("N hours ago", yesterday, last night, weekday names,
    month-day dates). General durations ("12-24 hours out", "within a week", "for 3 days") are NOT."""
    return (_elapsed_figures(text) or []) + _weekday_names(text) + re.findall(
        rf"\b(?:{_MONTH_ALT})\.?\s+\d{{1,2}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b", text or "", flags=re.IGNORECASE)


def _is_moment_ago(fig: Figure) -> bool:
    family, _n, key = fig.expr
    return (family == "deictic" and key == "just_now") or (family == "rel" and key in ("minute", "hour"))


def _weekday_names(text: str) -> list:
    return re.findall(r"\b(?:" + "|".join(calendar.day_name) + r")\b", text or "", flags=re.IGNORECASE)


def _check_elapsed(fig: Figure, conv_ts: Sequence[SourceItem], binder: _Binder, now: datetime,
                   min_overlap: int, pool: Sequence[SourceItem]) -> FigureFinding:
    if fig.anchor in ("duration", "noun"):   # "since yesterday" / "notes from yesterday": not an elapsed point
        return FigureFinding(fig, "out_of_scope", reason=f"anchor_{fig.anchor}")
    top, refs = binder.referents(fig.lemmas, conv_ts, min_overlap, by_turn=True)
    if not refs:
        return FigureFinding(fig, "unbound", reason="no_referent")
    ids = [r.id for r in refs]
    excerpt = refs[0].text[:200]
    # a referent that itself says the same elapsed phrase is an event-relative time the reply COPIES
    my_key = _elapsed_expr_key(fig)
    for r in refs:
        if any(_elapsed_expr_key(f) == my_key for f in _elapsed_figures(r.text)):
            return FigureFinding(fig, "copied", reason="copied_from_referent", referent_ids=ids, overlap=top,
                                 evidence_excerpt=excerpt)
    if any(_interval_ok(fig, r.timestamp, now) for r in refs):
        return FigureFinding(fig, "derived:elapsed", referent_ids=ids, overlap=top, evidence_excerpt=excerpt)
    ages = sorted((now - r.timestamp).total_seconds() for r in refs)
    newest = min(refs, key=lambda r: (now - r.timestamp).total_seconds())
    agree = (ages[-1] - ages[0]) <= max(300.0, 0.25 * ages[-1])
    if not fig.precise:
        return FigureFinding(fig, "elapsed_vague", reason="contradicted_vague", referent_ids=ids, overlap=top,
                             actual_age_s=ages[0], evidence_excerpt=excerpt)
    if not agree:
        return FigureFinding(fig, "elapsed_unconfirmed", reason="ambiguous_referents", referent_ids=ids, overlap=top,
                             actual_age_s=ages[0], evidence_excerpt=excerpt)
    # Only the "ago" family pointing at a MOMENT (just now / N minutes / N hours ago) is ever edited;
    # day-scale deictics (yesterday, N days ago, ...) are reported, never rewritten.
    if not _is_moment_ago(fig):
        return FigureFinding(fig, "elapsed_unconfirmed", reason="day_scale_not_edited", referent_ids=ids,
                             overlap=top, actual_age_s=ages[0], evidence_excerpt=excerpt)
    if min(ages) >= 6 * 3600:   # a moment-scale phrase is never rewritten into a day-scale one
        return FigureFinding(fig, "elapsed_unconfirmed", reason="age_not_moment_scale", referent_ids=ids,
                             overlap=top, actual_age_s=ages[0], evidence_excerpt=excerpt)
    # A correction needs evidence the phrase times the referent ITSELF, not an event it narrates:
    # a person in the clause, a binding on >= 2 content lemmas, and a referent that does not
    # itself narrate other times (measured 2026-10-25: single-lemma / person-less / time-narrating
    # referents produced the only false corrections on 102 stored replies).
    weak = None
    if not fig.person:
        weak = "elapsed_not_about_user"
    elif fig.anchor != "predicate":
        weak = "not_a_predicate_adverbial"
    elif top < max(2, min_overlap):
        weak = "weak_binding"
    elif binder.nlp is None or any(_referent_clause_narrates(binder.nlp, r.text, fig.lemmas) for r in refs):
        weak = "referent_narrates_time"
    if weak:
        return FigureFinding(fig, "elapsed_unconfirmed", reason=weak, referent_ids=ids, overlap=top,
                             actual_age_s=ages[0], evidence_excerpt=excerpt)
    phrase = _humanize_age(newest.timestamp, now, fig.text[:1].isupper())
    return FigureFinding(fig, "elapsed_precise", reason="contradicted_precise", referent_ids=ids, overlap=top,
                         actual_age_s=ages[0], replacement=phrase, evidence_excerpt=excerpt)


def _check_number(fig: Figure, person_in_clause: bool, user_items: Sequence[SourceItem], binder: _Binder,
                  now: datetime, min_overlap: int, pool: Sequence[SourceItem], pool_numbers: dict) -> FigureFinding:
    if not fig.unit:
        return FigureFinding(fig, "out_of_scope", reason="no_unit")
    top, refs = binder.referents(fig.lemmas, [i for i in user_items if i.role in ("user", "profile")], min_overlap)
    if not (person_in_clause or refs):
        return FigureFinding(fig, "out_of_scope", reason="world_fact")
    fig.in_scope = True
    key, base = _base(fig.value, fig.unit)
    same_key = pool_numbers.get(key, [])
    for v, unit, item_id in same_key:
        if _close(v, base) or (fig.approx and abs(v - base) <= 1.0):
            op = "copied" if unit == fig.unit else "derived:unit_conversion"
            return FigureFinding(fig, op, referent_ids=[item_id])
    op = _derive_number(fig, key, base, pool, now, same_key)
    if op:
        return FigureFinding(fig, f"derived:{op}")
    ref_ids = [r.id for r in refs]
    ref_has_unit = False
    for r in refs:
        for f in _number_figures(r.text, ()):
            if f.unit and _base(f.value, f.unit)[0] == key:
                ref_has_unit = True
                return FigureFinding(fig, "copied_mismatch", reason="referent_has_other_value",
                                     referent_ids=ref_ids, overlap=top, evidence_excerpt=r.text[:200])
    return FigureFinding(fig, "unsupported", reason="no_source" if not ref_has_unit else "other",
                         referent_ids=ref_ids, overlap=top)


# --------------------------------------------------------------------------- #
# Corrections
# --------------------------------------------------------------------------- #

def apply_corrections(reply: str, report: FigureReport,
                      correct_classes: Iterable[str] = ("elapsed_precise", "weekday_date")) -> tuple[str, int]:
    """Rewrite ONLY contradicted precise elapsed phrases and impossible weekday names."""
    wanted = set(correct_classes)
    edits: list[tuple[int, int, str]] = []
    for f in report.findings:
        if f.cls in wanted and f.replacement and f.cls in ("elapsed_precise", "weekday_date"):
            start, end = f.figure.span
            repl = f.replacement
            if f.cls == "weekday_date":
                orig = reply[start:end]
                repl = repl[:3] if len(orig) <= 3 else repl
                if orig.isupper():
                    repl = repl.upper()
            edits.append((start, end, repl))
    if not edits:
        return reply, 0
    edits.sort()
    out, cursor, n = [], 0, 0
    for start, end, repl in edits:
        if start < cursor:
            continue
        out.append(reply[cursor:start])
        out.append(repl)
        cursor, n = end, n + 1
    out.append(reply[cursor:])
    return "".join(out), n
