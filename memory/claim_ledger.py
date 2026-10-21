"""User-claim ledger: extractor prompt, deterministic validator, reply check (pure).

Incident 2026-10-20; class: BC-75 (self-authored artifact laundered into
evidence), BC-52 (lossy multi-hop derivation, a stale claim resurfaces).

The user is the only source of values about the user. A value has a date and a
modality, and the newest wins. Two things the user said were later restated by
Daemon and drifted: a CONDITIONAL ("if I can get 6 done today then I am only 6
days behind") came back as plain fact, and an ESTIMATE ("12 days i think?")
survived its worked-out replacement (~15 days). A deterministic lemma matcher
could not tell which subject a figure belonged to, so subject identity comes
from a small model and ADMISSION from deterministic checks:

  * the model proposes ``{kind, subject_key, value, unit, modality, quote}``;
  * the validator admits a claim only when the quote is an EXACT span of the
    user's text, the value appears in that quote, the enums hold, and the
    spaCy conditional parse (``fact_extractor._irrealis_conflict``) agrees --
    a parse that says conditional wins over the model, and a model's
    conditional/estimate is never upgraded to asserted;
  * a failed claim is DROPPED with a reason, never repaired (BC-84).

This module is pure: no model calls except through a caller-supplied async
callable, no I/O except the explicit ``ClaimLedger.load``/``save`` (path is a
parameter; nothing here knows a live store path). Nothing in the live turn path
imports it yet -- the live wiring is a later lane.

JSON parsing: ``core.personal_claim_check._strict_json`` is private and pinned
to the single key ``claims`` (this module also needs ``findings``); a shared
helper would mean editing that module for a ~25-line function, so ``_strict_json``
below is a local, key-parameterised sibling with the same shape vocabulary.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence

from utils import safe_json
from utils.temporal_resolver import (
    parse_said_on,
    resolve_date_expression,
    resolve_temporal_references,
)
from utils.read_time_markers import strip_machinery
from utils.trigger_match import normalize_ws

SCHEMA_VERSION = 1
KINDS = ("count", "duration", "date", "deadline", "status")
MODALITIES = ("asserted", "estimate", "conditional", "correction")
REPLY_MODALITIES = ("asserted", "estimate", "conditional", "plan_discussion")
STATUS_VALUES = ("done", "not_done", "planned")
SCOPES = ("standing", "occurrence")
CLASSES = ("matches_current", "stale", "conditional_as_fact", "estimate_as_fact",
           "differs_from_estimate")
# The only classes the LIVE reply check records (design 7a.4): modality findings on
# STANDING claims. stale / differs_from_estimate stay in the code for the offline
# baseline script, where they were measured (0 confirmed drift of 15 + 7 flagged).
REPLY_CHECK_LIVE_CLASSES = ("conditional_as_fact", "estimate_as_fact")

OPEN_SUBJECTS_CAP = 40
MAX_USER_CHARS = 4000       # authored head only; a paste's tail is not the user's figure
MAX_REPLY_CHARS = 6000
MAX_QUOTE_CHARS = 400
MAX_DESCRIPTION_CHARS = 160
MAX_CLAIMS_PER_TURN = 12
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,48}$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class LedgerClaim:
    id: str
    subject_key: str
    description: str
    kind: str
    value: str
    unit: str
    modality: str
    stated_at: str            # ISO timestamp of the turn that said it
    quote: str
    turn_id: str = ""
    superseded_by: Optional[str] = None
    unit_norm: str = ""       # lemma of the unit head; part of the subject key (A2)
    scope: str = "standing"   # standing = holds until it changes; occurrence = bound to one date
    occurrence_date: str = ""  # ISO date an occurrence claim is bound to ("" for standing)

    def __post_init__(self) -> None:
        if not self.unit_norm and self.unit:
            self.unit_norm = self.unit.strip().lower()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LedgerClaim":
        if not isinstance(data, Mapping):
            raise ValueError("claim is not an object")
        for name in ("id", "subject_key", "description", "kind", "value",
                     "unit", "modality", "stated_at", "quote"):
            if not isinstance(data.get(name), str):
                raise ValueError(f"claim field {name!r} missing or not text")
        if data["kind"] not in KINDS or data["modality"] not in MODALITIES:
            raise ValueError("claim kind/modality outside the enums")
        if _dt(data["stated_at"]) is None:
            raise ValueError("claim stated_at is not a timestamp")
        sup = data.get("superseded_by")
        if sup is not None and not isinstance(sup, str):
            raise ValueError("claim superseded_by is not text")
        # A claim stored before scope existed has no key: it is a standing claim.
        scope = data.get("scope", "standing")
        if scope not in SCOPES:
            raise ValueError("claim scope outside the enums")
        occ = data.get("occurrence_date", "")
        if not isinstance(occ, str) or (occ and not _ISO_RE.match(occ)):
            raise ValueError("claim occurrence_date is not an ISO date")
        return cls(
            id=data["id"], subject_key=data["subject_key"],
            description=data["description"], kind=data["kind"],
            value=data["value"], unit=data["unit"], modality=data["modality"],
            stated_at=data["stated_at"], quote=data["quote"],
            turn_id=str(data.get("turn_id") or ""), superseded_by=sup,
            unit_norm=str(data.get("unit_norm") or ""),
            scope=scope, occurrence_date=occ,
        )


def _dt(value: Any) -> Optional[datetime]:
    return parse_said_on(value)


def _iso(value: Any) -> str:
    dt = _dt(value)
    return dt.isoformat(timespec="seconds") if dt else ""


def make_claim_id(subject_key: str, kind: str, stated_at: str, value: str, quote: str) -> str:
    raw = f"{subject_key}|{kind}|{stated_at}|{value}|{normalize_ws(quote)}"
    return "clm_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]  # noqa: S324 - identity, not security


def _rank(claim: LedgerClaim) -> int:
    """Within one calendar day an asserted/correction claim outranks an estimate."""
    return 0 if claim.modality == "estimate" else 1


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class ClaimLedger:
    """In-memory claims with supersession; ``load``/``save`` are explicit."""

    def __init__(self, claims: Optional[Iterable[LedgerClaim]] = None):
        self.claims: list[LedgerClaim] = list(claims or [])

    # -- writing -----------------------------------------------------------
    def add(self, claim: LedgerClaim) -> list[str]:
        """Record ``claim`` and apply supersession; returns the ids it superseded.

        * a newer same (subject_key, kind, unit_norm) claim supersedes the older
          one (two values with different units are never the same subject); an
          OCCURRENCE claim supersedes only within the same occurrence_date, and a
          standing and an occurrence claim never touch each other;
        * an ``estimate`` never supersedes an asserted/correction claim of the
          same calendar day (kept alongside);
        * a ``correction`` supersedes every earlier claim of the subject;
        * a ``conditional`` is recorded, never supersedes, is never superseded,
          and never becomes the current value of a subject.
        A claim whose id is already present is ignored (resume-safe).
        """
        if not claim.id:
            claim.id = make_claim_id(claim.subject_key, claim.kind, claim.stated_at,
                                     claim.value, claim.quote)
        if any(c.id == claim.id for c in self.claims):
            return []
        existing = list(self.claims)
        self.claims.append(claim)
        if claim.modality == "conditional":
            return []
        ts = _dt(claim.stated_at)
        if ts is None:
            return []
        superseded: list[str] = []
        for old in existing:
            if (old.superseded_by or old.modality == "conditional"
                    or old.subject_key != claim.subject_key
                    or old.unit_norm != claim.unit_norm
                    or old.scope != claim.scope
                    or old.occurrence_date != claim.occurrence_date):
                continue
            if claim.modality != "correction" and old.kind != claim.kind:
                continue
            od = _dt(old.stated_at)
            if od is None or od > ts:
                continue
            if (claim.modality == "estimate" and old.modality in ("asserted", "correction")
                    and od.date() == ts.date()):
                continue
            old.superseded_by = claim.id
            superseded.append(old.id)
        # Out-of-order add: a NEWER non-conditional claim already exists.
        for newer in existing:
            if (newer.subject_key != claim.subject_key or newer.kind != claim.kind
                    or newer.unit_norm != claim.unit_norm or newer.modality == "conditional"
                    or newer.scope != claim.scope
                    or newer.occurrence_date != claim.occurrence_date):
                continue
            nd = _dt(newer.stated_at)
            if nd is None or nd <= ts:
                continue
            if (newer.modality == "estimate" and claim.modality in ("asserted", "correction")
                    and nd.date() == ts.date()):
                continue
            claim.superseded_by = newer.id
            break
        return superseded

    # -- reading -----------------------------------------------------------
    def as_of(self, when: Any) -> list[LedgerClaim]:
        """Copies of the claims that existed at ``when``, with supersession as it
        stood then (a superseder stated later does not count yet)."""
        ts = _dt(when)
        if ts is None:
            return [LedgerClaim(**c.to_dict()) for c in self.claims]
        by_id = {c.id: c for c in self.claims}
        out: list[LedgerClaim] = []
        for c in self.claims:
            cd = _dt(c.stated_at)
            if cd is None or cd > ts:
                continue
            copy = LedgerClaim(**c.to_dict())
            if copy.superseded_by:
                sup = by_id.get(copy.superseded_by)
                sd = _dt(sup.stated_at) if sup else None
                if sd is None or sd > ts:
                    copy.superseded_by = None
            out.append(copy)
        return out

    @staticmethod
    def _pick_current(claims: Sequence[LedgerClaim]) -> Optional[LedgerClaim]:
        live = [c for c in claims if not c.superseded_by and c.modality != "conditional"]
        if not live:
            return None
        return max(live, key=lambda c: (_dt(c.stated_at).date(), _rank(c), _dt(c.stated_at)))

    def current(self, subject_key: str, kind: Optional[str] = None,
                as_of: Any = None, unit: Optional[str] = None,
                scope: Optional[str] = None,
                occurrence_date: Optional[str] = None) -> Optional[LedgerClaim]:
        """Current (non-superseded, non-conditional) claim of a subject, or None.

        ``kind``/``unit`` (compared as ``unit_norm``)/``scope``/``occurrence_date``
        narrow it to one subject; omitted, the newest wins."""
        pool = self.as_of(as_of) if as_of is not None else self.claims
        unit_key = unit.strip().lower() if isinstance(unit, str) else None
        return self._pick_current([c for c in pool if c.subject_key == subject_key
                                   and (kind is None or c.kind == kind)
                                   and (unit_key is None or c.unit_norm == unit_key)
                                   and (scope is None or c.scope == scope)
                                   and (occurrence_date is None or c.occurrence_date == occurrence_date)])

    def open_subjects(self, n: int = OPEN_SUBJECTS_CAP, as_of: Any = None,
                      scope: Optional[str] = None) -> list[dict]:
        """Current subjects (one row per subject_key + kind + unit + scope), newest
        first, capped at ``n``. An occurrence subject is listed by key ONCE (its
        newest occurrence), not once per date. ``scope`` narrows to one scope.

        Each entry shows the subject's current value; a subject that only has
        conditional claims is listed with ``modality == "conditional"`` (so
        the extractor can reuse its key) but its value is never "current"."""
        pool = self.as_of(as_of) if as_of is not None else list(self.claims)
        groups: dict[tuple, list[LedgerClaim]] = {}
        for c in pool:
            if scope is not None and c.scope != scope:
                continue
            groups.setdefault((c.subject_key, c.kind, c.unit_norm, c.scope), []).append(c)
        rows: list[tuple[datetime, dict]] = []
        for (key, _kind, _unit, _scope), items in groups.items():
            rep = self._pick_current(items)
            if rep is None:
                conds = [c for c in items if c.modality == "conditional"]
                if not conds:
                    continue
                rep = max(conds, key=lambda c: _dt(c.stated_at))
            rows.append((_dt(rep.stated_at), {
                "subject_key": key, "description": rep.description, "kind": rep.kind,
                "value": rep.value, "unit": rep.unit, "unit_norm": rep.unit_norm,
                "modality": rep.modality, "stated_at": rep.stated_at,
                "scope": rep.scope, "occurrence_date": rep.occurrence_date,
            }))
        rows.sort(key=lambda r: r[0], reverse=True)
        return [row for _, row in rows[:max(0, n)]]

    # -- persistence -------------------------------------------------------
    def to_payload(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "claims": [c.to_dict() for c in self.claims]}

    def save(self, path: str) -> None:
        safe_json.atomic_write_json(path, self.to_payload())

    @classmethod
    def load(cls, path: str) -> "ClaimLedger":
        """Strict load: missing file -> empty ledger; corrupt or wrong-shape ->
        quarantine + ``CorruptStoreError`` (never continue with empty state)."""
        payload = safe_json.load_critical_json(str(path), "claim_ledger")
        if payload is None:
            return cls()
        safe_json.check_schema_version(payload, current=SCHEMA_VERSION, path=str(path),
                                       label="claim_ledger")
        try:
            if not isinstance(payload, dict) or not isinstance(payload.get("claims"), list):
                raise ValueError("claim ledger payload has the wrong shape")
            return cls(LedgerClaim.from_dict(c) for c in payload["claims"])
        except ValueError as exc:
            raise safe_json.corrupt_store(str(path), "claim_ledger", exc) from exc


# ---------------------------------------------------------------------------
# Strict JSON
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def _strict_json(raw: Any, key: str) -> list:
    """Parse ``{"<key>": [...]}`` exactly; prose around JSON is rejected.

    Raises ValueError("shape=...") describing the failure."""
    if not isinstance(raw, str):
        raise ValueError("shape=not_text")
    text = raw.strip()
    if not text:
        raise ValueError("shape=empty")
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    elif text.startswith("```"):
        raise ValueError("shape=fenced_unclosed")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        shape = "prose" if text[:1] not in "{[" else (
            "output_truncated" if exc.pos >= len(text) - 2 else "invalid_json")
        raise ValueError(f"shape={shape}") from None
    if not isinstance(parsed, dict) or set(parsed) != {key}:
        raise ValueError("shape=wrong_keys")
    if not isinstance(parsed[key], list):
        raise ValueError(f"shape={key}_not_list")
    return parsed[key]


# ---------------------------------------------------------------------------
# Values: spelled numbers, normalisation, presence in a quote
# ---------------------------------------------------------------------------

_UNITS_W = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS_W = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
           "seventy": 70, "eighty": 80, "ninety": 90}
_NUMWORD_ALT = "|".join(sorted(list(_UNITS_W) + list(_TENS_W), key=len, reverse=True))
_NUMBER_RE = re.compile(
    rf"(?<![\w.])(?:\d+(?:\.\d+)?|(?:{'|'.join(_TENS_W)})[\s-](?:{'|'.join(list(_UNITS_W)[1:10])})"
    rf"|(?:{_NUMWORD_ALT}))(?![\w])", re.IGNORECASE)


def _number_value(surface: str) -> Optional[float]:
    s = surface.strip().lower()
    try:
        return float(s)
    except ValueError:
        pass
    if s in _UNITS_W:
        return float(_UNITS_W[s])
    if s in _TENS_W:
        return float(_TENS_W[s])
    parts = re.split(r"[\s-]+", s)
    if len(parts) == 2 and parts[0] in _TENS_W and parts[1] in _UNITS_W:
        return float(_TENS_W[parts[0]] + _UNITS_W[parts[1]])
    return None


def _canon_number(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.4g}"


def numbers_in_text(text: str) -> list[tuple[str, int, str]]:
    """``(canonical value, offset, surface)`` for every digit or spelled number."""
    out = []
    for m in _NUMBER_RE.finditer(text or ""):
        v = _number_value(m.group(0))
        if v is not None:
            out.append((_canon_number(v), m.start(), m.group(0)))
    return out


def normalize_value(kind: str, value: Any) -> Optional[str]:
    """Canonical string form of a claimed value, or None when it is unusable."""
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    s = str(value).strip()
    if kind in ("count", "duration"):
        v = _number_value(s)
        return None if v is None else _canon_number(v)
    if kind in ("date", "deadline"):
        if not _ISO_RE.match(s):
            return None
        try:
            datetime.strptime(s, "%Y-%m-%d")
        except ValueError:
            return None
        return s
    if kind == "status":
        s = s.lower().replace(" ", "_").replace("-", "_")
        return s if s in STATUS_VALUES else None
    return None


_DATE_PAIR_RE = re.compile(
    r"(?=(?P<pair>[A-Za-z]+\.?\s+\d{1,2}(?:st|nd|rd|th)?\b|\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?[A-Za-z]+\b))")


def locate_value(kind: str, value: str, quote: str, said_on: Any) -> Optional[int]:
    """Offset in ``quote`` where the (normalised) value is stated, else None.

    count/duration: a digit or spelled number equal to the value. date/deadline:
    the ISO date itself, or an expression ``temporal_resolver`` resolves to it
    against ``said_on``. status: no textual anchor exists (the value is a closed
    enum), so only the exact-quote check protects it; offset 0."""
    if kind in ("count", "duration"):
        for canon, off, _surface in numbers_in_text(quote):
            if canon == value:
                return off
        return None
    if kind in ("date", "deadline"):
        idx = quote.find(value)
        if idx >= 0:
            return idx
        ref = _dt(said_on)
        if ref is None:
            return None
        if value in resolve_temporal_references(quote, ref):
            return 0
        # resolve_date_expression keys on the FIRST "<word> <day>" pair ("HW 7 is due
        # October 23" stops at "HW 7"), so offer it each candidate pair on its own.
        for m in _DATE_PAIR_RE.finditer(quote):
            iso = resolve_date_expression(m.group("pair"), ref)[0]
            if iso and (iso == value or iso[5:] == value[5:]):
                return m.start("pair")
        return None
    if kind == "status":
        return 0
    return None


# ---------------------------------------------------------------------------
# Modality cross-check (spaCy; the deployed conditional parse)
# ---------------------------------------------------------------------------

_EPISTEMIC_MODALS = frozenset({"might", "may", "could"})
_EPISTEMIC_VERBS = frozenset({"think", "guess", "believe", "reckon", "suppose"})
# Closed grammatical category: approximating degree adverbs that spaCy attaches
# to a number. Not a phrase list of examples; "only"/"exactly" are NOT hedges.
_APPROXIMATORS = frozenset({"about", "around", "approximately", "roughly", "nearly",
                            "almost", "circa", "~"})


def parse_modality(text: str, offset: int, nlp=None) -> Optional[str]:
    """``"conditional"``, ``"estimate"`` or None for the clause holding ``text[offset:]``.

    ``text`` must be whitespace-normalised. Conditional is the deployed
    ``fact_extractor._irrealis_conflict`` (obj = the suffix at ``offset``, which
    pins the position). Estimate is the drift report's epistemic rule: a
    first-person epistemic verb or an epistemic modal in the clause, an
    approximating adverb on the number, or a ``~``/``≈`` glyph."""
    from memory.fact_extractor import _get_nlp, _irrealis_conflict  # lazy import: startup-cost (spaCy model + extractor module)

    nlp = nlp or _get_nlp()
    if nlp is None or not text:
        return None
    suffix = text[offset:]
    if _irrealis_conflict(text, suffix, nlp=nlp):
        return "conditional"
    try:
        doc = nlp(text)
        tok = next((t for t in doc if t.idx <= offset < t.idx + len(t.text)), None)
        if tok is None:
            return None
        near = [tok, tok.head]
        if any(c.lower_ in _APPROXIMATORS for n in near for c in n.children):
            return "estimate"
        if any(text[i:i + 1] in ("~", "≈") for i in (offset - 1, offset - 2) if i >= 0):
            return "estimate"
        for t in tok.sent:
            if t.lower_ in _EPISTEMIC_MODALS and t.pos_ in ("AUX", "VERB"):
                return "estimate"
            if (t.lemma_ in _EPISTEMIC_VERBS and t.pos_ == "VERB"
                    and any(c.dep_ == "nsubj" and c.lower_ in ("i", "we") for c in t.children)):
                return "estimate"
    except Exception:  # degrades: hedge cross-check skipped, the model's modality stands
        return None
    return None


# ---------------------------------------------------------------------------
# Quote normalisation (A6): the exact-span check is exact up to this fold
# ---------------------------------------------------------------------------

_FOLD_CHARS = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
    "\u2026": "...",
}


def fold_text(text: str) -> tuple[str, str, list[int]]:
    """``(base, folded, index_map)``.

    ``base`` is ``normalize_ws(text)``. ``folded`` additionally folds curly
    quotes/apostrophes, en/em dashes, the ellipsis character and case.
    ``index_map[i]`` is the ``base`` index the folded char ``i`` came from, so a
    span found in the folded text maps back to original-case text. Nothing else is
    relaxed: no fuzzy matching."""
    base = normalize_ws(text or "")
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(base):
        rep = _FOLD_CHARS.get(ch, ch).lower()
        out.append(rep)
        idx.extend([i] * len(rep))
    return base, "".join(out), idx


def locate_quote(text: str, quote: str) -> Optional[tuple[str, int, str]]:
    """``(base_text, start, original_case_quote)`` when ``quote`` is an exact
    span of ``text`` up to ``fold_text``; else None."""
    base, folded, idx = fold_text(text)
    fq = fold_text(quote)[1]
    if not fq:
        return None
    pos = folded.find(fq)
    if pos < 0:
        return None
    start = idx[pos]
    end = idx[pos + len(fq) - 1] + 1
    return base, start, base[start:end]


# ---------------------------------------------------------------------------
# Number role (A1) and unit position (A1/A4), reply realisation (A5): spaCy structure
# ---------------------------------------------------------------------------

_unit_cache: dict[str, str] = {}


def _spacy(nlp=None):
    if nlp is not None:
        return nlp
    from memory.fact_extractor import _get_nlp  # lazy import: startup-cost (spaCy model + extractor module)
    return _get_nlp()


def norm_unit(unit: str, nlp=None) -> str:
    """Lowercased spaCy lemma of the unit's head noun ("Days" -> "day"). No synonym
    table: a different lemma is a different subject. Without spaCy: the lowercased text."""
    raw = " ".join((unit or "").split()).lower()
    if not raw:
        return ""
    if raw in _unit_cache:
        return _unit_cache[raw]
    model = _spacy(nlp)
    result = raw
    if model is not None:
        # A bare plural ("assignments") parses as a proper noun and keeps its -s; framed
        # after a number it is a counted noun and lemmatises properly.
        doc = model(f"6 {raw}")
        words = [t for t in doc if t.i > 0]
        if words:
            head = next((t for t in words if t.pos_ in ("NOUN", "PROPN")), words[-1])
            result = head.lemma_.lower().strip() or raw
    if nlp is None:
        _unit_cache[raw] = result
    return result


def _number_surface(quote: str, offset: int) -> str:
    for _canon, off, surface in numbers_in_text(quote):
        if off == offset:
            return surface
    return ""


def check_number_role(quote: str, offset: int, unit_forms: Sequence[str], nlp=None) -> Optional[str]:
    """Reject reason for a count/duration figure at ``quote[offset:]``, else None.

    * ``identifier_number``: in the spaCy parse of the quote the number directly
      follows a NOUN/PROPN token and does not itself modify a following noun
      ("HW 4", "Week 5", "question 3": the number NAMES an item);
    * ``unit_not_in_quote``: no token after the number has a lemma (or text) equal
      to one of ``unit_forms``. Skipped (degrades) without spaCy."""
    model = _spacy(nlp)
    if model is None:
        return None
    length = len(_number_surface(quote, offset)) or 1
    doc = model(quote)
    toks = [t for t in doc if t.idx < offset + length and t.idx + len(t.text) > offset]
    if not toks:
        return None
    first, last = toks[0], toks[-1]
    if first.i > 0 and doc[first.i - 1].pos_ in ("NOUN", "PROPN"):
        modifies_next = last.head.i > last.i and last.head.pos_ in ("NOUN", "PROPN")
        if not modifies_next:
            return "identifier_number"
    forms = {f for f in unit_forms if f}
    end = offset + length
    if forms and not any(t.idx >= end and (t.lemma_.lower() in forms or t.lower_ in forms) for t in doc):
        return "unit_not_in_quote"
    return None


_FUTURE_MODALS = frozenset({"will", "'ll", "ll", "shall", "would", "'d", "should", "might", "may",
                            "could", "can", "must", "wo", "ca"})
_COND_MARKERS = frozenset({"if", "unless", "provided", "providing", "assuming"})


def reply_states_realized(text: str, offset: int, nlp=None) -> bool:
    """True when the clause holding ``text[offset:]`` states the figure as a
    REALIZED state or past fact: its governing verb is past/perfect, or a copula
    with no modal / ``if`` / ``to`` / future ``going to`` marker on it or on the
    verbs above it. Anything else, and any parse failure, is False (never upgrade)."""
    model = _spacy(nlp)
    if model is None or not text:
        return False
    try:
        doc = model(text)
        tok = next((t for t in doc if t.idx <= offset < t.idx + len(t.text)), None)
        if tok is None:
            return False
        chain = [tok] + list(tok.ancestors)
        verbs = [a for a in chain if a.pos_ in ("VERB", "AUX")]
        if not verbs:
            return False
        for v in verbs:
            for c in v.children:
                if c.dep_ in ("aux", "auxpass") and c.lower_ in _FUTURE_MODALS:
                    return False
                if c.dep_ == "aux" and c.lower_ == "to":
                    return False
                if c.dep_ == "mark" and c.lower_ in _COND_MARKERS:
                    return False
            if v.lemma_ == "go" and any(c.lower_ == "to" for c in v.children):
                return False
        gov = verbs[0]
        if gov.lemma_ == "be":
            return True
        past = "Past" in gov.morph.get("Tense") or any(
            c.dep_ in ("aux", "auxpass") and c.lemma_ == "have" for c in gov.children)
        return bool(past)
    except Exception:  # degrades: realisation unconfirmed, the reply is not flagged
        return False


# ---------------------------------------------------------------------------
# Input exclusions
# ---------------------------------------------------------------------------


def prepare_user_text(user_text: str) -> tuple[str, str]:
    """Pre-extraction admission: ``(skip_reason, text)``.

    Probe turns ([test] envelope) yield nothing; lyrics/poem/quote pastes yield
    nothing; pasted correspondence and [test] blocks are removed (the SAME
    chokepoint the fact extractors use -- ``prepare_fact_extraction_source``).
    The surviving text is capped to its authored head."""
    from memory.fact_source import is_test_turn  # lazy import: layering (fact_source pulls the extractor stack)
    from memory.memory_storage import prepare_fact_extraction_source  # lazy import: startup-cost (memory_storage imports the chroma/graph stack)

    raw = user_text or ""
    if not raw.strip():
        return "empty", ""
    if is_test_turn(raw):
        return "test_probe", ""
    reason, source = prepare_fact_extraction_source(raw)
    if reason:
        return reason, ""
    source = source.strip()
    if not source:
        return "empty", ""
    return "", source[:MAX_USER_CHARS]


# ---------------------------------------------------------------------------
# Extraction prompt + validator
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """You extract CHECKABLE FIGURES THE USER STATES ABOUT THEMSELVES from one message, as strict JSON.

Extract only values about the user or their own plans/progress: counts, durations, dates/deadlines, and done/not-done status. DO NOT extract course content, general facts, worked examples, anything inside quoted or pasted material, or anything Daemon (the assistant) said.

Output ONLY a JSON object, no prose, no code fence:
{"claims":[{"kind":"count|duration|date|deadline|status","subject_key":"snake_case","new_subject":true|false,"description":"short phrase","value":"...","unit":"...","modality":"asserted|estimate|conditional|correction","scope":"standing|occurrence","occurrence_date":"YYYY-MM-DD or empty","quote":"..."}]}
If nothing qualifies output {"claims":[]}.

Field rules:
- quote: copy an exact, contiguous span of the user's message (character for character) that contains the value. Never paraphrase. quote must be copied from the text between <<< and >>> only; never from the subject catalogue.
- value: count/duration = one number (digits). date/deadline = ISO YYYY-MM-DD resolved against the message date given below. status = done | not_done | planned.
- unit: REQUIRED for count/duration: the unit noun written in the quote right after the number (days, weeks, assignments...). Empty for dates and status. A figure with no unit noun after it is not a count or duration: skip it.
- Labels are not counts: a number that NAMES an item ("HW 4", "Week 5", "question 3", "chapter 2", an ordinal) is an identifier, never a count or duration. Do not extract it.
- subject_key: names WHAT is measured, not the topic (e.g. days_behind_class, days_to_catch_up, assignments_finished, problems_planned_today). One topic can hold several subjects: from "I'm 6 days behind on the reading, it will take 3 days to catch up, and I finished 4 chapters" extract days_behind_reading (6 days), days_to_catch_up (3 days) and chapters_finished (4 chapters). Two figures with different units are never the same subject. When the figure is about the SAME quantity as one of the SUBJECT CATALOGUE entries (same thing, same unit), reuse that exact key and set new_subject=false; otherwise invent a short snake_case key and set new_subject=true.
- One claim per figure: a sentence with several figures yields one entry each. From "Covered 7 weeks of the material over the last 12 days I think" extract two: material_covered (7 weeks, asserted) and days_spent_on_material (12 days, estimate).
- scope: "standing" = the figure holds until it changes (days behind, a deadline, a total, a status); "occurrence" = bound to ONE date (last night's sleep, today's session, problems done today). For an occurrence set occurrence_date to that date in ISO form resolved against the message date; for standing leave it empty. Example: "slept 5 hours last night" is an occurrence (sleep_hours, 5 hours); "I'm 6 days behind" is standing.
- modality: "conditional" when the figure sits inside an if / unless / once-or-when-prospective clause (a hypothetical or a prospect, not a present fact); "estimate" when hedged ("I think", "maybe", "like", "about", "~", "roughly"); "correction" when the user explicitly fixes an earlier value; otherwise "asserted".
- At most 12 entries."""


def _render_open_subjects(open_subjects: Sequence[Mapping[str, Any]]) -> str:
    """Key catalogue as compact JSON: ``{"key","what","unit","kind"}`` and NO value,
    no date, no ``=`` -- a value line looks like quotable message text and the
    model echoed it as its quote (trial 2: 23 dropped quotes)."""
    if not open_subjects:
        return "[]"
    rows = []
    for s in open_subjects[:OPEN_SUBJECTS_CAP]:
        row = {"key": s.get("subject_key"), "what": s.get("description"),
               "unit": s.get("unit") or "", "kind": s.get("kind") or ""}
        if s.get("scope") == "occurrence":      # standing rows keep the 4-key shape
            row["scope"] = "occurrence"
        rows.append(row)
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def build_extraction_prompt(user_text: str, said_on: Any,
                            open_subjects: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """``(system, user)`` prompts for the claim extractor."""
    when = _iso(said_on) or str(said_on)
    user = (
        f"MESSAGE DATE: {when[:10]}\n\n"
        f"SUBJECT CATALOGUE (keys only, NOT message text; reuse a key only for the same thing; "
        f"never quote from it):\n{_render_open_subjects(open_subjects)}\n\n"
        f"USER MESSAGE:\n<<<\n{(user_text or '')[:MAX_USER_CHARS]}\n>>>"
    )
    return _EXTRACT_SYSTEM, user


_ISO_IN_TEXT_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def derive_occurrence_date(quote: str, said_on: Any) -> Optional[str]:
    """ISO date an occurrence claim is bound to, or None when the quote names more
    than one distinct date (never guess between them).

    A relative word ("last night", "today", "yesterday") or a month-day phrase in
    the quote is resolved with ``utils.temporal_resolver`` against ``said_on``;
    an ISO literal counts as itself; a quote with no date phrase is bound to the
    day it was said. A bare month-day resolves future-biased in the resolver, so a
    result more than ~half a year ahead of ``said_on`` is the previous year (an
    occurrence has happened or is happening, it is not scheduled)."""
    ref = _dt(said_on)
    if ref is None:
        return None
    found: set[str] = set(_ISO_IN_TEXT_RE.findall(resolve_temporal_references(quote or "", ref)))
    for m in _DATE_PAIR_RE.finditer(quote or ""):
        iso = resolve_date_expression(m.group("pair"), ref)[0]
        if iso:
            d = datetime.strptime(iso, "%Y-%m-%d")
            if (d.date() - ref.date()).days > 183:
                d = d.replace(year=d.year - 1)
            found.add(d.strftime("%Y-%m-%d"))
    if not found:
        return ref.date().isoformat()
    return next(iter(found)) if len(found) == 1 else None


def _drop(reason: str, item: Any = None, detail: str = "") -> dict:
    quote = ""
    key = ""
    if isinstance(item, Mapping):
        quote = str(item.get("quote") or "")[:120]
        key = str(item.get("subject_key") or "")[:48]
    return {"reason": reason, "subject_key": key, "quote": quote, "detail": detail}


def validate_claims(raw_json: Any, user_text: str, said_on: Any,
                    open_subjects: Any = (), *, turn_id: str = "",
                    nlp=None) -> tuple[list[LedgerClaim], list[dict]]:
    """Deterministic admission of the model's proposed claims.

    Returns ``(claims, drops)``. ``user_text`` is the text the model was shown
    (prepared + capped). A failed claim is dropped with a reason and never
    repaired; one bad claim never discards the rest."""
    drops: list[dict] = []
    try:
        items = _strict_json(raw_json, "claims")
    except ValueError as exc:
        return [], [_drop(str(exc))]
    stated_at = _iso(said_on)
    if not stated_at:
        return [], [_drop("bad_said_on")]
    text = user_text or ""
    claims: list[LedgerClaim] = []
    seen: set[tuple] = set()
    for n_item, item in enumerate(items):
        if n_item >= MAX_CLAIMS_PER_TURN * 2:
            drops.append(_drop("too_many_claims"))
            break
        if len(claims) >= MAX_CLAIMS_PER_TURN:
            drops.append(_drop("too_many_claims", item))
            continue
        if not isinstance(item, Mapping):
            drops.append(_drop("not_an_object"))
            continue
        kind, modality = item.get("kind"), item.get("modality")
        if kind not in KINDS:
            drops.append(_drop("bad_kind", item, str(kind)[:30]))
            continue
        if modality not in MODALITIES:
            drops.append(_drop("bad_modality", item, str(modality)[:30]))
            continue
        key = item.get("subject_key")
        if not isinstance(key, str) or not _KEY_RE.match(key):
            drops.append(_drop("bad_subject_key", item))
            continue
        desc = item.get("description")
        if not isinstance(desc, str) or not desc.strip():
            drops.append(_drop("bad_description", item))
            continue
        # A missing scope is a legacy-shaped claim = standing; a PRESENT bad scope drops.
        scope = item.get("scope", "standing")
        if scope not in SCOPES:
            drops.append(_drop("bad_scope", item, str(scope)[:30]))
            continue
        quote = item.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            drops.append(_drop("quote_missing", item))
            continue
        if len(normalize_ws(quote)) > MAX_QUOTE_CHARS:
            drops.append(_drop("quote_too_long", item))
            continue
        located = locate_quote(text, quote)
        if located is None:
            drops.append(_drop("quote_not_in_text", item))
            continue
        base_text, qpos, nq = located     # whitespace-normalised text, quote start, original-case span
        value = normalize_value(kind, item.get("value"))
        if value is None:
            drops.append(_drop("bad_value", item, str(item.get("value"))[:30]))
            continue
        voff = locate_value(kind, value, nq, said_on)
        if voff is None:
            drops.append(_drop("value_not_in_quote", item, value))
            continue
        unit = item.get("unit")
        unit = " ".join(unit.split()).lower()[:30] if isinstance(unit, str) else ""
        unit_norm = ""
        if kind in ("count", "duration"):
            if not unit:
                drops.append(_drop("unit_not_in_quote", item, "empty_unit"))
                continue
            unit_norm = norm_unit(unit, nlp)
            reason = check_number_role(nq, voff, (unit_norm, unit), nlp)
            if reason:
                drops.append(_drop(reason, item, value))
                continue
        else:
            unit = ""                      # dates and status carry no unit
        occurrence_date = ""
        if scope == "occurrence":
            # The model's own date is not trusted: a date phrase in the quote is resolved
            # against said_on, else the occurrence is the day it was said.
            occurrence_date = derive_occurrence_date(nq, said_on)
            if occurrence_date is None:
                drops.append(_drop("occurrence_date_ambiguous", item))
                continue
        final_modality = modality
        parsed = parse_modality(base_text, qpos + voff, nlp=nlp)
        if parsed == "conditional":
            final_modality = "conditional"   # the parse wins; never the other way round
        sig = (key, kind, value, final_modality, nq, scope)
        if sig in seen:
            drops.append(_drop("duplicate", item))
            continue
        seen.add(sig)
        # A reused key must exist in open_subjects; an unknown key is simply a new subject.
        claims.append(LedgerClaim(
            id=make_claim_id(key, kind, stated_at, value, nq),
            subject_key=key, description=desc.strip()[:MAX_DESCRIPTION_CHARS],
            kind=kind, value=value, unit=unit, modality=final_modality,
            stated_at=stated_at, quote=nq, turn_id=str(turn_id or ""),
            unit_norm=unit_norm, scope=scope, occurrence_date=occurrence_date,
        ))
    return claims, drops


# ---------------------------------------------------------------------------
# Reply check
# ---------------------------------------------------------------------------

_REPLY_SYSTEM = """You compare an ASSISTANT reply with figures the USER stated about themselves, as strict JSON.

You are given a SUBJECT CATALOGUE (key, what, unit, kind; no values). Report only statements in the reply that restate one of THESE subjects with a value. Ignore every other figure (course content, web facts, examples, figures not about a listed subject).

Output ONLY a JSON object, no prose, no code fence:
{"findings":[{"subject_key":"one of the listed keys","kind":"count|duration|date|deadline|status","value":"...","unit":"...","modality":"asserted|estimate|conditional|plan_discussion","quote":"..."}]}
If nothing qualifies output {"findings":[]}.

Rules:
- quote: an exact, contiguous span of the REPLY (character for character) that contains the value. quote must be copied from the text between <<< and >>> only; never from the subject catalogue.
- value: count/duration = one number (digits); date/deadline = ISO YYYY-MM-DD resolved against the reply date; status = done | not_done | planned.
- unit: the unit noun written in the quote right after the number; the figure must be about the listed subject's unit (a count of assignments is not a count of days).
- A number that NAMES an item ("HW 4", "Week 5", "question 3") is an identifier, not a figure: do not report it.
- modality is how the REPLY states it: "asserted" = plain fact about the present or past; "estimate" = hedged (roughly, about, I think); "conditional" = framed as an if/when hypothetical; "plan_discussion" = the reply talks about the user's plan, target or a what-if ("20 problems today is a huge lift", "finishing it today would put you...") rather than stating the figure as the user's actual state.
- Report the reply's value even if it is wrong; do not correct it."""


def build_reply_check_prompt(reply_text: str, said_at: Any,
                             current_claims: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """``(system, user)`` prompts for the reply check; ``current_claims`` are the
    ``ClaimLedger.open_subjects`` rows as of ``said_at``."""
    when = _iso(said_at) or str(said_at)
    user = (
        f"REPLY DATE: {when[:10]}\n\n"
        f"SUBJECT CATALOGUE (keys only, NOT reply text; never quote from it):\n"
        f"{_render_open_subjects(current_claims)}\n\n"
        f"ASSISTANT REPLY:\n<<<\n{(reply_text or '')[:MAX_REPLY_CHARS]}\n>>>"
    )
    return _REPLY_SYSTEM, user


@dataclass
class ReplyFinding:
    classification: str            # one of CLASSES
    subject_key: str
    kind: str
    quote: str
    value: str                     # the reply's value (normalised)
    unit: str
    reply_modality: str            # after the hedge cross-check
    claim_id: str
    claim_value: str
    claim_modality: str
    claim_stated_at: str
    matched_superseded: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def validate_reply_findings(raw_json: Any, reply_text: str, said_at: Any,
                            ledger: ClaimLedger, *, nlp=None) -> tuple[list[ReplyFinding], list[dict]]:
    """Admit the model's findings deterministically, then classify each against
    the ledger AS OF ``said_at``. Pure; returns ``(findings, drops)``."""
    try:
        items = _strict_json(raw_json, "findings")
    except ValueError as exc:
        return [], [_drop(str(exc))]
    if _dt(said_at) is None:
        return [], [_drop("bad_said_at")]
    text = reply_text or ""
    asof = ledger.as_of(said_at)
    findings: list[ReplyFinding] = []
    drops: list[dict] = []
    for n_item, item in enumerate(items):
        if n_item >= MAX_CLAIMS_PER_TURN * 2:
            drops.append(_drop("too_many_findings"))
            break
        if not isinstance(item, Mapping):
            drops.append(_drop("not_an_object"))
            continue
        kind, modality = item.get("kind"), item.get("modality")
        key, quote = item.get("subject_key"), item.get("quote")
        problem = None
        if kind not in KINDS:
            problem = "bad_kind"
        elif modality not in REPLY_MODALITIES:
            problem = "bad_modality"
        elif not isinstance(key, str) or not isinstance(quote, str) or not quote.strip():
            problem = "bad_fields"
        if problem:
            drops.append(_drop(problem, item))
            continue
        # Standing claims only: an occurrence claim (last night's sleep) is never compared with a reply.
        sub = [c for c in asof if c.subject_key == key and c.scope == "standing"]
        if not sub:
            drops.append(_drop("unknown_subject", item))
            continue
        sub = [c for c in sub if c.kind == kind]
        if not sub:
            drops.append(_drop("kind_mismatch", item))
            continue
        if len(normalize_ws(quote)) > MAX_QUOTE_CHARS:
            drops.append(_drop("quote_not_in_reply", item))
            continue
        located = locate_quote(text, quote)
        if located is None:
            drops.append(_drop("quote_not_in_reply", item))
            continue
        base_text, qpos, nq = located
        value = normalize_value(kind, item.get("value"))
        voff = locate_value(kind, value, nq, said_at) if value is not None else None
        if value is None or voff is None:
            drops.append(_drop("value_not_in_quote", item))
            continue
        if modality == "plan_discussion":
            drops.append(_drop("plan_discussion", item))      # the reply discusses a plan: never a flag
            continue
        if kind in ("count", "duration"):
            # A4: the reply's figure must carry THE claim's unit; groups of other units are other subjects.
            units = {}
            for c in sub:
                units.setdefault(c.unit_norm, []).append(c)
            matched_unit, role = None, None
            for un, group in units.items():
                forms = {un, group[0].unit.strip().lower(), norm_unit(un, nlp)}
                role = check_number_role(nq, voff, tuple(forms), nlp)
                if role == "identifier_number":
                    break
                if role is None:
                    matched_unit = un
                    break
            if role == "identifier_number":
                drops.append(_drop("identifier_number", item))
                continue
            if matched_unit is None:
                drops.append(_drop("unit_mismatch", item))
                continue
            sub = units[matched_unit]
        reply_mod = modality
        parsed = parse_modality(base_text, qpos + voff, nlp=nlp)
        if parsed == "conditional":
            reply_mod = "conditional"
        elif parsed == "estimate" and reply_mod == "asserted":
            reply_mod = "estimate"           # downgrade only; never upgrade
        verdict = _classify(value, reply_mod, sub,
                            lambda: reply_states_realized(base_text, qpos + voff, nlp))
        if verdict is None:
            drops.append(_drop("no_comparable_value", item))
            continue
        cls, ref, matched_sup = verdict
        if cls == "dropped_hedged_ok":
            drops.append(_drop("conditional_stated_as_conditional", item))
            continue
        if cls == "dropped_not_realized":
            drops.append(_drop("not_realized", item))
            continue
        unit = item.get("unit")
        findings.append(ReplyFinding(
            classification=cls, subject_key=key, kind=kind, quote=nq, value=value,
            unit=unit.strip().lower()[:30] if isinstance(unit, str) else "",
            reply_modality=reply_mod, claim_id=ref.id, claim_value=ref.value,
            claim_modality=ref.modality, claim_stated_at=ref.stated_at,
            matched_superseded=matched_sup,
        ))
    return findings, drops


def _classify(value: str, reply_mod: str, sub: Sequence[LedgerClaim],
              realized: Callable[[], bool]):
    """``(class, reference claim, matched_superseded)`` or None / a drop marker.

    ``conditional_as_fact`` (the ledger claim is conditional) and
    ``estimate_as_fact`` (it is an estimate) need the reply to state the value as
    a REALIZED state or past fact (``realized()``, the spaCy cross-check); a
    modal / if / future reply is never flagged."""
    cur = ClaimLedger._pick_current(sub)
    conds = [c for c in sub if c.modality == "conditional"]
    if cur is not None and cur.value == value:
        if reply_mod == "asserted" and cur.modality == "estimate" and realized():
            return "estimate_as_fact", cur, False
        return "matches_current", cur, False
    cond_hit = next((c for c in conds if c.value == value), None)
    if cond_hit is not None:
        if reply_mod == "conditional":
            return "dropped_hedged_ok", cond_hit, False
        if not realized():
            return "dropped_not_realized", cond_hit, False
        return "conditional_as_fact", cond_hit, False
    if cur is None:
        return None
    superseded_hit = next((c for c in sub if c.superseded_by and c.modality != "conditional"
                           and c.value == value), None)
    if cur.modality == "estimate" and superseded_hit is None:
        # contradicting a hedged figure is not the signal contradicting an asserted one is
        return "differs_from_estimate", cur, False
    return "stale", cur, superseded_hit is not None


# ---------------------------------------------------------------------------
# Render: the user's own figures, for the prompt (pure, no LLM)
# ---------------------------------------------------------------------------

MAX_RENDER_QUOTE_CHARS = 90
MAX_EARLIER_VALUES = 2


def _fmt_day(value: Any) -> str:
    """Weekday + MM-DD ("Fri 10-17"): a bare date read on another day is a BC-55 hazard."""
    dt = _dt(value)
    return dt.strftime("%a %m-%d") if dt else "undated"


def _fmt_figure(c: LedgerClaim) -> str:
    if c.kind in ("count", "duration"):
        return f"{c.value} {c.unit}".strip()
    if c.kind == "status":
        return c.value.replace("_", " ")
    return c.value


def _render_quote(text: str) -> str:
    one = " ".join(strip_machinery(text or "").split()).replace('"', "'")
    if len(one) > MAX_RENDER_QUOTE_CHARS:
        one = one[:MAX_RENDER_QUOTE_CHARS - 1].rstrip() + "…"
    return one


def _earlier_clause(items: Sequence[LedgerClaim], rep: LedgerClaim, *, always: bool) -> str:
    older = sorted((c for c in items if c.superseded_by and c.modality != "conditional"
                    and c.id != rep.id), key=lambda c: _dt(c.stated_at), reverse=True)
    older = older[:MAX_EARLIER_VALUES]
    if not older:
        return "; earlier: none" if always else ""
    return "; earlier: " + ", ".join(f"{_fmt_figure(c)} ({_fmt_day(c.stated_at)})" for c in older)


def _render_line(rep: LedgerClaim, items: Sequence[LedgerClaim]) -> str:
    label = " ".join(rep.description.split()) or rep.subject_key
    figure = _fmt_figure(rep)
    said = _fmt_day(rep.stated_at)
    quote = _render_quote(rep.quote)
    only = f"; applies to {_fmt_day(rep.occurrence_date)} only" if rep.scope == "occurrence" else ""
    if rep.modality == "conditional":
        return (f'- {label}: {figure} — CONDITIONAL, you said "{quote}" ({said}); not a fact{only}')
    if rep.modality == "estimate":
        return (f'- {label}: ~{figure} — your ESTIMATE ("{quote}", {said})'
                f'{_earlier_clause(items, rep, always=False)}{only}')
    return (f'- {label}: {figure} — you said "{quote}" ({said})'
            f'{_earlier_clause(items, rep, always=True)}{only}')


def render_stated_figures(ledger: ClaimLedger, now: Any, *, max_lines: int,
                          standing_window_days: int, deadline_window_days: int) -> list[str]:
    """One deterministic line per subject of the user's own stated figures, newest first.

    Selected (no LLM, no query matching):
      * standing subjects whose newest statement is within ``standing_window_days``;
      * date/deadline subjects whose VALUE lies within +-``deadline_window_days`` of ``now``;
      * occurrence subjects whose occurrence_date is today or yesterday.
    A conditional-only subject is shown as CONDITIONAL ("not a fact"); an estimate
    keeps its hedge; asserted lines carry up to two ``earlier:`` values. Quotes are
    cut to 90 characters and pass through ``strip_machinery``. Dates are
    weekday + MM-DD. Returns [] for an unusable ``now`` or ``max_lines <= 0``."""
    ref = _dt(now)
    if ref is None or max_lines <= 0:
        return []
    today = ref.date()
    yesterday = today - timedelta(days=1)
    groups: dict[tuple, list[LedgerClaim]] = {}
    for c in ledger.claims:
        groups.setdefault((c.subject_key, c.kind, c.unit_norm, c.scope,
                           c.occurrence_date if c.scope == "occurrence" else ""), []).append(c)
    picked: list[tuple[datetime, LedgerClaim, list[LedgerClaim]]] = []
    for (_key, kind, _unit, scope, occ), items in groups.items():
        rep = ClaimLedger._pick_current(items)
        if rep is None:
            conds = [c for c in items if c.modality == "conditional"]
            if not conds:
                continue
            rep = max(conds, key=lambda c: _dt(c.stated_at))
        said = _dt(rep.stated_at)
        if said is None:
            continue
        if scope == "occurrence":
            keep = occ in (today.isoformat(), yesterday.isoformat())
        else:
            keep = 0 <= (ref - said).total_seconds() <= standing_window_days * 86400
            if not keep and kind in ("date", "deadline") and rep.modality != "conditional":
                vd = _dt(rep.value)
                keep = vd is not None and abs((vd.date() - today).days) <= deadline_window_days
        if keep:
            picked.append((said, rep, items))
    picked.sort(key=lambda r: (r[0], r[1].subject_key), reverse=True)
    return [_render_line(rep, items) for _said, rep, items in picked[:max_lines]]


# ---------------------------------------------------------------------------
# Async orchestration helpers (the model call is injected)
# ---------------------------------------------------------------------------

EXTRACT_MAX_TOKENS = 500    # one call shape for the live write AND the offline scripts (not forked)
REPLY_MAX_TOKENS = 400


async def generate_for_ledger(model_manager: Any, model: str, system: str, user: str,
                              max_tokens: int) -> str:
    """THE deployed ``ModelManager.generate_once`` call the ledger makes (live write,
    baseline, backfill): temperature 0, small ``max_tokens``, reasoning OFF -- on a
    reasoning model a small budget is otherwise spent on thinking and the JSON never
    arrives (BC-89). Returns the raw text; a provider failure arrives as an
    API-error sentinel string, which validation drops."""
    return await model_manager.generate_once(
        user, model_name=model, system_prompt=system, max_tokens=max_tokens,
        temperature=0.0, disable_reasoning=True)


ModelCall = Callable[[str, str], Awaitable[str]]


async def extract_claims(user_text: str, said_on: Any, ledger: ClaimLedger,
                         call_model: ModelCall, *, turn_id: str = "",
                         nlp=None) -> tuple[list[LedgerClaim], list[dict], str]:
    """Prepare -> prompt -> model -> validate. Returns ``(claims, drops, skip_reason)``.

    ``call_model(system, user)`` returns the raw model text. The caller adds the
    claims to the ledger (so a failed add never half-applies)."""
    reason, text = prepare_user_text(user_text)
    if reason:
        return [], [], reason
    subjects = ledger.open_subjects(OPEN_SUBJECTS_CAP, as_of=said_on)
    system, user = build_extraction_prompt(text, said_on, subjects)
    raw = await call_model(system, user)
    claims, drops = validate_claims(raw, text, said_on, subjects, turn_id=turn_id, nlp=nlp)
    return claims, drops, ""


async def check_reply(reply_text: str, said_at: Any, ledger: ClaimLedger,
                      call_model: ModelCall, *, nlp=None) -> tuple[list[ReplyFinding], list[dict], str]:
    """Reply check against the ledger as of ``said_at``; skipped when the ledger
    has no subject at that time. Returns ``(findings, drops, skip_reason)``."""
    text = (reply_text or "")[:MAX_REPLY_CHARS]
    if not text.strip():
        return [], [], "empty"
    subjects = ledger.open_subjects(OPEN_SUBJECTS_CAP, as_of=said_at, scope="standing")
    if not subjects:
        return [], [], "no_subjects"
    system, user = build_reply_check_prompt(text, said_at, subjects)
    raw = await call_model(system, user)
    findings, drops = validate_reply_findings(raw, text, said_at, ledger, nlp=nlp)
    return findings, drops, ""
