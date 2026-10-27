"""Dated day-count anchors: store + prompt supply (2026-10-27; class: BC-103, BC-100, BC-75, BC-52).

Incident 2026-09-11..20: a note titled "... Day 23" was read as TODAY's count and echoed
forward (Daemon replies, shutdown summaries, "24 days" a week later) because no input
carried the start date. The user's own day counts ("23 days into X", "day 23 of X", a
dated "Day 23" note) are kept here with the date they were said; the prompt gets the count
already projected to today (``utils.streak_claims`` does the arithmetic).

Module Contract
- Writer: ONLY the post-response hook (``record_from_user_text``) -- the user's own text of
  a completed, non-[test] turn. The prompt build only READS (``supply_lines``) and extracts
  in memory from the notes it retrieved this turn (never persisted, never assistant text).
- Store: ``data/day_count_anchors.json`` via utils/safe_json (atomic write, strict load,
  ``schema_version``). A corrupt file raises CorruptStoreError; callers on the prompt path
  degrade to "no block" and never overwrite it.
- Config: ``day_counts.*`` read live (BC-82) as ``app_config.DAY_COUNTS_*``.
"""
from __future__ import annotations

import re
import threading
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional

import config.app_config as app_config
from utils import bootstrap, safe_json
from utils.date_coerce import coerce_date
from utils.logging_utils import get_logger
from utils.streak_claims import (
    DayCountClaim, day_count_lines, episode_anchors, extract_day_count_claims,
)

logger = get_logger("day_count_anchors")

SCHEMA_VERSION = 1
_LABEL = "day-count anchor store"
# Cheap gate before any parsing: a day number or an "N days" count.
_PREFILTER_RE = re.compile(r"\bday\s+(?:\d{1,3}|[a-z]{3,9})\b|\b(?:\d{1,3}|[a-z]{3,9})(?:\s*-\s*|\s+)days?\b", re.I)
# A paste is not the user's own count (mirrors the 08-29 paste guard).
MESSAGE_MAX_CHARS = 1500
_lock = threading.RLock()
_claims: Optional[List[DayCountClaim]] = None
_loaded_path: Optional[str] = None
_PATH_OVERRIDE: Optional[str] = None
_NLP_UNSET = object()
_nlp: Any = _NLP_UNSET


def store_path() -> str:
    return _PATH_OVERRIDE or bootstrap.store_path(app_config.DAY_COUNTS_STORE_PATH)


def _get_nlp() -> Any:
    """The shared spaCy pipeline when already loadable; None otherwise (stem fallback)."""
    global _nlp
    if _nlp is _NLP_UNSET:
        try:
            from memory.fact_extractor import _get_nlp as _fx_nlp  # lazy import: layering (extractor stack)
            _nlp = _fx_nlp()
        except Exception:  # degrades: activity lemmas fall back to the stem heuristic
            _nlp = None
    return _nlp


def _load() -> List[DayCountClaim]:
    """Process-wide claim list, strictly loaded once per path. Raises CorruptStoreError."""
    global _claims, _loaded_path
    path = store_path()
    with _lock:
        if _claims is None or _loaded_path != path:
            raw = safe_json.load_critical_json(path, _LABEL)
            claims: List[DayCountClaim] = []
            if isinstance(raw, dict):
                safe_json.check_schema_version(raw, current=SCHEMA_VERSION, path=path, label=_LABEL)
                for d in raw.get("claims") or []:
                    c = DayCountClaim.from_dict(d) if isinstance(d, dict) else None
                    if c is not None:
                        claims.append(c)
            elif raw is not None:
                raise safe_json.corrupt_store(path, _LABEL, ValueError("top-level value is not an object"))
            _claims, _loaded_path = claims, path
        return _claims


def _save(claims: List[DayCountClaim]) -> None:
    safe_json.atomic_write_json(
        store_path(), {"schema_version": SCHEMA_VERSION, "claims": [c.to_dict() for c in claims]})


def _identity(c: DayCountClaim) -> tuple:
    return (c.count, c.stated_on, tuple(sorted(c.lemmas)), c.source)


def record_from_user_text(text: str, stated_on: Any = None, *, source: str = "message") -> int:
    """Extract the user's day counts from ONE user-authored text and persist new ones.
    Returns the number added; 0 when disabled / nothing found. May raise CorruptStoreError."""
    if not app_config.DAY_COUNTS_ENABLED or not isinstance(text, str):
        return 0
    if len(text) > MESSAGE_MAX_CHARS or not _PREFILTER_RE.search(text):
        return 0
    day = coerce_date(stated_on) or date.today()
    found = extract_day_count_claims(text, day, source=source, nlp=_get_nlp())
    if not found:
        return 0
    with _lock:
        claims = _load()
        have = {_identity(c) for c in claims}
        new = [c for c in found if _identity(c) not in have]
        if not new:
            return 0
        horizon = app_config.DAY_COUNTS_MAX_AGE_DAYS
        kept = [c for c in claims if (day - c.stated_on).days <= horizon] + new
        _save(kept)
        global _claims
        _claims = kept
        return len(new)


def note_claims(notes: Iterable[Any]) -> List[DayCountClaim]:
    """In-memory claims from the notes RETRIEVED this turn. User-authored, content-dated notes
    only: a Daemon-generated summary or an index-time-dated chunk is never a source."""
    out: List[DayCountClaim] = []
    nlp = None
    for note in notes or []:
        if not isinstance(note, dict):
            continue
        meta = note.get("metadata") or {}
        if str(meta.get("author", "")).lower() == "daemon" or meta.get("source_type") == "daemon_daily_summary":
            continue
        if meta.get("date_basis") != "content":
            continue
        day = coerce_date(meta.get("note_date"))
        if day is None:
            continue
        title = str(meta.get("title") or "")
        body = str(note.get("content") or "")
        if not (_PREFILTER_RE.search(title) or _PREFILTER_RE.search(body)):
            continue
        nlp = nlp or _get_nlp()
        out += extract_day_count_claims(title, day, source="note", allow_bare=True, nlp=nlp)
        out += extract_day_count_claims(body[:4000], day, source="note", allow_bare=False, nlp=nlp)
    return out


def supply_lines(relevant_text: str, notes: Iterable[Any] = (), now: Optional[datetime] = None) -> List[str]:
    """The ``[YOUR DAY COUNTS]`` lines for this turn ([] when disabled, nothing relevant, or the
    store is unreadable -- a prompt build must never fail on this)."""
    if not app_config.DAY_COUNTS_ENABLED or not relevant_text:
        return []
    try:
        claims = list(_load()) + note_claims(notes)
        if not claims:
            return []
        as_of = (now or datetime.now()).date()
        return day_count_lines(
            episode_anchors(claims), as_of, relevant_text,
            max_lines=app_config.DAY_COUNTS_MAX_LINES,
            max_age_days=app_config.DAY_COUNTS_MAX_AGE_DAYS, nlp=_get_nlp())
    except Exception as exc:  # degrades: no [YOUR DAY COUNTS] block this turn
        logger.warning("[DayCounts] supply failed: %s", type(exc).__name__)
        return []


def reset_for_tests(path: Optional[str] = None) -> None:
    global _claims, _loaded_path, _PATH_OVERRIDE
    with _lock:
        _claims = _loaded_path = None
        _PATH_OVERRIDE = path
