"""
# utils/attachment_audit.py

Module Contract
- Purpose: Two DETERMINISTIC, no-LLM checks run once per turn that has file
  attachments, both born from the 2026-09-04 homework-attachment turn audit
  (docs/HANDOFF_20260904_hw_attachment_turn.md, items 8-9):
    1. referenced-but-missing attachment audit — the user's own text (or an
       attached document's text) names a file ("Housing.csv") that was never
       actually attached this turn, or an attached document's own title
       signals it's one PART of a multi-part assignment. Silent otherwise.
    2. deadline timezone conversion — a pasted deadline states an explicit
       timezone different from the user's own; convert and surface the local
       time so a Homework-due-11:59-PM-Eastern trap doesn't silently read as
       the user's own Central time.
- Key public functions:
  - audit_attachments(user_text, files, documents) -> str  [missing-file /
    multi-part-title note, or "" when nothing to flag]
  - deadline_timezone_note(text, user_tz=None) -> str  [zone-converted
    deadline note, or "" when no explicit zone token / zones already match]
- Both functions are pure string analysis: no network, no LLM calls, no
  ChromaDB/store access. `deadline_timezone_note` reads the user's resolved
  timezone via utils.timezone_resolver.get_user_timezone() (profile-backed,
  itself read-only) unless a `user_tz` override is passed (tests use this to
  avoid depending on the live profile store).
- Wired into gui/handlers.py where `merged_input` is assembled (append-only:
  a short note text, never a mutation of existing content).
"""

import os
import re
from datetime import datetime
from typing import Any, Iterable, List, Optional
from zoneinfo import ZoneInfo

from utils.logging_utils import get_logger
from utils.timezone_resolver import get_user_timezone

logger = get_logger("attachment_audit")

# ---------------------------------------------------------------------------
# Item 8: referenced-but-missing attachment audit
# ---------------------------------------------------------------------------

# Recognized attachment-shaped extensions (case-insensitive at match time).
_FILENAME_EXTS = r"csv|pdf|xlsx|xls|docx|txt|json|ipynb|R|py"
_EXT_RE = re.compile(r"\.(" + _FILENAME_EXTS + r")", re.IGNORECASE)
# Contiguous identifier run (no spaces, no dots) immediately preceding a
# recognized extension — the candidate filename "stem".
_STEM_RE = re.compile(r"[A-Za-z0-9_\-]+$")
# A glued Canvas-paste fragment ("onHousing.csvHomDownload"): a lowercase
# run directly followed by a TitleCase run. Trimming the lowercase run
# recovers the real filename stem ("Housing").
_LOWER_PREFIX_RE = re.compile(r"^([a-z]+)([A-Z][A-Za-z0-9_\-]*)$")

_PART_RE = re.compile(r"\bPart\s+(\d+|[IVX]+)\b")
_HOMEWORK_RANGE_RE = re.compile(r"Homework\s*\d+\s*[-–]\s*\d+")

_MULTIPART_SCAN_CHARS = 300
_DOC_EXT_LABELS = {
    ".pdf": "PDF",
    ".docx": "document",
    ".doc": "document",
    ".txt": "text file",
    ".md": "document",
}


def _extract_filenames(text: str) -> set:
    """Filename-shaped tokens ("Name.ext") referenced anywhere in `text`.

    Two shapes are recognized:
    - clean: the extension is followed by a non-identifier character (or
      end of string) — the ordinary case.
    - glued: the extension is immediately followed by more letters (a
      Canvas copy-paste concatenation, e.g. "onHousing.csvHomDownload")
      AND the stem itself has a lowercase-run-then-TitleCase shape, in
      which case the leading lowercase run is trimmed off and the
      TitleCase remainder is used as the recovered filename. Any other
      glued shape is skipped as too ambiguous to salvage.
    """
    found = set()
    if not text:
        return found
    for m in _EXT_RE.finditer(text):
        ext = m.group(1)
        start, end = m.start(), m.end()
        stem_match = _STEM_RE.search(text[:start])
        if not stem_match:
            continue
        stem = stem_match.group(0)

        next_char = text[end:end + 1]
        glued = bool(next_char) and (next_char.isalnum() or next_char == "_")
        if glued:
            trimmed = _LOWER_PREFIX_RE.match(stem)
            if not trimmed:
                # Ambiguous glue shape (e.g. "UsedCars.csvDownload") — skip
                # rather than guess a bogus filename.
                continue
            stem = trimmed.group(2)

        if not stem:
            continue
        found.add(f"{stem}.{ext}")
    return found


def _attached_basenames(files: Optional[Iterable[Any]]) -> set:
    """True original basenames of this turn's attachments (lowercased).

    Prefers `.orig_name` (the api/state.py resolve_uploads shim carries the
    real client filename separately from the server temp path); falls back
    to `os.path.basename(.name)` for Gradio-style file objects.
    """
    names = set()
    for f in files or []:
        orig = getattr(f, "orig_name", None)
        if orig:
            name = os.path.basename(str(orig))
        else:
            name = os.path.basename(str(getattr(f, "name", "") or ""))
        if name:
            names.add(name.lower())
    return names


def _doc_label(filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    return _DOC_EXT_LABELS.get(ext, "file")


def _detect_multipart_title(content_text: str) -> Optional[str]:
    """First ~300 chars of an attached document's text: does its own title
    signal it's one part of a multi-part assignment? Returns a short title
    snippet to surface, or None."""
    if not content_text:
        return None
    head = content_text[:_MULTIPART_SCAN_CHARS]
    part_m = _PART_RE.search(head)
    hw_m = _HOMEWORK_RANGE_RE.search(head)
    if not part_m and not hw_m:
        return None

    first_line = head.strip().splitlines()[0].strip() if head.strip() else ""
    if part_m and part_m.group(0) in first_line:
        return first_line[:120]
    if part_m:
        return part_m.group(0)
    if hw_m and hw_m.group(0) in first_line:
        return first_line[:120]
    return hw_m.group(0) if hw_m else None


def audit_attachments(user_text: str, files: Optional[Iterable[Any]],
                       documents: Optional[Iterable[Any]]) -> str:
    """Referenced-but-missing attachment audit (item 8).

    Args:
        user_text: The user's own typed message (pre-attachment).
        files: The raw uploaded-file objects for THIS turn (used only to
            resolve true original filenames — see `_attached_basenames`).
        documents: `ProcessedFilesResult.documents` for this turn — objects
            exposing `.filename` and `.content_text` (utils/file_processor.py
            ProcessedFile). Text documents only (images excluded upstream).

    Returns:
        A single "[ATTACHMENT NOTE] ..." line, or "" when nothing to flag.
    """
    try:
        documents = list(documents or [])

        referenced = set()
        referenced |= _extract_filenames(user_text or "")
        for doc in documents:
            referenced |= _extract_filenames(getattr(doc, "content_text", "") or "")

        attached = _attached_basenames(files)
        missing = sorted(
            {name for name in referenced if name.lower() not in attached},
            key=str.lower,
        )

        part_notes = []
        for doc in documents:
            title = _detect_multipart_title(getattr(doc, "content_text", "") or "")
            if title:
                label = _doc_label(getattr(doc, "filename", ""))
                part_notes.append((label, title))

        if not missing and not part_notes:
            return ""

        segments = []
        if missing:
            segments.append(
                f"Pasted material references files not attached: {', '.join(missing)}."
            )
        for label, title in part_notes:
            segments.append(
                f"Attached {label} is titled '{title}'; other parts may exist."
            )
        segments.append("Ask before estimating scope.")
        return "[ATTACHMENT NOTE] " + " ".join(segments)
    except Exception as e:
        logger.debug(f"[AttachmentAudit] audit_attachments failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Item 9: deadline timezone conversion
# ---------------------------------------------------------------------------

# Zone label/abbreviation -> (IANA zone, friendly display name).
_ZONE_MAP = {
    "eastern": ("America/New_York", "Eastern"),
    "et": ("America/New_York", "Eastern"),
    "est": ("America/New_York", "Eastern"),
    "edt": ("America/New_York", "Eastern"),
    "central": ("America/Chicago", "Central"),
    "ct": ("America/Chicago", "Central"),
    "cst": ("America/Chicago", "Central"),
    "cdt": ("America/Chicago", "Central"),
    "mountain": ("America/Denver", "Mountain"),
    "mt": ("America/Denver", "Mountain"),
    "mst": ("America/Denver", "Mountain"),
    "mdt": ("America/Denver", "Mountain"),
    "pacific": ("America/Los_Angeles", "Pacific"),
    "pt": ("America/Los_Angeles", "Pacific"),
    "pst": ("America/Los_Angeles", "Pacific"),
    "pdt": ("America/Los_Angeles", "Pacific"),
    "utc": ("UTC", "UTC"),
}

_IANA_TO_DISPLAY = {
    "America/New_York": "Eastern",
    "America/Chicago": "Central",
    "America/Denver": "Mountain",
    "America/Los_Angeles": "Pacific",
    "UTC": "UTC",
}

_DEADLINE_TIME_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\s*\(?\s*"
    r"(Eastern|Central|Mountain|Pacific|ET|EST|EDT|CT|CST|CDT|MT|PT|PST|PDT|UTC)\b",
    re.IGNORECASE,
)

# A zoned time is only a DEADLINE when a deadline cue sits near it (Fable
# review, 2026-09-04): the handlers feed this function the user text PLUS every
# attached document, so a lecture transcript's "office hours are at 3 pm ET"
# would otherwise become a [DEADLINE NOTE] — never wrong > always active.
# Strong cues may sit anywhere in a short window before or after the time;
# a bare "by" counts only when it directly precedes the time (the Canvas
# "Due Sep 13 by 11:59pm" shape). The FIRST cued match wins; uncued zoned
# times are ignored entirely.
_DEADLINE_CUE_RE = re.compile(
    r"\b(?:due|deadline|submi(?:t|ts|tted|tting|ssions?)|turn(?:ed)?\s+in|"
    r"no later than|cut-?off|closes|closed|must be (?:in|received|uploaded))\b",
    re.IGNORECASE,
)
_BY_BEFORE_TIME_RE = re.compile(r"\bby\s+(?:the\s+)?$", re.IGNORECASE)
_CUE_WINDOW_BEFORE = 120
_CUE_WINDOW_AFTER = 60
# Cue windows never cross a line break or a sentence boundary (terminator +
# whitespace + capital letter — "Sep. 13" and "p.m." don't split), so a
# transcript's "…at 3 pm ET on Tuesdays.\nHomework is due…" can't borrow the
# NEXT sentence's "due" for the office-hours time.
_SENTENCE_BREAK_RE = re.compile(r"\n|[.!?]\s+(?=[A-Z])")


def _first_deadline_match(text: str):
    """First zoned-time match with a deadline cue in the SAME sentence, else None."""
    for m in _DEADLINE_TIME_RE.finditer(text):
        before = _SENTENCE_BREAK_RE.split(
            text[max(0, m.start() - _CUE_WINDOW_BEFORE):m.start()]
        )[-1]
        after = _SENTENCE_BREAK_RE.split(text[m.end():m.end() + _CUE_WINDOW_AFTER])[0]
        if (_DEADLINE_CUE_RE.search(before) or _DEADLINE_CUE_RE.search(after)
                or _BY_BEFORE_TIME_RE.search(before)):
            return m
    return None


def _format_12h(dt: datetime) -> str:
    hour = dt.hour % 12
    if hour == 0:
        hour = 12
    return f"{hour}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def deadline_timezone_note(text: str, user_tz: Optional[str] = None) -> str:
    """Deadline timezone conversion (item 9).

    Scans `text` for a time carrying an explicit timezone token (e.g.
    "11:59pm Eastern Time") WITH a deadline cue nearby (due/deadline/submit/
    "by <time>"…). When the stated zone differs from the user's own resolved
    timezone, returns a "[DEADLINE NOTE] ..." line with the converted local
    time. Returns "" when there's no cued zone token, or when the stated zone
    already matches the user's own.
    """
    try:
        if not text:
            return ""
        m = _first_deadline_match(text)
        if not m:
            return ""

        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = m.group(3).lower().replace(".", "")
        zone_token = m.group(4).lower()

        if not (1 <= hour <= 12) or not (0 <= minute <= 59):
            return ""

        zone_entry = _ZONE_MAP.get(zone_token)
        if zone_entry is None:
            return ""
        source_iana, source_display = zone_entry

        if user_tz is None:
            user_tz = get_user_timezone()

        if source_iana == user_tz:
            return ""

        hour24 = hour % 12
        if ampm.startswith("p"):
            hour24 += 12

        today = datetime.now().date()
        try:
            source_dt = datetime(
                today.year, today.month, today.day, hour24, minute,
                tzinfo=ZoneInfo(source_iana),
            )
            target_dt = source_dt.astimezone(ZoneInfo(user_tz))
        except Exception as e:
            logger.debug(f"[AttachmentAudit] zone conversion failed: {e}")
            return ""

        target_display = _IANA_TO_DISPLAY.get(user_tz, user_tz)
        source_str = _format_12h(source_dt)
        target_str = _format_12h(target_dt)

        return (
            f"[DEADLINE NOTE] {source_str} {source_display} = "
            f"{target_str} {target_display} (your timezone)."
        )
    except Exception as e:
        logger.debug(f"[AttachmentAudit] deadline_timezone_note failed: {e}")
        return ""
