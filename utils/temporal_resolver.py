"""
Resolve relative temporal expressions to absolute dates in fact values,
normalize time-of-day expressions, and expand day abbreviations.

Module Contract
- Purpose: (1) Resolve relative date references ("tomorrow", "next Friday")
  to absolute dates at storage time. (2) Normalize time-of-day expressions
  ("3pm", "15:00", bare "3") to HH:MM format with confidence tracking.
  (3) Expand day abbreviations ("MWF", "TTh") to full day name lists.
  (4) Resolve date expressions ("April 15th") to ISO dates.
- Inputs:
  - resolve_temporal_references(text, reference_date) -> str
  - normalize_time_range(start_text, end_text, context) -> (start, end, basis, confidence)
  - expand_day_abbreviations(text) -> list[str]
  - resolve_date_expression(text, reference_date) -> (iso_date, basis, confidence)
  - has_temporal_reference(text) -> bool
- Outputs: Resolved strings, normalized HH:MM times, ISO dates, confidence floats
- Key behaviors:
  - Three-tier time resolution: explicit am/pm (0.95) → context heuristic (0.65-0.70) → ambiguous guess (0.50)
  - Date resolution is future-biased: past dates resolve to next year at lower confidence
  - Day abbreviations handle university shorthand: MWF, TTh, TR
- Side effects: None (pure functions, logging only)

Example:
    resolve_temporal_references("work tomorrow and the following day",
                                reference_date=datetime(2026, 3, 12))
    # → "work on Thu 2026-03-13 and Fri 2026-03-14"

    normalize_time_range("3", "10", context="work")
    # → ("15:00", "22:00", "work_pm_heuristic", 0.70)

    expand_day_abbreviations("MWF")
    # → ["monday", "wednesday", "friday"]
"""

import re
from datetime import datetime, timedelta
from typing import Optional

from utils.date_coerce import to_naive_local
from utils.logging_utils import get_logger

logger = get_logger("temporal_resolver")

DAY_NAMES = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "tues": 1, "wed": 2,
    "thu": 3, "thur": 3, "thurs": 3,
    "fri": 4, "sat": 5, "sun": 6,
}

# Regex to detect any relative temporal keyword in text
_TEMPORAL_KEYWORDS = re.compile(
    r'\b(?:'
    r'tomorrow|today|tonight|yesterday'
    r'|the\s+following\s+day|the\s+day\s+after'
    r'|next\s+(?:' + '|'.join(DAY_NAMES.keys()) + r'|week|month)'
    r'|in\s+\d+\s+days?'
    r'|\d+\s+days?\s+from\s+now'
    r')\b',
    re.IGNORECASE
)


def has_temporal_reference(text: str) -> bool:
    """Check if text contains relative temporal references that should be resolved."""
    if not text:
        return False
    return bool(_TEMPORAL_KEYWORDS.search(text))


def _next_weekday(ref_date: datetime, weekday: int) -> datetime:
    """Get the next occurrence of a weekday after ref_date (0=Monday)."""
    days_ahead = weekday - ref_date.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return ref_date + timedelta(days=days_ahead)


def _fmt(dt: datetime) -> str:
    """Format date as 'Thu 2026-03-13'."""
    return dt.strftime("%a %Y-%m-%d")


# An annotation this module already appended (idempotence guard for annotate mode).
_ANCHOR_BRACKET_AHEAD = (
    r'(?!\s*\[(?:week of )?(?:[A-Z][a-z]{2} \d{4}-\d{2}-\d{2}|[A-Z][a-z]+ \d{4})\])'
)


def resolve_temporal_references(text: str,
                                reference_date: Optional[datetime] = None) -> str:
    """
    Resolve relative temporal expressions to absolute dates.

    Replaces words like "tomorrow", "the following day", "next Monday"
    with concrete dates anchored to reference_date.

    Args:
        text: Text potentially containing relative temporal references.
        reference_date: Anchor date for resolution (defaults to now).

    Returns:
        Text with relative references replaced by absolute dates.
        If no temporal references found, returns text unchanged.
    """
    return _resolve(text, reference_date, annotate=False)


def anchor_temporal_references(text: str, said_on,
                               now: Optional[datetime] = None) -> str:
    """Annotate (never rewrite) the relative time words of RECALLED text.

    A stored turn said "finish HW 6 by end of day today" on a Saturday; read
    on Sunday the words stay but gain the date they meant:
    ``by end of day today [Sat 2026-10-17]``. Same-day text, text with no
    parseable ``said_on`` and text with nothing to anchor come back
    unchanged. One resolution code path (``_resolve``) decides what each
    expression means; annotate mode only changes how it is emitted.
    ``said_on`` may be a datetime or an ISO-ish string (see ``parse_said_on``).
    class: BC-100 (content-relative time words read as now-relative)
    """
    if not text or not has_temporal_reference(text):
        return text
    said_on = parse_said_on(said_on)
    if said_on is None:
        return text
    if now is None:
        now = datetime.now()
    if said_on.date() >= now.date():
        return text
    return _resolve(text, said_on, annotate=True)


def parse_said_on(value) -> Optional[datetime]:
    """Best-effort parse of a stored item's timestamp; None when unparseable."""
    if isinstance(value, datetime):
        return to_naive_local(value)
    if isinstance(value, str) and value.strip():
        try:
            return to_naive_local(
                datetime.fromisoformat(value.strip().replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def _resolve(text: str, reference_date: Optional[datetime], annotate: bool) -> str:
    if not text or not has_temporal_reference(text):
        return text

    if reference_date is None:
        reference_date = datetime.now()

    ref = reference_date.replace(hour=0, minute=0, second=0, microsecond=0)
    result = text
    guard = _ANCHOR_BRACKET_AHEAD if annotate else ''

    def _sub(pattern, render, string):
        """re.sub where ``render(match)`` is the absolute form; annotate mode
        keeps the original words and appends it in brackets."""
        def _r(m):
            absolute = render(m)
            return f"{m.group(0)} [{absolute}]" if annotate else absolute
        return re.sub(pattern + guard, _r, string, flags=re.IGNORECASE)

    # --- Order matters: resolve "tomorrow" before "the following day" ---

    has_tomorrow = bool(re.search(r'\btomorrow\b', result, re.IGNORECASE))

    # "tomorrow"
    if has_tomorrow:
        target = ref + timedelta(days=1)
        result = _sub(r'\btomorrow\b', lambda m: _fmt(target), result)

    # "the following day" / "the day after"  —  contextual:
    #   if "tomorrow" was present → day after tomorrow (+2)
    #   otherwise → day after today (+1)
    if re.search(r'\bthe\s+following\s+day\b', result, re.IGNORECASE) or \
       re.search(r'\bthe\s+day\s+after\b', result, re.IGNORECASE):
        offset = 2 if has_tomorrow else 1
        target = ref + timedelta(days=offset)
        result = _sub(r'\bthe\s+following\s+day\b', lambda m: _fmt(target), result)
        result = _sub(r'\bthe\s+day\s+after\b', lambda m: _fmt(target), result)

    # "today" / "tonight"
    result = _sub(r'\b(?:today|tonight)\b', lambda m: _fmt(ref), result)

    # "yesterday"
    if re.search(r'\byesterday\b', result, re.IGNORECASE):
        target = ref - timedelta(days=1)
        result = _sub(r'\byesterday\b', lambda m: _fmt(target), result)

    # "next [day_name]"
    def _next_day(match):
        day_str = match.group(1).lower()
        return _fmt(_next_weekday(ref, DAY_NAMES[day_str]))

    day_names_pattern = '|'.join(DAY_NAMES.keys())
    result = _sub(rf'\bnext\s+({day_names_pattern})\b', _next_day, result)

    # "next week"
    if re.search(r'\bnext\s+week\b', result, re.IGNORECASE):
        target = ref + timedelta(days=7)
        result = _sub(r'\bnext\s+week\b',
                      lambda m: f"week of {_fmt(target)}", result)

    # "next month"
    if re.search(r'\bnext\s+month\b', result, re.IGNORECASE):
        # Approximate: +30 days
        target = ref + timedelta(days=30)
        result = _sub(r'\bnext\s+month\b',
                      lambda m: target.strftime("%B %Y"), result)

    # "in N days" / "N days from now"
    def _in_n_days(match):
        return _fmt(ref + timedelta(days=int(match.group(1))))

    result = _sub(r'\bin\s+(\d+)\s+days?\b', _in_n_days, result)
    result = _sub(r'\b(\d+)\s+days?\s+from\s+now\b', _in_n_days, result)

    if result != text:
        logger.debug(f"[TemporalResolver] Resolved: '{text}' → '{result}'")

    return result


# -----------------------------------------------------------------------
# Day abbreviation expansion
# -----------------------------------------------------------------------

_DAY_ABBREV_MAP = {
    # University/common multi-day abbreviations
    "mwf": ["monday", "wednesday", "friday"],
    "mw": ["monday", "wednesday"],
    "tth": ["tuesday", "thursday"],
    "tr": ["tuesday", "thursday"],
    "tt": ["tuesday", "thursday"],
    "mtwthf": ["monday", "tuesday", "wednesday", "thursday", "friday"],
    "mtwrf": ["monday", "tuesday", "wednesday", "thursday", "friday"],
    "mtwtf": ["monday", "tuesday", "wednesday", "thursday", "friday"],
    "mf": ["monday", "friday"],
    "ss": ["saturday", "sunday"],
}


def expand_day_abbreviations(text: str) -> list:
    """Expand day abbreviations to full day name lists.

    Handles university shorthand (MWF, TTh, TR) and single day names.

    Args:
        text: Day abbreviation string (e.g. "MWF", "TTh", "Friday").

    Returns:
        List of lowercase full day names, e.g. ["monday", "wednesday", "friday"].
        Empty list if no valid day found.
    """
    if not text:
        return []
    cleaned = text.strip().lower()

    # Check multi-day abbreviation map first
    if cleaned in _DAY_ABBREV_MAP:
        return list(_DAY_ABBREV_MAP[cleaned])

    # Check single day name / abbreviation
    if cleaned in DAY_NAMES:
        # Map abbreviation to full name
        weekday_num = DAY_NAMES[cleaned]
        full_names = ["monday", "tuesday", "wednesday", "thursday",
                      "friday", "saturday", "sunday"]
        return [full_names[weekday_num]]

    # Check if it's already a full day name with trailing 's' (plural)
    if cleaned.endswith("s") and cleaned[:-1] in DAY_NAMES:
        weekday_num = DAY_NAMES[cleaned[:-1]]
        full_names = ["monday", "tuesday", "wednesday", "thursday",
                      "friday", "saturday", "sunday"]
        return [full_names[weekday_num]]

    return []


# -----------------------------------------------------------------------
# Time-of-day normalization
# -----------------------------------------------------------------------

_TIME_RE = re.compile(
    r'^(\d{1,2})'          # hour
    r'(?::(\d{2}))?'       # optional :MM
    r'\s*'
    r'(am|pm|a\.m\.|p\.m\.)?'  # optional am/pm
    r'$',
    re.IGNORECASE,
)

_NAMED_TIMES = {
    "noon": ("12:00", "explicit_named"),
    "midnight": ("00:00", "explicit_named"),
    "morning": ("08:00", "shift_heuristic"),
    "evenings": ("18:00", "shift_heuristic"),
    "evening": ("18:00", "shift_heuristic"),
    "nights": ("22:00", "shift_heuristic"),
    "night": ("22:00", "shift_heuristic"),
    "afternoons": ("13:00", "shift_heuristic"),
    "afternoon": ("13:00", "shift_heuristic"),
}


def _parse_single_time(text: str) -> tuple:
    """Parse a single time expression to (hour_int, minute_int, has_explicit_ampm).

    Returns (None, None, False) if unparseable.
    """
    if not text:
        return None, None, False
    cleaned = text.strip().lower()

    m = _TIME_RE.match(cleaned)
    if not m:
        return None, None, False

    hour = int(m.group(1))
    minute = int(m.group(2)) if m.group(2) else 0
    ampm = (m.group(3) or "").lower().replace(".", "")

    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0

    has_explicit = bool(ampm)
    return hour, minute, has_explicit


def normalize_time_range(
    start_text: str,
    end_text: str,
    context: str = "",
) -> tuple:
    """Normalize a time range to (start_hhmm, end_hhmm, resolution_basis, confidence).

    Three-tier resolution:
      - Explicit am/pm or 24h → high confidence (0.90-0.95)
      - Strong context heuristic → moderate confidence (0.65-0.70)
      - Ambiguous → low confidence with best guess (0.50)

    Args:
        start_text: Start time string (e.g. "3", "3pm", "15:00").
        end_text: End time string (e.g. "10", "10pm", "22:00").
        context: Hint for heuristic resolution: "work", "class", or "".

    Returns:
        Tuple of (start_hhmm, end_hhmm, resolution_basis, confidence).
        Returns (None, None, "unparseable", 0.0) if either time can't be parsed.
    """
    s_h, s_m, s_explicit = _parse_single_time(start_text)
    e_h, e_m, e_explicit = _parse_single_time(end_text)

    if s_h is None or e_h is None:
        return None, None, "unparseable", 0.0

    # Both have explicit am/pm or are in 24h format (hour >= 13)
    if s_explicit and e_explicit:
        basis = "explicit_ampm"
        confidence = 0.95
    elif s_h >= 13 or e_h >= 13:
        # At least one is unambiguously 24h
        basis = "explicit_24h"
        confidence = 0.95
    elif s_explicit or e_explicit:
        # One explicit, infer the other
        basis = "partial_explicit"
        confidence = 0.90
        if s_explicit and not e_explicit and e_h <= 12:
            # "3pm - 10" → 10 is probably PM too if > start
            if e_h < s_h or (s_h >= 12 and e_h < 12):
                e_h += 12
            elif s_h >= 12:
                e_h += 12
        elif e_explicit and not s_explicit and s_h <= 12:
            # "3 - 10pm" → 3 is probably PM too
            if s_h < e_h and e_h >= 12:
                s_h += 12
    else:
        # Both bare numbers — use context heuristics
        if s_h > e_h and s_h <= 12 and e_h <= 12:
            # e.g. "8-2" → likely 08:00-14:00 (AM start, PM end crossover)
            e_h += 12
            basis = "am_to_pm_crossover"
            confidence = 0.65
        elif context == "work":
            # Work context: bare numbers bias toward PM for typical shifts
            if s_h <= 6:
                s_h += 12
            if e_h <= 6:
                e_h += 12
            # Handle cases like "3-10" in work context → 15:00-22:00
            if s_h <= 12 and e_h > s_h:
                # Both could be PM
                if s_h < 7:
                    s_h += 12
                if e_h <= 12:
                    e_h += 12
            basis = "work_pm_heuristic"
            confidence = 0.70
        elif context == "class":
            # Class context: bare numbers bias toward AM
            basis = "am_context_heuristic"
            confidence = 0.70
        else:
            # Truly ambiguous — assume lower number is start, higher is end
            # Default to AM-to-PM interpretation
            if s_h < e_h:
                # Could be AM-AM or AM-PM, go with more common AM→PM
                if e_h > 6 and s_h < 7:
                    s_h += 12  # probably PM
                    e_h += 12
            basis = "ambiguous_guess"
            confidence = 0.50

    start_hhmm = f"{s_h:02d}:{s_m:02d}"
    end_hhmm = f"{e_h:02d}:{e_m:02d}"

    logger.debug(
        f"[TemporalResolver] Time range: '{start_text}'-'{end_text}' "
        f"(ctx={context}) → {start_hhmm}-{end_hhmm} "
        f"[{basis}, conf={confidence}]"
    )

    return start_hhmm, end_hhmm, basis, confidence


# -----------------------------------------------------------------------
# Date expression resolution
# -----------------------------------------------------------------------

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

_DATE_EXPR_RE = re.compile(
    r'(?:(?P<month_name>[a-zA-Z]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?)'
    r'|(?:(?P<day2>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<month_name2>[a-zA-Z]+))',
    re.IGNORECASE,
)


def resolve_date_expression(
    text: str,
    reference_date: Optional[datetime] = None,
) -> tuple:
    """Resolve a date expression to (iso_date, resolution_basis, confidence).

    Future-biased: if the date has already passed this year, resolves to
    next year at lower confidence.

    Args:
        text: Date expression like "April 15th", "15th of April", "Dec 3".
        reference_date: Anchor for year resolution (defaults to now).

    Returns:
        Tuple of (iso_date_str, resolution_basis, confidence).
        Returns (None, "unparseable", 0.0) if the expression can't be parsed.
    """
    if not text:
        return None, "unparseable", 0.0

    if reference_date is None:
        reference_date = datetime.now()

    m = _DATE_EXPR_RE.search(text)
    if not m:
        return None, "unparseable", 0.0

    month_str = m.group("month_name") or m.group("month_name2")
    day_str = m.group("day") or m.group("day2")

    if not month_str or not day_str:
        return None, "unparseable", 0.0

    month_num = _MONTH_NAMES.get(month_str.lower())
    if not month_num:
        return None, "unparseable", 0.0

    day_num = int(day_str)
    if day_num < 1 or day_num > 31:
        return None, "unparseable", 0.0

    # Try this year first
    year = reference_date.year
    try:
        candidate = datetime(year, month_num, day_num)
    except ValueError:
        return None, "unparseable", 0.0

    if candidate.date() >= reference_date.date():
        # Future or today — high confidence
        basis = "current_year_future"
        confidence = 0.90
    else:
        # Already passed — resolve to next year, lower confidence
        year += 1
        try:
            candidate = datetime(year, month_num, day_num)
        except ValueError:
            return None, "unparseable", 0.0
        basis = "next_year_rollover"
        confidence = 0.70

    iso_date = candidate.strftime("%Y-%m-%d")

    logger.debug(
        f"[TemporalResolver] Date: '{text}' → {iso_date} [{basis}, conf={confidence}]"
    )

    return iso_date, basis, confidence
