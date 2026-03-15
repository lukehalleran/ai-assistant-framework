"""
Resolve relative temporal expressions to absolute dates in fact values.

When users say "I work tomorrow", the fact extractor stores "work tomorrow"
which becomes stale/incorrect as time passes. This module resolves relative
references to absolute dates at storage time.

Example:
    resolve_temporal_references("work tomorrow and the following day",
                                reference_date=datetime(2026, 3, 12))
    # → "work on Thu 2026-03-13 and Fri 2026-03-14"
"""

import re
from datetime import datetime, timedelta
from typing import Optional

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
    if not text or not has_temporal_reference(text):
        return text

    if reference_date is None:
        reference_date = datetime.now()

    ref = reference_date.replace(hour=0, minute=0, second=0, microsecond=0)
    result = text

    # --- Order matters: resolve "tomorrow" before "the following day" ---

    has_tomorrow = bool(re.search(r'\btomorrow\b', result, re.IGNORECASE))

    # "tomorrow"
    if has_tomorrow:
        target = ref + timedelta(days=1)
        result = re.sub(r'\btomorrow\b', _fmt(target), result, flags=re.IGNORECASE)

    # "the following day" / "the day after"  —  contextual:
    #   if "tomorrow" was present → day after tomorrow (+2)
    #   otherwise → day after today (+1)
    if re.search(r'\bthe\s+following\s+day\b', result, re.IGNORECASE) or \
       re.search(r'\bthe\s+day\s+after\b', result, re.IGNORECASE):
        offset = 2 if has_tomorrow else 1
        target = ref + timedelta(days=offset)
        result = re.sub(r'\bthe\s+following\s+day\b', _fmt(target), result, flags=re.IGNORECASE)
        result = re.sub(r'\bthe\s+day\s+after\b', _fmt(target), result, flags=re.IGNORECASE)

    # "today" / "tonight"
    result = re.sub(r'\b(?:today|tonight)\b', _fmt(ref), result, flags=re.IGNORECASE)

    # "yesterday"
    if re.search(r'\byesterday\b', result, re.IGNORECASE):
        target = ref - timedelta(days=1)
        result = re.sub(r'\byesterday\b', _fmt(target), result, flags=re.IGNORECASE)

    # "next [day_name]"
    def _replace_next_day(match):
        day_str = match.group(1).lower()
        if day_str in DAY_NAMES:
            target = _next_weekday(ref, DAY_NAMES[day_str])
            return _fmt(target)
        return match.group(0)

    day_names_pattern = '|'.join(DAY_NAMES.keys())
    result = re.sub(
        rf'\bnext\s+({day_names_pattern})\b',
        _replace_next_day, result, flags=re.IGNORECASE
    )

    # "next week"
    if re.search(r'\bnext\s+week\b', result, re.IGNORECASE):
        target = ref + timedelta(days=7)
        result = re.sub(r'\bnext\s+week\b', f"week of {_fmt(target)}", result, flags=re.IGNORECASE)

    # "next month"
    if re.search(r'\bnext\s+month\b', result, re.IGNORECASE):
        # Approximate: +30 days
        target = ref + timedelta(days=30)
        result = re.sub(r'\bnext\s+month\b', target.strftime("%B %Y"), result, flags=re.IGNORECASE)

    # "in N days"
    def _replace_in_n_days(match):
        n = int(match.group(1))
        target = ref + timedelta(days=n)
        return _fmt(target)

    result = re.sub(r'\bin\s+(\d+)\s+days?\b', _replace_in_n_days, result, flags=re.IGNORECASE)

    # "N days from now"
    result = re.sub(r'\b(\d+)\s+days?\s+from\s+now\b', _replace_in_n_days, result, flags=re.IGNORECASE)

    if result != text:
        logger.debug(f"[TemporalResolver] Resolved: '{text}' → '{result}'")

    return result
