# utils/date_coerce.py
"""
Module Contract
- Purpose: shared `coerce_date` helper extracted 2026-09-16 (compaction) from
  the identical `_coerce_date` bodies duplicated in
  utils/completed_plan_claims.py and utils/streak_claims.py.
- Inputs:
  - coerce_date(value) -> Optional[date]  (accepts datetime/date/ISO string;
    any other shape, or an unparseable string, returns None)
  - to_naive_local(value) -> datetime  (2026-09-20: project convention is naive
    LOCAL time; an aware datetime is CONVERTED to local, then stripped)
- Dependencies: none beyond stdlib.
- Side effects: none; pure functions.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional


def coerce_date(value: Any) -> Optional[date]:
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


def to_naive_local(value: datetime) -> datetime:
    """Project convention is naive LOCAL time. An aware datetime is CONVERTED to local, then stripped
    (never a bare .replace(tzinfo=None), which misreads a UTC value by the UTC offset)."""
    return value.astimezone().replace(tzinfo=None) if value.tzinfo is not None else value
