# utils/date_coerce.py
"""
Module Contract
- Purpose: shared `coerce_date` helper extracted 2026-09-16 (compaction) from
  the identical `_coerce_date` bodies duplicated in
  utils/completed_plan_claims.py and utils/streak_claims.py.
- Inputs:
  - coerce_date(value) -> Optional[date]  (accepts datetime/date/ISO string;
    any other shape, or an unparseable string, returns None)
- Dependencies: none beyond stdlib.
- Side effects: none; pure function.
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
