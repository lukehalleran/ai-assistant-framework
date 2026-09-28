"""
# core/agentic/tool_thread.py

Module Contract
- Purpose (2026-09-27, session-audit plan E2 — BC-58, BC-74, BC-04, BC-15):
  an in-process, single-slot record of the READ tools the agentic
  controller dispatched THIS turn (email_search, web_search, search_memory,
  file tools, ...; never a write action like propose_action). The gate
  reads it on the NEXT turn to recognize a same-thread follow-up ("Search
  that", "Yes please" after an offer to run one more search, a terse
  correction of the search target) and route it back to the tool loop with
  the prior arguments carried forward — instead of falling through to
  plain chat, where the model has no memory of what it searched and
  confabulates ("must be in Outlook").
- Public interface:
  - record_read_tool_calls(calls: list[dict]) -> None
  - recent_read_tool_calls(max_age_s: float) -> list[dict]
  - reset() -> None (test hook)
- Dependencies: stdlib only (time). Deliberately a leaf module with no
  project imports — both core.agentic.gate and core.agentic.controller
  import it directly at module level with zero cycle risk.
- Side effects: module-level in-process state (single-user app; same
  one-slot pattern as gate._DEFERRED_REQUEST_SLOT). Never persisted —
  a process restart is an empty slot, same as any other in-memory
  continuation arm in this package.
"""

import time
from typing import Any, Dict, List

# Overwritten wholesale at the end of every agentic turn (never merged or
# appended across turns) — including with an empty list, so a turn that
# dispatched no trackable read tool (a computation-only turn, a forced
# write action) can never leave a STALE record for the gate to misread as
# "this turn continues that old search".
_STATE: Dict[str, Any] = {"calls": [], "ts": 0.0}


def record_read_tool_calls(calls: List[Dict[str, Any]]) -> None:
    """Overwrite the slot with this turn's READ-tool dispatch record.

    ``calls`` is a list of ``{"tool": name, "args": {...}}`` dicts, newest
    call last (the order the controller dispatched them in). An empty list
    is a deliberate, meaningful write: it clears whatever the PREVIOUS turn
    recorded so a turn that ran no read tools this time can't be mistaken
    for a continuation of an older search on the following turn.
    """
    _STATE["calls"] = list(calls or [])
    _STATE["ts"] = time.monotonic()


def recent_read_tool_calls(max_age_s: float) -> List[Dict[str, Any]]:
    """The last-recorded read-tool calls, or ``[]`` when none were ever
    recorded, the record itself was empty, or it is older than
    ``max_age_s`` seconds (a long-idle gap should not resurrect a stale
    search thread)."""
    calls = _STATE.get("calls") or []
    if not calls:
        return []
    ts = _STATE.get("ts") or 0.0
    if (time.monotonic() - ts) > max_age_s:
        return []
    return list(calls)


def reset() -> None:
    """Test hook: clear the slot back to its initial empty state."""
    _STATE["calls"] = []
    _STATE["ts"] = 0.0
