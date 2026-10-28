"""The ONE accessor for "recent turns visible to the current turn" (2026-10-28, BC-105).

Every turn-path reader of the corpus's recent turns (prompt, gate, STM/tone/thread/topic,
pacing, action guard...) asks THIS function instead of calling
``corpus_manager.get_recent_memories`` raw, so probe traffic (``[test]`` turns) is hidden
from a REAL turn in one place. Rule: ``core.prompt.token_manager.drop_prior_session_test_turns``
(the single predicate, built on ``memory.fact_source.is_test_turn``):

* ``current_query`` is a real turn -> no probe turn is returned (same session or not);
* ``current_query`` is a probe -> its own session's probes stay, prior-session probes drop;
* ``current_query`` is ``None`` (a caller outside a turn) -> same as a probe turn: prior-session
  probes drop (today's chokepoint behaviour).

Over-fetches so the count is still met after probes are dropped. Works on any object that
has ``get_recent_memories`` (test fakes included).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

RECENT_PROBE_LOOKBACK = 20


def recent_turns_visible_to(corpus_manager: Any, count: int, current_query: Optional[str] = None,
                            now: Optional[datetime] = None) -> List[Any]:
    """Newest-first recent turns the turn with ``current_query`` may see."""
    from core.prompt.token_manager import drop_prior_session_test_turns  # lazy import: layering (memory/ must not import core.prompt at module level; token_manager owns the session rule)
    pulled = list(corpus_manager.get_recent_memories(count + RECENT_PROBE_LOOKBACK) or [])
    return drop_prior_session_test_turns(pulled, now, current_query=current_query)[:count]
