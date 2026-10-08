"""
# memory/pending_turns.py

Module Contract
- Purpose: in-process registry of turns that were DELIVERED but are not yet in
  the corpus (2026-10-08, class: BC-38, BC-45). `_background_store_interaction`
  awaits the personal-claim receipt (up to ~10 s) before `store_interaction`,
  so a follow-up sent inside that window read a corpus that ended one turn
  early (gate, STM and web-trigger digest all described the turn BEFORE).
- Readers: `CorpusManager.get_recent_memories` / `get_recent_within_hours`
  merge `pending_entries(self)` on the READ path only. Nothing here is ever
  persisted: no disk, no corpus cache, bounded (cap + TTL), scoped per owner.
- Writers: gui/handlers `_dispatch_storage` registers; the storage task
  `forget`s in a `finally` (landed OR failed); TTL covers a crashed task.
- Never raises.
"""
import threading
import time
import weakref
from datetime import datetime
from typing import Any, Dict, List

PENDING_TURN_CAP = 8
PENDING_TURN_TTL_S = 120.0

_lock = threading.Lock()
_pending: list = []   # [(norm_query, entry_dict, monotonic_ts, owner_ref)]


def _norm(text) -> str:
    return " ".join(str(text or "").lower().split())


def _owner_ref(owner):
    try:
        return weakref.ref(owner)
    except TypeError:
        return lambda o=owner: o


def _live(now: float) -> None:
    _pending[:] = [e for e in _pending if (now - e[2]) < PENDING_TURN_TTL_S and e[3]() is not None]


def register(owner, query, response, user_text=None) -> None:
    try:
        norm = _norm(query)
        if not norm or not str(response or "").strip():
            return
        entry = {
            "query": str(query),
            "response": str(response),
            "timestamp": datetime.now(),
            "pending_storage": True,
        }
        if user_text is not None:
            entry["user_text"] = str(user_text)
        now = time.monotonic()
        with _lock:
            _live(now)
            _pending[:] = [e for e in _pending if not (e[0] == norm and e[3]() is owner)]
            _pending.append((norm, entry, now, _owner_ref(owner)))
            del _pending[:-PENDING_TURN_CAP]
    except Exception:  # degrades: a follow-up inside the storage window sees the corpus one turn early
        pass


def forget(owner, query) -> None:
    try:
        norm = _norm(query)
        with _lock:
            _pending[:] = [e for e in _pending if not (e[0] == norm and e[3]() is owner)]
    except Exception:  # degrades: entry lingers until its TTL
        pass


def pending_entries(owner) -> List[Dict[str, Any]]:
    """Fresh dict copies (newest first) of this owner's unexpired pending turns."""
    try:
        now = time.monotonic()
        with _lock:
            _live(now)
            return [dict(e[1]) for e in reversed(_pending) if e[3]() is owner]
    except Exception:  # degrades: pending turns invisible to readers
        return []
