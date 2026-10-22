"""Process-wide store for the user-claim ledger (``data/claim_ledger.json``).

Incident 2026-10-20; class: BC-75, BC-52. ``memory/claim_ledger.py`` is the pure
core; this module owns WHERE the ledger lives and WHO may write it.

* ``get_ledger()`` loads the store once per process through ``utils/safe_json``
  STRICT: a missing file is an empty ledger; an existing-but-corrupt file is
  quarantined (``.corrupt-<ts>`` copy) and raises ``CorruptStoreError``; a file
  written by a newer build raises ``StoreVersionError``. NEVER catch either and
  continue with empty state -- the next save would overwrite the user's data
  (CLAUDE.md Critical Rules). ``MemoryCoordinator`` calls it at startup when the
  ledger is enabled, so ``main.py``'s startup-abort path (the same one the
  knowledge graph and claim index use) turns it into an actionable message.
* ``save_ledger()`` is atomic (``ClaimLedger.save`` -> ``atomic_write_json``).
* ``write_lock()`` serialises writers within the running event loop.

The path is read at call time from ``app_config.CLAIM_LEDGER_STORE_PATH`` (BC-82)
and resolved against the one data-root authority (``bootstrap.store_path``).
Tests sandbox it with ``_PATH_OVERRIDE`` (tests/conftest.py).
"""
from __future__ import annotations

import asyncio
import threading
from typing import List, Optional

from datetime import datetime

from config import app_config
from memory import claim_ledger
from memory.claim_ledger import ClaimLedger
from utils import bootstrap
from utils.logging_utils import get_logger

logger = get_logger("claim_ledger_store")

_PATH_OVERRIDE: Optional[str] = None       # tests only
_ledger: Optional[ClaimLedger] = None
_loaded_path: Optional[str] = None
_state_lock = threading.RLock()
_async_lock: Optional[asyncio.Lock] = None
_async_lock_loop: Optional[asyncio.AbstractEventLoop] = None


def store_path() -> str:
    return _PATH_OVERRIDE or bootstrap.store_path(app_config.CLAIM_LEDGER_STORE_PATH)


def get_ledger() -> ClaimLedger:
    """The process-wide ledger, loaded strictly on first use (or when the path changed).

    Raises ``CorruptStoreError`` / ``StoreVersionError``; the caller must not
    substitute an empty ledger."""
    global _ledger, _loaded_path
    path = store_path()
    with _state_lock:
        if _ledger is None or _loaded_path != path:
            _ledger = ClaimLedger.load(path)
            _loaded_path = path
            logger.info("[ClaimLedger] loaded %d claims", len(_ledger.claims))
        return _ledger


def save_ledger() -> None:
    """Atomically persist the in-memory ledger. Only valid after ``get_ledger()``."""
    with _state_lock:
        if _ledger is None:
            raise RuntimeError("claim ledger was never loaded; refusing to save")
        _ledger.save(_loaded_path or store_path())


def write_lock() -> asyncio.Lock:
    """The writers' lock for the RUNNING loop (a lock built on another loop would raise
    on contention, so it is rebuilt when the loop changes)."""
    global _async_lock, _async_lock_loop
    loop = asyncio.get_running_loop()
    with _state_lock:
        if _async_lock is None or _async_lock_loop is not loop:
            _async_lock, _async_lock_loop = asyncio.Lock(), loop
        return _async_lock


def reset_for_tests() -> None:
    global _ledger, _loaded_path, _async_lock, _async_lock_loop
    with _state_lock:
        _ledger = _loaded_path = _async_lock = _async_lock_loop = None


def stated_figure_lines(now: Optional[datetime] = None) -> List[str]:
    """Lines for the ``[YOUR STATED FIGURES]`` prompt block, or [] (no LLM, no network).

    Returns [] unless ``claim_ledger.enabled`` AND ``claim_ledger.render_block`` are on
    (live reads, BC-82). Capped to ``max_block_lines`` lines and ``max_block_tokens``
    (estimated at chars/3 -- measured 3.2 chars/token on real blocks, so chars/4 would under-count by ~25%; trailing -- oldest -- lines are dropped whole, never cut mid-line). An
    unreadable store returns [] and logs: startup already aborted on a corrupt file,
    and a prompt build must never fail on the ledger."""
    if not (app_config.CLAIM_LEDGER_ENABLED and app_config.CLAIM_LEDGER_RENDER_BLOCK):
        return []
    try:
        lines = claim_ledger.render_stated_figures(
            get_ledger(), now or datetime.now(),
            max_lines=app_config.CLAIM_LEDGER_MAX_BLOCK_LINES,
            standing_window_days=app_config.CLAIM_LEDGER_STANDING_WINDOW_DAYS,
            deadline_window_days=app_config.CLAIM_LEDGER_DEADLINE_WINDOW_DAYS)
    except Exception as exc:  # degrades: no [YOUR STATED FIGURES] block this turn
        logger.warning("[ClaimLedger] stated-figures render failed: %s", type(exc).__name__)
        return []
    kept: List[str] = []
    used = 0
    for line in lines:
        cost = (len(line) + 2) // 3
        if used + cost > app_config.CLAIM_LEDGER_MAX_BLOCK_TOKENS:
            break
        kept.append(line)
        used += cost
    return kept
