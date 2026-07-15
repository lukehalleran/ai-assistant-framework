"""
# utils/turn_progress.py

Module Contract
- Purpose: Per-turn progress bus so deep pipeline stages (retrieval tasks,
  gating, prompt assembly) can surface live status lines to the streaming UI
  without threading callbacks through every layer. The GUI handler installs a
  queue at turn start and drains it while awaiting prepare_prompt; the prompt
  builder publishes fire-and-forget events.
- Inputs: emit(text) from anywhere in the pipeline (sync or async code).
- Outputs: begin_turn() -> asyncio.Queue consumed by the handler; drain(queue)
  -> list[str] of pending events.
- Concurrency model: SINGLE-USER — one chat turn in flight at a time (enforced
  by the API's stream lock and Gradio's queue). One module-level current queue;
  begin_turn replaces it, end_turn clears it. emit() is non-blocking and never
  raises: no active turn (CLI/tests/shutdown paths) or a full queue = drop.
- Side effects: none beyond the in-memory queue.
"""

import asyncio
from typing import List, Optional

from utils.logging_utils import get_logger

logger = get_logger("turn_progress")

_MAX_EVENTS = 200

_current_queue: Optional[asyncio.Queue] = None


def begin_turn() -> asyncio.Queue:
    """Install a fresh event queue for the turn and return it."""
    global _current_queue
    _current_queue = asyncio.Queue(maxsize=_MAX_EVENTS)
    return _current_queue


def end_turn() -> None:
    global _current_queue
    _current_queue = None


def emit(text: str) -> None:
    """Publish a progress line for the in-flight turn. Never raises, never blocks."""
    q = _current_queue
    if q is None or not text:
        return
    try:
        q.put_nowait(text)
    except asyncio.QueueFull:
        pass


def drain(queue: asyncio.Queue) -> List[str]:
    """Return all currently-pending events without waiting."""
    out: List[str] = []
    while True:
        try:
            out.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            return out
