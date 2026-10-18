"""
Event-loop stall detector (2026-10-15, class: BC-41 / BC-69).

Module Contract:
  Purpose: Name the synchronous code that blocks the asyncio loop. A live
           turn logged ``chroma_ms=8357`` under a 3.0 s ``wait_for`` with
           ``timed_out=False`` -- a timeout that cannot fire means the loop
           itself was blocked by a coroutine doing blocking work inline.
  Mechanism: a tiny heartbeat coroutine on the loop stamps a monotonic time
           every ``interval_s``; a daemon THREAD checks the stamp at the same
           cadence. When the stamp is older than the threshold the thread
           samples the LOOP thread's stack (``sys._current_frames()``) and
           keeps the innermost frames. One record per stall (the heartbeat's
           own measured gap is the duration).
  Output:  ``records()`` -> list of {"ms": int, "top": "path:line func",
           "entry": "path:line func" | None, "frames": [...]} longest first,
           max 5. ``top`` is the INNERMOST repo frame (where the blocking call
           sits); ``entry`` is the OUTERMOST repo frame (the code that started
           the chain, i.e. the caller to fix). A stall >= ``OTHER_THREADS_MIN_MS``
           also carries ``other_threads`` (2026-10-18): a loop thread waiting with
           no repo frame of its own is waiting on a GIL holder, so the same sample
           names what every OTHER thread was running ("name: frame <- frame ; ...").
           Appended to the same [LoopStall] log line. If the stack is deeper than
           ``MAX_STACK_WALK`` frames, ``entry`` is the outermost repo frame
           seen within that walk. Numbers
           and code locations ONLY: no locals, arguments or source text
           (turn records must stay content-free).
  Config:  ``turn_telemetry.loop_stall_detector`` (default true) and
           ``turn_telemetry.loop_stall_threshold_ms`` (default 250), read at
           call time (BC-82).
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from utils.logging_utils import get_logger

logger = get_logger("loop_stall")

MAX_RECORDS = 5
MAX_FRAMES = 8
MAX_STACK_WALK = 64
DEFAULT_THRESHOLD_MS = 250
OTHER_THREADS_MIN_MS = 1000   # read at call time; stalls this long also name the other threads
MAX_OTHER_THREADS = 8
MAX_OTHER_FRAMES = 3
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _frame_label(frame) -> str:
    path = frame.f_code.co_filename
    try:
        abspath = os.path.abspath(path)
        if abspath.startswith(_ROOT + os.sep):
            rel = os.path.relpath(abspath, _ROOT)
        elif "site-packages" + os.sep in abspath:
            rel = abspath.split("site-packages" + os.sep, 1)[1]
        else:
            rel = os.path.basename(abspath)
    except (ValueError, OSError):  # degrades: frame label falls back to the raw filename
        rel = path
    return f"{rel}:{frame.f_lineno} {frame.f_code.co_name}"


def _is_repo_label(label: str) -> bool:
    """True only for a frame in THIS repo's own code: the relative path must exist
    under the repo root (library frames are labelled relative to site-packages and
    stdlib frames by basename, so a not-absolute test alone would admit them)."""
    head = label.split(":", 1)[0]
    if os.path.isabs(head) or "site-packages" in head or head.startswith(("venv" + os.sep, ".venv" + os.sep)):
        return False
    return os.path.isfile(os.path.join(_ROOT, head)) and os.sep in head


def _other_threads_summary(loop_tid: Optional[int], watcher_tid: int, frames_by_tid: Dict[int, Any]) -> str:
    """Top frames of every thread except the loop and the watcher, capped.

    Threads running repo code sort first (idle pool workers parked in the stdlib
    would otherwise crowd out the one holding the GIL). Per thread: the innermost
    frame, then repo frames innermost-first, at most MAX_OTHER_FRAMES. Code
    locations only.
    """
    names = {t.ident: t.name for t in threading.enumerate()}
    rows = []
    for tid, frame in frames_by_tid.items():
        if tid in (loop_tid, watcher_tid):
            continue
        labels: List[str] = []
        depth = 0
        while frame is not None and depth < MAX_STACK_WALK:
            labels.append(_frame_label(frame))
            frame = frame.f_back
            depth += 1
        if not labels:
            continue
        repo = [lb for lb in labels if _is_repo_label(lb)]
        picked = [labels[0]] + [lb for lb in repo if lb != labels[0]]
        rows.append((not repo, f"{names.get(tid, f'thread-{tid}')}: " + " <- ".join(picked[:MAX_OTHER_FRAMES])))
    rows.sort(key=lambda r: r[0])  # stable: repo-running threads first
    return " ; ".join(text for _, text in rows[:MAX_OTHER_THREADS])


def read_settings() -> tuple:
    """(enabled, threshold_ms) read live from app_config (BC-82)."""
    import config.app_config as app_config  # lazy import: live-config
    enabled = bool(getattr(app_config, "LOOP_STALL_DETECTOR_ENABLED", True))
    try:
        threshold = int(getattr(app_config, "LOOP_STALL_THRESHOLD_MS", DEFAULT_THRESHOLD_MS))
    except (TypeError, ValueError):  # degrades: detector uses the default threshold
        threshold = DEFAULT_THRESHOLD_MS
    return enabled, max(20, threshold)


class LoopStallDetector:
    """Async context manager; a no-op when disabled."""

    def __init__(self, threshold_ms: Optional[int] = None, enabled: Optional[bool] = None,
                 interval_s: float = 0.05):
        cfg_enabled, cfg_threshold = read_settings()
        self.enabled = cfg_enabled if enabled is None else bool(enabled)
        self.threshold_ms = cfg_threshold if threshold_ms is None else int(threshold_ms)
        self.interval_s = interval_s
        self._records: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stamp = time.monotonic()
        self._loop_tid: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._hb_task: Optional[asyncio.Task] = None
        # Sample of the stall in progress (thread-written, loop/stop-read).
        self._pending_frames: Optional[List[str]] = None
        self._pending_ms: float = 0.0
        self._pending_entry: Optional[str] = None
        self._pending_other: Optional[str] = None

    # -- thread side ---------------------------------------------------
    def _watch(self) -> None:
        while not self._stop_evt.wait(self.interval_s):
            age_ms = (time.monotonic() - self._stamp) * 1000.0
            if age_ms < self.threshold_ms:
                continue
            all_frames = sys._current_frames()
            frame = all_frames.get(self._loop_tid)
            if frame is None:
                continue
            # Walk the whole stack (bounded) so the innermost REPO frame is found
            # even when the block sits deep inside a library (torch import, chroma);
            # keep the innermost MAX_FRAMES plus that repo frame.
            labels: List[str] = []
            repo_label: Optional[str] = None
            entry_label: Optional[str] = None
            depth = 0
            while frame is not None and depth < MAX_STACK_WALK:
                label = _frame_label(frame)
                if len(labels) < MAX_FRAMES:
                    labels.append(label)
                if repo_label is None and _is_repo_label(label):
                    repo_label = label
                if _is_repo_label(label):
                    entry_label = label  # keeps overwriting: ends as the outermost
                frame = frame.f_back
                depth += 1
            if repo_label is not None and repo_label not in labels:
                labels.append(repo_label)
            other = None
            if age_ms >= OTHER_THREADS_MIN_MS:
                other = _other_threads_summary(self._loop_tid, threading.get_ident(), all_frames) or None
            with self._lock:
                self._pending_frames = labels
                self._pending_entry = entry_label
                if other is not None:
                    self._pending_other = other
                self._pending_ms = max(self._pending_ms, age_ms)

    # -- loop side -----------------------------------------------------
    async def _heartbeat(self) -> None:
        while True:
            now = time.monotonic()
            gap_ms = (now - self._stamp) * 1000.0
            if gap_ms >= self.threshold_ms:
                self._finish(gap_ms)
            self._stamp = now
            await asyncio.sleep(self.interval_s)

    def _finish(self, ms: float) -> None:
        with self._lock:
            frames, self._pending_frames, self._pending_ms = self._pending_frames, None, 0.0
            entry, self._pending_entry = self._pending_entry, None
            other, self._pending_other = self._pending_other, None
        if not frames:
            return
        top = next((f for f in frames if _is_repo_label(f)), frames[0])
        record = {"ms": int(round(ms)), "top": top, "entry": entry, "frames": frames}
        suffix = ""
        if other and record["ms"] >= OTHER_THREADS_MIN_MS:
            record["other_threads"] = other
            suffix = " | other threads: " + other
        self._records.append(record)
        logger.warning("[LoopStall] event loop blocked %d ms at %s (entry: %s) | frames: %s%s",
                       record["ms"], top, entry, " <- ".join(frames), suffix)

    async def __aenter__(self) -> "LoopStallDetector":
        if not self.enabled:
            return self
        self._loop_tid = threading.get_ident()
        self._stamp = time.monotonic()
        self._stop_evt.clear()
        self._hb_task = asyncio.get_running_loop().create_task(self._heartbeat())
        self._thread = threading.Thread(target=self._watch, name="loop-stall-watch", daemon=True)
        self._thread.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_evt.set()
        self._thread.join(timeout=1.0)
        self._thread = None
        if self._hb_task is not None:
            self._hb_task.cancel()
            self._hb_task = None
        # A stall still open when the guard exits: close it from the thread's sample.
        with self._lock:
            open_ms = self._pending_ms
        if open_ms:
            self._finish(open_ms)

    def records(self) -> List[Dict[str, Any]]:
        out = sorted(self._records, key=lambda r: r["ms"], reverse=True)[:MAX_RECORDS]
        return [dict(r) for r in out]

    def turn_fields(self) -> List[Dict[str, Any]]:
        """Content-free form for the turn record: [{ms, top, entry}], longest first."""
        out = []
        for r in self.records():
            row = {"ms": r["ms"], "top": r["top"], "entry": r.get("entry")}
            if r.get("other_threads"):
                row["other_threads"] = r["other_threads"]  # added key only on long stalls
            out.append(row)
        return out
