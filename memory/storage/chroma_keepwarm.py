"""Keep Chroma's files warm (BC-41, BC-70).

``get_collection`` opens a collection without reading its data, so the first query
of a quiet session pays the cold-page-cache cost (CHR measurement: 0.3-6.6 s) inside
the user's first turn.  ``touch_collections`` reads one row per hot collection --
read-only, no embedder call, bounded -- and ``ChromaKeepWarm`` repeats that while the
process is idle so the pages stay resident.

Collections touched = the ones the turn path reads (conversations, summaries,
reflections, facts, procedural, procedural_skills, obsidian_notes, reference_docs,
proposals, threads, daemon_self_notes).  ``wiki_knowledge`` (large; wiki search is
FAISS-first), ``visual_memories`` (FAISS) and ``synthesis_results`` (insight-only) are
left out; ``chroma_keepwarm.collections`` overrides the list.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Iterable, List, Optional

from config import app_config
from utils.logging_utils import get_logger

logger = get_logger("chroma_keepwarm")

DEFAULT_COLLECTIONS = (
    "conversations", "summaries", "reflections", "facts", "procedural",
    "procedural_skills", "obsidian_notes", "reference_docs", "proposals",
    "threads", "daemon_self_notes",
)

# Filters the turn path really pushes into Chroma (thread_store.list_open_threads,
# gatherer_knowledge.get_user_uploads); the where-path is what is slow cold.
_TURN_PATH_FILTERS = {
    "threads": {"status": "open"},
    "reference_docs": {"type": "user_upload"},
}
_FALLBACK_DIM = 384

# A worker that outlives its timeout is still inside Chroma (a join cannot stop it). Until it
# returns, later passes skip that collection, so a hung store holds at most one keep-warm
# thread per collection instead of gaining one every pass.
_OUTSTANDING: Dict[str, threading.Thread] = {}
_OUTSTANDING_LOCK = threading.Lock()


def _vector(store) -> List[float]:
    dim = _FALLBACK_DIM
    try:
        model = getattr(getattr(store, "embedding_fn", None), "_model", None)
        d = model.get_sentence_embedding_dimension() if model is not None else None
        if isinstance(d, int) and d > 0:
            dim = d
    except Exception:  # degrades: falls back to the bge-small dimension
        pass
    return [dim ** -0.5] * dim


def _touch_one(store, name: str, vec: List[float]) -> None:
    coll = store._get_collection(name)
    errors = []
    try:
        coll.query(query_embeddings=[vec], n_results=1, include=["metadatas"])
    except Exception as e:  # degrades: this collection's vector pages stay cold
        errors.append(f"query: {e}")
    try:
        kwargs = {"limit": 1, "include": ["metadatas"]}
        where = _TURN_PATH_FILTERS.get(name)
        if where:
            kwargs["where"] = where
        coll.get(**kwargs)
    except Exception as e:  # degrades: this collection's metadata pages stay cold
        errors.append(f"get: {e}")
    if errors:
        raise RuntimeError("; ".join(errors))


def touch_collections(store, names: Optional[Iterable[str]] = None, *,
                      per_collection_timeout_s: Optional[float] = None,
                      total_budget_s: Optional[float] = None,
                      clock: Callable[[], float] = time.monotonic) -> Dict[str, Optional[float]]:
    """Read one row from each collection. Returns {name: ms, or None when it failed/timed out/was skipped}. Never raises.

    A collection whose previous touch is still running is skipped (see ``_OUTSTANDING``)."""
    result: Dict[str, Optional[float]] = {}
    if per_collection_timeout_s is None:
        per_collection_timeout_s = float(app_config.CHROMA_KEEPWARM_COLLECTION_TIMEOUT_S)  # live read (BC-82)
    if total_budget_s is None:
        total_budget_s = float(app_config.CHROMA_KEEPWARM_TOTAL_BUDGET_S)
    try:
        wanted = list(names) if names else list(DEFAULT_COLLECTIONS)
        known = getattr(store, "collections", None) or {}
        vec = _vector(store)
        start = clock()
        for name in wanted:
            if known and name not in known:
                continue
            if clock() - start >= total_budget_s:
                result[name] = None
                logger.debug(f"[KeepWarm] {name}: skipped, total budget {total_budget_s}s spent")
                continue
            with _OUTSTANDING_LOCK:
                previous = _OUTSTANDING.get(name)
                if previous is not None and not previous.is_alive():
                    del _OUTSTANDING[name]
                    previous = None
            if previous is not None:
                result[name] = None
                logger.debug(f"[KeepWarm] {name}: skipped, previous touch still running")
                continue
            box: Dict[str, object] = {}

            # Bind THIS worker's box: a worker that errors after its timeout must not write into
            # the box of the collection being touched at that moment.
            def _run(n=name, b=box):
                try:
                    _touch_one(store, n, vec)
                except Exception as e:  # degrades: reported below, other collections still touched
                    b["err"] = e

            t0 = clock()
            worker = threading.Thread(target=_run, daemon=True, name=f"keepwarm-{name}")
            worker.start()
            worker.join(per_collection_timeout_s)
            ms = (clock() - t0) * 1000.0
            if worker.is_alive():
                with _OUTSTANDING_LOCK:
                    _OUTSTANDING[name] = worker
                result[name] = None
                logger.debug(f"[KeepWarm] {name}: timed out after {per_collection_timeout_s}s")
            elif "err" in box:
                result[name] = None
                logger.debug(f"[KeepWarm] {name}: failed ({box['err']})")
            else:
                result[name] = ms
        logger.debug("[KeepWarm] pass: " + ", ".join(
            f"{n}={'fail' if v is None else f'{v:.0f}ms'}" for n, v in result.items()))
    except Exception as e:  # degrades: this pass is lost; the next one runs normally
        logger.debug(f"[KeepWarm] pass failed: {e}")
    return result


def _config_names() -> Optional[List[str]]:
    return list(app_config.CHROMA_KEEPWARM_COLLECTIONS) or None


def touch_from_config(store, extra: Iterable[str] = ()) -> Dict[str, Optional[float]]:
    names = list(dict.fromkeys([*extra, *(_config_names() or DEFAULT_COLLECTIONS)]))
    return touch_collections(store, names)


class ChromaKeepWarm:
    """Idle keep-warm thread. All time/IO inputs are injectable."""

    def __init__(self, store_fn: Callable[[], object], *,
                 interval_fn: Optional[Callable[[], float]] = None,
                 inflight_fn: Optional[Callable[[], bool]] = None,
                 last_activity_fn: Optional[Callable[[], float]] = None,
                 touch_fn: Optional[Callable[[object], object]] = None,
                 clock: Callable[[], float] = time.time,
                 poll_s: float = 15.0):
        self.store_fn = store_fn
        self.interval_fn = interval_fn or _default_interval
        self.inflight_fn = inflight_fn or _default_inflight
        self.last_activity_fn = last_activity_fn or _default_last_activity
        self.touch_fn = touch_fn or touch_from_config
        self.clock = clock
        self.poll_s = poll_s
        self.last_pass = clock()  # startup warmup just touched
        self.passes = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def tick(self) -> str:
        """One decision. Returns 'disabled' | 'not_due' | 'busy' | 'recent_activity' | 'touched' | 'error'."""
        try:
            interval = float(self.interval_fn())
        except Exception:  # degrades: unreadable interval counts as disabled
            return "disabled"
        if interval <= 0:
            return "disabled"
        now = self.clock()
        if now - self.last_pass < interval:
            return "not_due"
        try:
            if self.inflight_fn():
                return "busy"
        except Exception:  # degrades: unknown in-flight state counts as busy
            return "busy"
        try:
            if now - float(self.last_activity_fn()) < interval:
                return "recent_activity"
        except Exception:  # degrades: unknown activity time counts as recent
            return "recent_activity"
        try:
            store = self.store_fn()
            if store is None:
                return "error"
            self.touch_fn(store)
        except Exception as e:  # degrades: this pass is lost
            logger.debug(f"[KeepWarm] idle pass failed: {e}")
            self.last_pass = self.clock()
            return "error"
        self.last_pass = self.clock()
        self.passes += 1
        return "touched"

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            self.tick()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ChromaKeepWarm")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout)


def _default_interval() -> float:
    return float(app_config.CHROMA_KEEPWARM_INTERVAL_S)


def _default_inflight() -> bool:
    from gui.handlers import has_inflight_turns  # lazy import: cycle (gui imports memory)
    return has_inflight_turns()


def _default_last_activity() -> float:
    import main  # lazy import: cycle (main alias -- see main.py:802)
    return float(main._last_activity_time)


_instance: Optional[ChromaKeepWarm] = None
_instance_lock = threading.Lock()


def start_keepwarm(store_fn: Callable[[], object]) -> Optional[ChromaKeepWarm]:
    """Start the process-wide idle keep-warm thread (idempotent). Never raises."""
    global _instance
    try:
        with _instance_lock:
            if _instance is None:
                _instance = ChromaKeepWarm(store_fn)
            _instance.start()
            return _instance
    except Exception as e:  # degrades: no idle keep-warm this run
        logger.warning(f"[KeepWarm] not started: {e}")
        return None


def stop_keepwarm() -> None:
    global _instance
    with _instance_lock:
        inst, _instance = _instance, None
    if inst is not None:
        inst.stop()
