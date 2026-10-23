"""Pre-spawn the serving loop's default-executor workers (BC-41).

``asyncio.to_thread`` / ``run_in_executor(None, ...)`` start executor worker
threads lazily, ON THE EVENT LOOP, inside ``submit()``.  ``Thread.start()``
waits for the new thread to signal it started, which needs the GIL; when busy
workers (spaCy/thinc, chroma) hold it, the loop stalls (LoopStall 2995 ms on
the first turn after the 2026-10-23 restart; standalone measurement: ~2.9 s
blocked vs 0.1 ms once the workers exist).  Starting every worker once at
startup, from a plain thread, means a turn's gather never starts a thread.
"""
import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger("executor_prespawn")


def default_worker_count() -> int:
    """The same bound asyncio's own default executor uses."""
    return min(32, (os.cpu_count() or 1) + 4)


def install_prespawned_default_executor(
    loop: asyncio.AbstractEventLoop,
    max_workers: Optional[int] = None,
    timeout_s: float = 2.0,
) -> ThreadPoolExecutor:
    """Install a default executor on ``loop`` and start all its workers.

    The workers are started from a short-lived plain thread, never the loop.
    Returns immediately; the executor is usable at once (a submit that races the
    spawn simply behaves as before).
    """
    n = max_workers or default_worker_count()
    executor = ThreadPoolExecutor(max_workers=n, thread_name_prefix="asyncio")
    loop.set_default_executor(executor)

    def _spawn() -> None:
        # Each task waits until all n have started, so n DISTINCT workers exist
        # (a task that finished early would let the pool reuse its worker).
        # Short timeout: while it waits, held workers cannot serve real work
        # that raced the spawn; a broken barrier just leaves lazy starts.
        barrier = threading.Barrier(n, timeout=timeout_s)

        def _hold() -> None:
            try:
                barrier.wait()
            except threading.BrokenBarrierError:  # degrades: fewer pre-spawned workers; the rest start lazily as before
                pass

        try:
            for _ in range(n):
                executor.submit(_hold)
        except RuntimeError:  # degrades: executor already shut down; nothing to pre-spawn
            logger.debug("[ExecutorPrespawn] executor shut down before spawn finished")

    threading.Thread(target=_spawn, name="ExecutorPrespawn", daemon=True).start()
    return executor
