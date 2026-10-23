"""Pre-spawned default-executor workers (incident 2026-10-23, class BC-41).

First turn after restart: a LoopStall of 2995 ms inside asyncio.to_thread ->
Thread.start(), because the executor starts workers lazily on the loop and the
new thread needed the GIL that busy workers held.  The deployed lifespan hook
must leave every worker already running.
"""
import asyncio
import sys
import threading
import time
from unittest.mock import MagicMock, patch

from utils.executor_prespawn import install_prespawned_default_executor


def _asyncio_workers(executor):
    return len(executor._threads)


def _wait_for(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def test_install_starts_every_worker_and_to_thread_starts_none():
    async def go():
        loop = asyncio.get_running_loop()
        ex = install_prespawned_default_executor(loop, max_workers=6)
        assert _wait_for(lambda: _asyncio_workers(ex) == 6)
        before = _asyncio_workers(ex)
        # A turn's gather: many concurrent to_thread calls, same instance.
        await asyncio.gather(*(asyncio.to_thread(lambda: 1) for _ in range(12)))
        assert _asyncio_workers(ex) == before == 6
        ex.shutdown(wait=True)

    asyncio.run(go())


def test_without_prespawn_workers_start_lazily_on_submit():
    """Control: the base behaviour the hook removes (workers = 0 until used)."""
    async def go():
        from concurrent.futures import ThreadPoolExecutor
        ex = ThreadPoolExecutor(max_workers=6)
        asyncio.get_running_loop().set_default_executor(ex)
        assert _asyncio_workers(ex) == 0
        await asyncio.to_thread(lambda: 1)
        assert _asyncio_workers(ex) >= 1
        ex.shutdown(wait=True)

    asyncio.run(go())


def test_lifespan_installs_the_prespawned_executor_on_the_serving_loop():
    from api.app import create_app

    seen = {}
    real = install_prespawned_default_executor

    def _spy(loop, *a, **k):
        seen["loop"] = loop
        return real(loop, max_workers=4)

    fake_main = MagicMock()
    fake_main.run_shutdown_tasks_async = MagicMock(side_effect=lambda o: asyncio.sleep(0))
    orch = MagicMock()
    with patch("gui.launch.start_background_tasks"), \
         patch("utils.executor_prespawn.install_prespawned_default_executor", _spy), \
         patch.dict(sys.modules, {"main": fake_main}):
        app = create_app(orch, start_background=True)

        async def go():
            async with app.router.lifespan_context(app):
                assert seen["loop"] is asyncio.get_running_loop()
                ex = asyncio.get_running_loop()._default_executor
                assert _wait_for(lambda: _asyncio_workers(ex) == 4)

        asyncio.run(go())
    assert not any(t.name == "ExecutorPrespawn" and t.is_alive() for t in threading.enumerate())
