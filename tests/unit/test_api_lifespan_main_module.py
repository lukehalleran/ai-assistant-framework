"""Behavioural sibling: the FastAPI lifespan runs shutdown through the ALIASED ``main`` module (source-text row).

Incident date: 2026-08-22 (launched as ``python main.py`` the script is ``__main__``; the
lifespan's ``import main`` re-executed main.py as a second module and the full shutdown
sequence ran TWICE). Class: BC-63 (a grep for ``import main as main_mod`` stands in for
the alias contract), BC-81 (a second live instance of process-wide state).

The alias is ``sys.modules["main"] = <running module>``; the contract that makes it work is
that the lifespan resolves its idle monitor and its shutdown through ``sys.modules["main"]``
at run time. Drives THE deployed ``create_app`` lifespan with a stand-in module registered
under ``main`` and asserts both calls land on that module, with the orchestrator, and that
no second ``main`` is imported.
"""
import asyncio
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from api.app import create_app


def _fake_main():
    started = threading.Event()
    return SimpleNamespace(
        _orchestrator_ref=None,
        started=started,
        shutdown_calls=[],
        _idle_monitor_thread=lambda: started.set(),
        run_shutdown_tasks_async=None,
    )


def _run_lifespan(fake_main, orch):
    async def _shutdown(o):
        fake_main.shutdown_calls.append(o)

    fake_main.run_shutdown_tasks_async = _shutdown
    with patch("gui.launch.start_background_tasks"), \
         patch("utils.executor_prespawn.install_prespawned_default_executor"), \
         patch.dict(sys.modules, {"main": fake_main}):
        app = create_app(orch, start_background=True)

        async def go():
            async with app.router.lifespan_context(app):
                await asyncio.sleep(0.05)
                assert fake_main.started.wait(2), "idle monitor was not started on the aliased main"
                assert fake_main._orchestrator_ref is orch

        asyncio.run(go())
        assert sys.modules["main"] is fake_main  # no second module instance was created


def test_lifespan_starts_the_idle_monitor_and_runs_shutdown_on_the_aliased_main():
    fake_main, orch = _fake_main(), MagicMock()
    _run_lifespan(fake_main, orch)
    assert fake_main.shutdown_calls == [orch]


def test_lifespan_without_background_tasks_never_touches_main():
    fake_main, orch = _fake_main(), MagicMock()
    with patch.dict(sys.modules, {"main": fake_main}):
        app = create_app(orch, start_background=False)

        async def go():
            async with app.router.lifespan_context(app):
                pass

        asyncio.run(go())
    assert not fake_main.started.is_set()
    assert fake_main.shutdown_calls == []
