"""2026-09-08 homework-session audit, batch B3 (lazy sandbox acquisition, F5).

`AgenticSearchController.run_agentic_search()` used to call
`await self._get_sandbox_session()` (a remote E2B create-session call)
before round 1 whenever `sandbox_manager.is_available()` was true — a cheap
local flag, unlike session creation itself. A live session created 26
remote sandboxes and closed 13 of them with zero executions. Acquisition is
now deferred to `ToolExecutor._dispatch_sandbox`, the only site that ever
needs the session, via `core.agentic.tools.LazySandboxSession`.

Drives the deployed `run_agentic_search` (not a re-derivation) with a fake
sandbox_manager, faking only the model-decision and final-answer generation
layers — the real DISPATCH_TABLE routing and `_dispatch_sandbox` run.
"""

import inspect
from types import SimpleNamespace

import pytest

from core.agentic.controller import AgenticSearchController
from core.agentic.tools import LazySandboxSession
from core.agentic.types import SearchDecision


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeSandboxSession:
    def __init__(self):
        self.is_closed = False
        self.age_seconds = 0
        self.run_calls = []
        self.close_calls = 0

    def is_alive(self):
        return True

    async def run(self, code):
        self.run_calls.append(code)
        return SimpleNamespace(success=True, execution_time=0.01, stdout="ok", error=None)

    async def close(self):
        self.close_calls += 1
        self.is_closed = True


class FakeSandboxManager:
    def __init__(self, create_error=None):
        self.create_calls = 0
        self.sessions = []
        self.create_error = create_error
        self.execute_calls = []

    def is_available(self):
        return True

    async def create_session(self):
        self.create_calls += 1
        if self.create_error is not None:
            raise self.create_error
        s = FakeSandboxSession()
        self.sessions.append(s)
        return s

    def format_for_prompt(self, result, purpose):
        return f"[sandbox:{purpose}] {getattr(result, 'stdout', '') or getattr(result, 'error', '') or ''}"

    async def execute_code(self, code):
        # Ephemeral fallback path (no persistent session available) — the
        # honest degraded result, never an exception.
        self.execute_calls.append(code)
        return SimpleNamespace(success=True, execution_time=0.01, stdout="ephemeral-ok", error=None)


def _make_controller(sandbox_manager):
    from unittest.mock import MagicMock
    manager = MagicMock()
    manager.api_models = {}
    return AgenticSearchController(
        model_manager=manager,
        web_search_manager=MagicMock(),
        sandbox_manager=sandbox_manager,
    )


async def _run_turn(controller, monkeypatch, decision_batches):
    """Drive run_agentic_search with a scripted sequence of decision rounds
    (list of lists of SearchDecision); every round past the scripted ones is
    `is_done`. `_generate_final_response` is faked (no real generation)."""
    calls = {"n": 0}

    async def fake_decision(*args, **kwargs):
        idx = calls["n"]
        calls["n"] += 1
        if idx < len(decision_batches):
            return decision_batches[idx]
        return [SearchDecision(is_done=True)]

    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        yield "FINAL ANSWER."

    monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
    monkeypatch.setattr(controller, "_generate_final_response", fake_final)

    events = []
    async for ev in controller.run_agentic_search(
        query="run some code for me",
        system_prompt="sys",
        model_name="test-model",
        initial_search_terms=[],
        skip_initial_search=True,
    ):
        events.append(ev)
    return events


def _sandbox_decision(code="1 + 1", purpose="test"):
    return SearchDecision(wants_sandbox=True, sandbox_code=code, sandbox_purpose=purpose)


# ---------------------------------------------------------------------------
# LazySandboxSession — direct unit tests
# ---------------------------------------------------------------------------

class TestLazySandboxSessionDirect:
    @pytest.mark.asyncio
    async def test_is_closed_before_acquisition(self):
        lazy = LazySandboxSession(lambda: None)
        assert lazy.is_closed is True

    @pytest.mark.asyncio
    async def test_get_calls_acquire_once_and_caches(self):
        calls = {"n": 0}
        fake_session = SimpleNamespace(is_closed=False)

        async def acquire():
            calls["n"] += 1
            return fake_session

        lazy = LazySandboxSession(acquire)
        result1 = await lazy.get()
        result2 = await lazy.get()
        assert result1 is fake_session
        assert result2 is fake_session
        assert calls["n"] == 1
        assert lazy.is_closed is False

    @pytest.mark.asyncio
    async def test_none_result_is_cached_and_reports_closed(self):
        calls = {"n": 0}

        async def acquire():
            calls["n"] += 1
            return None

        lazy = LazySandboxSession(acquire)
        assert await lazy.get() is None
        assert await lazy.get() is None
        assert calls["n"] == 1
        assert lazy.is_closed is True


# ---------------------------------------------------------------------------
# run_agentic_search — no eager acquisition, sandbox only created on demand
# ---------------------------------------------------------------------------

class TestControllerLazyAcquisition:
    @pytest.mark.asyncio
    async def test_no_sandbox_decision_never_creates_a_session(self, monkeypatch):
        sandbox_manager = FakeSandboxManager()
        controller = _make_controller(sandbox_manager)
        await _run_turn(controller, monkeypatch, decision_batches=[])
        assert sandbox_manager.create_calls == 0

    @pytest.mark.asyncio
    async def test_one_sandbox_decision_creates_and_runs(self, monkeypatch):
        sandbox_manager = FakeSandboxManager()
        controller = _make_controller(sandbox_manager)
        await _run_turn(
            controller, monkeypatch,
            decision_batches=[[_sandbox_decision(code="print(1)")]],
        )
        assert sandbox_manager.create_calls == 1
        assert sandbox_manager.sessions[0].run_calls == ["print(1)"]
        assert sandbox_manager.execute_calls == []  # persistent path used, not ephemeral

    @pytest.mark.asyncio
    async def test_two_sandbox_rounds_in_one_turn_share_one_session(self, monkeypatch):
        sandbox_manager = FakeSandboxManager()
        controller = _make_controller(sandbox_manager)
        await _run_turn(
            controller, monkeypatch,
            decision_batches=[
                [_sandbox_decision(code="a = 1")],
                [_sandbox_decision(code="b = 2")],
            ],
        )
        assert sandbox_manager.create_calls == 1
        assert sandbox_manager.sessions[0].run_calls == ["a = 1", "b = 2"]

    @pytest.mark.asyncio
    async def test_expired_session_is_recreated_on_the_next_turn(self, monkeypatch):
        sandbox_manager = FakeSandboxManager()
        controller = _make_controller(sandbox_manager)

        await _run_turn(controller, monkeypatch, decision_batches=[[_sandbox_decision()]])
        assert sandbox_manager.create_calls == 1
        first_session = sandbox_manager.sessions[0]

        # Age it out between turns — _get_sandbox_session's own recycling
        # (untouched by this batch) must still fire.
        first_session.age_seconds = controller._sandbox_session_timeout + 1

        await _run_turn(controller, monkeypatch, decision_batches=[[_sandbox_decision()]])
        assert sandbox_manager.create_calls == 2
        assert sandbox_manager.sessions[1] is not first_session
        assert first_session.close_calls == 1  # the stale one was closed on recycle

    @pytest.mark.asyncio
    async def test_create_session_failure_falls_back_honestly(self, monkeypatch):
        sandbox_manager = FakeSandboxManager(create_error=RuntimeError("E2B unavailable"))
        controller = _make_controller(sandbox_manager)
        # Must not raise — the round degrades to the ephemeral execute_code
        # path (still honest: no exception escapes the turn).
        events = await _run_turn(
            controller, monkeypatch, decision_batches=[[_sandbox_decision(code="x = 1")]],
        )
        assert sandbox_manager.create_calls == 1
        assert sandbox_manager.execute_calls == ["x = 1"]
        assert any(isinstance(e, str) for e in events)  # final answer still streamed

    @pytest.mark.asyncio
    async def test_close_sandbox_is_idempotent(self, monkeypatch):
        sandbox_manager = FakeSandboxManager()
        controller = _make_controller(sandbox_manager)
        # Closing before anything was ever acquired must not raise.
        await controller.close_sandbox()

        await _run_turn(controller, monkeypatch, decision_batches=[[_sandbox_decision()]])
        session = sandbox_manager.sessions[0]

        await controller.close_sandbox()
        assert session.close_calls == 1
        await controller.close_sandbox()
        assert session.close_calls == 1  # second close is a no-op


# ---------------------------------------------------------------------------
# Source-level guard — run_agentic_search no longer awaits the eager call
# ---------------------------------------------------------------------------

class TestSourceLevelGuard:
    def test_run_agentic_search_does_not_eagerly_await_sandbox_acquisition(self):
        src = inspect.getsource(AgenticSearchController.run_agentic_search)
        assert "LazySandboxSession(" in src
        assert "await self._get_sandbox_session()" not in src

    def test_dispatch_sandbox_resolves_the_lazy_session(self):
        from core.agentic.tools import ToolExecutor
        src = inspect.getsource(ToolExecutor._dispatch_sandbox)
        assert "LazySandboxSession" in src
        assert ".get()" in src
