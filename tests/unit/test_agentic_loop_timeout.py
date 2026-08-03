"""Latency guards for the agentic loop (2026-07-24).

Incident: a turn hung ~2 minutes inside the agentic loop. kimi-3 narrated its
tool intent in prose instead of emitting XML markers, each decision round
streamed for ~55-60s, and there was NO wall-clock ceiling — the loop could run
every round to max_rounds (5 × ~55s) and a stalled provider call had no timeout
at all. The user gave up and hit "Retry".

Two guards now bound this:
  - round_timeout_s  → asyncio.wait_for around each decision-LLM call; on
    timeout the loop answers with whatever context it has (backstop vs. a
    stalled connection).
  - loop_timeout_s   → wall-clock budget for the rounds-2-N loop; once exceeded,
    no new round starts and the loop falls through to final synthesis.
"""
import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from core.agentic.controller import AgenticSearchController
from core.agentic.types import SearchDecision


@pytest.fixture
def controller():
    manager = MagicMock()
    manager.api_models = {}
    return AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())


def _make_final_spy(captured):
    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        captured["final_called"] = True
        captured["session"] = session
        yield "SYNTHESIZED ANSWER."
    return fake_final


async def _run(controller):
    events = []
    async for ev in controller.run_agentic_search(
        query="q", system_prompt="sys", model_name="test-model",
        initial_search_terms=[], skip_initial_search=True,
    ):
        events.append(ev)
    return "".join(c for c in events if isinstance(c, str))


# ===========================================================================
# Whole-loop wall-clock deadline
# ===========================================================================

class TestLoopDeadline:

    @pytest.mark.asyncio
    async def test_deadline_stops_runaway_and_synthesizes(self, controller, monkeypatch):
        """A model that keeps requesting tools every round would run to
        max_rounds; the wall-clock deadline must cut it short and hand off to
        final synthesis instead of hanging the turn."""
        captured = {"final_called": False}
        calls = {"n": 0}

        # Fake monotonic clock: deadline is set on the first call, then each
        # top-of-loop check advances 60s. With loop_timeout=100s the loop runs
        # exactly one round (60s < 100s), then the next check (120s) trips it.
        ticks = iter([0, 60, 120, 180, 240, 300, 360])
        last = {"t": 0}

        def fake_monotonic():
            try:
                last["t"] = next(ticks)
            except StopIteration:
                last["t"] += 60
            return last["t"]

        monkeypatch.setattr("core.agentic.controller.time.monotonic", fake_monotonic)
        monkeypatch.setattr("config.app_config.AGENTIC_LOOP_TIMEOUT_S", 100.0)

        async def fake_decision(*a, **k):
            calls["n"] += 1
            return [SearchDecision(
                wants_memory_search=True,
                memory_query="x", memory_collection="conversations",
            )]

        class _Res:
            start_events = []
            end_events = []
            round_data = MagicMock(duration_ms=1.0)
            formatted_context = "MEMORY: relevant context."
            memory_collection = "conversations"
            is_expand = False
            decision = SearchDecision()

        async def fake_dispatch(*a, **k):
            return _Res()

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_dispatch_single", fake_dispatch)
        monkeypatch.setattr(controller, "_generate_final_response", _make_final_spy(captured))

        text = await _run(controller)

        # Ran at most one round before the deadline tripped — NOT max_rounds.
        assert calls["n"] <= 1
        # Fell through to final synthesis rather than hanging / returning empty.
        assert captured["final_called"] is True
        assert "SYNTHESIZED ANSWER." in text


# ===========================================================================
# Per-round decision-LLM timeout
# ===========================================================================

class TestRoundTimeout:

    @pytest.mark.asyncio
    async def test_stalled_decision_call_times_out_to_answer(self, controller, monkeypatch):
        """A decision-LLM call that never returns must not hang the round; the
        wait_for backstop converts it into an implicit 'ready to answer'."""
        monkeypatch.setattr("config.app_config.AGENTIC_ROUND_TIMEOUT_S", 0.05)

        async def never_returns(*a, **k):
            await asyncio.sleep(5.0)  # far longer than the 0.05s timeout
            return "<memory>should never be parsed</memory>"

        monkeypatch.setattr(controller, "_generate_decision_no_reasoning", never_returns)

        handler = MagicMock()
        handler.parse_response = MagicMock(
            side_effect=AssertionError("parse_response must not run on timeout")
        )
        session = MagicMock()
        session.protocol = "xml_markers"  # not NATIVE_TOOLS → XML decision path

        decisions = await controller._get_model_decision(
            prompt="p", system_prompt="s", model_name="m",
            handler=handler, session=session,
        )

        assert len(decisions) == 1
        assert decisions[0].wants_answer is True
        handler.parse_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_fast_decision_call_is_unaffected(self, controller, monkeypatch):
        """A decision call that returns within budget parses normally — the
        timeout must not interfere with the happy path."""
        monkeypatch.setattr("config.app_config.AGENTIC_ROUND_TIMEOUT_S", 5.0)

        async def fast(*a, **k):
            return "<raw response>"

        parsed = [SearchDecision(wants_memory_search=True, memory_query="x")]
        monkeypatch.setattr(controller, "_generate_decision_no_reasoning", fast)

        handler = MagicMock()
        handler.parse_response = MagicMock(return_value=parsed)
        session = MagicMock()
        session.protocol = "xml_markers"

        decisions = await controller._get_model_decision(
            prompt="p", system_prompt="s", model_name="m",
            handler=handler, session=session,
        )

        assert decisions is parsed
        handler.parse_response.assert_called_once_with("<raw response>")


# ===========================================================================
# Persistent sandbox idle-recycle
# ===========================================================================

class TestSandboxRecycle:
    """_get_sandbox_session had two defects fixed 2026-07-24: the age recycle was
    DEAD (read `.age`; PersistentSession exposes `age_seconds`), and `is_closed`
    couldn't see a sandbox E2B killed server-side (idle/crash) — a dead handle was
    handed back and the next run() failed. Now: age_seconds recycle + a backend
    liveness probe (is_alive)."""

    @pytest.mark.asyncio
    async def test_stale_session_recycled(self, controller):
        old = MagicMock(is_closed=False)
        old.age_seconds = controller._sandbox_session_timeout + 1
        old.is_alive = MagicMock(return_value=True)  # alive, but too old
        old.close = AsyncMock()
        new = MagicMock(is_closed=False)
        controller.sandbox_manager = MagicMock()
        controller.sandbox_manager.create_session = AsyncMock(return_value=new)
        controller._sandbox_session = old

        result = await controller._get_sandbox_session()

        old.close.assert_awaited_once()
        assert result is new
        assert controller._sandbox_session is new

    @pytest.mark.asyncio
    async def test_dead_session_recreated(self, controller):
        """Young enough to reuse, but E2B killed it server-side — the liveness
        probe must catch it and recreate rather than return a dead handle."""
        dead = MagicMock(is_closed=False)
        dead.age_seconds = 1.0
        dead.is_alive = MagicMock(return_value=False)
        dead.close = AsyncMock()
        new = MagicMock(is_closed=False)
        controller.sandbox_manager = MagicMock()
        controller.sandbox_manager.create_session = AsyncMock(return_value=new)
        controller._sandbox_session = dead

        result = await controller._get_sandbox_session()

        dead.close.assert_awaited_once()
        assert result is new
        assert controller._sandbox_session is new

    @pytest.mark.asyncio
    async def test_fresh_live_session_reused(self, controller):
        fresh = MagicMock(is_closed=False)
        fresh.age_seconds = 1.0
        fresh.is_alive = MagicMock(return_value=True)
        fresh.close = AsyncMock()
        controller.sandbox_manager = MagicMock()
        controller.sandbox_manager.create_session = AsyncMock()
        controller._sandbox_session = fresh

        result = await controller._get_sandbox_session()

        fresh.close.assert_not_awaited()
        controller.sandbox_manager.create_session.assert_not_awaited()
        assert result is fresh


class TestPersistentSessionLiveness:
    """PersistentSession.is_alive() probes the E2B backend (is_running) so a
    server-side death is detected; local is_closed alone can't see it."""

    def _session(self, sandbox):
        from knowledge.sandbox_manager import PersistentSession
        return PersistentSession(sandbox, MagicMock())

    def test_alive_when_backend_running(self):
        sb = MagicMock()
        sb.is_running = MagicMock(return_value=True)
        assert self._session(sb).is_alive() is True

    def test_dead_when_backend_not_running(self):
        sb = MagicMock()
        sb.is_running = MagicMock(return_value=False)
        assert self._session(sb).is_alive() is False

    def test_probe_error_treated_as_dead(self):
        sb = MagicMock()
        sb.is_running = MagicMock(side_effect=RuntimeError("sandbox not found"))
        assert self._session(sb).is_alive() is False

    def test_closed_short_circuits_without_probe(self):
        sb = MagicMock()
        sb.is_running = MagicMock(return_value=True)
        sess = self._session(sb)
        sess._closed = True
        assert sess.is_alive() is False
        sb.is_running.assert_not_called()

    def test_no_probe_method_assumes_alive(self):
        sb = MagicMock(spec=[])  # no is_running attribute
        assert self._session(sb).is_alive() is True
