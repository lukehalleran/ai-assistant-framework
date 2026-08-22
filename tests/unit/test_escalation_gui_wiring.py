"""
Regression tests for the 2026-08-21 escalation-tracker dead-wiring fix.

Incident (2026-08-18 audit): escalation_tracker.update() + safety_canary.observe()
lived only in orchestrator._build_prompt_phase, reached via process_user_query —
but the GUI/SSE path calls prepare_prompt → build_full_prompt and never updated
the trackers. Counters stayed at zero across a ~50-turn CONCERN spiral and
GROUNDING_PRESENCE never fired in production. Same dead-wiring class as the
.intent_type incident.

Fix: the update moved INTO build_full_prompt (_update_safety_trackers), before
_build_system_prompt so the current turn counts toward the sustained-distress
threshold; _build_prompt_phase no longer updates (double-count guard); the GUI
telemetry hook now calls record_response so engagement detection can advance.
"""

import asyncio
import inspect
import re

import pytest

from core.context_pipeline import ContextResult, ToneLevel
from core.escalation_tracker import EscalationTracker, ResponseStrategy
from core.orchestrator import DaemonOrchestrator


def _make_context(query="I feel awful", tone=ToneLevel.CONCERN):
    return ContextResult(
        processed_query=query,
        original_query=query,
        tone_level=tone,
        tone_instructions="",
        emotional_context=None,
    )


def _bare_orchestrator(tracker):
    """Minimal orchestrator with only what build_full_prompt's tracker/system-prompt
    stages touch — never runs __init__ (no models, no stores)."""
    orch = object.__new__(DaemonOrchestrator)
    orch.escalation_tracker = tracker
    orch.safety_canary = None
    import logging
    orch.logger = logging.getLogger("test_escalation_wiring")
    return orch


class TestUpdateSafetyTrackers:
    def test_update_feeds_tracker(self):
        tracker = EscalationTracker()
        orch = _bare_orchestrator(tracker)
        orch._update_safety_trackers(_make_context())
        assert len(tracker.tone_history) == 1
        assert tracker.consecutive_distress_count == 1

    def test_sustained_concern_reaches_grounding(self):
        """The exact 08-18 failure: a CONCERN-only spiral must reach GROUNDING."""
        tracker = EscalationTracker(distress_threshold=5)
        orch = _bare_orchestrator(tracker)
        for _ in range(5):
            orch._update_safety_trackers(_make_context())
        assert tracker.current_strategy == ResponseStrategy.GROUNDING_PRESENCE
        assert "GROUNDING PRESENCE" in tracker.get_strategy_instructions()

    def test_never_raises_without_tracker(self):
        orch = _bare_orchestrator(None)
        orch._update_safety_trackers(_make_context())  # must not raise

    def test_never_raises_on_broken_tracker(self):
        class Boom:
            def update(self, *a, **k):
                raise RuntimeError("boom")
        orch = _bare_orchestrator(Boom())
        orch._update_safety_trackers(_make_context())  # must not raise


class TestBuildFullPromptWiring:
    def test_build_full_prompt_updates_tracker_before_system_prompt(self, monkeypatch):
        """Drive THE deployed build_full_prompt and assert the tracker update
        lands before the system prompt is built (this turn's tone must count
        toward the strategy the prompt renders)."""
        tracker = EscalationTracker(distress_threshold=1)
        orch = _bare_orchestrator(tracker)
        orch.response_planner = None
        order = []

        def fake_system_prompt(context, use_raw_mode):
            order.append(("system_prompt", tracker.consecutive_distress_count))
            return "SYSTEM"

        class FakeBuilder:
            async def build_prompt_from_context(self, context):
                return {}

            def _assemble_prompt(self, context, user_input, system_prompt):
                return "PROMPT"

        monkeypatch.setattr(orch, "_build_system_prompt", fake_system_prompt)
        orch.prompt_builder = FakeBuilder()

        prompt, system_prompt = asyncio.run(
            orch.build_full_prompt(_make_context(), use_raw_mode=False)
        )
        assert prompt == "PROMPT"
        # tracker was updated (count=1) BEFORE _build_system_prompt ran
        assert order == [("system_prompt", 1)]
        assert len(tracker.tone_history) == 1

    def test_raw_mode_skips_tracker(self, monkeypatch):
        tracker = EscalationTracker()
        orch = _bare_orchestrator(tracker)
        orch.response_planner = None
        monkeypatch.setattr(orch, "_build_system_prompt", lambda c, r: "SYSTEM")

        class FakeBuilder:
            async def build_prompt_from_context(self, context):
                return {}

            def _assemble_prompt(self, context, user_input, system_prompt):
                return "PROMPT"

        orch.prompt_builder = FakeBuilder()
        asyncio.run(orch.build_full_prompt(_make_context(), use_raw_mode=True))
        assert len(tracker.tone_history) == 0


class TestNoDoubleCount:
    """Source-level guards: exactly ONE update site, inside build_full_prompt."""

    def test_build_prompt_phase_does_not_update(self):
        src = inspect.getsource(DaemonOrchestrator._build_prompt_phase)
        assert "escalation_tracker.update(" not in src
        assert "safety_canary.observe(" not in src

    def test_build_full_prompt_calls_helper(self):
        src = inspect.getsource(DaemonOrchestrator.build_full_prompt)
        assert "_update_safety_trackers" in src

    def test_single_update_call_site_in_orchestrator(self):
        src = inspect.getsource(inspect.getmodule(DaemonOrchestrator))
        assert len(re.findall(r"escalation_tracker\.update\(", src)) == 1


class TestGuiRecordResponse:
    def test_telemetry_hook_records_response(self):
        """gui.handlers._write_turn_telemetry must feed the final response to
        the tracker (engagement detection was unreachable on the GUI path)."""
        import gui.handlers as handlers
        src = inspect.getsource(handlers._write_turn_telemetry)
        assert "record_response" in src

    def test_record_response_called(self):
        from gui.handlers import _write_turn_telemetry

        class FakeTracker:
            def __init__(self):
                self.recorded = []

            def record_response(self, text):
                self.recorded.append(text)

        class FakeOrch:
            _last_turn_signals = {}
        tracker = FakeTracker()
        orch = FakeOrch()
        orch.escalation_tracker = tracker

        class FakeCtx:
            orchestrator = orch
            telemetry = {}
            user_text = "hello"
            t_prepare_elapsed = 0.0

        _write_turn_telemetry(
            FakeCtx(), "enhanced", "sess", "model", 12,
            response_text="Try taking a short walk.",
        )
        assert tracker.recorded == ["Try taking a short walk."]
