"""Per-task gather spans (start/end offsets) reach the turn record.

Incident date: 2026-10-27 (first turn after a restart: tasks that cost ms
when warm took 3-10 s; durations alone cannot say whether a task STARTED
late or RAN long). Bug classes: BC-41 (latency), BC-70. Drives the deployed
UnifiedPromptBuilder.build_prompt (full_builder fakes, as in
test_section_outcomes.py) and the deployed build_full_prompt.
"""
import asyncio
import logging

import pytest

from core.context_pipeline import ContextResult, ToneLevel
from core.escalation_tracker import EscalationTracker
from core.orchestrator import DaemonOrchestrator
from tests.unit.test_independent_prompt_audit import full_builder, retrieval_limits

RECENT = [{"query": "Synthetic question", "response": "Synthetic retained answer."}]


class TestGatherRecordsSpans:
    @pytest.mark.asyncio
    async def test_late_starting_task_shows_late_start(self, monkeypatch):
        builder = full_builder(monkeypatch, RECENT, budget=10000)
        gate = asyncio.Lock()
        await gate.acquire()  # held until the early task finishes
        order = []

        async def recent(*args):
            await asyncio.sleep(0.15)
            gate.release()
            order.append("recent")
            return RECENT

        async def profile(*args, **kwargs):
            # The task coroutine begins at once; its OWN work waits on the lock.
            await gate.acquire()
            order.append("profile")
            return []

        builder.context_gatherer._get_recent_conversations = recent
        builder.context_gatherer.get_user_profile_context = profile
        result = await builder.build_prompt("Synthetic question", retrieval_overrides=retrieval_limits())

        spans = result["_task_spans"]
        assert set(spans) == set(result["_task_timings"]) == {"recent", "user_profile", "web_search"}
        for name, (start, end) in spans.items():
            assert 0 <= start <= end, name
        # Both started at once; the lock-bound task ENDED after the holder.
        assert spans["user_profile"][1] >= spans["recent"][1]
        assert spans["recent"][1] - spans["recent"][0] >= 0.14
        # Duration and span agree (same clock) and task_timings keeps its shape.
        assert all(isinstance(v, float) for v in result["_task_timings"].values())
        assert abs((spans["recent"][1] - spans["recent"][0]) - result["_task_timings"]["recent"]) < 0.01

    @pytest.mark.asyncio
    async def test_failed_task_still_has_a_span(self, monkeypatch):
        builder = full_builder(monkeypatch, RECENT, budget=10000)

        async def boom(*args, **kwargs):
            raise ValueError("synthetic")

        builder.context_gatherer.get_user_profile_context = boom
        result = await builder.build_prompt("Synthetic question", retrieval_overrides=retrieval_limits())
        assert "user_profile" in result["_task_spans"]


class _Builder:
    def __init__(self, ctx):
        self._ctx = ctx

    async def build_prompt_from_context(self, context):
        return self._ctx

    def _assemble_prompt(self, context, user_input, system_prompt):
        return "PROMPT"


def test_turn_signals_carry_task_spans():
    orch = object.__new__(DaemonOrchestrator)
    orch.escalation_tracker = EscalationTracker()
    orch.safety_canary = None
    orch.logger = logging.getLogger("test_task_spans")
    orch.response_planner = None
    orch.prompt_builder = _Builder({"_task_spans": {"recent": [0.0, 0.2], "user_profile": [0.0, 0.9]}})
    orch._build_system_prompt = lambda context, use_raw_mode: "SYSTEM"
    q = "Synthetic question"
    context = ContextResult(processed_query=q, original_query=q, tone_level=ToneLevel.CONVERSATIONAL,
                            tone_instructions="", emotional_context=None)
    asyncio.run(orch.build_full_prompt(context))
    assert orch._last_turn_signals["task_spans"] == {"recent": [0.0, 0.2], "user_profile": [0.0, 0.9]}
