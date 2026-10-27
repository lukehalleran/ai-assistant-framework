"""Behavioural sibling for ``TestNoDoubleCount`` in
tests/unit/test_escalation_gui_wiring.py (BC-63: three tests grepped the
source of ``_build_prompt_phase`` / ``build_full_prompt`` / the whole module
for ``escalation_tracker.update(`` and counted regex hits).

Incident date: 2026-08-18 (the GUI path never updated the EscalationTracker;
the fix moved the update into ``build_full_prompt`` and must not double-count
on ``process_user_query``'s ``_build_prompt_phase`` route). Bug classes:
BC-63, BC-62. THE deployed ``_build_prompt_phase`` is driven and the tracker
and canary are COUNTED: exactly one update/observe per turn.
"""
import logging
from types import SimpleNamespace

import pytest

from core.context_pipeline import ContextResult, ToneLevel
from core.escalation_tracker import EscalationTracker
from core.orchestrator import DaemonOrchestrator


class _Canary:
    def __init__(self):
        self.seen = []

    def observe(self, text, tone):
        self.seen.append(text)


class _Builder:
    async def build_prompt_from_context(self, context):
        return {}

    def _assemble_prompt(self, context, user_input, system_prompt):
        return "PROMPT"


def _context(query):
    return ContextResult(processed_query=query, original_query=query,
                         tone_level=ToneLevel.CONCERN, tone_instructions="",
                         emotional_context=None)


def _orch(query):
    orch = object.__new__(DaemonOrchestrator)
    orch.escalation_tracker = EscalationTracker()
    orch.safety_canary = _Canary()
    orch.logger = logging.getLogger("test_safety_tracker_single_update")
    orch.response_planner = None
    orch.prompt_builder = _Builder()
    orch._build_system_prompt = lambda context, use_raw_mode: "SYSTEM"

    async def build_context(user_input, files, use_raw_mode, personality):
        return _context(user_input)

    orch.build_context = build_context
    return orch


def _flow(query, use_raw_mode=False):
    return SimpleNamespace(user_input=query, files=[], use_raw_mode=use_raw_mode,
                           personality=None, debug_info={})


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "I feel awful today",
    "    I feel awful\n    today\n   ",
])
async def test_prompt_phase_counts_the_turn_once(query):
    orch = _orch(query)
    await orch._build_prompt_phase(_flow(query))
    assert len(orch.escalation_tracker.tone_history) == 1
    assert orch.escalation_tracker.consecutive_distress_count == 1
    assert orch.safety_canary.seen == [query]


@pytest.mark.asyncio
async def test_two_turns_count_twice_not_four_times():
    orch = _orch("x")
    await orch._build_prompt_phase(_flow("I feel awful today"))
    await orch._build_prompt_phase(_flow("still feeling awful"))
    assert len(orch.escalation_tracker.tone_history) == 2
    assert len(orch.safety_canary.seen) == 2


@pytest.mark.asyncio
async def test_raw_mode_turn_is_not_counted():
    orch = _orch("x")
    await orch._build_prompt_phase(_flow("I feel awful today", use_raw_mode=True))
    assert len(orch.escalation_tracker.tone_history) == 0
    assert orch.safety_canary.seen == []
