"""Behavioural sibling for ``test_turn_record_builder_writes_the_field_and_docs_list_it``
in tests/unit/test_tone_semantic_source_telemetry.py (BC-63: it grepped
``inspect.getsource(orchestrator)`` for the dict-literal text).

Incident date: 2026-10-15 (the tone semantic stage's scored-text source lived
only in an explanation string). Bug classes: BC-63, BC-62, BC-72 (a
telemetry field that is never populated). THE deployed ``build_full_prompt``
is driven; ``_last_turn_signals`` (which ``_hook_turn_telemetry`` merges into
the turn record) must carry the enum value, and only the enum.
"""
import asyncio
import logging
from types import SimpleNamespace

import pytest

from core.context_pipeline import ContextResult, ToneLevel
from core.escalation_tracker import EscalationTracker
from core.orchestrator import DaemonOrchestrator, PostResponseHookContext, _hook_turn_telemetry


class _Builder:
    async def build_prompt_from_context(self, context):
        return {}

    def _assemble_prompt(self, context, user_input, system_prompt):
        return "PROMPT"


def _orch():
    orch = object.__new__(DaemonOrchestrator)
    orch.escalation_tracker = EscalationTracker()
    orch.safety_canary = None
    orch.logger = logging.getLogger("test_turn_signals_tone_semantic_source")
    orch.response_planner = None
    orch.prompt_builder = _Builder()
    orch._build_system_prompt = lambda context, use_raw_mode: "SYSTEM"
    return orch


def _context(emotional):
    q = "I feel terrible about the whole day"
    return ContextResult(processed_query=q, original_query=q,
                         tone_level=ToneLevel.CONCERN, tone_instructions="",
                         emotional_context=emotional)


def _signals(emotional):
    orch = _orch()
    asyncio.run(orch.build_full_prompt(_context(emotional)))
    return orch, orch._last_turn_signals


@pytest.mark.parametrize("value", ["head", "whole"])
def test_enum_value_reaches_the_turn_signals(value):
    _, sig = _signals(SimpleNamespace(tone_semantic_source=value,
                                      tone_trigger="semantic", tone_confidence=0.7))
    assert sig["tone_semantic_source"] == value
    assert sig["tone_trigger"] == "semantic"


@pytest.mark.parametrize("value", ["some free text", "", None, 3])
def test_non_enum_value_is_recorded_as_none(value):
    _, sig = _signals(SimpleNamespace(tone_semantic_source=value))
    assert sig["tone_semantic_source"] is None


def test_no_emotional_context_is_recorded_as_none():
    _, sig = _signals(None)
    assert sig["tone_semantic_source"] is None


def test_turn_record_row_carries_the_field(monkeypatch):
    orch, _ = _signals(SimpleNamespace(tone_semantic_source="head"))
    rows = []
    import utils.turn_telemetry as tt
    monkeypatch.setattr(tt, "record_turn", rows.append)
    _hook_turn_telemetry(PostResponseHookContext(
        orchestrator=orch, user_input="I feel terrible about the whole day",
        response_text="I hear you.", telemetry={}))
    assert rows and rows[0]["tone_semantic_source"] == "head"
