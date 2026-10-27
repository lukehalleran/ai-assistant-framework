"""Behavioural sibling for ``TestCachePrefixPlacement`` in
tests/unit/test_response_guidance.py (BC-63: two tests read
core/orchestrator.py as text and grepped ``UNIVERSAL_GROUNDING +
PROMPT_CACHE_BREAKPOINT``).

Incident date: 2026-09-06 (a ~440-token incident-specific block sat in the
cached system-prompt prefix of every turn). Bug classes: BC-63, BC-62. THE
deployed ``DaemonOrchestrator._build_system_prompt`` is driven: the cached
prefix (before PROMPT_CACHE_BREAKPOINT) carries ONLY the universal block and
is byte-identical across different queries; the per-turn decision block never
precedes the breakpoint.
"""
from unittest.mock import AsyncMock, Mock

import pytest

from config.app_config import PROMPT_CACHE_BREAKPOINT
from core.context_pipeline import ContextResult, ToneLevel
from core.response_guidance import (
    CONTEXTUAL_GROUNDING,
    DECISION_SUPPORT_GROUNDING,
    UNIVERSAL_GROUNDING,
)


def _orch():
    from core.orchestrator import DaemonOrchestrator
    mm = Mock()
    mm.get_active_model_name = Mock(return_value="m")
    mm.supports_reasoning = Mock(return_value=False)
    rg = Mock()
    rg.time_manager = None
    ms = Mock()
    ms.corpus_manager = Mock()
    pb = Mock()
    pb.build_prompt = AsyncMock(return_value="x")
    return DaemonOrchestrator(model_manager=mm, response_generator=rg,
                              file_processor=Mock(), prompt_builder=pb, memory_system=ms)


def _ctx(query, tone=ToneLevel.CONVERSATIONAL):
    return ContextResult(processed_query=query, original_query=query,
                         tone_level=tone, tone_instructions="",
                         primary_topic="general")


def _norm(text):
    return " ".join(text.split())


QUERIES = [
    "Does my history support scheduling rest days? Weigh both sides.",
    "    Does my history support scheduling\n    rest days? Weigh both sides.\n   ",
]


@pytest.mark.parametrize("query", QUERIES)
def test_prefix_holds_universal_block_and_no_per_turn_blocks(query):
    prompt = _orch()._build_system_prompt(_ctx(query))
    assert prompt.count(PROMPT_CACHE_BREAKPOINT) == 1
    head = _norm(prompt.split(PROMPT_CACHE_BREAKPOINT)[0])
    assert _norm(UNIVERSAL_GROUNDING) in head
    assert _norm(DECISION_SUPPORT_GROUNDING) not in head
    assert _norm(CONTEXTUAL_GROUNDING) not in head


def test_prefix_is_identical_across_queries_and_tones():
    o = _orch()
    heads = {
        o._build_system_prompt(_ctx(q, tone)).split(PROMPT_CACHE_BREAKPOINT)[0]
        for q in QUERIES
        for tone in (ToneLevel.CONVERSATIONAL, ToneLevel.CONCERN)
    }
    assert len(heads) == 1, "per-turn content leaked into the cached prefix"
