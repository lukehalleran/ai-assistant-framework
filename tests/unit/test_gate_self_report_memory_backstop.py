"""Tier-4 LLM memory-search backstop for bare self-reports (2026-09-06).

Live 15:10 retest: the identical self-report that got "no trigger" that
morning flipped to `needs_memory_search=True` from the LLM trigger and ran a
memory loop. A bare first-person self-report with no recall cue is narration;
the enhanced path already retrieves memories for it. Drives THE deployed
evaluate_agentic_gate with the same harness as TestTier4LLMFallback.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agentic.gate import evaluate_agentic_gate

WRAPPED_SELF_REPORT = ("I took my stimulant at 10 AM today and I'm just\n  resting this "
                       "afternoon, feels good honestly even\n  though I got nothing done")


def _memory_decision():
    return MagicMock(
        should_search=False, search_terms=[],
        needs_memory_search=True, needs_knowledge_search=False,
        needs_document_generation=False, needs_pattern_analysis=False,
    )


@pytest.mark.asyncio
async def test_bare_self_report_does_not_trigger_memory_loop():
    with patch("utils.web_search_trigger.analyze_for_web_search_llm",
               new_callable=AsyncMock, return_value=_memory_decision()):
        d = await evaluate_agentic_gate(WRAPPED_SELF_REPORT, model_manager=MagicMock())
    assert not d.should_trigger
    assert "memory" not in (d.modes or [])


@pytest.mark.asyncio
async def test_recall_cue_still_honors_llm_memory_verdict():
    # First-person statement WITH a recall cue: the user is pointing back at
    # something said before — memory search stays on.
    with patch("utils.web_search_trigger.analyze_for_web_search_llm",
               new_callable=AsyncMock, return_value=_memory_decision()):
        d = await evaluate_agentic_gate(
            "I told you about my dentist before, the one who remembered my name",
            model_manager=MagicMock(),
        )
    assert d.should_trigger and "memory" in d.modes


@pytest.mark.asyncio
async def test_question_still_honors_llm_memory_verdict():
    with patch("utils.web_search_trigger.analyze_for_web_search_llm",
               new_callable=AsyncMock, return_value=_memory_decision()):
        d = await evaluate_agentic_gate(
            "Can you describe the color of my cat's fur exactly?", model_manager=MagicMock(),
        )
    assert d.should_trigger and "memory" in d.modes
