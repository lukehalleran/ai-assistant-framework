"""Gate must not route web on a verdict with empty search terms (2026-10-25, WEB2 W2).

Class: BC-05, BC-38. The controller's no-seed path ("_web_mode_no_seed")
insists on one real paid search, so a should_search=True with search_terms=[]
(e.g. a cached/prefetched verdict) bought a search nothing justified.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agentic.gate import evaluate_agentic_gate

CLEAN = "Rowan said the venue has been changing plans around this week"
WRAPPED = "Rowan said the venue has been\n   changing plans around this week"


def _verdict(terms, memory=False):
    return MagicMock(
        should_search=True, search_terms=terms,
        needs_memory_search=memory, needs_knowledge_search=False,
        needs_document_generation=False, needs_pattern_analysis=False,
        needs_email_search=False, blocked_reason="",
    )


async def _gate(text, verdict):
    with patch("utils.web_search_trigger.analyze_for_web_search_llm",
               new_callable=AsyncMock, return_value=verdict):
        return await evaluate_agentic_gate(text, model_manager=MagicMock())


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [CLEAN, WRAPPED])
async def test_empty_terms_routes_no_web(text):
    d = await _gate(text, _verdict([]))
    assert "web_search" not in (d.modes or [])
    assert not d.should_trigger


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [CLEAN, WRAPPED])
async def test_terms_present_still_routes_web(text):
    d = await _gate(text, _verdict(["venue schedule changes"]))
    assert d.should_trigger and "web_search" in d.modes
    assert d.search_terms == ["venue schedule changes"]
