import json

import pytest

from core.agentic.gate import evaluate_agentic_gate
from core.insight.detector import detect_insight_request
from utils.web_search_trigger import (
    _looks_like_pattern_candidate,
    analyze_for_web_search_llm,
)


class PatternClassifier:
    async def generate_once(self, *_args, **_kwargs):
        return json.dumps({
            "should_search": False,
            "confidence": 0.94,
            "reason": "asks to compare two personal variables across history and research",
            "search_terms": [],
            "search_depth": "deep",
            "num_searches": 0,
            "needs_memory_search": False,
            "needs_knowledge_search": False,
            "needs_pattern_analysis": True,
            "needs_document_generation": False,
        })


@pytest.mark.parametrize("query", [
    "Has my sleep changed since I moved?",
    "Compare how I was before and after starting night shift",
    "Compare my study hours before and after moving apartments; use my notes and web research",
    "What tends to happen when I skip breakfast?",
    "Does my mood track with exercise?",
    "Use my notes and outside research to test my theory",
])
def test_deterministic_common_english_shapes(query):
    intent = detect_insight_request(query)
    assert intent is not None
    assert intent.kind == "pattern_temporal"


@pytest.mark.asyncio
async def test_llm_fallback_owns_unusual_mixed_shape_even_with_online_source_word():
    query = (
        "Could my late work nights and my delivery orders be connected? "
        "Use online sources and whatever you have stored to evaluate it."
    )
    assert detect_insight_request(query) is None
    assert _looks_like_pattern_candidate(query)

    trigger = await analyze_for_web_search_llm(
        query, model_manager=PatternClassifier(), timeout=1,
    )
    assert trigger.needs_pattern_analysis
    assert trigger.should_search is False

    gate = await evaluate_agentic_gate(query, model_manager=PatternClassifier())
    assert gate.modes == ["insight"]
    assert gate.insight_intent["kind"] == "pattern_temporal"


@pytest.mark.parametrize("query", [
    "Why am I itchy tonight?",
    "Tell me about sleep cycles in adults",
    "I think my room is too warm",
])
def test_current_state_and_general_knowledge_do_not_deterministically_route(query):
    intent = detect_insight_request(query)
    assert intent is None or intent.kind != "pattern_temporal"


@pytest.mark.parametrize("query", [
    "How many times has Trump been impeached?",
    "Is my appointment before noon tomorrow?",
    "Have I told you about Morgan before?",
])
def test_factual_and_memory_lookup_questions_do_not_hijack_pattern_mode(query):
    intent = detect_insight_request(query)
    assert intent is None or intent.kind != "pattern_temporal"
