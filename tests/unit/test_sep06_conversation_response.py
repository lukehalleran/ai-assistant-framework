"""Conversation audit: evidence supports the reply without taking over it."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import json

import pytest

from core.agentic.gate import evaluate_agentic_gate
from core.insight.detector import allows_pattern_classification
from core.insight.synthesizer import (
    build_synthesis_prompts, recent_conversation_context,
    synthesize_stream, uses_conversational_synthesis,
)
from core.insight.types import EvidenceItem, InsightIntent
from core.prompt.gatherer_knowledge import _upload_is_live
from utils.web_search_trigger import analyze_for_web_search_llm


REFLECTION = (
    "Maybe I do need to schedule these. I'm not sure. It does not seem possible "
    "for me to exist and not work while on the ADHD meds. I know if I go more "
    "than 2 or 3 days off them, that leads to issues. Maybe like 1 or two days "
    "off a month idk"
)


class OvereagerClassifier:
    async def generate_once(self, *args, **kwargs):
        return json.dumps({
            "should_search": False, "confidence": 0.97,
            "reason": "personal pattern", "search_terms": [],
            "search_depth": "quick", "num_searches": 0,
            "needs_pattern_analysis": True,
        })


@pytest.mark.parametrize("query", [
    REFLECTION,
    "To be fair, 11 am haha I should have been clear",
    "Yeah. Honestly I feel good which is weird for me since I am also not being productive.",
])
def test_reflection_is_not_an_analysis_request(query):
    assert not allows_pattern_classification(query)


@pytest.mark.asyncio
async def test_classifier_cannot_force_reflection_into_report(monkeypatch):
    import utils.web_search_trigger as trigger
    trigger._llm_trigger_cache.clear()
    decision = await analyze_for_web_search_llm(REFLECTION, model_manager=OvereagerClassifier())
    assert not decision.needs_pattern_analysis
    # Also exercise the gate boundary with a malformed upstream decision.
    monkeypatch.setattr(trigger, "analyze_for_web_search_llm", AsyncMock(return_value=MagicMock(
        needs_pattern_analysis=True, should_search=False, search_terms=[],
        needs_memory_search=False, needs_knowledge_search=False,
        needs_document_generation=False,
    )))
    gate = await evaluate_agentic_gate(REFLECTION, model_manager=OvereagerClassifier())
    assert "insight" not in gate.modes


@pytest.mark.parametrize("query", [
    "Could my work hours and my sleep be connected? Use my history to assess it.",
    "Compare my study hours before and after moving apartments.",
    "Use my notes and outside research to test my theory.",
])
def test_analysis_requests_remain_eligible(query):
    assert allows_pattern_classification(query)


def test_conversation_is_bounded_and_speaker_attributed():
    context = recent_conversation_context([
        ("older " * 4000, "older reply " * 4000),
        ("The dose was at 11 AM yesterday.", "I misunderstood the timing."),
        ("I feel good while resting.", "That is useful to notice."),
    ])
    assert "User: The dose was at 11 AM yesterday." in context
    assert "User: I feel good while resting." in context
    assert "Assistant: I misunderstood" in context
    assert len(context) < 4200


def test_corpus_fallback_preserves_empty_authored_side():
    corpus = MagicMock()
    corpus.get_recent_memories.return_value = [
        {"query": "Attachment-only lecture", "user_text": "", "response": "I read it."},
        {"query": "typed plus attachment", "user_text": "Typed message", "response": "Reply"},
    ]
    context = recent_conversation_context([], corpus)
    assert context.index("Typed message") < context.index("I read it")
    assert "lecture" not in context
    assert "typed plus attachment" not in context


@pytest.mark.asyncio
async def test_actual_synthesis_gets_context_and_failed_analysis_is_not_zero():
    intent = InsightIntent(kind="pattern_temporal", theme="rest and focus",
                           raw_query="Does this fit what I have told you about rest and focus?")
    history = recent_conversation_context([
        {"role": "user", "content": "I took it at 11 AM yesterday."},
        {"role": "assistant", "content": "I had misread that."},
        {"role": "user", "content": "I feel good while resting today."},
    ])
    manifest = {"status": "insufficient", "limitations": ["phase specification failed"],
                "channels": [{"channel": "pattern", "attempted": False, "count": 0}]}
    evidence = [EvidenceItem(text="I struggled after several days off.",
                             collection="corpus", speaker="user", date="2026-08-10")]
    manager = MagicMock()
    manager.generate_async = AsyncMock(return_value="A contextual reply.")
    chunks = [chunk async for chunk in synthesize_stream(
        intent, evidence, None, model_manager=manager,
        conversation_context=history, deliberation_manifest=manifest,
    )]
    assert chunks == ["A contextual reply."]
    kwargs = manager.generate_async.call_args.kwargs
    assert "User: I feel good while resting today." in kwargs["prompt"]
    assert "11 AM yesterday" in kwargs["prompt"]
    assert kwargs["prompt"].endswith(intent.raw_query)
    assert "unavailable, not a measured zero" in kwargs["system_prompt"]
    assert "COMPUTED AGGREGATE:" not in kwargs["system_prompt"]
    # Generalized 2026-09-06: decision-support block is domain-neutral.
    assert "Decision support" in kwargs["system_prompt"]
    assert "for continuing and for" in kwargs["system_prompt"]
    # Debug serialization must be the same prompt actually sent.
    system, prompt = build_synthesis_prompts(
        intent, evidence, None, conversation_context=history, deliberation_manifest=manifest,
    )
    assert (system, prompt) == (kwargs["system_prompt"], kwargs["prompt"])
    manager.generate_async.assert_awaited_once()


@pytest.mark.parametrize("query,wants_document", [
    ("Give me a detailed analysis with a table by phase", False),
    ("Write this up for my appointment", True),
])
def test_explicit_report_keeps_analytical_presentation(query, wants_document):
    intent = InsightIntent(kind="pattern_temporal", theme="rest", raw_query=query,
                           wants_document=wants_document)
    assert not uses_conversational_synthesis(intent)
    system, _ = build_synthesis_prompts(intent, [], None)
    assert "COMPUTED AGGREGATE" in system


@pytest.mark.parametrize("query", [
    "To be fair, 11 am haha I should have been clear",
    "I feel good while resting today.", REFLECTION,
])
def test_fresh_homework_does_not_follow_user_into_personal_chat(query):
    doc = {"content": "Exponential smoothing exercise", "relevance_score": 0.1,
           "metadata": {"title": "upload:Homework.docx", "timestamp": datetime.now().isoformat()}}
    assert not _upload_is_live(doc, query=query)
    assert _upload_is_live(doc, query="What does task 2 in the assignment ask?")
    assert _upload_is_live(doc, query="Summarize Homework.docx")


def test_aware_fresh_upload_remains_available_for_document_work():
    doc = {"content": "Homework", "relevance_score": 0.0,
           "metadata": {"timestamp": datetime.now(timezone.utc).isoformat()}}
    assert _upload_is_live(doc, query="Help me with the uploaded assignment")
